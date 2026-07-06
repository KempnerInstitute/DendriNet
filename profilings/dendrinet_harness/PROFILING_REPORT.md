# DendriNet Profiling — Evidence Report

**Measurement only. Nothing was optimized, no precision changed, no kernel swapped.**
This pass classifies where forward + backward time and memory actually go, with
evidence, before any optimization.

- **Date:** 2026-07-06
- **Hardware:** NVIDIA H100 80GB HBM3 (`sm_90`), via SLURM (`kempner_requeue` / `kempner_undergrads`, `--gres=gpu:1`)
- **Stack:** PyTorch 2.11.0+cu128, CUDA 12.8, Python 3.11.15 (`dendritic_env`)
- **Harness:** `profile_dendrinet.py` · **Job:** `jobs/profile_dendrinet.sbatch`
- **Raw outputs:** `profilings/dendrinet_harness/{results.json, ops_*.txt, splitfb_*_b512.json}`

---

## TL;DR

> **~80% of the entire training step is a single eager op: the backward of the
> advanced-index synapse gather in `IndexedSparseLinear`** (`flat_x[:, idx]` →
> `indexing_backward_kernel` / `aten::_index_put_impl_`). It has **zero FLOPs**,
> is an **atomic scatter** (memory/atomic-bound), and is the only op worth a
> custom/fused kernel. Everything else — shunting divisor, top-k sort, BlockLinear
> convergence, E/I pooling — is cheap and already near-optimal for its FLOPs.

---

## Phase 0 — Discovery

### Environment
- The login node (`boslogin07`) has **no GPU** (`torch.cuda.is_available() == False`). All timing must run on a GPU node via SLURM.
- Reachable GPUs on `kempner_requeue`: H100 80GB, A100-SXM4 40GB, A100 MIG 3g.20gb, H200. Profiled on **H100**.
- `kempner_requeue` is **preemptible** (`--requeue`); jobs can be killed mid-run and restart from scratch. Keep runs short.

### Model forward path
Every example config in `configs/examples/` is a **`population_network`** core. Build chain:

```
load_config(path)
  -> get_architecture("population_network", core_cfg, input_dim)      # networks/architectures/factory.py
    -> PopulationNetwork(PopulationNetworkConfig(**mp_cfg))           # factory_recurrent.py:54
```

`forward(x)` signature (`recurrent/population_network.py:448`):
```python
def forward(self, x, hidden=None, return_hidden=False, seq_lengths=None): ...
```
For the profiled configs `rec_*_synapses: 0`, so `is_recurrent == False` → feedforward path,
`x` is `(B, input_dim)`, returns a plain tensor `(B, out)`.

Per-unit op sequence: **top-k/indexed synapse gather → E/I drive → branch-local
shunting divisor → axial BlockLinear convergence (child→parent) → E/I population pooling.**

### Implementation classification — everything is `[plain eager PyTorch]`
No custom CUDA, no Triton, no CuTe, no `load_inline`/`cpp_extension` anywhere in
`src/dendritic_modeling`. The **only** `torch.compile` is an opt-in wrapper on the
**feedforward `ei_network.py:599`** — NOT the `population_network` path used here, so
`torch.compile` is **inactive** for everything profiled (nothing to toggle on/off).

| Op family | Where (file) | Impl | Eager hotspots flagged |
|---|---|---|---|
| (a) top-k / indexed synapse gather | `synapse/topk.py`, `synapse/indexed_sparse.py`, `synapse/indexed_dynamic.py` | eager | `torch.topk`→`scatter`/`index_put`→dense matmul; **advanced-index gather** `flat_x[:, idx]` in a Python chunk loop (`indexed_sparse.py:264-267`); per-row `nonzero()` loops (`indexed_dynamic.py:223-241`) |
| (b) branch shunting / additive divisor | `dendritic/branch_dynamics.py` | eager | scalar-accumulate denominator, `numerator/(denom+eps)` at `:63`; no decorators |
| (c) axial child→parent (BlockLinear) | `dendritic/blocklinear.py` | eager | `EfficientBlockLinear`: `(x.view(B,out,bs)*w).sum(-1)` at `:317`; `DendriNet.forward` Python loop over branch layers (`dendrinet.py:438`) |
| (d) E/I population pooling | `recurrent/population_layer.py` | eager | `torch.cat`/`sum` per-population Python loops (`:464`, `:590`) |

Full flagged-hotspot inventory (scatter/gather/`index_put`/`.contiguous()`/boolean
masking/per-unit loops) is preserved in the harness classification and `ops_*.txt`.

---

## Phase 1 — Harness

`profile_dendrinet.py` (self-contained):

- Builds the **core network** directly from real configs (no dataloader) via `get_architecture`, so the signal is on the dendritic ops. Encoder is identity (MNIST/CIFAR), decoder is a tiny MLP — both excluded on purpose.
- Configs (shapes pulled from repo, not invented):
  - `population_network_local_learning_mnist` — e=128 `[3,3]`, i=32 `[1]`, ff-exc 40 / ff-inh 10, `indexed_rewire` top-k, input 784, 90,784 params.
  - `population_network_two_layer` — input 3072, 410,928 params (deepest).
  - `population_network_cifar10` — input 3072, 266,272 params.
- Batches **128 and 512** (separate launch-overhead from compute).
- **12 warmup + 50 timed iters.** `torch.cuda.Event` for authoritative fwd/bwd wall-clock (per-iter → mean + CV). Separate **forward-only / backward-only** `torch.profiler` passes (CUDA+CPU, `profile_memory=True`, `with_flops=True`) so per-op self-CUDA is cleanly attributable with no scope-nesting ambiguity.
- Reports coefficient of variation (CV) per timed op; flags CV > 5%.

Run: `sbatch jobs/profile_dendrinet.sbatch` (or `profile_dendrinet.py --cpu-smoke` to validate the build/forward/backward API on CPU without a GPU).

### Measurement corrections made (disclosed)
1. A first per-**module** hook table timed **inclusively** (parent modules contain children), so families summed >100%. Discarded in favor of profiler self-time.
2. torch 2.11 renamed `self_cuda_time_total` → `self_device_time_total`, silently zeroing the first programmatic table (`ka.table()` was unaffected). Fixed with robust attribute getters; re-ran.
3. `aten::_index_put_impl_` and `indexing_backward_kernel` are the **same work** (aten op + its CUDA kernel). Shares recomputed on dedup'd aten-only and kernel-only bases — they **agree to within 2%**.

---

## Phase 2 — Evidence

### Wall-clock (H100, `cuda.Event`, outside profiler)
Backward dominates and scales with batch; forward is sub-ms and noisy (⚠ = CV > 5%).

| config / batch | fwd ms (CV) | bwd ms (CV) | step ms | bwd % of step | peak mem | activation mem |
|---|---|---|---|---|---|---|
| local_learning / 128 | 3.05 (7%⚠) | 3.90 (46%⚠) | 6.95 | 56% | 101 MB | 50 MB |
| local_learning / 512 | 1.23 (78%⚠) | 8.44 (23%⚠) | 9.67 | 87% | 400 MB | 199 MB |
| two_layer / 128 | 6.85 (9%⚠) | 12.61 (0.6%) | 19.45 | 65% | 457 MB | 224 MB |
| **two_layer / 512** | 4.14 (25%⚠) | **35.04 (0.5%)** | **39.18** | **89%** | 1.79 GB | 882 MB |
| cifar10 / 128 | 1.24 (47%⚠) | 7.48 (0.2%) | 8.72 | 86% | 303 MB | 150 MB |
| cifar10 / 512 | 2.42 (16%⚠) | 22.16 (0.6%) | 24.58 | 90% | 1.19 GB | 593 MB |

### Clean forward/backward per-op self-CUDA (b512, dedup'd; both bases agree)

| phase | dominant op | share of phase |
|---|---|---|
| forward | `aten::sum` + `reduce_kernel` (BlockLinear sum + pooling) | ~35% |
| forward | `aten::index` (synapse gather) | 23–27% |
| forward | `aten::mul` (BlockLinear `x*w`, shunting; carries the real FLOPs) | ~11% |
| **backward** | **`aten::_index_put_impl_` → `indexing_backward_kernel` (index scatter)** | **88–91%** |
| backward | `aten::mul` | ~2% |
| backward | `aten::sum` | ~2% |
| backward | `DeviceRadixSort` (top-k sort) | ~0.6% |

### Ranked table — % of the whole step
(forward-share × fwd wall-frac + backward-share × bwd wall-frac; two_layer & local_learning @ b512)

| op (family) | current impl | % fwd | % bwd | mem traffic | gather/scatter? | measured-vs-FLOP gap | opt candidate? |
|---|---|---|---|---|---|---|---|
| **Advanced-index gather + backward scatter** — `aten::index` (fwd) / `_index_put_impl_`→`indexing_backward_kernel` (bwd), `IndexedSparseLinear` **(a)** | eager | ~3% | **~77–81%** | **huge**: gather materializes B×(out·k) (6.4 GB alloc, two_layer/512); bwd = atomic scatter | **YES — this IS the op** | **∞ (0 FLOPs, ~80% of time)** | **YES — #1, by far** |
| Reductions `aten::sum`/`reduce_kernel` — BlockLinear axial sum + pooling **(c/d)** | eager | ~6% | ~2% | moderate (reduce) | no | memory-bound reduce | marginal |
| Elementwise `aten::mul` — BlockLinear `x*w`, shunting **(b/c)** | eager | ~1.5% | ~2% | moderate | no | **has real FLOPs** (1.6 GF fwd), near-roofline | **no** |
| Shunting divisor `aten::div`/`add`/`neg` **(b)** | eager | <1% | <1% | small | no | trivial | **no** |
| Top-k **sort** `DeviceRadixSortOnesweep` — selection **(a)** | eager | <0.5% | ~0.6% | small | no | cheap | **no (surprise)** |
| BlockLinear whole `view+mul+sum` **(c)** | eager | ~2% | ~1% | moderate | no | cheap | **no** |
| `softplus` weight transform | eager | ~2% | ~1% | small | no | cheap | no |

---

## Direct answers

**1. Already efficient — don't touch.** Shunting/normalized-additive divisor **(b)**
(`div`/`add`/`sub`/`neg`, <1% each); the top-k **selection** itself (`DeviceRadixSort`
<0.6%); **BlockLinear axial coupling (c)** (`view+mul+sum`, ~2–3% total); `softplus`;
and the dense **`mul`** — which carries the real FLOPs (1.6 GFLOP fwd on two_layer)
yet costs ~2%, i.e. already near the H100 roofline. Fusing any of these recovers
single-digit-percent at best.

**2. Genuinely hot enough to justify a custom/fused kernel — exactly one.**
The **advanced-index synapse gather in `IndexedSparseLinear`, and specifically its
backward** (`flat_x[:, idx]` → `indexing_backward_kernel` / `_index_put_impl_`).
**~80% of the entire step**, ~90% of backward, at **zero FLOPs**. Forward gather
materializes multi-GB intermediates (6.4 GB at two_layer/512); backward is an
**atomic scatter** — many synapses gather the same input feature, so gradients must
atomically accumulate, serializing on the H100. A fused/segmented-reduction or
sparse-matmul kernel replacing the atomic scatter is the one place a kernel pays off.
*(Not implemented — measurement pass only.)*

**3. Surprises (explicit).**
- The **`torch.topk` sort is NOT the cost** (<0.6%). The intuitive suspect is wrong; the expense is *gathering* through the indices and, above all, *scattering gradients back* in backward.
- The **shunting divisor is negligible** (<1%) despite being the model's signature nonlinearity.
- **Forward is cheap and backward-dominated** (backward 56→90% of step as batch grows). Optimization attention belongs on the backward scatter, not forward.

**4. Compute-, memory-, or launch-bound? Neither compute.**
- **Backward = memory/atomic-bound.** Dominant kernel (`indexing_backward`) is pure memory traffic + atomics, **0 FLOPs**, with **low CV (0.5%)** at large batch → stable bottleneck.
- **Forward = launch/dispatch-overhead-bound.** Sub-millisecond, doesn't scale cleanly with batch (local_learning forward is *faster* at b512 than b128), **high CV (25–78%)** → many tiny eager kernels, not saturated compute.
- Nothing is compute-bound: the real-FLOP ops sit at ~2% of time.

---

## Reproduce

```bash
# full sweep (3 configs × batches 128,512)
sbatch jobs/profile_dendrinet.sbatch

# CPU API smoke test (no GPU needed)
PYTHONPATH=src python profile_dendrinet.py --cpu-smoke

# targeted clean fwd/bwd split (2 configs @ b512)
sbatch jobs/profile_dendrinet_split.sbatch
```

Outputs land in `profilings/dendrinet_harness/`:
`results.json` (all wall-clock + memory + CV), `ops_*.txt` (full `key_averages`
tables per config), `splitfb_*_b512.json` (clean per-op forward/backward self-CUDA).
Set `DENDRINET_TRACE=1` to also export Chrome traces (large/slow for index-heavy backward).
