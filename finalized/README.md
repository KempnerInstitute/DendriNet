# DendriNet kernels & optimization work — finalized snapshot

Curated entry point for the sparse-gather kernel and memory-optimization work
(branch `ryanconfig`). Everything here is a **snapshot copy**; the kernels
themselves live in `src/` (paths below), so you need the whole branch checked
out to run — this folder is where to start, not all you need.

## Quickstart (from the repo root, on a GPU node)

```bash
source finalized/setup_cuda_env.sh          # CUDA 12.9 + GCC 12 + conda env
export PYTHONPATH=src

# 1. prove the kernels are correct on your GPU (~2 min)
python finalized/checks/check_gradw_tile_equiv.py
python finalized/checks/check_fused_shunt.py

# 2. benchmark all four gather paths on the stress model (~15 min)
python finalized/harness/profile_dendrinet.py \
    --configs finalized/configs/kernel_stress_12L_16k_cuda.yaml --path cuda --batches 128

# 3. reproduce the memory story (~20 min)
python finalized/harness/sweep_batch_modes.py \
    --config finalized/configs/kernel_stress_12L_16k_cuda.yaml --out /tmp/sweep.json
```

Every training/profiling run logs which gather path it actually used
(`indexed gather path resolved: ...`) — if you see `loop`, your CUDA toolchain
didn't load and you are benchmarking the fallback.

## Where the actual kernels live (in `src/dendritic_modeling/`)

| file | what |
|---|---|
| `networks/architectures/excitation_inhibition/synapse/fused_gather.py` | custom-autograd gather (recompute-in-backward) |
| `.../synapse/cuda_gather.py` | CUDA warp-per-output fwd + split bwd kernels, adaptive tiles, variant picker |
| `.../synapse/triton_gather.py` | Triton gather (no nvcc needed) |
| `.../synapse/indexed_sparse.py` | the sparse layer; path dispatch cuda→triton→fused→loop |
| `networks/activations/parametric.py` | fused reactivation + fused shunt-divide autograd Functions |
| `analysis/utils/model_resources.py` | chunked (non-densifying) resource records |
| `training/fsdp_utils.py`, `training/strategies/fsdp_standard.py` | FSDP `cpu_init` + per-synapse wrapping |

**Flags** (all default to the fast/safe path; every one is reversible):
config: `indexed_use_{fused,triton,cuda}_gather`, `indexed_cache_recurrent_weight`,
`distributed.fsdp.{cpu_init, auto_wrap_policy: indexed_synapses}`.
env: `DENDRINET_POSITIVE_TANH_SIGMOID=0`, `DENDRINET_POSITIVE_TANH_FUSED=0`,
`DENDRINET_FUSED_SHUNT=0`, `DENDRINET_GATHER_FWD_VARIANT=staged|nostage`,
`DENDRINET_DENSE_RESOURCE_RECORDS=1` (each restores the pre-optimization behavior).

## Everything we did, in plain terms

**Feedforward (single GPU)**
1. Custom autograd gather ("fused") — stops storing the huge gather intermediate, recomputes it in backward. ~28× less memory; the reason batches above 128 run at all.
2. Triton gather kernel — same math, portable, no CUDA toolchain needed.
3. CUDA gather kernels (forward + split backward) — the fast path: 2.8× faster than fused, 9.2× faster than the naive loop at batch 128.
4. Adaptive backward tile — the backward kernel's grid now fills the GPU at real layer shapes: 1.17× whole step.
5. Un-staged forward variant + measured picker — 1.086× whole step, bit-identical output.
6. Reactivation memory rewrites (sigmoid identity → fused activation → fused shunt-divide) — the model keeps 1 tensor per activation instead of 3, and the divide stops pinning its operands: **2.19× less peak memory**, slightly faster.
7. Backward temporary cleanup — the fused backward itself was 16% of peak; rewritten in-place: smaller *and* faster.
8. int32 indices stay int32 — removed a silent int64 round-trip that kept an 889 MB copy alive every step.
9. Reporting fix — the post-training summary was densifying the sparse layers (~55 GB to compute five integers). Chunked, byte-identical output, **process peak 59.6 → 9.6 GB**.
10. Max trainable batch **1024 → 4096** (measured OOM frontier on Blackwell 96 GB and A100 80 GB; figure generators included).
11. Runtime gather-path logging — no more silent fallback; every run states which kernel it used.
12. Equivalence proof for every change — bit-identical where exact, ≤4e-7 where reordered, fp64-exact for the algebra, loss-trajectory-identical on the real trainer.

**RNN (the scaling track)**
13. Fused gather in real RNN training — 1.57× faster epoch, −36% memory, **loss bit-identical** to the naive loop (5k benchmark).
14. A100 cliff fixed — bf16 atomics were emulated (105× slower); fp32 accumulation makes fused win on both architectures.
15. int32 index buffer — halved the replicated index memory (40 → 20 GB/GPU at 1M scale).
16. Weight-transform hoist — stopped recomputing (and storing) the weight transform every timestep: 1.87× faster, −72.5% memory, and **100k neurons went from OOM-at-every-GPU-count to PASS**.
17. FSDP `cpu_init` + per-synapse wrapping — 1M no longer dies during initialization; the remaining 1M blocker is measured (replicated indices don't fit 40 GB cards → needs 80 GB cards or index sharding).

**Boundaries we found (measured, handed to the research track)**
18. The stress configs cannot learn from raw pixels at *any* depth or gain — the shunt's ~√K per-stage averaging erases input signal (entry layer alone: ~28×). This is architecture, not kernels: kernel outputs are bit-identical to the reference. Screen candidate morphologies with `harness/probe_signal_death.py` (~2 min) before spending training runs.
19. The backward gather kernel is *proven* optimal — all three alternative designs were built, measured, and lost (0.47×, 0.06×, 0.4×). Its 33% occupancy is the price of atomic locality, not a defect.

## Docs

- `docs/rnn_scaling_benchmark.md` — the ledger: every number above with its method and caveats.
- `docs/ff_gather_roofline.md` — the kernel investigation, including the failed hypotheses.
- `docs/OPEN_WORK.md` — what is deliberately not done (A100 crossover retune, K≠40 sweep, 12L trainability, 1M steady state).

## Closing result (2026-08-05)

The engineering contract — *the machinery changes nothing about what training does* —
was verified by a four-arm trajectory overlay on the real trainer, on both Blackwell
and A100 (`harness/closeout_overlay.sh`): the loss-curve deviation induced by the full
optimization stack, and separately by swapping the gather kernel (cuda↔fused), is
smaller than or equal to the deviation between two identical unoptimized runs.
The feedforward single-GPU engineering track is closed; exceptions and handoffs are
listed in `docs/rnn_scaling_benchmark.md` under "CLOSED".
