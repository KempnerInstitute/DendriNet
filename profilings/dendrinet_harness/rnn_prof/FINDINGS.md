# Batch-1 recurrent gather: gather-bound or launch-bound? (issue #383)

Measured 2026-07-30 on branch `ryanconfig`. Config
`profilings/dendrinet_harness/scale_ab/scale_5k_5k_off.yaml` (dendritic RNN,
4000 E + 1000 I, 10 `IndexedSparseLinear` layers, 25.0M synapses per timestep),
batch 1, T = 58, bf16. Two GPUs: RTX PRO 6000 Blackwell (sm_120) and
A100-SXM4-40GB (sm_80). Harness: `profile_rnn_gather.py` (repo root).

## Answer: both, split cleanly by phase

**Forward is launch-bound. Backward is work-bound.** They are separate problems
with separate fixes, and the profile separates them unambiguously.

| | Blackwell fwd | Blackwell bwd | A100 fwd | A100 bwd |
|---|---|---|---|---|
| GPU busy | **37%** | 98% | **31%** | ~100% |
| kernels / iter | 13,018 | 24,356 | 13,018 | 24,356 |
| median kernel | 1.4 us | 1.8 us | 2.8 us | 3.2 us |
| kernels under 5 us | 92% | 75% | 75% | 61% |

Three independent measurements agree:

1. **GPU-busy fraction.** The forward leaves the GPU idle 63–69% of the time
   while the CPU issues 13,018 kernels, 92% of which run under 5 us.
2. **Batch scaling.** Launch cost is per-call and batch-independent; work is not.
   Forward wall time is *flat* from batch 1 to batch 4 (Blackwell 71.6 / 71.3 /
   70.3 ms; A100 191.7 / 189.5 / 188.8 ms) and only starts scaling at batch 8.
   Backward scales linearly over the same range (scaling exponent ~0.42).
3. **CUDA-graph capture.** The full 58-step unroll captures and replays without
   modification. Replay removes 44–48% of forward time on Blackwell and 66–69%
   on A100 — that removed fraction *is* the launch overhead.

```
CUDA-graph ceiling (forward, no_grad, batch 1, T=58)
  Blackwell  loop   62.05 ms -> 34.61 ms   1.79x   (44.2% was launch overhead)
  Blackwell  fused  67.04 ms -> 34.86 ms   1.92x   (48.0%)
  A100       loop  163.27 ms -> 55.79 ms   2.93x   (65.8%)
  A100       fused 178.70 ms -> 55.27 ms   3.23x   (69.1%)
```

## The A100 fused regression was a dtype bug, not a hardware property

The previously recorded "fused is 8.8x slower on A100 but 1.51x faster on
Blackwell" was **one kernel**: `at::native::indexFuncLargeIndex<c10::BFloat16>`,
the `index_add_` that scatters `grad_x` in `_ChunkedSparseGatherMulSum.backward`.

Each of the ~1600 synapses per input slot issues an atomic add. bf16 `atomicAdd`
is emulated on sm_80 and collapses under that contention. Isolated at the real
layer shapes (`bench_index_add.py`):

| GPU | bf16 | fp32 | ratio |
|---|---|---|---|
| Blackwell | 1.19 ms | 0.61 ms | 1.9x |
| A100 | **170.10 ms** | **1.61 ms** | **105x** |

A100 *fp32* (1.61 ms) is within normal generational distance of Blackwell fp32
(0.61 ms). The entire arch-dependence was the bf16 atomic.

**Fix** (`synapse/fused_gather.py`): accumulate `grad_x` in fp32 when the input
is reduced-precision, then cast back. This is also the numerically correct
choice — aten's own `indexing_backward` accumulates in `opmath_t = float`, so
fp32 here *matches* the eager baseline rather than diverging from it. Verified:
forward bit-identical, `grad_w` bit-identical, and in bf16 the fused `grad_x` is
now **closer** to the fp64 reference than the loop baseline (2.85e-3 vs 3.77e-3).

## Results after the fix

Per-step (profiler, batch 1, T=58, bf16):

| GPU | path | fwd ms | bwd ms | step ms | vs loop | peak MB |
|---|---|---|---|---|---|---|
| Blackwell | loop | 95.5 | 401.2 | 496.7 | 1.00x | 6492 |
| Blackwell | fused (before) | 96.5 | 148.0 | 244.4 | 2.03x | 3429 |
| Blackwell | fused (after) | 97.4 | **128.8** | **226.2** | **2.20x** | 3450 |
| A100 | loop | 229.5 | 718.0 | 947.5 | 1.00x | 6492 |
| A100 | fused (before) | 236.6 | 6896.4 | 7132.9 | 0.13x | 3429 |
| A100 | fused (after) | 235.4 | **255.4** | **490.8** | **1.93x** | 3450 |

End-to-end, real FSDP trainer, 1 epoch — loss bit-identical (1.4123 train /
1.2014 valid) in every row:

| GPU | gather off | fused | speedup |
|---|---|---|---|
| Blackwell | 45.43 s | 28.41 s | 1.60x |
| A100 | 86.92 s | **63.07 s** | **1.38x** (was 758.56 s) |

**Fused now wins on both architectures.** The per-arch split is unnecessary.

## Decisions

1. **Turn `indexed_use_fused_gather` ON for the RNN configs, on every arch.**
   Applied to `configs/dendritic_rnn/large_scale_500k_5k/{base,canary}.yaml`.
   `triton` and `cuda` stay OFF: both remain far slower in the batch-1 recurrent
   regime, and because `cuda` takes precedence over `fused`, enabling it would
   silently disable the fast path. The canary run also shows the cuda path
   *changes the loss* (0.7073 vs the 0.7078 that both loop and fused produce),
   which is a second reason to keep it off here.

2. **#383 is now the CUDA-graph work, not a new backward kernel.** With the dtype
   fix in, the forward's launch overhead is roughly half the remaining step
   (Blackwell 97.4 of 226.2 ms; A100 235.4 of 490.8 ms), and the measured
   CUDA-graph ceiling recovers most of it. The connection indices are fixed for
   the life of the model and the shapes are static, so the unrolled cell is
   directly capturable — already demonstrated here. Expected: roughly 1.4x
   (Blackwell) to 1.5x (A100) on top of the current numbers, for ~3x total
   against the loop baseline. Use `torch.cuda.make_graphed_callables` (or
   `torch.compile(mode="reduce-overhead")`) so forward *and* backward are
   captured; note the untraced eager forward (95–97 ms) is ~35 ms above the
   no_grad forward (62–67 ms), and that gap is autograd graph construction, which
   graphing also removes.

3. **A recurrent-native backward kernel is worth doing, but second.**
   `indexFuncLargeIndex` is still the single fattest backward kernel (64 ms
   Blackwell / 138 ms A100, ~25% of backward). Since the index pattern is fixed,
   a precomputed CSR/segment structure would replace both the runtime radix sort
   (loop path) and the atomics (fused path) with a contention-free segmented
   reduction. Cost: an extra index-sized buffer, which is material at the 1e6
   target where `idx` is already ~40 GB.

4. **Indices are `int64`, and that dominates gather bandwidth.** Per forward:
   11.6 GB of index traffic against 2.9 GB of bf16 weights. `int32` indices
   would halve the gather's dominant memory term *and* halve the 40 GB `idx`
   allocation at the 1e6 target. Cheap, orthogonal, worth its own issue.

## Addendum: why the CUDA/Triton paths lose here — and how to fix them

The goal is custom CUDA/Triton kernels on *all* RNN configs, so the follow-up
question is whether those paths are fundamentally wrong for the batch-1 recurrent
regime or merely mistuned. **They are mistuned.** Measured per call at the real
layer shapes (`bench_gather_paths.py`, `bench_wpo_tile.py`, `bench_wpo_tile_bwd.py`).

The forward launches `grid = (ceil(O / P), M)` with **P hard-coded to 2048 outputs
per block** (`_pick_wpo_params`), one warp per output. That was tuned on
feedforward layers where `O` is 1e4–1e5. The dendritic RNN's proximal and somatic
compartments have the opposite geometry — `O` = 1000–4000 with `K` = 1168–3968 —
so at batch 1 the grid collapses to **1–2 blocks on a 188-SM GPU** while each warp
serially walks K. The measured slowdown tracks block count exactly:

| in | out | K | blocks @P=2048 | cuda kernel vs loop |
|---|---|---|---|---|
| 4000 | 100000 | 64 | 49 | 1.08x |
| 4000 | 20000 | 240 | 10 | 5.9x |
| 4000 | 4000 | 1168 | 2 | 12.7x |
| 4000 | 1000 | 3968 | **1** | **20.6x** |

Sweeping the tile per shape (Blackwell, batch 1, per timestep, all 10 layers):

| | loop / fused | shipped P=2048 | best tile | tile tuning recovers |
|---|---|---|---|---|
| forward | 0.445 ms (loop) | 1.541 ms (3.46x **slower**) | **0.091 ms (4.9x faster)** | **17.0x** |
| backward | 2.268 ms (fused) | 92.95 ms (41x **slower**) | **1.983 ms (1.14x faster)** | **46.9x** |

The best tile is roofline-class: 0.091 ms/timestep against a ~0.11 ms bandwidth
bound for the traffic involved. The optimal P falls with `O` exactly as the
occupancy argument predicts (P=128 at O=100000 down to P=2 at O=1000).

Per-call dtype conversion is a real but **secondary** cost. Both wrappers do
`weight.to(fp32)` and `idx.to(int32)` on every call even though both are invariant
across all 58 timesteps — 26.1 GB of conversion traffic per forward against 8.7 GB
of useful gather traffic. But measured, that is only **10% (Blackwell) / 24%
(A100)** of the CUDA path's cost. The tile is the dominant term.

**So: yes, custom kernels can win on RNN configs — the work is a tile heuristic,
not a new kernel.** Order of operations:

1. **Make the tile adaptive** in `_pick_wpo_params` / `_pick_bwd_wpo_params`:
   choose P so that `ceil(O/P) * M` is a few times the SM count, instead of a
   constant 2048. This is the prerequisite for enabling CUDA on any RNN config,
   and it is the single highest-leverage change available (17–47x on the kernels).
2. **Cache the int32 index** on the module (it never changes) and add a bf16
   kernel path, removing the remaining conversion tax.
3. **Then** CUDA graphs, and only then does the forward kernel win become visible
   in wall clock — at batch 1 the forward is launch-bound, so today a 17x faster
   forward gather saves GPU time that is already hidden behind the launch gap.
   The backward gain does translate immediately, since backward is GPU-bound.

Until step 1 lands, `fused` stays the correct default for the RNN configs: it is
the only path that is fast at these shapes without a retune.

Two caveats for whoever picks this up. The CUDA path computes in fp32 while
loop/fused compute in bf16, so it is not bit-equivalent — the canary loss is
0.7073 rather than 0.7078. That is *more* precise, not wrong, but it means the
cuda path cannot be A/B'd as a pure perf change. And `cuda_gather_available()`
returns False and **silently falls back to the loop** when the extension is not
built, so a config with `use_cuda_gather: true` can quietly run the slow path with
no warning; source `profilings/weight_transform_opt/setup_cuda_env.sh` first and
check the flag. Triton was not re-tuned here: at batch 1 it took >25 min for 5
steps (>250x the loop), consistent with the earlier epoch-level result.

## Reproducing

```bash
export PYTHONPATH=src
# phase split, GPU-busy fraction, per-op and per-kernel attribution
python profile_rnn_gather.py --config profilings/dendrinet_harness/scale_ab/scale_5k_5k_off.yaml \
    --path fused --mode profile --batch 1 --seq-len 58
# launch-bound test: flat in batch => launch-bound, linear => work-bound
python profile_rnn_gather.py --config ... --path fused --mode sweep
# hard ceiling on what CUDA-graphing the cell can recover
python profile_rnn_gather.py --config ... --path fused --mode cudagraph
# the isolated atomic-dtype result, and the equivalence checks
python profilings/dendrinet_harness/rnn_prof/bench_index_add.py
python profilings/dendrinet_harness/rnn_prof/check_fused_equiv.py
python profilings/dendrinet_harness/rnn_prof/check_module_equiv.py
```

Raw chrome traces are not committed (135 MB); `--mode profile` regenerates them.
Per-run summary JSON and console logs are under `a100/`, `blackwell/`,
`a100_fp32acc/`, `blackwell_fp32acc/`, and `logs/`.
