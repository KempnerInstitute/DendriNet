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
