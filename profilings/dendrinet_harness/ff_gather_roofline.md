# Is the feedforward gather done? No. (2026-07-30, Blackwell RTX PRO 6000)

Written to correct a claim I made and then disproved: that feedforward speed was
exhausted because "the gather is 75–92% of the step, so Amdahl caps everything else."

Two things were wrong with that.

## 1. It was benchmarked against the wrong baseline

The end-to-end runs behind that claim used the **`fused`** path. On the same 12-layer
16k stress config, batch 512 (`ff_stress/blackwell/extended`):

| path | step (ms) | vs fused |
|---|---|---|
| fused | 2845.5 | 1.00× |
| triton | 1816.8 | 1.57× |
| **cuda** | **1021.2** | **2.79×** |

A 2.8× factor was sitting in a config flag. Any "we've run out of speed" statement
measured on `fused` is measuring the wrong denominator.

The Amdahl *mechanism* was real, and re-testing confirmed it: the positive-tanh sigmoid
identity gives **1.007×** on `fused` but **1.034×** on `cuda` — 5× larger effect once the
dominant term shrinks. The logic held; the baseline didn't.

Also: the 75–92% figure came from the **RNN** profile, not a feedforward measurement.
It should not have been carried across architectures.

## 2. The gather kernel is nowhere near roofline

`ncu` on `gather_fwd_wpo_kernel` at the dominant shape (in=16,000, out=144,000, K=40,
batch 512), grid 71×512, block 512:

| metric | value |
|---|---|
| DRAM throughput | **1.11% of peak** (329 MB moved) |
| SM throughput | **35.3% of peak** |
| achieved warp occupancy | **33.1%** |
| L2 hit rate | **99.44%** |
| `launch__occupancy_limit_shared_mem` | **1 block** |

Backward is the same shape of result: `gather_gradx_stage` 1.21% DRAM / 22.4% SM,
`gather_gradw_transpose` 7.45% DRAM / 41.3% SM, both at 33% occupancy.

**Not bandwidth-bound. Not compute-bound. Occupancy-capped at 1 block/SM by shared
memory** — 512 threads = 16 warps, against an SM that holds far more.

### The lever this exposes

The shared-memory staging is what costs the occupancy, and the **99.44% L2 hit rate says
it is buying almost nothing** — the x-rows it stages are already resident in L2. A variant
that stages less (or nothing) would fit several blocks per SM.

Occupancy 33% → 100% is not automatically 3×; latency hiding saturates. But 1.1% DRAM and
35% SM is not a finished kernel by any reading, and the limiter is a design choice the
cache counters suggest is unnecessary.

## What IS closed

- **Forward tile `P`.** Swept the full ladder at all seven real shapes: best-per-shape
  beats the shipped P=2048 by **1.00× weighted**. Unlike the backward (which was badly
  mistuned and gave 1.32×), the forward tile is already right. Do not re-litigate.
- **The Python layer.** Passes A–D found no further Python-level speed win, and the
  largest allocation line in the model moved the step by 0.7–3.4%. That result stands —
  it was a statement about Python, and it should not have been generalized to the kernel.

## Method note

`roofline_gather.py` computes bandwidth from an analytic traffic model, and the answer
swings **1.7% → 12% → 53%** of peak depending on whether the x-gathers are assumed cached,
re-read per block, or uncached. That spread is too wide to conclude from; the model was
doing all the work. Use `ncu_gather_probe.py` under `ncu` instead — the counters settled
in one run what the model could not settle at all.
