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

### RESOLVED (2026-07-30) — un-staged forward variant

Added `gather_fwd_wpo_ns_kernel`: identical warp-per-output structure, **no shared
staging**, x read through `__ldg` and served by the L2 that was already at 99.44%.
Additive — the staged kernel is untouched.

It does not always win: it issues ~5× more L2 requests, which only pays back when the
grid has enough output work. Swept a 5×5 (IN, O) grid, batch 512, K=40 — **won 3 cells,
lost 22**:

| IN \ O | 2,000 | 8,000 | 16,000 | 48,000 | 144,000 | blk/SM |
|---|---|---|---|---|---|---|
| 3,072 | 0.91 | 0.92 | 0.91 | 0.92 | 0.89 | 8 |
| 4,000 | 0.90 | 0.89 | 0.88 | 0.86 | 0.88 | 6 |
| 8,000 | 0.78 | 0.74 | 0.63 | 0.71 | 0.82 | 3 |
| 16,000 | 0.84 | 0.81 | 0.87 | **1.11** | **1.50** | 1 |
| 24,000 | 0.73 | 0.70 | 0.72 | 0.84 | **1.06** | 1 |

`blk/SM <= 1 and reuse >= 120` (reuse = O·K/IN) classifies all 25 cells correctly. The
boundary **moves with IN** — at 16,000 it crosses by O=48,000, at 24,000 not until
144,000 — because a larger x-row is less L2-resident. That is why the rule has two terms
and not a threshold on either axis alone.

Shipped as `_pick_fwd_variant`, default `auto`, overridable with
`DENDRINET_GATHER_FWD_VARIANT=staged|nostage`.

**End-to-end, 12-layer stress config, batch 512, cuda path:**

| | step (ms) | peak (MB) |
|---|---|---|
| staged (before) | 908.6 | 28184.1 |
| auto picker | 837.0 | 28216.9 |
| **ratio** | **1.086×** | 0.999× |

Network output **bit-identical** (`torch.equal` → True across all 60 sparse layers).

**It also removes the `in_features` ceiling.** At IN=33,344 the staged kernel fails to
launch (`invalid argument`, needs 133,376 B > optin 101,376); the un-staged one runs. That
ceiling gated scaling `in_features` past 25,344 on Blackwell.

**Caveats.** Measured on Blackwell only. A100 has a larger optin (166,912 B), so `blk/SM`
at IN=16,000 is 2 and the rule selects `staged` — a no-op there, not a regression, but the
crossover has not been re-measured on A100. Only K=40 was swept; the reuse term carries K
on mechanism, not measurement. The **backward is untouched** — `gather_gradx_stage_kernel`
needs its shared buffer for fast atomics, so the same trick does not transfer directly.

Also: my earlier "forward tile P is closed at 1.00×" used a ladder topping out at the
shipped 2048. The staged optimum is actually **P=4096** (14.84 vs 15.07 ms, ~1.5%). Minor,
but the ladder failed to bracket the optimum from above.

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
