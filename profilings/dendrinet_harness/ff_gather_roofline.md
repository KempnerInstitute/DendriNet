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

**CORRECTION — it does NOT remove the `in_features` ceiling.** I claimed it did, on the
grounds that the staged kernel cannot launch above IN = optin/4 while the un-staged one
can. Both halves of that were misleading:

1. Production never hard-failed there. `_wpo_ok` already routed above-ceiling shapes to the
   **v2 shared-tiled forward**. The ceiling was a performance cliff, not a wall.
2. Measured against that v2 forward, the un-staged kernel **loses**: 0.96× / 0.93× / 0.89×
   at (33,344→48,000), (33,344→16,000), (50,688→16,000). v2 tiles over IN and keeps idx+w
   in shared, which still pays when the row is huge.

`_pick_fwd_variant` now returns `staged` above the ceiling, which routes through `_wpo_ok`
to v2 — the measured best. The 1.086× end-to-end win is unaffected; it comes entirely from
below-ceiling shapes.

**Caveats.** Measured on Blackwell only. A100 has a larger optin (166,912 B), so `blk/SM`
at IN=16,000 is 2 and the rule selects `staged` — a no-op there, not a regression, but the
crossover has not been re-measured on A100. Only K=40 was swept; the reuse term carries K
on mechanism, not measurement. The **backward is untouched** — `gather_gradx_stage_kernel`
needs its shared buffer for fast atomics, so the same trick does not transfer directly.

Also: my earlier "forward tile P is closed at 1.00×" used a ladder topping out at the
shipped 2048. The staged optimum is actually **P=4096** (14.84 vs 15.07 ms, ~1.5%). Minor,
but the ladder failed to bracket the optimum from above.

### BACKWARD: investigated, CLOSED — two hypotheses killed (2026-07-30)

`ncu` splits the backward cleanly at the dominant shape:

| kernel | time | occupancy | limiter | DRAM | SM |
|---|---|---|---|---|---|
| `gather_gradx_stage` | **21.01 ms** | 33.26% | **shared = 1 block** (regs allow 4) | 1.21% | 22.4% |
| `gather_gradw_transpose` | 3.16 ms | — | grid = 141 blocks | 7.38% | 41.3% |

grad_x is **87% of the backward**, and looks exactly like the forward did: 33% occupancy,
capped by an IN-float shared row. Both obvious fixes were tried, and **both lost**.

**Hypothesis 1 — "the row flush is the cost." FALSE.** Each block flushes the whole staged
row with global atomics, and there are `ceil(O/P)` blocks per m, so the flush is
`ceil(O/P)·IN·M` = **1.16 billion** atomics at the shipped P. Raising P cuts that linearly.
Swept P upward to O itself:

| P | blocks/m | flush atomics | ms | vs shipped |
|---|---|---|---|---|
| 1024 | 141 | 1.16 G | **20.46** | 1.00× |
| 4096 | 36 | 0.29 G | 27.32 | 0.75× |
| 16384 | 9 | 0.07 G | 57.23 | 0.36× |
| 147456 | 1 | 0.01 G | 364.81 | **0.06×** |

100× fewer atomics, **17× slower**. The flush is nearly free; parallelism is everything.
The shipped picker is already optimal — **1.00× weighted**, do not re-tune it.

**Hypothesis 2 — "the forward's no-stage fix transfers." FALSE.** Added
`gather_gradx_ns_kernel` (no shared, direct global fp32 atomics) and swept both:

| in | out | staged | no-stage | ratio |
|---|---|---|---|---|
| 16,000 | 144,000 | 20.42 ms | 41.50 ms | 0.49× |
| 16,000 | 48,000 | 7.42 ms | 13.47 ms | 0.55× |
| 3,072 | 144,000 | 8.05 ms | 44.14 ms | **0.18×** |
| | | | **weighted** | **0.47×** |

**2.1× slower.** Gradients agree (grad_x rel ≤ 9.1e-07; grad_w bit-identical), so this is
purely a speed verdict.

**Why the forward and backward differ.** The forward's shared row is a *read cache* for
data L2 already holds at 99.44% — deleting it costs only extra L2 requests. The backward's
shared row is an *atomic accumulator*, and shared atomics are far cheaper than global ones
under contention. The staged kernel is deliberately buying atomic locality with occupancy,
and that trade is strongly favourable. Its 33% occupancy is a *price*, not a defect.

**Third approach already rejected upstream**: the module docstring records a v3
scatter→gather transpose of grad_x, measured 2.2–2.4× slower. So all three routes —
global atomics, index transpose, shared staging — have now been measured, and the shipped
one wins.

**But the rejected kernel is not dead code — it wins ABOVE the ceiling.** Where the staged
grad_x cannot run at all, the comparison is not ns-vs-staged but **ns-vs-v2**, and there ns
wins:

| IN | out | v2 (was) | ns | speedup |
|---|---|---|---|---|
| 33,344 | 48,000 | 7.78 ms | 6.51 ms | **1.19×** |
| 33,344 | 16,000 | 2.60 ms | 2.33 ms | 1.12× |
| 50,688 | 16,000 | 2.61 ms | 2.35 ms | 1.11× |

Gradients agree to ≤3.3e-07. `_CudaSparseGather.backward` now uses
`gather_bwd_wpo_cuda2(..., gx_variant=1)` above the ceiling instead of `gather_bwd_cuda`.

**Note the asymmetry: above the ceiling the forward wants v2 and the backward wants ns.**
The two directions do not agree, and assuming they would is what produced the wrong forward
rule in the first place.

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
