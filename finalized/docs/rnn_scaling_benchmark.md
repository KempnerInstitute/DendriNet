# RNN Scaling Benchmark — baseline (the numbers to beat)

Reference metrics for dendritic-RNN training **speed** and **memory**. Every future
optimization is measured against the **BASELINE** row (fused + int32 idx). Two questions
this ledger answers: *"is my change faster/lighter than baseline?"* and *"how far up the
scale ladder does it fit?"*

## Method
- Configs: `profilings/dendrinet_harness/scale_ab/scale_5k_5k_{off,fused}.yaml`
  (5,000 neurons = 4,000 E + 1,000 I, 5k contacts/neuron, Romo task, FSDP).
- Metric: trainer-reported **epoch wall-clock** + **peak GPU memory** (nvidia-smi poll), 2 trials averaged.
- Hardware: **1 GPU** (FSDP world_size=1 → NO_SHARD). Multi-GPU rows note nproc.
- Code: commit `cbaa88eb` — gather path = **fused** (`use_fused_gather`), int32 index buffer, fp32 grad_x accumulation.
- Equivalence: fused vs loop loss is **bit-identical** (1.4123) — pure perf.

## BASELINE — 5k/5k, Blackwell RTX PRO 6000 (2026-07-30)

| scale | GPU | path | epoch (s) | peak (MB) | speed vs loop | mem vs loop | loss |
|---|---|---|---|---|---|---|---|
| 5k/5k | Blackwell | loop (naive) | 47.29 | 8087 | 1.00× | — | 1.4123 |
| 5k/5k | Blackwell | **fused+int32 — BASELINE** | **30.16** | **5189** | **1.57× (−36.2% time)** | **−35.8%** | 1.4123 |

*"Our work made the 5k RNN 1.57× faster (36% less epoch time) at 36% less peak memory, loss bit-identical."*
(Per-step isolated gather speedup is higher — ~2.2× — but epoch-level is diluted by FSDP/data overhead; this ledger tracks the honest end-to-end number.)

## NEW — recurrent weight-transform hoist (2026-07-30, commit `6acdd5d4`)

`indexed_cache_recurrent_weight`. `IndexedSparseLinear` recomputed
`apply_weight_transform(pre_w)` every timestep and autograd saved each copy, costing
`T · out_features · K` of **activation** memory. `pre_w` is constant within a forward, so
this is loop-invariant. Critically this term is activation, not parameter, memory — **FSDP
does not shard it**, so every rank pays it in full and adding GPUs does not help.

| scale | GPU | path | epoch (s) | peak (MB) | speed vs loop | mem vs loop | loss |
|---|---|---|---|---|---|---|---|
| 5k/5k | Blackwell | loop (naive) | 46.27 | 8103 | 1.00× | — | 1.4123 |
| 5k/5k | Blackwell | fused+int32 (prev baseline) | 30.16 | 5189 | 1.57× | −35.8% | 1.4123 |
| 5k/5k | Blackwell | **fused+int32+hoist — NEW BASELINE** | **24.70** | **2227** | **1.87× (−46.6% time)** | **−72.5%** | 1.4123 |

*"1.87× faster at 72% less peak memory than naive, loss bit-identical."* Supersedes the
1.57× / −36% row. Caveat: **1 successful trial** for the hoist row (loop is 2 trials,
47.11/45.43); the 2nd hoist trial was lost to the `workspace_mb` NameError below, not to a
real failure. Re-run for a 2-trial average when HEAD is repaired.

**Memory-model validation** (`rnn_prof/check_weight_cache.py`, 5k/5k, A100):

| | slope (MB/timestep) | T=58 peak (MB) |
|---|---|---|
| cache OFF | **51.87** (predicted `nnz·2B` = 50.0) | 3737 |
| cache ON | **2.08** | 1018 |

Forward **bit-identical** (max|off−on| = 0.0). Gradients differ by accumulation order only:
fp32 **2.98e−07**, bf16 1.18e−02 (one shared node accumulating T grads vs T nodes
contributing one each). The fp32/loop path has a *second* T-scaling term — the saved
`selected` gather intermediate, measured 197.7 MB/timestep ≈ `2·nnz·4B` — which the fused
backward already recomputes, so **fused + hoist removes both**.

## Scale gates — measured, not extrapolated

| scale | hardware | nproc | result | peak/GPU (MB) | epoch (s) |
|---|---|---|---|---|---|
| 100k/5k | 4× A100-40GB | 2 | **OOM** (before hoist) | 40375 | — |
| 100k/5k | 4× A100-40GB | 4 | **OOM** (before hoist) | 40437 | — |
| 100k/5k | 4× A100-40GB | 2 | **PASS** (hoist on) | **21721** | 112.60 |
| 100k/5k | 4× A100-40GB | 4 | **PASS** (hoist on) | **17737** | 56.23 |
| 1M/5k | 4× A100-40GB | 4 | **OOM** (hoist on) | 38995 | — (tried to alloc 2.98 GiB) |

**100k is unblocked** — it previously OOM'd at *every* GPU count; note 40375 → 40437 MB going
2→4 GPUs, i.e. the binding term did not shard at all, which is what identified it as
activation memory. Post-hoist it does respond to N.

Fitting `mem(N) = F + K/N` to the two passing 100k points: **F ≈ 13.75 GB fixed per GPU**,
K ≈ 15.9 GB sharded. The replicated int32 `idx` is only 1.86 GB of that F — so **~11.9 GB of
the per-GPU floor is still unattributed** (activations + FSDP working buffers) and is the
next thing to decompose, *before* sharding `idx`.

**1M does not fit 4× A100-40GB.** Scaling the 100k constants ×10 puts the fixed term near
137 GB/GPU, which no GPU count reduces. 1M therefore needs the fixed floor attacked, not
more GPUs — confirming the replicated-buffer / model-parallel item as the gate for 1M, but
`idx` alone (18.6 GB at 1M) is not sufficient to explain the floor.

## NEW — positive-tanh sigmoid identity (2026-07-30)
`(tanh(z)+1)/2` is exactly `sigmoid(2z)`, but the naive spelling runs two extra
full-size elementwise kernels and allocates a temporary for each. Pass D of the hot-path
audit attributed **39.9% of live allocation** to that one line. Flag: default OFF,
`DENDRINET_POSITIVE_TANH_SIGMOID=1` or `set_positive_tanh_sigmoid(True)`.

Isolated op (Blackwell, 8 shape/dtype combos): fwd **1.43×**, bwd **1.12×**, peak **1.17×**.

End-to-end, 12-layer 16k stress config, batch 512, fused path:

| | step (ms) | peak (MB) |
|---|---|---|
| identity off | 2843.6 | 29831.8 |
| identity on | 2824.9 | 25021.8 |
| **ratio** | **1.007×** | **1.192×** |

**This is a memory optimization, not a speed one.** The op gets 1.43× faster and the step
gets 0.7% faster — Amdahl, the gather still dominates. But it returns **4.8 GB at batch
512** for free. That is the same class as the hoist: an *activation-layout* fix aimed at
the fixed per-GPU floor, which is the 1M gate. Correct an earlier claim of "1.46× faster,
1.33× less memory" — that was the isolated forward, and it does not survive end-to-end.

Equivalence (`check_positive_tanh_equiv.py`): **bit-identical in fp64** for every touched
activation, so the rewrite is structurally exact and the fp32 gap is pure rounding
(1.2e-07 value, 6.0e-07 grad; 2.4e-07 through the whole 12-layer network). 200 Adam steps
down each path: final loss relative difference **0.00e+00**, params agree to 2.4e-07 — it
does not compound. `param_linear_tanh` is deliberately excluded: its `(1-exp(-3x))` factor
reaches ~1e5 and amplified that rounding to 4e-4 on parameter gradients, and no config
selects it.

## NEW — reactivation retains one tensor instead of three (2026-07-30)
Allocation profiling on the **cuda** path (the earlier pass D had been forced onto `fused`
at batch 64 by a harness bug) put `parametric.py` at **57.6% of live allocation** —
14,943 MB, the largest single line in the model. Two changes, both shipped on by default:

1. `(tanh(z)+1)/2 -> sigmoid(2z)` (`1d1b1159`) — removes transient temporaries.
2. `_PositiveTanhCentered` (`05063123`) — one autograd Function retaining **only** `y`,
   reconstructing `z = logit(y)` in backward for `grad_m`. Safe *because of* saturation:
   where the fp32 sigmoid pins to 1.0 and `logit` returns inf, `y(1-y)` is exactly 0 and
   PyTorch's own sigmoid backward drops the same terms, so bounding the logit argument
   reproduces the reference rather than approximating it.

12-layer 16k stress config, batch 512, Blackwell, cuda path:

3. `_ShuntDividePositiveTanh` — fuses the **shunt divide** into the same Function. The
   accumulation chain was a red herring (`a+b` retains nothing); the cost was `DivBackward`
   pinning *both* operands of `voltage = n/(d+eps)`. Fused, the numerator is never needed
   (`dL/dn = dL/dv/(d+eps)`) and `v` is reconstructible from `y`.

| mode | step (ms) | peak (MB) | saved |
|---|---|---|---|
| eager | 857.7 | 28,184.1 | — |
| + identity | 820.5 | 23,384.7 | 4,799 MB |
| + fused activation | 841.1 | 18,551.5 | 9,633 MB (1.52×) |
| **+ fused shunt** | 844.8 | **14,343.3** | **13,841 MB (1.96×)** |
| + int32 indices (C6) | — | **13,370.1** | **14,814 MB (2.11×)** |

C6 removes the gather wrapper's int32->int64->int32 index round trip, which had been
retaining an 889 MB int64 copy per step and partly undoing the int32-index scaling lever.
Verified on Blackwell AND A100-40GB (`verify_c5_c6.py` PASS on both): retained dtype read
off the autograd node is int32, gradients agree with the fused reference to <= 4e-07. The
retained copy costs **0.0 MB**, not the predicted ~445 -- `.to(torch.int32)` on an
already-int32 contiguous tensor returns the same tensor, so it aliases the module's buffer.

**The A100-40GB now runs this config**, which needed 28.2 GB before.

Mechanism confirmed by re-profiling against an arithmetic one-tensor cost of 4,968.8 MB:
the reactivation went 3 retained tensors → 1, and the divide+activation pair went
**2.92 → 1.92**, with `branch_dynamics` now holding 0.00 MB. Step time stays below eager
throughout.

**This is activation memory — the term FSDP does not shard** ([[rnn-activation-memory-dominates]]),
i.e. the same class as the weight-transform hoist and aimed squarely at the fixed per-GPU
floor that gates 1M. Worth re-running the 100k multi-GPU gate against it.

Equivalence: fp64 output *and* every gradient bit-identical up to 46% saturation; fp32 gap
2.4e-07 end-to-end and does not grow with saturation; 200 Adam steps with a saturating
input end at final-loss relative difference 0.00e+00.

## Batch scaling — feedforward stress config, Blackwell (97,252 MB), cuda path

Full sweep of batch x mode. **Batch 4096 runs — and only with the complete stack.**

| batch | eager | identity | fused-act | fused-shunt |
|---|---|---|---|---|
| 128 | 7,924.8 | 6,723.4 | 5,503.5 | **4,292.6** |
| 256 | 14,361.9 | 11,949.5 | 9,518.7 | **7,096.9** |
| 512 | 27,236.2 | 22,401.9 | 17,549.2 | **12,705.5** |
| 1024 | 52,984.6 | 43,306.6 | 33,610.2 | **23,922.7** |
| 2048 | **OOM** | 85,116.0 | 65,732.1 | **46,357.1** |
| 4096 | — | **OOM** | **OOM** | **91,226.0** |

*(peak MB; OOM = did not fit 97,252 MB)*

**The maximum trainable batch quadrupled**, 1024 → 4096. Without this work the config
cannot pass batch 1024 on a 97 GB card; the 2048 and 4096 runs exist only because of it.

The ratio against eager grows with batch, because the ~1.7 GB batch-independent term is a
shrinking share of the total: 1.85x at 128, 2.02x at 256, 2.14x at 512, **2.21x at 1024**
(the largest batch where all four modes still run, so the last honest comparison).

**Speed is flat to slightly better in every cell** — 193.0 vs 204.7 ms at batch 128,
1,658.8 vs 1,680.1 at 1024. None of this trades speed for memory.

Prediction check, recorded because it was contested: fitting **21.84 MB/sample + 1,678 MB**
to the 512 and 2048 points predicted 91,100 MB at batch 4096. Measured **91,226 MB** —
0.14% off. An intermediate claim that 4096 would "most likely OOM" came from doubling the
2048 figure instead of fitting the line, which ignores the fixed term; it was wrong and the
original "tight but plausible" was right.

Raw data: `ff_stress/batchsweep/blackwell.json` (untracked — `profilings/**/*.json` is
gitignored by project convention).

## REAL-TRAINER VALIDATION (2026-07-31) — the 2.19x is real, but it is not the peak

Everything above was measured on a synthetic step (`out.float().pow(2).mean().backward()`,
no optimizer, no loss head, no data loader). This runs `train_experiments.py` for one real
epoch — Adam, cross-entropy, real CIFAR-10 — twice, differing only by
`DENDRINET_POSITIVE_TANH_SIGMOID`.

**The real trainer uses mixed precision** (`autocast` + `GradScaler`), which the fp32
profiling harness did not. The fused paths had never been tested in half precision.

| | eager | eager (control) | fused | |
|---|---|---|---|---|
| `_epoch_train` peak | **16,927.1 MB** | — | **9,646.6 MB** | **1.75×** |
| process peak (torch allocated) | 59,606.4 MB | 59,606.4 MB | 59,599.9 MB | **1.0001×** |
| epoch wall-clock | 152 s | 154 s | 153 s | neutral |
| train loss | 2.5568 | 2.5568 | 2.5557 | 4.3e-04 |
| valid loss | 2.3123 | 2.3123 | 2.3132 | 3.9e-04 |

**The control run matters.** The config sets `deterministic: false`, so a loss difference
means nothing without knowing the run-to-run spread. Two eager runs came out **bit-identical**
(2.5568 / 2.3123 both), so the fused difference is attributable — and it is 4e-04, consistent
with half-precision rounding rather than a defect.

**The training phase does respond: 1.75×.** That is consistent with the synthetic 2.02× at
batch 256 once ~2.6 GB of mode-independent optimizer state (params + grads + Adam moments)
is added to both sides.

**But the process peak does not move at all.** `_epoch_train` peaks at 9.6 GB against a
59.6 GB process peak, and the two modes differ by 6.5 MB. **Something outside the training
and validation phases allocates ~60 GB** — six times the training step — and it is
unaffected by any of this work. It is not in `_epoch_train` or `_epoch_valid_stats`, and the
log shows no analysis phase after training. **Unlocated; this is now the binding constraint
on this config's memory, and it is larger than anything fixed today.**

Half-precision equivalence, checked because the trainer forced the question: no NaN/Inf at
any saturation fraction up to 60%, output bit-identical, and gradient deviations of 1–2.5 ulp
in both bf16 (7.6e-03–2.0e-02) and fp16 (1.0e-03–2.4e-03) — at the format's own floor, and
**not growing with saturation**, so the `logit(y)` reconstruction does not degrade in half
precision.

## The unlocated 60 GB: FOUND and FIXED (2026-08-04)

The real trainer's process peak (59.6 GB vs a 9.6 GB training step) was localized by a
whole-run allocator-trace replay: **`effective_synapse_snapshot`**, called per pathway from
the unconditional post-training save path, materializes ~5 dense `[144,000 x 16,000]`
matrices per indexed layer (8,911.1 MB each) to produce five integers of connectivity
reporting. ~55 GB of dense copies of layers whose entire design is K=40 sparsity —
mode-independent, after training, 6x the training step.

Fix (`fd9533a6`): `_indexed_record_counts` computes the same scatters in <=256 MB row
chunks. Bit-exact by construction (chunked scatter, not analytic re-derivation); EXACT
MATCH on duplicate-index / forbidden / relu traps, and `model_resources.json` from the
real 60-pathway model is byte-identical across paths. Legacy path retained behind
`DENDRINET_DENSE_RESOURCE_RECORDS=1` and for cached-mask / candidate_mask() / stochastic
modules.

| real trainer, 12L/16k, batch 256, 1 epoch | before | after |
|---|---|---|
| process peak (trace replay) | 59,558 MB | **9,598 MB (6.2x)** |
| where the peak occurs | post-training reporting | **the training step itself** |

With this, the process peak IS the training-step peak, whose composition is the known
structural floor (y, d+eps, optimizer state). The feedforward single-GPU memory story is
now closed end to end: synthetic harness, real trainer, and save path all agree.

## cpu_init + per-synapse wrapping: 100k regression PASS, 1M OOM MOVED (2026-08-04)

Re-ran both gates with `cpu_init: true` + `auto_wrap_policy: indexed_synapses`
(jobs 37256960 / 37256964).

**100k, 4x A100-40GB — PASS, losses byte-identical** to the pre-fix run (Train 1.0037,
Valid 0.6953). Peak 17,737 → **16,033 MB/GPU**. Honest cost: epoch 56.23 → **72.25 s**
(1.29x slower) — per-synapse FSDP units mean more all-gathers. The granularity that makes
init fit is paid in communication; a coarser wrap tier is a future speed lever.

**1M, 4x A100-40GB — still OOM, but the failure MOVED**, which is the point:

| | before | after |
|---|---|---|
| dies in | `FSDP.__init__` → `_sync_module_params_and_buffers` | `_alloc_storage` (unshard all-gather, first forward) |
| meaning | every rank materializes the whole 37.6 GB model | init PASSES; steady state does not fit |
| resident at failure | — | 33.87 GB allocated + 2.38 GiB unshard ask vs 39.49 GB |

The steady-state arithmetic now measured, not estimated: replicated int32 idx (~19 GB) +
fp32 flat-param shard (~5.2 GB) + grads + bf16 working copies ≈ 34 GB **before optimizer
state even exists**. 4x A100-40GB cannot hold 1M/5k with a replicated index buffer,
independent of any further activation work.

Three exits, in increasing effort:
1. **Bigger cards**: 4x A100-80GB (partition `gpu`) or Blackwell — fits trivially; would
   give the first real 1M training epoch. Cheapest decisive next step.
2. **More ranks**: 8x A100-40 (2 nodes) — shards halve → ~29 GB + activations; likely fits
   but needs multi-node FSDP.
3. **Shard the idx buffer** — the real systems item; removes the 19 GB replication that
   dominates the floor.

## CLOSED: feedforward, single-GPU, engineering track (2026-08-05)

Closing evidence: four-arm trajectory overlay on the real trainer (5 epochs each,
`closeout_overlay.sh`), on BOTH architectures, with the acceptance criterion fixed before
running: optimization-induced deviation is judged against the eager-vs-eager control gap,
not against zero. Every arm's gather path is a logged fact, not an inference.

| max |Δloss| across all 10 points | Blackwell | A100-80GB |
|---|---|---|
| eager vs eager control (noise floor) | 1.19e-02 | 3.88e-03 |
| reactivation stack ON vs eager | **2.75e-03** | **3.81e-03** |
| fused gather kernel vs cuda kernel | **5.47e-03** | **2.29e-03** |

On both cards the optimizations' effect on training is **smaller than or equal to the
difference between two identical eager runs** — statistically invisible. Corroborating
live speed: the cuda arm trained in 90 s vs the fused arm's 277 s (3.1×), on the same
trainer, same epoch count.

**Closed** = operators bit-identical/≤4e-7; 2.19× peak memory at the proven structural
floor; batch 1024→4096; process peak = training step; save path fixed; equivalence at
operator, half-precision, optimizer-trajectory, and now real-trainer-trajectory level,
cross-architecture. Lab entry point: `finalized/`.

**Excepted** (declared, not forgotten) -- REVISED 2026-08-06, the original three items
(A100 crossover unmeasured, K=40-only sweep, P=4096) were all closed by measurement; the
remaining declared set: H200 unmeasured (takes the measured sm90 branch -- same cc/SM/
shared budget, margins may shift); H100 1-blk/SM regime (IN 29-58k) beyond the swept
grid, staged default; B200/sm100 unmeasured (its 232 KB optin routes these INs to >=3
blk/SM where the blk/SM predicate keeps staged); batch-64 tiny-O no-stage wins (~30 us
cells) declined, the picker has no batch input; Triton path correct but unoptimized
(1.6-1.8x behind cuda, kept as the no-nvcc fallback).

## Picker edge closure -- 2026-08-06
The three declared forward-picker edges, closed by measurement on three architectures
(`sweep_picker_edges.sh`, one run each -> `rnn_prof/logs/edges_{h100,blackwell,a100}`):
- **(a) H100**: new sm90 branch. Staged nearly everywhere (232 KB shared keeps >=3
  blk/SM to IN~19k); no-stage only at 2 blk/SM (whole row K<=15, the 144k cell to
  K<=60); hybrid loses everywhere (0.54-0.94x), no band. Dispatch verified live;
  shipped-band verify PASS 12/12.
- **(b) batch**: winner sets identical at batch 64/128/512 on every arch's band cells.
- **(c) K edges**: the interpolated edges were wrong BOTH ways. Blackwell: the
  reuse>=120 rule shipped 0.58-0.89x regressions (IN=24k, K=50-60) and forfeited
  1.27-1.28x wins at (16k,144k,K70/80) -- reuse heuristic RETIRED, bands are measured
  lookups. A100: the K-blind rule shipped 0.58-0.94x at high-K small-O and forfeited
  the K<=15 whole-row wins at 2 blk/SM -- refit with K-banded no-stage P.
- **Shipped-band verify** (`bench_shipped_bands.py`: sweeps report per-cell BEST P,
  dispatch ships BANDED P -- this times the exact dispatched config vs staged per
  winner cell): caught 3 A100 cells (P=128 refit) and 1 Blackwell cell (hybrid
  (48k,K60) trimmed at 0.94x).
- **sm80 hardware variants**: A100-40GB (same cc, ~20% less bandwidth) tips 5 marginal
  cells negative (0.93-0.95x) that the 80GB fit had at 0.99-1.05x. Band trimmed to the
  cells that win on BOTH variants -- costs ~nothing on 80GB, restores never-worse on
  40GB.
- Standing regression tests: `check_picker_dispatch.py` (per-arch expected winner sets;
  simulate mode needs no GPU) + `bench_shipped_bands.py` (hardware). Final states:
  H100 12 cells PASS; Blackwell 37 cells, A100 37 cells (80GB re-verify queued).

## 1M gate -- 2026-08-06 (4x H100-80GB, job 37522508)
Init + FSDP wrap + full FORWARD of the first 1M x 5k batch all succeed (the cpu_init
fix holds at scale). OOM in the first `loss.backward()` inside the fused gather:
992 MiB requested with 2.55 GiB reserved-but-unallocated and 651 MiB free -- a
FRAGMENTATION miss, not capacity (74.4 GiB truly allocated of 79.2). peak_per_gpu
80,967 MB. Retry queued with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
(allocator policy only, no numeric effect); next lever if it misses: low-precision
optimizer state (~7.8 GB/rank).

**Handed off**: raw-input learning / morphology co-design → research track (screen with
`probe_signal_death.py`); multi-GPU + 1M steady state → scaling track (init fixed; boundary
measured: replicated idx needs 80 GB cards, 8 ranks, or idx sharding).

## Open rows
| scale | GPU | path | epoch (s) | peak (MB) | notes |
|---|---|---|---|---|---|
| 100k/5k | multi-GPU | + sigmoid identity | — | — | does it cut the 13.75 GB floor? |
| 5k/5k | Blackwell | fused+int32+hoist, trial 2 | — | — | re-run for 2-trial avg |
| 5k/5k | A100 | loop / fused | — | — | cross-arch |
| 100k/5k | multi-GPU | decompose the 13.75 GB floor | — | — | **do before sharding idx** |
| 1M/5k | multi-GPU | fused + sharded idx / model-parallel | — | — | gate for 1M |
| 5k/5k | Blackwell | + 8-bit optimizer | — | — | touches sharded term only |

## Kernel chapter: CLOSED
Op-level speed/mem for the RNN is done — fused autograd (speed+mem) + int32 idx (mem).
CUDA/Triton kernels do NOT apply (slower at batch-1). torch.compile blocked. Everything
below the baseline is **distributed/memory-systems** work — and the hoist above shows the
largest wins there may be *activation-layout* fixes, not sharding.
