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

## Batch scaling — feedforward stress config, Blackwell, cuda path

| batch | peak (MB) | vs batch 512 |
|---|---|---|
| 512 | 13,370.1 | 1.00× |
| **2048** | **48,810.2** | 3.65× |

Sub-linear in batch because ~450 MB (the transformed weights) is batch-independent.
Predicted ~52 GB from the batch-512 composition; measured 48.8 GB.

**Batch 2048 was not reachable this morning.** At the pre-fusion 28,184 MB it would have
needed ~103 GB, which is no single GPU here; it now fits a 96 GB Blackwell with 47 GB spare.
The transient gap grows with it — 9,880 MB, still ~20% of peak — so that remains the
un-decomposed term at any batch.

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
