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

## Open rows
| scale | GPU | path | epoch (s) | peak (MB) | notes |
|---|---|---|---|---|---|
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
