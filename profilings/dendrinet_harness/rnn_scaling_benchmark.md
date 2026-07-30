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

## Open rows — fill as gates / optimizations run
| scale | GPU | path | epoch (s) | peak (MB) | speed vs loop | mem vs loop | notes |
|---|---|---|---|---|---|---|---|
| 5k/5k | A100 | loop / fused | — | — | — | — | cross-arch baseline |
| 100k/5k | multi-GPU | fused | — | — | — | — | memory gate (nproc=2,4) |
| 1M/5k | multi-GPU | fused | — | — | — | — | fits? peak/GPU |
| 5k/5k | Blackwell | fused + 8-bit optimizer | — | — | — | — | optimizer-state mem |
| 1M/5k | multi-GPU | fused + sharded idx | — | — | — | — | replicated-buffer fix |

## Kernel chapter: CLOSED
Op-level speed/mem for the RNN is done — fused autograd (speed+mem) + int32 idx (mem).
CUDA/Triton kernels do NOT apply (slower at batch-1). torch.compile blocked. Everything
below the baseline is now **distributed/memory-systems** work, not kernels.
