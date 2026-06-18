# Findings

Profiled config: `configs/examples/standard_mnist.yaml` (MNIST, hidden `[32]`,
`strategy: standard`, batch 256). Hardware: 1× A100 (sm_80; the `cutlass_80` /
`ampere_sgemm` kernel names confirm Ampere). PyTorch 2.11.0+cu128.

## TL;DR

DendriNet training is **CPU-launch-bound and synchronization-bound, not
compute-bound.** The GPU sits idle most of every step. The dominant GPU kernel is
the **top-k** selection in the sparse dendritic layers. There are **no custom
kernels** to optimize, so the levers are higher-level.

## Phase 1 — investigation (before any instrumentation)

- **Framework:** PyTorch only (TF/JAX are optional extras, not on the training
  path). Eager mode. Defaults are reproducible-but-slow (`deterministic=True`,
  `cudnn_benchmark=False`, `allow_tf32=False`).
- **Training loop:** `Trainer._epoch_train` in `training/strategies/standard.py`.
- **Data:** in-memory `TensorDataset`s — no per-item disk I/O, so data loading is
  not a bottleneck. (Exception: `PoissonGeneratorDataset` does per-sample CPU
  work, but the profiled configs don't use it.)
- **Custom kernels:** none. No CUDA/Triton; one `autograd.Function`
  (`SurrogateSpike`) but its backward is plain element-wise PyTorch. Optional
  `torch.compile` on one model. → **Nsight Compute is characterization-only.**
- **AMP:** opt-in (`use_amp`, default off). **Grad accumulation:** not
  implemented. **Distributed:** DDP/FSDP via `distributed.mode`.

## Rung 1 — torch.profiler (3 active steps)

| Metric | Value |
|--------|-------|
| Self **CPU** time (3 steps) | **78.2 ms** (~26 ms/step) |
| Self **CUDA** time (3 steps) | **10.7 ms** (~3.6 ms/step) |
| Top GPU op | `aten::topk` (`sbtopk::gatherTopK`) — **33%** |
| Adam optimizer | 26% |
| Matmuls (`aten::mm`) | 20% |
| `data_h2d` | **2%** (negligible) |

CPU does ~7× the GPU's work → the GPU idles waiting on CPU dispatch.

## Rung 2 — Nsight Systems (6 steps)

**CUDA API summary — the decisive evidence:**

| API call | Total | Calls | Per step |
|----------|-------|-------|----------|
| `cudaLaunchKernel` | **1.235 s** | **2,593** | **~430 launches/step** |
| `cudaStreamSynchronize` | 8.96 ms | 297 | **~50 syncs/step** |
| `cudaMemcpyAsync` | 10.4 ms | 313 | ~52/step |

Total GPU kernel time across the whole capture ≈ **33 ms**. So the CPU spent
**~1.2 s launching kernels** to feed **33 ms** of GPU compute → launch-bound.

**NVTX per-step (median; ignore first-step init outliers):** forward 3.75 ms,
backward 2.22 ms, optimizer 2.78 ms, data_h2d 0.21 ms. (`cuBLAS:cublasCreate_v2`
= 108 ms, one-time.) **Top GPU kernel:** `sbtopk::gatherTopK` = **38.6%**.

## Verdict

Two measured causes of GPU starvation:
1. **Kernel-launch overhead** — ~430 micro-kernels per step (the dendritic
   elementwise math: `vectorized_elementwise_kernel`, `multi_tensor_apply`,
   softplus, …). Each is µs of GPU work but a full CPU dispatch.
2. **Excessive synchronization** — ~50 `cudaStreamSynchronize`/step, driven by
   per-step `.item()` loss logging plus internal syncs.

Plus: the profiled model is **small for an A100**, so some underutilization is
inherent to this (deliberately tiny) config.

## Recommended levers (priority order)

1. **`torch.compile`** — fuses the hundreds of micro-kernels into far fewer
   launches. The codebase already exposes an optional `compile_forward` flag.
2. **Cut per-step syncs** — accumulate loss on-GPU; call `.item()` once per epoch.
3. **Larger batch / model** — to actually saturate the GPU on real workloads.
4. **CUDA graphs** — amortize launch overhead across the repeated step.

## Why Rung 3 was (correctly) skipped for optimization

No custom kernels exist; `topk`/`sgemm` are stock NVIDIA kernels you cannot
rewrite, and the GPU is idle — so kernel-level occupancy metrics would profile a
kernel that isn't the bottleneck. Nsight Compute remains useful only to
*characterize* top-k (memory- vs compute-bound), which is a learning exercise.

## Caveats

- Numbers are from the **smallest** config (chosen for fast iteration) — they
  characterize the *small-model regime*. Re-profile a production-scale config
  before drawing scaling conclusions.
- Read **medians**, not averages — the first step is a one-time CUDA/cuBLAS-init
  outlier that the warmup step is meant to absorb.

## Configs profiled in this effort

| Config | Strategy | Notes |
|--------|----------|-------|
| `standard_mnist` | standard | The original Rung 1/2/3 target. |
| `cifar10_two_layer_population_network` | standard | CIFAR; needs the dataset cached first. |
| `ryan_two_layer_population_network` | standard | CIFAR; same. |
| `population_network_recurrent_pyr_pv_sst` | **recurrent** | Sequence/temporal loss; the `RecurrentTrainer` path (separately instrumented). Synthetic data, no download. |
