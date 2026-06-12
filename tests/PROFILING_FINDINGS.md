# DendriNet Profiling — Findings (Rung 1 + Rung 2)

**Date:** 2026-06-11  ·  **Config:** `configs/examples/standard_mnist.yaml`
(MNIST, hidden `[32]`, `strategy: standard`, batch 256)  ·  **Hardware:** 1× A100
(sm_80; FASRC/Kempner `holygpu8a`)  ·  **PyTorch:** 2.11.0+cu128
**Method:** opt-in instrumentation, a few steps after warmup, then early-exit
(`--profile` / `--profile_backend nsys`). No change to numerics.

---

## TL;DR

The dendritic model is **CPU-launch-bound and synchronization-bound, not
compute-bound.** On an A100, the GPU sits idle most of every step: the CPU
issues **~430 kernel launches and ~50 GPU syncs per step** to feed only **~5 ms**
of actual GPU compute. The single most expensive GPU kernel is the **top-k**
selection in the sparse dendritic layers (`TopKLinear`), ~39% of GPU time — but
total GPU work is tiny. There are **no custom CUDA/Triton kernels**, so
kernel-level profiling (Nsight Compute, Rung 3) is not warranted.

---

## Rung 1 — `torch.profiler` (3 active steps)

| Metric | Value |
|---|---|
| Self **CPU** time (3 steps) | **78.2 ms** (~26 ms/step) |
| Self **CUDA** time (3 steps) | **10.7 ms** (~3.6 ms/step) |
| Top GPU op | `aten::topk` (`sbtopk::gatherTopK`) — **33%** |
| Optimizer (Adam) | 26% |
| Matmuls (`aten::mm`, `ampere_sgemm`) | 20% |
| `data_h2d` (host→device copy) | **2%** (negligible) |

**Read:** CPU does ~7× the work of the GPU → GPU idles waiting on CPU dispatch.
Data loading is a non-issue (in-memory `TensorDataset`).

## Rung 2 — Nsight Systems (`nsys`, 6 steps)

**CUDA API summary — the decisive evidence:**

| API call | Total | Calls | Per step |
|---|---|---|---|
| `cudaLaunchKernel` | **1.235 s** | **2,593** | **~430 launches/step** |
| `cudaStreamSynchronize` | 8.96 ms | 297 | **~50 syncs/step** |
| `cudaMemcpyAsync` | 10.4 ms | 313 | ~52/step |

Total **GPU kernel time across the whole capture ≈ 33 ms.** So the CPU spent
**~1.2 s launching kernels** to feed **33 ms** of GPU compute → launch-bound.

**NVTX per-step (steady-state median; ignore first-step init outliers):**

| Range | Median/step | Note |
|---|---|---|
| `forward` | 3.75 ms | first step 142 ms = one-time CUDA init |
| `backward` | 2.22 ms | first step 197 ms |
| `optimizer` | 2.78 ms | first step 288 ms |
| `data_h2d` | 0.21 ms | negligible |
| `cuBLAS:cublasCreate_v2` | — | 108 ms, **one-time** handle creation |

**Top GPU kernel:** `sbtopk::gatherTopK` = **38.6%** (confirms Rung 1) — the
sparse dendritic routing dominates GPU compute.

---

## Verdict

Two concrete, measured causes of GPU starvation:

1. **Kernel-launch overhead** — the dendritic computation decomposes into
   hundreds of micro-kernels per step (`vectorized_elementwise_kernel`,
   `multi_tensor_apply`, softplus, etc.), ~430 launches/step. Each is µs of GPU
   work but a full CPU dispatch.
2. **Excessive synchronization** — ~50 `cudaStreamSynchronize`/step, driven by
   per-step `.item()` loss logging (3×/step) plus internal syncs. Each is a
   CPU↔GPU stall.

Plus: the profiled model is **small for an A100**, so some underutilization is
inherent to this (deliberately tiny) config.

## Recommended levers (priority order)

1. **`torch.compile`** — fuses the hundreds of micro-kernels into far fewer
   launches; biggest win for the lowest effort. *The codebase already exposes an
   optional `compile_forward` flag* (`ei_network.py`).
2. **Cut per-step syncs** — accumulate loss on-GPU; call `.item()` once per epoch
   instead of every step.
3. **Larger batch / model** — to actually saturate the GPU on real workloads
   (validate whether production-scale runs are GPU-bound, unlike this tiny one).
4. **CUDA graphs** — amortize launch overhead across the repeated step.

## Why Rung 3 (Nsight Compute) was skipped

Verified in Phase 1 and confirmed by both rungs: **no custom CUDA/Triton kernels,
no performance-critical custom autograd backward.** The bottleneck is the
*timeline* (launches + syncs), not a slow kernel. `topk`/`sgemm` are stock NVIDIA
kernels that cannot be rewritten, and the GPU is idle — so kernel-level occupancy
metrics would profile a kernel that is not the bottleneck.

## Caveats

- Numbers are from the **smallest** config, chosen for fast iteration; they
  characterize the *small-model regime*. Re-profile a representative
  production-scale config before drawing scaling conclusions.
- First-step costs (one-time CUDA/cuBLAS init) are excluded by reading medians,
  not averages — the warmup step intentionally absorbs them.

## Reproduce

```bash
# Rung 1 (torch.profiler):
python src/dendritic_modeling/scripts/training/train_experiments.py \
    configs/examples/standard_mnist.yaml --profile --profile_dir profilings/PyTorch-General

# Rung 2 (Nsight Systems):
nsys profile --trace=cuda,nvtx,osrt,cudnn,cublas \
    --output=profilings/NSights/mnist_rung2 --force-overwrite=true \
    python src/dendritic_modeling/scripts/training/train_experiments.py \
        configs/examples/standard_mnist.yaml --profile --profile_backend nsys
nsys stats --report nvtx_sum --report cuda_gpu_kern_sum --report cuda_api_sum \
    profilings/NSights/mnist_rung2.nsys-rep
```
