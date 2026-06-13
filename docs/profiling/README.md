# DendriNet Performance Profiling

This directory documents the performance-profiling effort on DendriNet: how the
profiling instrumentation works, how to run it (locally and on the FASRC/Kempner
cluster), what we found, and every gotcha we hit so you don't re-hit them.

> **Goal of profiling here:** understand *where training time goes* — and in
> particular, **is the GPU actually busy, or idling while the CPU/Python feeds
> it?** That single question drives almost everything below.

## Read these in order

| Doc | What it covers |
|-----|----------------|
| [setup.md](setup.md) | Building the conda environment (incl. the CUDA 12.9 cluster), and the FASRC/Kempner Slurm basics. |
| [usage.md](usage.md) | How to actually run a profile — the opt-in flags/env vars, the three "rungs," step vs epoch mode, the batch jobs, and how to read the output. |
| [instrumentation.md](instrumentation.md) | What was added to the codebase, the design (opt-in / off-by-default), and how to instrument another trainer. |
| [findings.md](findings.md) | The results: the GPU is launch/sync-bound, top-k is the dominant kernel, and what to do about it. |
| [troubleshooting.md](troubleshooting.md) | Every problem we hit (preemption, dataset caching, 0-byte traces, permissions, git artifacts…) and the fix. |

## The profiling ladder (the mental model)

Three tools, increasing in detail and cost. **Climb only as far as the workload
justifies.**

| Rung | Tool | Answers | Worth it for DendriNet? |
|------|------|---------|--------------------------|
| 1 | `torch.profiler` | Per-op CPU vs GPU time; is the GPU idling? | ✅ start here |
| 2 | **Nsight Systems** (`nsys`) | A full timeline — overlap, gaps, sync stalls | ✅ the main tool |
| 3 | **Nsight Compute** (`ncu`) | The internals of *one* GPU kernel | ⚠️ characterization only — **no custom kernels to optimize** |

The Rung-3 caveat is important and was decided up front: DendriNet has **no
custom CUDA/Triton kernels** — every kernel is stock PyTorch/cuBLAS/cuDNN code
you cannot rewrite. So Nsight Compute is a *learning* tool here, not an
optimization one. See [findings.md](findings.md).

## Headline finding (one paragraph)

On an A100, DendriNet training is **CPU-launch-bound and synchronization-bound,
not compute-bound.** Per step the CPU issues **~430 kernel launches** and **~50
GPU syncs** to feed only **~5 ms of actual GPU compute** — so the GPU sits idle
most of every step. The single most expensive GPU kernel is the **top-k
selection** in the sparse dendritic layers (`TopKLinear`), ~39% of GPU time. The
actionable levers are higher-level: `torch.compile` (fuse the hundreds of tiny
kernels), fewer per-step `.item()` syncs, and larger batch/model to actually
feed the GPU. Full numbers in [findings.md](findings.md).

## 60-second quick start (on the cluster)

```bash
# one-time: build the env (CUDA 12.9 cluster)
bash setup/setup_cuda129.sh
conda activate dendritic_env

# profile a config with Nsight Systems (auto-requeues through preemption)
sbatch --export=ALL,NSYS_MODULE=cuda/12.9.1-fasrc01,\
CONFIG=configs/examples/standard_mnist.yaml jobs/profile_nsys.sbatch

# when [job] DONE appears with a non-zero .nsys-rep, copy it down (on your Mac):
rsync -avz <user>@login.rc.fas.harvard.edu:DendriNet/profilings/NSights/ ~/DendriNet/profilings/NSights/
# then open the .nsys-rep in the Nsight Systems GUI
```

## Related files in the repo

- `setup/setup_cuda129.sh` — the env builder for a CUDA 12.9 cluster.
- `jobs/profile_nsys.sbatch` — parameterized Nsight Systems capture (Rung 2).
- `jobs/profile_ncu.sbatch` — Nsight Compute capture of the top-k kernel (Rung 3).
- `src/dendritic_modeling/training/utils/profiling.py` — the instrumentation.
- Root-level `PROFILING_FINDINGS.md`, `PROFILING_REPORT.md`, `PROFILING_HANDOFF.md`
  — earlier standalone write-ups; [findings.md](findings.md) supersedes the first,
  and `PROFILING_REPORT.md` is a plain-language explainer good for pasting into a chat.
