# Usage: how to run a profile

All profiling is **opt-in and off by default.** With no flag/env var set,
training behaves exactly as before — same numerics, same speed.

## The controls

### CLI flags (on `train_experiments.py`)
| Flag | Effect |
|------|--------|
| `--profile` | Enable profiling (Rung 1 torch.profiler by default). |
| `--profile_dir DIR` | Where traces go (default `./profiler_logs`). Implies `--profile`. |
| `--profile_backend {torch,nsys}` | `torch` = torch.profiler (Rung 1). `nsys` = NVTX-only for Nsight Systems (Rung 2). Implies `--profile`. |

### Environment variables (read directly by the instrumentation)
| Var | Default | Effect |
|-----|---------|--------|
| `DENDRINET_PROFILE` | unset | Master gate (the flags set this to `1`). |
| `DENDRINET_PROFILE_DIR` | `./profiler_logs` | Trace output dir. |
| `DENDRINET_PROFILE_BACKEND` | `torch` | `torch` or `nsys`. |
| `DENDRINET_PROFILE_WAIT` | `1` | Schedule wait steps. |
| `DENDRINET_PROFILE_WARMUP` | `1` | Schedule warmup steps. |
| `DENDRINET_PROFILE_ACTIVE` | `3` | Profiled steps (step mode). |
| `DENDRINET_PROFILE_EPOCHS` | `0` | If `>0`, profile N full epochs instead of a step count. |

### Step mode vs epoch mode
- **Step mode** (default, `DENDRINET_PROFILE_EPOCHS=0`): capture
  `wait+warmup+active+1` steps, then **exit**. Best for *per-step* performance
  patterns. The batch job sets `ACTIVE=20` so you see ~20 clean repeating cycles.
- **Epoch mode** (`DENDRINET_PROFILE_EPOCHS=N`): NVTX labels on, **no** per-step
  early-exit; run N full epochs (with their validation passes), then exit before
  the heavy post-training analysis. Best for seeing **epoch boundaries**
  (train→validation). Bigger, slower trace.

> Why profile so few steps? Every training step is essentially identical, so a
> handful of steps shows the same pattern a full run would — just without a
> multi-GB trace. The first step is always a one-time-init outlier; that's what
> the `wait`/`warmup` are for, and why you "zoom past step 1" in the GUI.

## Rung 1 — torch.profiler

Direct run (inside a GPU allocation):
```bash
python src/dendritic_modeling/scripts/training/train_experiments.py \
    configs/examples/standard_mnist.yaml \
    --profile --profile_dir profilings/PyTorch-General
```
Outputs in `profilings/PyTorch-General/`: a TensorBoard trace, `chrome_trace.json`
(open at `chrome://tracing` or [ui.perfetto.dev](https://ui.perfetto.dev)), and a
**per-operator CPU/GPU table printed to the log** (the fastest signal — no tooling
needed).

## Rung 2 — Nsight Systems (the main tool)

Use the batch job — it survives preemption (auto-requeue) and names the report
after the config:

```bash
sbatch --export=ALL,NSYS_MODULE=cuda/12.9.1-fasrc01,\
CONFIG=configs/examples/standard_mnist.yaml jobs/profile_nsys.sbatch
```

Job parameters (via `--export=ALL,KEY=VAL,...`):
| Key | Default | Effect |
|-----|---------|--------|
| `CONFIG` | `standard_mnist.yaml` | Which YAML to profile. Report = `profilings/NSights/<config-name>.nsys-rep`. |
| `NSYS_MODULE` | unset | CUDA module providing `nsys` (use `cuda/12.9.1-fasrc01`). |
| `PROFILE_ACTIVE` | `20` | Step-mode step count. |
| `PROFILE_EPOCHS` | `0` | `>0` → epoch mode (N full epochs). |

Examples for the configs we profiled:
```bash
# step mode (per-step patterns)
sbatch --export=ALL,NSYS_MODULE=cuda/12.9.1-fasrc01,CONFIG=configs/examples/cifar10_two_layer_population_network.yaml jobs/profile_nsys.sbatch

# epoch mode (3 epochs, to see boundaries)
sbatch --export=ALL,NSYS_MODULE=cuda/12.9.1-fasrc01,PROFILE_EPOCHS=3,CONFIG=configs/examples/ryan_two_layer_population_network.yaml jobs/profile_nsys.sbatch

# recurrent config (uses the RecurrentTrainer)
sbatch --export=ALL,NSYS_MODULE=cuda/12.9.1-fasrc01,PROFILE_EPOCHS=3,CONFIG=configs/examples/population_network_recurrent_pyr_pv_sst.yaml jobs/profile_nsys.sbatch
```

Why `nsys` and not torch.profiler under nsys: both use CUPTI and conflict. The
`nsys` backend emits **NVTX ranges only** (no torch.profiler), so the timeline is
clean and labeled (`forward`/`backward`/`optimizer`/`data_h2d`).

### Reading an nsys report
Text summary on the cluster (write to a file — see the "stray `>`" gotcha in
[troubleshooting.md](troubleshooting.md)):
```bash
nsys stats --report nvtx_sum --report cuda_gpu_kern_sum --report cuda_api_sum \
  profilings/NSights/<name>.nsys-rep > rung2_stats.txt
```
- `nvtx_sum` — time in your `forward`/`backward`/`optimizer`/`data_h2d` ranges
  (read the **median**, not the average — the first step is an init outlier).
- `cuda_gpu_kern_sum` — GPU kernel ranking (top-k should top it).
- `cuda_api_sum` — watch `cudaLaunchKernel` (launch overhead) and
  `cudaStreamSynchronize` (the `.item()` sync stalls).

Visual timeline: `rsync` the `.nsys-rep` to your Mac, open in the **Nsight Systems
GUI**, zoom past step 1, stack the **NVTX row** over the **CUDA HW (GPU)** row, and
look at the idle gaps in the GPU row.

## Rung 3 — Nsight Compute (characterization only)

```bash
sbatch --export=ALL,NCU_MODULE=cuda/12.9.1-fasrc01 jobs/profile_ncu.sbatch
```
Profiles the top-k kernel (`sbtopk::gatherTopK`) on `standard_mnist`. Reuses the
`nsys` backend for early-exit and filters to a few launches (`ncu` is slow — it
replays each kernel many times).

> **The Rung-3 gate:** `ncu` needs GPU performance-counter access. If the log shows
> `ERR_NVGPUCTRPERM`, that's a cluster driver setting only RC admins can enable —
> email `rchelp@rc.fas.harvard.edu`. See [troubleshooting.md](troubleshooting.md).

Open the `.ncu-rep` in the **Nsight Compute** GUI (a *different app* from Nsight
Systems). For top-k, look at **Compute vs Memory throughput** (bound by which?)
and **achieved occupancy**.

## Getting reports onto your Mac

Preferred — `rsync` (no repo bloat), **run on your Mac**:
```bash
rsync -avz <user>@login.rc.fas.harvard.edu:DendriNet/profilings/ ~/DendriNet/profilings/
```
The trace artifacts are gitignored, so they sit locally for reference without
bloating git. Only commit reports to git deliberately (see
[troubleshooting.md](troubleshooting.md) → "Transferring reports").
