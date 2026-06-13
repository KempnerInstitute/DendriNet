# Setup: environment + cluster

## The CUDA 12.9 problem (and why we use cu128 wheels)

The cluster (FASRC/Kempner) has a **CUDA 12.9** driver. The repo's
`setup/environment.yml` pins `pytorch-cuda=12.1` from the (deprecated) `pytorch`
conda channel — that cannot produce a 12.8/12.9-compatible build, and there is
**no `cu129` pip wheel** for stable PyTorch.

**Key fact:** PyTorch wheels *bundle their own CUDA runtime*; they do **not** use
the system CUDA toolkit. The only thing that must be compatible is the NVIDIA
**driver**, and within CUDA major version 12 a newer driver runs any older 12.x
runtime. So a **`cu128` wheel runs natively on a CUDA 12.9 driver** — `cu128`
*is* the "CUDA 12.9-compatible" build. (Verified: the env installs
`torch 2.11.0+cu128`, bundled CUDA 12.8, and `torch.cuda.is_available()` is
`True` on the GPU nodes.)

## Build the environment

Use the helper script (do **not** use `conda env create -f setup/environment.yml`):

```bash
bash setup/setup_cuda129.sh
conda activate dendritic_env
```

It creates a fresh Python 3.11 env, installs `torch`/`torchvision` from the
`cu128` wheel index, then `pip install -e .` for the package. Configurable via
env vars (see the header of `setup/setup_cuda129.sh`):

| Var | Purpose |
|-----|---------|
| `ENV_NAME` | env name (default `dendritic_env`) |
| `ENV_PREFIX` | create at a path instead (for HPC home-quota limits) |
| `CUDA_WHL` | wheel tag (default `cu128`; set `cu129` only if your torch ships it) |
| `EXTRAS` | pip extras, e.g. `[dev]`, `[all]` |
| `FORCE` | `1` to remove an existing env first |

The script's last lines print the exact `conda activate` line to use.

## FASRC / Kempner specifics

This cluster runs **Slurm** on **Linux**. A few things that bit us repeatedly:

### `conda` and `nsys`/`ncu` come from modules
A fresh shell has neither on `PATH`. Load them:
```bash
module load python                 # provides conda + mamba on FASRC
module load cuda/12.9.1-fasrc01    # provides nsys and ncu (Nsight CLIs)
```
Confirm: `conda --version`, `nsys --version`, `ncu --version`.

### Login nodes vs compute nodes
- **Login node** (`holylogin*`): no GPU. `torch.cuda.is_available()` is **False**
  here — that's expected, not a broken install. Good for git, editing, and
  pre-downloading datasets.
- **Compute/GPU node** (`holygpu*`): where the GPU lives. Only here will
  `torch.cuda.is_available()` be `True`. Get one interactively with `salloc`, or
  run a batch job with `sbatch`.

### Interactive GPU session
```bash
salloc -p kempner_requeue -A kempner_undergrads -c 4 --gres=gpu:1 --mem=32G -t 01:00:00
# inside the allocation (it's a fresh shell — reload modules):
module load python && conda activate dendritic_env
module load cuda/12.9.1-fasrc01
python -c "import torch; print(torch.version.cuda, torch.cuda.is_available())"   # expect 12.8 True
```

> **Preemption warning:** `kempner_requeue` is **preemptible** — interactive jobs
> get killed within minutes when higher-priority work arrives. For anything that
> isn't a quick check, **use the batch jobs** (`jobs/*.sbatch`), which auto-requeue
> through preemption. See [usage.md](usage.md) and [troubleshooting.md](troubleshooting.md).

### Finding your account / partitions
```bash
sacctmgr -n show assoc user=$USER format=account%40 | sort -u    # accounts (-A)
sinfo -o "%P %a %l %G" | grep -i kempner                          # partitions (*_requeue = preemptible)
```

## A note for non-cluster / macOS

Apple Silicon Macs have **no NVIDIA CUDA at all** — `pytorch-cuda` and the
`cu*` wheels cannot install there. GPU profiling (Rungs 2–3) must run on the
cluster. A local Mac env (CPU/MPS build, plain `pip install torch torchvision`)
can import the package and run CPU-only `torch.profiler` (Rung 1), but not the
GPU-timeline tools.
