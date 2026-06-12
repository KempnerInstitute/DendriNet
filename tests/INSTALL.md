# Installation

## Automated Setup

```bash
bash install.sh
```

Useful variants:

```bash
bash install.sh --quick
bash install.sh --fix-only
bash install.sh --verify
```

## Manual Setup

Create the default environment and install the package in editable mode:

```bash
conda env create -f setup/environment.yml
conda activate dendritic_env
pip install -e .
```

Optional extras:

```bash
pip install -e .[dev]
pip install -e .[train]
pip install -e .[all]
```

## Cluster Setup (CUDA 12.9)

On a CUDA 12.9 cluster (e.g. the Kempner cluster), the conda `environment.yml`
pins an incompatible `pytorch-cuda=12.1`. Use the helper script instead, which
installs a CUDA-12.8 PyTorch wheel (bundled CUDA runtime; runs natively on a
CUDA 12.9 driver) and then `pip install -e .`:

```bash
bash setup/setup_cuda129.sh
conda activate dendritic_env
```

Configurable via env vars — e.g. install into scratch space, or add extras:

```bash
ENV_PREFIX=/scratch/$USER/envs/dendritic_env bash setup/setup_cuda129.sh
EXTRAS="[dev]" bash setup/setup_cuda129.sh
```

See the comments at the top of `setup/setup_cuda129.sh` for all options and the
cu128-vs-cu129 compatibility rationale.

## Optional Dependency

`ffmpeg` is only needed for ablation video export.

```bash
conda install -c conda-forge ffmpeg
```

## Verify

```bash
python -c "import dendritic_modeling; print('ok')"
python src/dendritic_modeling/scripts/training/train_experiments.py --help
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --help
```

## Notes

- The supported workflow is `pip install -e .`.
- Set `PYTHONPATH` manually only if you are debugging a broken environment.
