# Installation

Requires Python 3.10+. The default conda environment uses Python 3.11 and
CUDA 12.1 for GPU installs; adjust or remove `pytorch-cuda` in
`setup/environment.yml` for CPU-only or site-specific CUDA builds.

## Conda Setup

Create the default environment and install the package in editable mode:

```bash
conda env create -f setup/environment.yml
conda activate dendritic_env
pip install -e .
```

Optional extras:

```bash
pip install -e ".[dev]"
pip install -e ".[train]"
pip install -e ".[all]"
```

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
