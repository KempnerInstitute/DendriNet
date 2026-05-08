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

## Profiling Example

`configs/examples/standard_mnist.yaml` is a small DendriNet training run. Enable
`experiment.enable_profiling: true` for a PyTorch profiler trace, or run with
NVTX annotations for Nsight:

```bash
DENDRITIC_NVTX=1 PYTHONPATH=src:$PYTHONPATH nsys profile \
  -o outputs/profiles/standard_mnist_nvtx \
  python src/dendritic_modeling/scripts/training/train_experiments.py \
  configs/examples/standard_mnist.yaml
```

## Notes

- The supported workflow is `pip install -e .`.
- Set `PYTHONPATH` manually only if you are debugging a broken environment.
