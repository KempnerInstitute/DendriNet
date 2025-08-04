# Installation Guide

## Quick Install

```bash
# Clone the repository
git clone https://github.com/KempnerInstitute/dendritic_modeling.git
cd dendritic_modeling

# Create conda environment
conda env create -f environment.yml
conda activate dendritic_env

# Install the package
pip install -e .
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

## Manual Installation

If you prefer to set up the environment manually:

```bash
# Create environment
conda create -n dendritic_env python=3.9
conda activate dendritic_env

# Install PyTorch (choose appropriate version for your system)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Verify Installation

```bash
# Run tests to verify everything works
pytest tests/ -v

# Quick functionality test
python -c "from dendritic_modeling.config import load_config; print('Installation successful!')"
```

## GPU Setup

The framework automatically detects and uses GPU when available. To force CPU usage:

```bash
CUDA_VISIBLE_DEVICES="" python your_script.py
```

## Development Setup

For development with code quality tools:

```bash
pip install -e ".[dev]"
pre-commit install
```

## Troubleshooting

**Common Issues:**

1. **CUDA version mismatch**: Ensure PyTorch CUDA version matches your system CUDA
2. **Memory errors**: Reduce batch size in your config files
3. **Import errors**: Verify all dependencies are installed: `pip install -r requirements.txt`

**Get Help:**
- Check [GitHub Issues](https://github.com/KempnerInstitute/dendritic_modeling/issues)
- Verify your environment: `conda list` and `pip list` 