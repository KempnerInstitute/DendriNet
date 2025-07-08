# Installation Guide

This guide provides detailed installation instructions for the Dendritic Modeling Framework.

## Quick Start

```bash
# Create and activate conda environment with all dependencies
conda env create -f environment.yml
conda activate dendritic_env

# Install the package in development mode
pip install -e .
```

## Installation Options

### 1. Full Development Setup (Recommended for Contributors)

```bash
# Option A: Using conda environment file
conda env create -f environment.yml
conda activate dendritic_env
pip install -e ".[dev]"

# Option B: Using pip only
python -m venv dendritic_env
source dendritic_env/bin/activate  # Windows: dendritic_env\Scripts\activate
pip install -r requirements.txt
pip install -e .

# Set up pre-commit hooks
pre-commit install
```

### 2. Core Package Only (Minimal Installation)

```bash
# Create virtual environment
python -m venv dendritic_env
source dendritic_env/bin/activate  # Windows: dendritic_env\Scripts\activate

# Install core dependencies only
pip install -e .
```

### 3. Specific Use Cases

#### For Testing and CI
```bash
pip install -e ".[test]"
```

#### For Training and Analysis
```bash
pip install -e ".[train]"
```

#### For Documentation Building
```bash
pip install -e ".[docs]"
```

#### For Code Quality Tools
```bash
pip install -e ".[lint]"
```

#### Everything (Development + Training + Docs)
```bash
pip install -e ".[all]"
```

## Dependency Groups

| Group | Purpose | Key Packages |
|-------|---------|--------------|
| `core` | Basic functionality | torch, numpy, matplotlib, tensorflow |
| `test` | Testing suite | pytest, pytest-cov, pytest-mock, coverage |
| `lint` | Code quality | ruff, black, isort, mypy, pre-commit |
| `train` | Training tools | jupyter, plotly, seaborn, tensorboard |
| `docs` | Documentation | sphinx, sphinx-rtd-theme, nbsphinx |
| `dev` | Development | Combines test + lint + build tools |
| `all` | Everything | All dependency groups combined |

## Environment Files

### Conda Environment (`environment.yml`)
- Installs packages from conda-forge, PyTorch, and NVIDIA channels
- Includes CUDA support (pytorch-cuda=12.1)
- Falls back to pip for packages not available in conda

### Requirements File (`requirements.txt`)
- Pure pip installation with version specifications
- Includes all dependencies for full development setup
- Can be used with any Python environment manager

### PyProject.toml Configuration
- Modern Python project configuration
- Defines optional dependency groups
- Includes tool configurations for pytest, coverage, etc.

## Verification

After installation, verify everything works:

```bash
# Basic functionality test
python -c "import dendritic_modeling; print('Package imported successfully')"

# Full dependency test
python -c "
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import dendritic_modeling
print('All core dependencies working')
"

# Run test suite (if test dependencies installed)
pytest tests/ -v

# Check code quality tools (if lint dependencies installed)
ruff check src/
black --check src/
mypy src/
```

## GPU Support

### CUDA Setup
The conda environment includes CUDA support by default:
```bash
conda env create -f environment.yml
# pytorch-cuda=12.1 is included
```

### Verifying GPU Support
```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'Current GPU: {torch.cuda.get_device_name()}')
"
```

## Troubleshooting

### Common Issues

#### Missing libbz2.so.1.0 (Linux)
```bash
# Create symlink to available version
ln -sf /lib64/libbz2.so.1.0.6 $CONDA_PREFIX/lib/libbz2.so.1.0
```

#### NetworkX Import Errors
NetworkX is optional for plotting. The package will work without it:
```bash
# Install networkx if you need network plotting
pip install networkx
```

#### TensorFlow GPU Issues
```bash
# Verify TensorFlow can see GPU
python -c "
import tensorflow as tf
print('TF version:', tf.__version__)
print('GPU available:', tf.config.list_physical_devices('GPU'))
"
```

#### Pre-commit Hook Failures
```bash
# Update pre-commit hooks
pre-commit autoupdate

# Clear cache and reinstall
pre-commit clean
pre-commit install
```

### Kempner Cluster Specific Setup

```bash
# Load required modules
module load python/3.12.5-fasrc01
module load cuda/12.4.1-fasrc01
module load cudnn/8.9.2.26_cuda12-fasrc01

# Then follow standard installation
conda env create -f environment.yml
conda activate dendritic_env
pip install -e .
```

## Development Workflow

### Setting Up for Development

1. **Fork and clone the repository**
2. **Install with development dependencies**:
   ```bash
   conda env create -f environment.yml
   conda activate dendritic_env
   pip install -e ".[dev]"
   ```
3. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```
4. **Verify setup**:
   ```bash
   python tests/test_runner.py --mode full
   ```

### Before Committing

```bash
# Run code quality checks
pre-commit run --all-files

# Run tests
pytest

# Check coverage
pytest --cov=dendritic_modeling --cov-report=term-missing
```

## Alternative Package Managers

### Using Poetry
```bash
# If you prefer Poetry (experimental)
poetry install --all-extras
poetry shell
```

### Using Pipenv
```bash
# If you prefer Pipenv (experimental)
pipenv install -e .
pipenv shell
```

### Using Mamba (Faster than Conda)
```bash
# Install mamba first
conda install mamba -n base -c conda-forge

# Use mamba instead of conda
mamba env create -f environment.yml
mamba activate dendritic_env
```

## Getting Help

- **Installation Issues**: Check this guide and [CI_SETUP.md](CI_SETUP.md)
- **Testing Problems**: See [tests/README.md](tests/README.md)
- **Development Questions**: Review the main [README.md](README.md)
- **Bug Reports**: Open an issue on GitHub with your installation details 