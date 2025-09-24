# Dendritic Modeling Installation Guide

## Quick Installation

```bash
bash install.sh
```

The automated script handles environment creation, dependency installation, package setup, and verification.

## Installation Options

### Option 1: Full Automated Install
```bash
bash install.sh
```

### Option 2: Quick Install (Skip Optional Components)
```bash
bash install.sh --quick
```

### Option 3: Fix Existing Installation
```bash
bash install.sh --fix-only
```

### Option 4: Verify Current Installation
```bash
bash install.sh --verify
```

## Manual Installation

### Prerequisites
- Conda/Miniconda/Anaconda installed
- CUDA 12.1+ (for GPU support)

### Step-by-Step Manual Install

#### 1. Remove Existing Environment (if corrupted)
```bash
conda env remove -n dendritic_env -y
conda clean --all -y
```

#### 2. Create Environment
```bash
conda env create -f setup/environment.yml
# OR
conda create -n dendritic_env python=3.11 -y
conda activate dendritic_env
pip install -e .
```

#### 3. Activate and Install Package
```bash
conda activate dendritic_env
pip install -e .
```

#### 4. Fix Import Issues (if needed)
```bash
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"
echo 'export PYTHONPATH="$PYTHONPATH:'$(pwd)'/src"' >> ~/.bashrc
```

#### 5. Verify Installation
```bash
python -c "import torch; import dendritic_modeling; print('Success!')"
python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --help
```

## Installation Options

```bash
pip install -e .         # Core dependencies
pip install -e .[dev]    # Development tools
pip install -e .[train]  # Training/analysis
pip install -e .[all]    # All dependencies
```
