.. _installation_section:

Installation and Setup
=====================

This document provides comprehensive instructions for installing and setting up the dendritic modeling framework, including environment configuration, dependencies, and troubleshooting common issues.

Overview
--------

The dendritic modeling framework is a PyTorch-based library for implementing and experimenting with dendritic neural networks. This guide will help you:

1. Set up a Python environment for the framework
2. Install the framework and its dependencies
3. Configure development tools and testing
4. Verify the installation
5. Configure your environment for different hardware
6. Troubleshoot common installation issues

System Requirements
------------------

Before installing, ensure your system meets the following requirements:

* **Python**: 3.9 or higher (3.12 recommended)
* **CUDA** (for GPU support): Version 11.8 or higher recommended
* **Operating System**: Linux (recommended), macOS, or Windows
* **Disk Space**: At least 8GB for the framework, dependencies, and datasets
* **Memory**: At least 8GB RAM (16GB or more recommended)
* **GPU**: NVIDIA GPU with at least 4GB VRAM (recommended for training)

Quick Installation (Recommended)
--------------------------------

The fastest way to get started is using our conda environment file:

.. code-block:: bash

    # Create and activate conda environment with all dependencies
    conda env create -f environment.yml
    conda activate dendritic_env

    # Install the package in development mode
    pip install -e .

    # Verify installation
    python -c "import dendritic_modeling; print('Installation successful!')"

This creates a complete development environment with testing, code quality tools, and all optional dependencies.

Installation Options
-------------------

Choose the installation method that best fits your needs:

Full Development Setup (Recommended for Contributors)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Option A: Using conda environment file (recommended)
    conda env create -f environment.yml
    conda activate dendritic_env
    pip install -e ".[dev]"

    # Option B: Using pip only
    python -m venv dendritic_env
    source dendritic_env/bin/activate  # Windows: dendritic_env\Scripts\activate
    pip install -r requirements.txt
    pip install -e .

    # Set up pre-commit hooks for code quality
    pre-commit install

    # Verify development setup
    python tests/test_runner.py --mode full

Core Package Only (Minimal Installation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Create virtual environment
    python -m venv dendritic_env
    source dendritic_env/bin/activate  # Windows: dendritic_env\Scripts\activate

    # Install core dependencies only
    pip install -e .

Specialized Installations
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # For testing and CI development
    pip install -e ".[test]"

    # For training and analysis work
    pip install -e ".[train]"

    # For documentation building
    pip install -e ".[docs]"

    # For code quality tools only
    pip install -e ".[lint]"

    # Everything (development + training + docs)
    pip install -e ".[all]"

Dependency Groups Explained
--------------------------

The framework uses a modular dependency system:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Group
     - Purpose
     - Key Packages
   * - ``core``
     - Basic functionality
     - torch, numpy, matplotlib, tensorflow, omegaconf
   * - ``test``
     - Testing suite
     - pytest, pytest-cov, pytest-mock, coverage, networkx
   * - ``lint``
     - Code quality
     - ruff, black, isort, mypy, pre-commit
   * - ``train``
     - Training tools
     - jupyter, plotly, seaborn, tensorboard
   * - ``docs``
     - Documentation
     - sphinx, sphinx-rtd-theme, nbsphinx
   * - ``dev``
     - Development
     - Combines test + lint + build tools
   * - ``all``
     - Everything
     - All dependency groups combined

Detailed Installation Steps
-------------------------

Step 1: Set up the Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Using Conda (Recommended):**

.. code-block:: bash

    # Create and activate environment with full dependencies
    conda env create -f environment.yml
    conda activate dendritic_env

**Using Python venv:**

.. code-block:: bash

    # Create virtual environment
    python -m venv dendritic_env
    source dendritic_env/bin/activate  # Windows: dendritic_env\Scripts\activate

**Using Mamba (Faster alternative to conda):**

.. code-block:: bash

    # Install mamba first
    conda install mamba -n base -c conda-forge

    # Create environment
    mamba env create -f environment.yml
    mamba activate dendritic_env

Step 2: Install PyTorch
~~~~~~~~~~~~~~~~~~~~~

PyTorch is automatically installed via our environment files, but you can install manually:

.. code-block:: bash

    # For CUDA 12.1 (recommended)
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    # For CUDA 11.8
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

    # For CPU only
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

Verify PyTorch installation:

.. code-block:: bash

    python -c "
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU count: {torch.cuda.device_count()}')
    "

Step 3: Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    git clone https://github.com/KempnerInstitute/dendritic-modeling.git
    cd dendritic-modeling

Step 4: Install the package
~~~~~~~~~~~~~~~~~~~~~~~~

Choose your installation method:

.. code-block:: bash

    # Basic installation
    pip install -e .

    # Development installation (recommended for contributors)
    pip install -e ".[dev]"

    # Full installation with all optional dependencies
    pip install -e ".[all]"

Step 5: Set up development tools (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For development work, set up code quality tools:

.. code-block:: bash

    # Install pre-commit hooks
    pre-commit install

    # Test the setup
    pre-commit run --all-files

    # Run tests to verify everything works
    pytest tests/ -v

Verification and Testing
-----------------------

Basic Verification
~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Test basic functionality
    python -c "import dendritic_modeling; print('Installation successful!')"

    # Test all core dependencies
    python -c "
    import torch
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import dendritic_modeling
    print('All core dependencies working')
    "

Comprehensive Testing
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Run the full test suite
    pytest tests/ -v

    # Run tests with coverage
    pytest --cov=dendritic_modeling --cov-report=html
    open htmlcov/index.html  # View coverage report

    # Run specific test categories
    pytest -m unit              # Unit tests only
    pytest -m integration       # Integration tests only
    pytest -m "not slow"        # Skip slow tests

    # Use our custom test runner
    python tests/test_runner.py --mode full

Code Quality Verification
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Check code style
    ruff check src/ tests/
    black --check src/ tests/
    isort --check-only src/ tests/

    # Type checking
    mypy src/

    # Run all pre-commit hooks
    pre-commit run --all-files

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~

**Missing libbz2.so.1.0 (Linux)**

.. code-block:: bash

    # Create symlink to available version
    ln -sf /lib64/libbz2.so.1.0.6 $CONDA_PREFIX/lib/libbz2.so.1.0

**NetworkX Import Errors**

NetworkX is optional for plotting. Install if needed:

.. code-block:: bash

    pip install networkx

**CUDA Compatibility Issues**

.. code-block:: bash

    # Check CUDA version
    nvcc --version

    # Check PyTorch CUDA compatibility
    python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"

    # Reinstall PyTorch with matching CUDA version
    pip uninstall torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

**Pre-commit Hook Failures**

.. code-block:: bash

    # Update pre-commit hooks
    pre-commit autoupdate

    # Clear cache and reinstall
    pre-commit clean
    pre-commit install

Running Your First Experiment
----------------------------

After installation, test with a simple experiment:

.. code-block:: bash

    # Run a basic experiment
    python src/dendritic_modeling/scripts/train_experiments.py configs/config_exp.yaml

    # For interactive exploration
    jupyter notebook examples/quickstart.ipynb

Continuous Integration Setup
--------------------------

The project includes comprehensive CI/CD with GitHub Actions. Every pull request automatically runs:

- **Quick PR Check** (~2-3 minutes): Basic tests, linting, import validation  
- **Full CI Pipeline** (~10-15 minutes): Complete test suite across multiple Python versions

**Local CI Simulation**

.. code-block:: bash

    # Run the same checks as CI locally
    python tests/test_runner.py --mode full
    pre-commit run --all-files
    pytest --cov=dendritic_modeling --cov-fail-under=75

For more details, see the :doc:`../../development/contributing` guide.

Next Steps
----------

Now that you have installed the framework, you can:

- Follow the :doc:`quickstart` guide
- Explore :doc:`examples` 
- Learn about :doc:`../user_guide/concepts`
- Set up your development environment with :doc:`../../development/contributing`

For additional help:

- **Installation Issues**: Check the `INSTALL.md <https://github.com/KempnerInstitute/dendritic-modeling/blob/main/INSTALL.md>`_ file
- **Development Setup**: See the `CI_SETUP.md <https://github.com/KempnerInstitute/dendritic-modeling/blob/main/CI_SETUP.md>`_ guide
- **Bug Reports**: Open an issue on `GitHub <https://github.com/KempnerInstitute/dendritic-modeling/issues>`_ 