.. _installation_section:

Installation and Setup
=====================

This document provides comprehensive instructions for installing and setting up the dendritic modeling framework, including environment configuration, dependencies, and troubleshooting common issues.

Overview
--------

The dendritic modeling framework is a PyTorch-based library for implementing and experimenting with dendritic neural networks. This guide will help you:

1. Set up a Python environment for the framework
2. Install the framework and its dependencies
3. Verify the installation
4. Configure your environment for different hardware
5. Troubleshoot common installation issues

System Requirements
------------------

Before installing, ensure your system meets the following requirements:

* **Python**: 3.8 or higher
* **CUDA** (for GPU support): Version 11.0 or higher recommended
* **Operating System**: Linux (recommended), macOS, or Windows
* **Disk Space**: At least 5GB for the framework and datasets
* **Memory**: At least 8GB RAM (16GB or more recommended)
* **GPU**: NVIDIA GPU with at least 4GB VRAM (recommended for training)

Installation Methods
------------------

There are two main methods to install the framework:

Quick Installation (Recommended for Mamba/Conda Users)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a quick installation using Mamba:

.. code-block:: bash
    
    # Create a new environment
    mamba create -n dendritic python=3.10
    mamba activate dendritic
    
    # Clone the repository
    git clone https://github.com/KempnerInstitute/dendritic-modeling.git
    cd dendritic-modeling
    
    # Install the package
    pip install .

Development Installation
~~~~~~~~~~~~~~~~~~~~~~

For development or the latest features, install from source with development dependencies:

.. code-block:: bash

    # Create a new Mamba environment
    mamba create -n dendritic python=3.10
    mamba activate dendritic
    
    # Clone the repository
    git clone https://github.com/KempnerInstitute/dendritic-modeling.git
    cd dendritic-modeling
    
    # Install in development mode with dev dependencies
    pip install -e '.[dev]'
    
    # Install documentation dependencies (optional)
    pip install -e '.[docs]'

Detailed Installation Steps
-------------------------

Here's a more detailed guide to installing the framework from source:

Step 1: Set up the Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using Mamba:

.. code-block:: bash
    
    # Create and activate a new environment
    mamba create -n dendritic python=3.10
    mamba activate dendritic


Step 2: Install PyTorch
~~~~~~~~~~~~~~~~~~~~~

Install PyTorch with GPU support:

.. code-block:: bash

    # For CUDA 11.8
    mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    
    # For CPU only
    mamba install pytorch torchvision torchaudio cpuonly -c pytorch

Verify PyTorch installation:

.. code-block:: bash

    # Verify PyTorch and CUDA
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

Step 3: Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~~~

Get the source code:

.. code-block:: bash

    git clone https://github.com/KempnerInstitute/dendritic-modeling.git
    cd dendritic-modeling

Step 4: Install the package
~~~~~~~~~~~~~~~~~~~~~~~~

Install the package and its dependencies:

.. code-block:: bash

    # Install the basic package (for users)
    pip install .
    
    # OR install in development mode (for developers)
    pip install -e .
    
    # Optional: Install with development dependencies
    pip install -e '.[dev]'
    
    # Optional: Install with documentation dependencies
    pip install -e '.[docs]'

The project's dependencies are defined in the pyproject.toml file and include:

.. code-block:: text

    # Core dependencies
    torch
    numpy
    torchvision
    torchinfo
    matplotlib
    wandb
    pandas
    omegaconf>=2.3.0
    scikit-learn
    
    # Development dependencies (optional)
    ruff
    mypy
    black
    isort
    pytest
    pytest-cov
    etc.

Step 5: Verify installation
~~~~~~~~~~~~~~~~~~~~~~~

Ensure that the installation was successful:

.. code-block:: bash

    # Run a simple test
    python -c "from dendritic_modeling.networks import ExcitationInhibitionNetwork; print('Installation successful!')"
    
    # Run the test suite (if development dependencies were installed)
    pytest tests/

Configuring for Different Hardware
-------------------------------

CPU-only configuration
~~~~~~~~~~~~~~~~~~

For systems without a GPU:

.. code-block:: bash

    # Set this environment variable to disable CUDA
    export CUDA_VISIBLE_DEVICES=""
    
    # Install PyTorch for CPU only
    mamba install pytorch torchvision torchaudio cpuonly -c pytorch

Multi-GPU configuration
~~~~~~~~~~~~~~~~~~

For systems with multiple GPUs:

.. code-block:: bash

    # Set visible devices (e.g., use only GPUs 0 and 1)
    export CUDA_VISIBLE_DEVICES="0,1"
    
    # Check available GPUs
    python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
    
    # In your Python code
    model = model.to('cuda')  # Uses GPU 0
    # Or for multi-GPU training
    model = torch.nn.DataParallel(model)

Cluster configuration (SLURM)
~~~~~~~~~~~~~~~~~~~~~~~~~

For running on a SLURM-managed cluster, such as the Kempner Institute cluster:

.. code-block:: bash

    # Load necessary modules (adjust versions as needed)
    module load python/3.12.5-fasrc01
    module load cuda/12.4.1-fasrc01
    module load cudnn/8.9.2.26_cuda12-fasrc01
    
    # Clone the repository and enter directory
    git clone https://github.com/KempnerInstitute/dendritic-modeling.git
    cd dendritic-modeling
    
    # Install the package
    pip install -e .
    
    # Create a SLURM job submission script (example.sh)
    cat > example.sh << 'EOL'
    #!/bin/bash
    #SBATCH --job-name=dendritic_job
    #SBATCH --output=dendritic_%j.out
    #SBATCH --error=dendritic_%j.err
    #SBATCH --time=04:00:00
    #SBATCH --gres=gpu:1
    #SBATCH --mem=16G
    #SBATCH --cpus-per-task=4
    
    # Load modules
    module load python/3.12.5-fasrc01
    module load cuda/12.4.1-fasrc01
    module load cudnn/8.9.2.26_cuda12-fasrc01
    
    # Activate environment if needed
    # source /path/to/your/env/bin/activate
    
    # Run the experiment with specified config
    python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml
    EOL
    
    # Submit the job
    sbatch example.sh

Virtual Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer using venv instead of Conda/Mamba:

.. code-block:: bash

    # Create a virtual environment
    python -m venv dendritic-env
    
    # Activate the environment
    # On Linux/Mac:
    source dendritic-env/bin/activate
    # On Windows:
    # dendritic-env\Scripts\activate
    
    # Install PyTorch with appropriate CUDA version
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Clone the repository
    git clone https://github.com/KempnerInstitute/dendritic-modeling.git
    cd dendritic-modeling
    
    # Install the package
    pip install -e .

Troubleshooting
--------------

CUDA Compatibility Issues
~~~~~~~~~~~~~~~~~~~~~

If you encounter CUDA compatibility issues:

.. code-block:: bash

    # Check CUDA version
    nvcc --version
    
    # Check PyTorch CUDA compatibility
    python -c "import torch; print(torch.version.cuda)"
    
    # If versions mismatch, reinstall PyTorch with the matching CUDA version
    pip uninstall torch torchvision torchaudio
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu[version]
    # Replace [version] with your CUDA version (e.g., cu118 for CUDA 11.8)

ImportError Issues
~~~~~~~~~~~~~~

If you encounter import errors:

.. code-block:: bash

    # Ensure the package is installed
    pip show dendritic_modeling
    
    # If not installed or in development mode:
    pip install -e .
    
    # Check your PYTHONPATH
    echo $PYTHONPATH
    
    # Add the repository to your PYTHONPATH if needed
    export PYTHONPATH=$PYTHONPATH:/path/to/dendritic-modeling

Memory Issues
~~~~~~~~~

If you encounter out-of-memory errors during training:

.. code-block:: bash

    # Reduce batch size in your configuration file or training parameters
    # config.yaml:
    train:
      batch_size: 32  # Try smaller batch size
    
    # Use gradient accumulation
    # In your code or config:
    gradient_accumulation_steps: 2  # Accumulate gradients before updating
    
    # Clean up GPU memory
    python -c "import torch; torch.cuda.empty_cache()"

Starting Your First Experiment
------------------------------

To get started with your first experiment:

.. code-block:: bash

    # Create a configuration file (or use an example)
    cp configs/config_exp.yaml configs/my_experiment.yaml
    
    # Edit the configuration
    nano configs/my_experiment.yaml
    
    # Run the experiment
    python scripts/train_experiments.py --config configs/my_experiment.yaml
    
    # For interactive exploration, use a notebook
    jupyter notebook examples/quickstart.ipynb

Next Steps
----------

Now that you have installed the framework, you can:

- Explore the [Quickstart Guide](./quickstart.rst)
- Run through the [Examples](./examples.rst)
- Learn about the main [Concepts](./concepts.rst)
- Configure your [Experiments](./configuration.rst) 