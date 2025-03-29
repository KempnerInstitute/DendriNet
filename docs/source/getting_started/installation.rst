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

Quick Installation (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a quick installation using pip:

.. code-block:: bash

    # Create a new conda environment
    conda create -n dendritic python=3.8
    conda activate dendritic
    
    # Install PyTorch with CUDA (modify cudatoolkit version if needed)
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    
    # Install the dendritic modeling framework
    pip install dendritic-modeling

Development Installation
~~~~~~~~~~~~~~~~~~~~~~

For development or the latest features, install from source:

.. code-block:: bash

    # Create a new conda environment
    conda create -n dendritic python=3.8
    conda activate dendritic
    
    # Install PyTorch with CUDA
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    
    # Clone the repository
    git clone https://github.com/KempnerInstitute/dendritic-modeling.git
    cd dendritic-modeling
    
    # Install in development mode
    pip install -e .

Detailed Installation Steps
-------------------------

Here's a more detailed guide to installing the framework from source:

Step 1: Set up the Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using conda (recommended):

.. code-block:: bash

    # Install Miniconda if you don't have it
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh
    
    # Create and activate a new environment
    conda create -n dendritic python=3.8
    conda activate dendritic

Using virtualenv (alternative):

.. code-block:: bash

    # Install virtualenv if you don't have it
    pip install virtualenv
    
    # Create and activate a new environment
    virtualenv dendritic-env
    source dendritic-env/bin/activate  # On Windows: dendritic-env\Scripts\activate

Step 2: Install PyTorch
~~~~~~~~~~~~~~~~~~~~~

Install PyTorch with GPU support:

.. code-block:: bash

    # For CUDA 11.7
    pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
    
    # For CPU only
    pip install torch==2.0.0 torchvision==0.15.1

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

Step 4: Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~

Install the package and its dependencies:

.. code-block:: bash

    # Install required packages
    pip install -r requirements.txt
    
    # Install the package in development mode
    pip install -e .
    
    # Install additional development dependencies (optional)
    pip install -r requirements-dev.txt

The ``requirements.txt`` file includes all the necessary dependencies:

.. code-block:: text

    numpy>=1.20.0
    pandas>=1.3.0
    matplotlib>=3.4.0
    seaborn>=0.11.0
    scikit-learn>=0.24.0
    tqdm>=4.60.0
    wandb>=0.12.0
    omegaconf>=2.1.0
    pytest>=6.2.0
    jupyter>=1.0.0
    networkx>=2.6.0

Step 5: Verify installation
~~~~~~~~~~~~~~~~~~~~~~~

Ensure that the installation was successful:

.. code-block:: bash

    # Run a simple test
    python -c "from dendritic_modeling.networks import ExcitationInhibitionNetwork; print('Installation successful!')"
    
    # Run the test suite
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
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

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

For HPC clusters with SLURM:

.. code-block:: bash

    #!/bin/bash
    #SBATCH --job-name=dendritic
    #SBATCH --gres=gpu:1
    #SBATCH --mem=32G
    #SBATCH --time=24:00:00
    
    module load python/3.8 cuda/11.7
    source activate dendritic
    
    python train_experiments.py --config configs/config_exp.yaml

Directory Structure
----------------

After installation, the framework has the following directory structure:

.. code-block:: text

    dendritic-modeling/
    ├── src/                     # Source code
    │   └── dendritic_modeling/  # Main package
    │       ├── activation_functions.py
    │       ├── config.py
    │       ├── dendrinet.py
    │       └── ...              # Other modules
    ├── configs/                 # Configuration files
    ├── scripts/                 # Utility scripts
    ├── tests/                   # Test suite
    ├── docs/                    # Documentation
    └── notebooks/              # Example notebooks

Common Issues and Solutions
------------------------

CUDA version mismatch
~~~~~~~~~~~~~~~~~~

If you encounter CUDA version errors:

.. code-block:: text

    RuntimeError: CUDA error: no kernel image is available for execution on the device

Solution:

.. code-block:: bash

    # Check CUDA version
    nvidia-smi
    
    # Install PyTorch for your specific CUDA version
    pip install torch==2.0.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html

Out of memory errors
~~~~~~~~~~~~~~~~~~

If you encounter GPU memory issues:

.. code-block:: text

    RuntimeError: CUDA out of memory

Solution:

.. code-block:: python

    # Reduce batch size
    batch_size = 32  # Try smaller values
    
    # Use gradient accumulation
    optimizer.zero_grad()
    for i, (inputs, targets) in enumerate(data_loader):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss = loss / accumulation_steps
        loss.backward()
        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

Module not found errors
~~~~~~~~~~~~~~~~~~

If the package cannot be found:

.. code-block:: text

    ModuleNotFoundError: No module named 'dendritic_modeling'

Solution:

.. code-block:: bash

    # Make sure you're in the correct environment
    conda activate dendritic
    
    # Install the package in development mode
    cd /path/to/dendritic-modeling
    pip install -e .

Slow training
~~~~~~~~~~~~~

If training is unusually slow:

Solutions:

1. Enable mixed precision training:

.. code-block:: python

    # Use PyTorch's automatic mixed precision
    from torch.cuda.amp import autocast, GradScaler
    
    scaler = GradScaler()
    for inputs, targets in data_loader:
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

2. Check for bottlenecks with profiling:

.. code-block:: python

    # Profile your training loop
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    print(prof.key_averages().table(sort_by="cuda_time_total"))

Development Setup
-----------------

For developers contributing to the project:

.. code-block:: bash

    # Install development dependencies
    pip install -r requirements-dev.txt
    
    # Set up pre-commit hooks
    pip install pre-commit
    pre-commit install
    
    # Create a new branch for your feature
    git checkout -b feature/your-feature-name
    
    # Make your changes, then test
    pytest
    
    # Format and lint code
    black src/ tests/
    flake8 src/ tests/

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