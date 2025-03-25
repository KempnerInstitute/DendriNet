.. _installation_setup:

Installation and Setup
======================

This document provides step-by-step instructions for installing and setting up the dendritic modeling framework, including environment configuration, dependencies, and troubleshooting common issues.

Overview
--------

The dendritic modeling framework is a PyTorch-based library for implementing and experimenting with dendritic neural networks. This guide will help you:

1. Set up a Python environment for the framework
2. Install the framework and its dependencies
3. Verify the installation
4. Configure your environment for different hardware
5. Troubleshoot common installation issues

System Requirements
-------------------

Before installing, ensure your system meets the following requirements:

* **Python**: 3.8 or higher
* **CUDA** (for GPU support): Version 11.0 or higher recommended
* **Operating System**: Linux (recommended), macOS, or Windows
* **Disk Space**: At least 5GB for the framework and datasets
* **Memory**: At least 8GB RAM (16GB or more recommended)
* **GPU**: NVIDIA GPU with at least 4GB VRAM (recommended for training)

Installation Methods
--------------------

There are two main methods to install the framework:

1. Using pip (recommended for most users)
2. Installing from source (recommended for developers)

Method 1: Using pip
~~~~~~~~~~~~~~~~~~~

The simplest way to install the framework is using pip:

.. code-block:: bash

    # Create a new conda environment
    conda create -n dendritic python=3.8
    conda activate dendritic
    
    # Install PyTorch with CUDA (modify cudatoolkit version if needed)
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    
    # Install the dendritic modeling framework
    pip install dendritic-modeling

Method 2: Installing from source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the latest features or if you plan to contribute to the framework:

.. code-block:: bash

    # Create a new conda environment
    conda create -n dendritic python=3.8
    conda activate dendritic
    
    # Install PyTorch with CUDA
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    
    # Clone the repository
    git clone https://github.com/yourusername/dendritic-modeling.git
    cd dendritic-modeling
    
    # Install in development mode
    pip install -e .

Detailed Installation Steps
---------------------------

Here's a more detailed guide to installing the framework from source:

Step 1: Set up the Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    source dendritic-env/bin/activate

Step 2: Install PyTorch
~~~~~~~~~~~~~~~~~~~~~~~

Install PyTorch with GPU support:

.. code-block:: bash

    # For CUDA 11.3
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    
    # For CUDA 11.6
    conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch
    
    # For CPU only
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

Verify PyTorch installation:

.. code-block:: bash

    # Verify PyTorch and CUDA
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

Step 3: Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/yourusername/dendritic-modeling.git
    cd dendritic-modeling

Step 4: Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Install dependencies
    pip install -r requirements.txt
    
    # Install development dependencies (optional)
    pip install -r requirements-dev.txt
    
    # Install the package in development mode
    pip install -e .

The ``requirements.txt`` file includes all the necessary dependencies:

.. code-block:: bash

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run a simple test to verify the installation:

.. code-block:: bash

    # Run a simple test
    python -c "from dendritic_modeling.networks import ExcitationInhibitionNetwork; print('Installation successful!')"
    
    # Run the test suite
    pytest -xvs tests/

Configuring for Different Hardware
----------------------------------

CPU-only configuration
~~~~~~~~~~~~~~~~~~~~~~

If you don't have a GPU, modify the installation:

.. code-block:: bash

    # Install PyTorch for CPU only
    conda install pytorch torchvision torchaudio cpuonly -c pytorch

Multi-GPU configuration
~~~~~~~~~~~~~~~~~~~~~~~

For systems with multiple GPUs:

.. code-block:: bash

    # Set visible devices (e.g., use only GPUs 0 and 1)
    export CUDA_VISIBLE_DEVICES=0,1
    
    # Check available GPUs
    python -c "import torch; print(f'Number of GPUs: {torch.cuda.device_count()}'); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"

The framework automatically uses all available GPUs for data-parallel training.

Cluster configuration (SLURM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For HPC clusters with SLURM:

1. Create a conda environment in your home directory
2. Install the framework as described above
3. Use the provided SLURM scripts for job submission (see ``sweep_job_submission.rst``)

.. code-block:: bash

    # Example SLURM job submission
    sbatch scripts/slurm/run_experiment.sh configs/config_exp.yaml

Directory Structure
-------------------

After installation, the framework has the following directory structure:

.. code-block:: text

    dendritic-modeling/
    ├── configs/                # Configuration files
    ├── data/                   # Dataset storage (created when needed)
    ├── dendritic_modeling/     # Source code
    │   ├── __init__.py
    │   ├── networks.py         # Network architectures
    │   ├── activation_functions.py  # Activation functions
    │   ├── learning_strategies.py   # Learning strategies
    │   ├── config.py           # Configuration utilities
    │   ├── plot_utils.py       # Visualization utilities
    │   └── ...
    ├── docs/                   # Documentation
    ├── outputs/                # Experiment outputs
    ├── scripts/                # Utility scripts
    ├── tests/                  # Unit tests
    ├── README.md               # Overview and instructions
    ├── LICENSE                 # License information
    ├── pyproject.toml          # Python package configuration
    └── requirements.txt        # Dependencies

Common Issues and Solutions
---------------------------

CUDA version mismatch
~~~~~~~~~~~~~~~~~~~~~

**Issue**: Error messages related to CUDA version incompatibility.

**Solution**:

1. Check your CUDA version:

   .. code-block:: bash

       nvcc --version

2. Install the matching PyTorch version:

   .. code-block:: bash

       conda install pytorch torchvision torchaudio cudatoolkit=<your_cuda_version> -c pytorch

3. If needed, reinstall the framework:

   .. code-block:: bash

       pip uninstall dendritic-modeling
       pip install -e .

Out of memory errors
~~~~~~~~~~~~~~~~~~~~

**Issue**: CUDA out of memory errors during training.

**Solution**:

1. Reduce batch size in the configuration:

   .. code-block:: yaml

       train:
         batch_size: 32  # Try a smaller batch size

2. Use gradient accumulation:

   .. code-block:: yaml

       train:
         accumulate_grad_batches: 2  # Accumulate gradients over 2 batches

3. Use a smaller model:

   .. code-block:: yaml

       model:
         network:
           parameters:
             excitatory_layer_sizes: [10]  # Try smaller layer sizes
             inhibitory_layer_sizes: [20]

Module not found errors
~~~~~~~~~~~~~~~~~~~~~~~

**Issue**: ``ModuleNotFoundError`` when importing from the framework.

**Solution**:

1. Ensure the package is installed:

   .. code-block:: bash

       pip list | grep dendritic-modeling

2. Reinstall in development mode:

   .. code-block:: bash

       pip install -e .

3. Check your Python path:

   .. code-block:: bash

       python -c "import sys; print(sys.path)"

Slow training
~~~~~~~~~~~~~

**Issue**: Training is unexpectedly slow.

**Solution**:

1. Check if CUDA is being used:

   .. code-block:: bash

       python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

2. Enable GPU acceleration in your configuration:

   .. code-block:: yaml

       # No specific setting needed; the framework uses CUDA automatically if available

3. Use a profiler to identify bottlenecks:

   .. code-block:: python

       from torch.profiler import profile, record_function, ProfilerActivity
       
       with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
           # Your training code here
       
       print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

Development Setup
-----------------

If you plan to contribute to the framework, additional setup is recommended:

1. Install development dependencies:

   .. code-block:: bash

       pip install -r requirements-dev.txt

2. Set up pre-commit hooks:

   .. code-block:: bash

       pre-commit install

3. Configure your IDE (e.g., VSCode, PyCharm) with:
   - Black formatter
   - Flake8 or Pylint linter
   - pytest for running tests

Starting Your First Experiment
------------------------------

After installation, you can run your first experiment:

1. Prepare a configuration file:

   .. code-block:: bash

       cp configs/config_exp.yaml configs/my_experiment.yaml
       # Edit configs/my_experiment.yaml as needed

2. Run the training script:

   .. code-block:: bash

       python scripts/train_experiments.py configs/my_experiment.yaml

3. Monitor the output:

   .. code-block:: bash

       # Check training progress
       tail -f outputs/my_experiment/output.log

4. Visualize results:

   .. code-block:: bash

       # If wandb is enabled
       echo "Visit https://wandb.ai to see your results"
       
       # Or use local visualizations
       ls outputs/my_experiment/figures/

Next Steps
----------

After setting up the framework, you can:

1. Read the documentation on network structures and learning strategies
2. Explore example configurations in the ``configs`` directory
3. Try training on different datasets
4. Experiment with different dendritic architectures

.. figure:: figures/pdf/place_holder.pdf
   :alt: Installation Workflow
   :width: 600px
   :align: center

   Overview of the installation and setup process. [PLACEHOLDER FOR DIAGRAM] 