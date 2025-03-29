.. _installation_section:

Installation
============

This guide will help you set up the Dendritic Modeling framework on your system.

Prerequisites
-------------

Before installing the package, ensure you have the following:

* Python 3.8 or higher
* CUDA-compatible GPU (recommended but not required)
* Git

Installation Methods
--------------------

There are several ways to install the framework:

Quick Installation
~~~~~~~~~~~~~~~~~~

For a quick installation using pip:

.. code-block:: bash

    pip install dendritic-modeling

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For development or the latest features, install from source:

.. code-block:: bash

    # Clone the repository
    git clone https://github.com/KempnerInstitute/dendritic-modeling.git
    cd dendritic-modeling
    
    # Install in development mode
    pip install -e .

Detailed Installation Steps
---------------------------

If you prefer a step-by-step approach:

Step 1: Set up the Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We recommend using conda or virtualenv to create a clean environment:

.. code-block:: bash

    # Using conda
    conda create -n dendritic python=3.8
    conda activate dendritic
    
    # Or using virtualenv
    python -m venv dendritic-env
    source dendritic-env/bin/activate  # On Windows: dendritic-env\Scripts\activate

Step 2: Install PyTorch
~~~~~~~~~~~~~~~~~~~~~~~

Install PyTorch with the appropriate CUDA version for your system:

.. code-block:: bash

    # For CUDA 11.7
    pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
    
    # For CPU only
    pip install torch==2.0.0 torchvision==0.15.1

Step 3: Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Get the source code:

.. code-block:: bash

    git clone https://github.com/KempnerInstitute/dendritic-modeling.git
    cd dendritic-modeling

Step 4: Install dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install the package and its dependencies:

.. code-block:: bash

    # Install required packages
    pip install -r requirements.txt
    
    # Install the package in development mode
    pip install -e .
    
    # Install additional development dependencies (optional)
    pip install -r requirements-dev.txt

Step 5: Verify installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure that the installation was successful:

.. code-block:: bash

    # Run a simple test
    python -c "import dendritic_modeling; print(dendritic_modeling.__version__)"
    
    # Run the test suite
    pytest tests/

Configuring for Different Hardware
----------------------------------

CPU-only configuration
~~~~~~~~~~~~~~~~~~~~~~

For systems without a GPU:

.. code-block:: bash

    # Set this environment variable to disable CUDA
    export CUDA_VISIBLE_DEVICES=""

Multi-GPU configuration
~~~~~~~~~~~~~~~~~~~~~~~

For systems with multiple GPUs:

.. code-block:: bash

    # To use specific GPUs
    export CUDA_VISIBLE_DEVICES="0,1"
    
    # In your Python code
    model = model.to('cuda')  # Uses GPU 0
    # Or for multi-GPU training
    model = torch.nn.DataParallel(model)

Cluster configuration (SLURM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For HPC environments using SLURM:

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
-------------------

Once installed, the package structure is as follows:

.. code-block:: none

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
    └── notebooks/                # Example notebooks

Common Issues and Solutions
---------------------------

CUDA version mismatch
~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~

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