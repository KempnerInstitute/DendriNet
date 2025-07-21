Installation and Setup
====================

This guide covers everything you need to get the Dendritic Modeling Framework up and running, from basic installation to advanced multi-GPU setups.

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~

**Minimum Requirements:**
- Python 3.8 or higher
- 8 GB RAM  
- CUDA-compatible GPU (recommended)
- 10 GB free disk space

**Recommended Requirements:**
- Python 3.9+
- 16+ GB RAM
- NVIDIA GPU with 8+ GB VRAM
- CUDA 11.8 or 12.x
- 50+ GB free disk space (for datasets and results)

**For Large-Scale Training:**
- Multiple GPUs (4-8 recommended)
- 32+ GB RAM
- High-speed interconnect (NVLink/InfiniBand)
- NCCL for multi-GPU communication

Software Dependencies
~~~~~~~~~~~~~~~~~~~~

**Required:**
- Git
- Conda or Miniconda
- CUDA Toolkit (for GPU support)

**Optional but Recommended:**
- cuDNN (for optimized training)
- Weights & Biases account (for experiment tracking)

Quick Installation
------------------

The fastest way to get started:

.. code-block:: bash

   # 1. Clone the repository
   git clone https://github.com/KempnerInstitute/dendritic-modeling.git
   cd dendritic-modeling

   # 2. Create and activate conda environment
   conda env create -f environment.yml
   conda activate dendritic_env

   # 3. Install the package
   pip install -e .

   # 4. Verify installation
   python -c "import dendritic_modeling; print('Installation successful!')"

That's it! You're ready to run your first experiment.

Detailed Installation Options
-----------------------------

Development Installation
~~~~~~~~~~~~~~~~~~~~~~~~

For contributors and developers:

.. code-block:: bash

   # Clone with development tools
   git clone https://github.com/KempnerInstitute/dendritic-modeling.git
   cd dendritic-modeling

   # Create environment with dev dependencies
   conda env create -f environment.yml
   conda activate dendritic_env

   # Install in development mode with extras
   pip install -e ".[dev]"

   # Set up pre-commit hooks
   pre-commit install

   # Run tests to verify
   pytest tests/

Custom Environment Installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer managing dependencies manually:

.. code-block:: bash

   # Create custom environment
   conda create -n dendritic_custom python=3.9
   conda activate dendritic_custom

   # Install PyTorch (choose appropriate CUDA version)
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

   # Install other dependencies
   pip install -r requirements.txt

   # Install framework
   pip install -e .

Docker Installation
~~~~~~~~~~~~~~~~~~

For containerized environments:

.. code-block:: bash

   # Pull pre-built image (if available)
   docker pull ghcr.io/kempnerinstitute/dendritic-modeling:latest

   # Or build from source
   docker build -t dendritic-modeling .

   # Run container with GPU support
   docker run --gpus all -it dendritic-modeling

Cluster/HPC Installation
~~~~~~~~~~~~~~~~~~~~~~~

For high-performance computing environments:

.. code-block:: bash

   # Load required modules (example for SLURM)
   module load python/3.9
   module load cuda/11.8
   module load cudnn/8.9.2

   # Install in user space
   pip install --user -e .

   # Verify NCCL for multi-node training
   python -c "import torch; print(torch.distributed.is_nccl_available())"

Verification
------------

Basic Verification
~~~~~~~~~~~~~~~~~

Run these commands to verify your installation:

.. code-block:: bash

   # Test basic import
   python -c "import dendritic_modeling; print('✓ Framework imported successfully')"

   # Test GPU availability
   python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')"

   # Test framework components
   python -c "from dendritic_modeling.models import Classifier; print('✓ Models working')"

Quick Test Run
~~~~~~~~~~~~~

Run a minimal experiment to verify everything works:

.. code-block:: bash

   # Run quick test (should complete in <2 minutes)
   python src/dendritic_modeling/scripts/train_experiments.py configs/config_exp.yaml \
       --output_dir ./test_run

   # Check for output files
   ls test_run/

Expected output structure:

.. code-block:: text

   test_run/
   ├── checkpoints/
   ├── logs/
   ├── analysis/
   └── config.json

GPU Verification
~~~~~~~~~~~~~~~

For GPU setups, verify CUDA integration:

.. code-block:: python

   import torch
   import dendritic_modeling

   # Check CUDA
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU count: {torch.cuda.device_count()}")

   # Test GPU memory
   if torch.cuda.is_available():
       device = torch.device('cuda:0')
       x = torch.randn(1000, 1000, device=device)
       print(f"✓ GPU tensor created: {x.device}")

Advanced Setup
--------------

Multi-GPU Configuration
~~~~~~~~~~~~~~~~~~~~~~

For distributed training with FSDP:

.. code-block:: bash

   # Install additional dependencies
   pip install accelerate

   # Verify NCCL for communication
   python -c "import torch; print(f'NCCL available: {torch.distributed.is_nccl_available()}')"

   # Test multi-GPU setup
   torchrun --nproc_per_node=2 -m torch.distributed.launch \
       --nnodes=1 --master_port=29500 \
       src/dendritic_modeling/scripts/train_experiments.py \
       configs/config_exp.yaml --use_fsdp

Weights & Biases Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For experiment tracking:

.. code-block:: bash

   # Install wandb
   pip install wandb

   # Login (creates ~/.netrc)
   wandb login

   # Test integration
   python -c "import wandb; print('✓ WandB ready')"

Custom Dataset Setup
~~~~~~~~~~~~~~~~~~~

For custom datasets, create appropriate directory structure:

.. code-block:: bash

   # Create data directory
   mkdir -p data/custom_dataset

   # Set environment variable (optional)
   export DENDRITIC_DATA_PATH="$(pwd)/data"

Environment Variables
~~~~~~~~~~~~~~~~~~~~

Useful environment variables for customization:

.. code-block:: bash

   # Data paths
   export DENDRITIC_DATA_PATH="/path/to/datasets"
   export DENDRITIC_RESULTS_PATH="/path/to/results"

   # CUDA settings
   export CUDA_VISIBLE_DEVICES="0,1,2,3"  # Specify GPUs
   export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"  # Memory management

   # For FSDP/distributed training
   export MASTER_ADDR="localhost"
   export MASTER_PORT="29500"
   export WORLD_SIZE="4"

Troubleshooting
---------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Installation Fails with Permission Errors**

.. code-block:: bash

   # Use --user flag
   pip install --user -e .

   # Or fix conda permissions
   conda config --set always_yes true

**CUDA/GPU Not Detected**

.. code-block:: bash

   # Check CUDA installation
   nvidia-smi
   nvcc --version

   # Reinstall PyTorch with correct CUDA version
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

**Import Errors**

.. code-block:: bash

   # Verify package installation
   pip list | grep dendritic

   # Reinstall in development mode
   pip install -e . --force-reinstall

**Memory Issues on Large Models**

.. code-block:: bash

   # Enable memory efficient settings
   export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

   # Use smaller batch sizes
   # Modify config: train.trainer.batch_size: 32

**FSDP Communication Timeouts**

See detailed troubleshooting in :doc:`troubleshooting` and the FSDP troubleshooting guide.

**Environment Conflicts**

.. code-block:: bash

   # Create clean environment
   conda deactivate
   conda env remove -n dendritic_env
   conda env create -f environment.yml

Platform-Specific Notes
~~~~~~~~~~~~~~~~~~~~~~

**Windows**

.. code-block:: bash

   # Use conda for package management
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

   # Git Bash recommended for commands
   # Adjust paths with backslashes if needed

**macOS**

.. code-block:: bash

   # For Apple Silicon Macs
   conda install pytorch torchvision torchaudio -c pytorch

   # Note: GPU training not supported on macOS

**Linux Clusters**

.. code-block:: bash

   # Often require module loading
   module load gcc/9.3.0
   module load cuda/11.8
   module load python/3.9

   # Check cluster-specific documentation

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~

For optimal performance:

.. code-block:: bash

   # Enable optimized PyTorch build
   pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118

   # Install NVIDIA Apex (optional, for advanced optimizations)
   git clone https://github.com/NVIDIA/apex
   cd apex
   pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

Next Steps
----------

Now that you have the framework installed:

1. **Quick Start**: Try the :doc:`quickstart` tutorial
2. **Core Concepts**: Understand the framework in :doc:`core_concepts`  
3. **Configuration**: Learn about options in :doc:`configuration/index`
4. **Examples**: Explore :doc:`tutorials/index`

.. note::
   If you encounter any installation issues not covered here, please check our `GitHub Issues <https://github.com/KempnerInstitute/dendritic-modeling/issues>`_ or create a new issue with your system details.

Keeping Updated
---------------

To update to the latest version:

.. code-block:: bash

   # Pull latest changes
   cd dendritic-modeling
   git pull origin main

   # Update dependencies
   conda env update -f environment.yml

   # Reinstall package
   pip install -e . --force-reinstall

   # Run tests to verify
   python -c "import dendritic_modeling; print('✓ Update successful')" 