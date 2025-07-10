Installation Guide
==================

This guide provides detailed instructions for installing the Dendritic Modeling Framework on various platforms and configurations.

Requirements
------------

System Requirements
~~~~~~~~~~~~~~~~~~

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **CUDA** (optional): 11.0+ for GPU acceleration
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **Storage**: 2GB free space

Python Dependencies
~~~~~~~~~~~~~~~~~~

Core dependencies:

- PyTorch >= 1.10.0
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0
- tqdm >= 4.62.0

Optional dependencies:

- CUDA Toolkit (for GPU support)
- Weights & Biases (for experiment tracking)
- TensorBoard (for visualization)
- Jupyter (for notebooks)

Quick Installation
-----------------

Using pip
~~~~~~~~~

The simplest way to install:

.. code-block:: bash

   pip install dendritic-modeling

For GPU support:

.. code-block:: bash

   pip install dendritic-modeling[gpu]

Using conda
~~~~~~~~~~~

With conda/mamba:

.. code-block:: bash

   conda install -c conda-forge dendritic-modeling

Development Installation
-----------------------

Clone and Install
~~~~~~~~~~~~~~~~

For the latest development version:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/dendritic-modeling.git
   cd dendritic-modeling
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e ".[dev]"

Using Poetry
~~~~~~~~~~~

If you prefer Poetry for dependency management:

.. code-block:: bash

   # Clone repository
   git clone https://github.com/yourusername/dendritic-modeling.git
   cd dendritic-modeling
   
   # Install dependencies
   poetry install
   
   # Activate environment
   poetry shell

Platform-Specific Instructions
-----------------------------

Linux
~~~~~

Ubuntu/Debian:

.. code-block:: bash

   # System dependencies
   sudo apt-get update
   sudo apt-get install python3-dev python3-pip
   
   # Install framework
   pip3 install dendritic-modeling

Red Hat/CentOS:

.. code-block:: bash

   # System dependencies
   sudo yum install python3-devel
   
   # Install framework
   pip3 install dendritic-modeling

macOS
~~~~~

Using Homebrew:

.. code-block:: bash

   # Install Python if needed
   brew install python@3.9
   
   # Install framework
   pip3 install dendritic-modeling

For Apple Silicon (M1/M2):

.. code-block:: bash

   # Install with Metal Performance Shaders support
   pip3 install dendritic-modeling[mps]

Windows
~~~~~~~

Using Anaconda (recommended):

.. code-block:: bash

   # Create new environment
   conda create -n dendritic python=3.9
   conda activate dendritic
   
   # Install PyTorch (CPU)
   conda install pytorch torchvision cpuonly -c pytorch
   
   # Install framework
   pip install dendritic-modeling

For GPU support on Windows:

.. code-block:: bash

   # Install CUDA-enabled PyTorch
   conda install pytorch torchvision cudatoolkit=11.3 -c pytorch
   
   # Install framework
   pip install dendritic-modeling[gpu]

GPU Setup
---------

NVIDIA GPUs
~~~~~~~~~~

1. Install CUDA Toolkit:

   .. code-block:: bash

      # Check CUDA version
      nvidia-smi
      
      # Install matching PyTorch
      pip install torch==1.12.0+cu116 -f https://download.pytorch.org/whl/torch_stable.html

2. Verify installation:

   .. code-block:: python

      import torch
      print(torch.cuda.is_available())  # Should print True
      print(torch.cuda.device_count())  # Number of GPUs

AMD GPUs
~~~~~~~~

For ROCm support:

.. code-block:: bash

   # Install ROCm-enabled PyTorch
   pip install torch -f https://download.pytorch.org/whl/rocm5.2/torch_stable.html
   
   # Install framework
   pip install dendritic-modeling

Apple Silicon
~~~~~~~~~~~~

For M1/M2 Macs:

.. code-block:: python

   import torch
   print(torch.backends.mps.is_available())  # Should print True

Docker Installation
------------------

Using Pre-built Image
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Pull image
   docker pull dendriticmodeling/framework:latest
   
   # Run container
   docker run -it --gpus all dendriticmodeling/framework:latest

Building Custom Image
~~~~~~~~~~~~~~~~~~~~

Create a ``Dockerfile``:

.. code-block:: dockerfile

   FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
   
   # Install dependencies
   RUN pip install dendritic-modeling[gpu]
   
   # Set working directory
   WORKDIR /workspace
   
   # Copy your code
   COPY . .
   
   CMD ["python", "train.py"]

Build and run:

.. code-block:: bash

   docker build -t my-dendritic-app .
   docker run -it --gpus all my-dendritic-app

Cluster Installation
-------------------

SLURM Systems
~~~~~~~~~~~~

Example installation script:

.. code-block:: bash

   #!/bin/bash
   #SBATCH --job-name=install
   #SBATCH --time=01:00:00
   #SBATCH --mem=16GB
   
   # Load modules
   module load python/3.9
   module load cuda/11.3
   
   # Create virtual environment
   python -m venv ~/dendritic-env
   source ~/dendritic-env/bin/activate
   
   # Install framework
   pip install dendritic-modeling[gpu]

Distributed Setup
~~~~~~~~~~~~~~~~

For multi-node training:

.. code-block:: bash

   # Install with MPI support
   pip install dendritic-modeling[distributed]
   
   # Install Horovod
   HOROVOD_GPU_OPERATIONS=NCCL pip install horovod

Verification
-----------

Basic Test
~~~~~~~~~

Verify installation:

.. code-block:: python

   import dendritic_modeling as dm
   
   # Check version
   print(dm.__version__)
   
   # Create simple model
   model = dm.networks.ExcitationInhibitionNetwork(
       input_dim=100,
       excitatory_layer_sizes=[64, 32],
       inhibitory_layer_sizes=[16, 8]
   )
   
   # Test forward pass
   import torch
   x = torch.randn(10, 100)
   output = model(x)
   print(f"Output shape: {output.shape}")

GPU Test
~~~~~~~~

Test GPU functionality:

.. code-block:: python

   import torch
   import dendritic_modeling as dm
   
   # Check GPU
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")
   
   # Create model on GPU
   model = dm.networks.ExcitationInhibitionNetwork(
       input_dim=100,
       excitatory_layer_sizes=[64, 32],
       inhibitory_layer_sizes=[16, 8]
   ).to(device)
   
   # Test on GPU
   x = torch.randn(10, 100).to(device)
   output = model(x)
   print(f"GPU computation successful: {output.is_cuda}")

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

**ImportError: No module named 'dendritic_modeling'**
   - Solution: Ensure installation completed successfully
   - Try: ``pip install --upgrade dendritic-modeling``

**CUDA out of memory**
   - Solution: Reduce batch size or model size
   - Try: ``torch.cuda.empty_cache()``

**Slow performance**
   - Check: GPU is being used (``model.cuda()``)
   - Verify: CUDA and cuDNN are properly installed

**Version conflicts**
   - Solution: Create fresh virtual environment
   - Use: ``pip install --upgrade --force-reinstall dendritic-modeling``

Getting Help
~~~~~~~~~~~

If you encounter issues:

1. Check the `FAQ <faq.html>`_
2. Search `GitHub Issues <https://github.com/yourusername/dendritic-modeling/issues>`_
3. Join our `Discord community <https://discord.gg/dendritic>`_
4. Post a detailed issue report

Environment Variables
--------------------

Optional configuration:

.. code-block:: bash

   # Set device preference
   export DENDRITE_DEVICE="cuda:0"
   
   # Enable debug mode
   export DENDRITE_DEBUG="1"
   
   # Set random seed
   export DENDRITE_SEED="42"
   
   # Configure logging
   export DENDRITE_LOG_LEVEL="INFO"

Next Steps
----------

After installation:

1. Follow the `Quick Start Tutorial <tutorials.html#quick-start>`_
2. Explore `Example Scripts <examples.html>`_
3. Read the `API Reference <api_reference.html>`_
4. Join the community!

.. note::

   For the latest updates and announcements, follow our `GitHub repository <https://github.com/yourusername/dendritic-modeling>`_ and star the project! 