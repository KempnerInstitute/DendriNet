Installation Guide
==================

This guide provides simple instructions for installing the Dendritic Modeling Framework using conda.

Requirements
------------

System Requirements
~~~~~~~~~~~~~~~~~~

- **Operating System**: Linux, macOS, or Windows
- **Python**: 3.12
- **CUDA** (optional): 11.0+ for GPU acceleration
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **Storage**: 2GB free space

Prerequisites
~~~~~~~~~~~~

- Anaconda or Miniconda installed on your system
- Git for cloning the repository

Quick Installation
-----------------

Step 1: Clone the Repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/KempnerInstitute/dendritic-modeling.git
   cd dendritic-modeling

Step 2: Create Conda Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create the environment from the provided environment.yml file
   conda env create -f environment.yml
   
   # Activate the environment
   conda activate dendritic_env

Step 3: Install the Package
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install the package in development mode
   pip install -e .

That's it! The framework is now installed and ready to use.

GPU Support
----------

The environment.yml file includes PyTorch with CUDA support. To verify GPU availability:

.. code-block:: python

   import torch
   print(torch.cuda.is_available())  # Should print True if GPU is available
   print(torch.cuda.device_count())  # Number of GPUs

If you need a different CUDA version, modify the ``pytorch-cuda`` version in environment.yml before creating the environment:

.. code-block:: yaml

   dependencies:
     - pytorch-cuda=11.8  # Change to your CUDA version

Platform-Specific Notes
----------------------

macOS with Apple Silicon
~~~~~~~~~~~~~~~~~~~~~~~

For M1/M2/M3 Macs, PyTorch will automatically use Metal Performance Shaders (MPS):

.. code-block:: python

   import torch
   print(torch.backends.mps.is_available())  # Should print True

Windows
~~~~~~~

On Windows, use Anaconda Prompt or PowerShell to run the installation commands.

Cluster Installation
-------------------

For HPC clusters with module systems:

.. code-block:: bash

   # Load required modules (example)
   module load anaconda3
   module load cuda/11.8
   
   # Create and activate environment
   conda env create -f environment.yml
   conda activate dendritic_env
   
   # Install package
   pip install -e .

Verification
-----------

Verify your installation:

.. code-block:: python

   # Test import
   import dendritic_modeling as dm
   print(dm.__version__)
   
   # Create a simple model
   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   
   model = ExcitationInhibitionNetwork(
       input_dim=100,
       transfer_params={"type": "identity"},
       excitatory_layer_sizes=[64, 32],
       inhibitory_layer_sizes=[32, 16],
       excitatory_branch_factors=[2],
       inhibitory_branch_factors=[2]
   )
   
   # Test forward pass
   import torch
   x = torch.randn(10, 100)
   output = model(x)
   print(f"Output shape: {output.shape}")

Common Issues
~~~~~~~~~~~~

**Import errors after installation**
   - Make sure you activated the conda environment: ``conda activate dendritic_env``
   - Verify installation: ``pip list | grep dendritic``

**CUDA/GPU not detected**
   - Check CUDA installation: ``nvidia-smi``
   - Ensure PyTorch CUDA version matches your system CUDA version

**Memory errors during training**
   - Reduce batch size in your configuration
   - Use gradient accumulation for larger effective batch sizes

Next Steps
----------

1. Follow the :doc:`Quick Start Tutorial <tutorials>` to run your first experiment
2. Read about :doc:`Network Architectures <NETWORK_ARCHITECTURES>` 
3. Learn about :doc:`Training Strategies <TRAINING_STRATEGIES>`
4. Explore :doc:`Example Configurations <examples>`

For help, please check our `GitHub Issues <https://github.com/KempnerInstitute/dendritic-modeling/issues>`_ or create a new issue with your question. 