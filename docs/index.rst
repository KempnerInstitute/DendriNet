Dendritic Modeling Framework Documentation
==========================================

A PyTorch-based framework for biologically-inspired neural networks with dendritic computation and excitation-inhibition dynamics.

Quick Start
-----------

.. code-block:: bash

   # Installation
   git clone https://github.com/KempnerInstitute/dendritic-modeling.git
   cd dendritic-modeling
   conda env create -f environment.yml
   conda activate dendritic_env
   pip install -e .
   
   # Simple run
   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

Core Components
---------------

**Models**
   Classifier and Regressor with encoder-core-decoder architecture

**Networks**
   - **EINet**: Excitation-inhibition networks with dendritic branching
   - **DendriNet**: Pure dendritic networks with hierarchical processing
   - **Classical**: MLP, CNN for baseline comparisons

**Training**
   - Standard training for single GPU
   - FSDP training for large models
   - Parameter sweeps for systematic exploration

Examples
--------

Simple Training Example
^^^^^^^^^^^^^^^^^^^^^^^

Train an EINet model on MNIST:

.. code-block:: bash

   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

FSDP Training Example
^^^^^^^^^^^^^^^^^^^^^

For large models that don't fit on single GPU:

.. code-block:: bash

   # Multi-GPU training
   torchrun --nproc_per_node=4 src/dendritic_modeling/scripts/training/train_experiments_fsdp.py configs/config_exp.yaml

Sweeps Example
^^^^^^^^^^^^^^

Run parameter sweeps for systematic exploration:

.. code-block:: bash

   # EI parameter sweep
   python src/dendritic_modeling/scripts/sweeps/unified/sweep_manager.py ei --config configs/sweeps/ei_sweep.yaml --run
   
   # Branch architecture sweep
   python src/dendritic_modeling/scripts/sweeps/unified/sweep_manager.py branch --config configs/sweeps/branch_sweep.yaml --run

Configuration Guide
-------------------

Core Configuration Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

YAML configuration files control all aspects of training and analysis:

.. code-block:: yaml

   model:
     task: classification  # or regression
     
     core_network:
       type: EINet
       parameters:
         excitatory_layer_sizes: [128, 64]
         inhibitory_layer_sizes: [64, 32]
         use_shunting: true
   
   train:
     learning_strategy: standard  # or two_step, voltage_stabilization
     trainer:
       epochs: 100
       batch_size: 32
       
   task:
     dataset: mnist
     parameters:
       train_size: 8000
       valid_size: 1000

Key Configuration Sections
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table:: Configuration Parameters
   :header-rows: 1
   :widths: 20 30 50

   * - Section
     - Parameter
     - Description
   * - model.core_network
     - type
     - Network architecture ("EINet", "DendriNet", "MLP")
   * - model.core_network.parameters
     - excitatory_layer_sizes
     - List of excitatory layer sizes [128, 64]
   * - model.core_network.parameters
     - inhibitory_layer_sizes
     - List of inhibitory layer sizes [64, 32]
   * - model.core_network.parameters
     - use_shunting
     - Enable shunting inhibition (true/false)
   * - train
     - learning_strategy
     - Training approach ("standard", "two_step", "voltage_stabilization")
   * - train.trainer
     - epochs
     - Number of training epochs
   * - train.trainer
     - batch_size
     - Training batch size
   * - task
     - dataset
     - Dataset name ("mnist", "cifar10", "line")
   * - analysis.information_analysis
     - enabled
     - Enable information-theoretic analysis
   * - wandb
     - use_wandb
     - Enable Weights & Biases logging

Detailed Documentation
----------------------

.. toctree::
   :maxdepth: 1

   installation
   CONFIGURATION_GUIDE
   sweeps
   api_reference
   faq

**Need Help?** Check the FAQ or open an issue on `GitHub <https://github.com/KempnerInstitute/dendritic-modeling/issues>`_.

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
