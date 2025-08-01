Quick Start Guide
================

Basic experiments to get started with dendritic modeling.

Installation
-----------

.. code-block:: bash

   git clone https://github.com/KempnerInstitute/dendritic-modeling.git
   cd dendritic-modeling
   conda env create -f environment.yml
   conda activate dendritic_env
   pip install -e .

Basic Experiment
---------------

Single-GPU training with CIFAR-10:

.. code-block:: bash

   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

This trains an EI network on CIFAR-10 with performance analysis.

Sweep Experiment
---------------

Parameter sweep across multiple configurations:

.. code-block:: bash

   # Generate sweep configurations
   python -m dendritic_modeling.scripts.sweeps.unified.sweep_manager \
     --config configs/examples/general_sweep_example.yaml --generate-only

   # Run sweep
   python -m dendritic_modeling.scripts.sweeps.unified.sweep_manager \
     --config configs/examples/general_sweep_example.yaml --run

FSDP Experiment
--------------

Large model training across multiple GPUs:

.. code-block:: bash

   # Automated FSDP workflow
   python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py \
     configs/config_exp_fsdp.yaml --submit

   # Direct FSDP training (4 GPUs)
   torchrun --nproc_per_node=4 \
     src/dendritic_modeling/scripts/training/train_experiments_fsdp.py \
     configs/config_exp_fsdp.yaml

Configuration
------------

Basic configuration structure:

.. code-block:: yaml

   model:
     task: classification
     core_network:
       type: EINet
       parameters:
         excitatory_layer_sizes: [10]
         inhibitory_layer_sizes: [3]
         excitatory_branch_factors: [2]
         inhibitory_branch_factors: [1]
         ee_synapses_per_branch_per_layer: [16]
         ei_synapses_per_branch_per_layer: [8]
         ie_synapses_per_branch_per_layer: [4]
         ii_synapses_per_branch_per_layer: [2]

   train:
     main:
       trainer:
         epochs: 100
         batch_size: 128
         loss_function: ce

   task:
     dataset: cifar10

   analysis:
     performance_analysis:
       enabled: true

Output Structure
---------------

Training generates organized results:

.. code-block:: text

   results/experiment_TIMESTAMP/
   ├── performance/           # Performance metrics
   ├── final_model.pt        # Trained model
   └── config.yaml          # Used configuration

Next Steps
---------

- See :doc:`CONFIGURATION_GUIDE` for detailed configuration options
- See :doc:`FSDP_TRAINING_GUIDE` for large model training  
- See :doc:`sweeps` for parameter exploration
- See example configurations in ``configs/examples/``