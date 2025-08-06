Dendritic Modeling Framework
============================

A framework for biologically-inspired neural networks with excitation-inhibition dynamics and dendritic computation.

Quick Start
-----------

Installation:

.. code-block:: bash

   git clone https://github.com/KempnerInstitute/dendritic-modeling.git
   cd dendritic-modeling
   conda env create -f environment.yml
   conda activate dendritic_env
   pip install -e .

Basic Training:

.. code-block:: bash

   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

Architecture Overview
---------------------

The framework uses a three-part model architecture:

1. Encoder Network: Processes raw input data (Identity, MLP, CNN)
2. Core Network: Main computational unit (EINet, DendriNet, MLP)
3. Decoder Network: Maps to output space (Identity, MLP)

Core Networks
^^^^^^^^^^^^^

EINet (Excitation-Inhibition Network)
   Implements biologically-inspired dendritic computation with separate excitatory and inhibitory pathways.

DendriNet
   Pure dendritic network with hierarchical branching structure.

MLP
   Standard multi-layer perceptron for baseline comparisons.

Training Options
----------------

Standard Training
   Single-GPU training with standard backpropagation.

   .. code-block:: bash

      # Standard Training
      python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

      # Or with command line overrides
      python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml --epochs 100

FSDP Training
   Distributed training for large models across multiple GPUs.

   .. code-block:: bash

      # Generate and submit FSDP job (default behavior)
      python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml

      # Generate SLURM script only
      python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --generate-only

      # Dry run to see what would be submitted
      python src/dendritic_modeling/scripts/tools/submit_fsdp_job.py configs/config_exp_fsdp.yaml --dry-run

Parameter Sweeps
   Systematic exploration of hyperparameter spaces.

   .. code-block:: bash

      # Generate and run sweep (default behavior)
      python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/unified_all_sweeps_example.yaml

      # Generate configs only (no job submission)
      python src/dendritic_modeling/scripts/sweeps/sweep_manager.py --config configs/unified_all_sweeps_example.yaml --generate-only

      # Analyze sweep results with auto-detection
      python src/dendritic_modeling/scripts/sweeps/result_analyzer.py --results-dir output/sweep_20240101120000 --auto-detect

Configuration
-------------

All experiments are configured through YAML files. Key sections:

- ``model``: Network architecture and parameters
- ``train``: Training hyperparameters and strategies
- ``task``: Dataset configuration
- ``analysis``: Optional analysis settings
- ``wandb``: Experiment tracking

Documentation
-------------

.. toctree::
   :maxdepth: 2

   CONFIGURATION_GUIDE
   sweeps
   FSDP_TROUBLESHOOTING

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
