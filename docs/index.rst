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
   
   # Simple training
   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

Architecture Overview
---------------------

The framework uses a three-part model architecture:

1. **Encoder Network**: Processes raw input data (Identity, MLP, CNN)
2. **Core Network**: Main computational unit (EINet, DendriNet, MLP)
3. **Decoder Network**: Maps to output space (Identity, MLP)

Core Networks
^^^^^^^^^^^^^

**EINet (Excitation-Inhibition Network)**
   Implements biologically-inspired dendritic computation with separate excitatory and inhibitory pathways.

**DendriNet**
   Pure dendritic network with hierarchical branching structure.

**MLP**
   Standard multi-layer perceptron for baseline comparisons.

Training Options
----------------

**Standard Training**
   Single-GPU training with standard backpropagation.

**Enhanced Training**
   Advanced training with multiple optimizers, regularization, and pruning:
   
   - Multiple optimizers: Adam, SGD, RMSprop, AdamW
   - L1/L2 regularization with selective application
   - Weight pruning with magnitude thresholds and post-pruning analysis
   - Synaptic turnover analysis with active synapse tracking
   - Performance comparison plots and detailed branch-level statistics
   - Complete re-analysis on pruned models

**FSDP Training**
   Distributed training for large models across multiple GPUs.

**Parameter Sweeps**
   Systematic exploration of hyperparameter spaces.

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
