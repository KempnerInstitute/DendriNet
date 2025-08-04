Dendritic Modeling Framework
============================

PyTorch framework for biologically-inspired neural networks with dendritic computation and excitation-inhibition dynamics.

Architecture
-----------

Three-part model architecture:

1. Encoder Network: Processes input data
2. Core Network: Main computation (EINet, DendriNet, MLP)  
3. Decoder Network: Maps to output space

Core Networks:

- EINet: Excitation-inhibition networks with dendritic computation
- DendriNet: Hierarchical branching dendritic networks
- MLP: Standard multi-layer perceptrons

Training Options
---------------

Standard Training
   Single-GPU training with analysis

Enhanced Training
   Multiple optimizers, regularization, pruning, and detailed analysis

FSDP Training
   Multi-GPU distributed training with automated resource estimation

Parameter Sweeps
   Systematic hyperparameter exploration

Quick Start
----------

See :doc:`QUICK_START` for basic experiment examples.

Documentation
-------------

.. toctree::
   :maxdepth: 2

   QUICK_START
   CONFIGURATION_GUIDE
   FSDP_TRAINING_GUIDE
   sweeps

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
