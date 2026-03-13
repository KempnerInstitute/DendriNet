Dendritic Modeling Framework
============================

A framework for biologically-inspired neural networks with excitation-inhibition
dynamics and dendritic computation.

.. code-block:: bash

   pip install -e .
   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

The framework uses a three-part model: **encoder** (Identity, MLP, CNN) --
**core network** (EINet, DendriNet, MLP, RNN) -- **decoder** (Identity, MLP).
All experiments are configured through YAML files.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   network_architectures
   training_strategies
   datasets
   analysis
   CONFIGURATION_GUIDE
   sweeps
   FSDP_TROUBLESHOOTING

.. toctree::
   :maxdepth: 2
   :caption: Reference

   config_reference
   available_components
   contributing

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
