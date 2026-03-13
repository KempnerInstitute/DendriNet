Dendritic Modeling Framework
============================

A framework for excitation-inhibition networks, dendritic architectures,
baseline MLPs, recurrent models, training strategies, and analysis tools.

.. code-block:: bash

   pip install -e .
   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

Every model follows the same three-part layout: **encoder** (Identity, MLP,
CNN), **core network** (E-I core, dendritic MLP, MLP, or recurrent core), and
**decoder** (Identity or MLP). All experiments are configured through YAML
files.

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
   configuration_guide
   sweeps
   fsdp_troubleshooting

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
