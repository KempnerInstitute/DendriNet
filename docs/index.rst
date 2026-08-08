Dendritic Modeling Framework
============================

A framework for excitation-inhibition networks, dendritic architectures,
baseline MLPs, recurrent models, training strategies, and analysis tools.

.. figure:: _static/architecture/system-overview.svg
   :alt: System overview showing configs, canonicalization, registries, model factory, training, analysis, and artifacts.
   :width: 100%

   The framework is configured from YAML, canonicalized into typed model
   settings, built through architecture registries and factories, trained with
   standard or local learning rules, and analyzed into reusable artifacts.

.. code-block:: bash

   pip install -e .
   python src/dendritic_modeling/scripts/training/train_experiments.py configs/config_exp.yaml

Every model follows the same three-part layout: **encoder** (Identity, MLP,
CNN), **core network** (E-I core, dendritic MLP, MLP, or recurrent core), and
**decoder** (Identity or MLP). All experiments are configured through YAML
files.

Start here:

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Goal
     - Page
     - First config to open
   * - Run an E/I MNIST model
     - :doc:`quickstart`
     - ``configs/examples/population_network_feedforward.yaml``
   * - Configure feedforward or recurrent population models
     - :doc:`network_architectures`
     - ``configs/examples/population_network_feedforward.yaml``
   * - Browse config examples
     - :doc:`examples`
     - ``configs/examples/population_network_feedforward.yaml``
   * - Add heterogeneous morphologies, synapse types, spiking soma, or feedback
     - :doc:`available_components`
     - ``configs/examples/population_network_heterogeneous_feedforward.yaml``
   * - Sweep E/I balance, morphology, taus, or connection probabilities
     - :doc:`sweeps`
     - ``configs/sweep_config.yaml``
   * - Inspect metrics, routing, gates, spikes, and branch analyses
     - :doc:`analysis`
     - ``analysis:`` in ``configs/config_exp.yaml``

``model.core.type: population_network`` is the main notation for feedforward
stacks, recurrent microcircuits, named E/I populations, per-population
morphologies, typed synapses, spiking soma, and soma-to-dendrite feedback. E/I
alias configs are supported for saved experiments and comparison runs.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   examples

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   network_architectures
   training_strategies
   datasets
   analysis
   engineering_architecture
   configuration_guide
   transformer_llm_scaling
   initialization_comparison_policy
   sweeps
   fsdp_troubleshooting

.. toctree::
   :maxdepth: 2
   :caption: Reference

   config_reference
   available_components
   iss373_integration_notes
   contributing

Indices and Tables
==================

* :ref:`genindex`
* :ref:`search`
