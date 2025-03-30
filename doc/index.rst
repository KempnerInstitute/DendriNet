.. Dendritic Modeling documentation master file

Welcome to Dendritic Modeling's Documentation
=============================================

.. image:: _static/dendritic_network_banner.png
   :alt: Dendritic Network Illustration
   :align: center
   :width: 600px

**Dendritic Modeling** is a Python framework for creating, training, and analyzing artificial neural networks with dendritic-inspired architectures. These networks incorporate biological principles from dendrites, facilitating more complex computational strategies than traditional neural networks.

.. important::
   This framework is designed for researchers in computational neuroscience, machine learning, and artificial intelligence interested in biologically-inspired computing.

Key Features
------------

* **Biologically-inspired architectures**: Models with dendritic branching structures
* **Excitation-Inhibition Networks**: Separate excitatory and inhibitory pathways
* **Specialized learning strategies**: Algorithms designed for complex dendritic topologies
* **Visualization tools**: Comprehensive analysis of network behaviors
* **Configurable experiments**: YAML-based experiment configuration

Getting Started
---------------

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   users/installation
   users/quickstart
   users/examples
   users/concepts
   users/configuration
   users/experiment_tracking

.. toctree::
   :maxdepth: 1
   :caption: Components

   dend_m/network_structures
   dend_m/learning_strategies
   dend_m/gradient_scaling
   dend_m/activation_functions
   dend_m/weight_initialization
   dend_m/synthetic_datasets
   dend_m/evaluation_visualization

.. toctree::
   :maxdepth: 1
   :caption: Contributing

   dend_m/contributing
   dend_m/visualization_guide

.. toctree::
   :maxdepth: 1
   :caption: Reference

   dendritic_modeling
   about
   glossary
   additional_resources

Citation
--------

If you use this framework in your research, please cite:

.. code-block:: bibtex

   @software{dendritic_modeling,
     author = {Kempner Lab},
     title = {Dendritic Modeling: A Framework for Biologically-Inspired Neural Networks},
     url = {https://github.com/kempnerlab/dendritic-modeling},
     version = {0.1.0},
     year = {2023},
   }

See :doc:`about` for more information on the project and its contributors.

Indices and Tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
