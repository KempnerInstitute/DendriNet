User Guide
==========

This comprehensive user guide covers everything you need to know about using the Dendritic Modeling Framework, from basic concepts to advanced configurations and analysis techniques.

.. toctree::
   :maxdepth: 2
   :caption: User Guide Contents

   getting_started
   architecture_and_models
   training_and_configuration
   analysis_and_visualization

Overview
--------

The user guide is organized into four main sections:

**Getting Started**
   Installation, quickstart tutorial, and basic usage patterns. Learn how to set up the framework and run your first experiments.

**Architecture and Models**  
   Detailed coverage of network architectures, model types, and biological foundations. Understand how to build and configure dendritic networks.

**Training and Configuration**
   Comprehensive guide to training strategies, configuration systems, and experiment management. Master the tools for reproducible research.

**Analysis and Visualization**
   Advanced techniques for analyzing network behavior, information flow, and creating publication-quality visualizations.

Key Concepts
-----------

Before diving into the specific sections, here are the fundamental concepts that underpin the framework:

Shunting Inhibition
~~~~~~~~~~~~~~~~~~

The core computational mechanism implementing divisive normalization:

.. math::

   V_{out} = \frac{E}{E + I + \epsilon}

Where:
- :math:`E` = sum of excitatory inputs
- :math:`I` = sum of inhibitory inputs  
- :math:`\epsilon` = small constant for numerical stability

This computation provides gain control and normalization, similar to mechanisms found throughout the nervous system.

Dendritic Architecture
~~~~~~~~~~~~~~~~~~~~~

Multi-dendrite neurons where each dendrite:

1. Receives excitatory and inhibitory synaptic inputs
2. Performs local shunting inhibition computation
3. Outputs processed signal to soma for integration

The soma integrates outputs from all dendrites to produce the final neuron output.

Excitation-Inhibition Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Network architectures that maintain separate excitatory and inhibitory populations following Dale's principle:

- **Excitatory neurons**: Can only make excitatory connections
- **Inhibitory neurons**: Can only make inhibitory connections  
- **Realistic connectivity**: Based on biological synaptic densities and patterns

Framework Components
-------------------

The framework consists of several key modules:

Core Modules
~~~~~~~~~~~

- **Networks** (``dendritic_modeling.networks``): Core neural network architectures
- **Models** (``dendritic_modeling.models``): High-level model wrappers with task-specific heads
- **Training** (``dendritic_modeling.training``): Training utilities and learning strategies
- **Analysis** (``dendritic_modeling.analysis``): Tools for network analysis and interpretation
- **Configuration** (``dendritic_modeling.config``): Type-safe configuration system
- **Datasets** (``dendritic_modeling.datasets``): Synthetic and real datasets

Supporting Modules
~~~~~~~~~~~~~~~~

- **Utils** (``dendritic_modeling.utils``): Utility functions and helpers
- **Plotting** (``dendritic_modeling.plotting``): Visualization tools and plotting utilities
- **Managers** (``dendritic_modeling.managers``): High-level experiment management

Quick Reference
--------------

Common Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic network creation
   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   from dendritic_modeling.config import TransferFunctionParams
   
   params = TransferFunctionParams().asdict()
   network = ExcitationInhibitionNetwork(
       input_dim=784,
       transfer_params=params,
       excitatory_layer_sizes=[100, 10],
       inhibitory_layer_sizes=[50, 0]
   )

.. code-block:: python

   # Model creation and training
   from dendritic_modeling.models import Classifier
   from dendritic_modeling.training.utils import train_model
   
   model = Classifier(core_network=network)
   train_model(model, train_dataset, val_dataset)

.. code-block:: python

   # Dataset loading
   from dendritic_modeling.datasets import get_unified_datasets
   
   train_ds, val_ds, test_ds = get_unified_datasets("mnist")

.. code-block:: python

   # Analysis and visualization
   from dendritic_modeling.analysis import InformationAnalyzer
   from dendritic_modeling.plotting import plot_network_activity
   
   analyzer = InformationAnalyzer(model)
   info_metrics = analyzer.compute_information_metrics(dataset)
   plot_network_activity(model, sample_batch)

Configuration Examples
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   # Basic configuration
   model:
     task: classification
     network:
       type: EINet
       parameters:
         excitatory_layer_sizes: [100, 10]
         inhibitory_layer_sizes: [50, 0]
         
   train:
     learning_strategy: mle
     lr: 0.001
     epochs: 50
     batch_size: 64

.. code-block:: yaml

   # Advanced configuration with branch-wise training
   model:
     network:
       type: EINet
       parameters:
         excitatory_layer_sizes: [200, 100, 10]
         inhibitory_layer_sizes: [100, 50, 0]
         excitatory_branch_factors: [10, 5, 1]
         
   train:
     learning_strategy: freeze_branches_reversed
     epochs_per_layer: 20
     final_tune_epochs: 10

Next Steps
---------

1. **New Users**: Start with :doc:`getting_started` for installation and basic usage
2. **Architecture Focus**: Go to :doc:`architecture_and_models` to understand network designs
3. **Training Focus**: See :doc:`training_and_configuration` for advanced training strategies
4. **Analysis Focus**: Explore :doc:`analysis_and_visualization` for research tools

For specific API details, see the :doc:`../api/index` section.
