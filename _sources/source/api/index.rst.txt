API Reference
=============

This section provides detailed API documentation for all modules in the Dendritic Modeling Framework.

.. toctree::
   :maxdepth: 2
   :caption: Core Modules

   networks/index
   models/index
   config/index

.. toctree::
   :maxdepth: 2
   :caption: Training and Optimization

   training/index
   optimization/index

.. toctree::
   :maxdepth: 2
   :caption: Analysis and Visualization

   analysis/index
   plotting/index

.. toctree::
   :maxdepth: 2
   :caption: Data and Utilities

   datasets/index
   utils/index

.. toctree::
   :maxdepth: 2
   :caption: Managers and Workflows

   managers/index

Overview
--------

The Dendritic Modeling Framework is organized into several key modules:

Core Modules
~~~~~~~~~~~

**Networks** (``dendritic_modeling.networks``)
   Core neural network architectures implementing dendritic computation, including excitation-inhibition networks, MLPs, CNNs, and Vision Transformers.

**Models** (``dendritic_modeling.models``)
   High-level model wrappers that combine networks with task-specific components for classification, regression, and other applications.

**Configuration** (``dendritic_modeling.config``)
   Type-safe configuration system for experiments, models, and training parameters with YAML support.

Training and Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

**Training** (``dendritic_modeling.training``)
   Comprehensive training utilities including multiple learning strategies, distributed training, and experiment management.

**Optimization** (``dendritic_modeling.optimization``)
   Optimization algorithms and utilities specifically designed for dendritic networks.

Analysis and Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~

**Analysis** (``dendritic_modeling.analysis``)
   Tools for analyzing network behavior, information flow, biological plausibility, and performance evaluation.

**Plotting** (``dendritic_modeling.plotting``)
   Visualization tools for networks, training curves, analysis results, and publication-quality figures.

Data and Utilities
~~~~~~~~~~~~~~~~~

**Datasets** (``dendritic_modeling.datasets``)
   Dataset utilities for loading standard datasets and creating synthetic data for dendritic modeling research.

**Utils** (``dendritic_modeling.utils``)
   Utility functions for logging, file operations, mathematical operations, and other common tasks.

Managers and Workflows
~~~~~~~~~~~~~~~~~~~~~

**Managers** (``dendritic_modeling.managers``)
   High-level managers for experiments, analysis workflows, and automated pipelines.

Quick Reference
--------------

Common Import Patterns
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Core components
   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   from dendritic_modeling.models import Classifier
   from dendritic_modeling.config import TransferFunctionParams
   
   # Training
   from dendritic_modeling.training import train_model
   from dendritic_modeling.training.utils import evaluate_model
   
   # Analysis
   from dendritic_modeling.analysis import InformationAnalyzer
   from dendritic_modeling.plotting import NetworkVisualizer
   
   # Data
   from dendritic_modeling.datasets import get_unified_datasets

Key Classes and Functions
~~~~~~~~~~~~~~~~~~~~~~~~

**Network Creation**

.. code-block:: python

   # Create E/I network
   ei_network = ExcitationInhibitionNetwork(
       input_dim=784,
       transfer_params=TransferFunctionParams().asdict(),
       excitatory_layer_sizes=[100, 10],
       inhibitory_layer_sizes=[50, 0]
   )

**Model Creation**

.. code-block:: python

   # Create classifier
   model = Classifier(
       core_network=ei_network,
       encoder_network=None,
       decoder_network=None
   )

**Training**

.. code-block:: python

   # Train model
   train_model(
       model=model,
       train_dataset=train_dataset,
       val_dataset=val_dataset,
       epochs=50
   )

**Analysis**

.. code-block:: python

   # Analyze information flow
   analyzer = InformationAnalyzer(model)
   info_metrics = analyzer.compute_information_metrics(test_dataset)

**Visualization**

.. code-block:: python

   # Visualize network
   visualizer = NetworkVisualizer(model)
   visualizer.plot_network_architecture()

Module Dependencies
------------------

The framework modules have the following dependency structure:

.. code-block:: text

   dendritic_modeling/
   ├── config/          # No dependencies (base configuration)
   ├── utils/           # Depends on: config
   ├── networks/        # Depends on: config, utils
   ├── models/          # Depends on: networks, config, utils
   ├── datasets/        # Depends on: utils
   ├── training/        # Depends on: models, datasets, utils, config
   ├── optimization/    # Depends on: training, models, utils
   ├── analysis/        # Depends on: models, training, utils
   ├── plotting/        # Depends on: analysis, models, utils
   └── managers/        # Depends on: all other modules

Version Information
------------------

.. code-block:: python

   import dendritic_modeling
   print(dendritic_modeling.__version__)

For detailed documentation of each module, see the corresponding sections in the API reference.
