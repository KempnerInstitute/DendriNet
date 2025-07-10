Dendritic Modeling Framework
============================

.. image:: _static/dendritic_diagram.png
   :alt: Dendritic neuron architecture
   :align: center
   :width: 600px

A PyTorch-based framework for biologically-inspired neural networks with dendritic computation.

Key Features
------------

* **Dendritic Computation**: Implements shunting inhibition and multi-branch dendritic processing
* **E/I Networks**: Separate excitatory and inhibitory pathways with biologically-plausible connectivity
* **Sparse Connectivity**: TopK and DEEPST algorithms for dynamic sparse connections
* **Vision Models**: Pre-built units for Gabor filters, center-surround, and motion detection
* **Analysis Tools**: Comprehensive suite for ablation, information theory, and synaptic dynamics

Quick Start
-----------

.. code-block:: python

   from dendritic_modeling.networks import ExcitationInhibitionNetwork
   from dendritic_modeling.models import Classifier
   
   # Create E/I network with dendritic branches
   network = ExcitationInhibitionNetwork(
       input_dim=784,
       excitatory_layer_sizes=[100],
       inhibitory_layer_sizes=[50],
       excitatory_branch_factors=[2, 2],
       use_shunting=True
   )
   
   # Wrap in classifier
   model = Classifier(
       encoder_network=None,
       core_network=network,
       decoder_network=None
   )

Documentation
-------------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   tutorials
   api_reference
   theory
   examples
   contributing

Core Components
---------------

**Networks**
   - :class:`~dendritic_modeling.networks.ExcitationInhibitionNetwork`: Main E/I network architecture
   - :class:`~dendritic_modeling.networks.DendriticBranchLayer`: Individual dendritic branch processing
   - :class:`~dendritic_modeling.networks.DendriNet`: Multi-layer dendritic network
   - :class:`~dendritic_modeling.networks.layers.DendriteConv2d`: Convolutional dendritic layer

**Models**
   - :class:`~dendritic_modeling.models.Classifier`: Classification wrapper
   - :class:`~dendritic_modeling.models.Regressor`: Regression wrapper

**Training**
   - :class:`~dendritic_modeling.training.Trainer`: Standard training
   - :class:`~dendritic_modeling.training.LayerWiseTrainer`: Layer-by-layer training
   - :class:`~dendritic_modeling.training.LocalCreditAssignment`: Local learning rules

**Analysis**
   - :class:`~dendritic_modeling.analysis.AblationAnalyzer`: Systematic ablation studies
   - :class:`~dendritic_modeling.analysis.InformationAnalyzer`: Information-theoretic analysis
   - :class:`~dendritic_modeling.analysis.SynapticTurnoverAnalyzer`: Synaptic dynamics tracking

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
