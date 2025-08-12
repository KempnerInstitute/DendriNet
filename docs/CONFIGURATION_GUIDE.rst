Configuration Guide
===================

Quick Start
-----------

All experiments use YAML configuration files. Start with these templates:

- ``configs/config_exp.yaml`` - Standard training
- ``configs/config_exp_fsdp.yaml`` - FSDP training
- ``configs/unified_all_sweeps_example.yaml`` - Parameter sweeps

Basic Structure
---------------

.. code-block:: yaml

   model:
     task: classification
     encoder_network:
       type: Identity
     core_network:
       type: EINet
     decoder_network:
       type: Identity

   train:
     main:
       epochs: 100
       batch_size: 64
       learning_rate: 0.001

   task:
     name: mnist
     data_dir: data/

Core Networks
-------------

EINet (Excitation-Inhibition Network)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Biologically-inspired network with separate excitatory and inhibitory pathways.

.. code-block:: yaml

   core_network:
     type: EINet
     parameters:
       layer_sizes: [128, 64, 32]
       ee_synapses_per_branch_per_layer: [[2], [2], [2]]
       ie_synapses_per_branch_per_layer: [[1], [1], [1]]
       use_shunting: true

Key parameters:
- ``layer_sizes``: Neurons per layer
- ``ee_synapses_per_branch_per_layer``: Excitatory connections
- ``ie_synapses_per_branch_per_layer``: Inhibitory connections
- ``use_shunting``: Enable shunting inhibition

DendriNet
~~~~~~~~~

Pure dendritic network with hierarchical branching.

.. code-block:: yaml

   core_network:
     type: DendriNet
     parameters:
       layer_sizes: [128, 64, 32]
       branch_factors: [[2, 2], [2, 2], [2, 2]]
       topk_type: standard

MLP
~~~

Standard multi-layer perceptron.

.. code-block:: yaml

   core_network:
     type: MLP
     parameters:
       hidden_sizes: [128, 64, 32]
       activation: ReLU

Training Options
----------------

Standard Training
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   train:
     main:
       epochs: 100
       batch_size: 64
       learning_rate: 0.001
       optimizer: Adam

FSDP Training
~~~~~~~~~~~~~

.. code-block:: yaml

   train:
     main:
       epochs: 100
       batch_size: 32  # Smaller for FSDP
       learning_rate: 0.001

   fsdp:
     enabled: true
     sharding_strategy: FULL_SHARD

Learning Strategies
~~~~~~~~~~~~~~~~~~~

Available strategies:
- ``standard`` - Regular backpropagation
- ``freeze_layers`` - Freeze specific layers
- ``freeze_branches`` - Freeze dendritic branches
- ``two_step`` - Two-phase training
- ``voltage_stabilization`` - Stabilize dendritic voltages
- ``local_ca`` - Local credit assignment

.. code-block:: yaml

   train:
     main:
       learning_strategy: standard

Datasets
--------

Supported datasets:
- ``mnist`` - MNIST handwritten digits
- ``cifar10`` - CIFAR-10 images
- ``imagenet`` - ImageNet (requires download)
- ``mnist_modulo10`` - MNIST with modulo arithmetic

.. code-block:: yaml

   task:
     name: mnist
     data_dir: data/
     batch_size: 64

Analysis Options
----------------

Enable analysis during training:

.. code-block:: yaml

   analysis:
     performance_analysis:
       enabled: true
     information_analysis:
       enabled: true
       method: kraskov
     ablation_analysis:
       enabled: true
       layer_ablation: true

WandB Integration
-----------------

Track experiments with Weights & Biases:

.. code-block:: yaml

   wandb:
     enabled: true
     project: dendritic-modeling
     name: my_experiment

Common Configurations
--------------------

Small Model (Development)
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   model:
     core_network:
       type: EINet
       parameters:
         layer_sizes: [64, 32, 10]

   train:
     main:
       epochs: 50
       batch_size: 128

Large Model (Production)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   model:
     core_network:
       type: EINet
       parameters:
         layer_sizes: [512, 256, 128, 64]

   train:
     main:
       epochs: 200
       batch_size: 32

   fsdp:
     enabled: true

Parameter Sweep
~~~~~~~~~~~~~~~

.. code-block:: yaml

   sweep_config:
     model.core_network.parameters.layer_sizes: [[64,32,10], [128,64,32,10]]
     train.main.learning_rate: [0.001, 0.01, 0.1]

   filter_config:
     mode: "ei-sum"
     value: 4

Information Analysis
--------------------

The framework provides flexible information-theoretic analysis with configurable computation levels.

Computation Levels
~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   analysis:
     information_analysis:
       params:
         computation_level: "branch"  # Options: "branch", "layer", "network"
         branch_aggregation: "mean"   # Options: "mean", "multivariate", "sample"

- ``branch``: Compute information for individual branches (most granular)
- ``layer``: Aggregate branches within each layer then compute information
- ``network``: Aggregate all branches across the network then compute information

Branch Aggregation
~~~~~~~~~~~~~~~~~~

When using ``layer`` or ``network`` computation levels:

- ``mean``: Average branch activations to create scalar values
- ``multivariate``: Use raw multi-dimensional branch data for joint information computation
- ``sample``: Treat each branch as a separate observation by flattening branch dimension into sample dimension

Enhanced EI Network Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides enhanced analysis specifically designed for excitation-inhibition networks.

.. code-block:: yaml

   analysis:
     information_analysis:
       params:
         # Neuron selection
         neuron_selection: "all"          # Options: "all", "excitatory", "inhibitory"
         
         # Enhanced analysis modes
         per_neuron_analysis: false       # Compute MI for each neuron individually
         ei_specific_analysis: false      # Analyze E and I networks separately

Neuron Selection Options:

- ``all``: Use both excitatory and inhibitory neurons (default)
- ``excitatory``: Analyze only excitatory neurons and their synapses
- ``inhibitory``: Analyze only inhibitory neurons and their synapses

Per-Neuron Analysis:

When enabled, computes mutual information for each individual neuron separately, providing detailed per-neuron MI statistics and averaged results across all neurons.

EI-Specific Analysis:

When enabled, performs three separate analyses:
- Excitatory network only (inhibitory inputs set to zero)
- Inhibitory network only (excitatory inputs set to zero)  
- Combined E+I network (standard analysis)

Training Optimization
---------------------

Configure performance and debugging options for training.

Trainer Structure
~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   train:
     encoder_net_trainer:
       loss_function: mse
       epochs: 50
       batch_size: 128
     main_trainer:
       loss_function: ce
       epochs: 100
       batch_size: 64
       enable_nan_checking: false
       enable_adaptive_clipping: false

Performance Options
~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   train:
     main_trainer:
       enable_nan_checking: false      # Enable NaN detection (expensive)
       enable_adaptive_clipping: false # Enable adaptive gradient clipping (expensive)
       use_amp: false                  # Enable automatic mixed precision

Encoder Network Caching
~~~~~~~~~~~~~~~~~~~~~~~~

Hash-based caching automatically reuses trained encoder networks with identical configurations.

.. code-block:: yaml

   train:
     encoder:
       load_save_root: ./.trained_encoder_networks/

Models are saved to ``{load_save_root}/{config_hash}/best_model.pt`` and automatically loaded when configurations match.

Troubleshooting
---------------

Common Issues:
- Model too large: Use FSDP or reduce layer sizes
- Training slow: Increase batch size or use multiple GPUs
- Low accuracy: Adjust learning rate or model architecture
- Memory errors: Reduce batch size or enable gradient checkpointing

For detailed troubleshooting, see ``FSDP_TROUBLESHOOTING.rst``.
