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
       params:
         method: "kraskov"
     ablation_analysis:
       enabled: true
       params:
         layer_ablation: true
         ablation_methods: ["lesion", "shuffle", "mean_clamp"]
         shuffle_seed: 0
         clamp_statistic: "mean"
         compute_pred_label_mi: true

Ablation / Intervention Analysis
--------------------------------

Our ablation analysis supports both **structural lesions** (removing connections) and
**signal-level interventions** (removing label-aligned information while preserving the
operating point / baseline distribution).

``lesion``:
   Structural ablation by disabling the relevant connection weights. This can change the
   layer's gain/baseline and may cause saturation/vanishing.

``shuffle``:
   Signal intervention that permutes the targeted signal across samples in a batch. This
   preserves the marginal distribution of that signal but destroys sample/label alignment.

``mean_clamp``:
   Signal intervention that replaces the targeted signal with its batch mean. This preserves
   a stable operating point but removes trial-to-trial variability.

For an information-theoretic complement aligned to these interventions, enable
``compute_ablation_aligned_mi`` in the information analyzer to compute conditional-MI terms
such as ``I(E;C|I,Vb)``, ``I(I;C|E,Vb)``, and synapse-vs-upstream decompositions like
``I(Vout;C|Vb)`` (class information added by synapses beyond upstream input) and
``I(Vb;C|Vout)`` (upstream class information not reflected in the layer output).

.. code-block:: yaml

   analysis:
     ablation_analysis:
       enabled: true
       training: false
       params:
         ablation_methods: ["lesion", "shuffle", "mean_clamp"]
         shuffle_seed: 0
         clamp_statistic: "mean"
         compute_pred_label_mi: true  # Adds I(C; Ĉ) in bits, where Ĉ=argmax predicted class
         layer_ablation: true
         compartment_ablation: false
         ablate_all_synapses: true
         ablate_excitation: true
         ablate_inhibition: true
         ablate_upstream: true

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
         # Enhanced analysis modes
         per_neuron_analysis: false       # Compute MI for each neuron individually
         per_einet_analysis: false        # Analyze E and I networks separately
         per_layer_analysis: false        # Compute enhanced analysis per layer

Per-Neuron Analysis:

When enabled, computes mutual information for each individual neuron separately, providing detailed per-neuron MI statistics and averaged results across all neurons.

Per-EI-Network Analysis:

When enabled, performs three separate analyses:
- Excitatory network only (inhibitory inputs set to zero)
- Inhibitory network only (excitatory inputs set to zero)
- Combined E+I network (standard analysis)

Per-Layer Analysis:

When enabled, computes enhanced analysis (per-neuron and per-EI-network) for each layer separately in multilayer networks, providing layer-specific insights into information processing.

Branch Input Analysis
~~~~~~~~~~~~~~~~~~~~~

The framework supports analysis of upstream dendritic inputs (branch inputs) in hierarchical dendritic networks.

Branch-input (Vb) metrics are computed automatically when branch inputs exist (no configuration flag needed). This adds the following metrics:

Basic MI:
   ``I(Vb;C)`` - Mutual information between branch input and class labels

Pairwise MI:
   - ``I(E;Vb)`` - Mutual information between excitation and branch input
   - ``I(I;Vb)`` - Mutual information between inhibition and branch input
   - ``I(Vb;Vout)`` - Mutual information between branch input and output

Conditional MI:
   - ``I(E;Vb|C)`` - Conditional mutual information between E and Vb given class
   - ``I(I;Vb|C)`` - Conditional mutual information between I and Vb given class
   - ``I(Vb;Vout|C)`` - Conditional mutual information between Vb and Vout given class

Note: For single-layer networks without hierarchical dendritic structure, branch input is zero and these metrics provide baseline comparisons.

Linear Discriminant Analysis Weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework can compute alternative aggregation weights using Linear Discriminant Analysis (LDA) instead of network-learned weights.

.. code-block:: yaml

   analysis:
     information_analysis:
       params:
         compute_lda_weights: false    # Compute LDA-based aggregations

When enabled, the system fits separate LDA models on raw synaptic inputs to find optimal linear combinations that maximize class separation. This provides three LDA-weighted signals:

E_lin:
   Optimal linear combination of excitatory synaptic inputs using discriminant weights

I_lin:
   Optimal linear combination of inhibitory synaptic inputs using discriminant weights

Vb_lin:
   Optimal linear combination of upstream branch inputs using discriminant weights

The analysis then computes mutual information for both network-learned aggregations and LDA-optimized aggregations:

Network weights (w_opt):
   - ``I(E;C)``, ``I(I;C)``, ``I(Vb;C)``
   - Standard pairwise and conditional MI

LDA weights (w_lda):
   - ``I(E_lin;C)``, ``I(I_lin;C)``, ``I(Vb_lin;C)``
   - Pairwise: ``I(E_lin;I_lin)``, ``I(E_lin;Vb_lin)``, etc.
   - Conditional: ``I(E_lin;I_lin|C)``, etc.

This comparison reveals whether the network has learned synaptic weights that approach the discriminant-optimal configuration or whether additional structure beyond class discrimination is encoded.

Complete Information Analysis Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   analysis:
     information_analysis:
       enabled: true
       training: false
       params:
         # Estimation method
         method: "kraskov"              # Options: "kraskov", "sklearn", "copula"
         max_samples: 1000               # Limit samples for computational efficiency
         n_neighbors: 10                 # K-NN parameter for Kraskov estimator

         # Which metrics to compute
         compute_basic_mi: true          # I(X;C) for all variables
         compute_pairwise_mi: true       # I(X;Y) between variables
         compute_conditional_mi: true    # I(X;Y|C) conditioned on class

         # Computation granularity
         computation_level: "single_branch"  # Options: "single_branch", "layer_branch", "all_branch"
         branch_aggregation: "mean"          # Options: "mean", "multivariate", "sample"

         # Enhanced analysis modes
         per_neuron_analysis: false      # Per-neuron MI computation
         per_einet_analysis: false       # Separate E and I network analysis
         per_layer_analysis: false       # Per-layer enhanced analysis

         # LDA analysis
         compute_lda_weights: false      # Compute LDA-based aggregations
         lda_n_shuffles: 0              # Random-weight baseline shuffles (0 = disabled)

         # Ablation-aligned information extras
         compute_ablation_aligned_mi: true   # Adds conditional-MI proxies aligned with E/I/Vb ablations

         # Single-branch -> layer-total proxy aggregations (avoids high-dim multivariate MI)
         compute_layer_total_proxies: true   # Adds *_sum/*_union/*_topK_sum per-layer proxy metrics
         layer_total_topk: 10                # K used for *_topK_sum metrics

         # Plotting note: figures include both *_sum proxies and a per-branch mean view
         # ("ablation_proxy_branch_mean") so you can see branch-count scaling effects.

         # Additional options
         include_vinf: true              # Include soma activation
         normalize_mi: false             # Normalize by entropy

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
