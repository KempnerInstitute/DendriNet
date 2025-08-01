Configuration Guide
===================

This guide covers all configuration options for the Dendritic Modeling Framework. All configurations use YAML files passed to training scripts.

Training Features
-----------------

- **Regularization**: L1/L2 regularization with global or parameter group specific weights
- **Pruning**: Magnitude-based weight pruning with selective application
- **Optimizers**: Adam, SGD, RMSprop, AdamW with parameter group support
- **Learning Rates**: Different learning rates for different network components

Model Architecture
------------------

The framework uses a three-part model structure:

.. code-block:: yaml

   model:
     task: classification  # or "regression"
     
     encoder_network:
       type: Identity      # Options: Identity, MLP, CNN, AlexNet
       parameters:
         # Parameters specific to encoder type
         
     core_network:
       type: EINet         # Options: EINet, DendriNet, MLP, MatchedParamMLP
       parameters:
         # Parameters specific to core network type
         
     decoder_network:
       type: Identity      # Options: Identity, MLP
       parameters:
         # Parameters specific to decoder type

Encoder Networks
^^^^^^^^^^^^^^^^

**Identity** (default)
   Passes input unchanged to core network.

.. code-block:: yaml

      encoder_network:
        type: Identity
        parameters: {}

**MLP**
   Multi-layer perceptron encoder.

   .. code-block:: yaml

  encoder_network:
        type: MLP
    parameters:
      input_dim: 784
          hidden_dims: [512, 256]
          output_dim: 128
          activation: relu

**CNN**
   Convolutional encoder for image data.

   .. code-block:: yaml

      encoder_network:
        type: CNN
    parameters:
          input_channels: 1
          output_dim: 128

Core Networks
^^^^^^^^^^^^^

**EINet (Excitation-Inhibition Network)**

The main dendritic architecture with separate excitatory and inhibitory pathways:

.. code-block:: yaml

  core_network:
    type: EINet
    parameters:
       # Layer architecture
       excitatory_layer_sizes: [128, 64, 10]    # Neurons per layer
       inhibitory_layer_sizes: [64, 32, 0]      # Neurons per layer
      
      # Branching structure
       excitatory_branch_factors: [2, 2]        # Branching per layer
       inhibitory_branch_factors: []            # Usually empty
      
       # Synaptic connectivity
      ee_synapses_per_branch_per_layer: [100, 50]  # E→E connections
      ei_synapses_per_branch_per_layer: [50, 25]   # E→I connections  
      ie_synapses_per_branch_per_layer: [25, 10]   # I→E connections
       ii_synapses_per_branch_per_layer: [0, 0]     # I→I (usually 0)
       
       # Core features
       use_shunting: true                       # Shunting inhibition
       somatic_synapses: true                   # Direct soma connections
       epsilon: 1e-6                            # Numerical stability
       
       # TopK selection
       topk_type: standard                      # Synapse selection method
       topk_init_method: xavier_normal          # Weight initialization
       
       # Activation functions
       reactivate: true                         # Enable branch activation
       reactivation_type: param_tanh            # Activation function
       reactivation_init_m: 1.5                 # Initial slope
       reactivation_init_b: 0.5                 # Initial bias
       
       # Gradient scaling
       reactivation_strategy: inverse           # Gradient strategy
       topk_strategy: conductance_dynamic       # TopK gradient scaling
       blocklinear_strategy: conductance_dynamic # Output gradient scaling
       
       # Input processing
       input_mode: 1                            # Input routing mode
       transfer_params: {}                      # Transfer function params

**Key EINet Parameters:**

- **Layer Sizes**: Define the number of neurons in each layer for excitatory and inhibitory pathways
- **Branch Factors**: Determine dendritic branching structure (e.g., [2, 2] means each neuron branches into 2 at each level)
- **Synaptic Connectivity**: Number of connections per branch between different neuron types
- **TopK Types**:
  - ``standard``: Select top-k synapses by weight magnitude
  - ``stochastic``: Probabilistic selection with temperature
  - ``variance``: Selection based on input variance
  - ``deepst``: Dynamic sparse training with rewiring
- **Activation Types**:
  - ``param_tanh``: Parametric tanh with learnable m and b
  - ``relu``, ``sigmoid``, ``tanh``: Standard activations
  - ``param_linear_sigmoid``, ``param_linear_tanh``: Interpolated activations

**DendriNet**

Pure dendritic network without explicit E/I separation:

.. code-block:: yaml

   core_network:
     type: DendriNet
     parameters:
       n_soma: 10                               # Output neurons
       branch_factors: [2, 2]                   # Branching structure
       excitatory_input_dim: 784                # Input dimension
       excitatory_synapses_per_branch: 100      # Synapses per branch
       use_shunting: true                       # Enable shunting
       reactivate: true                         # Branch activation

**MLP**

Standard multi-layer perceptron:

.. code-block:: yaml

  core_network:
    type: MLP
    parameters:
      input_dim: 784
      hidden_dims: [512, 256, 128]
      output_dim: 10
       activation: relu

**MatchedParamMLP**

MLP with parameter count matched to EINet for fair comparison:

.. code-block:: yaml

  core_network:
    type: MatchedParamMLP
    parameters:
      # Uses same parameters as EINet to match parameter count
      excitatory_layer_sizes: [128, 64]
      inhibitory_layer_sizes: [64, 32]
       # Other EINet parameters...

Decoder Networks
^^^^^^^^^^^^^^^^

**Identity** (default)
   Uses core network output directly.

   .. code-block:: yaml

      decoder_network:
        type: Identity
        parameters: {}

**MLP**
   Additional processing of core network output.

   .. code-block:: yaml

      decoder_network:
        type: MLP
        parameters:
          input_dim: 10
          hidden_dims: [32]
          output_dim: 10
          activation: relu

Training Configuration
----------------------

**Training Workflow:**

The training process follows this sequence:

1. **Training**: Model trains for specified epochs
2. **Original Analysis**: All enabled analyses run on the trained model
3. **Pruning** (if enabled): Weights below threshold are pruned, performance is re-evaluated
4. **Pruned Model Analysis** (if ``redo_analysis_after_pruning: true``): All analyses re-run on pruned model

**Learning Strategies:**

- ``standard``: Standard training with optional regularization, pruning, and post-pruning analysis  
- ``two_step``: Train branches then soma connections
- ``voltage_stabilization``: Stabilize dendritic voltages during training
- ``freeze_layers``: Layer-wise training from output to input
- ``freeze_branches``: Branch-wise training
- ``local_ca``: Local credit assignment

**Configuration Example:**

.. code-block:: yaml

   train:
     main:
       learning_strategy: standard
       
       # Parameter groups for differential learning rates
       param_groups:
         lr: 0.001                   # Default learning rate
         split_params: true          # Enable parameter grouping
         topk_lr: 0.001             # TopK layer learning rate
         blocklinear_lr: 0.0001     # BlockLinear layer learning rate
         reactivation_lr: 0.0001    # Reactivation layer learning rate
         decoder_lr: 0.0001         # Decoder network learning rate
       
       # Regularization configuration
       regularization:
         l1_weight: 0.001            # Global L1 penalty weight (used when split_params=false)
         l2_weight: 0.01             # Global L2 penalty weight (used when split_params=false)
         selective:
           layers: []                # Apply to specific layers (empty = all)
           branches: []              # Apply to specific branches (empty = all)
           weight_types: ["all"]     # Options: "all", "recurrent", "input", "output", "linear"
         
         # Parameter group specific regularization
         split_params: true          # Split parameters into groups for differential regularization
         param_group_weights:
           topk:                     # TopK layer parameters
             l1_weight: 0.002        # Stronger L1 for sparsity
             l2_weight: 0.005
           blocklinear:              # BlockLinear parameters
             l1_weight: 0.0          # No L1 to preserve structure
             l2_weight: 0.001        # Light L2 only
           reactivation:             # Reactivation parameters
             l1_weight: 0.0005       # Light L1
             l2_weight: 0.0          # No L2
           decoder:                  # Decoder network parameters
             l1_weight: 0.001
             l2_weight: 0.005
       
       # Weight pruning
       pruning:
         enabled: false              # Enable weight pruning
         threshold: 0.001            # Magnitude threshold for pruning
         dynamic_pruning: false      # Prune during training (false = prune after training)
         pruning_frequency: 10       # Epochs between pruning (if dynamic)
         re_evaluate_after: true     # Re-evaluate model performance after pruning
         selective:
           layers: []                # Prune specific layers (empty = all)
           branches: []              # Prune specific branches (empty = all)
           weight_types: ["all"]     # Options: "all", "excitatory", "inhibitory", "recurrent", "input", "output", "linear"
       
       # Reporting and post-analysis options
       reporting:
         report_non_pruned: true          # Print detailed pruning report
         save_pruning_stats: true         # Save pruning statistics to JSON
         detailed_branch_report: true     # Generate synaptic analysis plots
         redo_analysis_after_pruning: true  # Re-run all enabled analyses on pruned model

**Pruning Outputs:**

When pruning is enabled, the following files are generated:

- ``pruning_stats.json``: Detailed pruning statistics by layer, branch, and weight type
- ``pruning_performance_comparison.png``: Performance before and after pruning
- ``pruned_analysis/``: Complete re-analysis results on pruned model (if enabled)

If ``detailed_branch_report: true``, additional plots are generated:

- ``active_synapses_per_branch.png``: Number of active synapses per branch per layer
- ``active_synapses_summary_by_layer.png``: Summary statistics of active synapses by layer

**Regularization**

``split_params: false`` - Uses global ``l1_weight`` and ``l2_weight`` for all parameters.

``split_params: true`` - Uses ``param_group_weights`` for component-specific regularization:

- ``topk``: TopK layer parameters
- ``blocklinear``: BlockLinear layer parameters  
- ``reactivation``: Reactivation layer parameters
- ``decoder``: Decoder network parameters

If no ``param_group_weights`` specified when ``split_params=true``, no regularization is applied.

**Optimizer Configuration**

.. code-block:: yaml

   train:
     main:
       learning_strategy: standard
       
       # Internal optimizer creation (respects param_groups above)
       optimizer:
         name: adam                  # Options: adam, sgd, rmsprop, adamw
         lr: 0.001                  # Base learning rate
         weight_decay: 0.01         # Global L2 regularization
         momentum: 0.9              # For SGD
         betas: [0.9, 0.999]        # For Adam/AdamW
         eps: 1e-8                  # For Adam/AdamW

Task Configuration
------------------

.. code-block:: yaml

task:
     dataset: mnist              # Dataset name
  parameters:
       # Dataset-specific parameters
       train_size: 50000         # Training samples
       valid_size: 10000         # Validation samples
       test_size: 10000          # Test samples
       normalize: true           # Apply normalization
       flatten: true             # Flatten images

**Supported Datasets:**

- ``mnist``: Handwritten digits (28x28)
- ``cifar10``: Natural images (32x32x3)
- ``cifar100``: Natural images with 100 classes

Analysis Configuration
----------------------

Optional analysis during or after training. Analysis runs twice if pruning is enabled: once on the original trained model, and again on the pruned model (if ``redo_analysis_after_pruning: true``):

.. code-block:: yaml

analysis:
     # Performance metrics
     performance_analysis:
    enabled: true
       training: false           # Run during training
    params:
      accuracy: true
      categorical_loglikelihood: true
         mse: false              # For regression
     
     # Information theory (optional)
     information_analysis:
       enabled: false            # Expensive computation
    training: false
    params:
         method: kraskov         # MI estimation method
         n_neighbors: 10
         max_samples: 5000
     
     # Synaptic turnover analysis
     synapse_turnover_analysis:
       enabled: true
       training: true
       snapshot_interval: 5        # Frequency of snapshots
       synaptic_analysis: true     # Enable synaptic connectivity analysis
       save_synaptic_report: true  # Save synaptic analysis to file

Output Configuration
--------------------

.. code-block:: yaml

outputs:
     output_dir: results/        # Base output directory
     checkpoint_dir: checkpoints/
     save_frequency: 10          # Save every N epochs
     
   visualization:
     enabled: true
     plot_frequency: 10
     plots:
       - loss_curves
       - accuracy_curves
       - weight_distributions

Weights & Biases Configuration
------------------------------

.. code-block:: yaml

wandb:
     use_wandb: true
     project: dendritic-modeling
     entity: my-team
     tags: [einet, mnist]
     log_frequency: 100          # Log every N batches

Complete Example Configuration
------------------------------

**Basic EINet on MNIST:**

.. code-block:: yaml

model:
  task: classification
     
  encoder_network:
    type: Identity
       parameters: {}
       
  core_network:
    type: EINet
    parameters:
         excitatory_layer_sizes: [512, 256, 10]
         inhibitory_layer_sizes: [128, 64, 0]
         excitatory_branch_factors: [2, 2]
         inhibitory_branch_factors: []
         ee_synapses_per_branch_per_layer: [100, 50]
         ei_synapses_per_branch_per_layer: [50, 25]
         ie_synapses_per_branch_per_layer: [25, 10]
         ii_synapses_per_branch_per_layer: [0, 0]
      use_shunting: true
         reactivate: true
      reactivation_type: param_tanh
         topk_type: standard
         
  decoder_network:
    type: Identity
       parameters: {}

train:
  learning_strategy: standard
     run_name: einet_mnist_basic
     seed: 42
     
    trainer:
      epochs: 100
      batch_size: 64
       optimizer: adam
       learning_rate: 0.001
       weight_decay: 0.0001

task:
  dataset: mnist
  parameters:
       train_size: 50000
       valid_size: 10000
       normalize: true
    flatten: true

outputs:
     output_dir: results/einet_mnist/
     
   wandb:
     use_wandb: false

**FSDP Configuration for Large Models:**

.. code-block:: yaml

   # Add to train section for distributed training
train:
     learning_strategy: fsdp
     
     fsdp:
       sharding_strategy: FULL_SHARD
       cpu_offload: true
       mixed_precision: true
       
  trainer:
       gradient_accumulation_steps: 4
       # Other trainer settings...

**Parameter Sweep Configuration:**

.. code-block:: yaml

   # Sweep-specific configuration
sweep_config:
     model.core_network.parameters.ee_synapses_per_branch_per_layer: 
       - [50, 25]
       - [100, 50]
       - [200, 100]
     model.core_network.parameters.use_shunting: 
       - true
       - false
   
   experiment_settings:
     seeds_per_condition: 3
     base_seed: 42
     
   # Rest of configuration... 