Configuration Guide
===================

All configurations use YAML files parsed by OmegaConf. Key sections:

Model Configuration
------------------

1. **Network Configuration**:
   - ``network.type``: String - 'EINet' or 'DendriNet' (default: 'EINet').
   - ``network.parameters.input_dim``: Int - Input dimension (e.g., 784 for MNIST).
   - ``network.parameters.output_dim``: Int - Output dimension (e.g., 10 for classification).
   - ``network.parameters.num_layers``: Int - Number of layers (default: 3).
   - ``network.parameters.excitatory_layer_sizes``: List[Int] - Sizes for excitatory paths (e.g., [512, 256, 10]).
   - ``network.parameters.inhibitory_layer_sizes``: List[Int] - Sizes for inhibitory paths (e.g., [128, 64, 0]).
   - ``network.parameters.use_shunting``: Bool - Enable shunting (explained in core_concepts).
   - ``network.parameters.ee_synapses_per_branch_per_layer``: List[Int] - Excitatory synapses per branch per layer.
   - ``network.parameters.ie_synapses_per_branch_per_layer``: List[Int] - Inhibitory synapses per branch per layer.
   - ``network.parameters.reactivation_strategy``: String - 'inverse' or 'none'.
   - ``network.parameters.epsilon``: Float - Numerical stability constant (default: 1e-6).

Training Configuration
---------------------

2. **Train Config**:
   - ``optimizer``: String - 'adam' or 'sgd' (default: 'adam').
   - ``learning_rate``: Float - Initial LR (default: 0.001).
   - ``batch_size``: Int - Batch size (default: 64).
   - ``num_epochs``: Int - Training epochs (default: 100).
   - ``seed``: Int - Random seed (default: 42).
   - ``learning_strategy``: String - 'mle', 'voltage_stabilization', etc.
   - ``weight_decay``: Float - L2 regularization strength (default: 0.0001).

Task Configuration
-----------------

3. **Task Config**:
   - ``dataset``: String - 'mnist', 'cifar10', or 'multixor' (synthetic).
   - ``parameters.num_samples``: Int - For synthetic datasets (default: 5000).
   - ``parameters.input_dim``: Int - Dataset input dimension.
   - ``parameters.num_classes``: Int - Number of output classes.

Sweep Configuration
------------------

4. **Sweep Config** (for unified sweeps):
   - ``type``: String - 'ei', 'branch', 'noise', 'general'.
   - ``parameters``: Dict - Parameters to sweep (e.g., ee_synapses: [10,20,30]).
   - ``slurm_config``: Dict - SLURM job configuration for cluster runs.
   - ``experiment_settings.seeds_per_condition``: Int - Number of random seeds per parameter combination.
   - ``experiment_settings.base_seed``: Int - Starting random seed.

Example Configuration Files
---------------------------

**Basic Training Config (configs/basic_training.yaml)**:

.. code-block:: yaml

   task:
     dataset: "mnist"
     parameters:
       input_dim: 784
       num_classes: 10

   model:
     network:
       type: "EINet"
       parameters:
         input_dim: 784
         output_dim: 10
         num_layers: 3
         excitatory_layer_sizes: [512, 256, 10]
         inhibitory_layer_sizes: [128, 64, 0]
         use_shunting: true
         epsilon: 1e-6

   train:
     optimizer: "adam"
     learning_rate: 0.001
     batch_size: 64
     num_epochs: 100
     seed: 42

**EI Sweep Config (configs/ei_sweep.yaml)**:

.. code-block:: yaml

   sweep_config:
     model.network.parameters.ee_synapses_per_branch_per_layer: [[10], [20], [30]]
     model.network.parameters.ie_synapses_per_branch_per_layer: [[10], [20], [30]]
     model.network.parameters.use_shunting: [true, false]

   experiment_settings:
     seeds_per_condition: 3
     base_seed: 42

   slurm_config:
     account: "kempner_dev"
     partition: "kempner"
     time: "02:00:00"
     gpus_per_node: 1

**Implementation**: Configs are loaded in ``config.py`` and used in factory functions. The ``load_config()`` function parses YAML files and creates dataclass instances for type safety.

Dendritic Modeling Configuration Guide
==================================================

This comprehensive guide covers all configuration options available in the Dendritic Modeling Framework. All configurations are specified using YAML files that are passed to the training script.

Table of Contents
------------------------------

#. [Overview](#overview)
#. [Model Configuration](#model-configuration)
#. [Training Configuration](#training-configuration)
#. [Task/Dataset Configuration](#taskdataset-configuration)
#. [Analysis Configuration](#analysis-configuration)
#. [Visualization Configuration](#visualization-configuration)
#. [Output Configuration](#output-configuration)
#. [Weights & Biases Configuration](#weights--biases-configuration)
#. [Complete Examples](#complete-examples)

Overview
------------------------------

Configuration files define all aspects of dendritic modeling experiments:

`````yaml
Basic structure
==================================================
model:          # Network architecture and parameters
train:          # Training strategy and hyperparameters  
task:           # Dataset and task configuration
analysis:       # Analysis and metrics configuration
visualization:  # Plotting and visualization options
outputs:        # Output directories and saving options
wandb:          # Weights & Biases logging configuration
`````

Model Configuration
------------------------------

Network Types
~~~~~~~~~~~~~~~~~~~~

The framework supports multiple network architectures:

`````yaml
model:
  task: classification  # or "regression"
  
  # Three-part model structure
  encoder_network:
    type: Identity      # Input encoder (Identity, CNN, MLP, AlexNet)
    parameters:
      input_dim: 784
      
  core_network:
    type: EINet         # Main network (EINet, MLP, MatchedParamMLP)
    parameters:
      # See detailed parameters below
      
  decoder_network:
    type: Identity      # Output decoder (Identity, MLP)
    parameters:
      output_dim: 10
`````

EINet Parameters
~~~~~~~~~~~~~~~~~~~~

**EINet** (Excitation-Inhibition Network) is the main dendritic architecture:

Architecture Parameters
^^^^^^^^^^^^^^^

`````yaml
model:
  core_network:
    type: EINet
    parameters:
      # Layer sizes
      excitatory_layer_sizes: [128, 64, 32]    # Excitatory neurons per layer
      inhibitory_layer_sizes: [64, 32, 16]     # Inhibitory neurons per layer
      
      # Branching structure
      excitatory_branch_factors: [2, 2]        # Branching factors for E neurons
      inhibitory_branch_factors: []            # Usually empty for I neurons
      
      # Connectivity patterns
      ee_synapses_per_branch_per_layer: [100, 50]  # E→E connections
      ei_synapses_per_branch_per_layer: [50, 25]   # E→I connections  
      ie_synapses_per_branch_per_layer: [25, 10]   # I→E connections
      ii_synapses_per_branch_per_layer: [0, 0]     # I→I connections (usually 0)
      
      # Synaptic properties
      somatic_synapses: true                   # Allow direct soma connections
      use_shunting: true                       # Shunting vs subtractive inhibition
`````

Synaptic Selection Mechanisms
^^^^^^^^^^^^^^^

`````yaml
      # TopK synapse selection
      topk_type: standard           # "standard", "stochastic", "variance", "deepst"
      topk_init_method: xavier_normal  # Weight initialization method
      topk_noise_level: 0.0         # Noise for exploration (0.0-1.0)
      
      # Stochastic TopK parameters
      topk_temperature: 0.5         # Temperature for stochastic selection
      topk_ultrafast: true          # Enable fast approximation
      
      # Dynamic Sparse Training (DeepST) parameters
      excitatory_target_density: 0.1    # Target connection density
      inhibitory_target_density: 0.1
      rewiring_mode: global         # "global", "local", "constant_branch"
      rewire_frequency: 1           # Rewiring frequency (epochs)
      use_noise: true               # Add Gaussian noise
      sigma: 0.05                   # Noise standard deviation
      freeze_excitatory_connectivity: false
      freeze_inhibitory_connectivity: false
      weight_threshold: 1e-6        # Pruning threshold
`````

Activation Functions
^^^^^^^^^^^^^^^

`````yaml
      # Reactivation functions
      reactivate: true              # Enable reactivation after branches
      reactivation_type: param_tanh # Activation function type
      reactivation_init_m: 1.5      # Initial slope parameter
      reactivation_init_b: 0.5      # Initial bias parameter
`````

**Available reactivation types:**
* ``none``: No reactivation (identity)
* ``param_tanh``: Parametric tanh with learnable m and b
* ``param_tanh_only_m``: Parametric tanh with only m learnable
* ``relu``: Standard ReLU
* ``sigmoid``: Standard sigmoid
* ``tanh``: Standard tanh
* ``param_linear_sigmoid``: Interpolates between linear and sigmoid
* ``linear_sigmoid``: Non-trainable linear-sigmoid interpolation
* ``param_linear_tanh``: Trainable linear-tanh interpolation
* ``linear_tanh``: Non-trainable linear-tanh interpolation
* ``param_linear_tanh_transition``: Smooth transition from linear to tanh
* ``linear_tanh_transition``: Non-trainable version

Gradient Scaling
^^^^^^^^^^^^^^^

`````yaml
      # Gradient scaling strategies
      reactivation_strategy: inverse    # "none", "inverse"
      topk_strategy: conductance_dynamic # "none", "conductance_dynamic"
      blocklinear_strategy: conductance_dynamic # "none", "distal_upweight_by_idx", "conductance_dynamic"
      
      # Debugging
      print_hooks: false            # Print gradient statistics
`````

Input Processing
^^^^^^^^^^^^^^^

`````yaml
      # Input modes
      input_mode: 1                 # 0: raw→E, None→I; 1: same→E,I; 2: network transform
      
      # Input network (for input_mode=2)
      input_net_dict:
        type: CNNInputTransform     # Transform network type
        
        # CNN parameters
        input_shape: [3, 32, 32]    # Input dimensions [C, H, W]
        excitatory_hidden_dims: [16, 32]      # E pathway conv channels
        inhibitory_hidden_dims: [8, 16]       # I pathway conv channels
        excitatory_kernel_sizes: [5, 3]       # E pathway kernel sizes
        inhibitory_kernel_sizes: [7, 5]       # I pathway kernel sizes
        excitatory_strides: [1, 1]            # E pathway strides
        inhibitory_strides: [1, 1]            # I pathway strides
        
        # MLP parameters (for MLPInputTransform)
        hidden_dims: [512, 256]     # Hidden layer sizes
        excitatory_dim: 256          # E pathway output dimension
        inhibitory_dim: 64           # I pathway output dimension
`````

**Available input transform types:**
* ``IdentityInputTransform``: Passes input unchanged to both E/I pathways
* ``IndexInputTransform``: Random index-based splitting of input dimensions
* **Note**: The following are referenced in documentation but not fully implemented:
  * ``MLPInputTransform``: MLP-based transformation (placeholder)
  * ``CNNInputTransform``: CNN-based transformation (placeholder)  
  * ``AlexNetTransform``: Pre-trained AlexNet features (placeholder)
  * ``FeatureExtractorMLP``: Feature extraction MLP (placeholder)
  * ``SigmoidInputNetTransform``: MLP with sigmoid output (placeholder)

Network Configuration Details
----------------------------

**TopK Selection Types** (``topk_type``):
   - ``standard``: Basic TopK selection by weight magnitude (default)
   - ``stochastic``: Probabilistic selection with temperature control
     - ``topk_temperature``: Float - Selection sharpness (default: 0.5)
     - ``topk_ultrafast``: Bool - Enable faster computation (default: False)
   - ``variance``: Selection based on input variance and weight magnitude
   - ``deepst``: Dynamic sparse training with rewiring

**Rewiring Modes** (``rewiring_mode`` for DeepST):
   - ``global``: Connections can be rewired anywhere in the network (default)
   - ``local``: Connections rewired only within the same neuron
   - ``constant_branch``: Maintain constant number of synapses per branch

**DeepST-Specific Options**:
   - ``use_noise``: Bool - Enable Gaussian noise for Brownian motion (default: True)
   - ``sigma``: Float - Noise standard deviation (default: 0.05)
   - ``freeze_excitatory_connectivity``: Bool - Freeze excitatory synapse locations (default: False)
   - ``freeze_inhibitory_connectivity``: Bool - Freeze inhibitory synapse locations (default: False)
   - ``weight_threshold``: Float - Pruning threshold for weights (default: 1e-6)

**Gradient Scaling Strategies**:
   - ``reactivation_strategy``: 'none', 'inverse' (default: 'inverse')
   - ``topk_strategy``: 'none', 'conductance_dynamic' (default: 'conductance_dynamic')
   - ``blocklinear_strategy``: 'none', 'distal_upweight_by_idx', 'conductance_dynamic' (default: 'conductance_dynamic')

**Reactivation Functions** (``reactivation_type``):
   - ``none``: Identity function
   - ``param_tanh``: Parametric tanh with learnable parameters (default)
   - ``param_tanh_only_m``: Parametric tanh with learnable slope only
   - ``relu``: Standard ReLU
   - ``sigmoid``: Standard sigmoid  
   - ``tanh``: Standard tanh
   - ``param_linear_sigmoid``: Trainable linear-sigmoid interpolation
   - ``linear_sigmoid``: Non-trainable linear-sigmoid interpolation
   - ``param_linear_tanh``: Trainable linear-tanh interpolation
   - ``linear_tanh``: Non-trainable linear-tanh interpolation
   - ``param_linear_tanh_transition``: Trainable linear-tanh transition
   - ``linear_tanh_transition``: Non-trainable linear-tanh transition

**Initialization Methods** (``topk_init_method``, ``init_method``):
   - ``xavier_normal``: Xavier normal initialization (default)
   - ``xavier_uniform``: Xavier uniform initialization
   - ``kaiming_normal``: Kaiming normal initialization
   - ``kaiming_uniform``: Kaiming uniform initialization
   - ``orthogonal``: Orthogonal initialization
   - ``normal``: Standard normal initialization
   - ``uniform``: Uniform initialization
   - ``eye``: Identity matrix initialization
   - ``adaptive_initialization``: Bool - Use network-aware weight scaling (default: True)

**Weight Parameterization**:
   - ``param_space``: String - Weight parameterization: 'log', 'presigmoid', 'uniform' (default: 'log')

**Transfer Function Options** (``transfer_params``):
   - ``input_mode``: 0 (raw→E, None→I), 1 (same→E,I), 2 (network transform)
   - ``independent_pathways``: Bool - Use separate input streams for E/I
   - ``excitatory_dim``: Int - Output dim for E pathway (independent mode)
   - ``inhibitory_dim``: Int - Output dim for I pathway (independent mode)  
   - ``output_activation``: 'sigmoid', 'relu', 'tanh', 'softplus', 'none'

Other Network Types
~~~~~~~~~~~~~~~~~~~~

Standard MLP
^^^^^^^^^^^^^^^

`````yaml
model:
  core_network:
    type: MLP
    parameters:
      input_dim: 784
      hidden_dims: [512, 256, 128]
      activation: relu              # "relu", "sigmoid", "tanh"
      output_dim: 10
`````

Parameter-Matched MLP
^^^^^^^^^^^^^^^

`````yaml
model:
  core_network:
    type: MatchedParamMLP
    parameters:
      # Uses same parameters as EINet to match parameter count
      input_dim: 784
      excitatory_layer_sizes: [128, 64]
      inhibitory_layer_sizes: [64, 32]
      # ... other EINet parameters
`````

Training Configuration
------------------------------

Learning Strategies
~~~~~~~~~~~~~~~~~~~~

`````yaml
train:
  learning_strategy: standard      # Training strategy
  
  # Basic parameters
  run_name: my_experiment          # Experiment name
  seed: 42                         # Random seed
  train_valid_split: 0.8           # Train/validation split ratio
  weight_decay_rate: 0.1           # L2 regularization
`````

**Available learning strategies:**
* ``standard``: Traditional backpropagation
* ``freeze_layers``: Train layers sequentially
* ``freeze_branches``: Train branches sequentially (local learning)
* ``two_step``: Local pretraining + global fine-tuning
* ``two_step_kl``: Two-step with KL divergence
* ``voltage_stabilization``: Biologically-motivated warm-up

Optimizer Configuration
~~~~~~~~~~~~~~~~~~~~

`````yaml
train:
  optim_dict:
    # Simple configuration
    split_params: false            # Use single learning rate
    lr: 0.001                     # Learning rate
    
    # Advanced configuration  
    split_params: true            # Use different LRs for components
    input_net_lr: 0.001          # Input network learning rate
    topk_lr: 0.005               # Synaptic weight learning rate
    blocklinear_lr: 0.001        # Branch aggregation learning rate
    reactivation_lr: 0.01        # Activation parameter learning rate
    output_layer_lr: 0.001       # Output layer learning rate
`````

Trainer Parameters
~~~~~~~~~~~~~~~~~~~~

`````yaml
train:
  trainer:
    # Basic training
    epochs: 100                   # Number of training epochs
    batch_size: 128              # Batch size
    loss_function: categorical_nll # Loss function
    shuffle: true                # Shuffle training data
    
    # Optimization
    grad_clip_value: 5.0         # Gradient clipping
    
    # Early stopping
    early_stopping: true         # Enable early stopping
    patience: 10                 # Early stopping patience
    load_best_state_dict: true   # Load best model after training
    
    # Logging
    plot_losses: true            # Plot training curves
    suppress_prints: false       # Suppress console output
    print_every: 10              # Print frequency (epochs)
    
    # Strategy-specific parameters
    pretrain_epochs: 20          # Pretraining epochs (two-step, voltage_stabilization)
    target_voltage: 0.5          # Target voltage (voltage_stabilization)
    kl_weight: 0.1              # KL divergence weight (two_step_kl)
    reverse_training: false      # Train in reverse order (freeze strategies)
    epochs_per_layer: 10         # Epochs per layer (freeze_layers)
    epochs_per_branch: 10        # Epochs per branch (freeze_branches)
    final_tune_epochs: 20        # Final tuning epochs (multi-stage strategies)
`````

**Available loss functions:**
* ``categorical_nll``: Categorical negative log-likelihood
* ``mse``: Mean squared error
* ``cross_entropy``: Cross-entropy loss
* ``bce``: Binary cross-entropy

Encoder Network Training
~~~~~~~~~~~~~~~~~~~~

For models with sophisticated encoder networks:

`````yaml
train:
  encoder:
    # Training mode
    training_mode: sequential     # "joint", "sequential"
    train_encoder: true          # Whether to train encoder
    encoder_epochs: 20           # Encoder pretraining epochs
    
    # Independent head training
    train_heads_independently: false
    inhibitory_head_objective: classification  # "classification", "diversity", "combined"
    diversity_weight: 0.5        # Weight for diversity loss
    
    # Pre-trained weights
    pretrained_options:
      source: none               # "file", "pytorch", "none"
      weights_path: ""           # Path to weights file
      pytorch_model: ""          # PyTorch model name (e.g., "alexnet")
`````

Main Network Training
~~~~~~~~~~~~~~~~~~~~

For multi-stage training strategies:

`````yaml
train:
  main:
    # Parameter groups for different learning rates
    param_groups:
      topk: 0.005               # Synaptic weights
      blocklinear: 0.001        # Branch aggregation
      reactivation: 0.01        # Activation parameters
      output_layer: 0.001       # Output layer
    
    weight_decay_rate: 0.1      # Weight decay
    
    trainer:
      # Same parameters as above
      epochs: 80
      batch_size: 128
      # ...
`````

Task/Dataset Configuration
------------------------------

Dataset Selection
~~~~~~~~~~~~~~~~~~~~

`````yaml
task:
  dataset: mnist                 # Dataset name
  data_path: ""                  # Custom data path (empty = default)
  
  parameters:
    # Common parameters
    train_size: 8000             # Training samples
    valid_size: 1000             # Validation samples  
    test_size: 1000              # Test samples
    flatten: true                # Flatten images
    normalize: false             # Normalize inputs
`````

Supported Datasets
~~~~~~~~~~~~~~~~~~~~

Vision Datasets
^^^^^^^^^^^^^^^

`````yaml
MNIST handwritten digits
==================================================
task:
  dataset: mnist
  parameters:
    flatten: true               # Flatten 28x28 → 784
    normalize: true             # Apply normalization

CIFAR-10 natural images  
==================================================
task:
  dataset: cifar10
  parameters:
    flatten: false              # Keep spatial structure
    normalize: true
`````

Synthetic Datasets
^^^^^^^^^^^^^^^

`````yaml
Line detection
==================================================
task:
  dataset: line
  parameters:
    image_size: 16              # Image dimensions
    
Center-surround patterns
==================================================
task:
  dataset: center_surround  
  parameters:
    image_size: 15

Orientation bars
==================================================
task:
  dataset: orientation_bars
  parameters:
    image_size: 16
    n_orient_classes: 8         # Number of orientations

Multi-bit XOR
==================================================
task:
  dataset: multi_xor
  parameters:
    n_bits: 8                   # Number of input bits
    
Information routing
==================================================
task:
  dataset: info_shunting
  parameters:
    input_dim: 10               # Input dimensionality
    
Noisy line detection
==================================================
task:
  dataset: noise_resilience
  parameters:
    image_size: 16
    noise_level: 0.2            # Gaussian noise level
`````

Advanced Datasets
^^^^^^^^^^^^^^^

`````yaml
Correlated Gaussian inputs
==================================================
task:
  dataset: synthetic_correlated
  parameters:
    input_dim: 300              # Input dimensionality
    correlation_range: [-0.1, 0.1]  # Correlation range
    mean_magnitude: 0.5         # Class separation
    
Poisson-sampled data
==================================================
task:
  dataset: poisson_generator
  parameters:
    base_dataset: mnist         # Base dataset
    rate_multiplier: 1.0        # Poisson rate
    
Orthogonal network data
==================================================
task:
  dataset: orthonet
  parameters:
    input_dim: 10
    n_layers: 5                 # Network depth
    nonlinearity: sigmoid       # "sigmoid", "tanh", None
    repeat_layers: false        # Repeat layer weights
    noise_std: 0.1             # Noise for repeated layers
`````

MNIST Variants
^^^^^^^^^^^^^^^

`````yaml
MNIST with modulo labels
==================================================
task:
  dataset: mnist_modulo10
  parameters:
    shuffle_iterations: 2
    
MNIST with random flips
==================================================
task:
  dataset: random_flip_mnist
  parameters:
    random_flip: true
    
Sparse MNIST (feature selectivity)
==================================================
task:
  dataset: feature_selectivity
  parameters:
    threshold: 0.3
    
Contextual figure-ground
==================================================
task:
  dataset: context_gating
  parameters:
    noise_level: 0.1
    
Compositional MNIST
==================================================
task:
  dataset: hierarchical_processing
  parameters:
    n_samples: 10000
`````

Analysis Configuration
------------------------------

Performance Analysis
~~~~~~~~~~~~~~~~~~~~

`````yaml
analysis:
  performance_analysis:
    enabled: true               # Enable performance metrics
    training: false             # Run during training
    params:
      accuracy: true            # Classification accuracy
      categorical_loglikelihood: true  # Log-likelihood
      mse: false               # Mean squared error (regression)
`````

Information-Theoretic Analysis
~~~~~~~~~~~~~~~~~~~~

`````yaml
analysis:
  information_analysis:
    enabled: true
    training: false             # Usually false (expensive)
    params:
      # Estimation method
      method: kraskov           # "kraskov", "sklearn", "copula", "auto"
      max_samples: 5000         # Limit samples for efficiency
      
      # k-NN estimator (kraskov)
      n_neighbors: 10           # Number of neighbors (5-20)
      
      # Histogram estimator (sklearn)
      n_bins: 20               # Histogram bins
      
      # Copula estimator
      copula_type: gaussian     # "gaussian", "t", "clayton"
      
      # Analysis types
      compute_basic_mi: true    # I(E;C), I(I;C), I(Vout;C)
      compute_pairwise_mi: true # I(E;I), I(E;Vout), I(I;Vout)
      compute_conditional_mi: true # I(E;I|C), I(E;Vout|C)
      compute_pid: false        # Partial Information Decomposition
      
      # PID settings
      pid_binarize: true
      pid_binarize_method: median # "median", "mean", "quantile"
      pid_binarize_threshold: 0.5
      
      # Additional options
      include_vinf: true        # Include soma activation
      normalize_mi: false       # Normalize MI by entropy
      verbose: false           # Detailed progress
      compute_per_branch: true  # Per-branch analysis
      branch_aggregation: none  # Branch aggregation method
`````

Ablation Analysis
~~~~~~~~~~~~~~~~~~~~

`````yaml
analysis:
  ablation_analysis:
    enabled: true
    training: false
    params:
      layer_ablation: true      # Test removing layers
      compartment_ablation: true # Test removing E/I compartments
      ablate_all_synapses: true # Test removing all synapses
      ablate_excitation: true   # Test removing excitation
      ablate_inhibition: true   # Test removing inhibition
      ablate_upstream: true     # Test removing branches
      accuracy: true           # Measure accuracy
      categorical_loglikelihood: true # Measure log-likelihood
`````

Specialized Analyses
~~~~~~~~~~~~~~~~~~~~

`````yaml
analysis:
  # Synaptic turnover tracking
  synapse_turnover_analysis:
    enabled: false
    training: false            # Track during training
    
  # Noise robustness
  noise_perturbation_analysis:
    enabled: false
    training: false
    params:
      uniform_noise: true      # Test uniform noise
      uniform_magnitudes: [0.1, 0.2]
      gaussian_noise: true     # Test Gaussian noise
      gaussian_sdevs: [0.1, 0.2]
      n_samples: 10           # Samples per noise level
      accuracy: true
      categorical_loglikelihood: true
      
  # Weight analysis
  weight_analysis:
    enabled: false
    training: false
    params:
      synapse_weights: true    # Analyze synapse weights
      branch_weights: true     # Analyze branch weights
      
  # Synaptic activation analysis
  synaptic_activation_analysis:
    enabled: false
    training: false
    params:
      network_summary: true    # Network-wide stats
      layer_summary: true      # Per-layer stats
      branch_summary: true     # Per-branch stats
      raw_summary: true        # Raw values
      logspace: false         # Log-transformed values
      downsample: true        # Downsample for efficiency
      samples: 1000           # Sample size
      
  # Branch activation analysis
  branch_activation_analysis:
    enabled: false
    training: false
    params:
      synapse_activations: true # E/I synapse activations
      branch_activations: true  # Branch outputs
      network_summary: true
      layer_summary: true
      branch_summary: true
      raw_summary: true
      logspace: false
      downsample: true
      samples: 1000
      
  # Single layer contribution
  single_layer_contribution:
    enabled: true
    training: false
    params:
      save_plots: true         # Generate plots
      save_csv: true          # Save numerical results
`````

Visualization Configuration
------------------------------

`````yaml
visualization:
  enabled: true                # Enable visualization
  visualize_epochs: [10, 30, 50] # Epochs for visualization
  
  # Available plot types
  plots:
    - weights                  # Synaptic weight matrices
    - activations             # Activation distributions  
    - gradients               # Gradient flow
    - profiles                # Dendritic profiles
    - branch_info             # Branch information content
    - branch_ablation         # Branch ablation effects
    - layer_ablation          # Layer ablation effects
    - dendritic_ablation_all  # Comprehensive ablation
    - dendritic_information   # Information processing
    - dendritic_information_inputs      # Input information
    - dendritic_information_activations # Activation information
`````

Output Configuration
------------------------------

`````yaml
outputs:
  dir: ./outputs/              # Base output directory
  run_name: experiment_001     # Run identifier
  
  # Subdirectories created automatically:
  # - models/: Model checkpoints
  # - logs/: Training logs  
  # - analysis/: Analysis results
  # - visualizations/: Generated plots
`````

Weights & Biases Configuration
------------------------------

`````yaml
wandb:
  use_wandb: true             # Enable W&B logging
  entity: my_username         # W&B username/team
  project: dendritic_modeling # Project name
  group: experiments          # Run group
  tags: [dendritic, ei_network] # Run tags
`````

FSDP Configuration
------------------------------

For large models, enable FSDP in train section:
`````yaml
train:
  use_fsdp: true
  fsdp_config:
    mixed_precision: true
    cpu_offload: false
    sharding_strategy: full
`````

Run with torchrun for multi-GPU.

Complete Examples
------------------------------

Simple EINet Classification
~~~~~~~~~~~~~~~~~~~~

`````yaml
configs/simple_einet.yaml
==================================================
model:
  task: classification
  encoder_network:
    type: Identity
    parameters:
      input_dim: 784
  core_network:
    type: EINet
    parameters:
      excitatory_layer_sizes: [64]
      inhibitory_layer_sizes: [32] 
      ee_synapses_per_branch_per_layer: [100]
      ie_synapses_per_branch_per_layer: [50]
      use_shunting: true
      reactivation_type: param_tanh
  decoder_network:
    type: Identity
    parameters:
      output_dim: 10

train:
  learning_strategy: standard
  trainer:
    epochs: 50
    batch_size: 128
    loss_function: categorical_nll

task:
  dataset: mnist
  parameters:
    flatten: true
    normalize: true

analysis:
  performance_analysis:
    enabled: true
    
outputs:
  dir: ./results/simple_einet/
`````

Advanced Multi-Layer EINet
~~~~~~~~~~~~~~~~~~~~

`````yaml
configs/advanced_einet.yaml
==================================================
model:
  task: classification
  encoder_network:
    type: CNNDownsample
    parameters:
      input_shape: [3, 32, 32]
      hidden_dims: [32, 64]
      kernel_sizes: [5, 3]
      strides: [1, 2]
      output_dim: 512
  core_network:
    type: EINet
    parameters:
      excitatory_layer_sizes: [256, 128, 64]
      inhibitory_layer_sizes: [128, 64, 32]
      excitatory_branch_factors: [2, 2]
      ee_synapses_per_branch_per_layer: [200, 100]
      ei_synapses_per_branch_per_layer: [100, 50]
      ie_synapses_per_branch_per_layer: [50, 25]
      
      # Advanced synaptic properties
      topk_type: stochastic
      topk_temperature: 0.3
      reactivation_type: param_tanh
      reactivation_strategy: inverse
      topk_strategy: conductance_dynamic
      blocklinear_strategy: conductance_dynamic
  decoder_network:
    type: MLP
    parameters:
      input_dim: 64
      hidden_dims: [32]
      output_dim: 10

train:
  learning_strategy: two_step
  encoder:
    training_mode: sequential
    encoder_epochs: 20
  main:
    param_groups:
      topk: 0.005
      blocklinear: 0.001
      reactivation: 0.01
      output_layer: 0.001
    weight_decay_rate: 0.1
    trainer:
      epochs: 100
      batch_size: 64
      pretrain_epochs: 20
      final_tune_epochs: 30

task:
  dataset: cifar10
  parameters:
    normalize: true

analysis:
  performance_analysis:
    enabled: true
  information_analysis:
    enabled: true
    params:
      method: kraskov
      compute_basic_mi: true
      compute_pairwise_mi: true
      max_samples: 3000
  ablation_analysis:
    enabled: true
    params:
      layer_ablation: true
      compartment_ablation: true

visualization:
  enabled: true
  plots:
    - weights
    - activations
    - branch_info
    - branch_ablation

wandb:
  use_wandb: true
  project: advanced_dendritic
  tags: [cifar10, two_step, stochastic]

outputs:
  dir: ./results/advanced_einet/
`````

Dynamic Sparse Training Example
~~~~~~~~~~~~~~~~~~~~

`````yaml
configs/deepst_example.yaml
==================================================
model:
  task: classification
  encoder_network:
    type: Identity
    parameters:
      input_dim: 784
  core_network:
    type: EINet
    parameters:
      excitatory_layer_sizes: [128, 64]
      inhibitory_layer_sizes: [64, 32]
      excitatory_branch_factors: [2]
      ee_synapses_per_branch_per_layer: [150]
      ie_synapses_per_branch_per_layer: [75]
      
      # Dynamic sparse training
      topk_type: deepst
      excitatory_target_density: 0.1
      inhibitory_target_density: 0.05
      rewiring_mode: global
      rewire_frequency: 5
      use_noise: true
      sigma: 0.05
      
      # Standard reactivation
      reactivation_type: param_tanh
  decoder_network:
    type: Identity
    parameters:
      output_dim: 10

train:
  learning_strategy: standard
  trainer:
    epochs: 200
    batch_size: 128

task:
  dataset: mnist
  parameters:
    flatten: true

analysis:
  synapse_turnover_analysis:
    enabled: true
    training: true
  performance_analysis:
    enabled: true

outputs:
  dir: ./results/deepst_experiment/
`````

Regression Example
~~~~~~~~~~~~~~~~~~~~

`````yaml
configs/regression_example.yaml
==================================================
model:
  task: regression
  encoder_network:
    type: Identity
    parameters:
      input_dim: 100
  core_network:
    type: EINet
    parameters:
      excitatory_layer_sizes: [64, 32]
      inhibitory_layer_sizes: [32, 16]
      ee_synapses_per_branch_per_layer: [80]
      ie_synapses_per_branch_per_layer: [40]
      use_shunting: true
  decoder_network:
    type: MLP
    parameters:
      input_dim: 32
      hidden_dims: [16]
      output_dim: 1

train:
  learning_strategy: standard
  trainer:
    epochs: 100
    batch_size: 64
    loss_function: mse

task:
  dataset: orthonet
  parameters:
    input_dim: 100
    n_layers: 3
    nonlinearity: sigmoid

analysis:
  performance_analysis:
    enabled: true
    params:
      accuracy: false
      mse: true
      categorical_loglikelihood: false

outputs:
  dir: ./results/regression/
`````

FSDP Large Model Example
~~~~~~~~~~~~~~~~~~~~

`````yaml
configs/fsdp_large_model.yaml
==================================================
model:
  task: classification
  core_network:
    type: EINet
    parameters:
      excitatory_layer_sizes: [1024, 512]
train:
  use_fsdp: true
  trainer:
    batch_size: 64
`````

Sweep Configuration Example
~~~~~~~~~~~~~~~~~~~~

`````yaml
configs/sweep_config.yaml
==================================================
sweep_config:
  model.network.parameters.excitatory_layer_sizes: [[10], [20]]
train:
  use_fsdp: true  # FSDP in sweeps
`````

---

This configuration guide covers all the available options in the Dendritic Modeling Framework. For additional examples and tutorials, see the ``configs/`` directory and the framework documentation. 