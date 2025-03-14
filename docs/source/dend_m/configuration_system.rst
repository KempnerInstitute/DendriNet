.. _configuration_system:

Configuration System
====================

This document explains the YAML configuration system used in the dendritic modeling framework, which provides a flexible way to define experiment parameters without changing code.

Overview
--------

The dendritic modeling framework uses a hierarchical YAML configuration system loaded via OmegaConf that allows you to specify all aspects of an experiment, including:

1. Model architecture and parameters
2. Training procedure
3. Task and dataset settings
4. Visualization options
5. Experiment tracking with Weights & Biases (optional)

This design separates the configuration from the code, making it easier to run experiments with different settings and ensuring reproducibility.

Running Experiments
-------------------

You can run experiments using the provided training script with a configuration file:

.. code-block:: bash

    python scripts/train_experiments.py configs/config_exp.yaml

Configuration Structure
-----------------------

The configuration is organized into six main sections:

1. **model**: Network architecture and parameters
2. **train**: Training settings and hyperparameters
3. **task**: Dataset configuration and parameters
4. **visualization**: Settings for visualizing model components
5. **wandb**: Weights & Biases integration
6. **outputs**: Output directory settings

Here's an example of a complete configuration (simplified):

.. code-block:: yaml

    model:
      task: classification
      probabilistic: false
      network:
        type: EINet
        parameters:
          excitatory_layer_sizes: [10]
          inhibitory_layer_sizes: [20]
          excitatory_branch_factors: [1,2]
          inhibitory_branch_factors: []
          reactivate: true
          somatic_synapses: true
          ee_synapses_per_branch_per_layer: [24]
          ei_synapses_per_branch_per_layer: [200]
          ie_synapses_per_branch_per_layer: [2]
          ii_synapses_per_branch_per_layer: [0]
          topk_init_method: xavier_normal
          use_shunting: true
          reactivation_type: param_tanh
          reactivation_strategy: inverse
          blocklinear_strategy: block_conductance_dynamic
          input_mode: 2
          input_net_dict:
            type: "MLPInputTransform"
            hidden_dims: []
            output_dim: 128

    train:
      learning_strategy: mle
      run_name: mnist_einet
      seed: 42
      train_valid_split: 0.8
      lr: 0.001
      epochs: 50
      batch_size: 64
      grad_clip_value: 1.0
      weight_decay_rate: 0.01
      input_net_training:
        training_mode: "sequential"
        train_input_net: true
        input_net_epochs: 20
        train_heads_independently: false
        inhibitory_head_objective: "classification"

    task:
      dataset: mnist
      data_path: 
      parameters: {}

    wandb:
      entity: 
      project: 
      group: 
      tags: 
      use_wandb: false

    visualization:
      enabled: false
      visualize_epochs: [10, 30]
      plots:
        - weights
        - activations
        - gradients

    outputs:
      dir: outputs

Model Configuration
-------------------

The ``model`` section defines the neural network architecture and its parameters:

.. code-block:: yaml

    model:
      task: classification  # Type of task: "classification", "regression"
      probabilistic: false  # Whether to use probabilistic outputs
      network:
        type: EINet  # Network type: "EINet", "MLP"
        parameters:  # Network-specific parameters
          # EINet specific parameters
          excitatory_layer_sizes: [10]  # Number of excitatory cells per layer
          inhibitory_layer_sizes: [20]  # Number of inhibitory cells per layer
          excitatory_branch_factors: [1,2]  # Branching factors for excitatory dendrites
          inhibitory_branch_factors: []  # Branching factors for inhibitory dendrites
          # Synaptic connections
          ee_synapses_per_branch_per_layer: [24]  # Excitatory-to-excitatory synapses
          ei_synapses_per_branch_per_layer: [200]  # Excitatory-to-inhibitory synapses
          ie_synapses_per_branch_per_layer: [2]  # Inhibitory-to-excitatory synapses
          ii_synapses_per_branch_per_layer: [0]  # Inhibitory-to-inhibitory synapses
          # Activation and initialization
          reactivate: true  # Whether to use reactivation
          somatic_synapses: true  # Whether to use somatic synapses
          topk_init_method: xavier_normal  # Initialization method for TopK layers
          # Computational strategies
          use_shunting: true  # Whether to use shunting inhibition
          reactivation_type: param_tanh  # Type of reactivation function
          reactivation_strategy: inverse  # Strategy for reactivation
          blocklinear_strategy: block_conductance_dynamic  # Strategy for block linear layers
          # Output options
          enable_branch_outputs: false  # Whether to keep branch activations
          output_layer: false  # Whether to add an output classification layer
          output_dim: 10  # Dimension of output layer
          # Input processing
          input_mode: 2  # 0=original, 1=exc=x+inh=x, 2=input_net
          input_net_dict:  # Configuration for input network (used when input_mode=2)
            type: "MLPInputTransform"  # Type of input transform
            hidden_dims: []  # Hidden dimensions for input network
            output_dim: 128  # Output dimension for input network

Input Modes
~~~~~~~~~~~

The framework supports three input modes:

1. **input_mode=0**: Original mode with raw inputs
2. **input_mode=1**: Direct mode where excitatory and inhibitory inputs are the same
3. **input_mode=2**: Input network mode where inputs are first transformed through a separate network

Input Network
~~~~~~~~~~~~~

When ``input_mode=2``, the ``input_net_dict`` specifies the architecture of the input transformation network:

.. code-block:: yaml

    input_net_dict:
      type: "MLPInputTransform"  # Type of input network
      hidden_dims: [512, 256]    # Hidden layer dimensions
      output_dim: 128           # Output dimension

Training Configuration
----------------------

The ``train`` section defines the training procedure and hyperparameters:

.. code-block:: yaml

    train:
      learning_strategy: mle  # Learning strategy: "mle", "freeze_layers", "local_credit_assignment", "two_step", "two_step_kl"
      run_name: mnist_einet  # Name for the experiment run
      seed: 42  # Random seed for reproducibility
      train_valid_split: 0.8  # Train/validation split ratio
      lr: 0.001  # Learning rate
      weight_decay_rate: 0.01  # Weight decay rate
      epochs: 50  # Number of training epochs
      batch_size: 64  # Batch size
      grad_clip_value: 1.0  # Gradient clipping value
      epochs_per_layer: 50  # Epochs per layer for layer-wise training
      pretrain_epochs: 10  # Pretraining epochs for two-step strategies
      # Input network training options
      input_net_training:
        training_mode: "sequential"  # "joint" or "sequential"
        train_input_net: true  # Whether to train the input network
        input_net_epochs: 20  # Number of epochs for input network training
        train_heads_independently: false  # Train excitatory and inhibitory heads independently
        inhibitory_head_objective: "classification"  # "classification", "diversity", "combined"
        diversity_weight: 0.5  # Weight for diversity loss

Task Configuration
------------------

The ``task`` section defines the dataset and task-specific parameters:

.. code-block:: yaml

    task:
      dataset: mnist  # Dataset name
      data_path:  # Path to dataset (optional)
      parameters:  # Dataset-specific parameters
        train_size: 8000  # Number of training samples
        valid_size: 1000  # Number of validation samples
        test_size: 1000  # Number of test samples
        image_size: 16  # Image size for some datasets

Available datasets include:
- Standard datasets: ``"mnist"``, ``"cifar10"``
- Synthetic datasets: ``"line"``, ``"center_surround"``, ``"orientation_bars"``, ``"multi_xor"``, ``"mnist_modulo10"``, ``"mnist_switch2"``, ``"info_shunting"``, ``"noise_resilience"``, ``"feature_selectivity"``

Visualization Configuration
---------------------------

The ``visualization`` section controls what aspects of the model to visualize:

.. code-block:: yaml

    visualization:
      enabled: true  # Whether to enable visualizations
      visualize_epochs: [10, 30]  # Epochs at which to visualize
      plots:  # Types of plots to generate
        - weights  # Plot weights
        - activations  # Plot activations
        - gradients  # Plot gradients
        - profiles  # Plot dendritic compartment profiles
        - branch_info  # Plot mutual information
        - ablation  # Plot ablation studies

WandB Configuration
-------------------

The ``wandb`` section configures Weights & Biases integration:

.. code-block:: yaml

    wandb:
      entity:  # WandB entity name
      project:  # WandB project name
      group:  # WandB group name
      tags:  # Tags for the experiment
      use_wandb: false  # Whether to use WandB

Outputs Configuration
---------------------

The ``outputs`` section specifies where to save outputs:

.. code-block:: yaml

    outputs:
      dir: outputs  # Directory for saving outputs

Generating Sweep Configurations
-------------------------------

The framework includes a powerful tool for generating parameter sweeps. Using the ``generate_sweep_configs.py`` script, you can create multiple configuration files that cover a range of parameter values:

.. code-block:: bash

    python scripts/generate_sweep_configs.py --param_file configs/sweep_params.yaml --output_dir configs/sweeps

Sweep Parameters File
~~~~~~~~~~~~~~~~~~~~~

Create a ``sweep_params.yaml`` file that defines the parameters to sweep:

.. code-block:: yaml

    base_config: configs/config_exp.yaml  # Base configuration to modify
    
    # Parameters to sweep, with their range of values
    sweep_params:
      model.network.parameters.excitatory_layer_sizes:
        - [10]
        - [20]
        - [50]
      train.lr:
        - 0.001
        - 0.0005
        - 0.0001
      train.batch_size:
        - 32
        - 64
        - 128

This will generate 27 configuration files (3×3×3 combinations) in the specified output directory.

Generating Job Submission Scripts
---------------------------------

The ``generate_sweep_configs.py`` script can also generate SLURM job submission scripts:

.. code-block:: bash

    python scripts/generate_sweep_configs.py --param_file configs/sweep_params.yaml --output_dir configs/sweeps --generate_run_script --slurm

This will create a ``run.sh`` file that submits all generated configuration files as a SLURM array job.

.. figure:: figures/pdf/place_holder.pdf
   :alt: Configuration System Diagram
   :width: 600px
   :align: center

   Diagram showing how the configuration system integrates with different components. [PLACEHOLDER FOR DIAGRAM]

Loading and Extending Configurations
------------------------------------

In custom scripts, you can load and modify configurations programmatically:

.. code-block:: python

    from dendritic_modeling.config import load_config
    
    # Load configuration from file
    config = load_config("configs/config_exp.yaml")
    
    # Modify configuration
    config.train.lr = 0.0005
    config.model.network.parameters.excitatory_layer_sizes = [20]
    
    # Use configuration in your script
    # ...

This allows for flexible experimentation while maintaining reproducibility. 