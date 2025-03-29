Configuration System
====================

The Dendritic Modeling framework uses a flexible YAML-based configuration system to define experiment parameters. This approach allows you to:

1. Define complex experiments without changing code
2. Track experiment parameters for reproducibility
3. Run parameter sweeps by varying configuration values
4. Share experiment setups with colleagues

Basic Configuration Structure
-----------------------------

Configuration files are organized into several sections:

.. code-block:: yaml

    model:
      # Model architecture and parameters
      
    train:
      # Training parameters
      
    task:
      # Dataset and task parameters
      
    wandb:
      # Experiment tracking settings
      
    visualization:
      # Visualization parameters
      
    outputs:
      # Output directory settings

Let's explore each section in detail:

Model Configuration
-------------------

The `model` section defines the network architecture and parameters:

.. code-block:: yaml

    model:
      task: classification
      probabilistic: false
      network:
        type: EINet  # "EINet" or "MLP"
        parameters:
          excitatory_layer_sizes: [10]
          inhibitory_layer_sizes: [20]
          excitatory_branch_factors: [2, 2]
          inhibitory_branch_factors: []
          ee_synapses_per_branch_per_layer: [24]
          ei_synapses_per_branch_per_layer: [100]
          ie_synapses_per_branch_per_layer: [2]
          ii_synapses_per_branch_per_layer: [0]
          reactivate: true
          somatic_synapses: true
          topk_init_method: xavier_normal
          use_shunting: true
          reactivation_type: param_tanh
          reactivation_strategy: inverse
          blocklinear_strategy: block_conductance_dynamic
          enable_branch_outputs: true
          output_layer: false
          output_dim: 10
          
          # Input mode (0=original, 1=exc=x+inh=x, 2=input_net)
          input_mode: 2
          input_net_dict:
            type: "MLPInputTransform"
            hidden_dims: [512, 256]
            output_dim: 128

Key parameters include:

- **network.type**: Network architecture to use (`EINet` or `MLP`)
- **excitatory_layer_sizes/inhibitory_layer_sizes**: Number of cells in each layer
- **excitatory_branch_factors/inhibitory_branch_factors**: Branching structure
- **synapses_per_branch_per_layer**: Connectivity parameters
- **input_mode**: Method for handling inputs (direct or through input network)

Training Configuration
----------------------

The `train` section defines the training process and hyperparameters:

.. code-block:: yaml

    train:
      learning_strategy: mle  # Multiple strategies available (see below)
      run_name: mnist_einet
      seed: 42
      train_valid_split: 0.8
      lr: 0.001
      weight_decay_rate: 0.01
      epochs: 50
      batch_size: 64
      grad_clip_value: 1.0
      
      # For branch-wise learning strategies
      epochs_per_layer: 10   # Number of epochs to train each branch
      final_tune_epochs: 5   # Number of epochs for final fine-tuning
      
      # For strategies with pretraining
      pretrain_epochs: 10    # Number of epochs for initial full network training
      
      # For voltage stabilization strategy
      warmup_epochs: 1       # Number of epochs for activation stabilization
      stability_target_voltage: 0.5  # Target activation value
      
      # Input network training configuration
      input_net_training:
        training_mode: "sequential"  # "joint" or "sequential"
        train_input_net: true
        input_net_epochs: 20
        train_heads_independently: false

Key parameters include:

- **learning_strategy**: Algorithm for training the network
- **lr**, **batch_size**, **epochs**: Standard training hyperparameters
- **epochs_per_layer**, **final_tune_epochs**: Parameters for branch-wise training strategies
- **pretrain_epochs**: Parameter for strategies with initial full-network training
- **input_net_training**: Configuration for input transformation networks

Available Learning Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides several learning strategies that can be specified in the configuration:

1. **MLE (Maximum Likelihood Estimation)** - `learning_strategy: mle`
   
   Standard end-to-end training where all parameters are updated simultaneously.

2. **Freeze Branches** - `learning_strategy: freeze_branches`
   
   Trains the network one branch at a time, from the soma (output) to the input, focusing on:
   
   - Training soma-proximal branches first, then progressively moving to distal branches
   - Using `epochs_per_layer` epochs for each branch (parameter name remains the same despite operating on branches)
   - Performing final tuning of all branches for `final_tune_epochs` epochs
   
   Example configuration:
   
   .. code-block:: yaml
   
       train:
         learning_strategy: freeze_branches
         epochs_per_layer: 20    # Each branch is trained for 20 epochs
         final_tune_epochs: 10   # 10 epochs of full-network fine-tuning

3. **Freeze Branches Reversed** - `learning_strategy: freeze_branches_reversed`
   
   Similar to freeze_branches, but trains the network in the opposite direction:
   
   - Starting from distal branches (input) and moving toward the soma (output)
   - Following the natural flow of information in neural dendrites
   - More biologically plausible than standard freeze_branches
   
   Example configuration:
   
   .. code-block:: yaml
   
       train:
         learning_strategy: freeze_branches_reversed
         epochs_per_layer: 20    # Each branch is trained for 20 epochs
         final_tune_epochs: 10   # 10 epochs of full-network fine-tuning

4. **Freeze Branches with Pretraining** - `learning_strategy: freeze_branches_with_pretraining`
   
   Three-phase training approach:
   
   - Initial full-network training for `pretrain_epochs` epochs
   - Branch-by-branch training from soma to input for `epochs_per_layer` epochs per branch
   - Final full-network fine-tuning for `final_tune_epochs` epochs
   
   Example configuration:
   
   .. code-block:: yaml
   
       train:
         learning_strategy: freeze_branches_with_pretraining
         pretrain_epochs: 10     # Initial full-network training
         epochs_per_layer: 20    # Each branch is trained for 20 epochs
         final_tune_epochs: 10   # 10 epochs of full-network fine-tuning

5. **Freeze Branches Reversed with Pretraining** - `learning_strategy: freeze_branches_reversed_with_pretraining`
   
   Combines pretraining with the reversed branch training approach:
   
   - Initial full-network training for `pretrain_epochs` epochs
   - Branch-by-branch training from input to soma for `epochs_per_layer` epochs per branch
   - Final full-network fine-tuning for `final_tune_epochs` epochs
   
   Example configuration:
   
   .. code-block:: yaml
   
       train:
         learning_strategy: freeze_branches_reversed_with_pretraining
         pretrain_epochs: 10     # Initial full-network training
         epochs_per_layer: 20    # Each branch is trained for 20 epochs
         final_tune_epochs: 10   # 10 epochs of full-network fine-tuning

6. **Two-Step Training** - `learning_strategy: two_step`
   
   Two-phase training approach:
   
   - Pretraining phase for `pretrain_epochs` epochs
   - Main training phase for remaining epochs
   
   Example configuration:
   
   .. code-block:: yaml
   
       train:
         learning_strategy: two_step
         pretrain_epochs: 10     # Pretraining phase
         epochs: 50              # Total epochs (including pretraining)

7. **Voltage Stabilization** - `learning_strategy: voltage_stabilization`
   
   Includes a warm-up phase to stabilize activations:
   
   - Initial warm-up phase for `warmup_epochs` epochs to stabilize branch voltages
   - Main training phase for remaining epochs
   
   Example configuration:
   
   .. code-block:: yaml
   
       train:
         learning_strategy: voltage_stabilization
         warmup_epochs: 5        # Activation stabilization phase
         stability_target_voltage: 0.5  # Target activation value
         epochs: 50              # Total epochs (including warmup)

Complete Example
~~~~~~~~~~~~~~~

A complete example of a configuration using the freeze_branches_reversed_with_pretraining strategy:

.. code-block:: yaml

    train:
      learning_strategy: freeze_branches_reversed_with_pretraining
      run_name: mnist_einet_freeze_reversed_pretrain
      seed: 42
      train_valid_split: 0.8
      lr: 0.005
      weight_decay_rate: 0.01
      epochs: 60             # Total epochs across all phases
      batch_size: 128
      grad_clip_value: 5.0
      pretrain_epochs: 10    # Initial full-network training
      epochs_per_layer: 20   # Each branch is trained for 20 epochs
      final_tune_epochs: 10  # Final full-network fine-tuning

Task Configuration
-----------------

The `task` section defines the dataset and task-specific parameters:

.. code-block:: yaml

    task:
      dataset: mnist  # Dataset to use
      data_path: ""   # Optional path to data (will use default if empty)
      parameters:
        # Dataset-specific parameters
        transform: "normalize"
        target_transform: ""

Wandb Configuration
------------------

The `wandb` section configures experiment tracking with Weights & Biases:

.. code-block:: yaml

    wandb:
      entity: "your_username"
      project: "dendritic_modeling"
      group: "mnist_experiments"
      tags: ["einet", "mnist", "exploration"]
      use_wandb: true  # Set to false to disable wandb logging

Visualization Configuration
--------------------------

The `visualization` section controls model visualization:

.. code-block:: yaml

    visualization:
      enabled: true
      visualize_epochs: [0, 10, 20, 30, 40]  # Epochs at which to generate visualizations
      plots: ["branch_info", "ablation"]  # Plot types to generate
      parameters:
        # Visualization-specific parameters
        save_figures: true
        figure_format: "png"

Outputs Configuration
-------------------

The `outputs` section defines where experiment results are saved:

.. code-block:: yaml

    outputs:
      dir: "outputs"  # Directory for experiment outputs

Using the Configuration System
-----------------------------

To run an experiment with a specific configuration file:

.. code-block:: bash

    python src/dendritic_modeling/train_experiments.py config_exp.yaml

To override specific parameters at runtime:

.. code-block:: bash

    python src/dendritic_modeling/train_experiments.py config_exp.yaml --train.learning_strategy freeze_branches --train.lr 0.01

This flexible configuration system allows you to:

1. Define a wide range of experiments without code changes
2. Ensure reproducibility by capturing all experiment parameters
3. Easily iterate on hyperparameters
4. Share experiment setups with colleagues

The configuration is fully typed and validated, with detailed error messages if parameters are invalid. 