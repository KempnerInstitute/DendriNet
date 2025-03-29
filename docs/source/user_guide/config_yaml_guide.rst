.. _config_yaml_guide:

Understanding the Config YAML File
=================================

Overview
--------

The dendritic modeling framework uses YAML configuration files to define experiment parameters. This approach provides flexibility, reproducibility, and makes it easy to run parameter sweeps without modifying code.

This guide explains the structure of the config.yaml file, what each section controls, and how to customize it for your experiments.

Basic Structure
--------------

The configuration file has several main sections:

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

Let's examine each section in detail.

Model Configuration
------------------

The ``model`` section defines the network architecture and parameters:

.. code-block:: yaml

    model:
      task: classification
      probabilistic: false
      network:
        type: EINet  # "EINet" or "MLP"
        parameters:
          # Network architecture parameters
          excitatory_layer_sizes: [512]
          inhibitory_layer_sizes: [200]
          excitatory_branch_factors: [64]
          inhibitory_branch_factors: []
          
          # Connection parameters
          ee_synapses_per_branch_per_layer: [24]
          ei_synapses_per_branch_per_layer: [200]
          ie_synapses_per_branch_per_layer: [2]
          ii_synapses_per_branch_per_layer: [0]
          
          # Activation parameters
          reactivate: true
          somatic_synapses: true
          topk_init_method: xavier_normal
          use_shunting: true
          
          # Reactivation settings
          reactivation_type: param_tanh
          reactivation_strategy: inverse
          blocklinear_strategy: block_conductance_dynamic
          
          # Output settings
          enable_branch_outputs: true
          output_layer: false
          output_dim: 10
          
          # Input processing
          input_mode: 0  # 0=original, 1=exc=x+inh=x, 2=input_net
          input_net_dict:
            type: "MLPInputTransform"
            hidden_dims: [512, 256]
            output_dim: 128

Key parameters include:

* **task**: Type of task ("classification")
* **probabilistic**: Whether to use probabilistic classification
* **network.type**: Network architecture ("EINet" or "MLP")
* **excitatory_layer_sizes**: Number of excitatory cells in each layer
* **inhibitory_layer_sizes**: Number of inhibitory cells in each layer
* **excitatory_branch_factors**: Branching structure for excitatory dendrites
* **inhibitory_branch_factors**: Branching structure for inhibitory dendrites
* **synapses_per_branch_per_layer**: Connectivity parameters for different neuron types
* **reactivation_type**: Type of reactivation function ("param_tanh", "relu", "tanh", etc.)
* **reactivation_strategy**: Strategy for reactivation ("inverse" or "none")
* **input_mode**: Method for handling inputs (0=direct, 1=exc=inh=input, 2=through input network)
* **input_net_dict**: Configuration for input transformation network (used if input_mode=2)

Train Configuration
------------------

The ``train`` section defines the training process and hyperparameters:

.. code-block:: yaml

    train:
      learning_strategy: mle
      seed: 42
      train_valid_split: 0.8
      lr: 0.001
      weight_decay_rate: 0.01
      epochs: 50
      batch_size: 64
      grad_clip_value: 1.0
      
      # For branch-wise training strategies
      epochs_per_layer: 20
      final_tune_epochs: 10
      
      # For strategies with pretraining
      pretrain_epochs: 10
      
      # For voltage stabilization
      warmup_epochs: 5
      stability_target_voltage: 0.5
      
      # Input network training
      input_net_training:
        training_mode: "sequential"
        train_input_net: true
        input_net_epochs: 20
        train_heads_independently: false

Key parameters include:

* **learning_strategy**: Learning algorithm (see below for options)
* **seed**: Random seed for reproducibility
* **lr**, **weight_decay_rate**: Optimizer parameters
* **epochs**, **batch_size**: Basic training parameters
* **epochs_per_layer**, **final_tune_epochs**: Parameters for branch-wise training
* **pretrain_epochs**: Parameter for strategies with initial full-network training
* **warmup_epochs**, **stability_target_voltage**: Parameters for voltage stabilization
* **input_net_training**: Configuration for input network training

Available Learning Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework implements several learning strategies:

1. **mle** (Maximum Likelihood Estimation):
   
   Standard end-to-end training where all parameters are updated simultaneously.

2. **freeze_branches**:
   
   Trains branches sequentially from soma to distal branches.
   
   .. code-block:: yaml
   
       train:
         learning_strategy: freeze_branches
         epochs_per_layer: 20
         final_tune_epochs: 10

3. **freeze_branches_reversed**:
   
   Trains branches sequentially from distal branches to soma.
   
   .. code-block:: yaml
   
       train:
         learning_strategy: freeze_branches_reversed
         epochs_per_layer: 20
         final_tune_epochs: 10

4. **freeze_branches_with_pretraining**:
   
   Includes pretraining phase before branch-by-branch training.
   
   .. code-block:: yaml
   
       train:
         learning_strategy: freeze_branches_with_pretraining
         pretrain_epochs: 10
         epochs_per_layer: 20
         final_tune_epochs: 10

5. **freeze_branches_reversed_with_pretraining**:
   
   Includes pretraining phase before reversed branch-by-branch training.
   
   .. code-block:: yaml
   
       train:
         learning_strategy: freeze_branches_reversed_with_pretraining
         pretrain_epochs: 10
         epochs_per_layer: 20
         final_tune_epochs: 10

6. **two_step**:
   
   Two-phase training approach.
   
   .. code-block:: yaml
   
       train:
         learning_strategy: two_step
         pretrain_epochs: 10
         epochs: 50

7. **voltage_stabilization**:
   
   Includes a warm-up phase to stabilize activations.
   
   .. code-block:: yaml
   
       train:
         learning_strategy: voltage_stabilization
         warmup_epochs: 5
         stability_target_voltage: 0.5
         epochs: 50

Task Configuration
-----------------

The ``task`` section defines the dataset and task-specific parameters:

.. code-block:: yaml

    task:
      dataset: mnist
      data_path: ""  # Optional path to data (will use default if empty)
      parameters:
        # Dataset-specific parameters
        transform: "normalize"
        target_transform: ""

Available datasets include:

* **mnist**: Standard MNIST digit recognition
* **cifar10**: CIFAR-10 image classification
* **random_flip_mnist**: MNIST with random flip transformations
* **mnist_modulo10**: MNIST with modulo operation for label transformations

Wandb Configuration
------------------

The ``wandb`` section configures experiment tracking with Weights & Biases:

.. code-block:: yaml

    wandb:
      entity: "your_username"
      project: "dendritic_modeling"
      group: "mnist_experiments"
      tags: ["einet", "mnist", "exploration"]
      use_wandb: true  # Set to false to disable wandb logging

Visualization Configuration
--------------------------

The ``visualization`` section controls model visualization:

.. code-block:: yaml

    visualization:
      enabled: true
      visualize_epochs: [0, 10, 20, 30, 40]  # Epochs at which to visualize
      plots:
        - weights
        - activations
        - branch_info
        - branch_ablation
        - layer_ablation
      parameters:
        # Visualization-specific parameters
        save_figures: true
        figure_format: "png"

Available plot types:

* **weights**: Visualize network weights
* **activations**: Visualize neuron activations
* **branch_info**: Analyze branch-specific information
* **branch_ablation**: Analyze the effect of ablating individual branches
* **layer_ablation**: Analyze the effect of ablating whole layers

Outputs Configuration
-------------------

The ``outputs`` section defines where experiment results are saved:

.. code-block:: yaml

    outputs:
      dir: "outputs"  # Directory for experiment outputs

Complete Example Configuration
-----------------------------

Here's a complete example configuration for an MNIST experiment:

.. code-block:: yaml

    model:
      task: classification
      probabilistic: true
      network:
        type: EINet
        parameters:
          input_dim: 784
          excitatory_layer_sizes: [512]
          inhibitory_layer_sizes: [200]
          excitatory_branch_factors: [64]
          inhibitory_branch_factors: []
          ee_synapses_per_branch_per_layer: [24]
          ei_synapses_per_branch_per_layer: [200]
          ie_synapses_per_branch_per_layer: [1]
          ii_synapses_per_branch_per_layer: []
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
          input_mode: 0

    train:
      learning_strategy: mle
      run_name: mnist_einet
      seed: 42
      train_valid_split: 0.8
      lr: 0.001
      weight_decay_rate: 0.01
      epochs: 50
      batch_size: 128
      grad_clip_value: 1.0
      epochs_per_layer: 20
      final_tune_epochs: 10
      pretrain_epochs: 10

    task:
      dataset: mnist
      data_path: ""
      parameters: {}

    wandb:
      entity: ""
      project: dendritic_modeling
      group: mnist_experiments
      tags: [einet, mnist]
      use_wandb: true

    visualization:
      enabled: true
      visualize_epochs: [1, 25, 50]
      plots:
        - branch_info
        - branch_ablation

    outputs:
      dir: outputs

Tips for Configuration
--------------------

1. **Reproducibility**:
   - Always set a fixed seed for reproducible results
   - Save your configuration files with your experiment results

2. **Parameter Sweeps**:
   - Create multiple configuration files or use command-line overrides
   - For larger sweeps, use the sweep functionality with wandb

3. **Network Architecture**:
   - Start with simpler networks (fewer layers, smaller branch factors)
   - Gradually increase complexity as needed

4. **Training Strategy**:
   - Try different learning strategies based on your task
   - For complex tasks, branch-wise strategies often work better

5. **Custom Datasets**:
   - For custom datasets, specify the data_path
   - Add any required dataset-specific parameters

Using the Configuration System
-----------------------------

To run an experiment with a specific configuration file:

.. code-block:: bash

    python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml

To override specific parameters at runtime:

.. code-block:: bash

    python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml \
      --train.learning_strategy freeze_branches \
      --train.lr 0.01

This flexible configuration system allows you to:
- Define a wide range of experiments without code changes
- Ensure reproducibility by capturing all experiment parameters
- Easily iterate on hyperparameters
- Share experiment setups with colleagues 