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
      learning_strategy: mle  # "mle", "freeze_layers", "local_credit_assignment", "two_step", "two_step_kl"
      run_name: mnist_einet
      seed: 42
      train_valid_split: 0.8
      lr: 0.001
      weight_decay_rate: 0.01
      epochs: 50
      batch_size: 64
      grad_clip_value: 1.0
      
      # For specialized learning strategies
      epochs_per_layer: 10  # For "freeze_layers"
      pretrain_epochs: 5    # For "two_step"
      
      # Input network training configuration
      input_net_training:
        training_mode: "sequential"  # "joint" or "sequential"
        train_input_net: true
        input_net_epochs: 20
        train_heads_independently: false
        inhibitory_head_objective: "classification"
        diversity_weight: 0.5
        pretrained_options:
          source: "file"
          weights_path: ""
          pytorch_model: ""

Key parameters include:

- **learning_strategy**: Algorithm for training the network
- **lr**, **batch_size**, **epochs**: Standard training hyperparameters
- **input_net_training**: Configuration for input transformation networks

Task Configuration
------------------

The `task` section defines the dataset and task-specific parameters:

.. code-block:: yaml

    task:
      dataset: mnist  # "mnist", "cifar10", or synthetic datasets
      data_path: "/path/to/data"  # Optional custom path
      parameters:
        # For synthetic datasets
        train_size: 8000
        valid_size: 1000
        test_size: 1000
        image_size: 16

For synthetic datasets, additional parameters may be required based on the dataset type.

Experiment Tracking
-------------------

The `wandb` section configures Weights & Biases integration:

.. code-block:: yaml

    wandb:
      entity: "your-username"
      project: "dendritic-experiments"
      group: "mnist-comparison"
      tags: ["dendritic", "mnist", "experiment1"]
      use_wandb: true

Set `use_wandb: false` to disable Weights & Biases tracking.

Visualization Configuration
---------------------------

The `visualization` section controls the generation of visualizations:

.. code-block:: yaml

    visualization:
      enabled: true
      visualize_epochs: [10, 30, 50]  # At which epochs to generate visualizations
      plots:
        - weights      # Plot dendritic weights
        - activations  # Plot activation distributions
        - gradients    # Plot gradient distributions
        - branch_info  # Plot mutual information in branches
        - ablation     # Perform and plot ablation studies

Output Configuration
--------------------

The `outputs` section configures where results are saved:

.. code-block:: yaml

    outputs:
      dir: "results/experiment1"

Loading Configurations in Code
------------------------------

To use configurations in your code:

.. code-block:: python

    from dendritic_modeling.config import load_config
    
    # Load the configuration
    config = load_config("path/to/config.yaml")
    
    # Access configuration values
    learning_rate = config.train.lr
    batch_size = config.train.batch_size
    
    # Create model from configuration
    from dendritic_modeling.networks import ExcitationInhibitionNetwork
    
    net_params = config.model.network.parameters
    model = ExcitationInhibitionNetwork(
        input_dim=784,  # This might come from your dataset
        excitatory_layer_sizes=net_params.excitatory_layer_sizes,
        inhibitory_layer_sizes=net_params.inhibitory_layer_sizes,
        excitatory_branch_factors=net_params.excitatory_branch_factors,
        inhibitory_branch_factors=net_params.inhibitory_branch_factors,
        ee_synapses_per_branch_per_layer=net_params.ee_synapses_per_branch_per_layer,
        ei_synapses_per_branch_per_layer=net_params.ei_synapses_per_branch_per_layer,
        ie_synapses_per_branch_per_layer=net_params.ie_synapses_per_branch_per_layer,
        ii_synapses_per_branch_per_layer=net_params.ii_synapses_per_branch_per_layer,
        reactivate=net_params.reactivate,
        somatic_synapses=net_params.somatic_synapses
    )

Parameter Sweeps
----------------

For parameter sweeps, you can use the provided utility scripts:

.. code-block:: bash

    python scripts/generate_sweep_configs.py --base-config configs/base_config.yaml --sweep-params configs/sweep_params.yaml --output-dir configs/sweep

A sweep parameter file might look like:

.. code-block:: yaml

    # sweep_params.yaml
    model.network.parameters.excitatory_branch_factors:
      - [1, 2]
      - [2, 2]
      - [2, 4]
    
    train.lr:
      - 0.0001
      - 0.001
      - 0.01
    
    train.batch_size:
      - 32
      - 64
      - 128

This would generate 27 configuration files (3 branch factors × 3 learning rates × 3 batch sizes) in the specified output directory.

Best Practices
--------------

1. **Version Control**: Keep configuration files in version control to track experiment changes
2. **Comments**: Add comments to complex parameters to explain their purpose
3. **Defaults**: Use reasonable defaults in your base configurations
4. **Validation**: The framework validates configurations, but check for logical consistency
5. **Reproducibility**: Set the random seed in the configuration for reproducible results
6. **Modularity**: Create base configurations that can be extended for specific experiments

Example Configurations
----------------------

The framework includes several example configurations in the `configs` directory:

- `mnist_einet.yaml`: Basic EINet for MNIST classification
- `cifar10_einet.yaml`: EINet for CIFAR-10 classification
- `synthetic_tasks.yaml`: Configuration for synthetic datasets
- `ablation_study.yaml`: Setup for branch ablation experiments

You can use these as starting points for your own experiments. 