.. _configuration:

Configuration System
==================

The Dendritic Modeling framework uses a flexible YAML-based configuration system to define experiments. This approach makes it easy to reproduce experiments, track different configurations, and modify parameters without changing code.

Overview
--------

The configuration system is built around the ``train_experiments.py`` script, which takes a YAML configuration file as input and handles:

1. Setting up the specified dataset
2. Creating the model architecture
3. Applying the chosen learning strategy
4. Running training and evaluation
5. Generating visualizations and analysis

Configuration File Structure
---------------------------

A typical configuration file has the following top-level sections:

``model``
~~~~~~~~~

Defines the model architecture and parameters:

.. code-block:: yaml

    model:
      task: classification  # Type of task: classification, regression, etc.
      network:
        type: EINet  # Network type: EINet, MLP, etc.
        parameters:
          input_dim: 784
          excitatory_layer_sizes: [512, 256]
          inhibitory_layer_sizes: [200, 100]
          excitatory_branch_factors: [64, 32]
          inhibitory_branch_factors: [32, 16]
          reactivate: true
          somatic_synapses: true
          # ... other architecture parameters

See :doc:`../network_structures/index` for details on all available model parameters.

``train``
~~~~~~~~~

Defines the training strategy and hyperparameters:

.. code-block:: yaml

    train:
      learning_strategy: mle  # mle, freeze_layers, two_step, etc.
      seed: 42  # Random seed for reproducibility
      lr: 0.001  # Learning rate
      epochs: 20  # Training epochs
      batch_size: 64  # Batch size
      grad_clip_value: 5.0  # Gradient clipping
      # Optional parameters for specific strategies:
      pretrain_epochs: 5  # For two_step strategy
      warmup_epochs: 2  # For voltage_stabilization

See :doc:`../learning/index` for details on all available learning strategies and parameters.

``task``
~~~~~~~~

Defines the dataset and task-specific parameters:

.. code-block:: yaml

    task:
      dataset: mnist  # mnist, cifar10, multi_xor, etc.
      data_path: ""  # Path to data (if needed)
      parameters:
        # Dataset-specific parameters
        train_size: 8000  # For synthetic datasets

Available datasets include:
- Standard datasets: ``mnist``, ``fashion_mnist``, ``cifar10``, ``cifar100``
- Synthetic datasets: ``multi_xor``, ``line``, ``center_surround``, ``orientation_bars``
- Modified datasets: ``mnist_modulo10``, ``random_flip_mnist``

See :doc:`../analysis/synthetic_datasets` for more information on synthetic datasets.

``visualization``
~~~~~~~~~~~~~~~~

Controls visualization and analysis:

.. code-block:: yaml

    visualization:
      enabled: true  # Enable visualization
      visualize_epochs: [10, 20]  # Epochs to visualize
      plots:
        - weights  # Plot weight visualizations
        - activations  # Plot activation distributions
        - branch_ablation  # Run branch ablation studies
        - layer_ablation  # Run layer ablation studies
        - branch_info  # Analyze branch information content
        - gradients  # Plot gradient distributions
        # ... other visualization options

See :doc:`../visualization/index` for details on available visualization options.

``wandb``
~~~~~~~~~

Configuration for Weights & Biases integration:

.. code-block:: yaml

    wandb:
      use_wandb: true  # Enable W&B logging
      entity: your_username  # W&B username
      project: dendritic_modeling  # W&B project name
      tags: [experiment1, mnist]  # Tags for this run

See :doc:`../analysis/experiment_tracking` for more details on experiment tracking.

``outputs``
~~~~~~~~~~

Controls output saving:

.. code-block:: yaml

    outputs:
      dir: "./outputs"  # Directory to save outputs

Running Experiments
------------------

To run an experiment using a configuration file:

.. code-block:: bash

    python src/dendritic_modeling/train_experiments.py path/to/config.yaml

You can override configuration values from the command line:

.. code-block:: bash

    # Override learning rate
    python src/dendritic_modeling/train_experiments.py config.yaml --train.lr 0.0001

    # Change dataset
    python src/dendritic_modeling/train_experiments.py config.yaml --task.dataset cifar10

    # Enable visualization
    python src/dendritic_modeling/train_experiments.py config.yaml --visualization.enabled true

Configuration Templates
----------------------

Several example configuration templates are provided in the ``configs/`` directory:

- ``config_exp.yaml``: Basic configuration for standard classification tasks
- ``sweep_params.yaml``: Example configuration for parameter sweeps

Best Practices
-------------

1. **Version Control**: Keep your configuration files in version control alongside your code
2. **Documentation**: Add comments to your configuration files to explain parameter choices
3. **Parameter Sweeps**: Use parameter sweeps with Weights & Biases to find optimal parameters
4. **Reproducibility**: Set a fixed random seed for reproducible experiments
5. **Modularization**: Create base configuration files and extend them for specific experiments

Advanced Configuration Options
----------------------------

For advanced users, the configuration system supports:

- **Parameter Sweeps**: Running multiple experiments with different parameter values
- **Multi-GPU Training**: Configuring distributed training across multiple GPUs
- **Custom Datasets**: Defining and using custom datasets
- **Experiment Tracking**: Integration with experiment tracking tools

See the example configurations and the API documentation for more details on these advanced options.
