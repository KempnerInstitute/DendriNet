.. _examples:

Examples
========

This page contains a collection of example scripts and use cases for the Dendritic Modeling framework. These examples demonstrate the framework's capabilities and provide templates for your own experiments.

.. note::
   While these examples show how to use the low-level API directly, we recommend using the configuration-based approach with ``train_experiments.py`` for most experiments. See the :doc:`quickstart` guide for a step-by-step tutorial on using configurations.

Basic Examples
--------------

MNIST Classification
~~~~~~~~~~~~~~~~~~~~

A simple example of training a dendritic network on MNIST:

.. code-block:: python

    import torch
    from dendritic_modeling.networks import ExcitationInhibitionNetwork
    from dendritic_modeling.learning_strategies import TrainerMLE
    from dendritic_modeling.utils import load_mnist_as_datasets, accuracy_score

    # Load MNIST dataset
    train_ds, valid_ds, test_ds = load_mnist_as_datasets(train_valid_split=0.8)

    # Create model
    model = ExcitationInhibitionNetwork(
        input_dim=784,
        excitatory_layer_sizes=[512, 256],
        inhibitory_layer_sizes=[200, 100],
        excitatory_branch_factors=[64, 32],
        inhibitory_branch_factors=[32, 16],
        reactivate=True,
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        reactivation_type="param_tanh",
        reactivation_strategy="inverse",
        blocklinear_strategy="block_conductance_dynamic",
        enable_branch_outputs=True,
        output_layer=False,
        output_dim=10
    )

    # Create classifier
    classifier = torch.nn.Sequential(
        model,
        torch.nn.LogSoftmax(dim=-1)
    )

    # Train model
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    trainer = TrainerMLE(optimizer=optimizer)

    results = trainer.train(
        model=classifier,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=20,
        batch_size=64
    )

    # Evaluate on test set
    test_accuracy = accuracy_score(classifier, *test_ds[:])
    print(f"Test accuracy: {test_accuracy:.4f}")

**Equivalent Configuration Approach:**

The above code could be expressed in a configuration file (``mnist_config.yaml``) like this:

.. code-block:: yaml

    model:
      task: classification
      network:
        type: EINet
        parameters:
          input_dim: 784
          excitatory_layer_sizes: [512, 256]
          inhibitory_layer_sizes: [200, 100]
          excitatory_branch_factors: [64, 32]
          inhibitory_branch_factors: [32, 16]
          reactivate: true
          somatic_synapses: true
          topk_init_method: xavier_normal
          reactivation_type: param_tanh
          reactivation_strategy: inverse
          blocklinear_strategy: block_conductance_dynamic
          enable_branch_outputs: true
          output_layer: false
          output_dim: 10

    train:
      learning_strategy: mle
      lr: 0.001
      epochs: 20
      batch_size: 64

    task:
      dataset: mnist

    outputs:
      dir: "./outputs"

Then run with:

.. code-block:: bash

    python src/dendritic_modeling/scripts/train_experiments.py mnist_config.yaml

Synthetic Dataset Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Training a dendritic network on a synthetic dataset designed to test specific computational capabilities:

.. code-block:: python

    from dendritic_modeling.datasets import MultiXORDataset
    from torch.utils.data import random_split

    # Create a Multi-XOR dataset
    dataset = MultiXORDataset(n_samples=10000, n_bits=8)

    # Split into train/valid/test
    train_size = int(0.7 * len(dataset))
    valid_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_ds, valid_ds, test_ds = random_split(dataset, [train_size, valid_size, test_size])

    # Create a model with suitable architecture for XOR tasks
    model = ExcitationInhibitionNetwork(
        input_dim=8,  # 8 input bits
        excitatory_layer_sizes=[16, 8],
        inhibitory_layer_sizes=[12, 6],
        excitatory_branch_factors=[4, 2],
        inhibitory_branch_factors=[3, 2],
        reactivate=True,
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        reactivation_type="param_tanh",
        reactivation_strategy="inverse",
        blocklinear_strategy="block_conductance_dynamic",
        enable_branch_outputs=True,
        output_layer=False,
        output_dim=2  # 2 output classes (0 or 1)
    )

    # Continue with training as in the MNIST example...

**Equivalent Configuration Approach:**

.. code-block:: yaml

    model:
      task: classification
      network:
        type: EINet
        parameters:
          input_dim: 8
          excitatory_layer_sizes: [16, 8]
          inhibitory_layer_sizes: [12, 6]
          excitatory_branch_factors: [4, 2]
          inhibitory_branch_factors: [3, 2]
          reactivate: true
          somatic_synapses: true
          # ... other parameters as above

    task:
      dataset: multi_xor
      parameters:
        n_bits: 8
        n_samples: 10000

Advanced Examples
-----------------

Two-Step Training
~~~~~~~~~~~~~~~~~

Using the two-step training strategy for more complex networks:

.. code-block:: python

    from dendritic_modeling.learning_strategies import TwoStepTrainer

    # Create model (as in previous examples)
    model = ExcitationInhibitionNetwork(...)

    # Initialize two-step trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TwoStepTrainer(optimizer=optimizer)

    # Train with two-step approach
    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        pretrain_epochs=5,  # First phase with simplified learning
        maintrain_epochs=15,  # Second phase with full learning
        batch_size=64
    )

**Equivalent Configuration Approach:**

.. code-block:: yaml

    train:
      learning_strategy: two_step
      lr: 0.001
      epochs: 15  # maintrain_epochs
      pretrain_epochs: 5
      batch_size: 64

For more details on training strategies, see :doc:`../advanced/learning/index`.

Input Transformation Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using an input transformation network for better feature extraction:

.. code-block:: python

    # Create an EINet with input transformation
    model = ExcitationInhibitionNetwork(
        input_dim=784,
        excitatory_layer_sizes=[512, 256],
        inhibitory_layer_sizes=[200, 100],
        excitatory_branch_factors=[64, 32],
        inhibitory_branch_factors=[32, 16],
        reactivate=True,
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        reactivation_type="param_tanh",
        reactivation_strategy="inverse",
        blocklinear_strategy="block_conductance_dynamic",
        input_mode=2,  # Use input transformation network
        input_net_dict={
            "type": "MLPInputTransform",
            "hidden_dims": [512, 256],
            "output_dim": 128
        },
        enable_branch_outputs=True,
        output_layer=False,
        output_dim=10
    )

    # Train the model with appropriate learning strategy
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = TrainerMLE(optimizer=optimizer)

    results = trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=20,
        batch_size=64
    )

**Equivalent Configuration Approach:**

.. code-block:: yaml

    model:
      network:
        parameters:
          # ... other parameters
          input_mode: 2
          input_net_dict:
            type: "MLPInputTransform"
            hidden_dims: [512, 256]
            output_dim: 128

For more details on network architectures and input transformations, see :doc:`../advanced/network_structures/index`.

Branch Information Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzing information content in dendritic branches:

.. code-block:: python

    from dendritic_modeling.utils import analyze_dendritic_information

    # Create and train a model (as in previous examples)
    model = ExcitationInhibitionNetwork(...)
    # ... train the model ...

    # Analyze branch information
    branch_info = analyze_dendritic_information(
        model=model,
        dataset=test_ds,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Plot branch information (example: mutual information)
    import matplotlib.pyplot as plt

    # Check if results are available and have the expected structure
    if "mutual_information" in branch_info and branch_info["mutual_information"] is not None:
        mi_values = branch_info["mutual_information"] # Assuming MI is a flat list or array
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(mi_values)), mi_values)
        plt.xlabel('Branch Index (across all layers)')
        plt.ylabel('Mutual Information with Label')
        plt.title('Information Content in Dendritic Branches')
        plt.show()
    else:
        print("Mutual information data not found in analysis results.")

**Equivalent Configuration Approach:**

To enable information analysis in the configuration:

.. code-block:: yaml

    visualization:
      enabled: true
      plots:
        - branch_info

For more details on information analysis, see :doc:`../advanced/analysis/branch_information`.

Ablation Studies
~~~~~~~~~~~~~~~~

Performing ablation studies to understand branch importance:

.. code-block:: python

    from dendritic_modeling.utils import analyze_branch_ablation

    # Create and train a model (as in previous examples)
    model = ExcitationInhibitionNetwork(...)
    # ... train the model ...

    # Analyze branch ablation
    results = analyze_branch_ablation(
        model=model,
        dataset=test_ds,
        mode='direct',  # Ablation mode: 'direct' or 'indirect'
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Extract results
    baseline_acc, branch_results = results

    print(f"Baseline accuracy: {baseline_acc:.4f}")

    # Calculate branch importance from results
    branch_importances = {}
    for branch_id, (acc, delta) in branch_results.items():
        branch_importances[branch_id] = delta

    # Visualize branch importances
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    branches = list(branch_importances.keys())
    importance_values = list(branch_importances.values())
    plt.bar(branches, importance_values)
    plt.xlabel('Branch Index')
    plt.ylabel('Accuracy Drop')
    plt.title('Branch Importance (Accuracy Drop)')
    plt.show()

**Equivalent Configuration Approach:**

To enable ablation studies in the configuration:

.. code-block:: yaml

    visualization:
      enabled: true
      plots:
        - branch_ablation
        - layer_ablation

For more details on ablation studies, see :doc:`../advanced/analysis/ablation_studies`.

Configuration File Structure
---------------------------

The configuration file used with ``train_experiments.py`` has several top-level sections:

``model``
~~~~~~~~~

Defines the model architecture and parameters:

.. code-block:: yaml

    model:
      task: classification  # Type of task: classification, regression, etc.
      network:
        type: EINet  # Network type: EINet, MLP, etc.
        parameters:
          # Network-specific parameters
          excitatory_layer_sizes: [512, 256]
          inhibitory_layer_sizes: [200, 100]
          # ... other architecture parameters

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
      # ... other training parameters

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
        # ... other visualization options

``wandb``
~~~~~~~~~

Configuration for Weights & Biases integration:

.. code-block:: yaml

    wandb:
      use_wandb: true  # Enable W&B logging
      entity: your_username  # W&B username
      project: dendritic_modeling  # W&B project name
      tags: [experiment1, mnist]  # Tags for this run

``outputs``
~~~~~~~~~~

Controls output saving:

.. code-block:: yaml

    outputs:
      dir: "./outputs"  # Directory to save outputs

Running Experiments with train_experiments.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides a flexible experiment runner through ``train_experiments.py``. Here's how to use it:

**Basic Usage**:

.. code-block:: bash

    python src/dendritic_modeling/scripts/train_experiments.py config_exp.yaml

**Overriding Configuration Values**:

.. code-block:: bash

    # Override learning strategy
    python src/dendritic_modeling/scripts/train_experiments.py config_exp.yaml --train.learning_strategy freeze_branches

    # Enable visualization
    python src/dendritic_modeling/scripts/train_experiments.py config_exp.yaml --visualization.enabled true

For more information about specific configuration options, see:
- :doc:`../advanced/network_structures/index` for network architecture options
- :doc:`../advanced/learning/index` for learning strategies
- :doc:`../advanced/analysis/index` for analysis tools
- :doc:`../advanced/visualization/visualization_guide` for visualization options

Notebook Examples
-----------------

For additional examples, explore the Jupyter notebooks in the `examples` directory of the repository:

1. `01_mnist_classification.ipynb` - Basic classification with dendritic networks
2. `02_synthetic_tasks.ipynb` - Working with synthetic datasets
3. `03_branch_visualization.ipynb` - Visualizing dendritic branches and their activity
4. `04_ablation_studies.ipynb` - Performing ablation studies to understand network behavior
5. `05_comparison_with_mlp.ipynb` - Comparing dendritic networks with standard MLPs
6. `06_advanced_training_strategies.ipynb` - Using specialized learning strategies

These notebooks provide executable examples with detailed explanations and visualizations.
