.. _examples:

Examples
========

This page contains a collection of example scripts and use cases for the Dendritic Modeling framework. These examples demonstrate the framework's capabilities and provide templates for your own experiments.

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

Synthetic Dataset Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Training a dendritic network on a synthetic dataset designed to test specific computational capabilities:

.. code-block:: python

    from dendritic_modeling.synthetic_datasets import MultiXORDataset
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

Branch Information Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzing information content in dendritic branches:

.. code-block:: python

    from dendritic_modeling.utils import compute_branch_info, record_branch_activations
    
    # Create and train a model (as in previous examples)
    model = ExcitationInhibitionNetwork(...)
    # ... train the model ...
    
    # Record branch activations
    activations, labels = record_branch_activations(
        model=model,
        dataset=test_ds,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Compute branch information
    branch_info = compute_branch_info(
        activations=activations,
        labels=labels,
        n_bins=10  # Number of bins for information calculation
    )
    
    # Plot branch information
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(branch_info['mi'])), branch_info['mi'])
    plt.xlabel('Branch Index')
    plt.ylabel('Mutual Information with Label')
    plt.title('Information Content in Dendritic Branches')
    plt.show()

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
        ablation_mode='direct',  # Ablation mode: 'direct' or 'indirect'
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Extract results
    baseline_acc = results['baseline_acc']
    branch_importances = results['branch_importances']
    
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # Visualize branch importances
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.imshow(branch_importances, cmap='RdBu_r', aspect='auto')
    plt.colorbar(label='Accuracy Drop')
    plt.xlabel('Branch Index')
    plt.ylabel('Layer Index')
    plt.title('Branch Importance (Accuracy Drop)')
    plt.show()

Running Experiments with train_experiments.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides a flexible experiment runner through train_experiments.py. Here's how to use it:

1. **Basic Usage**:
.. code-block:: python

    python src/dendritic_modeling/train_experiments.py config_exp.yaml

2. **Configuration File Structure**:
.. code-block:: yaml

    # config_exp.yaml
    model:
      task: "classification"
      network:
        type: "EINet"
        parameters:
          input_dim: 784
          excitatory_layer_sizes: [512, 256]
          inhibitory_layer_sizes: [200, 100]
          excitatory_branch_factors: [64, 32]
          inhibitory_branch_factors: [32, 16]
          reactivate: true
          somatic_synapses: true
          topk_init_method: "xavier_normal"
          reactivation_type: "param_tanh"
          reactivation_strategy: "inverse"
          blocklinear_strategy: "block_conductance_dynamic"
          enable_branch_outputs: true
          output_layer: false
          output_dim: 10
    
    train:
      learning_strategy: "mle"
      run_name: "mnist_einet"
      seed: 0
      train_valid_split: 0.8
      lr: 0.01
      weight_decay_rate: 0.1
      epochs: 50
      batch_size: 128
      grad_clip_value: 5.0
      epochs_per_layer: 50
      final_tune_epochs: 10
      pretrain_epochs: 50
    
    task:
      dataset: "mnist"
      data_path: ""
      parameters: {}
    
    visualization:
      enabled: false
      visualize_epochs: []
      plots: ["branch_info", "ablation"]
      parameters:
        save_figures: true
        figure_format: "png"
    
    wandb:
      entity: ""
      project: ""
      group: ""
      tags: []
      use_wandb: false
    
    outputs:
      dir: "outputs"

3. **Available Learning Strategies**:
- mle: Standard Maximum Likelihood Estimation (default)
- freeze_branches: Train branches sequentially
- freeze_branches_reversed: Train branches in reverse order
- freeze_branches_with_pretraining: Pretrain then train branches sequentially
- freeze_branches_reversed_with_pretraining: Pretrain then train branches in reverse order
- two_step: Two-phase training with pretraining and main training
- voltage_stabilization: Add voltage stabilization phase

4. **Command Line Arguments**:
.. code-block:: bash

    # Basic usage
    python src/dendritic_modeling/train_experiments.py config_exp.yaml
    
    # Override learning strategy
    python src/dendritic_modeling/train_experiments.py config_exp.yaml --train.learning_strategy freeze_branches
    
    # Enable visualization
    python src/dendritic_modeling/train_experiments.py config_exp.yaml --visualization.enabled true

For more information about specific configuration options, see the :doc:`../advanced/network_structures/index` and :doc:`../advanced/learning/index` sections.

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