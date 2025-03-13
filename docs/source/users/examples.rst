Examples
========

This page contains a collection of example scripts and use cases for the Dendritic Modeling framework. These examples demonstrate the framework's capabilities and provide templates for your own experiments.

Basic Examples
------------

MNIST Classification
~~~~~~~~~~~~~~~~~~

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
        excitatory_layer_sizes=[10],
        inhibitory_layer_sizes=[20],
        excitatory_branch_factors=[2, 2],
        inhibitory_branch_factors=[],
        ee_synapses_per_branch_per_layer=[24],
        ei_synapses_per_branch_per_layer=[100],
        ie_synapses_per_branch_per_layer=[2],
        ii_synapses_per_branch_per_layer=[]
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
~~~~~~~~~~~~~~~~~~~~~~~

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
        excitatory_layer_sizes=[2],  # 2 output classes (0 or 1)
        inhibitory_layer_sizes=[4],
        excitatory_branch_factors=[2, 2],  # 4 branches per cell
        inhibitory_branch_factors=[],
        ee_synapses_per_branch_per_layer=[4],  # Each branch connects to 4 inputs
        ei_synapses_per_branch_per_layer=[6],
        ie_synapses_per_branch_per_layer=[2],
        ii_synapses_per_branch_per_layer=[]
    )
    
    # Continue with training as in the MNIST example...

Advanced Examples
---------------

Two-Step Training
~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~

Using an input transformation network for better feature extraction:

.. code-block:: python

    # Create an EINet with input transformation
    model = ExcitationInhibitionNetwork(
        input_dim=784,
        excitatory_layer_sizes=[10],
        inhibitory_layer_sizes=[20],
        excitatory_branch_factors=[2, 2],
        inhibitory_branch_factors=[],
        ee_synapses_per_branch_per_layer=[24],
        ei_synapses_per_branch_per_layer=[100],
        ie_synapses_per_branch_per_layer=[2],
        ii_synapses_per_branch_per_layer=[],
        input_mode=2,  # Use input transformation network
        input_net_dict={
            "type": "MLPInputTransform",
            "hidden_dims": [512, 256],
            "output_dim": 128
        }
    )
    
    # Configure input network training
    from dendritic_modeling.config import Config, TrainConfig
    
    train_config = TrainConfig(
        input_net_training={
            "training_mode": "sequential",
            "train_input_net": True,
            "input_net_epochs": 10,
            "train_heads_independently": True,
            "inhibitory_head_objective": "diversity"
        }
    )
    
    # Use appropriate trainer for this configuration...

Branch Information Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

Analyzing information content in dendritic branches:

.. code-block:: python

    from dendritic_modeling.utils import analyze_branch_information
    
    # Create and train a model (as in previous examples)
    model = ExcitationInhibitionNetwork(...)
    # ... train the model ...
    
    # Analyze branch information
    branch_info = analyze_branch_information(
        model=model,
        dataset=test_ds,
        batch_size=64
    )
    
    # Plot branch information
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.bar(branch_info.keys(), branch_info.values())
    plt.xlabel('Branch Layer')
    plt.ylabel('Mutual Information with Label')
    plt.title('Information Content in Dendritic Branches')
    plt.show()

Ablation Studies
~~~~~~~~~~~~~~

Performing ablation studies to understand branch importance:

.. code-block:: python

    from dendritic_modeling.utils import analyze_branch_ablation
    
    # Create and train a model (as in previous examples)
    model = ExcitationInhibitionNetwork(...)
    # ... train the model ...
    
    # Analyze branch ablation
    baseline_acc, results = analyze_branch_ablation(
        model=model,
        dataset=test_ds,
        layer_idx=0,  # First branch layer
        branch_ids=[0, 1, 2, 3]  # Analyze each branch separately
    )
    
    print(f"Baseline accuracy: {baseline_acc:.4f}")
    for branch_id, (acc, delta) in results.items():
        print(f"Branch {branch_id}: Accuracy drop = {delta:.4f}")

Configuration-Based Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Running an experiment with YAML configuration:

.. code-block:: python

    from dendritic_modeling.config import load_config
    from dendritic_modeling.utils import get_unified_datasets
    
    # Load configuration
    config = load_config("configs/experiment_config.yaml")
    
    # Get datasets
    datasets = get_unified_datasets(config.task, config.train)
    
    # Initialize model from config
    from dendritic_modeling.networks import ExcitationInhibitionNetwork
    
    net_params = config.model.network.parameters
    model = ExcitationInhibitionNetwork(
        input_dim=datasets['train'][0][0].shape[1],
        excitatory_layer_sizes=net_params.excitatory_layer_sizes,
        inhibitory_layer_sizes=net_params.inhibitory_layer_sizes,
        excitatory_branch_factors=net_params.excitatory_branch_factors,
        inhibitory_branch_factors=net_params.inhibitory_branch_factors,
        ee_synapses_per_branch_per_layer=net_params.ee_synapses_per_branch_per_layer,
        ei_synapses_per_branch_per_layer=net_params.ei_synapses_per_branch_per_layer,
        ie_synapses_per_branch_per_layer=net_params.ie_synapses_per_branch_per_layer,
        ii_synapses_per_branch_per_layer=net_params.ii_synapses_per_branch_per_layer,
        reactivate=net_params.reactivate,
        somatic_synapses=net_params.somatic_synapses,
        input_mode=net_params.input_mode,
        input_net_dict=net_params.input_net_dict
    )
    
    # Get appropriate trainer from config
    from dendritic_modeling.learning_strategies import get_trainer
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.train.lr)
    trainer = get_trainer(
        strategy=config.train.learning_strategy,
        optimizer=optimizer
    )
    
    # Train
    results = trainer.train(
        model=model,
        train_data=datasets['train'],
        valid_data=datasets['valid'],
        epochs=config.train.epochs,
        batch_size=config.train.batch_size,
        grad_clip_value=config.train.grad_clip_value
    )
    
    # Save results
    import os
    import json
    
    os.makedirs(config.outputs.dir, exist_ok=True)
    with open(os.path.join(config.outputs.dir, "results.json"), "w") as f:
        json.dump({
            "test_accuracy": accuracy_score(model, *datasets['test'][:]),
            "train_losses": [float(loss) for loss in results["train losses"]],
            "valid_losses": [float(loss) for loss in results["valid losses"]]
        }, f)

Notebook Examples
--------------

For additional examples, explore the Jupyter notebooks in the `examples` directory of the repository:

1. `01_mnist_classification.ipynb` - Basic classification with dendritic networks
2. `02_synthetic_tasks.ipynb` - Working with synthetic datasets
3. `03_branch_visualization.ipynb` - Visualizing dendritic branches and their activity
4. `04_ablation_studies.ipynb` - Performing ablation studies to understand network behavior
5. `05_comparison_with_mlp.ipynb` - Comparing dendritic networks with standard MLPs
6. `06_advanced_training_strategies.ipynb` - Using specialized learning strategies

These notebooks provide executable examples with detailed explanations and visualizations. 