.. _quickstart:

Quickstart
==========

This quickstart guide will help you get up and running with the Dendritic Modeling framework. We'll cover the basics of creating, training, and evaluating a dendritic network model.

Creating a Simple Dendritic Network
-----------------------------------

Let's start by creating a simple dendritic network model for MNIST classification:

.. code-block:: python

    import torch
    from dendritic_modeling.networks import ExcitationInhibitionNetwork
    
    # Define a simple EINet for MNIST
    model = ExcitationInhibitionNetwork(
        input_dim=784,  # 28x28 images flattened
        excitatory_layer_sizes=[10],  # 10 excitatory cells (one per digit)
        inhibitory_layer_sizes=[20],  # 20 inhibitory cells
        excitatory_branch_factors=[2, 2],  # Creating 4 branches per cell (2x2)
        inhibitory_branch_factors=[],  # No branching for inhibitory cells
        ee_synapses_per_branch_per_layer=[24],  # 24 synapses from input to each excitatory branch
        ei_synapses_per_branch_per_layer=[100],  # 100 synapses from input to each inhibitory branch
        ie_synapses_per_branch_per_layer=[2],  # 2 synapses from inhibitory cells to each excitatory branch
        ii_synapses_per_branch_per_layer=[],  # No inhibitory-inhibitory connections
        reactivate=True,  # Use reactivation for branches
        somatic_synapses=True  # Allow somatic connections
    )
    
    # Apply a softmax for classification
    classifier = torch.nn.Sequential(
        model,
        torch.nn.LogSoftmax(dim=-1)
    )

Loading a Dataset
-----------------

Now, let's load the MNIST dataset:

.. code-block:: python

    from dendritic_modeling.utils import load_mnist_as_datasets
    
    # Load MNIST and split into train/validation/test sets
    train_ds, valid_ds, test_ds = load_mnist_as_datasets(train_valid_split=0.8)
    
    # Peek at the data
    x_sample, y_sample = train_ds[0]
    print(f"Input shape: {x_sample.shape}, Label: {y_sample}")

Training the Model
------------------

Now, let's train our model using a standard learning strategy:

.. code-block:: python

    from dendritic_modeling.learning_strategies import TrainerMLE
    
    # Set up optimizer with Adam
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    
    # Initialize the MLE (Maximum Likelihood Estimation) trainer
    trainer = TrainerMLE(optimizer=optimizer)
    
    # Train the model
    results = trainer.train(
        model=classifier,
        train_data=train_ds,
        valid_data=valid_ds,
        grad_clip_value=5.0,
        epochs=10,
        batch_size=64,
        shuffle=True
    )
    
    # Plot training curves
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 5))
    plt.plot(results['train losses'], label='Train Loss')
    plt.plot(results['valid losses'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

Evaluating the Model
--------------------

Let's evaluate our trained model:

.. code-block:: python

    from dendritic_modeling.utils import accuracy_score
    
    # Evaluate on test set
    test_accuracy = accuracy_score(classifier, *test_ds[:])
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Make predictions on a few examples
    x_test, y_test = test_ds[:10]
    with torch.no_grad():
        logits = classifier(x_test)
        predictions = torch.argmax(logits, dim=1)
    
    print("Ground truth:", y_test)
    print("Predictions:", predictions)

Visualizing the Network
-----------------------

The framework provides tools to visualize the dendritic network:

.. code-block:: python

    from dendritic_modeling.plot_utils import plot_einet_params, plot_einet_activations
    
    # Visualize network weights
    plot_einet_params(
        einet=model,
        save_root="./visualizations",
        filename="mnist_model"
    )
    
    # Visualize activations for different inputs
    input_list = [x_test[i:i+1] for i in range(5)]  # First 5 test examples
    plot_einet_activations(
        einet=model,
        input_list=input_list,
        save_root="./visualizations",
        filename="mnist_activations"
    )

Using YAML Configuration
------------------------

For more complex experiments, you can use YAML configuration files:

.. code-block:: python

    from dendritic_modeling.config import load_config
    from dendritic_modeling.utils import get_unified_datasets
    
    # Load configuration
    config = load_config("configs/mnist_einet.yaml")
    
    # Get datasets based on configuration
    datasets = get_unified_datasets(config.task, config.train)
    
    # Create model from configuration
    from dendritic_modeling.networks import create_model_from_config
    
    model = create_model_from_config(config.model)
    
    # Train using configuration parameters
    trainer = TrainerMLE(optimizer=torch.optim.Adam(model.parameters(), lr=config.train.lr))
    results = trainer.train(
        model=model,
        train_data=datasets['train'],
        valid_data=datasets['valid'],
        epochs=config.train.epochs,
        batch_size=config.train.batch_size
    )

Next Steps
----------

Now that you've learned the basics of the Dendritic Modeling framework, you can:

- Explore more complex network architectures with different branch factors and connection patterns
- Try different learning strategies like `TwoStepTrainer` or `LocalCreditAssignment`
- Experiment with synthetic datasets designed to test specific dendritic computational capabilities
- Use experiment tracking with Weights & Biases
- Check out the example notebooks for more advanced use cases

Refer to the detailed documentation for each component for more information. 