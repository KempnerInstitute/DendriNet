.. _quickstart:

Quickstart
==========

This quickstart guide will help you get up and running with the Dendritic Modeling framework. We'll cover the basics of creating, training, and evaluating a dendritic network model using both the programmatic API and the recommended configuration-based approach.

.. note::
   The recommended way to run experiments is using the configuration-based approach with ``train_experiments.py``. We cover both approaches to help you understand the components, but for production use, the config approach offers better reproducibility and experiment tracking.

Option 1: Configuration-based Approach (Recommended)
---------------------------------------------------

The most straightforward way to run experiments is using a YAML configuration file with the ``train_experiments.py`` script.

Step 1: Create a Configuration File
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Create a file named ``my_mnist_config.yaml`` with the following content:

.. code-block:: yaml

    model:
      task: classification
      network:
        type: EINet
        parameters:
          input_dim: 784  # For MNIST (28x28 pixels)
          excitatory_layer_sizes: [512, 256]
          inhibitory_layer_sizes: [200, 100]
          excitatory_branch_factors: [4, 4]
          inhibitory_branch_factors: [2, 2]
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
      seed: 42
      train_valid_split: 0.8
      lr: 0.001
      epochs: 20
      batch_size: 64
      grad_clip_value: 5.0

    task:
      dataset: mnist
      data_path: ""  # Default location
      parameters: {}

    visualization:
      enabled: true
      plots:
        - weights
        - activations
        - branch_ablation

    wandb:
      use_wandb: false

    outputs:
      dir: "./outputs"

Step 2: Run the Experiment
~~~~~~~~~~~~~~~~~~~~~~~~~

Run the experiment with:

.. code-block:: bash

    python src/dendritic_modeling/train_experiments.py my_mnist_config.yaml

This will:
1. Load the MNIST dataset
2. Create the model as specified in the config
3. Train the model for the specified number of epochs
4. Generate visualizations if enabled
5. Save the model and results to the output directory

Step 3: View the Results
~~~~~~~~~~~~~~~~~~~~~~

Check the output directory for:
- Saved model checkpoints
- Training logs
- Visualizations (if enabled)
- Evaluation metrics

You can adjust the configuration parameters to experiment with different architectures and training strategies. See :doc:`../advanced/configuration/index` for a complete guide to configuration options.

Option 2: Programmatic API Approach
----------------------------------

If you prefer to have more control or want to integrate with existing code, you can use the programmatic API.

Creating a Simple Dendritic Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's start by creating a simple dendritic network model for MNIST classification:

.. code-block:: python

    import torch
    from dendritic_modeling.networks import ExcitationInhibitionNetwork

    # Define a simple EINet for MNIST
    model = ExcitationInhibitionNetwork(
        input_dim=784,  # 28x28 images flattened
        excitatory_layer_sizes=[512, 256],
        inhibitory_layer_sizes=[200, 100],
        excitatory_branch_factors=[4, 4],  # 16 branches per cell
        inhibitory_branch_factors=[2, 2],  # 4 branches per inhibitory cell
        reactivate=True,  # Use reactivation for branches
        somatic_synapses=True,  # Allow somatic connections
        topk_init_method="xavier_normal",
        reactivation_type="param_tanh",
        reactivation_strategy="inverse",
        blocklinear_strategy="block_conductance_dynamic",
        enable_branch_outputs=True,
        output_dim=10
    )

    # Apply a softmax for classification
    classifier = torch.nn.Sequential(
        model,
        torch.nn.LogSoftmax(dim=-1)
    )

Loading a Dataset
~~~~~~~~~~~~~~~~

Now, let's load the MNIST dataset:

.. code-block:: python

    from dendritic_modeling.utils import load_mnist_as_datasets

    # Load MNIST and split into train/validation/test sets
    train_ds, valid_ds, test_ds = load_mnist_as_datasets(train_valid_split=0.8)

    # Peek at the data
    x_sample, y_sample = train_ds[0]
    print(f"Input shape: {x_sample.shape}, Label: {y_sample}")

Training the Model
~~~~~~~~~~~~~~~~

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
        epochs=20,
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
~~~~~~~~~~~~~~~~~~

Let's evaluate our trained model:

.. code-block:: python

    from dendritic_modeling.utils import accuracy_score

    # Evaluate on test set
    test_accuracy = accuracy_score(classifier, *test_ds[:])
    print(f"Test accuracy: {test_accuracy:.4f}")

Mapping Between Approaches
-------------------------

Here's how the programmatic API corresponds to the configuration-based approach:

.. list-table::
   :header-rows: 1
   :width: 100%

   * - Programmatic API
     - Configuration Entry
   * - ``ExcitationInhibitionNetwork(input_dim=784, ...)``
     - ``model.network.parameters`` section
   * - ``TrainerMLE(optimizer=...)``
     - ``train.learning_strategy: mle``
   * - ``trainer.train(epochs=20, batch_size=64)``
     - ``train.epochs: 20, train.batch_size: 64``
   * - ``load_mnist_as_datasets()``
     - ``task.dataset: mnist``

Next Steps
----------

Now that you've learned the basics of the Dendritic Modeling framework, you can:

- Explore more complex network architectures in :doc:`../advanced/network_structures/index`
- Try different learning strategies detailed in :doc:`../advanced/learning/index`
- Run ablation studies as described in :doc:`../advanced/analysis/ablation_studies`
- Visualize your models with tools in :doc:`../advanced/visualization/visualization_guide`
- Experiment with synthetic datasets designed to test specific dendritic computational capabilities

Refer to the detailed documentation for each component for more information.
