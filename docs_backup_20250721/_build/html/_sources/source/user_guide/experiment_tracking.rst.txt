Experiment Tracking
===================

The Dendritic Modeling framework provides comprehensive tools for tracking experiments, logging results, and visualizing outcomes. This guide explains how to use these features effectively.

Logging System
--------------

The framework includes a built-in logging system that records important information during training and evaluation:

.. code-block:: python

    from dendritic_modeling.logging_config import setup_logging

    # Get logger
    logger = setup_logging()

    # Log information
    logger.info("Starting experiment with learning rate: %f", learning_rate)
    logger.debug("Model parameters: %s", model_parameters)
    logger.warning("Validation loss increasing, consider early stopping")
    logger.error("Failed to load dataset from %s", data_path)

By default, logs are written to both console and a file called `dendritic_modeling.log` in your working directory.

Configuring the Logger
~~~~~~~~~~~~~~~~~~~~~~

You can customize the logging behavior:

.. code-block:: python

    import logging
    from dendritic_modeling.logging_config import setup_logging, LOGGER_NAME

    # Get logger with custom settings
    logger = setup_logging()

    # Change log level
    logger.setLevel(logging.DEBUG)  # Show all logs

    # Change handlers
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.DEBUG)
        elif isinstance(handler, logging.StreamHandler):
            handler.setLevel(logging.INFO)

    # Create a new handler for a specific experiment
    experiment_handler = logging.FileHandler('experiment_results.log')
    experiment_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(experiment_handler)

Local Experiment Tracking
-------------------------

For basic experiment tracking, the framework provides utilities to save results locally:

.. code-block:: python

    from dendritic_modeling.utils import save_dict

    # Training loop
    train_losses = []
    valid_losses = []
    accuracies = []

    for epoch in range(epochs):
        # ... training code ...
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        accuracies.append(valid_accuracy)

    # Save results
    results = {
        'train_losses': train_losses,
        'valid_losses': valid_losses,
        'accuracies': accuracies,
        'hyperparameters': {
            'lr': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs
        },
        'model_config': model.config
    }

    save_dict(results, save_path='results/', fname='experiment_results.json')

This saves a structured JSON file that you can later analyze.

Visualizing Local Results
~~~~~~~~~~~~~~~~~~~~~~~~~

The framework includes visualization utilities to plot results:

.. code-block:: python

    from dendritic_modeling.plot_utils import plot_NLL_loss_curves

    # Plot loss curves
    plot_NLL_loss_curves(
        train_losses=train_losses,
        valid_losses=valid_losses,
        num_epochs=epochs,
        save_path='results/plots/'
    )

Integration with Weights & Biases
---------------------------------

For more advanced experiment tracking, the framework integrates with Weights & Biases (wandb):

Setting Up Weights & Biases
~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Install wandb:

   .. code-block:: bash

      pip install wandb

2. Configure wandb in your YAML configuration:

   .. code-block:: yaml

      wandb:
        entity: "your-username"
        project: "dendritic-experiments"
        group: "mnist-comparison"
        tags: ["dendritic", "mnist"]
        use_wandb: true

3. Initialize wandb in your code:

   .. code-block:: python

      import wandb
      from dendritic_modeling.config import load_config

      # Load configuration
      config = load_config("path/to/config.yaml")

      # Initialize wandb
      if config.wandb.use_wandb:
          wandb.init(
              entity=config.wandb.entity,
              project=config.wandb.project,
              group=config.wandb.group,
              tags=config.wandb.tags,
              config=config.asdict()  # Log configuration as wandb config
          )

Logging Metrics to Weights & Biases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During training, log metrics to wandb:

.. code-block:: python

    # Training loop
    for epoch in range(epochs):
        # ... training code ...

        # Log metrics
        if config.wandb.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "accuracy": accuracy,
                "learning_rate": current_lr
            })

Logging Model Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~

Log the model architecture to wandb:

.. code-block:: python

    if config.wandb.use_wandb:
        # Log model architecture
        wandb.watch(model, log="all", log_freq=100)

Saving and Logging Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Save model checkpoints and visualizations as wandb artifacts:

.. code-block:: python

    # Save model checkpoint
    checkpoint_path = f"checkpoints/model_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': valid_loss,
    }, checkpoint_path)

    # Log as wandb artifact
    if config.wandb.use_wandb:
        artifact = wandb.Artifact(f"model-checkpoint-{epoch}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)

    # Save and log visualizations
    if epoch in config.visualization.visualize_epochs:
        # Generate visualizations
        visualization_path = f"visualizations/epoch_{epoch}/"

        # Save branch visualizations
        plot_einet_params(
            einet=model.net,
            save_root=visualization_path,
            filename=f"weights_epoch_{epoch}"
        )

        # Log to wandb
        if config.wandb.use_wandb:
            wandb.log({
                "weights_visualization": wandb.Image(f"{visualization_path}/weights_epoch_{epoch}.png"),
            })

Running Sweeps with Weights & Biases
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   While Weights & Biases does support parameter sweeps as described below, the standard workflow in this codebase uses dedicated sweep generation scripts like ``generate_ei_sweep.py`` rather than wandb's native sweep functionality. See the :ref:`ei_sweep_guide` for the recommended approach.

Wandb can support parameter sweeps for hyperparameter optimization, though this is not the primary method used in this codebase:

.. code-block:: python

    # NOTE: This is provided for reference but is not the standard workflow in this codebase
    sweep_config = {
        'method': 'grid',  # or 'random', 'bayes'
        'parameters': {
            'train.lr': {'values': [0.0001, 0.001, 0.01]},
            'train.batch_size': {'values': [32, 64, 128]},
            'model.network.parameters.excitatory_branch_factors': {'values': [[1, 2], [2, 2], [2, 4]]}
        }
    }

    # Initialize sweep
    sweep_id = wandb.sweep(sweep_config, project=config.wandb.project)

    # Run sweep agent
    wandb.agent(sweep_id, train_function, count=9)

Specialized EI Parameter Sweeps (Recommended Approach)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For detailed exploration of excitatory and inhibitory parameter combinations, the framework provides specialized scripts that are the standard workflow for parameter sweeps in this codebase:

1. **Generate parameter sweep configurations** using ``generate_ei_sweep.py`` with ``sweep_params_seeds_ei.yaml``
2. **Run the experiments** using the generated ``run_ei_sweep.sh`` script
3. **Analyze results** with ``analyze_ei_sweep.py`` to visualize performance across parameter combinations

This is the recommended approach for parameter sweeps in this codebase. For detailed instructions, see the :ref:`ei_sweep_guide`.

The workflow looks like:

.. code-block:: bash

    # Step 1: Generate sweep configurations
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml

    # Step 2: Run the sweep experiments
    cd output/ei_<timestamp>
    ./run_ei_sweep.sh

    # Step 3: Analyze the results
    python scripts/analyze_ei_sweep.py --folder output/ei_<timestamp>

This approach is particularly useful for understanding how excitatory and inhibitory balance affects model performance and finding the optimal parameter configuration for your tasks.

Comparing Experiments
---------------------

Analyzing Experiment Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The framework provides utilities to compare multiple experiments:

.. code-block:: python

    import json
    import matplotlib.pyplot as plt

    # Load multiple experiment results
    def load_results(experiment_paths):
        results = {}
        for name, path in experiment_paths.items():
            with open(f"{path}/experiment_results.json", "r") as f:
                results[name] = json.load(f)
        return results

    experiment_paths = {
        "EINet-Standard": "results/einet_standard",
        "EINet-TwoStep": "results/einet_twostep",
        "MLP-Baseline": "results/mlp_baseline"
    }

    results = load_results(experiment_paths)

    # Compare learning curves
    plt.figure(figsize=(12, 6))
    for name, data in results.items():
        plt.plot(data["valid_losses"], label=f"{name} (Acc: {data['accuracies'][-1]:.4f})")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss")
    plt.title("Comparison of Learning Curves")
    plt.legend()
    plt.savefig("results/comparison_curves.png")

With Weights & Biases, you can use their dashboard for more advanced comparisons.

Reproducibility
---------------

To ensure experiment reproducibility:

1. Always set random seeds in your configuration:

   .. code-block:: yaml

      train:
        seed: 42

2. Record the exact environment:

   .. code-block:: python

      import torch
      import numpy as np
      import platform

      # Log environment details
      env_info = {
          "python_version": platform.python_version(),
          "pytorch_version": torch.__version__,
          "numpy_version": np.__version__,
          "cuda_available": torch.cuda.is_available(),
          "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
          "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
      }

      # Save environment info with results
      results["environment"] = env_info
      save_dict(results, save_path='results/', fname='experiment_results.json')

3. Version control your code and configurations

Best Practices
--------------

1. **Systematic Naming**: Use consistent naming conventions for experiments
2. **Hyperparameter Tracking**: Always record all hyperparameters used
3. **Regular Checkpoints**: Save model checkpoints frequently during training
4. **Resource Monitoring**: Track GPU memory usage and training time
5. **Automation**: Use scripts to automate experiment execution
6. **Reproducibility**: Set random seeds and record environment details
7. **Visualization**: Generate visualizations to help interpret results

For research publications, maintain a record of all experiments, including unsuccessful ones, to provide complete context.
