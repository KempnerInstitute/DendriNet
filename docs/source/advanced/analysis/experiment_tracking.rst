.. _experiment_tracking:

Experiment Tracking and Logging
===============================

This document explains the experiment tracking and logging capabilities of the dendritic modeling framework, including integration with Weights & Biases (W&B) and local logging features.

Overview
--------

Tracking experiments is crucial for research reproducibility and comparison of results. The dendritic modeling framework provides comprehensive experiment tracking through:

1. **Weights & Biases (W&B) integration**: Cloud-based experiment tracking
2. **Local CSV logging**: Recording metrics to local files
3. **Checkpoint saving**: Saving model states for resuming training
4. **Visualization generation**: Creating plots of network properties at specified intervals

These tracking mechanisms help maintain a record of experiments, compare different configurations, and identify the most promising approaches.

Weights & Biases Integration
----------------------------

The framework integrates with Weights & Biases (W&B) for experiment tracking. This is configured in the ``wandb`` section of the configuration file:

.. code-block:: yaml

    wandb:
      entity: your_username          # Your W&B username or team name
      project: dendritic-networks    # Project name
      group: mnist-experiments       # Group for related experiments
      tags: [dendritic, mnist]       # Tags for filtering experiments
      use_wandb: true                # Whether to use W&B logging

When ``use_wandb`` is set to ``true``, the framework initializes W&B at the start of training and logs metrics, hyperparameters, and visualizations.

Initialization
~~~~~~~~~~~~~~

W&B is initialized in the training script:

.. code-block:: python

    # Initialize W&B if enabled
    if config.wandb.use_wandb:
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            group=config.wandb.group,
            name=config.train.run_name,
            config=OmegaConf.to_container(config, resolve=True),
            tags=config.wandb.tags
        )

This logs the complete configuration for reproducibility and creates a new run with the specified name.

Metrics Logging
~~~~~~~~~~~~~~~

During training, metrics are logged to W&B at the end of each epoch:

.. code-block:: python

    # Log metrics to W&B
    if config.wandb.use_wandb:
        wandb.log({
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "train_accuracy": train_accuracy,
            "valid_accuracy": valid_accuracy,
            "epoch": epoch
        })

Additional metrics can be logged during training, such as learning rate, gradient norms, and custom metrics specific to dendritic networks.

Model Checkpoints
~~~~~~~~~~~~~~~~~

W&B can also track model checkpoints:

.. code-block:: python

    # Save model checkpoint to W&B
    if config.wandb.use_wandb:
        wandb.save(checkpoint_path)

This allows downloading model checkpoints directly from the W&B interface.

Visualizations
~~~~~~~~~~~~~~

Visualizations are uploaded to W&B when generated:

.. code-block:: python

    # Log visualization to W&B
    if config.wandb.use_wandb:
        wandb.log({"weight_visualization": wandb.Image(viz_path)})

This makes it easy to compare visualizations across different runs.

Best Practices for W&B
~~~~~~~~~~~~~~~~~~~~~~

1. **Use consistent project names**: Keep related experiments in the same project
2. **Use groups for experiment series**: Group runs that test similar hypotheses
3. **Tag runs appropriately**: Add tags to categorize experiments
4. **Version your code**: Link W&B runs to specific code commits
5. **Log all relevant metrics**: Include custom metrics that give insight into dendritic dynamics

Local CSV Logging
-----------------

Even when W&B is disabled, the framework logs metrics to local CSV files:

.. code-block:: python

    # Log metrics to CSV
    with open(os.path.join(output_dir, "metrics.csv"), "a") as f:
        if epoch == 0:
            f.write("epoch,train_loss,valid_loss,train_accuracy,valid_accuracy\n")
        f.write(f"{epoch},{train_loss},{valid_loss},{train_accuracy},{valid_accuracy}\n")

This creates a ``metrics.csv`` file in the output directory with all training metrics.

Structure of the Logging System
-------------------------------

The logging system uses Python's built-in ``logging`` module, configured in ``logging_config.py``:

.. code-block:: python

    def setup_logging(log_dir, experiment_name, log_level=logging.INFO):
        """
        Setup logging configuration.
        
        Parameters
        ----------
        log_dir : str
            Directory to save log files
        experiment_name : str
            Name of the experiment
        log_level : int
            Logging level, default is INFO
            
        Returns
        -------
        logging.Logger
            Configured logger
        """
        log_path = os.path.join(log_dir, f"{experiment_name}.log")
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        
        # Create logger
        logger = logging.getLogger(experiment_name)
        logger.setLevel(log_level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

This creates a logger that writes both to a file and to the console.

Using the Logger
~~~~~~~~~~~~~~~~

The logger is used throughout the codebase to record events and errors:

.. code-block:: python

    # Get logger
    logger = setup_logging(output_dir, config.train.run_name)
    
    # Log training progress
    logger.info(f"Epoch {epoch}/{config.train.epochs} - "
                f"Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}, "
                f"Train Acc: {train_accuracy:.4f}, Valid Acc: {valid_accuracy:.4f}")
    
    # Log warnings
    logger.warning("Learning rate is unusually high")
    
    # Log errors (without stopping execution)
    try:
        # Some code that might raise an exception
    except Exception as e:
        logger.error(f"Error during training: {e}")

Log files are saved in the output directory specified in the configuration.

Model Checkpointing
-------------------

The framework automatically saves model checkpoints during training:

.. code-block:: python

    # Save checkpoint
    checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
    }, checkpoint_path)

By default, checkpoints are saved at the end of each epoch. For longer training runs, you can configure checkpoint frequency in the configuration:

.. code-block:: yaml

    train:
      checkpoint_frequency: 10  # Save checkpoint every 10 epochs

Best Model Tracking
~~~~~~~~~~~~~~~~~~~

The framework also keeps track of the best model according to validation metrics:

.. code-block:: python

    # Check if this is the best model so far
    if valid_accuracy > best_valid_accuracy:
        best_valid_accuracy = valid_accuracy
        best_model_path = os.path.join(output_dir, "best_model.pt")
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"New best model saved with validation accuracy: {valid_accuracy:.4f}")

This ensures you always have access to the best-performing model.

Resuming Training
~~~~~~~~~~~~~~~~~

Checkpoints allow resuming training from a specific epoch:

.. code-block:: python

    if args.resume_from:
        checkpoint = torch.load(args.resume_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming training from epoch {start_epoch}")

Visualization Generation
------------------------

The framework can generate visualizations at specified intervals during training:

.. code-block:: yaml

    visualization:
      enabled: true
      visualize_epochs: [0, 10, 50, 100]  # Epochs at which to generate visualizations
      plots:
        - weights
        - activations
        - gradients

When visualizations are enabled, the framework generates the specified plots at the specified epochs:

.. code-block:: python

    # Generate visualizations if necessary
    if config.visualization.enabled and epoch in config.visualization.visualize_epochs:
        visualize_model(model, data, output_dir, epoch)

Visualizations are saved in the output directory and can also be logged to W&B.

.. figure:: figures/pdf/place_holder.pdf
   :alt: Experiment Tracking System
   :width: 600px
   :align: center

   Overview of the experiment tracking and logging system in the dendritic modeling framework. [PLACEHOLDER FOR DIAGRAM]

Custom Metrics for Dendritic Networks
-------------------------------------

The framework includes custom metrics specific to dendritic networks:

1. **Branch activation ratio**: The fraction of branches that are active for each input
2. **Branch specialization metric**: How specialized branches are for different inputs
3. **Synaptic weight sparsity**: The fraction of synaptic weights that are close to zero
4. **Mutual information**: Information content between branch activations and labels

These metrics are calculated during training and logged along with standard metrics.

Best Practices for Tracking
---------------------------

1. **Unique run names**: Use descriptive, unique names for each run
2. **Version your configurations**: Keep configurations under version control
3. **Log hardware details**: Note GPU/CPU models for performance comparison
4. **Set reasonable checkpoint frequency**: Balance storage space vs. safety
5. **Archive important runs**: Keep archives of particularly important results

Example: Using the Tracking System
----------------------------------

Here's an example of how to use the tracking system in a custom script:

.. code-block:: python

    from dendritic_modeling.config import load_config
    from dendritic_modeling.logging_config import setup_logging
    import wandb
    
    # Load configuration
    config = load_config("configs/my_experiment.yaml")
    
    # Set up logging
    logger = setup_logging("outputs", config.train.run_name)
    
    # Initialize W&B
    if config.wandb.use_wandb:
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            name=config.train.run_name,
            config=config
        )
    
    # Log parameters
    logger.info(f"Starting experiment with learning rate: {config.train.lr}")
    
    # Training loop
    for epoch in range(config.train.epochs):
        # Train model
        train_loss, train_accuracy = train_epoch(model, train_loader)
        valid_loss, valid_accuracy = validate_epoch(model, valid_loader)
        
        # Log metrics
        logger.info(f"Epoch {epoch}: Train Acc={train_accuracy:.4f}, Valid Acc={valid_accuracy:.4f}")
        
        if config.wandb.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "valid_loss": valid_loss,
                "train_accuracy": train_accuracy,
                "valid_accuracy": valid_accuracy,
                "epoch": epoch
            })
    
    # Save final model
    torch.save(model.state_dict(), "outputs/final_model.pt")
    logger.info("Training complete, model saved.") 