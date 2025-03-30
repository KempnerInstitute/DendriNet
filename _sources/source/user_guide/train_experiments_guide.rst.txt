.. _train_experiments_guide:

Using train_experiments.py
========================

Overview
--------

The ``train_experiments.py`` script is the primary entry point for running experiments with the dendritic modeling framework. It handles:

1. Loading and parsing YAML configuration files
2. Setting up the experiment environment
3. Building the model based on configuration
4. Applying the appropriate learning strategy
5. Training the model and saving results
6. Generating visualizations (if enabled)

This guide explains how to use the script to run experiments, override configuration parameters, and customize your workflow.

Basic Usage
----------

To run an experiment, you need a configuration file (see :ref:`config_yaml_guide` for details on configuration files):

.. code-block:: bash

    python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml

This command will:

1. Read the configuration file ``configs/config_exp.yaml``
2. Set up the experiment environment (random seed, output directory, etc.)
3. Load the specified dataset
4. Build the model based on the configuration
5. Apply the specified learning strategy
6. Train the model for the specified number of epochs
7. Evaluate the model on test data
8. Save the model and results
9. Generate visualizations (if enabled)

Script Workflow
--------------

Let's examine the workflow of ``train_experiments.py`` in more detail:

1. **Parsing Configuration**:
   The script loads the YAML configuration file and parses it into a structured object using OmegaConf.

2. **Setting Up Environment**:
   - Creates an output directory with a unique timestamp
   - Sets up logging to file and console
   - Sets random seeds for reproducibility
   - Initializes Weights & Biases if enabled

3. **Loading Data**:
   - Loads the specified dataset (MNIST, CIFAR10, etc.)
   - Splits the data into train, validation, and test sets
   - Applies any specified transformations

4. **Building Model**:
   - Determines model architecture based on configuration
   - Initializes the network with specified parameters
   - Wraps the network in a classifier (probabilistic or standard)

5. **Training**:
   - Creates an optimizer with specified learning rate and weight decay
   - Applies the specified learning strategy
   - Trains the model for the specified number of epochs
   - Saves checkpoints during training

6. **Evaluation**:
   - Evaluates the trained model on test data
   - Computes performance metrics (accuracy, loss)
   - Logs results to file and Weights & Biases

7. **Visualization**:
   - Generates visualizations if enabled
   - Creates plots for weights, activations, branch information, etc.
   - Saves visualization files to the output directory

8. **Saving Results**:
   - Saves the trained model
   - Saves configuration and parameters
   - Saves performance metrics

Command-Line Arguments
--------------------

The ``train_experiments.py`` script supports command-line arguments to override configuration parameters:

.. code-block:: bash

    python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml --train.learning_strategy freeze_branches --train.lr 0.01

This example overrides the learning strategy and learning rate specified in the configuration file.

You can override any parameter in the configuration using the dotted notation:

.. code-block:: bash

    python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml \
      --model.network.parameters.excitatory_layer_sizes "[256]" \
      --train.epochs 100 \
      --visualization.enabled true

Example: Running an MNIST Experiment
----------------------------------

Let's walk through a complete example of running an MNIST experiment:

1. **Create a Configuration File**:

   Create a file named ``configs/mnist_exp.yaml`` with the following content:

   .. code-block:: yaml

       model:
         task: classification
         probabilistic: true
         network:
           type: EINet
           parameters:
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

2. **Run the Experiment**:

   .. code-block:: bash

       python src/dendritic_modeling/train_experiments.py configs/mnist_exp.yaml

3. **Monitor Training Progress**:

   You'll see output similar to:

   .. code-block:: text

       INFO train_experiments.py:109  Loading dataset...
       INFO train_experiments.py:53  Running EINet with probabilistic classifier
       INFO train_experiments.py:147  Using learning strategy: mle
       INFO train_experiments.py:203  [Epoch 1/50] train_loss=0.452, valid_loss=0.402
       INFO train_experiments.py:203  [Epoch 2/50] train_loss=0.381, valid_loss=0.350
       ...
       INFO train_experiments.py:203  [Epoch 50/50] train_loss=0.123, valid_loss=0.142
       INFO train_experiments.py:270  Evaluating performance...
       INFO train_experiments.py:279  train acc=0.965, valid acc=0.951, test acc=0.948

4. **View Results**:

   Results will be saved in the output directory (e.g., ``outputs/mnist_einet_20230401_123456/``).

   If you enabled visualization, you'll find plots in the ``plots`` subdirectory.

5. **Analyze with Weights & Biases**:

   If you enabled W&B logging, you can view detailed metrics and visualizations on the W&B dashboard.

Example: Training with Different Learning Strategies
--------------------------------------------------

Let's run the same experiment with different learning strategies:

1. **Freeze Branches Strategy**:

   .. code-block:: bash

       python src/dendritic_modeling/train_experiments.py configs/mnist_exp.yaml \
         --train.learning_strategy freeze_branches \
         --train.epochs_per_layer 20 \
         --train.final_tune_epochs 10

2. **Two-Step Strategy**:

   .. code-block:: bash

       python src/dendritic_modeling/train_experiments.py configs/mnist_exp.yaml \
         --train.learning_strategy two_step \
         --train.pretrain_epochs 10 \
         --train.epochs 50

3. **Voltage Stabilization Strategy**:

   .. code-block:: bash

       python src/dendritic_modeling/train_experiments.py configs/mnist_exp.yaml \
         --train.learning_strategy voltage_stabilization \
         --train.warmup_epochs 5 \
         --train.stability_target_voltage 0.5

Example: Parameter Sweeps
-----------------------

For parameter sweeps, you can use shell scripts or Weights & Biases sweeps.

Here's a simple bash script for a learning rate sweep:

.. code-block:: bash

    #!/bin/bash

    for lr in 0.0001 0.0005 0.001 0.005 0.01
    do
        python src/dendritic_modeling/train_experiments.py configs/mnist_exp.yaml \
          --train.lr $lr \
          --train.run_name "mnist_lr_${lr}"
    done

For more advanced parameter sweeps, you can use W&B:

1. Create a sweep configuration file ``sweep_config.yaml``:

   .. code-block:: yaml

       program: src/dendritic_modeling/train_experiments.py
       method: grid
       parameters:
         config:
           value: configs/mnist_exp.yaml
         train.lr:
           values: [0.0001, 0.0005, 0.001, 0.005, 0.01]
         train.batch_size:
           values: [64, 128, 256]

2. Initialize the sweep:

   .. code-block:: bash

       wandb sweep sweep_config.yaml

3. Run the sweep agent:

   .. code-block:: bash

       wandb agent YOUR_SWEEP_ID

Debug and Troubleshooting
------------------------

Common issues and solutions:

1. **Out of memory errors**:
   - Reduce batch size: ``--train.batch_size 32``
   - Reduce model size: ``--model.network.parameters.excitatory_layer_sizes "[128]"``

2. **Slow convergence**:
   - Adjust learning rate: ``--train.lr 0.005``
   - Try different learning strategies: ``--train.learning_strategy freeze_branches_with_pretraining``

3. **Dataset not found**:
   - Specify full path to data: ``--task.data_path "/path/to/data"``

4. **Visualization errors**:
   - Disable visualization for debugging: ``--visualization.enabled false``

5. **General debugging**:
   - Add verbose output: ``--verbose true``
   - Enable gradient logging: ``--train.log_gradients true``

Advanced Usage
------------

1. **Custom Datasets**:

   To use a custom dataset, you need to:
   - Add the dataset to ``synthetic_datasets.py``
   - Reference it in your configuration file
   - Specify the data path

2. **Custom Learning Strategies**:

   To implement a custom learning strategy:
   - Add the strategy to ``learning_strategies.py``
   - Register it in the ``get_trainer()`` function
   - Use it in your configuration file

3. **Custom Visualizations**:

   To add custom visualizations:
   - Add the visualization function to ``visualization_manager.py``
   - Register it in the ``PlotManager`` class
   - Enable it in your configuration file

Conclusion
---------

The ``train_experiments.py`` script provides a flexible and powerful interface for running experiments with the dendritic modeling framework. By combining it with the configuration system, you can easily:

- Run a wide variety of experiments without modifying code
- Ensure reproducibility through fixed seeds and saved configurations
- Explore different model architectures and learning strategies
- Perform parameter sweeps to optimize performance
- Generate visualizations to understand model behavior

For more detailed information, see the :ref:`config_yaml_guide` for configuration options and the API reference for implementation details.
