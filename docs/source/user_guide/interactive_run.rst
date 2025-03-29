Running an interactive session
==============================

This section provides a step-by-step guide to running an interactive session with the `dendritic_modeling` package. Interactive sessions are useful for exploring the package's features, testing different network configurations, and visualizing the results of your experiments.

To run a job with the `dendritic_modeling` package, we need the following components:

- A Python environment with the required dependencies installed
- A configuration file
- A run script

In the following sections, we will walk you through the process of setting up an interactive session with the `dendritic_modeling` package.

Step 1: Prepare your run environment
------------------------------------

The first step is setting up your compute environment. It can be your laptop or an an interactive session on the cluster. You will need to get the code and install the package in your conda environment. Please refer to the :ref:`installation_section` for detailed instructions on setting up your environment.


Step 2: Create a configuration file
-----------------------------------

Each compute job requires a configuration file that specifies the network architecture, training parameters, and other experimental settings. The following shows an example of a configuration file:

.. code-block:: yaml

    model:
      task: classification
      probabilistic: True
      network:
        type: "EINet"
        parameters:
          input_dim: 784
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
      epochs_per_layer: 20
      final_tune_epochs: 10
      pretrain_epochs: 10

    task:
      dataset: "mnist"
      data_path: ""
      parameters: {}

    wandb:
      entity: ""
      project: "dendritic_modeling"
      group: "mnist_experiments"
      tags: ["einet", "mnist"]
      use_wandb: true

    visualization:
      enabled: true
      visualize_epochs: [1, 10, 50]
      plots: ["branch_info", "ablation"]
      parameters: {}

    outputs:
      dir: "outputs"


The configuration file defines several main sections:
  - **model**: Specifies the network architecture, including the type of network (e.g., EINet), layer sizes, branch factors, synaptic connections, and reactivation settings.
  - **train**: Defines the training parameters, such as the learning strategy (e.g., "mle", "freeze_branches", "two_step", etc.), learning rate, number of epochs, batch size, etc.
  - **task**: Specifies the dataset to use and any dataset-specific parameters.
  - **wandb**: Contains settings for logging the experiment to Weights & Biases (W&B).
  - **visualization**: Contains settings for visualization, including which epochs to generate visualizations for and which types of plots to create.
  - **outputs**: Specifies the directory where outputs will be saved.

Please refer to the :ref:`concept_overview` for more details on the network architecture.

Step 3: Run the interactive session
-----------------------------------

Once you have set up your environment and created the configuration file, you can run the interactive session using the following command:

.. code-block:: bash

  $ python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml


You can also override specific configuration values at runtime:

.. code-block:: bash

  $ python src/dendritic_modeling/train_experiments.py configs/config_exp.yaml --train.learning_strategy freeze_branches --train.lr 0.005


The `train_experiments.py` script will load the configuration file `config_exp.yaml` and start the training process. You can monitor the training progress in the terminal output and visualize the results using the W&B dashboard if you've enabled it.


Step 4: Monitor outputs and visualizations
------------------------------------------

During training, the log messages will indicate:
  - The completion of each epoch with training and validation metrics
  - The learning strategy being used
  - Saving of checkpoint models
  - Final performance metrics (train, validation, and test accuracy)

If W&B is enabled, you can monitor the training progress and visualizations on the W&B dashboard. The experiment results, including model weights and performance metrics, will be saved in the directory specified in the `outputs.dir` parameter of your configuration file.

The visualizations, if enabled, will be generated at the epochs specified in the `visualization.visualize_epochs` list and will include the plot types specified in the `visualization.plots` list.
