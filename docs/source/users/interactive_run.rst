Running an interactive session
==============================

This section provides a step-by-step guide to running an interactive session with the Dendritic Modeling package. Interactive sessions are useful for exploring the package's features, testing different network configurations, and visualizing the results of your experiments.

To run a job with the `dendritic_modeling` package, we need the following components:

- A Python environment with the required dependencies installed
- A configuration file
- A run script 

In the following sections, we will walk you through the process of setting up an interactive session with the Dendritic Modeling package.

Step 1: Prepare your run environment
------------------------------------

The first step is setting up your compute environment. It can be your laptop or an an interactive session on the cluster. You will need to get the code and install the package in your conda environment. Please refer to the :ref:`installation_guide` for detailed instructions on setting up your environment.


Step 2: Create a configuration file

Each compute job requires a configuration file that specifies the network architecture, training parameters, and other experimental settings. The following shows an example of a configuration file:

.. code-block:: yaml

    model:
        input_dim: 784
        excitatory_layer_sizes: [10]
        inhibitory_layer_sizes: [20]
        excitatory_branch_factors: [24]
        inhibitory_branch_factors: []
        reactivate: true
        somatic_synapses: true
        ee_synapses_per_branch_per_layer: [24]
        ei_synapses_per_branch_per_layer: [100]
        ie_synapses_per_branch_per_layer: [1]
        ii_synapses_per_branch_per_layer: []

    train:
        run_name: mnist_1
        seed: 42
        train_valid_split: 0.8
        lr: 0.001
        epochs: 10
        batch_size: 32

    wandb:
        entity: your_entity  # <---- Add your entity
        project: dendritic_modeling_pr
        group: debug
        tags: [mnist_1]


The configuration (let's say `config_exp.yaml` located in config/) has three major sections: `model`, `train`, and `wandb`. The `model` section specifies the network architecture, including the input dimension, layer sizes, branch factors, and synaptic connections. The `train` section defines the training parameters, such as the learning rate, number of epochs, and batch size. The `wandb` section contains the settings for logging the experiment results to Weights & Biases (W&B). Please refer to the :ref:`concept_overview` for more details on the network architecture.


Step 2: Run the interactive session
-----------------------------------

Once you have set up your environment and created the configuration file, you can run the interactive session using the following command:

.. code-block:: bash

    $ python scripts/train_mnist.py configs/mnist_1.yaml --output_dir output_folder --experiment_name mnist_1


The `output_dir` and `experiment_name` arguments specify the output directory and the name of the experiment, respectively. The `train_mnist.py` script will load the configuration file `mnist_1.yaml` and start the training process. You can monitor the training progress and visualize the results using the W&B dashboard.






