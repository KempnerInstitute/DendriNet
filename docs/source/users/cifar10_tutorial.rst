.. _cifar10_tutorial:

CIFAR10 Tutorial
===============

This tutorial demonstrates how to train a dendritic network on the CIFAR10 dataset using the dendritic modeling framework.

Prerequisites
------------

Before starting this tutorial, make sure you have:
- Installed the dendritic modeling package and its dependencies
- Set up a conda environment as described in the :ref:`installation_section`
- Basic familiarity with the concepts described in :ref:`concept_overview`

Step 1: Prepare the configuration file
--------------------------------------

Create a configuration file named ``config_cifar10.yaml`` with the following content:

.. code-block:: yaml

    model:
      task: classification         
      probabilistic: True
      network:
        type: "EINet"
        parameters:
          input_dim: 3072  # 3 channels x 32 x 32
          excitatory_layer_sizes: [512, 256]
          inhibitory_layer_sizes: [200, 100]
          excitatory_branch_factors: [64, 32]
          inhibitory_branch_factors: [32, 16]
          reactivate: true
          somatic_synapses: true
          ee_synapses_per_branch_per_layer: [24, 24]
          ei_synapses_per_branch_per_layer: [200, 100]
          ie_synapses_per_branch_per_layer: [20, 10]
          ii_synapses_per_branch_per_layer: [20, 10]
          topk_init_method: xavier_normal
          use_shunting: true
          init_gain: 1.0
          init_inhib_offset: 1.3
          reactivation_initial_m: 2
          reactivation_initial_b: 0.5
          reactivation_type: param_tanh       
          train_reactivation_b_only: false
          gradient_strategy: block_conductance_dynamic  
          gradient_scale_factor: 2
          weight_decay_rate: 0.1
          enable_branch_outputs: true
          output_dim: 10

    train:
      learning_strategy: "mle"           
      run_name: "cifar10_einet"
      seed: 0
      train_valid_split: 0.8
      lr: 0.001
      epochs: 50
      batch_size: 128
      grad_clip_value: 5
      weight_decay_rate: 0.0001

    task:
      dataset: "cifar10"
      parameters: {}

    wandb:
      entity: your_wandb_entity
      project: "dendritic_cifar10"
      group: "cifar10_experiments"
      tags: [cifar10]
      use_wandb: true

    outputs:
      dir: "outputs"

    visualization:
      enabled: true
      plots:
        - "weights"        
        - "activations"    
        - "gradients"      
        - "branch_info"    
        - "ablation"

Let's examine the key differences from MNIST configuration:

1. **Input dimension**: Set to 3072 (3 channels × 32 × 32 pixels) instead of 784 for MNIST
2. **Network architecture**: Deeper architecture with multiple layers
3. **Learning rate**: Lower learning rate (0.001) for more stable training
4. **Batch size**: Larger batch size for better gradient estimates

Step 2: Run the training script
-------------------------------

Execute the following command to start training:

.. code-block:: bash

    $ python src/dendritic_modeling/train_experiments.py configs/config_cifar10.yaml --output_dir results

This will:
1. Load the CIFAR10 dataset (downloading it if necessary)
2. Initialize the dendritic network according to the configuration
3. Train the network for the specified number of epochs
4. Save the results in the output directory

Step 3: Monitor training progress
---------------------------------

During training, you can monitor:
- Training and validation loss/accuracy in the console output
- Detailed metrics in Weights & Biases if enabled
- Generated visualizations in the output directory

Step 4: Analyze results
-----------------------

After training completes, you can analyze the results:

1. **Final accuracy**: Check the test accuracy to evaluate model performance
2. **Visualization analysis**: Examine the generated visualizations to understand:
   - Weight distributions and patterns
   - Activation patterns for different classes
   - Branch-specific activities
   - Effects of ablating different branches or layers

3. **Learning dynamics**: Analyze how loss and accuracy changed during training

Optimizing for CIFAR10
----------------------

CIFAR10 is more challenging than MNIST. Here are some tips for improving performance:

1. **Network architecture adjustments**:
   - Increase the number of layers and their sizes
   - Experiment with different branch factors
   - Try different values for synapses_per_branch parameters

2. **Training strategy modifications**:
   - Use learning rate scheduling (e.g., reduce on plateau)
   - Try different learning strategies (e.g., two_step, freeze_layers)
   - Experiment with different gradient strategies

3. **Data augmentation**:
   - Although not directly supported in the config, you can modify the dataset loading to include data augmentation

Advanced: Adding Data Augmentation
----------------------------------

For improved performance, you can add data augmentation by modifying the dataset loading code:

1. Create a custom script based on `train_experiments.py` with added data augmentation:

.. code-block:: python

    # Add to imports
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, TensorDataset

    # Define augmentation transforms
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Apply augmentation during training
    # (This would need to be integrated with the existing data loading pipeline)

2. Use the modified script for training with augmentation

Comparison with Standard Neural Networks
----------------------------------------

It's instructive to compare dendritic networks with standard neural networks on CIFAR10:

1. **Performance**: Dendritic networks often achieve better accuracy with fewer parameters due to their sophisticated branch-specific processing
2. **Robustness**: Dendritic networks can be more robust to noise and adversarial examples
3. **Interpretability**: The branch structure offers additional interpretability by examining branch-specific activities

Troubleshooting
---------------

Common issues when training on CIFAR10:

1. **Out of memory errors**: Reduce batch size or model size
2. **Slow convergence**: Try different learning rates or optimization strategies
3. **Dimension errors**: Ensure input_dim is correctly set to 3072
4. **Data path issues**: If data loading fails, check that data is being saved to the correct path (outputs/data/cifar)

For more detailed information on troubleshooting, refer to the general documentation. 