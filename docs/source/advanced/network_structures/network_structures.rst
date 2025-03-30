.. _network_structures:

Network Structures and Architecture
=================================

Overview
--------

This document describes the neural network architectures available in the dendritic modeling framework,
their configurations, input processing modes, and how to customize them for specific tasks.

Available Network Types
----------------------

The framework provides several network types with different architectures:

1. **EINet (Excitation-Inhibition Network)**: A biologically-inspired architecture with dendritic branches
   and separate excitatory and inhibitory pathways.

2. **Standard MLP**: A traditional multilayer perceptron with fully-connected layers.

3. **Sparse MLP**: An MLP with sparse connectivity patterns.

Core Network Architecture
------------------------

The core network architecture is defined by the following parameters:

.. code-block:: yaml

    model:
      network:
        type: "EINet"  # ExcitationInhibitionNetwork
        parameters:
          input_dim: 784
          excitatory_layer_sizes: [512, 256]
          inhibitory_layer_sizes: [200, 100]
          excitatory_branch_factors: [64, 32]
          inhibitory_branch_factors: [32, 16]
          output_dim: 10

These parameters define:

1. **input_dim**: Dimension of the input data
2. **excitatory_layer_sizes**: Sizes of the excitatory layers
3. **inhibitory_layer_sizes**: Sizes of the inhibitory layers
4. **excitatory_branch_factors**: Branch factors for excitatory dendrites in each layer
5. **inhibitory_branch_factors**: Branch factors for inhibitory dendrites in each layer
6. **output_dim**: Dimension of the output (e.g., number of classes for classification)

Excitation-Inhibition Network (EINet)
-----------------------------------

The EINet is the primary architecture of the framework, implementing a biologically-inspired neural network
with dendritic branches, excitatory-inhibitory balance, and sparse connectivity.

.. note::
   The EINet architecture is inspired by observations of biological neurons, where
   excitatory and inhibitory inputs are processed differently and integrated in dendritic branches.

Implementation
~~~~~~~~~~~~~

The EINet is implemented through several core classes:

- `ExcitationInhibitionNetwork`: The main network class
- `ExcitationInhibitionLayer`: Individual layer with excitatory and inhibitory cells
- `TopKLinear`: Sparse connectivity implementation
- `BlockLinear`: Layer for integrating branch outputs

.. code-block:: python

    class ExcitationInhibitionNetwork(nn.Module):
        """
        A network with multiple excitatory-inhibitory layers.
        """

        def __init__(self, input_dim, excitatory_layer_sizes, inhibitory_layer_sizes,
                     excitatory_branch_factors, inhibitory_branch_factors,
                     input_mode=0, input_net_dict=None, ...):
            # Initialize the network with the specified parameters

        def forward(self, x):
            # Process input through the network
            # Return output

Excitation-Inhibition Layer
~~~~~~~~~~~~~~~~~~~~~~~~~

Each layer in an EINet contains both excitatory and inhibitory cells. The excitatory cells
receive inputs and produce outputs that propagate to the next layer, while inhibitory cells
modulate the activity of excitatory cells through inhibitory connections.

.. code-block:: python

    # Example configuration of an EI layer
    layer = ExcitationInhibitionLayer(
        n_excitatory_cells=10,
        n_inhibitory_cells=20,
        excitatory_branch_factors=[2, 2],
        inhibitory_branch_factors=[],
        excitatory_input_dim=784,
        ee_synapses_per_branch=24,
        ei_synapses_per_branch=100,
        inhibitory_input_dim=784,
        ie_synapses_per_branch=2,
        reactivate=True,
        somatic_synapses=True
    )

Dendritic Branch Processing
~~~~~~~~~~~~~~~~~~~~~~~~~

Dendritic branches implement a form of nonlinear integration of inputs, inspired by the
computation performed in biological dendrites.

.. code-block:: python

    # Simplified pseudocode for dendritic computation
    def dendritic_computation(excitatory_input, inhibitory_input):
        # Compute excitatory and inhibitory contributions
        excitation = weighted_sum(excitatory_input)
        inhibition = weighted_sum(inhibitory_input)

        # Shunting inhibition (division-based)
        voltage = excitation / (excitation + inhibition + 1)

        # Apply nonlinear activation
        output = activation_function(voltage)

        return output

Branch Parameters
~~~~~~~~~~~~~~~

The EINet architecture is highly configurable through various parameters:

- **Layer sizes**: Number of excitatory and inhibitory cells in each layer
- **Branch factors**: Number of dendritic branches per cell
- **Synapse counts**: Number of synapses per branch for each connection type
   - `ee_synapses_per_branch_per_layer`: Excitatory-to-excitatory connections
   - `ei_synapses_per_branch_per_layer`: Excitatory-to-inhibitory connections
   - `ie_synapses_per_branch_per_layer`: Inhibitory-to-excitatory connections
   - `ii_synapses_per_branch_per_layer`: Inhibitory-to-inhibitory connections
- **Reactivation parameters**: Controls the nonlinear activation in branches
- **Somatic synapses**: Whether to include direct synapses to the soma

Input Processing Modes
--------------------

The framework supports three input modes for processing data before it enters the dendritic layers:

.. code-block:: yaml

    model:
      network:
        parameters:
          input_mode: 0  # Raw input mode (default)
          # Alternative modes: 1 (linear transform) or 2 (network transform)

1. **Mode 0: Raw Input (Direct)**
   - Raw inputs are fed directly to the dendritic layers
   - No input transformation is applied
   - Suitable for simple tasks or pre-processed inputs

2. **Mode 1: Linear Transform**
   - Inputs are transformed through a linear layer
   - Allows for basic dimensionality reduction or expansion
   - Faster than network transform, less powerful but more stable

3. **Mode 2: Network Transform**
   - Inputs are transformed through a multi-layer network
   - More expressive than linear transform
   - Can learn complex feature representations
   - Requires more careful training

Input Network Types
-----------------

The framework supports several types of input transformation networks:

1. **MLPInputTransform**: Multi-layer perceptron with configurable hidden layers
   - Simple but effective for many tasks
   - Fully connected architecture

2. **ConvInputTransform**: Convolutional network for image data
   - Better suited for spatial data like images
   - Preserves spatial relationships

3. **AlexNetTransform**: Based on AlexNet architecture (for transfer learning)

4. **ResNetTransform**: Based on ResNet architecture (for complex feature extraction)

Input Network Configuration
-------------------------

When using input mode 2, you can configure the input network through the ``input_net_dict`` parameter:

.. code-block:: yaml

    model:
      network:
        parameters:
          input_mode: 2
          input_net_dict:
            type: "MLPInputTransform"
            hidden_dims: [512, 256]
            output_dim: 128
            activation: "relu"
            dropout: 0.2

Training Strategies
-----------------

Several strategies are available for training input transformation networks:

Sequential vs. Joint Training
~~~~~~~~~~~~~~~~~~~~~~~~~~

Input networks can be trained either sequentially (before the dendritic network)
or jointly (simultaneously with the dendritic network).

.. code-block:: python

    # Configuration for sequential training
    input_net_training = {
        'training_mode': 'sequential',
        'train_input_net': True,
        'input_net_epochs': 20
    }

    # Configuration for joint training
    input_net_training = {
        'training_mode': 'joint'
    }

Independent Head Training
~~~~~~~~~~~~~~~~~~~~~~

The excitatory and inhibitory heads can be trained independently with different objectives.

.. code-block:: python

    # Configuration for independent head training
    input_net_training = {
        'train_heads_independently': True,
        'inhibitory_head_objective': 'diversity',
        'diversity_weight': 0.5
    }

Pre-trained Input Networks
~~~~~~~~~~~~~~~~~~~~~~~

For complex tasks, pre-trained networks can be used to extract features.

.. code-block:: python

    # Configuration for using pre-trained weights
    input_net_training = {
        'train_input_net': False,
        'pretrained_options': {
            'source': 'pytorch',
            'pytorch_model': 'resnet18'
        }
    }

Transfer Learning Options
~~~~~~~~~~~~~~~~~~~~~~

When using pre-trained networks, several transfer learning approaches are available:

1. **Feature extraction**: Freeze pre-trained layers and train only new layers
2. **Fine-tuning**: Update all layers but with lower learning rates for pre-trained layers
3. **Progressive unfreezing**: Gradually unfreeze deeper layers during training

Example Configurations
-------------------

Here are some example configurations for different network types:

.. code-block:: yaml

    # EINet with input transformation
    model:
      network:
        type: "EINet"
        parameters:
          input_mode: 2
          input_net_dict:
            type: "MLPInputTransform"
            hidden_dims: [512, 256]
            output_dim: 128
          excitatory_layer_sizes: [256, 128]
          inhibitory_layer_sizes: [128, 64]
          excitatory_branch_factors: [32, 16]
          inhibitory_branch_factors: [16, 8]
          output_dim: 10

    # Sparse MLP configuration
    model:
      network:
        type: "SparseMLP"
        parameters:
          input_dim: 784
          hidden_dims: [512, 256]
          output_dim: 10
          sparsity: 0.1  # Only 10% of connections are active

    # Standard MLP with dropout
    model:
      network:
        type: "SimpleMLP"
        parameters:
          input_dim: 784
          hidden_dims: [512, 256]
          output_dim: 10
          activation: "relu"
          dropout: 0.2

Implementation Details (Overview)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implementation of network structures is modular and follows object-oriented principles:

1. Base classes define common interfaces and methods
2. Derived classes implement specific network architectures
3. Factory functions create appropriate networks based on configuration

Network Output Types
--------------------

The framework supports different output types depending on the task:

1. **Probabilistic Classification**: Returns a probability distribution over classes
2. **Deterministic Classification**: Returns class logits or predictions
3. **Regression**: Returns continuous values

.. code-block:: python

    # Probabilistic classifier
    class ProbabilisticClassifier(nn.Module):
        def forward(self, x):
            logits = self.network(x)
            return torch.distributions.Categorical(logits=logits)

    # Deterministic classifier
    class Classifier(nn.Module):
        def forward(self, x):
            return self.network(x)

Creating Custom Networks
------------------------

The framework is designed to be extensible, allowing users to create custom network architectures.

.. code-block:: python

    # Example of creating a custom network
    class CustomEINet(ExcitationInhibitionNetwork):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Add custom layers or components
            self.attention_layer = SelfAttention(dim=kwargs['excitatory_layer_sizes'][-1])

        def forward(self, x):
            # Custom forward pass
            features = super().forward(x)
            return self.attention_layer(features)

Computational Properties
------------------------

Dendritic networks have unique computational properties compared to traditional ANNs:

1. **Local nonlinearity**: Dendritic branches implement nonlinear computations before integration
2. **Sparse connectivity**: Each branch connects to a small subset of potential inputs
3. **Shunting inhibition**: Division-based inhibition instead of subtraction
4. **Hierarchical processing**: Information is processed at multiple levels (branch, cell, layer)

Choosing the Right Architecture
-------------------------------

Guidelines for selecting the appropriate network architecture:

1. For simple tasks with limited data, start with a standard MLP
2. For tasks requiring complex feature extraction, use input transformation with EINet
3. For tasks with multiple input modalities, consider separate input networks for each modality
4. For tasks requiring hierarchical processing, leverage deep EINets with multiple layers

Implementation Details
----------------------

Under the hood, the network structures are implemented using PyTorch modules and custom layers.
The implementation is designed to be efficient, leveraging techniques like sparse operations
and parallelized computation.
