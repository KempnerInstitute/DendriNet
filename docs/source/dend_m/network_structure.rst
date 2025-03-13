.. _network_structure:

Network Structure and Input Modes
===============================

The dendritic modeling framework provides a flexible architecture that can be configured in various ways. This document explains the network structure and the different input processing modes available.

Overview
--------

The dendritic network architecture consists of several key components:

1. **Input processing layer**: Optional transformation of raw inputs before dendritic processing
2. **Excitatory dendritic layers**: Process inputs through excitatory pathways
3. **Inhibitory dendritic layers**: Process inputs through inhibitory pathways
4. **Output layer**: Combines excitatory and inhibitory pathways to produce final output

The framework allows for different configurations of these components, particularly in how inputs are processed before entering the dendritic layers.

Network Architecture
-----------------

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

ExcitationInhibitionNetwork
-------------------------

The ``ExcitationInhibitionNetwork`` is the main network class that implements the excitatory-inhibitory architecture:

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

The network consists of a sequence of ``ExcitationInhibitionLayer`` instances, each containing excitatory and inhibitory dendritic networks.

Input Modes
---------

One of the most important configuration options is the ``input_mode`` parameter, which determines how inputs are processed before entering the dendritic layers:

.. code-block:: yaml

    model:
      network:
        parameters:
          input_mode: 0  # Raw input mode (default)
          # Alternative modes: 1 (linear transform) or 2 (network transform)

The framework supports three input modes:

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

Input Mode 0: Raw Input
---------------------

In input mode 0, the raw inputs are passed directly to the dendritic layers:

.. code-block:: python

    # Input mode 0: Raw input
    def forward(self, x):
        # No transformation
        excitatory_input = x
        inhibitory_input = None
        
        # Pass directly to dendritic layers
        for layer in self.layers:
            excitatory_output, inhibitory_output = layer(excitatory_input, inhibitory_input)
            excitatory_input = excitatory_output
            inhibitory_input = inhibitory_output
            
        return excitatory_output

This mode is suitable when:
- The input data is already well-structured
- You want to minimize the number of parameters
- You want to ensure the dendritic layers see the raw input patterns

Input Mode 1: Linear Transform
---------------------------

In input mode 1, the input is transformed through a linear layer:

.. code-block:: python

    # Input mode 1: Linear transform
    def forward(self, x):
        # Apply linear transformation
        transformed_input = self.input_transform(x)
        
        excitatory_input = transformed_input
        inhibitory_input = None
        
        # Continue with dendritic layers
        for layer in self.layers:
            excitatory_output, inhibitory_output = layer(excitatory_input, inhibitory_input)
            excitatory_input = excitatory_output
            inhibitory_input = inhibitory_output
            
        return excitatory_output

This mode is useful when:
- You want to adapt the input dimensionality to match the dendritic network
- The input features need simple re-weighting or projection
- You want to maintain a relatively direct connection to the input data

Input Mode 2: Network Transform
----------------------------

In input mode 2, the input is transformed through a more complex network:

.. code-block:: python

    # Input mode 2: Network transform
    def forward(self, x):
        # Apply network transformation
        transformed_input = self.input_net(x)
        
        excitatory_input = transformed_input
        inhibitory_input = None
        
        # Continue with dendritic layers
        for layer in self.layers:
            excitatory_output, inhibitory_output = layer(excitatory_input, inhibitory_input)
            excitatory_input = excitatory_output
            inhibitory_input = inhibitory_output
            
        return excitatory_output

This mode is suitable when:
- The input data requires significant feature extraction
- You want to leverage pre-trained feature extractors
- The task benefits from hierarchical feature processing

Input Network Configuration
------------------------

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

Available input network types include:

1. **MLPInputTransform**: Multi-layer perceptron with configurable hidden layers
   - Simple but effective for many tasks
   - Fully connected architecture

2. **ConvInputTransform**: Convolutional network for image data
   - Better suited for spatial data like images
   - Preserves spatial relationships

3. **SigmoidInputNetTransform**: MLP with sigmoid activation at the output
   - Produces bounded outputs which can be beneficial for dendritic processing
   - Often used with the sequential input network training strategy

Implementation details:

.. code-block:: python

    class MLPInputTransform(nn.Module):
        """
        Multi-layer perceptron input transformation network.
        
        Parameters
        ----------
        input_dim : int
            Dimension of the input data
        hidden_dims : list of int
            Dimensions of the hidden layers
        output_dim : int
            Dimension of the output
        activation : str
            Activation function to use
        dropout : float
            Dropout probability
        """
        
        def __init__(self, input_dim, hidden_dims, output_dim, activation="relu", dropout=0.0):
            super().__init__()
            
            # Build MLP layers
            self.layers = []
            prev_dim = input_dim
            
            for h_dim in hidden_dims:
                self.layers.append(nn.Linear(prev_dim, h_dim))
                self.layers.append(self._get_activation(activation))
                
                if dropout > 0:
                    self.layers.append(nn.Dropout(dropout))
                    
                prev_dim = h_dim
                
            # Output layer
            self.layers.append(nn.Linear(prev_dim, output_dim))
            
            # Combine into sequential model
            self.model = nn.Sequential(*self.layers)
            
        def forward(self, x):
            return self.model(x)

Sequential Input Network Training
------------------------------

When using input mode 2, you can enable sequential training of the input network:

.. code-block:: yaml

    train:
      input_net_training:
        training_mode: "sequential"
        train_input_net: true
        input_net_epochs: 20

With sequential training:
1. The input network is trained first, often as a standalone classifier
2. The trained input network is then used to initialize the dendritic model
3. The full model can be fine-tuned with the pre-trained input network

This approach can lead to more stable training and better final performance, especially for complex datasets.

Choosing the Right Input Mode
---------------------------

The choice of input mode depends on the specific task and data:

1. **Input Mode 0 (Raw)** is suitable for:
   - Simple, well-structured data with clear patterns
   - Tasks where interpretability of dendrite responses to raw inputs is important
   - When you want to minimize parameters and maximize the role of dendritic computation

2. **Input Mode 1 (Linear)** is suitable for:
   - Data that needs basic preprocessing or dimensionality adjustment
   - When you want a good balance between simplicity and flexibility
   - Situations where training stability is a concern

3. **Input Mode 2 (Network)** is suitable for:
   - Complex, high-dimensional data like images
   - When leveraging pretrained feature extractors is beneficial
   - Tasks requiring sophisticated feature extraction before dendritic processing

Combining with Other Configuration Options
---------------------------------------

The input mode should be considered in relation to other configuration options:

1. **With learning strategies**:
   - For input mode 2, sequential training often works well
   - For input mode 0, freeze_layers can be effective to build representations gradually

2. **With gradient scaling**:
   - Input mode 0 may require more aggressive gradient scaling
   - Input mode 2 may work well with milder gradient scaling

3. **With network depth**:
   - Input mode 0 often works with deeper dendritic networks
   - Input mode 2 may allow for shallower dendritic networks

Example Complete Configuration
----------------------------

Here's an example of a complete configuration with input mode 2:

.. code-block:: yaml

    model:
      task: classification
      probabilistic: true
      network:
        type: "EINet"
        parameters:
          input_dim: 784
          input_mode: 2
          input_net_dict:
            type: "MLPInputTransform"
            hidden_dims: [512, 256]
            output_dim: 128
            activation: "relu"
            dropout: 0.1
          excitatory_layer_sizes: [64, 32]
          inhibitory_layer_sizes: [32, 16]
          excitatory_branch_factors: [8, 4]
          inhibitory_branch_factors: [4, 2]
          output_dim: 10
          
    train:
      learning_strategy: "mle"
      input_net_training:
        training_mode: "sequential"
        train_input_net: true
        input_net_epochs: 20
      run_name: "mnist_einet_input_net"
      epochs: 50
      batch_size: 128 