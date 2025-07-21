Networks API Reference
======================

This section provides detailed API documentation for the network components in the dendritic modeling framework.

Core Network Classes
--------------------

ExcitationInhibitionNetwork
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.networks.ExcitationInhibitionNetwork
   :members:
   :undoc-members:
   :show-inheritance:

   The main network class implementing multi-layer excitatory-inhibitory networks with dendritic computation.

   **Example usage:**

   .. code-block:: python

      from dendritic_modeling.networks import ExcitationInhibitionNetwork

      # Create a 3-layer E/I network
      network = ExcitationInhibitionNetwork(
          input_dim=784,
          n_layers=3,
          n_excitatory=[200, 100, 10],
          n_inhibitory=[50, 25, 0],
          n_branches=[20, 15, 10],
          connectivity="sparse",
          sparsity=0.2,
          epsilon=1e-6
      )

      # Forward pass
      output = network(input_tensor)

DendriticNetwork
~~~~~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.networks.DendriticNetwork
   :members:
   :undoc-members:
   :show-inheritance:

   Basic dendritic network without separate E/I populations.

Layer Components
----------------

DendriticLayer
~~~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.networks.layers.DendriticLayer
   :members:
   :undoc-members:
   :show-inheritance:

   Core dendritic layer implementing shunting inhibition.

   **Key methods:**

   - ``forward(exc_input, inh_input)``: Compute dendritic output
   - ``reset_parameters()``: Initialize weights
   - ``get_effective_weights()``: Get E/(E+I+ε) weights

DendriticBranchLayer
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.networks.layers.DendriticBranchLayer
   :members:
   :undoc-members:
   :show-inheritance:

   Extended dendritic layer with explicit branch structure.

LinearExcitation / LinearInhibition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.networks.layers.linear.LinearExcitation
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: dendritic_modeling.networks.layers.linear.LinearInhibition
   :members:
   :undoc-members:
   :show-inheritance:

   Specialized linear layers for excitatory and inhibitory pathways with proper weight constraints.

Input Networks
--------------

InputNetwork
~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.networks.input_network.InputNetwork
   :members:
   :undoc-members:
   :show-inheritance:

   Flexible input transformation network.

   **Supported transforms:**

   - ``identity``: No transformation
   - ``conv2d``: Convolutional preprocessing
   - ``mlp``: Fully connected preprocessing  
   - ``gabor``: Gabor filter banks
   - ``dog``: Difference of Gaussians
   - ``custom``: User-defined transform

Input Transforms
~~~~~~~~~~~~~~~~

Input transformation modules are available for preprocessing data before feeding to dendritic networks. These include convolutional transforms, Gabor filter banks, and other preprocessing utilities.

Specialized Architectures
-------------------------

DendriticCNN
~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.networks.architectures.dendritic.DendriticCNN
   :members:
   :undoc-members:
   :show-inheritance:

   CNN architecture with dendritic layers replacing standard convolutions.

   **Example:**

   .. code-block:: python

      cnn = DendriticCNN(
          input_channels=3,
          num_classes=10,
          channels=[32, 64, 128],
          n_branches=[10, 8, 6],
          kernel_size=3,
          use_batch_norm=True
      )

ViT_Dendrite
~~~~~~~~~~~~

.. autoclass:: dendritic_modeling.networks.architectures.dendritic.ViT_Dendrite
   :members:
   :undoc-members:
   :show-inheritance:

   Vision Transformer with dendritic attention mechanism.

Factory Functions
-----------------

Creating Visual Units
~~~~~~~~~~~~~~~~~~~~~

Factory functions are available for creating biologically-inspired visual units including Gabor units, center-surround units, motion detectors, and end-stopped units.

Converting CNNs
~~~~~~~~~~~~~~~

.. automodule:: dendritic_modeling.networks.convert.cnn_to_dendrite
   :members:

Convert pre-trained CNNs to dendritic networks:

.. autofunction:: dendritic_modeling.networks.convert.cnn_to_dendrite.convert_conv2d_to_dendrite
.. autofunction:: dendritic_modeling.networks.convert.cnn_to_dendrite.convert_model_to_dendrite

Utilities
---------

Network Analysis
~~~~~~~~~~~~~~~~

Helper functions for network analysis and visualization are available in the utils modules:

- ``count_parameters(model)``: Count trainable parameters
- ``get_activation_shapes(model, input_shape)``: Get activation shapes at each layer
- ``visualize_connectivity(layer)``: Visualize connection patterns

Configuration
-------------

Network configurations can be specified in YAML:

.. code-block:: yaml

   network:
     name: ExcitationInhibitionNetwork
     params:
       input_dim: 784
       n_layers: 3
       n_excitatory: [200, 100, 10]
       n_inhibitory: [50, 25, 0]
       n_branches: [20, 15, 10]
       connectivity: sparse
       sparsity: 0.2
       activation: relu
       use_batch_norm: true
       dropout_rate: 0.1
       epsilon: 1e-6

   input_network:
     name: conv2d
     params:
       in_channels: 1
       out_channels: 32
       kernel_size: 5
       stride: 1
       padding: 2

See Also
--------

- :doc:`models` - Model wrapper classes
- :doc:`training` - Training utilities
- :doc:`/user_guide/architecture` - Architecture overview 