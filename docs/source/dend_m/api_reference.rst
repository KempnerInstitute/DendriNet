.. _api_reference:

API Reference
=============

This section provides detailed documentation of the key classes and methods in the dendritic modeling framework.

ExcitationInhibitionNetwork
---------------------------

The ``ExcitationInhibitionNetwork`` is the core network class that implements the excitatory-inhibitory architecture.

.. code-block:: python

    class ExcitationInhibitionNetwork(nn.Module):
        """
        A network with multiple excitatory-inhibitory layers.
        
        This network implements a sequence of excitatory and inhibitory dendritic processing
        layers that model the complex interactions in biological neural networks.
        
        Parameters
        ----------
        input_dim : int
            Dimension of the input data
        excitatory_layer_sizes : list of int
            Sizes of the excitatory layers
        inhibitory_layer_sizes : list of int
            Sizes of the inhibitory layers
        excitatory_branch_factors : list of int
            Branch factors for excitatory dendrites in each layer
        inhibitory_branch_factors : list of int
            Branch factors for inhibitory dendrites in each layer
        ee_synapses_per_branch_per_layer : list of int
            Number of excitatory-to-excitatory synapses per branch in each layer
        ei_synapses_per_branch_per_layer : list of int
            Number of excitatory-to-inhibitory synapses per branch in each layer
        ie_synapses_per_branch_per_layer : list of int
            Number of inhibitory-to-excitatory synapses per branch in each layer
        ii_synapses_per_branch_per_layer : list of int
            Number of inhibitory-to-inhibitory synapses per branch in each layer
        reactivate : bool
            Whether to apply reactivation to prevent branch "death" during training
        somatic_synapses : bool
            Whether to allow direct connections to the somatic layer
        topk_init_method : str
            Initialization method for TopKLinear weights
        use_shunting : bool
            Whether to enable shunting inhibition
        gradient_strategy : str
            Strategy for gradient computation and scaling
        gradient_scale_factor : float
            Factor for scaling gradients
        enable_branch_outputs : bool
            Whether to enable access to branch outputs for visualization
        learning_strategy : str
            Strategy for training the network
        """
        
        def forward(self, x):
            """
            Forward pass through the network.
            
            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape (batch_size, input_dim)
                
            Returns
            -------
            torch.Tensor
                Output tensor of shape (batch_size, output_dim)
            """
            # Implementation details...

ExcitationInhibitionLayer
-------------------------

The ``ExcitationInhibitionLayer`` implements a single layer of excitatory and inhibitory processing.

.. code-block:: python

    class ExcitationInhibitionLayer(nn.Module):
        """
        A layer that processes inputs through excitatory and inhibitory pathways.
        
        This layer contains an excitatory dendrinet and an inhibitory dendrinet,
        which interact to produce the layer's output.
        
        Parameters
        ----------
        input_dim : int
            Dimension of the input data
        excitatory_output_dim : int
            Size of the excitatory output
        inhibitory_output_dim : int
            Size of the inhibitory output
        excitatory_branch_factor : int
            Branch factor for excitatory dendrites
        inhibitory_branch_factor : int
            Branch factor for inhibitory dendrites
        ee_synapses_per_branch : int
            Number of excitatory-to-excitatory synapses per branch
        ei_synapses_per_branch : int
            Number of excitatory-to-inhibitory synapses per branch
        ie_synapses_per_branch : int
            Number of inhibitory-to-excitatory synapses per branch
        ii_synapses_per_branch : int
            Number of inhibitory-to-inhibitory synapses per branch
        somatic_synapses : bool
            Whether to allow direct connections to the somatic layer
        use_shunting : bool
            Whether to enable shunting inhibition
        """
        
        def forward(self, excitatory_input, inhibitory_input=None):
            """
            Forward pass through the layer.
            
            Parameters
            ----------
            excitatory_input : torch.Tensor
                Excitatory input tensor
            inhibitory_input : torch.Tensor, optional
                Inhibitory input tensor
                
            Returns
            -------
            tuple of torch.Tensor
                Excitatory and inhibitory outputs
            """
            # Implementation details...

DendriNet
---------

The ``DendriNet`` class implements a dendritic network that processes inputs through a series of branch layers.

.. code-block:: python

    class DendriNet(nn.Module):
        """
        A network that models dendritic computation with branch-specific processing.
        
        Parameters
        ----------
        input_dim : int
            Dimension of the input data
        branch_layers_sizes : list of int
            Sizes of the branch layers
        branch_factors : list of int
            Branching factors for each layer
        synapses_per_branch : list of int
            Number of synapses (connections) per branch for each layer
        branch_layers_factors : list of int
            Factors for determining branch layer dimensions
        enable_outputs : bool
            Whether to enable outputs from branches
        init_method : str
            Method for initializing weights
        """
        
        def forward(self, x, inhibitory_input=None, upstream_output=None):
            """
            Forward pass through the dendritic network.
            
            Parameters
            ----------
            x : torch.Tensor
                Input tensor
            inhibitory_input : torch.Tensor, optional
                Inhibitory input for modulation
            upstream_output : torch.Tensor, optional
                Output from upstream layers
                
            Returns
            -------
            torch.Tensor
                Output tensor
            """
            # Implementation details...

DendriticBranchLayer
--------------------

The ``DendriticBranchLayer`` implements a single layer of dendritic branches.

.. code-block:: python

    class DendriticBranchLayer(nn.Module):
        """
        A layer that models computation in dendritic branches.
        
        Parameters
        ----------
        input_dim : int
            Dimension of the input data
        output_dim : int
            Dimension of the output
        inhibitory_dim : int
            Dimension of the inhibitory input
        upstream_dim : int
            Dimension of the upstream input
        branch_factor : int
            Branching factor for this layer
        synapses_per_branch : int
            Number of synapses per branch
        init_method : str
            Method for initializing weights
        use_shunting : bool
            Whether to enable shunting inhibition
        """
        
        def forward(self, x, inhibitory_input=None, upstream_output=None):
            """
            Forward pass through the branch layer.
            
            Parameters
            ----------
            x : torch.Tensor
                Input tensor
            inhibitory_input : torch.Tensor, optional
                Inhibitory input for modulation
            upstream_output : torch.Tensor, optional
                Output from upstream layers
                
            Returns
            -------
            torch.Tensor
                Output tensor
            """
            # Implementation details...

TopKLinear
----------

The ``TopKLinear`` class implements a linear transformation with top-k selection of weights.

.. code-block:: python

    class TopKLinear(nn.Module):
        """
        A linear layer that selectively activates only the top-k weights for each output unit.
        
        This models the selective activation of synapses in biological dendrites.
        
        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        bias : bool
            Whether to include a bias term
        k : int
            Number of top weights to select per output unit
        init_method : str
            Method for initializing weights
        """
        
        def forward(self, x):
            """
            Forward pass with top-k weight selection.
            
            Parameters
            ----------
            x : torch.Tensor
                Input tensor
                
            Returns
            -------
            torch.Tensor
                Output tensor
            """
            # Get weight and create top-k mask
            weight = self.weight
            
            # Create mask for selecting top-k weights
            with torch.no_grad():
                if self.k < self.in_features:
                    # Compute absolute weight values
                    abs_weight = torch.abs(weight)
                    
                    # Find top-k indices for each output unit
                    _, indices = torch.topk(abs_weight, self.k, dim=1)
                    
                    # Create binary mask with 1s for top-k weights
                    mask = torch.zeros_like(weight)
                    mask.scatter_(1, indices, 1)
                else:
                    # If k >= in_features, use all weights
                    mask = torch.ones_like(weight)
            
            # Apply mask to weights and perform linear transformation
            pruned_weight = weight * mask
            output = torch.mm(x, pruned_weight.t())
            
            if self.bias is not None:
                output += self.bias
                
            return output

BlockLinear
-----------

The ``BlockLinear`` class implements a masked linear transformation that models branching in dendrites.

.. code-block:: python

    class BlockLinear(nn.Module):
        """
        A linear layer with block structure that models dendritic branching.
        
        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        branch_factor : int
            Branching factor for creating block structure
        bias : bool
            Whether to include a bias term
        """
        
        def forward(self, x):
            """
            Forward pass with block-structured connections.
            
            Parameters
            ----------
            x : torch.Tensor
                Input tensor
                
            Returns
            -------
            torch.Tensor
                Output tensor
            """
            # Create block mask based on branch factor
            # Perform masked linear transformation
            # Implementation details...

ParametricTanh
--------------

The ``ParametricTanh`` class implements a trainable activation function that models nonlinear responses in dendrites.

.. code-block:: python

    class ParametricTanh(nn.Module):
        """
        A trainable activation function with learnable slope and offset parameters.
        
        Parameters
        ----------
        num_features : int
            Number of features (one set of parameters per feature)
        train_b : bool
            Whether to train the offset parameter b
        initial_m : float
            Initial value for slope parameter m
        initial_b : float
            Initial value for offset parameter b
        """
        
        def forward(self, x):
            """
            Forward pass with parametric tanh activation.
            
            Parameters
            ----------
            x : torch.Tensor
                Input tensor
                
            Returns
            -------
            torch.Tensor
                Output tensor after activation
            """
            # Apply parametric tanh: m * tanh(x + b)
            # Implementation details...

Learning Strategies
-------------------

The framework supports several learning strategies specifically designed for dendritic networks:

1. **MLE (Maximum Likelihood Estimation)**
   
   Standard end-to-end training with backpropagation through the entire network.

2. **Freeze Layers**
   
   Progressive training where each layer is trained sequentially while keeping previously trained layers frozen.

3. **Two-Step Training**
   
   A two-phase approach with:
   - Phase 1: Pretraining of specific components
   - Phase 2: Fine-tuning of the entire network

4. **Sequential Input Network**
   
   Training input transformation networks before the dendritic components.

Each strategy can be selected through the configuration file by setting the ``learning_strategy`` parameter. 