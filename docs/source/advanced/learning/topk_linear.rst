.. _topk_linear:

TopKLinear Layers and Dendritic Processing
==========================================

The TopKLinear layer is the core building block of dendritic computation in the framework. This document explains how TopKLinear layers model biological dendritic computation and how they can be configured.

Overview
--------

Biological dendrites perform selective integration of inputs by activating only the strongest synaptic connections. The ``TopKLinear`` layer implements this selective activation mechanism by:

1. Computing the potential contribution of all inputs
2. Selecting the top k strongest inputs for each output unit
3. Summing only the selected inputs to produce the output

This selective integration is a fundamental departure from standard neural network layers and allows dendritic networks to implement powerful computational properties like context-dependent processing and feature binding.

TopKLinear Architecture
-----------------------

The ``TopKLinear`` layer extends the standard linear layer with a top-k selection mechanism:

.. code-block:: python

    class TopKLinear(nn.Module):
        """
        Linear layer with top-k activation.

        This layer processes inputs by selecting the top k inputs for each output unit,
        modeling the selective integration property of biological dendrites.

        Parameters
        ----------
        in_features : int
            Number of input features
        out_features : int
            Number of output features
        k : int
            Number of inputs to select for each output unit
        bias : bool
            Whether to use a bias term
        """

        def __init__(self, in_features, out_features, k, bias=True):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.k = k

            # Initialize weights and bias
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.Tensor(out_features))
            else:
                self.register_parameter('bias', None)

            # Initialize parameters
            self.reset_parameters()

        def forward(self, input):
            # Compute all potential contributions
            potential = F.linear(input, self.weight, None)

            # Get the indices of the top k inputs for each output unit
            _, indices = torch.topk(potential, self.k, dim=1)

            # Create a mask for the selected inputs
            mask = torch.zeros_like(potential)
            mask.scatter_(1, indices, 1)

            # Apply the mask to select only the top k inputs
            selected = potential * mask

            # Sum the selected inputs and add bias if present
            if self.bias is not None:
                return selected.sum(dim=1) + self.bias
            else:
                return selected.sum(dim=1)

The key components of the TopKLinear layer are:

1. **Weight matrix** (out_features × in_features): Stores the connection strengths
2. **k parameter**: Controls how many inputs are selected for each output
3. **Selection mechanism**: Identifies and activates only the strongest connections
4. **Masked summation**: Sums only the selected inputs

Configuration
-------------

The TopKLinear layer is configured through several parameters:

.. code-block:: yaml

    model:
      network:
        parameters:
          # TopKLinear configuration
          excitatory_branch_factors: [64, 32]  # k values for excitatory layers
          inhibitory_branch_factors: [32, 16]  # k values for inhibitory layers
          topk_init_method: "xavier_normal"    # Weight initialization method

The ``branch_factors`` parameters define the k value for each layer, which determines how many inputs are selected by each dendritic branch.

Branch Factor Selection
-----------------------

The branch factor (k) is a critical parameter that affects the computational properties of the dendritic network:

1. **Small k values** (1-10):
   - More selective branches that respond to specific input patterns
   - Greater sparsity and specialization
   - May require more branches to cover the input space

2. **Medium k values** (10-100):
   - Balance between selectivity and generalization
   - Good for most applications
   - Approximate biological values for many neural types

3. **Large k values** (>100):
   - Less selective, more like standard linear layers
   - Good for dense, distributed representations
   - May require fewer branches overall

The optimal branch factor depends on the input dimensionality and the desired properties of the network. A common rule of thumb is to set k to approximately 10-20% of the input dimensionality for each branch.

Biological Intuition
--------------------

The TopKLinear layer is inspired by several properties of biological dendrites:

1. **Dendritic spikes**: Dendrites generate local spikes when sufficient inputs are activated
2. **Nonlinear integration**: Inputs that arrive on the same dendritic branch can interact nonlinearly
3. **Branch-specific processing**: Each dendritic branch can function as a semi-independent processing unit
4. **Sparse activation**: Only a subset of synapses are strongly active for any given input

The top-k selection mechanism is a simplified model of these biological properties, capturing the essential computational feature of selective integration.

Forward Pass Implementation
---------------------------

The forward pass in TopKLinear involves several key steps:

1. **Linear projection**: Calculate the potential contribution of each input
2. **Top-k selection**: Select the k highest-potential inputs
3. **Masked summation**: Sum only the selected inputs

Implementation details:

.. code-block:: python

    def forward(self, input):
        # Compute all potential contributions
        # Shape: [batch_size, in_features]
        potential = F.linear(input, self.weight, None)

        # Get the indices of the top k inputs for each output unit
        # Shape of values: [batch_size, k]
        # Shape of indices: [batch_size, k]
        values, indices = torch.topk(potential, self.k, dim=1)

        # Create a mask for the selected inputs
        # Shape: [batch_size, in_features]
        mask = torch.zeros_like(potential)
        mask.scatter_(1, indices, 1)

        # Apply the mask to select only the top k inputs
        # Shape: [batch_size, in_features]
        selected = potential * mask

        # Sum the selected inputs and add bias if present
        # Shape: [batch_size, out_features]
        if self.bias is not None:
            return selected.sum(dim=1) + self.bias
        else:
            return selected.sum(dim=1)

Reactivation Strategies
-----------------------

The framework supports reactivation strategies that modify the behavior of TopKLinear layers:

1. **None**: Standard top-k selection without any modification
2. **Inverse**: Scales the activations by the inverse of the sum, encouraging more balanced activation
3. **Param_tanh**: Uses a parametric tanh activation after selection for smoother activation profiles

These can be configured in the YAML configuration:

.. code-block:: yaml

    model:
      network:
        parameters:
          reactivation_strategy: "inverse"  # Options: "none", "inverse"
          reactivation_type: "param_tanh"   # Options: "none", "param_tanh", "relu", "sigmoid", "tanh"

Gradient Flow in TopKLinear
---------------------------

The top-k selection creates a sparse computational graph that affects gradient flow:

1. During the forward pass, only k inputs are selected
2. During backpropagation, gradients flow only through the selected inputs
3. This creates sparse gradients that can lead to unstable training

To address this, the framework uses gradient clipping and appropriate initialization strategies to stabilize training.

Comparison with Standard Linear Layers
--------------------------------------

Compared to standard linear layers, TopKLinear has several key differences:

1. **Computational complexity**:
   - Forward pass: O(n log k) for top-k selection vs. O(n) for standard linear
   - More computationally intensive due to sorting operation

2. **Memory usage**:
   - Similar number of parameters
   - Higher memory usage during training due to storing indices and masks

3. **Representational power**:
   - Can implement nonlinear operations that standard linear layers cannot
   - Can adapt its computational behavior to different inputs (context-dependent processing)

4. **Sparsity**:
   - Naturally sparse activations (only k connections are active)
   - Sparse gradient flow during training

BlockLinear Strategies
---------------------

The framework implements different blocklinear strategies that affect how inputs are processed:

1. **None**: No special processing
2. **Distal upweight by idx**: Upweights inputs based on their distance
3. **Block conductance dynamic**: Uses a dynamic conductance mechanism

Configure these strategies in the YAML configuration:

.. code-block:: yaml

    model:
      network:
        parameters:
          blocklinear_strategy: "block_conductance_dynamic"  # Options: "none", "distal_upweight_by_idx", "block_conductance_dynamic"

Best Practices
--------------

For optimal performance with TopKLinear layers:

1. **Start with reasonable branch factors**:
   - For initial experimentation, set k to 10-20% of the input dimension
   - Adjust based on performance and desired properties

2. **Use appropriate reactivation strategies**:
   - The "inverse" reactivation strategy often works well with dendritic networks
   - Param_tanh reactivation type provides smooth activation profiles

3. **Combine with appropriate activation functions**:
   - Use reactivation_type param_tanh for better performance
   - Consider the interaction between selectivity and nonlinearity

4. **Monitor branch activation**:
   - Track which branches are active during training
   - Ensure branches aren't "dying" (never being selected)

5. **Adjust learning rates carefully**:
   - Use gradient clipping to stabilize training
   - Start with conservative learning rates (0.001) and adjust as needed

Example Configuration
---------------------

Here's a complete configuration example for a network using TopKLinear layers:

.. code-block:: yaml

    model:
      network:
        type: "EINet"
        parameters:
          input_dim: 784
          excitatory_layer_sizes: [512, 256]
          inhibitory_layer_sizes: [200, 100]
          excitatory_branch_factors: [64, 32]
          inhibitory_branch_factors: [32, 16]
          reactivate: true
          somatic_synapses: true
          topk_init_method: "xavier_normal"
          reactivation_type: "param_tanh"
          reactivation_strategy: "inverse"
          blocklinear_strategy: "block_conductance_dynamic"
          output_dim: 10
