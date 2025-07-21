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

TopKLinear Variants
-------------------

The framework provides three variants of TopKLinear layers, each with different connection selection strategies:

1. **Standard TopKLinear**: Deterministically selects the k strongest connections for each branch.
2. **StochasticTopKLinear**: Selects connections probabilistically based on their strength rank.
3. **VarianceTopKLinear**: Selects connections based on a combination of weight magnitude and input variance.

These variants can be configured using the ``topk_strategy`` parameter:

.. code-block:: yaml

    model:
      network:
        parameters:
          topk_strategy: "stochastic"  # Options: "standard", "stochastic", "variance"
          # Additional parameters for specific strategies
          topk_temperature: 0.5        # For stochastic TopK (controls randomness)
          topk_noise_level: 0.1        # Adds noise to weight selection (all strategies)
          topk_ultrafast: true         # Enables highly optimized implementation for stochastic TopK

Standard TopKLinear
~~~~~~~~~~~~~~~~~~~

The standard TopKLinear implementation deterministically selects the k strongest inputs for each branch:

1. **Deterministic selection**: Always selects the same k connections for identical inputs
2. **Based purely on weight magnitude**: Only the absolute strength of the weight matters
3. **Sharp selection boundary**: There is a hard cutoff between selected and non-selected inputs

This approach is the fastest and most predictable, but might lead to less flexible representations.

StochasticTopKLinear
~~~~~~~~~~~~~~~~~~~~

The StochasticTopKLinear layer introduces controlled randomness to the selection process:

1. **Probabilistic selection**: Each connection has a probability of being selected based on its rank
2. **Sigmoid probability function**: P(select) = sigmoid((k - rank)/(k * temperature))
3. **Temperature parameter**: Controls the "sharpness" of the selection threshold
   - Lower temperature (e.g., 0.1): More deterministic, similar to standard TopK
   - Higher temperature (e.g., 1.0): More random, allowing weaker connections to participate

This stochastic approach has several benefits:
- Helps escape local minima during training by occasionally trying different connection patterns
- Prevents overfitting by introducing controlled noise into the model
- Closer to biological reality where synaptic activation has stochastic elements

Configure StochasticTopKLinear with:

.. code-block:: yaml

    model:
      network:
        parameters:
          topk_strategy: "stochastic"     # Select stochastic TopK
          topk_temperature: 0.5           # Control randomness (0.1-1.0)
          topk_noise_level: 0.1           # Add noise to weight values
          topk_ultrafast: true            # Enable ultrafast implementation

Ultrafast Mode for Stochastic TopK
++++++++++++++++++++++++++++++++++

The ``topk_ultrafast`` option enables a highly optimized implementation that significantly speeds up training:

1. **Simplified algorithm**: Replaces the full stochastic procedure with a noise-based approximation
2. **10-50x faster**: Greatly reduces computational overhead during training
3. **Similar behavior**: Maintains the key properties of stochastic selection

With ultrafast mode enabled, noise is directly added to the weights before a single standard TopK selection, approximating the more complex sampling procedure of the full implementation.

VarianceTopKLinear
~~~~~~~~~~~~~~~~~~

The VarianceTopKLinear layer selects connections based on both weight magnitude and input statistics:

1. **Variance tracking**: Maintains exponential moving averages of input activation statistics
2. **Information-based selection**: Prioritizes inputs with high variance (information content)
3. **Combined score**: Select based on weight_magnitude * input_variance

This approach focuses connections on inputs that:
- Carry more information (high variance = more discriminative signal)
- Already have strong weights in the model

This can be especially valuable when some inputs are more informative than others, or when dealing with noisy or redundant inputs.

Configure VarianceTopKLinear with:

.. code-block:: yaml

    model:
      network:
        parameters:
          topk_strategy: "variance"      # Select variance-based TopK
          topk_noise_level: 0.0          # Typically less noise is needed with this approach

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
          topk_strategy: "stochastic"          # Selection strategy
          topk_temperature: 0.5                # For stochastic TopK
          topk_noise_level: 0.1                # Noise level for weight selection
          topk_ultrafast: true                 # Enable faster implementation

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
5. **Stochastic activation**: Biological synapses have probabilistic activation properties

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

1. **Choosing K values**:
   - For deterministic TopK: Start with 10-20% of input dimensionality
   - For stochastic TopK: Use smaller K values (5-15% of input dim) with temperature 0.3-0.7
   - For variance-based TopK: Similar to deterministic, but monitor input statistics

2. **Training stability**:
   - Use gradient clipping (values between 1.0-10.0 work well)
   - Consider lower learning rates for stochastic variants (0.5-0.8× standard rate)

3. **Computational efficiency**:
   - Enable ultrafast mode for stochastic TopK during large-scale training
   - Use standard TopK for inference if deterministic outputs are required

4. **Hyperparameter tuning**:
   - For stochastic TopK: Sweep temperature (0.1-1.0) and noise level (0-0.2)
   - For variance TopK: Focus on momentum values (0.9-0.99)

5. **When to use each variant**:
   - Standard TopK: When deterministic behavior is needed or for baseline comparisons
   - Stochastic TopK: When training gets stuck or for more flexible representations
   - Variance TopK: When dealing with varied information content across inputs or noisy data

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
