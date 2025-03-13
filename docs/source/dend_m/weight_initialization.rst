.. _weight_initialization:

Weight Initialization Methods
===========================

Proper weight initialization is critical for training dendritic networks successfully. This document explains the initialization methods used in the dendritic modeling framework and their importance for dendritic computation.

Overview
--------

Dendritic networks have unique initialization requirements due to:

1. **Sparse connectivity**: The top-k selection in TopKLinear layers means only a subset of weights are active
2. **Branch-specific processing**: Different branches may require different initialization scales
3. **Excitatory-inhibitory balance**: Initial balance between these pathways is crucial for stable training
4. **Multiple processing layers**: Deeper networks require careful initialization to maintain signal propagation

The framework provides several initialization methods that address these challenges, with specific consideration for the selective activation patterns in dendritic branches.

TopKLinear Initialization
-----------------------

The ``TopKLinear`` layer, which models selective synaptic activation in dendrites, has specialized initialization options that can be selected through the ``topk_init_method`` parameter:

.. code-block:: yaml

    model:
      network:
        parameters:
          topk_init_method: "xavier_normal"
          init_gain: 1.0

Available initialization methods include:

1. **xavier_normal**: Xavier (Glorot) initialization with normal distribution
2. **xavier_uniform**: Xavier initialization with uniform distribution
3. **kaiming_normal**: Kaiming (He) initialization with normal distribution
4. **kaiming_uniform**: Kaiming initialization with uniform distribution
5. **sparse_normal**: Sparse initialization with normal distribution, tailored for top-k connections

Implementation details:

.. code-block:: python

    def initialize_topk_weights(weight, init_method, k, gain=1.0):
        """
        Initialize weights for TopKLinear layers.
        
        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor to initialize
        init_method : str
            Initialization method to use
        k : int
            Number of active connections per output unit
        gain : float
            Gain factor to scale the initialization
            
        Returns
        -------
        torch.Tensor
            Initialized weight tensor
        """
        fan_in, fan_out = weight.shape
        
        if init_method == 'xavier_normal':
            std = gain * math.sqrt(2.0 / (fan_in + fan_out))
            nn.init.normal_(weight, mean=0.0, std=std)
            
        elif init_method == 'xavier_uniform':
            bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
            nn.init.uniform_(weight, -bound, bound)
            
        elif init_method == 'kaiming_normal':
            # For ReLU-like activations
            nn.init.kaiming_normal_(weight, a=0, mode='fan_in', nonlinearity='relu')
            
        elif init_method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(weight, a=0, mode='fan_in', nonlinearity='relu')
            
        elif init_method == 'sparse_normal':
            # Specialized for top-k connections
            # Scale by sqrt(fan_in/k) to account for sparse activation
            std = gain * math.sqrt(1.0 / k)
            nn.init.normal_(weight, mean=0.0, std=std)
            
        return weight

Xavier vs. Kaiming Initialization
-------------------------------

For dendritic networks, the choice between Xavier and Kaiming initialization depends on the activation functions and network depth:

1. **Xavier initialization** (also called Glorot initialization):
   - Designed for symmetric activations like tanh
   - Maintains variance across layers by considering both fan-in and fan-out
   - Works well with ParametricTanh activations in dendritic networks
   - Better for shallower networks

2. **Kaiming initialization** (also called He initialization):
   - Designed for ReLU-like activations with positive skew
   - Focuses on fan-in to maintain variance for forward pass
   - May need adaptation for ParametricTanh (adjusting the 'a' parameter)
   - Better for deeper networks

The init_gain Parameter
---------------------

The ``init_gain`` parameter scales the standard deviation or bounds of the initialization:

.. code-block:: yaml

    model:
      network:
        parameters:
          init_gain: 1.0  # Default value

A higher gain increases the initial magnitude of weights, which can help with vanishing gradients but risks saturation. Values between 0.5 and 2.0 are most common.

Guidelines for different activation functions:
- For ParametricTanh with default parameters: gain ≈ 1.0
- For steeper ParametricTanh (higher m): gain ≈ 0.5-0.8
- For flatter ParametricTanh (lower m): gain ≈ 1.2-1.5

Inhibitory Weight Initialization
------------------------------

Inhibitory connections often require different initialization to maintain proper excitation-inhibition balance. The framework provides the ``init_inhib_offset`` parameter:

.. code-block:: yaml

    model:
      network:
        parameters:
          init_inhib_offset: 1.3

This parameter affects how inhibitory weights are initialized:

1. Values > 1.0 initialize inhibitory weights with larger magnitude
2. Values < 1.0 initialize inhibitory weights with smaller magnitude
3. Value of 1.0 treats inhibitory weights the same as excitatory

Proper initialization of inhibitory weights helps establish the desired excitatory-inhibitory balance from the beginning of training.

Implementation:

.. code-block:: python

    # For inhibitory pathway initialization
    if pathway == 'inhibitory':
        std = std * init_inhib_offset  # Scale the standard deviation
        nn.init.normal_(weight, mean=0.0, std=std)

Sparse Initialization for TopK
----------------------------

For TopKLinear layers, standard initialization methods may not be optimal due to the sparse connectivity pattern. The sparse_normal method adjusts the scale based on the number of active connections:

.. code-block:: python

    def sparse_topk_init(weight, k, gain=1.0):
        """
        Initialize weights specifically for TopKLinear with consideration for sparsity.
        
        Parameters
        ----------
        weight : torch.Tensor
            The weight tensor to initialize
        k : int
            Number of active connections per output
        gain : float
            Scaling factor
            
        Returns
        -------
        torch.Tensor
            Initialized weight tensor
        """
        fan_in = weight.size(1)
        # Scale by sqrt(fan_in/k) to account for only k active inputs
        adjustment = math.sqrt(fan_in / k)
        std = gain / adjustment
        
        return nn.init.normal_(weight, mean=0.0, std=std)

This accounts for the fact that during forward propagation, only k out of fan_in inputs will be active for each output.

BlockLinear Initialization
------------------------

For BlockLinear layers, which handle information flow between branches, initialization needs to consider the block structure:

.. code-block:: python

    def initialize_block_linear(weight, branch_factor, init_method='xavier_normal', gain=1.0):
        """
        Initialize BlockLinear weights with consideration for block structure.
        
        Parameters
        ----------
        weight : torch.Tensor
            Weight tensor to initialize
        branch_factor : int
            Branch factor that determines block structure
        init_method : str
            Initialization method
        gain : float
            Gain factor
            
        Returns
        -------
        torch.Tensor
            Initialized weight tensor
        """
        # Calculate effective fan_in based on branch structure
        effective_fan_in = weight.size(1) // branch_factor
        
        if init_method == 'xavier_normal':
            std = gain * math.sqrt(2.0 / (effective_fan_in + weight.size(0)))
            nn.init.normal_(weight, mean=0.0, std=std)
        
        # Create block mask
        mask = create_block_mask(weight.size(0), weight.size(1), branch_factor)
        
        # Apply mask to zero out non-block connections
        return weight * mask

Initialization of ParametricTanh Parameters
----------------------------------------

The learnable parameters in ParametricTanh activations (m and b) also require careful initialization:

.. code-block:: yaml

    model:
      network:
        parameters:
          reactivation_initial_m: 2.0  # Initial slope parameter
          reactivation_initial_b: 0.5  # Initial offset parameter

These values strongly influence the initial response properties of dendritic branches:

1. **reactivation_initial_m**: Controls the initial slope of the tanh function
   - Higher values create steeper curves with stronger saturation
   - Lower values create more gradual, linear-like responses
   - Biological dendrites often exhibit steep nonlinearities, so values ≥ 1.0 are common

2. **reactivation_initial_b**: Controls the initial offset of the tanh function
   - Positive values shift the sensitive region toward negative inputs
   - Negative values shift the sensitive region toward positive inputs
   - The optimal value depends on the distribution of pre-activation values

Best Practices for Initialization
-------------------------------

For optimal performance when training dendritic networks:

1. **Match initialization to activation**: Use xavier_normal for ParametricTanh, kaiming for ReLU-like functions
2. **Adjust for network depth**: Deeper networks may need lower gain values
3. **Balance excitation-inhibition**: Adjust init_inhib_offset based on network architecture
4. **Monitor initial activations**: Check activation statistics at initialization to verify signal propagation
5. **Consider layer-specific initialization**: Different layers may benefit from different initialization parameters

Example complete initialization configuration:

.. code-block:: yaml

    model:
      network:
        parameters:
          topk_init_method: "xavier_normal"
          init_gain: 1.0
          init_inhib_offset: 1.3
          reactivation_initial_m: 2.0
          reactivation_initial_b: 0.5 