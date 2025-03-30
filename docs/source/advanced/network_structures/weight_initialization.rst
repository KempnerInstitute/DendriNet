.. _weight_initialization:

Weight Initialization Methods
=============================

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
-------------------------

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
5. **orthogonal**: Orthogonal initialization
6. **normal**: Standard normal distribution initialization
7. **uniform**: Uniform distribution initialization
8. **eye**: Identity matrix initialization

Implementation details:

The initialization is handled internally by the TopKLinear layer:

.. code-block:: python

    TOPK_INIT_METHODS = {
        'xavier_normal': nn.init.xavier_normal_,
        'xavier_uniform': nn.init.xavier_uniform_,
        'kaiming_normal': nn.init.kaiming_normal_,
        'kaiming_uniform': nn.init.kaiming_uniform_,
        'orthogonal': nn.init.orthogonal_,
        'normal': nn.init.normal_,
        'uniform': nn.init.uniform_,
        'eye': nn.init.eye_,
    }

    def initialize(self):
        if self.init_method in TOPK_INIT_METHODS:
            init_func = TOPK_INIT_METHODS[self.init_method]
            init_func(self.pre_w)
            self.pre_w.data.mul_(self.init_gain)  # scale
        else:
            raise ValueError(
                f'Invalid initialization method: {self.init_method}. ',
                f'Choose from {list(TOPK_INIT_METHODS.keys())}')

Xavier vs. Kaiming Initialization
---------------------------------

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
-----------------------

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

BlockLinear Initialization
--------------------------

For BlockLinear layers, which handle information flow between branches, initialization is handled internally by the layer. The layer ensures that the input features are divisible by the output features to maintain proper block structure.

Initialization of ParametricTanh Parameters
-------------------------------------------

The learnable parameters in ParametricTanh activations (m and b) also require careful initialization:

.. code-block:: yaml

    model:
      network:
        parameters:
          reactivation_initial_m: 1.5  # Initial slope parameter (default)
          reactivation_initial_b: 1.0  # Initial offset parameter (default)

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
---------------------------------

For optimal performance when training dendritic networks:

1. **Match initialization to activation**: Use xavier_normal for ParametricTanh, kaiming for ReLU-like functions
2. **Adjust for network depth**: Deeper networks may need lower gain values
3. **Monitor initial activations**: Check activation statistics at initialization to verify signal propagation
4. **Consider layer-specific initialization**: Different layers may benefit from different initialization parameters

Example complete initialization configuration:

.. code-block:: yaml

    model:
      network:
        parameters:
          topk_init_method: "xavier_normal"
          init_gain: 1.0
          reactivation_initial_m: 1.5
          reactivation_initial_b: 1.0
