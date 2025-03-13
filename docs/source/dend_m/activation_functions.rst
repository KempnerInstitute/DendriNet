.. _activation_functions:

Activation Functions
==================

This document explains the specialized activation functions available in the dendritic modeling framework for modeling nonlinear response properties of biological dendrites.

Overview
--------

Biological dendrites exhibit complex nonlinear response properties that are crucial for their computational capabilities. The dendritic modeling framework provides several activation functions to model these properties, with a particular focus on parametric activations that can learn the specific nonlinearities for each dendritic branch.

Available Activation Functions
---------------------------

The framework implements several activation functions in ``activation_functions.py``:

.. code-block:: yaml

    model:
      network:
        parameters:
          reactivation_type: param_tanh  # Options: "none", "param_tanh", "param_tanh_only_m", "relu", "sigmoid", "tanh"
          reactivation_initial_m: 1.5    # Initial slope parameter for param_tanh
          reactivation_initial_b: 1.0    # Initial offset parameter for param_tanh

These activation functions are created using the ``ActivationFactory`` class, which provides a unified interface for creating different activation types.

ParametricTanh
-----------

The ``ParametricTanh`` is a specialized activation function with learnable parameters ``m`` (slope) and ``b`` (offset):

.. code-block:: python

    class ParametricTanh(nn.Module):
        def __init__(self, output_dim, init_m=1.5, init_b=1.0):
            super(ParametricTanh, self).__init__()
            self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)
            self.log_b = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)

            nn.init.constant_(self.log_m, math.log(init_m))
            nn.init.constant_(self.log_b, math.log(init_b))
            
        def forward(self, V):
            m = self.log_m.exp()  # slope
            b = self.log_b.exp()  # midpoint
            return (torch.tanh(m * (V - b)) + 1) / 2

This activation computes a normalized sigmoid-like response from the membrane potential ``V``, with:
- ``m`` controlling the steepness of the response (higher values give sharper transitions)
- ``b`` controlling the threshold/midpoint of the response

The parameters are stored in log space to ensure positivity and for more stable gradient-based optimization. The ``init_m`` and ``init_b`` parameters allow control over the initial parameter values.

ParametricTanhOnlyM
----------------

The ``ParametricTanhOnlyM`` is a variant where only the slope parameter ``m`` is learnable, while the offset ``b`` is fixed:

.. code-block:: python

    class ParametricTanhOnlyM(nn.Module):
        def __init__(self, output_dim, init_m=1.5, fixed_b=0.5):
            super(ParametricTanhOnlyM, self).__init__()
            self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad=True)
            self.fixed_b = fixed_b  # fixed midpoint value
            with torch.no_grad():
                self.log_m.data.fill_(math.log(init_m))

            self.register_full_backward_hook(
                lambda mod, g_in, g_out: scale_activation_grad_hook(mod, g_in, g_out)
            )
                        
        def forward(self, V):
            m = self.log_m.exp()  # slope
            b = self.fixed_b       # fixed midpoint
            return (torch.tanh(m * (V - b)) + 1) / 2

This implementation also registers a gradient scaling hook to help with training stability. The hook is defined in ``gradient_scaling.py`` and helps ensure appropriate gradient flow through the activation function.

Standard Activation Functions
--------------------------

The framework also provides standard activation functions with consistent interfaces:

1. **SimpleReLU**: Standard ReLU activation
   
   .. code-block:: python
   
       class SimpleReLU(nn.Module):
           def __init__(self):
               super(SimpleReLU, self).__init__()
           def forward(self, x):
               return torch.relu(x)

2. **SimpleSigmoid**: Standard sigmoid activation
   
   .. code-block:: python
   
       class SimpleSigmoid(nn.Module):
           def __init__(self):
               super(SimpleSigmoid, self).__init__()
           def forward(self, x):
               return torch.sigmoid(x)

3. **SimpleTanh**: Standard tanh activation
   
   .. code-block:: python
   
       class SimpleTanh(nn.Module):
           def __init__(self):
               super(SimpleTanh, self).__init__()
           def forward(self, x):
               return torch.tanh(x)

ActivationFactory
--------------

To simplify creating the right activation function, the framework provides an ``ActivationFactory`` class:

.. code-block:: python

    class ActivationFactory:
        @staticmethod
        def create(act_type, output_dim=None, init_m=1.5, init_b=1.0, fixed_b=0.5, **kwargs):
            """
            Create an activation module of the desired type.

            Parameters
            ----------
            act_type : str
                One of ["none", "param_tanh", "param_tanh_only_m", "relu", "sigmoid", "tanh"].
            output_dim : int or None
                Used by param_tanh and param_tanh_only_m.
            init_m : float
                Initial value for m.
            init_b : float
                Initial value for b (used only for param_tanh).
            fixed_b : float
                The fixed value of b when using param_tanh_only_m.
            **kwargs : dict
                Additional keyword arguments are accepted but ignored.

            Returns
            -------
            nn.Module
                The activation module.
            """
            act_type = act_type.lower()
            if act_type == "none":
                return nn.Identity()
            elif act_type == "param_tanh":
                if output_dim is None:
                    raise ValueError("ParametricTanh requires output_dim.")
                return ParametricTanh(output_dim, init_m, init_b)
            elif act_type == "param_tanh_only_m":
                if output_dim is None:
                    raise ValueError("ParametricTanhOnlyM requires output_dim.")
                return ParametricTanhOnlyM(output_dim, init_m, fixed_b)
            elif act_type == "relu":
                return SimpleReLU()
            elif act_type == "sigmoid":
                return SimpleSigmoid()
            elif act_type == "tanh":
                return SimpleTanh()
            else:
                raise ValueError(f"Unknown activation type: {act_type}")

This factory pattern makes it easy to create the appropriate activation function based on the configuration.

Configuration
-----------

The activation functions are configured in the model section of the configuration:

.. code-block:: yaml

    model:
      network:
        parameters:
          reactivation_type: param_tanh  # Type of activation function
          reactivation_strategy: inverse  # Gradient scaling strategy for reactivation
          reactivate: true               # Whether to use reactivation
          use_shunting: true             # Whether to use shunting inhibition

The ``reactivation_strategy`` parameter affects how gradients flow through the activation, with the "inverse" strategy implemented in the gradient scaling module.

Usage in the Network
-----------------

The activation functions are primarily used in dendritic branch layers, where they model the response of dendritic branches to incoming signals. The ``ExcitationInhibitionLayer`` in ``networks.py`` uses these activations to process dendritic inputs.

Biological Intuition
------------------

The parametric activation functions are motivated by biological observations:

1. **Sigmoidal response**: Dendritic branches show sigmoidal voltage responses to increasing input
2. **Variable thresholds**: Different branches can have different activation thresholds
3. **Activity-dependent plasticity**: The response properties of dendrites can change with experience

The ``ParametricTanh`` activation captures these properties by allowing each branch to learn its own response curve through the ``m`` and ``b`` parameters.

Best Practices
-----------

When working with dendritic activations:

1. **Initial parameters**: Start with reasonable initial parameters (typical values: ``init_m=1.5``, ``init_b=1.0``)
2. **Gradient scaling**: Use appropriate gradient scaling to ensure stable training
3. **Monitor parameters**: Track the learned ``m`` and ``b`` values to understand branch specialization
4. **Consider constraints**: For some applications, ``ParametricTanhOnlyM`` with fixed ``b`` may be more stable

By carefully configuring activation functions, you can create dendritic networks that capture the computational properties of biological dendrites. 