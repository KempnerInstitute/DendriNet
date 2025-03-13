.. _gradient_scaling:

Gradient Scaling Techniques
==========================

Dendritic networks employ specialized gradient scaling techniques to address unique training challenges. This document explains these techniques in detail.

Overview
--------

Effective training of dendritic networks requires careful management of gradient flow. The complex branching structure, selective activation through top-k connections, and interaction between excitatory and inhibitory pathways can lead to:

1. **Vanishing gradients**: Especially in deeper branches or layers
2. **Exploding gradients**: Particularly through highly active branches
3. **Unbalanced learning**: Where some branches learn much faster than others
4. **Gradient sparsity**: Due to the selective activation in TopKLinear layers

The framework implements several gradient scaling strategies to address these challenges, which can be selected using the ``gradient_strategy`` parameter in your model configuration.

Available Gradient Strategies
---------------------------

.. code-block:: yaml

    model:
      network:
        parameters:
          gradient_strategy: "block_conductance_dynamic"
          gradient_scale_factor: 2.0

The available strategies include:

1. **block_conductance**: Scale gradients based on block conductance
2. **block_conductance_dynamic**: Dynamically adjust gradient scaling based on network activity
3. **none**: No special gradient handling (standard backpropagation)
4. **conductance**: Scale gradients based on overall network conductance

Block Conductance Strategy
------------------------

The block conductance strategy scales gradients based on the "conductance" of each dendritic branch. Conductance here refers to how strongly a branch responds to its inputs.

This strategy:
1. Computes a measure of branch conductance based on activation patterns
2. Scales gradients inversely proportional to the conductance
3. Helps prevent branches with strong activation from dominating learning

Implementation details:

.. code-block:: python

    def block_conductance_scaling(grads, activations, scale_factor):
        """
        Scale gradients based on block conductance.
        
        Parameters
        ----------
        grads : torch.Tensor
            The input gradients
        activations : torch.Tensor
            The activations of the branch
        scale_factor : float
            Factor to control the scaling strength
            
        Returns
        -------
        torch.Tensor
            Scaled gradients
        """
        # Compute conductance as a function of activations
        conductance = compute_conductance(activations)
        
        # Scale gradients inversely with conductance
        scaled_grads = grads / (conductance * scale_factor + 1.0)
        
        return scaled_grads

Dynamic Block Conductance Strategy
--------------------------------

The dynamic block conductance strategy extends the basic approach by adaptively adjusting the scaling based on network activity during training.

Key features:
1. Monitors overall network activity and gradient magnitudes
2. Dynamically adjusts scaling factors for different branches
3. Can increase or decrease scaling based on recent learning progress
4. Helps maintain balanced learning across different branches

This strategy is particularly useful for deep dendritic networks with many branches, where static scaling might not be optimal throughout training.

Implementation details:

.. code-block:: python

    def dynamic_block_conductance_scaling(grads, activations, scale_factor, history):
        """
        Dynamically scale gradients based on block conductance and learning history.
        
        Parameters
        ----------
        grads : torch.Tensor
            The input gradients
        activations : torch.Tensor
            The activations of the branch
        scale_factor : float
            Base factor to control the scaling strength
        history : dict
            Dictionary containing learning history
            
        Returns
        -------
        torch.Tensor
            Scaled gradients
        """
        # Compute conductance as in the basic strategy
        conductance = compute_conductance(activations)
        
        # Compute dynamic adjustment factor based on learning history
        adjustment = compute_adjustment_factor(history)
        
        # Apply dynamic scaling
        effective_scale = scale_factor * adjustment
        scaled_grads = grads / (conductance * effective_scale + 1.0)
        
        return scaled_grads

Gradient Scale Factor
-------------------

The ``gradient_scale_factor`` parameter controls the strength of gradient scaling. Higher values result in more aggressive scaling, which can help with very deep networks but might slow down learning. Lower values apply more subtle scaling.

Guidelines for setting this parameter:
- Start with values between 1.0 and 2.0
- Increase for deeper networks or if training is unstable
- Decrease if learning seems too slow

Custom Gradient Scaling
---------------------

You can implement custom gradient scaling by creating a custom ``torch.autograd.Function`` that applies your scaling logic during the backward pass:

.. code-block:: python

    class CustomGradientScaling(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, scaling_args):
            ctx.save_for_backward(input)
            ctx.scaling_args = scaling_args
            return input.clone()
            
        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            scaling_args = ctx.scaling_args
            
            # Apply custom scaling logic
            scaled_grad = your_custom_scaling_function(grad_output, input, scaling_args)
            
            return scaled_grad, None

Interaction with Activation Functions
-----------------------------------

Gradient scaling interacts with activation functions, particularly the ``ParametricTanh`` used in dendritic branches:

1. The adaptive slope parameter in ParametricTanh effectively scales gradients
2. Branches with high activation values through tanh can have small gradients due to saturation
3. The gradient scaling strategies help counteract these effects

Best Practices
------------

When using gradient scaling:

1. **Monitor gradient magnitudes**: Use the visualization tools to track gradient distributions
2. **Adjust scaling during training**: Start with stronger scaling and reduce if learning stalls
3. **Combine with gradient clipping**: Use the ``grad_clip_value`` parameter to prevent extreme values
4. **Experiment with different strategies**: The optimal strategy may depend on your specific architecture

Integration with Optimizers
-------------------------

Gradient scaling happens before the optimization step. The framework integrates with various optimizers:

1. **Adam**: Works well with gradient scaling, often the best choice
2. **SGD**: May require more careful tuning of scaling factors
3. **AdamW**: Combines well with weight decay and gradient scaling

Example configuration with Adam:

.. code-block:: yaml

    train:
      optimizer:
        name: "adam"
        lr: 0.001
        weight_decay: 0.0001
      grad_clip_value: 5.0
      
    model:
      network:
        parameters:
          gradient_strategy: "block_conductance_dynamic"
          gradient_scale_factor: 1.5 