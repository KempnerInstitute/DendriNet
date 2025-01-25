"""
gradient_scaling.py
===================
This module defines classes/functions for gradient-scaling strategies,
removing direct references from dendritic modules. We collect all
gradient-scaling logic here and can apply it to modules from the outside.
"""

import torch

def scale_layer_gradients(module, grad_input, grad_output, scale_factor):
    """
    Original function from dendrinet.py, now relocated here.
    Scales layer gradients by a constant factor.
    """
    if grad_input is None:
        return None
    return tuple(
        g * scale_factor if (g is not None) else None
        for g in grad_input
    )

def scale_reactivation_gradients(module, grad_input, grad_output):
    """
    Original function from dendrinet.py, now relocated here.
    By default, does no special scaling, but you can adapt it.
    """
    return grad_output

class GradientScaler:
    """
    A class to manage registration of gradient-scaling hooks and
    methods based on a config or argument.
    """

    def __init__(self, strategy="none", scale_factor=1.0, layer_idx=0):
        """
        Parameters
        ----------
        strategy : str
            The gradient scaling strategy (e.g. 'none', 'uniform_scale',
            'distal_upweight', 'distal_upweight_by_idx', etc.).
        scale_factor : float
            Base scaling factor used by some strategies.
        layer_idx : int
            Layer index, if needed by certain strategies (e.g. 'distal_upweight_by_idx').
        """
        self.strategy = strategy
        self.scale_factor = scale_factor
        self.layer_idx = layer_idx

    def compute_scale(self, block_linear_obj=None):
        """
        Decide how to compute the scale for the gradient.

        If you had more advanced logic, you could incorporate or
        read data from 'block_linear_obj' or the layer's state.
        """
        if self.strategy == 'uniform_scale':
            return self.scale_factor

        elif self.strategy == 'distal_upweight':
            # If you have a block_linear or advanced logic
            # that sets e.g. block_linear_obj.grad_scale, you can use it
            if block_linear_obj and hasattr(block_linear_obj, 'grad_scale'):
                return block_linear_obj.grad_scale.mean().item() * self.scale_factor
            else:
                return self.scale_factor

        elif self.strategy == 'distal_upweight_by_idx':
            # A simpler layer-based formula: scale = (1 + layer_idx)* factor
            return (1 + self.layer_idx) * self.scale_factor

        else:
            return 1.0

    def register_block_linear(self, block_linear_layer):
        """
        Attaches a backward hook to the BlockLinear object to scale gradients
        according to the chosen strategy.
        """
        def hook_fn(module, grad_input, grad_output):
            s = self.compute_scale(block_linear_obj=block_linear_layer)
            return scale_layer_gradients(module, grad_input, grad_output, s)

        block_linear_layer.register_full_backward_hook(hook_fn)

    def register_reactivation(self, reactivation_layer):
        """
        Attaches a backward hook to the Reactivation object (for example).
        """
        reactivation_layer.register_full_backward_hook(scale_reactivation_gradients)

    def register_generic_layer(self, layer):
        """
        Attaches a backward hook to a generic layer (like a DendriticBranchLayer).
        """
        def hook_fn(module, grad_input, grad_output):
            s = self.compute_scale()
            return scale_layer_gradients(module, grad_input, grad_output, s)

        layer.register_full_backward_hook(hook_fn)