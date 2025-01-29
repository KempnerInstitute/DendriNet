"""
gradient_scaling.py
===================
This module defines classes/functions for gradient-scaling strategies,
removing direct references from dendritic modules. 

Possible gradient strategies for the config:
    - 'none'
    - 'uniform_scale'
    - 'distal_upweight'
    - 'distal_upweight_by_idx'
    - 'block_conductance_dynamic'
      (uses total conductance from BlockLinear + pruned weights)
"""

import torch

def scale_layer_gradients(module, grad_input, grad_output, scale_factor):
    """
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
    For now, does no special scaling. We can adapt it if needed later. #TODO
    """
    return grad_output

class GradientScaler:
    """
    A class to manage registration of gradient-scaling hooks and
    methods based on a config or argument.

    The recognized strategies are:
      "none"                 -> no special scaling
      "uniform_scale"        -> multiply gradient by scale_factor
      "distal_upweight"      -> multiply gradient by scale_factor * (optionally block.grad_scale)
      "distal_upweight_by_idx" -> multiply gradient by (layer_idx+1)*scale_factor
      "block_conductance_dynamic" -> dynamic scale using sum of conduction from block + excit/inhib

    Typically, we instantiate with:
        scaler = GradientScaler(strategy='block_conductance_dynamic')
    and call:
        scaler.register_block_linear_dynamic(my_block, exc_layer, inh_layer)
    to attach a backward hook that does the dynamic scaling.
    """

    def __init__(self, strategy="none", scale_factor=1.0, layer_idx=0):
        """
        Parameters
        ----------
        strategy : str
            The gradient scaling strategy (e.g. 'none', 'uniform_scale',
            'distal_upweight', 'distal_upweight_by_idx', 'block_conductance_dynamic').
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

        we might come up with more advanced logic, and incorporate or
        read data from 'block_linear_obj' or the layer's state.
        """
        if self.strategy == 'uniform_scale':
            return self.scale_factor

        elif self.strategy == 'distal_upweight':
            if block_linear_obj and hasattr(block_linear_obj, 'grad_scale'):
                return block_linear_obj.grad_scale.mean().item() * self.scale_factor
            else:
                return self.scale_factor

        elif self.strategy == 'distal_upweight_by_idx':
            # A simpler layer-based formula: scale = (1 + layer_idx)* factor
            return (1 + self.layer_idx) * self.scale_factor

        elif self.strategy == 'block_conductance_dynamic':
            # We'll just return 1.0 here; the actual dynamic logic
            # is done in register_block_linear_dynamic's hook.
            return 1.0

        else:
            # 'none' or unknown
            return 1.0

    def register_block_linear(self, block_linear_layer):
        """
        Attaches a backward hook to the BlockLinear object to scale gradients
        according to the chosen strategy (constant).
        """
        def hook_fn(module, grad_input, grad_output):
            s = self.compute_scale(block_linear_obj=block_linear_layer)
            return scale_layer_gradients(module, grad_input, grad_output, s)

        block_linear_layer.register_full_backward_hook(hook_fn)

    def register_reactivation(self, reactivation_layer):
        """
        Attaches a backward hook to a reactivation module.
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

    def register_block_linear_dynamic(self, block_linear_layer, branch_excitation=None, branch_inhibition=None):
        """
        A  method that uses the old 'set_branches_to_output_grad_scale' logic 
        dynamically during backward if strategy=='block_conductance_dynamic'.
        We compute the scale factor from sum of conductances in block_linear + 
        excit/inhib pruned weights, then multiply the gradient.

        branch_excitation, branch_inhibition are references to the 
        TopKLinear modules for excitatory or inhibitory inputs.
        """
        def hook_fn(module, grad_input, grad_output):
            if self.strategy != 'block_conductance_dynamic':
                
                return grad_input

            # compute the dynamic scale factor
            grad_scale_value = compute_block_linear_grad_scale(
                block_linear_layer=block_linear_layer,
                branch_excitation=branch_excitation,
                branch_inhibition=branch_inhibition
            )
            return scale_layer_gradients(module, grad_input, grad_output, grad_scale_value)

        block_linear_layer.register_full_backward_hook(hook_fn)


def compute_block_linear_grad_scale(block_linear_layer, branch_excitation=None, branch_inhibition=None):
    """
    Emulates the old logic from set_branches_to_output_grad_scale:
    total_conductance = block_linear_layer.sum_conductances() + 1
    plus pruned_weight sums from excit/inhib.

    We'll then do an average scale from ratio of total_conductance / block_conductance
    as demonstration. Adjust shape logic if you want a 1D scale per row.

    Returns a single float scale or we can return a per-row scale if needed.
    """
    with torch.no_grad():
        total_conductance = block_linear_layer.sum_conductances() + 1.0

        if branch_excitation is not None:
            total_conductance += branch_excitation.pruned_weight().sum(dim=1)
        if branch_inhibition is not None:
            total_conductance += branch_inhibition.pruned_weight().sum(dim=1)

        # block_linear_layer.log_weight is shape [out_features, block_size]
        branch_conductance = block_linear_layer.log_weight.exp()

        # The old code flattened, here I make it .mean() for a single scalar:
        ratio = (total_conductance[:,None] / branch_conductance).mean().item()

    return ratio