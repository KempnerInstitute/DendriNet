"""
gradient_scaling.py
===================
Provides a GradientScaler class that registers backward hooks to implement 
various gradient-scaling strategies for BlockLinear or other modules.
"""

import torch
import functools


def safe_hook(hook_fn):
    """
    Decorator for backward hooks to ensure they return None when no inputs require gradients.
    This prevents the "Backward hook for Modules where no input requires gradient" error.
    
    Args:
        hook_fn: The original hook function to decorate
        
    Returns:
        A decorated hook function that properly handles the case when no inputs require gradients
    """
    @functools.wraps(hook_fn)
    def safe_hook_fn(module, grad_input, grad_output):
        # Check if any grad_input requires gradient
        requires_grad = False
        if grad_input is not None:
            for g in grad_input:
                if g is not None:
                    requires_grad = True
                    break
                    
        # Check parameters too (some hooks need this)
        if not requires_grad:
            for p in module.parameters(recurse=True):
                if p.requires_grad:
                    requires_grad = True
                    break
        
        # If nothing requires gradients, return None
        if not requires_grad:
            return None
            
        # Otherwise, call the original hook function
        try:
            result = hook_fn(module, grad_input, grad_output)
            return result
        except Exception as e:
            # If anything goes wrong, return the original grad_input as a fallback
            print(f"Warning: Hook error: {e}")
            return grad_input
            
    return safe_hook_fn


def scale_activation_grad_hook(module, grad_input, grad_output):
    """
    A hook for reactivation functions. If inputs don't require gradients,
    return None as required.
    """
    # Check if any input requires gradients
    inputs_require_grad = False
    for p in module.parameters(recurse=True):
        if p.requires_grad:
            inputs_require_grad = True
            break
            
    # If no inputs require gradients, return None
    if not inputs_require_grad:
        return None
        
    return grad_output


def scale_layer_gradients_elementwise(module, grad_input, grad_output, scale_vec):
    """
    Apply element-wise scaling to the gradients.
    """
    return tuple(
        grad * scale_vec if grad is not None else None for grad in grad_input
    )


def scale_layer_gradients_scalar(module, grad_input, grad_output, scale_factor):
    """
    Multiply entire grad_input by a single scalar factor.
    """
    return tuple(
        g * scale_factor if (g is not None) else None
        for g in grad_input
    )


# Apply the safe_hook decorator to create safe versions
safe_scale_layer_gradients_elementwise = lambda module, grad_input, grad_output, scale_vec: \
    safe_hook(lambda m, gi, go: scale_layer_gradients_elementwise(m, gi, go, scale_vec))(module, grad_input, grad_output)

safe_scale_layer_gradients_scalar = lambda module, grad_input, grad_output, scale_factor: \
    safe_hook(lambda m, gi, go: scale_layer_gradients_scalar(m, gi, go, scale_factor))(module, grad_input, grad_output)

class GradientScaler:
    """
    A class for hooking into BlockLinear or other modules to apply
    user-selected gradient scaling strategies.
    """

    def __init__(self,reactivation_strategy="none", blocklinear_strategy="none", scale_factor=1.0, layer_idx=0):
        """
        Parameters
        ----------
        strategy : str
            One of ['none', 'distal_upweight_by_idx', 'block_conductance_dynamic'].
        scale_factor : float
            Base scaling factor for certain strategies.
        layer_idx : int
            For 'distal_upweight_by_idx' if needed.
        """
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.scale_factor = scale_factor
        self.layer_idx = layer_idx
    
    def register_reactivation_inverse(self, reactivation_layer):
        """
        Register a backward hook to invert the reactivation function gradient.
        """
        if self.reactivation_strategy != 'none':
            # We now use the decorator-wrapped version of the hook
            reactivation_layer.register_full_backward_hook(scale_activation_grad_hook)


    def register_block_linear_dynamic(self, block_linear_layer, branch_excitation=None, branch_inhibition=None):
        """
        If strategy='block_conductance_dynamic', attach a backward hook that 
        computes conduction-based scaling from the current block plus excit/inhib.

        If 'distal_upweight_by_idx', we do (layer_idx+1) * scale_factor for entire param.
        If 'none', no special scaling.
        """

        def hook_fn(module, grad_input, grad_output):
            # Apply the appropriate scaling strategy
            if self.blocklinear_strategy == 'block_conductance_dynamic':
                try:
                    scale_vec = compute_block_linear_grad_scale(
                        block_linear_layer=block_linear_layer,
                        branch_excitation=branch_excitation,
                        branch_inhibition=branch_inhibition
                    )
                    return scale_layer_gradients_elementwise(module, grad_input, grad_output, scale_vec)
                except Exception as e:
                    # If there's any error in computing the scale, return the original gradients
                    return grad_input

            elif self.blocklinear_strategy == 'distal_upweight_by_idx':
                try:
                    scale_val = (self.layer_idx + 1) * self.scale_factor
                    return scale_layer_gradients_scalar(module, grad_input, grad_output, scale_val)
                except Exception as e:
                    # If there's any error in applying the scale, return the original gradients
                    return grad_input

            else:
                return grad_input

        # Wrap our hook function with the safe_hook decorator
        safe_hook_fn = safe_hook(hook_fn)

        if self.blocklinear_strategy != 'none':
            block_linear_layer.register_full_backward_hook(safe_hook_fn)


def compute_block_linear_grad_scale(block_linear_layer, branch_excitation=None, branch_inhibition=None):
    """
    A conduction-based dynamic scale, returning a flattened vector so each weight 
    can get a distinct multiplier. shape [out_features * block_size].
    """
    with torch.no_grad():
        total_conductance = block_linear_layer.sum_conductances() + 1.0

        # Safer access to branch_excitation, checking if it exists and has the right attributes
        if branch_excitation is not None and hasattr(branch_excitation, 'pruned_weight'):
            try:
                total_conductance += branch_excitation.pruned_weight().sum(dim=1)
            except Exception as e:
                # If there's any error, we just skip this part
                pass
                
        # Safer access to branch_inhibition, checking if it exists and has the right attributes
        if branch_inhibition is not None and hasattr(branch_inhibition, 'pruned_weight'):
            try:
                total_conductance += branch_inhibition.pruned_weight().sum(dim=1)
            except Exception as e:
                # If there's any error, we just skip this part
                pass

        block_exp = block_linear_layer.log_weight.exp()
        ratio_2d = total_conductance[:, None] / block_exp  # shape [out_features, block_size]
        ratio_1d = ratio_2d.flatten()                      # shape [out_features * block_size]
    return ratio_1d

# Apply the safe_hook decorator to our existing functions
scale_activation_grad_hook = safe_hook(scale_activation_grad_hook)