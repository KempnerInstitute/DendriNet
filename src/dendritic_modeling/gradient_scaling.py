"""
gradient_scaling.py
===================
Provides a GradientScaler class that registers backward hooks to implement
various gradient-scaling strategies for BlockLinear or other modules.
"""

import functools

import torch


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
            print(f"Warning: Hook error: {e}", flush=True)
            return grad_input

    return safe_hook_fn


def scale_activation_grad_hook(module, grad_input, grad_output):
    """
    A hook for reactivation functions. If inputs don't require gradients,
    return None as required.
    """
    # Check if prints are enabled
    print_hooks = getattr(module, "print_hooks", False)

    if print_hooks:
        print("\nscaling reactivation grad\n", flush=True)

    return grad_output


def scale_layer_input_gradients_elementwise(module, grad_input, grad_output):
    """
    Apply element-wise scaling to the input gradients.
    """
    if hasattr(module, "input_scale_vec"):
        if module.input_scale_vec is not None:
            input_scale_vec = module.input_scale_vec
    else:
        input_scale_vec = 1

    # Check if prints are enabled
    print_hooks = getattr(module, "print_hooks", False)

    for i, grad in enumerate(grad_input):
        if grad is not None:
            mean_abs_before = grad.abs().mean().item()
            mean_abs_after = (grad * input_scale_vec).abs().mean().item()
            if print_hooks:
                print(
                    f"Gradient {i} - Mean Abs Before: {mean_abs_before:.6f}, Mean Abs After: {mean_abs_after:.6f}",
                    flush=True,
                )
    return tuple(
        grad * input_scale_vec if grad is not None else None for grad in grad_input
    )


def scale_layer_param_gradients_elementwise(grad, param_scale_vec, print_hooks=False):
    """
    Apply element-wise scaling to the parameter gradients.
    """
    if grad is not None:
        mean_abs_before = grad.abs().mean().item()
        param_scale_vec = 1 if param_scale_vec is None else param_scale_vec
        mean_abs_after = (grad * param_scale_vec).abs().mean().item()
        if print_hooks:
            print(
                f"Parameter Gradient - Mean Abs Before: {mean_abs_before:.6f}, Mean Abs After: {mean_abs_after:.6f}",
                flush=True,
            )
    else:
        param_scale_vec = 1 if param_scale_vec is None else param_scale_vec
    return grad * param_scale_vec if grad is not None else None


# def scale_activation_grad_hook(module, grad_input, grad_output):
#     return grad_output


# def scale_layer_gradients_elementwise(module, grad_input, grad_output, scale_vec):
#     print('\nbackward hook call', flush=True)
#     print(f'grad_input:\n{grad_input[0].mean(dim = 0)}\n{grad_input[0].std(dim = 0)}')
#     print(f'grad_output:\n{grad_output[0].mean(dim = 0)}\n{grad_output[0].std(dim = 0)}')
#     print(f'scale_vec: {scale_vec.shape}\n{scale_vec}')
#     return tuple(
#         grad * scale_vec if grad is not None else None for grad in grad_input
#     )


def scale_layer_gradients_scalar(module, grad_input, grad_output, scale_factor):
    """
    Multiply entire grad_input by a single scalar factor.
    """
    # Check if prints are enabled
    print_hooks = getattr(module, "print_hooks", False)

    # Calculate and print mean absolute gradient values before and after scaling
    for i, g in enumerate(grad_input):
        if g is not None:
            mean_abs_before = g.abs().mean().item()
            mean_abs_after = (g * scale_factor).abs().mean().item()
            if print_hooks:
                print(
                    f"Scalar Scaling Gradient {i} - Mean Abs Before: {mean_abs_before:.6f}, Mean Abs After: {mean_abs_after:.6f} (scale={scale_factor:.4f})",
                    flush=True,
                )

    return tuple(g * scale_factor if (g is not None) else None for g in grad_input)


# Apply the safe_hook decorator to create safe versions
# safe_scale_layer_input_gradients_elementwise = (
#     lambda module, grad_input, grad_output: safe_hook(
#         lambda m, gi, go: scale_layer_input_gradients_elementwise(m, gi, go)
#     )(module, grad_input, grad_output)
# )

# safe_scale_layer_gradients_scalar = (
#     lambda module, grad_input, grad_output, scale_factor: safe_hook(
#         lambda m, gi, go: scale_layer_gradients_scalar(m, gi, go, scale_factor)
#     )(module, grad_input, grad_output)
# )


class GradientScaler:
    """
    A class for hooking into BlockLinear or other modules to apply
    user-selected gradient scaling strategies.
    """

    def __init__(
        self,
        reactivation_strategy="none",
        blocklinear_strategy="none",
        topk_strategy="none",
        scale_factor=1.0,
        layer_idx=0,
        print_hooks=False,
    ):
        """
        Parameters
        ----------
        strategy : str
            One of ['none', 'distal_upweight_by_idx', 'block_conductance_dynamic'].
        scale_factor : float
            Base scaling factor for certain strategies.
        layer_idx : int
            For 'distal_upweight_by_idx' if needed.
        print_hooks : bool
            Whether to print gradient statistics in hooks. Default is False.
        """
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.topk_strategy = topk_strategy
        self.scale_factor = scale_factor
        self.layer_idx = layer_idx
        self.print_hooks = print_hooks

    def register_reactivation_inverse(self, reactivation_layer):
        """
        Register a backward hook to invert the reactivation function gradient.
        """
        if self.reactivation_strategy != "none":
            # First remove any existing hooks to prevent conflicts
            if hasattr(reactivation_layer, "_backward_hooks"):
                reactivation_layer._backward_hooks.clear()
            if hasattr(reactivation_layer, "_backward_pre_hooks"):
                reactivation_layer._backward_pre_hooks.clear()

            # Set the print_hooks attribute
            reactivation_layer.print_hooks = self.print_hooks

            # Register with full backward hook consistently
            reactivation_layer.register_full_backward_hook(
                # safe_hook(scale_activation_grad_hook)
                scale_activation_grad_hook
            )

    def register_block_linear_dynamic(self, block_linear_layer):
        """
        If strategy='conductance_dynamic', attach a backward hook that
        computes conduction-based scaling from the current block plus excit/inhib.

        If 'distal_upweight_by_idx', we do (layer_idx+1) * scale_factor for entire param.
        If 'none', no special scaling.
        """
        if self.print_hooks:
            print(
                "registering block linear dynamic, strategy:",
                self.blocklinear_strategy,
                flush=True,
            )

        def input_hook_fn(module, grad_input, grad_output):
            # Apply the appropriate scaling strategy
            if self.print_hooks:
                print("\n\nblocklinear input hook call", flush=True)
            if self.blocklinear_strategy == "conductance_dynamic":
                try:
                    return scale_layer_input_gradients_elementwise(
                        module, grad_input, grad_output
                    )
                except Exception as e:
                    # If there's any error in computing the scale, return the original gradients
                    return grad_input

            elif self.blocklinear_strategy == "distal_upweight_by_idx":
                try:
                    scale_val = (self.layer_idx + 1) * self.scale_factor
                    return scale_layer_gradients_scalar(
                        module, grad_input, grad_output, scale_val
                    )
                except Exception as e:
                    # If there's any error in applying the scale, return the original gradients
                    return grad_input

            else:
                return grad_input

        def param_hook_fn(grad):
            if self.print_hooks:
                print("\n\nblocklinear param hook call", flush=True)
            if self.blocklinear_strategy == "conductance_dynamic":
                try:
                    return scale_layer_param_gradients_elementwise(
                        grad, block_linear_layer.param_scale_vec, self.print_hooks
                    )
                except Exception as e:
                    # If there's any error in applying the scale, return the original gradients
                    return grad

            else:
                return grad

        if self.blocklinear_strategy != "none":
            # First remove any existing hooks to prevent conflicts
            if hasattr(block_linear_layer, "_backward_hooks"):
                block_linear_layer._backward_hooks.clear()
            if hasattr(block_linear_layer, "_backward_pre_hooks"):
                block_linear_layer._backward_pre_hooks.clear()

            # Set the print_hooks attribute
            block_linear_layer.print_hooks = self.print_hooks

            # Register consistently with full backward hook
            block_linear_layer.register_full_backward_hook(input_hook_fn)
            block_linear_layer.log_weight.register_hook(param_hook_fn)
            if self.print_hooks:
                print("registered block linear dynamic", flush=True)

    def register_topk_dynamic(self, topk_layer):
        """
        If strategy='conductance_dynamic', attach a backward hook that
        computes conduction-based scaling from the current block plus excit/inhib.

        If 'distal_upweight_by_idx', we do (layer_idx+1) * scale_factor for entire param.
        If 'none', no special scaling.
        """
        if self.print_hooks:
            print("registering topk dynamic, strategy:", self.topk_strategy, flush=True)

        def input_hook_fn(module, grad_input, grad_output):
            # Apply the appropriate scaling strategy
            if self.print_hooks:
                print("\n\ntopk input hook call", flush=True)
            if self.topk_strategy == "conductance_dynamic":
                try:
                    return scale_layer_input_gradients_elementwise(
                        module, grad_input, grad_output
                    )
                except Exception as e:
                    # If there's any error in computing the scale, return the original gradients
                    return grad_input

            elif self.topk_strategy == "distal_upweight_by_idx":
                try:
                    scale_val = (self.layer_idx + 1) * self.scale_factor
                    return scale_layer_gradients_scalar(
                        module, grad_input, grad_output, scale_val
                    )
                except Exception as e:
                    # If there's any error in applying the scale, return the original gradients
                    return grad_input

            else:
                return grad_input

        def param_hook_fn(grad):
            if self.print_hooks:
                print("\n\ntopk param hook call", flush=True)
            if self.topk_strategy == "conductance_dynamic":
                try:
                    return scale_layer_param_gradients_elementwise(
                        grad, topk_layer.param_scale_vec, self.print_hooks
                    )
                except Exception as e:
                    # If there's any error in applying the scale, return the original gradients
                    return grad

            else:
                return grad

        if self.topk_strategy != "none":
            # First remove any existing hooks to prevent conflicts
            if hasattr(topk_layer, "_backward_hooks"):
                topk_layer._backward_hooks.clear()
            if hasattr(topk_layer, "_backward_pre_hooks"):
                topk_layer._backward_pre_hooks.clear()

            # Set the print_hooks attribute
            topk_layer.print_hooks = self.print_hooks

            # Register consistently with full backward hook
            topk_layer.register_full_backward_hook(input_hook_fn)
            topk_layer.pre_w.register_hook(param_hook_fn)


# Apply the safe_hook decorator to our existing functions
# scale_activation_grad_hook = safe_hook(scale_activation_grad_hook)
