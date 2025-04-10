"""
gradient_scaling.py
===================
Provides a GradientScaler class that registers backward hooks to implement
various gradient-scaling strategies for BlockLinear or other modules.
"""

import functools
import traceback

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
        requires_grad = True
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
    # Check if any input requires gradients
    inputs_require_grad = False
    for p in module.parameters(recurse=True):
        if p.requires_grad:
            inputs_require_grad = True
            break

    # If no inputs require gradients, return None
    if not inputs_require_grad:
        return None

    print("\nscaling reactivation grad\n", flush=True)
    return grad_output


def scale_layer_input_gradients_elementwise(module, grad_input, grad_output):
    """
    Apply element-wise scaling to the input gradients.
    """
    print("\n\n***INPUT HOOK CALLED***", flush=True)
    # print(f"Module: {module.__class__.__name__}", flush=True)
    # print(f"grad_input types: {[type(g) for g in grad_input]}", flush=True)
    # print(f"grad_input shapes: {[g.shape if g is not None else None for g in grad_input]}", flush=True)

    if hasattr(module, "input_scale_vec"):
        if module.input_scale_vec is not None:
            input_scale_vec = module.input_scale_vec
    else:
        input_scale_vec = 1

    # Print gradient norms before scaling
    # for i, grad in enumerate(grad_input):
    #     if grad is not None:
    #         print(f"Before scaling - Module: {module.__class__.__name__}, Input grad {i} norm: {grad.norm()}", flush=True)

    # print("scale_layer_input_gradients_elementwise CALLED", flush=True)

    # Scale gradients
    scaled_grad_input = tuple(
        grad * input_scale_vec if grad is not None else None for grad in grad_input
    )

    # Print gradient norms after scaling
    # for i, grad in enumerate(scaled_grad_input):
    #     if grad is not None:
    #         print(f"After scaling - Module: {module.__class__.__name__}, Input grad {i} norm: {grad.norm()}", flush=True)

    # for grad in grad_input:
    # if grad is not None:
    # print(f'grad std:\n{grad.std(dim = 0)}', flush=True)
    # print('scaled grad std:\n', (grad * input_scale_vec).std(dim = 0), flush=True)
    return scaled_grad_input


def scale_layer_param_gradients_elementwise(grad, param_scale_vec):
    """
    Apply element-wise scaling to the parameter gradients.
    """
    # print(f"Param grad shape: {grad.shape if grad is not None else None}", flush=True)
    # print(f"Scale vec shape: {param_scale_vec.shape if hasattr(param_scale_vec, 'shape') else 'scalar'}", flush=True)

    param_scale_vec = 1 if param_scale_vec is None else param_scale_vec

    # Print gradient norm before scaling
    # if grad is not None:
    #     print(f"Before scaling - Parameter grad norm: {grad.norm()}", flush=True)

    # Scale gradient
    scaled_grad = grad * param_scale_vec if grad is not None else None

    # Print gradient norm after scaling
    # if scaled_grad is not None:
    #     print(f"After scaling - Parameter grad norm: {scaled_grad.norm()}", flush=True)

    return scaled_grad


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
    # Print gradient norms before scaling
    # for i, grad in enumerate(grad_input):
    #     if grad is not None:
    #         print(f"Before scaling - Module: {module.__class__.__name__}, Scalar scaling with factor {scale_factor}, Input grad {i} norm: {grad.norm()}", flush=True)

    # print(f"Scaling gradients by scalar factor: {scale_factor}", flush=True)

    # Scale gradients
    scaled_grad_input = tuple(
        g * scale_factor if (g is not None) else None for g in grad_input
    )

    # Print gradient norms after scaling
    # for i, grad in enumerate(scaled_grad_input):
    #     if grad is not None:
    #         print(f"After scaling - Module: {module.__class__.__name__}, Scalar scaling with factor {scale_factor}, Input grad {i} norm: {grad.norm()}", flush=True)

    return scaled_grad_input


# Apply the safe_hook decorator to create safe versions
safe_scale_layer_input_gradients_elementwise = (
    lambda module, grad_input, grad_output: safe_hook(
        lambda m, gi, go: scale_layer_input_gradients_elementwise(m, gi, go)
    )(module, grad_input, grad_output)
)

safe_scale_layer_gradients_scalar = (
    lambda module, grad_input, grad_output, scale_factor: safe_hook(
        lambda m, gi, go: scale_layer_gradients_scalar(m, gi, go, scale_factor)
    )(module, grad_input, grad_output)
)


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
        """
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.topk_strategy = topk_strategy
        self.scale_factor = scale_factor
        self.layer_idx = layer_idx

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

            # Register with full backward hook consistently
            # print(f"Registering reactivation hook for {reactivation_layer}", flush=True)
            reactivation_layer.register_full_backward_hook(
                safe_hook(scale_activation_grad_hook)
            )

    def register_block_linear_dynamic(self, block_linear_layer):
        """
        If strategy='conductance_dynamic', attach a backward hook that
        computes conduction-based scaling from the current block plus excit/inhib.

        If 'distal_upweight_by_idx', we do (layer_idx+1) * scale_factor for entire param.
        If 'none', no special scaling.
        """
        print(
            "registering block linear dynamic, strategy:",
            self.blocklinear_strategy,
            flush=True,
        )

        # For conductance_dynamic, make sure the layer has an input_scale_vec attribute
        if self.blocklinear_strategy == "conductance_dynamic":
            if not hasattr(block_linear_layer, "input_scale_vec"):
                # Create a default scale vector of 1.0 to make hooks work
                block_linear_layer.register_buffer(
                    "input_scale_vec",
                    torch.ones(
                        block_linear_layer.in_features,
                        device=block_linear_layer.log_weight.device,
                    ),
                )
                print(
                    f"Created input_scale_vec for {block_linear_layer.__class__.__name__}",
                    flush=True,
                )

        # Keep standard backward hook for debugging
        def input_hook_fn(module, grad_input, grad_output):
            # Apply the appropriate scaling strategy
            print("\n\nBLOCKLINEAR INPUT HOOK CALLED", flush=True)
            print(f"Module: {module.__class__.__name__}", flush=True)
            print(f"grad_input types: {[type(g) for g in grad_input]}", flush=True)
            print(
                f"grad_input shapes: {[g.shape if g is not None else None for g in grad_input]}",
                flush=True,
            )
            print(
                f"grad_output shapes: {[g.shape if g is not None else None for g in grad_output]}",
                flush=True,
            )

            if self.blocklinear_strategy == "conductance_dynamic":
                try:
                    if (
                        hasattr(module, "input_scale_vec")
                        and module.input_scale_vec is not None
                    ):
                        # print(f"Using conductance_dynamic with scale shape: {module.input_scale_vec.shape}", flush=True)
                        return scale_layer_input_gradients_elementwise(
                            module, grad_input, grad_output
                        )
                    else:
                        print(
                            "No input_scale_vec found in module hook, creating default",
                            flush=True,
                        )
                        # Create a fallback scale vector
                        if grad_input[0] is not None:
                            scale_vec = torch.ones_like(grad_input[0][0])
                            print(
                                f"Created default scale vector with shape {scale_vec.shape}",
                                flush=True,
                            )
                            return tuple(
                                grad * scale_vec.view(1, -1)
                                if grad is not None
                                else None
                                for grad in grad_input
                            )
                        else:
                            return grad_input
                except Exception as e:
                    # If there's any error in computing the scale, return the original gradients
                    print(f"Error in conductance_dynamic: {e}", flush=True)
                    print(f"Traceback: {traceback.format_exc()}", flush=True)
                    return grad_input

            elif self.blocklinear_strategy == "distal_upweight_by_idx":
                try:
                    scale_val = (self.layer_idx + 1) * self.scale_factor
                    print(f"Using distal_upweight with scale: {scale_val}", flush=True)
                    return scale_layer_gradients_scalar(
                        module, grad_input, grad_output, scale_val
                    )
                except Exception as e:
                    # If there's any error in applying the scale, return the original gradients
                    print(f"Error in distal_upweight_by_idx: {e}", flush=True)
                    print(f"Traceback: {traceback.format_exc()}", flush=True)
                    return grad_input

            else:
                return grad_input

        original_forward = block_linear_layer.forward

        def forward_with_hooks(x):
            # First call the original forward
            result = original_forward(x)

            # Check if we should apply gradient scaling
            if self.blocklinear_strategy != "none" and x.requires_grad:

                def tensor_hook_fn(grad):
                    print("\n\nBLOCKLINEAR TENSOR HOOK CALLED", flush=True)
                    print(
                        f"Tensor hook for layer: {block_linear_layer.__class__.__name__}",
                        flush=True,
                    )
                    print(f"grad tensor shape: {grad.shape}", flush=True)
                    print(
                        f"grad tensor requires_grad: {grad.requires_grad}", flush=True
                    )
                    print(
                        f"BEFORE SCALING - grad tensor norm: {grad.norm()}", flush=True
                    )

                    if self.blocklinear_strategy == "conductance_dynamic":
                        try:
                            if (
                                hasattr(block_linear_layer, "input_scale_vec")
                                and block_linear_layer.input_scale_vec is not None
                            ):
                                input_scale_vec = block_linear_layer.input_scale_vec
                                # print(f"Scaling tensor grad with shape {grad.shape}", flush=True)
                                # Make sure dimensions match
                                if len(grad.shape) == 2 and input_scale_vec.dim() == 1:
                                    # We need to broadcast properly
                                    scaled_grad = grad * input_scale_vec.view(1, -1)
                                else:
                                    scaled_grad = grad * input_scale_vec
                                print(
                                    f"AFTER SCALING - grad tensor norm: {scaled_grad.norm()}",
                                    flush=True,
                                )
                                return scaled_grad
                            else:
                                print(
                                    "No input_scale_vec found, creating default ones",
                                    flush=True,
                                )
                                # Create a default scale vector
                                scale_vec = torch.ones_like(grad[0])
                                print(
                                    f"Created temporary scale vector with shape {scale_vec.shape}",
                                    flush=True,
                                )
                                scaled_grad = grad * scale_vec.view(1, -1)
                                print(
                                    f"AFTER SCALING - grad tensor norm: {scaled_grad.norm()}",
                                    flush=True,
                                )
                                return scaled_grad
                        except Exception as e:
                            print(f"Error in tensor hook: {e}", flush=True)
                            print(f"Traceback: {traceback.format_exc()}", flush=True)
                            print(
                                f"AFTER ERROR - returning original grad norm: {grad.norm()}",
                                flush=True,
                            )
                            return grad
                    elif self.blocklinear_strategy == "distal_upweight_by_idx":
                        try:
                            scale_val = (self.layer_idx + 1) * self.scale_factor
                            print(
                                f"Scaling tensor grad with factor {scale_val}",
                                flush=True,
                            )
                            scaled_grad = grad * scale_val
                            print(
                                f"AFTER SCALING - grad tensor norm: {scaled_grad.norm()}",
                                flush=True,
                            )
                            return scaled_grad
                        except Exception as e:
                            print(f"Error in tensor hook: {e}", flush=True)
                            print(f"Traceback: {traceback.format_exc()}", flush=True)
                            print(
                                f"AFTER ERROR - returning original grad norm: {grad.norm()}",
                                flush=True,
                            )
                            return grad
                    else:
                        print(
                            f"NO SCALING APPLIED - grad tensor norm: {grad.norm()}",
                            flush=True,
                        )
                        return grad

                # Register the hook directly on the input tensor
                if hasattr(x, "register_hook") and x.requires_grad:
                    x.register_hook(tensor_hook_fn)

            return result

        # Replace the forward method with our hooked version
        block_linear_layer.forward = forward_with_hooks

        def param_hook_fn(grad):
            print("\n\nBLOCKLINEAR PARAM HOOK CALLED", flush=True)
            print(
                f"Parameter hook for: {block_linear_layer.__class__.__name__}",
                flush=True,
            )
            print(f"param grad shape: {grad.shape}", flush=True)
            print(f"BEFORE SCALING - param grad norm: {grad.norm()}", flush=True)

            if self.blocklinear_strategy == "conductance_dynamic":
                try:
                    scaled_grad = scale_layer_param_gradients_elementwise(
                        grad, block_linear_layer.param_scale_vec
                    )
                    print(
                        f"AFTER SCALING - param grad norm: {scaled_grad.norm()}",
                        flush=True,
                    )
                    return scaled_grad
                except Exception as e:
                    # If there's any error in applying the scale, return the original gradients
                    print(f"Error in param conductance_dynamic: {e}", flush=True)
                    print(f"Traceback: {traceback.format_exc()}", flush=True)
                    print(
                        f"AFTER ERROR - returning original param grad norm: {grad.norm()}",
                        flush=True,
                    )
                    return grad

            else:
                print(
                    f"NO SCALING APPLIED - param grad norm: {grad.norm()}", flush=True
                )
                return grad

        # Wrap our hook function with the safe_hook decorator
        safe_input_hook_fn = safe_hook(input_hook_fn)
        safe_param_hook_fn = (
            param_hook_fn  # Parameter hooks don't need the safe wrapper
        )

        if self.blocklinear_strategy != "none":
            # First remove any existing hooks to prevent conflicts
            if hasattr(block_linear_layer, "_backward_hooks"):
                block_linear_layer._backward_hooks.clear()
            if hasattr(block_linear_layer, "_backward_pre_hooks"):
                block_linear_layer._backward_pre_hooks.clear()

            # Register consistently with full backward hook
            # print(f"Registering block linear hook for {block_linear_layer}", flush=True)
            # Use the direct hook function instead of the safe_hook wrapped version
            print(
                f"REGISTERING INPUT HOOK DIRECTLY for {block_linear_layer.__class__.__name__}",
                flush=True,
            )
            block_linear_layer.register_full_backward_hook(
                input_hook_fn
            )  # Use direct hook instead of safe_hook
            block_linear_layer.log_weight.register_hook(safe_param_hook_fn)
            # print("registered block linear dynamic", flush=True)

    def register_topk_dynamic(self, topk_layer):
        """
        If strategy='conductance_dynamic', attach a backward hook that
        computes conduction-based scaling from the current block plus excit/inhib.

        If 'distal_upweight_by_idx', we do (layer_idx+1) * scale_factor for entire param.
        If 'none', no special scaling.
        """
        # print("registering topk dynamic, strategy:", self.topk_strategy, flush=True)

        # For conductance_dynamic, make sure the layer has an input_scale_vec attribute
        if self.topk_strategy == "conductance_dynamic":
            if not hasattr(topk_layer, "input_scale_vec"):
                # Create a default scale vector of 1.0 to make hooks work
                topk_layer.register_buffer(
                    "input_scale_vec",
                    torch.ones(topk_layer.in_features, device=topk_layer.pre_w.device),
                )
                print(
                    f"Created input_scale_vec for {topk_layer.__class__.__name__}",
                    flush=True,
                )

        def input_hook_fn(module, grad_input, grad_output):
            # Apply the appropriate scaling strategy
            # print("\n\ntopk input hook call", flush=True)
            if self.topk_strategy == "conductance_dynamic":
                try:
                    if (
                        hasattr(module, "input_scale_vec")
                        and module.input_scale_vec is not None
                    ):
                        # print(f"Using topk conductance_dynamic with scale shape: {module.input_scale_vec.shape}", flush=True)
                        return scale_layer_input_gradients_elementwise(
                            module, grad_input, grad_output
                        )
                    else:
                        print(
                            "No input_scale_vec found in topk module hook, creating default",
                            flush=True,
                        )
                        # Create a fallback scale vector
                        if grad_input[0] is not None:
                            scale_vec = torch.ones_like(grad_input[0][0])
                            print(
                                f"Created default scale vector with shape {scale_vec.shape}",
                                flush=True,
                            )
                            return tuple(
                                grad * scale_vec.view(1, -1)
                                if grad is not None
                                else None
                                for grad in grad_input
                            )
                        else:
                            return grad_input
                except Exception as e:
                    # If there's any error in computing the scale, return the original gradients
                    print(f"Error in topk conductance_dynamic: {e}", flush=True)
                    print(f"Traceback: {traceback.format_exc()}", flush=True)
                    return grad_input

            elif self.topk_strategy == "distal_upweight_by_idx":
                try:
                    scale_val = (self.layer_idx + 1) * self.scale_factor
                    print(
                        f"Using topk distal_upweight with scale: {scale_val}",
                        flush=True,
                    )
                    return scale_layer_gradients_scalar(
                        module, grad_input, grad_output, scale_val
                    )
                except Exception as e:
                    # If there's any error in applying the scale, return the original gradients
                    print(f"Error in topk distal_upweight_by_idx: {e}", flush=True)
                    print(f"Traceback: {traceback.format_exc()}", flush=True)
                    return grad_input

            else:
                return grad_input

        def param_hook_fn(grad):
            # print("\n\ntopk param hook call", flush=True)
            if self.topk_strategy == "conductance_dynamic":
                try:
                    return scale_layer_param_gradients_elementwise(
                        grad, topk_layer.param_scale_vec
                    )
                except Exception as e:
                    # If there's any error in applying the scale, return the original gradients
                    print(f"Error in topk param conductance_dynamic: {e}", flush=True)
                    print(f"Traceback: {traceback.format_exc()}", flush=True)
                    return grad

            else:
                return grad

        # Wrap our hook function with the safe_hook decorator
        safe_input_hook_fn = safe_hook(input_hook_fn)
        safe_param_hook_fn = (
            param_hook_fn  # Parameter hooks don't need the safe wrapper
        )

        if self.topk_strategy != "none":
            # First remove any existing hooks to prevent conflicts
            if hasattr(topk_layer, "_backward_hooks"):
                topk_layer._backward_hooks.clear()
            if hasattr(topk_layer, "_backward_pre_hooks"):
                topk_layer._backward_pre_hooks.clear()

            # Register consistently with full backward hook
            # print(f"Registering topk hook for {topk_layer}", flush=True)
            topk_layer.register_full_backward_hook(safe_input_hook_fn)
            topk_layer.pre_w.register_hook(safe_param_hook_fn)
            # print(f"Successfully registered hooks for {topk_layer.__class__.__name__}", flush=True)


# Apply the safe_hook decorator to our existing functions
scale_activation_grad_hook = safe_hook(scale_activation_grad_hook)
