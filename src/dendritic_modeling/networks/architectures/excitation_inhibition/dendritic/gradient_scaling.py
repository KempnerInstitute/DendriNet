"""
gradient_scaling.py
===================
Provides a GradientScaler class that registers backward hooks to implement
various gradient-scaling strategies for BlockLinear or other modules.
"""

import logging

import torch

from dendritic_modeling.config.choices import normalize_gradient_scaling

logger = logging.getLogger(__name__)


def amp_safe_dtype_cast(grad, target_dtype=torch.float32):
    """
    Safely cast gradient to target dtype for AMP compatibility.

    Args:
        grad: Input gradient tensor
        target_dtype: Target dtype (default: float32)

    Returns:
        Gradient cast to target dtype, or original if None
    """
    if grad is None:
        return None

    # Handle different gradient types and ensure compatibility
    if isinstance(grad, torch.Tensor):
        if grad.dtype != target_dtype:
            return grad.to(target_dtype)

    return grad


def scale_activation_grad_hook(module, grad_input, grad_output):
    """
    A hook for reactivation functions. If inputs don't require gradients,
    return None as required.

    Gradient scaling is a training rule: in eval mode the hook leaves the
    activation gradient unchanged, so analyses see the model's true gradient.
    """
    if not module.training:
        return None

    if getattr(module, "print_hooks", False):
        logger.info("Scaling reactivation gradient")

    # Simply return grad_output to preserve dtypes
    # The actual reactivation gradient scaling should be implemented
    # in the forward pass or through other means that don't change dtypes
    return grad_output


def scale_layer_input_gradients_elementwise(module, grad_input, grad_output):
    """
    Apply element-wise scaling to the input gradients.
    """
    input_scale_vec = 1
    if hasattr(module, "input_scale_vec"):
        if module.input_scale_vec is not None:
            input_scale_vec = module.input_scale_vec

    if getattr(module, "print_hooks", False):
        for i, grad in enumerate(grad_input):
            if grad is not None:
                # Use AMP-safe dtype casting for statistics
                grad_float32 = amp_safe_dtype_cast(grad, torch.float32)
                scaled_grad_float32 = grad_float32 * input_scale_vec

                mean_abs_before = grad_float32.abs().mean().item()
                mean_abs_after = scaled_grad_float32.abs().mean().item()
                logger.info(
                    "Gradient %s - Mean Abs Before: %.6f, Mean Abs After: %.6f",
                    i,
                    mean_abs_before,
                    mean_abs_after,
                )

    # Apply scaling while preserving original dtypes
    scaled_gradients = []
    for grad in grad_input:
        if grad is not None:
            # Direct scaling to preserve dtype
            scaled_grad = grad * input_scale_vec
            scaled_gradients.append(scaled_grad)
        else:
            scaled_gradients.append(None)

    return tuple(scaled_gradients)


def scale_layer_param_gradients_elementwise(grad, param_scale_vec, print_hooks=False):
    """
    Apply element-wise scaling to the parameter gradients.
    """
    param_scale_vec = 1 if param_scale_vec is None else param_scale_vec

    if grad is None:
        return None

    if print_hooks:
        # Use AMP-safe dtype casting for statistics only
        grad_float32 = amp_safe_dtype_cast(grad, torch.float32)
        scaled_grad_float32 = grad_float32 * param_scale_vec

        mean_abs_before = grad_float32.abs().mean().item()
        mean_abs_after = scaled_grad_float32.abs().mean().item()
        logger.info(
            "Parameter Gradient - Mean Abs Before: %.6f, Mean Abs After: %.6f",
            mean_abs_before,
            mean_abs_after,
        )

    # Direct scaling to preserve dtype
    scaled_grad = grad * param_scale_vec
    return scaled_grad


def scale_layer_gradients_scalar(module, grad_input, grad_output, scale_factor):
    """
    Multiply entire grad_input by a single scalar factor.
    """
    # Check if prints are enabled
    print_hooks = getattr(module, "print_hooks", False)

    # Calculate and print mean absolute gradient values before and after scaling
    if print_hooks:
        for i, g in enumerate(grad_input):
            if g is not None:
                mean_abs_before = g.abs().mean().item()
                mean_abs_after = (g * scale_factor).abs().mean().item()
                logger.info(
                    "Scalar Scaling Gradient %s - Mean Abs Before: %.6f, "
                    "Mean Abs After: %.6f (scale=%.4f)",
                    i,
                    mean_abs_before,
                    mean_abs_after,
                    scale_factor,
                )

    return tuple(g * scale_factor if (g is not None) else None for g in grad_input)


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
        reactivation_strategy : str
            ``"none"`` or ``"straight_through"`` (the gate passes its output
            gradient to its input in training mode).
        blocklinear_strategy : str
            ``"none"``, ``"conductance_dynamic"`` or ``"distal_upweight_by_idx"``.
        topk_strategy : str
            ``"none"`` or ``"conductance_dynamic"``.
        scale_factor : float
            Base factor of ``distal_upweight_by_idx``.
        layer_idx : int
            Layer index used by ``distal_upweight_by_idx``.
        print_hooks : bool
            Whether to print gradient statistics in hooks. Default is False.
        """
        self.reactivation_strategy = normalize_gradient_scaling(
            reactivation_strategy, "reactivation"
        )
        self.blocklinear_strategy = normalize_gradient_scaling(
            blocklinear_strategy, "blocklinear"
        )
        self.topk_strategy = normalize_gradient_scaling(topk_strategy, "topk")
        self.scale_factor = scale_factor
        self.layer_idx = layer_idx
        self.print_hooks = print_hooks

    def register_reactivation_inverse(self, reactivation_layer):
        """
        Register a backward hook to invert the reactivation function gradient.
        """
        if self.reactivation_strategy != "none":

            reactivation_layer.print_hooks = self.print_hooks
            reactivation_layer.register_full_backward_hook(scale_activation_grad_hook)

    def register_block_linear_dynamic(self, block_linear_layer):
        """
        If strategy='conductance_dynamic', attach a backward hook that
        computes conduction-based scaling from the current block plus excit/inhib.

        If 'distal_upweight_by_idx', we do (layer_idx+1) * scale_factor for entire param.
        If 'none', no special scaling.

        The hooks scale gradients only while the layer is in training mode; in
        eval mode gradients are left unscaled (stale training scales are not
        applied to analysis gradients).
        """
        if self.print_hooks:
            logger.info(
                "Registering block linear dynamic, strategy: %s",
                self.blocklinear_strategy,
            )

        def input_hook_fn(module, grad_input, grad_output):
            if not module.training:
                return None
            # Apply the appropriate scaling strategy
            if self.print_hooks:
                logger.info("BlockLinear input hook call")
            if self.blocklinear_strategy == "conductance_dynamic":
                if getattr(module, "_use_forward_dynamic_grad_scaling", False):
                    return grad_input
                return scale_layer_input_gradients_elementwise(
                    module, grad_input, grad_output
                )

            elif self.blocklinear_strategy == "distal_upweight_by_idx":
                scale_val = (self.layer_idx + 1) * self.scale_factor
                return scale_layer_gradients_scalar(
                    module, grad_input, grad_output, scale_val
                )

            else:
                return grad_input

        if self.blocklinear_strategy != "none":

            block_linear_layer.print_hooks = self.print_hooks
            block_linear_layer.register_full_backward_hook(input_hook_fn)
            self.register_block_linear_param_hook(block_linear_layer)
            if self.print_hooks:
                logger.info("Registered block linear dynamic")

    def register_block_linear_param_hook(self, block_linear_layer):
        """Register the ``log_weight`` gradient hook of the BlockLinear strategy.

        Part of :meth:`register_block_linear_dynamic`. Tensor hooks are not
        carried over by ``copy.deepcopy`` or pickling, so owners call this to
        restore them on a copied layer (module hooks are copied).
        """
        if self.blocklinear_strategy == "none":
            return

        def param_hook_fn(grad):
            if not block_linear_layer.training:
                return grad
            if self.print_hooks:
                logger.info("BlockLinear parameter hook call")
            if self.blocklinear_strategy == "conductance_dynamic":
                if getattr(
                    block_linear_layer, "_use_forward_dynamic_grad_scaling", False
                ):
                    return grad
                return scale_layer_param_gradients_elementwise(
                    grad, block_linear_layer.param_scale_vec, self.print_hooks
                )

            else:
                return grad

        block_linear_layer.log_weight.register_hook(param_hook_fn)

    def register_topk_dynamic(self, topk_layer):
        """Attach the ``conductance_dynamic`` TopK gradient scaling.

        The scale is applied to the ``pre_w`` gradient in training mode (or by
        the forward hook path of stateful layers). ``"none"`` attaches nothing.
        """
        if self.print_hooks:
            logger.info("Registering TopK dynamic, strategy: %s", self.topk_strategy)

        if self.topk_strategy != "none":
            if self.topk_strategy == "conductance_dynamic":
                topk_layer._use_forward_dynamic_grad_scaling = True

            topk_layer.print_hooks = self.print_hooks
            self.register_topk_param_hook(topk_layer)

    def register_topk_param_hook(self, topk_layer):
        """Register the ``pre_w`` gradient hook of the TopK strategy.

        Part of :meth:`register_topk_dynamic`. Tensor hooks are not carried
        over by ``copy.deepcopy`` or pickling, so owners call this to restore
        them on a copied layer.
        """
        if self.topk_strategy == "none":
            return

        def param_hook_fn(grad):
            if not topk_layer.training:
                return grad
            if self.print_hooks:
                logger.info("TopK parameter hook call")
            if self.topk_strategy == "conductance_dynamic":
                if getattr(topk_layer, "_use_forward_dynamic_grad_scaling", False):
                    return grad
                return scale_layer_param_gradients_elementwise(
                    grad, topk_layer.param_scale_vec, self.print_hooks
                )

            else:
                return grad

        topk_layer.pre_w.register_hook(param_hook_fn)
