"""
Weight transformation utilities for positive weight constraints.

This module provides different transformation functions to ensure synaptic weights
remain positive during training. Each transformation has different characteristics:

"""

import math
from typing import Literal

import torch
import torch.nn.functional as functional

WeightTransformType = Literal["identity", "exp", "relu", "softplus"]


def safe_exp(
    weights: torch.Tensor,
    margin: float = 1.0,
    max_exponent: float | None = None,
) -> torch.Tensor:
    """Exponentiate without overflowing finite floating-point tensors."""
    if not torch.is_floating_point(weights):
        return torch.exp(weights)
    max_log = math.log(torch.finfo(weights.dtype).max) - float(margin)
    if max_exponent is not None:
        max_log = min(max_log, float(max_exponent))
    return torch.exp(weights.clamp(max=max_log))


def inverse_softplus(
    positive_weights: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Numerically stable inverse of ``softplus`` for positive values.

    Uses ``x + log(-expm1(-x))`` instead of ``log(exp(x) - 1)`` so large
    weights map back to raw parameters without overflowing.
    """
    x = positive_weights.clamp_min(eps)
    return x + torch.log(-torch.expm1(-x))


def apply_weight_transform(
    weights: torch.Tensor, transform_type: WeightTransformType = "exp"
) -> torch.Tensor:
    """
    Apply the specified transformation to ensure positive weights.

    Args:
        weights: Raw weight parameters (can be negative)
        transform_type: Type of transformation to apply

    Returns:
        Transformed weights (guaranteed to be positive)

    Raises:
        ValueError: If transform_type is not supported
    """
    if transform_type == "exp":
        return safe_exp(weights)
    elif transform_type == "relu":
        return functional.relu(weights)
    elif transform_type == "softplus":
        return functional.softplus(weights)
    elif transform_type == "identity":
        # No constraint: signed weights allowed
        return weights
    else:
        raise ValueError(
            f"Unsupported weight transform type: {transform_type}. "
            f"Supported types: 'identity', 'exp', 'relu', 'softplus'"
        )


def get_weight_transform_fn(transform_type: WeightTransformType):
    """
    Get a transformation function for the specified type.

    Args:
        transform_type: Type of transformation

    Returns:
        Function that applies the transformation
    """
    if transform_type == "exp":
        return safe_exp
    elif transform_type == "relu":
        return functional.relu
    elif transform_type == "softplus":
        return functional.softplus
    elif transform_type == "identity":
        return lambda x: x
    else:
        raise ValueError(
            f"Unsupported weight transform type: {transform_type}. "
            f"Supported types: 'identity', 'exp', 'relu', 'softplus'"
        )


def inverse_weight_transform(
    positive_weights: torch.Tensor, transform_type: WeightTransformType = "exp"
) -> torch.Tensor:
    """
    Apply the inverse transformation to get raw parameters from positive weights.

    This is useful for initialization or analysis.

    Args:
        positive_weights: Positive weight values
        transform_type: Type of transformation that was applied

    Returns:
        Raw parameters that would produce the given positive weights
    """
    if transform_type == "exp":
        return torch.log(
            positive_weights + 1e-8
        )  # Add small epsilon for numerical stability
    elif transform_type == "relu":
        # ReLU doesn't have a unique inverse, return the weights themselves
        return positive_weights
    elif transform_type == "softplus":
        return inverse_softplus(positive_weights)
    elif transform_type == "identity":
        # Identity transform is its own inverse (signed weights allowed)
        return positive_weights
    else:
        raise ValueError(
            f"Unsupported weight transform type: {transform_type}. "
            f"Supported types: 'identity', 'exp', 'relu', 'softplus'"
        )


__all__ = [
    "WeightTransformType",
    "apply_weight_transform",
    "get_weight_transform_fn",
    "inverse_softplus",
    "inverse_weight_transform",
    "safe_exp",
]
