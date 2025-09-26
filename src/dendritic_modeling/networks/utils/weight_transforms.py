"""
Weight transformation utilities for positive weight constraints.

This module provides different transformation functions to ensure synaptic weights
remain positive during training. Each transformation has different characteristics:

"""

from typing import Literal

import torch
import torch.nn.functional as functional

WeightTransformType = Literal["exp", "relu", "softplus"]


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
        return torch.exp(weights)
    elif transform_type == "relu":
        return functional.relu(weights)
    elif transform_type == "softplus":
        return functional.softplus(weights)
    else:
        raise ValueError(
            f"Unsupported weight transform type: {transform_type}. "
            f"Supported types: 'exp', 'relu', 'softplus'"
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
        return torch.exp
    elif transform_type == "relu":
        return functional.relu
    elif transform_type == "softplus":
        return functional.softplus
    else:
        raise ValueError(
            f"Unsupported weight transform type: {transform_type}. "
            f"Supported types: 'exp', 'relu', 'softplus'"
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
        # Inverse of softplus: log(exp(x) - 1)
        # For numerical stability, use the identity: softplus_inverse(x) = log(exp(x) - 1) = x + log(1 - exp(-x)) for x > 0
        return torch.log(torch.exp(positive_weights) - 1 + 1e-8)
    else:
        raise ValueError(
            f"Unsupported weight transform type: {transform_type}. "
            f"Supported types: 'exp', 'relu', 'softplus'"
        )


__all__ = [
    "WeightTransformType",
    "apply_weight_transform",
    "get_weight_transform_fn",
    "inverse_weight_transform",
]
