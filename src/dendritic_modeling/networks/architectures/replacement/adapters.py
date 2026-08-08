"""Input adapters shared by replacement modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_VALID_INPUT_TRANSFORMS = frozenset({"identity", "relu", "signed_split"})


def validate_input_transform(transform: str) -> str:
    """Return a normalized replacement input transform name."""
    normalized = str(transform)
    if normalized not in _VALID_INPUT_TRANSFORMS:
        raise ValueError("input_transform must be identity, relu, or signed_split")
    return normalized


def transformed_feature_dim(input_dim: int, transform: str) -> int:
    """Feature dimension after applying a replacement input adapter."""
    normalized = validate_input_transform(transform)
    multiplier = 2 if normalized == "signed_split" else 1
    return int(input_dim) * multiplier


class NonNegativeInputAdapter(nn.Module):
    """Map signed features into the non-negative domain expected by dendrites.

    ``identity`` is available for callers that already guarantee non-negative
    inputs.  ``relu`` keeps only positive activity.  ``signed_split`` preserves
    both polarities as ``[ReLU(x), ReLU(-x)]`` using non-negative channels.
    """

    def __init__(
        self,
        input_dim: int,
        transform: str = "signed_split",
        *,
        validate_shape: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.transform = validate_input_transform(transform)
        self.output_dim = transformed_feature_dim(self.input_dim, self.transform)
        self.validate_shape = bool(validate_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.validate_shape and x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim}, got {x.shape[-1]}"
            )
        if self.transform == "identity":
            return x
        if self.transform == "relu":
            return torch.relu(x)
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=-1)

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
            f"transform={self.transform!r}"
        )


class SpatialNonNegativeInputAdapter(nn.Module):
    """Apply a non-negative transform to BCHW feature maps.

    Signed splitting concatenates positive and negative magnitudes along the
    channel dimension, preserving the signed tensor exactly while presenting
    only non-negative inputs to a dendritic convolution.
    """

    def __init__(
        self,
        input_channels: int,
        transform: str = "signed_split",
        *,
        validate_shape: bool = True,
    ):
        super().__init__()
        self.input_channels = int(input_channels)
        self.transform = validate_input_transform(transform)
        self.output_channels = transformed_feature_dim(
            self.input_channels, self.transform
        )
        self.validate_shape = bool(validate_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                "SpatialNonNegativeInputAdapter expects BCHW input, "
                f"got shape {tuple(x.shape)}"
            )
        if self.validate_shape and x.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {x.shape[1]}"
            )
        if self.transform == "identity":
            return x
        if self.transform == "relu":
            return torch.relu(x)
        return torch.cat([torch.relu(x), torch.relu(-x)], dim=1)

    def extra_repr(self) -> str:
        return (
            f"input_channels={self.input_channels}, "
            f"output_channels={self.output_channels}, "
            f"transform={self.transform!r}"
        )


class PositiveFeatureScale(nn.Module):
    """Apply a positive scalar or featurewise gain without changing layout.

    The adapter preserves non-negativity and channel identity. It is therefore
    suitable between dendritic blocks, where an unrestricted affine projection
    would obscure the representation being tested.
    """

    def __init__(
        self,
        num_features: int,
        *,
        feature_axis: int,
        initial_scale: float = 1.0,
        per_feature: bool = True,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.feature_axis = int(feature_axis)
        self.per_feature = bool(per_feature)
        self.learnable = bool(learnable)
        initial_scale = float(initial_scale)
        if self.num_features < 1:
            raise ValueError("num_features must be positive")
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("initial_scale must be finite and positive")
        size = self.num_features if self.per_feature else 1
        value = torch.full((size,), math.log(initial_scale), dtype=torch.float32)
        if self.learnable:
            self.log_scale = nn.Parameter(value)
        else:
            self.register_buffer("log_scale", value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axis = (
            self.feature_axis if self.feature_axis >= 0 else x.ndim + self.feature_axis
        )
        if axis < 0 or axis >= x.ndim:
            raise ValueError(
                f"feature_axis={self.feature_axis} is invalid for shape {tuple(x.shape)}"
            )
        if self.per_feature and x.shape[axis] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features on axis {axis}, "
                f"got {x.shape[axis]}"
            )
        shape = [1] * x.ndim
        shape[axis] = self.log_scale.numel()
        return x * self.log_scale.exp().view(shape)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, feature_axis={self.feature_axis}, "
            f"per_feature={self.per_feature}, learnable={self.learnable}"
        )
