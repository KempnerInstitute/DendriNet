"""Input adapters shared by replacement modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

_VALID_INPUT_TRANSFORMS = frozenset({"identity", "relu", "signed_split"})
_VALID_POPULATION_REDUCTIONS = frozenset(
    {"one_to_one", "fixed_mean", "learned_positive"}
)
_IDENTITY_OUTPUT_ADAPTERS = frozenset({"identity", "none", "off"})
_GLOBAL_SCALE_OUTPUT_ADAPTERS = frozenset({"learned_global", "global"})
_FEATURE_SCALE_OUTPUT_ADAPTERS = frozenset(
    {"learned_per_channel", "per_channel", "per_feature"}
)
_SIGNED_AFFINE_OUTPUT_ADAPTERS = frozenset(
    {"signed_per_channel_affine", "per_channel_affine"}
)
_GLOBAL_THRESHOLD_OUTPUT_ADAPTERS = frozenset(
    {"learned_global_threshold_relu", "global_threshold_relu"}
)
_FEATURE_THRESHOLD_OUTPUT_ADAPTERS = frozenset(
    {
        "learned_per_channel_threshold_relu",
        "per_channel_threshold_relu",
        "per_feature_threshold_relu",
    }
)


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


class PositiveThresholdReLU(nn.Module):
    """Apply a positive gain and threshold without mixing feature identities.

    The transform ``ReLU(scale * x - threshold)`` supplies an explicit sparse
    output contract for a nonnegative replacement. Both quantities are kept
    positive through a log parameterization and can be global or featurewise.
    """

    def __init__(
        self,
        num_features: int,
        *,
        feature_axis: int,
        initial_scale: float = 1.0,
        initial_threshold: float = 0.25,
        per_feature: bool = True,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.feature_axis = int(feature_axis)
        self.per_feature = bool(per_feature)
        self.learnable = bool(learnable)
        initial_scale = float(initial_scale)
        initial_threshold = float(initial_threshold)
        if self.num_features < 1:
            raise ValueError("num_features must be positive")
        if not math.isfinite(initial_scale) or initial_scale <= 0:
            raise ValueError("initial_scale must be finite and positive")
        if not math.isfinite(initial_threshold) or initial_threshold <= 0:
            raise ValueError("initial_threshold must be finite and positive")
        size = self.num_features if self.per_feature else 1
        scale = torch.full((size,), math.log(initial_scale), dtype=torch.float32)
        threshold = torch.full(
            (size,), math.log(initial_threshold), dtype=torch.float32
        )
        if self.learnable:
            self.log_scale = nn.Parameter(scale)
            self.log_threshold = nn.Parameter(threshold)
        else:
            self.register_buffer("log_scale", scale)
            self.register_buffer("log_threshold", threshold)

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
        scale = self.log_scale.exp().view(shape)
        threshold = self.log_threshold.exp().view(shape)
        return torch.relu(x * scale - threshold)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, feature_axis={self.feature_axis}, "
            f"per_feature={self.per_feature}, learnable={self.learnable}"
        )


class SignedPerChannelAffine(nn.Module):
    """Signed per-channel affine ``gain * x + bias`` without channel mixing.

    The signed counterpart of :class:`PositiveFeatureScale`: gains and biases
    are unconstrained, so a core can carry the bias term of a layer it stands
    in for, or recenter a code. Initialized to the identity (gain 1, bias 0),
    so selecting the adapter changes nothing until trained or explicitly set.
    """

    def __init__(
        self,
        num_features: int,
        *,
        feature_axis: int,
        learnable: bool = True,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.feature_axis = int(feature_axis)
        if self.num_features < 1:
            raise ValueError("num_features must be positive")
        gain = torch.ones(self.num_features)
        bias = torch.zeros(self.num_features)
        if learnable:
            self.gain = nn.Parameter(gain)
            self.bias = nn.Parameter(bias)
        else:
            self.register_buffer("gain", gain)
            self.register_buffer("bias", bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axis = (
            self.feature_axis if self.feature_axis >= 0 else x.ndim + self.feature_axis
        )
        if axis < 0 or axis >= x.ndim:
            raise ValueError(
                f"feature_axis={self.feature_axis} is invalid for shape {tuple(x.shape)}"
            )
        if x.shape[axis] != self.num_features:
            raise ValueError(
                f"Expected {self.num_features} features on axis {axis}, "
                f"got {x.shape[axis]}"
            )
        shape = [1] * x.ndim
        shape[axis] = self.num_features
        return x * self.gain.view(shape) + self.bias.view(shape)

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, feature_axis={self.feature_axis}"


def make_nonmixing_output_adapter(
    *,
    num_features: int,
    feature_axis: int,
    mode: str,
    initial_scale: float = 1.0,
    initial_threshold: float = 0.25,
) -> nn.Module:
    """Construct a channel-preserving positive output adapter by name."""

    normalized = str(mode).strip().lower()
    if normalized in _IDENTITY_OUTPUT_ADAPTERS:
        return nn.Identity()
    if normalized in _SIGNED_AFFINE_OUTPUT_ADAPTERS:
        return SignedPerChannelAffine(
            num_features,
            feature_axis=feature_axis,
            learnable=True,
        )
    if normalized in _GLOBAL_SCALE_OUTPUT_ADAPTERS:
        return PositiveFeatureScale(
            num_features,
            feature_axis=feature_axis,
            initial_scale=initial_scale,
            per_feature=False,
            learnable=True,
        )
    if normalized in _FEATURE_SCALE_OUTPUT_ADAPTERS:
        return PositiveFeatureScale(
            num_features,
            feature_axis=feature_axis,
            initial_scale=initial_scale,
            per_feature=True,
            learnable=True,
        )
    if normalized in _GLOBAL_THRESHOLD_OUTPUT_ADAPTERS:
        return PositiveThresholdReLU(
            num_features,
            feature_axis=feature_axis,
            initial_scale=initial_scale,
            initial_threshold=initial_threshold,
            per_feature=False,
            learnable=True,
        )
    if normalized in _FEATURE_THRESHOLD_OUTPUT_ADAPTERS:
        return PositiveThresholdReLU(
            num_features,
            feature_axis=feature_axis,
            initial_scale=initial_scale,
            initial_threshold=initial_threshold,
            per_feature=True,
            learnable=True,
        )
    raise ValueError(
        "output adapter mode must be identity, learned_global, "
        "learned_per_channel, learned_global_threshold_relu, or "
        f"learned_per_channel_threshold_relu; got {mode!r}"
    )


class PopulationToFeatureReduction(nn.Module):
    """Reduce a fixed soma group to each downstream feature.

    The input feature axis is organized as contiguous groups with
    ``somas_per_feature`` somas per output feature. ``fixed_mean`` gives every
    soma equal weight. ``learned_positive`` learns a separate simplex over the
    soma group of every feature, retaining population identity without mixing
    across output channels or units.
    """

    def __init__(
        self,
        num_features: int,
        *,
        somas_per_feature: int = 1,
        feature_axis: int = -1,
        mode: str = "one_to_one",
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.somas_per_feature = int(somas_per_feature)
        self.feature_axis = int(feature_axis)
        self.mode = str(mode).strip().lower()
        if self.num_features < 1:
            raise ValueError("num_features must be positive")
        if self.somas_per_feature < 1:
            raise ValueError("somas_per_feature must be positive")
        if self.mode not in _VALID_POPULATION_REDUCTIONS:
            raise ValueError("mode must be one_to_one, fixed_mean, or learned_positive")
        if self.mode == "one_to_one" and self.somas_per_feature != 1:
            raise ValueError("one_to_one requires somas_per_feature=1")
        if self.mode == "learned_positive":
            self.logits = nn.Parameter(
                torch.zeros(self.num_features, self.somas_per_feature)
            )
        else:
            self.register_parameter("logits", None)

    @property
    def input_features(self) -> int:
        return self.num_features * self.somas_per_feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        axis = (
            self.feature_axis if self.feature_axis >= 0 else x.ndim + self.feature_axis
        )
        if axis < 0 or axis >= x.ndim:
            raise ValueError(
                f"feature_axis={self.feature_axis} is invalid for shape {tuple(x.shape)}"
            )
        if x.shape[axis] != self.input_features:
            raise ValueError(
                f"Expected {self.input_features} population features on axis {axis}, "
                f"got {x.shape[axis]}"
            )
        if self.mode == "one_to_one":
            return x

        grouped_shape = [*x.shape[:axis], self.num_features, self.somas_per_feature]
        grouped_shape.extend(x.shape[axis + 1 :])
        grouped = x.reshape(grouped_shape)
        soma_axis = axis + 1
        if self.mode == "fixed_mean":
            return grouped.mean(dim=soma_axis)

        weights = torch.softmax(self.logits, dim=-1)
        weight_shape = [1] * grouped.ndim
        weight_shape[axis] = self.num_features
        weight_shape[soma_axis] = self.somas_per_feature
        return (grouped * weights.view(weight_shape)).sum(dim=soma_axis)

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, "
            f"somas_per_feature={self.somas_per_feature}, "
            f"feature_axis={self.feature_axis}, mode={self.mode!r}"
        )
