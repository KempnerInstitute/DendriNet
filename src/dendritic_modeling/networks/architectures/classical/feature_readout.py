"""Conventional feature normalization composed with the existing affine builder."""

import copy
import math
from dataclasses import asdict

import torch
from torch import nn

from dendritic_modeling.config.feature_readout import FeatureReadoutConfig
from dendritic_modeling.networks.architectures.classical.mlp import build_affine_mlp


class FeatureReadout(nn.Module):
    """Normalize channel territories, flatten and apply an affine readout.

    Group-mean division uses ``offset + max(mean(group) * noise, 0)``.
    It is neither L1 normalization nor division by an E/I total conductance.
    The readout is constructed first; no wrapper-specific random draws occur.
    """

    def __init__(self, config: FeatureReadoutConfig):
        super().__init__()
        config.validate()
        self.config = copy.deepcopy(config)
        self.feature_readout_config = asdict(config)
        self.input_dim = math.prod(config.input_shape)
        self.output_dim = config.readout.dimensions[-1]
        readout = build_affine_mlp(config.readout)
        # Preserve plain Linear keys for a single unactivated projection.
        self.readout = readout[0] if len(readout) == 1 else readout
        if config.normalization == "layer_norm":
            self.normalizer = nn.LayerNorm(
                tuple(config.input_shape),
                eps=config.normalization_epsilon,
                elementwise_affine=config.normalization_affine,
            )
        elif config.normalization == "group_norm":
            self.normalizer = nn.GroupNorm(
                config.groups,
                config.input_shape[0],
                eps=config.normalization_epsilon,
                affine=config.normalization_affine,
            )
        else:
            self.normalizer = None

    def representation(self, x, *, divisor_noise=None):
        """Return flattened coordinates; noise is an explicit per-group multiplier.

        Inputs may be feature maps or already flattened, with one batch axis.
        Noise has shape [batch, groups], the same dtype/device, and must be
        finite and nonnegative. No random noise is generated internally.
        """
        if not isinstance(x, torch.Tensor) or not x.is_floating_point():
            raise TypeError("features must be a floating-point tensor")
        shape = tuple(self.config.input_shape)
        if x.ndim == 2 and x.shape[1] == self.input_dim:
            x = x.reshape(x.shape[0], *shape)
        elif x.ndim != len(shape) + 1 or tuple(x.shape[1:]) != shape:
            raise ValueError("features must match input_shape or its flattened size")
        if divisor_noise is not None:
            if self.config.normalization != "group_mean":
                raise ValueError("divisor_noise only applies to group_mean")
            if (
                not isinstance(divisor_noise, torch.Tensor)
                or divisor_noise.shape != (x.shape[0], self.config.groups)
                or divisor_noise.dtype != x.dtype
                or divisor_noise.device != x.device
            ):
                raise ValueError(
                    "divisor_noise must match batch/groups and input dtype/device"
                )
            if not torch.isfinite(divisor_noise).all() or (divisor_noise < 0).any():
                raise ValueError("divisor_noise must be finite and nonnegative")
        if self.config.normalization == "group_mean":
            groups = self.config.groups
            grouped = x.reshape(x.shape[0], groups, shape[0] // groups, *shape[1:])
            divisor = grouped.mean(dim=tuple(range(2, grouped.ndim)), keepdim=True)
            if divisor_noise is not None:
                divisor = divisor * divisor_noise.reshape(
                    *divisor_noise.shape, *([1] * (grouped.ndim - 2))
                )
            x = (
                grouped / (self.config.divisor_offset + divisor.clamp_min(0.0))
            ).reshape_as(x)
        elif self.normalizer is not None:
            x = self.normalizer(x)
        return x.flatten(1)

    def forward(self, x, *, divisor_noise=None):
        return self.readout(self.representation(x, divisor_noise=divisor_noise))


def build_feature_readout(config: FeatureReadoutConfig) -> FeatureReadout:
    return FeatureReadout(config)
