"""Spatial cores as the core of an encoder-core-decoder model."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from dendritic_modeling.networks.base import BaseNetwork


class FlattenedSpatialCore(BaseNetwork):
    """Apply a spatial core to images and flatten its feature maps.

    Spatial cores (``hierarchical_dendritic_conv``,
    ``sparse_active_matched_point_conv`` and their aliases) take the place of a
    convolution: they map ``(B, C, H, W)`` images to ``(B, C_out, H_out, W_out)``
    feature maps. As the core of an encoder-core-decoder model the decoder
    reads vectors, so the model builder wraps the spatial core in this module,
    which flattens the feature maps channel first:
    ``output_dim = C_out * H_out * W_out``. The image shape is fixed when the
    model is built; other shapes raise.

    Args:
        core: The spatial core, with ``out_channels`` and
            ``compute_output_shape(height, width)``.
        input_shape: The image shape ``(C, H, W)`` without the batch dimension.
    """

    def __init__(self, core: nn.Module, input_shape: Sequence[int]) -> None:
        super().__init__()
        if len(input_shape) != 3:
            raise ValueError(f"input_shape must be (C, H, W), got {tuple(input_shape)}")
        channels, height, width = (int(value) for value in input_shape)
        out_height, out_width = core.compute_output_shape(height, width)
        if out_height < 1 or out_width < 1:
            raise ValueError(
                f"{type(core).__name__} produces an empty feature map from "
                f"{height}x{width} images; reduce its kernel size or stride or "
                "add padding"
            )
        self.core = core
        self.input_shape = (channels, height, width)
        self.output_shape = (int(core.out_channels), int(out_height), int(out_width))
        self.output_dim = math.prod(self.output_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[1:]) != self.input_shape:
            raise ValueError(
                f"{type(self.core).__name__} was built for images of shape "
                f"{self.input_shape}, got input of shape {tuple(x.shape)}"
            )
        return self.core(x).flatten(1)

    def get_output_dim(self) -> int:
        return self.output_dim

    def get_effective_params(self) -> int:
        """The wrapped core's effective parameter count (flattening has none)."""
        count = getattr(self.core, "get_effective_params", None)
        if count is None:
            return sum(parameter.numel() for parameter in self.core.parameters())
        return int(count())


__all__ = ["FlattenedSpatialCore"]
