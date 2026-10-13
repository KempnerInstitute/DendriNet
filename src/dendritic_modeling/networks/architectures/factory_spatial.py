"""Spatial dendritic architecture factory helpers."""

from __future__ import annotations

from typing import Any

import torch.nn as nn
from omegaconf import DictConfig

from dendritic_modeling.config.conversion import to_plain_dict as _to_plain_mapping

# The type names live in the configuration package, which checks the sections
# each type reads without importing torch.
# Spatial dendritic cores: flatten encoder spatial output -> EINet -> project.
# Tensor-map baseline: flatten -> DendriNet -> project (no spatial locality).
# Spatial dendritic conv: F.unfold -> shared DendriNet on local patches -> fold.
# Preserves spatial structure (B, C_out, H_out, W_out).
# Canonical name: hierarchical_dendritic_conv
from dendritic_modeling.config.core_sections import (
    SPATIAL_DENDRITIC_CONV_TYPES,
    SPATIAL_DENDRITIC_TYPES,
    SPATIAL_PATCH_POINT_TYPES,
)
from dendritic_modeling.networks.architectures.recurrent.structured_einet_factory import (
    reject_recurrent_request_for_feedforward_core,
)

# Cores that output feature maps (B, C_out, H_out, W_out).
FEATURE_MAP_CORE_TYPES: frozenset[str] = (
    SPATIAL_DENDRITIC_CONV_TYPES | SPATIAL_PATCH_POINT_TYPES
)

# Cores that read images (B, C, H, W): the architecture factory receives their
# channel count C as ``input_dim``.
IMAGE_INPUT_CORE_TYPES: frozenset[str] = FEATURE_MAP_CORE_TYPES | {"dendritic_alexnet"}


def _build_spatial_dendritic_architecture(
    type: str,
    parameters: dict[str, Any] | DictConfig,
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> nn.Module:
    """Build flattened spatial dendritic tensor-map cores."""

    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
        HierarchicalDendriticTensorMap,
    )

    if input_dim is None:
        raise ValueError(
            f"input_dim required when creating spatial dendritic core (type='{type}')"
        )
    reject_recurrent_request_for_feedforward_core(type, _to_plain_mapping(parameters))

    return HierarchicalDendriticTensorMap(
        config=parameters,
        input_dim=input_dim,
        suffix_input_dim=suffix_input_dim,
    )


def _build_spatial_dendritic_conv_architecture(
    type: str,
    parameters: dict[str, Any] | DictConfig,
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> nn.Module:
    """Build shared-patch spatial dendritic convolution cores."""

    del suffix_input_dim
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.spatial_dendritic import (
        HierarchicalDendriticConv,
    )

    raw_params = _to_plain_mapping(parameters)
    reject_recurrent_request_for_feedforward_core(type, raw_params)
    spatial_cfg = _to_plain_mapping(raw_params.get("spatial", {}))
    kernel_size = spatial_cfg.get("kernel_size", 3)
    stride = spatial_cfg.get("stride", 1)
    padding = spatial_cfg.get("padding", 0)
    input_transform = spatial_cfg.get("input_transform", "identity")
    input_scale = spatial_cfg.get("input_scale", 1.0)
    output_adapter_mode = spatial_cfg.get("output_adapter_mode", "identity")
    output_initial_scale = spatial_cfg.get("output_initial_scale", 1.0)
    output_initial_threshold = spatial_cfg.get("output_initial_threshold", 0.25)

    if input_dim is None:
        raise ValueError(
            f"input_dim required when creating spatial dendritic conv (type='{type}')"
        )

    # input_dim here is the number of input channels (C_in). For spatial
    # cores, input_dim = prefix_output_spec.channels.
    return HierarchicalDendriticConv(
        config=parameters,
        in_channels=input_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        input_transform=input_transform,
        input_scale=input_scale,
        output_adapter_mode=output_adapter_mode,
        output_initial_scale=output_initial_scale,
        output_initial_threshold=output_initial_threshold,
    )


def _build_spatial_patch_point_architecture(
    type: str,
    parameters: dict[str, Any] | DictConfig,
    input_dim: int | None,
    suffix_input_dim: int | None,
) -> nn.Module:
    """Build a shared-patch fixed-index affine control."""

    del type, suffix_input_dim
    from dendritic_modeling.networks.architectures.classical.cnn.sparse_active_matched import (
        SparseActiveMatchedPointConv,
    )

    if input_dim is None:
        raise ValueError("input_dim is required for a spatial sparse point core")
    raw_params = _to_plain_mapping(parameters)
    spatial_cfg = _to_plain_mapping(raw_params.get("spatial", {}))
    return SparseActiveMatchedPointConv(
        config=parameters,
        in_channels=input_dim,
        kernel_size=spatial_cfg.get("kernel_size", 3),
        stride=spatial_cfg.get("stride", 1),
        padding=spatial_cfg.get("padding", 0),
    )


__all__ = [
    "FEATURE_MAP_CORE_TYPES",
    "IMAGE_INPUT_CORE_TYPES",
    "SPATIAL_DENDRITIC_CONV_TYPES",
    "SPATIAL_DENDRITIC_TYPES",
    "SPATIAL_PATCH_POINT_TYPES",
    "_build_spatial_dendritic_architecture",
    "_build_spatial_dendritic_conv_architecture",
    "_build_spatial_patch_point_architecture",
]
