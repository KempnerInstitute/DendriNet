"""
Pretrained backbone splitter for encoder-core-decoder pipelines.

Splits Sequential-based torchvision backbones (e.g., AlexNet, VGG) at named
layers to create a prefix encoder and suffix decoder, with a dendritic core
network replacing the layers in between.

Models whose ``forward()`` is not a simple Sequential walk (e.g., ResNet with
skip connections) are supported only when the split points fall *outside* the
residual blocks — typically ``avgpool`` or ``fc``.

Config::

    model:
      pretrained_replacement:
        backbone: "alexnet"
        weights: "IMAGENET1K_V1"
        input_shape: [3, 224, 224]

        replace:
          start: "classifier.4"       # first replaced layer (inclusive)
          end: "classifier.5"         # last replaced layer (inclusive)

        omit_layers: ["classifier.0", "classifier.3"]   # e.g. dropout

        trainability:
          encoder:
            mode: "frozen"
          decoder:
            mode: "trainable"

      core:
        type: "einet"
        ...
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import get_model

from dendritic_modeling.networks.base import BaseNetwork

logger = logging.getLogger(__name__)

_DEFAULT_INPUT_SIZE = (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------
def _load_backbone(backbone: str, weights: str | None = None) -> nn.Module:
    """Load a torchvision model, optionally with pretrained weights or a local file."""
    if weights is None:
        return get_model(backbone, weights=None)

    if os.path.sep in weights or weights.endswith((".pt", ".pth")):
        model = get_model(backbone, weights=None)
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded weights from local file: %s", weights)
        return model

    return get_model(backbone, weights=weights)


# ---------------------------------------------------------------------------
# Layer utilities
# ---------------------------------------------------------------------------
def _flatten_model_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Walk top-level children, expanding ``nn.Sequential`` containers.

    Returns ``(dotted_name, module)`` pairs in forward-execution order.
    Non-Sequential children (e.g. ResNet residual blocks) are kept atomic.
    """
    layers: list[tuple[str, nn.Module]] = []
    for name, child in model.named_children():
        if isinstance(child, nn.Sequential):
            for sub_name, sub_child in child.named_children():
                layers.append((f"{name}.{sub_name}", sub_child))
        else:
            layers.append((name, child))
    return layers


def list_backbone_layers(backbone: str) -> list[str]:
    """Return the flattened layer names for a torchvision backbone.

    Useful for discovering valid split points::

        >>> list_backbone_layers("alexnet")
        ['features.0', 'features.1', ..., 'classifier.6']
    """
    model = get_model(backbone, weights=None)
    return [name for name, _ in _flatten_model_layers(model)]


def _extract_layers(
    model: nn.Module,
    *,
    split_after: str | None = None,
    split_from: str | None = None,
    drop_layers: list[str] | None = None,
) -> tuple[list[nn.Module], list[str]]:
    """Extract a contiguous slice of layers from a model.

    Used internally by :func:`build_split_plan`.
    """
    all_layers = _flatten_model_layers(model)
    all_names = [name for name, _ in all_layers]
    drop_set = set(drop_layers or [])

    if split_after is not None:
        if split_after not in all_names:
            raise ValueError(
                f"split_after='{split_after}' not found. "
                f"Available layers: {all_names}"
            )
        cut_idx = all_names.index(split_after) + 1
        selected = all_layers[:cut_idx]
    elif split_from is not None:
        if split_from not in all_names:
            raise ValueError(
                f"split_from='{split_from}' not found. "
                f"Available layers: {all_names}"
            )
        cut_idx = all_names.index(split_from)
        selected = all_layers[cut_idx:]
    else:
        raise ValueError("Either split_after or split_from must be specified")

    modules = []
    names = []
    for name, mod in selected:
        if name not in drop_set:
            modules.append(mod)
            names.append(name)

    return modules, names


def _needs_flatten(modules: list[nn.Module]) -> bool:
    has_spatial = any(
        isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d))
        for m in modules
    )
    has_linear = any(isinstance(m, nn.Linear) for m in modules)
    return has_spatial and has_linear


def _insert_flatten(modules: list[nn.Module]) -> list[nn.Module]:
    result = []
    inserted = False
    for m in modules:
        if isinstance(m, nn.Linear) and not inserted:
            result.append(nn.Flatten(start_dim=1))
            inserted = True
        result.append(m)
    return result


# ---------------------------------------------------------------------------
# BackboneSegment — generic wrapper for a prefix or suffix slice
# ---------------------------------------------------------------------------
class BackboneSegment(BaseNetwork):
    """A contiguous slice of a pretrained backbone.

    Wraps a list of modules into an ``nn.Sequential``, handles spatial→flat
    transitions automatically, and exposes ``output_dim``.
    """

    def __init__(
        self,
        modules: list[nn.Module],
        names: list[str],
        *,
        freeze: bool = False,
        input_size: tuple[int, ...] = _DEFAULT_INPUT_SIZE,
        auto_flatten: bool = True,
    ):
        super().__init__()

        if auto_flatten and _needs_flatten(modules):
            modules = _insert_flatten(modules)

        self.layers = nn.Sequential(*modules)
        self.frozen = freeze
        self._layer_names = names

        # Compute output_dim via dummy forward
        with torch.no_grad():
            dummy = torch.zeros(*input_size)
            out = self.layers(dummy)

        if auto_flatten and out.dim() > 2:
            self.layers.append(nn.Flatten(start_dim=1))
            out = out.flatten(start_dim=1)

        self.output_dim = out.shape[-1]

        if freeze:
            self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeezed = True

        batch_shape = x.shape[:-3] if x.dim() > 2 else x.shape[:-1]
        if x.dim() > 2:
            x = x.reshape(-1, *x.shape[-3:])

        out = self.layers(x)
        out = out.reshape(*batch_shape, -1)

        if squeezed:
            out = out.squeeze(0)
        return out

    def get_output_dim(self):
        return self.output_dim


# ---------------------------------------------------------------------------
# BackboneSplitPlan — single-load, three-way split
# ---------------------------------------------------------------------------
@dataclass
class BackboneSplitPlan:
    """Result of splitting a pretrained backbone into prefix / replaced / suffix.

    Built by :func:`build_split_plan`. Holds the resolved modules and metadata
    needed to construct encoder (prefix), core, and decoder (suffix).
    """

    backbone: str
    weights: str | None

    # Layer names
    all_layer_names: list[str]
    prefix_names: list[str]
    replaced_names: list[str]
    suffix_names: list[str]

    # Modules (from a single loaded model instance)
    prefix_modules: list[nn.Module]
    suffix_modules: list[nn.Module]

    # Shape metadata
    input_size: tuple[int, ...]
    prefix_output_dim: int
    suffix_input_dim: int  # expected input dim for the first suffix Linear
    suffix_output_dim: int

    def summary(self) -> str:
        lines = [
            f"BackboneSplitPlan(backbone={self.backbone!r})",
            f"  prefix  ({len(self.prefix_names)} layers): {self.prefix_names}",
            f"  replace ({len(self.replaced_names)} layers): {self.replaced_names}",
            f"  suffix  ({len(self.suffix_names)} layers): {self.suffix_names}",
            f"  prefix_output_dim={self.prefix_output_dim}",
            f"  suffix_input_dim={self.suffix_input_dim}",
            f"  suffix_output_dim={self.suffix_output_dim}",
        ]
        return "\n".join(lines)


def build_split_plan(
    backbone: str,
    weights: str | None = None,
    *,
    replace_start: str,
    replace_end: str,
    omit_layers: list[str] | None = None,
    input_shape: list[int] | None = None,
) -> BackboneSplitPlan:
    """Build a split plan by loading the backbone once and slicing it three ways.

    Args:
        backbone: Torchvision model name.
        weights: Weight string, local path, or None.
        replace_start: First layer to replace (inclusive).
        replace_end: Last layer to replace (inclusive).
        omit_layers: Layer names to drop entirely (e.g. dropout).
        input_shape: Model input shape ``[C, H, W]``. Defaults to ``[3, 224, 224]``.
    """
    model = _load_backbone(backbone, weights)
    all_layers = _flatten_model_layers(model)
    all_names = [name for name, _ in all_layers]
    omit_set = set(omit_layers or [])

    for required in (replace_start, replace_end):
        if required not in all_names:
            raise ValueError(
                f"Layer {required!r} not found in {backbone}. "
                f"Available: {all_names}"
            )

    start_idx = all_names.index(replace_start)
    end_idx = all_names.index(replace_end)
    if start_idx > end_idx:
        raise ValueError(
            f"replace_start={replace_start!r} comes after "
            f"replace_end={replace_end!r} in the layer order"
        )

    prefix_layers = all_layers[:start_idx]
    replaced_layers = all_layers[start_idx : end_idx + 1]
    suffix_layers = all_layers[end_idx + 1 :]

    def _filter(layers):
        return [(n, m) for n, m in layers if n not in omit_set]

    prefix_filtered = _filter(prefix_layers)
    suffix_filtered = _filter(suffix_layers)

    prefix_names = [n for n, _ in prefix_filtered]
    prefix_modules = [m for _, m in prefix_filtered]
    replaced_names = [n for n, _ in replaced_layers]
    suffix_names = [n for n, _ in suffix_filtered]
    suffix_modules = [m for _, m in suffix_filtered]

    # Compute prefix output shape
    if input_shape:
        input_size = (1, *input_shape)
    else:
        input_size = _DEFAULT_INPUT_SIZE

    # Build temporary prefix to measure its output dim
    tmp_prefix_mods = list(prefix_modules)
    if _needs_flatten(tmp_prefix_mods):
        tmp_prefix_mods = _insert_flatten(tmp_prefix_mods)
    tmp_prefix = nn.Sequential(*tmp_prefix_mods)
    with torch.no_grad():
        prefix_out = tmp_prefix(torch.zeros(*input_size))
    if prefix_out.dim() > 2:
        prefix_out = prefix_out.flatten(start_dim=1)
    prefix_output_dim = prefix_out.shape[-1]

    # Infer suffix input_dim from its first Linear layer
    suffix_input_dim = prefix_output_dim  # default: assume core matches prefix
    for m in suffix_modules:
        if isinstance(m, nn.Linear):
            suffix_input_dim = m.in_features
            break

    # Infer suffix output_dim from its last Linear layer
    suffix_output_dim = 0
    for m in reversed(suffix_modules):
        if isinstance(m, nn.Linear):
            suffix_output_dim = m.out_features
            break
    if suffix_output_dim == 0 and suffix_modules:
        # Dummy forward to figure it out
        with torch.no_grad():
            dummy_in = torch.zeros(1, suffix_input_dim)
            dummy_out = nn.Sequential(*suffix_modules)(dummy_in)
        suffix_output_dim = dummy_out.shape[-1]

    plan = BackboneSplitPlan(
        backbone=backbone,
        weights=weights,
        all_layer_names=all_names,
        prefix_names=prefix_names,
        replaced_names=replaced_names,
        suffix_names=suffix_names,
        prefix_modules=prefix_modules,
        suffix_modules=suffix_modules,
        input_size=input_size,
        prefix_output_dim=prefix_output_dim,
        suffix_input_dim=suffix_input_dim,
        suffix_output_dim=suffix_output_dim,
    )

    logger.info("Built split plan:\n%s", plan.summary())
    return plan


def build_encoder_decoder_from_plan(
    plan: BackboneSplitPlan,
    *,
    prefix_freeze: bool = True,
    suffix_freeze: bool = False,
) -> tuple[BackboneSegment, BackboneSegment]:
    """Create encoder and decoder ``BackboneSegment`` instances from a plan."""
    encoder = BackboneSegment(
        plan.prefix_modules,
        plan.prefix_names,
        freeze=prefix_freeze,
        input_size=plan.input_size,
        auto_flatten=True,
    )
    decoder = BackboneSegment(
        plan.suffix_modules,
        plan.suffix_names,
        freeze=suffix_freeze,
        input_size=(1, plan.suffix_input_dim),
        auto_flatten=False,
    )
    return encoder, decoder


__all__ = [
    "BackboneSegment",
    "BackboneSplitPlan",
    "build_encoder_decoder_from_plan",
    "build_split_plan",
    "list_backbone_layers",
]
