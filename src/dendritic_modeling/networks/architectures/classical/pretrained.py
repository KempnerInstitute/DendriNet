"""
Pretrained model splitter for encoder-core-decoder pipelines.

Splits Sequential-based torchvision backbones (e.g., AlexNet, VGG) at a
named layer to create a frozen encoder (pre-split layers) and trainable
decoder (post-split layers), with a dendritic core network replacing the
layers in between.

Models whose forward() is not a simple Sequential walk (e.g., ResNet with
skip connections) are supported only when the split point is *outside* the
residual blocks — typically ``avgpool`` or ``fc``.

Example config::

    model:
      encoder:
        type: "pretrained"
        params:
          backbone: "alexnet"
          weights: "IMAGENET1K_V1"
          split_after: "classifier.2"
          freeze: true
          drop_layers: ["classifier.0", "classifier.3"]

      core:
        type: "einet"
        ...

      decoder:
        type: "pretrained_decoder"
        params:
          backbone: "alexnet"
          weights: "IMAGENET1K_V1"
          split_from: "classifier.6"
          freeze: false
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import get_model

from dendritic_modeling.networks.base import BaseNetwork

logger = logging.getLogger(__name__)

# Default input size for torchvision models (ImageNet).
_DEFAULT_INPUT_SIZE = (1, 3, 224, 224)


def _load_backbone(backbone: str, weights: Optional[str] = None) -> nn.Module:
    """Load a torchvision model with pretrained weights or from a local file.

    Args:
        backbone: Torchvision model name (e.g., "alexnet", "resnet50").
        weights: One of:
            - Torchvision weights string (e.g., "IMAGENET1K_V1", "DEFAULT")
            - Local file path ending in .pt/.pth
            - None for random initialization
    """
    if weights is None:
        return get_model(backbone, weights=None)

    # Local checkpoint
    if os.path.sep in weights or weights.endswith((".pt", ".pth")):
        model = get_model(backbone, weights=None)
        state_dict = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        logger.info("Loaded weights from local file: %s", weights)
        return model

    # Torchvision weights enum string
    return get_model(backbone, weights=weights)


def _flatten_model_layers(model: nn.Module) -> list[tuple[str, nn.Module]]:
    """Walk model's top-level children, expanding Sequential containers.

    Returns an ordered list of (dotted_name, module) pairs in forward-execution
    order. Sequential containers are expanded so that individual sublayers can
    be referenced (e.g., "classifier.2" instead of just "classifier").

    Non-Sequential children are kept as atomic units (e.g., ResNet's "layer1"
    which contains residual blocks executed via its own forward()).
    """
    layers: list[tuple[str, nn.Module]] = []
    for name, child in model.named_children():
        if isinstance(child, nn.Sequential):
            for sub_name, sub_child in child.named_children():
                layers.append((f"{name}.{sub_name}", sub_child))
        else:
            layers.append((name, child))
    return layers


def _extract_layers(
    model: nn.Module,
    *,
    split_after: Optional[str] = None,
    split_from: Optional[str] = None,
    drop_layers: Optional[list[str]] = None,
) -> tuple[list[nn.Module], list[str]]:
    """Extract a slice of layers from a model.

    Args:
        model: The full torchvision model.
        split_after: Include layers up to and including this named layer.
        split_from: Include layers starting from this named layer.
        drop_layers: Layer names to exclude (e.g., dropout layers).

    Returns:
        Tuple of (list of modules, list of their names).
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

    # Filter out dropped layers
    modules = []
    names = []
    for name, mod in selected:
        if name not in drop_set:
            modules.append(mod)
            names.append(name)

    return modules, names


def _needs_flatten(modules: list[nn.Module]) -> bool:
    """Check if modules transition from spatial (conv/pool) to linear layers."""
    has_spatial = False
    has_linear = False
    for m in modules:
        if isinstance(m, (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            has_spatial = True
        if isinstance(m, nn.Linear):
            has_linear = True
    return has_spatial and has_linear


def _insert_flatten(modules: list[nn.Module]) -> list[nn.Module]:
    """Insert nn.Flatten before the first nn.Linear if preceded by spatial layers."""
    result = []
    inserted = False
    for m in modules:
        if isinstance(m, nn.Linear) and not inserted:
            result.append(nn.Flatten(start_dim=1))
            inserted = True
        result.append(m)
    return result


class PretrainedEncoder(BaseNetwork):
    """Frozen (or trainable) encoder from a pretrained torchvision model.

    Extracts layers from the model up to and including ``split_after``.
    Computes ``output_dim`` via a single dummy forward pass.

    Args:
        backbone: Torchvision model name.
        weights: Weights string, local path, or None.
        split_after: Named layer to include up to (inclusive).
        freeze: If True, freeze all parameters.
        drop_layers: Layer names to skip (e.g., dropout layers).
        input_size: Input tensor size for output_dim calculation.
            Defaults to (1, 3, 224, 224).
    """

    def __init__(
        self,
        backbone: str = "alexnet",
        weights: Optional[str] = "IMAGENET1K_V1",
        split_after: str = "classifier.2",
        freeze: bool = True,
        drop_layers: Optional[list[str]] = None,
        input_size: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__()

        model = _load_backbone(backbone, weights)
        modules, names = _extract_layers(
            model, split_after=split_after, drop_layers=drop_layers
        )

        # Auto-insert flatten at conv→FC boundary within the layer sequence
        if _needs_flatten(modules):
            modules = _insert_flatten(modules)

        self.layers = nn.Sequential(*modules)
        self.frozen = freeze

        # Compute output_dim via dummy forward
        size = tuple(input_size) if input_size else _DEFAULT_INPUT_SIZE
        with torch.no_grad():
            dummy = torch.zeros(*size)
            out = self.layers(dummy)

        # If output is still spatial (e.g., split ends at a pool layer),
        # append a flatten so output is always (batch, features).
        if out.dim() > 2:
            self.layers.append(nn.Flatten(start_dim=1))
            out = out.flatten(start_dim=1)

        self.output_dim = out.shape[-1]

        if freeze:
            self.requires_grad_(False)

        logger.info(
            "PretrainedEncoder: backbone=%s, split_after=%s, "
            "output_dim=%d, freeze=%s, layers=%s",
            backbone,
            split_after,
            self.output_dim,
            freeze,
            names,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = False
        if x.dim() == 3:  # (C, H, W) -> (1, C, H, W)
            x = x.unsqueeze(0)
            squeezed = True

        # Handle extra batch dimensions: (*, C, H, W) -> (B, C, H, W)
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])

        out = self.layers(x)
        out = out.reshape(*batch_shape, -1)

        if squeezed:
            out = out.squeeze(0)
        return out

    def get_output_dim(self):
        return self.output_dim


class PretrainedDecoder(BaseNetwork):
    """Decoder from a pretrained torchvision model.

    Extracts layers from ``split_from`` onward. Typically used for the
    output head (e.g., AlexNet's fc8).

    Args:
        backbone: Torchvision model name.
        weights: Weights string, local path, or None.
        split_from: Named layer to start from (inclusive).
        freeze: If True, freeze all parameters.
        drop_layers: Layer names to skip.
        input_dim: Expected input dimension (from core network output).
    """

    def __init__(
        self,
        backbone: str = "alexnet",
        weights: Optional[str] = "IMAGENET1K_V1",
        split_from: str = "classifier.6",
        freeze: bool = False,
        drop_layers: Optional[list[str]] = None,
        input_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        model = _load_backbone(backbone, weights)
        modules, names = _extract_layers(
            model, split_from=split_from, drop_layers=drop_layers
        )

        self.layers = nn.Sequential(*modules)

        # Validate input_dim against the first Linear layer if both are known
        if input_dim is not None:
            first_linear = None
            for m in modules:
                if isinstance(m, nn.Linear):
                    first_linear = m
                    break
            if first_linear is not None and first_linear.in_features != input_dim:
                raise ValueError(
                    f"PretrainedDecoder: input_dim={input_dim} does not match "
                    f"first Linear layer in_features={first_linear.in_features} "
                    f"(backbone={backbone!r}, split_from={split_from!r}). "
                    f"Check that core.output_dim matches decoder input."
                )

        # Determine output_dim from the last Linear layer or dummy forward
        last_linear = None
        for m in reversed(modules):
            if isinstance(m, nn.Linear):
                last_linear = m
                break

        if last_linear is not None:
            self.output_dim = last_linear.out_features
        elif input_dim is not None:
            with torch.no_grad():
                dummy = torch.zeros(1, input_dim)
                out = self.layers(dummy)
            self.output_dim = out.shape[-1]
        else:
            raise ValueError(
                "Cannot determine output_dim: no Linear layer found and "
                "input_dim not provided"
            )

        if freeze:
            self.requires_grad_(False)

        logger.info(
            "PretrainedDecoder: backbone=%s, split_from=%s, "
            "output_dim=%d, freeze=%s, layers=%s",
            backbone,
            split_from,
            self.output_dim,
            freeze,
            names,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

    def get_output_dim(self):
        return self.output_dim


__all__ = ["PretrainedDecoder", "PretrainedEncoder"]
