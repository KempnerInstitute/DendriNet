"""Patching utilities for replacing transformer MLP modules."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from dendritic_modeling.networks.architectures.transformer.config_translation import (
    build_dendritic_ffn_kwargs_from_core_config,
    build_ei_stack_kwargs_from_core_config,
)
from dendritic_modeling.networks.architectures.transformer.modules import (
    DendriticFFNReplacement,
    EIStackMLPSlot,
    GatedDendriticFFNReplacement,
    LayerwiseEIStackReplacement,
)
from dendritic_modeling.networks.architectures.transformer.utils import (
    _first_parameter,
    _get_attr_path,
    _infer_mlp_dims,
    _set_attr_path,
    _to_plain_mapping,
)

logger = logging.getLogger(__name__)


def resolve_transformer_layers(
    model: nn.Module,
    *,
    layers_attr: str | None = None,
) -> nn.ModuleList | list[nn.Module]:
    """Resolve a decoder layer list from common Hugging Face model layouts.

    DeepSeek/Llama-style causal language models usually expose decoder layers
    as ``model.model.layers``.  Passing ``layers_attr`` makes the lookup
    explicit for custom wrappers.
    """
    if layers_attr is not None:
        layers = _get_attr_path(model, layers_attr)
        if not isinstance(layers, (nn.ModuleList, list)):
            raise TypeError(f"{layers_attr!r} does not resolve to a layer list")
        return layers

    candidates = (
        "model.layers",
        "model.model.layers",
        "layers",
        "transformer.h",
        "gpt_neox.layers",
    )
    for path in candidates:
        try:
            layers = _get_attr_path(model, path)
        except AttributeError:
            continue
        if isinstance(layers, (nn.ModuleList, list)):
            return layers
    raise ValueError(
        "Could not find transformer decoder layers. Pass layers_attr explicitly."
    )


@dataclass
class ReplacementRecord:
    """Metadata for one patched transformer target module."""

    layer_index: int
    mlp_attr: str
    original_mlp: nn.Module
    replacement: nn.Module


def _resolve_layer_target(
    layers: nn.ModuleList | list[nn.Module],
    layer_index: int,
    mlp_attr: str,
) -> tuple[nn.Module, nn.Module]:
    """Return a transformer layer and its target module with consistent errors."""
    if layer_index < 0 or layer_index >= len(layers):
        raise IndexError(
            f"layer index {layer_index} out of range for {len(layers)} layers"
        )
    layer = layers[layer_index]
    try:
        target = _get_attr_path(layer, mlp_attr)
    except AttributeError as exc:
        raise AttributeError(
            f"layer {layer_index} has no target module path {mlp_attr!r}"
        ) from exc
    if not isinstance(target, nn.Module):
        raise TypeError(f"layer {layer_index} target {mlp_attr!r} is not an nn.Module")
    return layer, target


def _move_replacement_like_source(
    replacement: nn.Module,
    source: nn.Module,
) -> nn.Module:
    """Move replacement to the first source parameter's device and floating dtype."""
    param = _first_parameter(source)
    if param is None:
        return replacement
    dtype = param.dtype if param.dtype.is_floating_point else None
    return replacement.to(device=param.device, dtype=dtype)


def _validate_layer_indices(
    layer_indices: Sequence[int],
) -> list[int]:
    """Return ordered unique layer indices or fail before mutating a model."""
    ordered = [int(index) for index in layer_indices]
    if not ordered:
        raise ValueError("layer_indices must contain at least one layer")
    if len(ordered) != len(set(ordered)):
        raise ValueError("layer_indices must not contain duplicates")
    return ordered


def replace_transformer_mlp_layers(
    model: nn.Module,
    layer_indices: Sequence[int],
    *,
    replacement_kwargs: dict[str, object] | None = None,
    layers_attr: str | None = None,
    mlp_attr: str = "mlp",
    preserve_device_dtype: bool = True,
    replacement_class: type[nn.Module] = DendriticFFNReplacement,
) -> list[ReplacementRecord]:
    """Replace selected transformer layer target modules with dendritic FFNs.

    This is intentionally model-object based rather than config-file based so
    it can work with Hugging Face DeepSeek/Llama-like modules after loading.
    ``mlp_attr`` may be a nested path relative to each transformer layer.
    """
    replacement_kwargs = dict(replacement_kwargs or {})
    layers = resolve_transformer_layers(model, layers_attr=layers_attr)
    ordered_indices = _validate_layer_indices(layer_indices)

    # Resolve and construct every replacement first. If any layer path, shape,
    # or constructor is invalid, the source model remains completely untouched.
    targets = [
        (*_resolve_layer_target(layers, layer_index, mlp_attr), layer_index)
        for layer_index in ordered_indices
    ]
    prepared: list[tuple[nn.Module, nn.Module, nn.Module, int]] = []
    for layer, original_mlp, layer_index in targets:
        from_mlp = getattr(replacement_class, "from_mlp", None)
        if not callable(from_mlp):
            raise TypeError("replacement_class must provide a callable from_mlp")
        replacement = from_mlp(original_mlp, **replacement_kwargs)

        if preserve_device_dtype:
            replacement = _move_replacement_like_source(replacement, original_mlp)
        prepared.append((layer, original_mlp, replacement, layer_index))

    records: list[ReplacementRecord] = []
    for layer, original_mlp, replacement, layer_index in prepared:
        _set_attr_path(layer, mlp_attr, replacement)
        records.append(
            ReplacementRecord(
                layer_index=int(layer_index),
                mlp_attr=mlp_attr,
                original_mlp=original_mlp,
                replacement=replacement,
            )
        )

    return records


def replace_transformer_mlp_layers_with_ei_stack(
    model: nn.Module,
    layer_indices: Sequence[int],
    *,
    replacement_kwargs: dict[str, object] | None = None,
    layers_attr: str | None = None,
    mlp_attr: str = "mlp",
    preserve_device_dtype: bool = True,
) -> list[ReplacementRecord]:
    """Replace selected target modules with slots from one shared EI stack."""
    replacement_kwargs = dict(replacement_kwargs or {})
    layers = resolve_transformer_layers(model, layers_attr=layers_attr)
    ordered_indices = _validate_layer_indices(layer_indices)
    for layer_index in ordered_indices:
        _resolve_layer_target(layers, layer_index, mlp_attr)

    _, first_mlp = _resolve_layer_target(layers, ordered_indices[0], mlp_attr)
    hidden_size, _ = _infer_mlp_dims(first_mlp)
    stack_num_layers = int(replacement_kwargs.pop("num_layers", len(ordered_indices)))
    if stack_num_layers != len(ordered_indices):
        raise ValueError(
            "EI stack num_layers must match the number of replaced transformer "
            f"layers ({stack_num_layers} != {len(ordered_indices)})"
        )
    stack = LayerwiseEIStackReplacement(
        hidden_size=hidden_size,
        num_layers=stack_num_layers,
        **replacement_kwargs,
    )
    if preserve_device_dtype:
        stack = _move_replacement_like_source(stack, first_mlp)

    records: list[ReplacementRecord] = []
    for stack_pos, layer_index in enumerate(ordered_indices):
        layer, original_mlp = _resolve_layer_target(layers, layer_index, mlp_attr)
        replacement = EIStackMLPSlot(stack, stack_pos)
        _set_attr_path(layer, mlp_attr, replacement)
        records.append(
            ReplacementRecord(
                layer_index=int(layer_index),
                mlp_attr=mlp_attr,
                original_mlp=original_mlp,
                replacement=replacement,
            )
        )
    return records


def apply_transformer_replacement_config(
    model: nn.Module,
    transformer_replacement: Any,
    core_config: Any,
    *,
    replacement_kwargs: dict[str, Any] | None = None,
) -> list[ReplacementRecord]:
    """Apply a YAML-style transformer replacement config to a loaded model."""
    tr = _to_plain_mapping(transformer_replacement)
    if not bool(tr.get("enabled", True)):
        return []

    if "layer_indices" in tr and "layers" not in tr:
        logger.warning(
            "transformer_replacement.layer_indices is deprecated; use "
            "transformer_replacement.layers instead."
        )
    if "mlp_attr" in tr and "target_module" not in tr:
        logger.warning(
            "transformer_replacement.mlp_attr is deprecated; use "
            "transformer_replacement.target_module instead."
        )
    layer_indices = tr.get("layers", tr.get("layer_indices", []))
    if not layer_indices:
        raise ValueError(
            "transformer_replacement.layers must contain at least one layer"
        )

    replacement_overrides = _to_plain_mapping(tr.get("replacement_kwargs", {}))
    replacement_kind = str(
        replacement_overrides.get(
            "kind", replacement_overrides.get("type", "dendritic_ffn")
        )
    ).lower()
    if replacement_kind in {"ei_stack", "layerwise_ei_stack", "einet_stack"}:
        kwargs = build_ei_stack_kwargs_from_core_config(
            core_config,
            transformer_replacement=tr,
            num_layers=len(layer_indices),
            overrides=replacement_kwargs,
        )
        return replace_transformer_mlp_layers_with_ei_stack(
            model,
            [int(idx) for idx in layer_indices],
            replacement_kwargs=kwargs,
            layers_attr=tr.get("layers_attr"),
            mlp_attr=tr.get("target_module", tr.get("mlp_attr", "mlp")),
            preserve_device_dtype=bool(tr.get("preserve_device_dtype", True)),
        )

    kwargs = build_dendritic_ffn_kwargs_from_core_config(
        core_config,
        transformer_replacement=tr,
        overrides=replacement_kwargs,
    )
    return replace_transformer_mlp_layers(
        model,
        [int(idx) for idx in layer_indices],
        replacement_kwargs=kwargs,
        layers_attr=tr.get("layers_attr"),
        mlp_attr=tr.get("target_module", tr.get("mlp_attr", "mlp")),
        preserve_device_dtype=bool(tr.get("preserve_device_dtype", True)),
        replacement_class=(
            GatedDendriticFFNReplacement
            if replacement_kind
            in {"gated_dendritic_ffn", "gated_dendritic", "dendritic_glu"}
            else DendriticFFNReplacement
        ),
    )


def load_transformer_with_dendritic_replacements(
    transformer_replacement: Any,
    core_config: Any,
) -> tuple[nn.Module, list[ReplacementRecord]]:
    """Load a Hugging Face causal LM and apply dendritic MLP replacements."""
    tr = _to_plain_mapping(transformer_replacement)
    if "backbone" in tr and "model_name" not in tr:
        logger.warning(
            "transformer_replacement.backbone is deprecated; use "
            "transformer_replacement.model_name instead."
        )
    model_name = tr.get("model_name", tr.get("backbone", ""))
    if not model_name:
        raise ValueError("transformer_replacement.model_name is required")

    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise ImportError(
            "transformers is required to load transformer_replacement models"
        ) from exc

    model_kwargs = _to_plain_mapping(tr.get("model_kwargs", {}))
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    records = apply_transformer_replacement_config(model, tr, core_config)
    return model, records
