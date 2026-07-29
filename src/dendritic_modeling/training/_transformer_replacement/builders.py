"""Replacement builders for transformer distillation units."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.config import Config
from dendritic_modeling.networks.architectures.transformer import (
    DendriticFFNReplacement,
    EIStackMLPSlot,
    GatedDendriticFFNReplacement,
    LayerwiseEIStackReplacement,
    build_dendritic_ffn_kwargs_from_core_config,
    build_ei_stack_kwargs_from_core_config,
)
from dendritic_modeling.training._transformer_replacement.common import (
    DistillationUnit,
    SyntheticTransformerMLP,
    _to_plain_mapping,
    resolve_configured_replacement_layers,
)


def _replacement_kind(config: Config) -> str:
    replacement_kwargs = _to_plain_mapping(
        config.model.transformer_replacement.replacement_kwargs
    )
    return str(
        replacement_kwargs.get("kind", replacement_kwargs.get("type", "dendritic_ffn"))
    ).lower()


def _make_single_layer_ei_stack_replacement(
    *,
    hidden_size: int,
    config: Config,
    dtype: torch.dtype,
    device: torch.device,
) -> EIStackMLPSlot:
    kwargs = build_ei_stack_kwargs_from_core_config(
        config.model.core,
        transformer_replacement=config.model.transformer_replacement,
        num_layers=1,
    )
    stack = LayerwiseEIStackReplacement(hidden_size=hidden_size, **kwargs)
    stack = stack.to(device=device, dtype=dtype)
    return EIStackMLPSlot(stack, 0)


def _make_replacement(
    *,
    hidden_size: int,
    config: Config,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    kind = _replacement_kind(config)
    if kind in {"dense_mlp_control", "dense_control", "dense_mlp"}:
        raise ValueError(
            "dense_mlp_control is a synthetic training baseline; construct it "
            "through _make_synthetic_units"
        )
    if kind in {"ei_stack", "layerwise_ei_stack", "einet_stack"}:
        return _make_single_layer_ei_stack_replacement(
            hidden_size=hidden_size,
            config=config,
            dtype=dtype,
            device=device,
        )

    kwargs = build_dendritic_ffn_kwargs_from_core_config(
        config.model.core,
        transformer_replacement=config.model.transformer_replacement,
    )
    replacement_cls = (
        GatedDendriticFFNReplacement
        if kind in {"gated_dendritic_ffn", "gated_dendritic", "dendritic_glu"}
        else DendriticFFNReplacement
    )
    replacement = replacement_cls(hidden_size=hidden_size, **kwargs)
    return replacement.to(device=device, dtype=dtype)


def _make_synthetic_units(
    config: Config,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> list[DistillationUnit]:
    train_cfg = config.training.transformer_replacement
    teacher_cfg = _to_plain_mapping(train_cfg.synthetic_teacher)
    hidden_size = int(train_cfg.hidden_size or 32)
    intermediate_size = int(train_cfg.intermediate_size or hidden_size * 4)
    layer_indices = resolve_configured_replacement_layers(config, default=[0])

    units: list[DistillationUnit] = []
    for layer_index in layer_indices:
        teacher = SyntheticTransformerMLP(
            hidden_size,
            intermediate_size,
            kind=str(teacher_cfg.get("kind", "gated_mlp")),
            activation=str(teacher_cfg.get("activation", "silu")),
            bias=bool(teacher_cfg.get("bias", False)),
        ).to(device=device, dtype=dtype)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad_(False)

        if _replacement_kind(config) in {
            "dense_mlp_control",
            "dense_control",
            "dense_mlp",
        }:
            replacement = SyntheticTransformerMLP(
                hidden_size,
                intermediate_size,
                kind=str(teacher_cfg.get("kind", "gated_mlp")),
                activation=str(teacher_cfg.get("activation", "silu")),
                bias=bool(teacher_cfg.get("bias", False)),
            ).to(device=device, dtype=dtype)
        else:
            replacement = _make_replacement(
                hidden_size=hidden_size,
                config=config,
                dtype=dtype,
                device=device,
            )
            if (
                isinstance(replacement, GatedDendriticFFNReplacement)
                and replacement.teacher_topk_init
            ):
                replacement.initialize_from_mlp(teacher)
        units.append(
            DistillationUnit(
                layer_index=int(layer_index),
                teacher_mlp=teacher,
                replacement=replacement,
            )
        )
    return units


def _records_to_units(records: Sequence[Any]) -> list[DistillationUnit]:
    return [
        DistillationUnit(
            layer_index=int(record.layer_index),
            replacement=record.replacement,
            teacher_mlp=None,
        )
        for record in records
    ]
