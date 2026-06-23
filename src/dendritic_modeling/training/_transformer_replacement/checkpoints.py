"""Checkpoint loading helpers for transformer replacements."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.transformer import EIStackMLPSlot


def _remap_single_layer_stack_state_dict(
    replacement: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor] | None:
    """Map a single-layer EI-stack checkpoint into one slot of a shared stack."""
    if not isinstance(replacement, EIStackMLPSlot):
        return None

    slot_index = int(replacement.layer_index)
    remapped: dict[str, torch.Tensor] = {}
    prefix_map = {
        "stack.layers.0.": f"stack.layers.{slot_index}.",
        "stack.output_projections.0.": f"stack.output_projections.{slot_index}.",
        "stack.pre_norm.0.": f"stack.pre_norm.{slot_index}.",
    }
    for key, value in state_dict.items():
        for old_prefix, new_prefix in prefix_map.items():
            if key.startswith(old_prefix):
                remapped[f"{new_prefix}{key[len(old_prefix):]}"] = value
                break

    return remapped or None


def _load_replacement_state_dict(
    replacement: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load a replacement checkpoint, including layerwise EI-stack slots."""
    remapped = _remap_single_layer_stack_state_dict(replacement, state_dict)
    if remapped is not None and set(state_dict) != set(replacement.state_dict()):
        _copy_replacement_state_dict(replacement, remapped)
        return

    try:
        replacement.load_state_dict(state_dict)
        return
    except RuntimeError as exc:
        if remapped is None:
            raise exc

    _copy_replacement_state_dict(replacement, remapped)


def _copy_replacement_state_dict(
    replacement: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    current_state = replacement.state_dict()
    missing = [key for key in state_dict if key not in current_state]
    if missing:
        raise RuntimeError(
            "EI-stack checkpoint keys do not match the replacement slot: "
            f"{missing[:5]}"
        )
    with torch.no_grad():
        for key, value in state_dict.items():
            target = current_state[key]
            target.copy_(value.to(device=target.device, dtype=target.dtype))


def _load_replacement_record_checkpoints(
    records: Sequence[Any],
    checkpoint_dir: str,
    *,
    device: torch.device,
) -> None:
    if not checkpoint_dir:
        return
    for record in records:
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"layer_{int(record.layer_index)}_replacement.pt",
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)
        payload = torch.load(checkpoint_path, map_location=device)
        state_dict = payload.get("state_dict", payload)
        _load_replacement_state_dict(record.replacement, state_dict)
