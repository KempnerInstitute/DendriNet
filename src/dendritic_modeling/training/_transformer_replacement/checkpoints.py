"""Checkpoint loading helpers for transformer replacements."""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from dendritic_modeling.networks.architectures.transformer import EIStackMLPSlot
from dendritic_modeling.training.replacement_common import ReplacementTrainingHistory

from .distributed import is_transformer_main_process, transformer_process_rank

logger = logging.getLogger(__name__)
BEST_REPLACEMENT_CHECKPOINT = "best_replacement_checkpoint.pt"
_BITMASK_SUFFIX = ".__bitmask"
_BITMASK_METADATA_SUFFIX = ".__bitmask_metadata"


def _compact_sparse_index_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Pack non-negative sparse topology indices for checkpoint storage."""
    compact: dict[str, torch.Tensor] = {}
    encoding: dict[str, str] = {}
    for key, value in state_dict.items():
        if key.endswith("._last_forward_param_tensor") or key == (
            "_last_forward_param_tensor"
        ):
            continue
        if key.endswith("connection_indices") and value.dtype == torch.int64:
            max_index = int(value.max().item()) if value.numel() else 0
            min_index = int(value.min().item()) if value.numel() else 0
            if min_index >= 0 and max_index <= 65_535:
                value = value.to(dtype=torch.uint16)
            elif min_index >= 0 and max_index <= 4_294_967_295:
                value = value.to(dtype=torch.uint32)
            encoding[key] = str(value.dtype).removeprefix("torch.")
        compact[key] = value
    return compact, encoding


def _compact_sparse_bitmask_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Encode fixed row-wise topology as bitmasks instead of per-weight indices.

    Sparse weights are reordered into ascending input-index order so the mask
    contains all information required to reconstruct their alignment. This is
    most useful for moderately dense fixed topology, where even uint16 costs
    two index bytes per learned weight while a mask costs one bit per possible
    connection.
    """
    compact = {
        key: value
        for key, value in state_dict.items()
        if not key.endswith("._last_forward_param_tensor")
        and key != "_last_forward_param_tensor"
    }
    encoding: dict[str, str] = {}
    for key, value in tuple(compact.items()):
        if not key.endswith("connection_indices"):
            continue
        if value.ndim != 2 or value.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        }:
            continue

        weight_key = f"{key[: -len('connection_indices')]}pre_w"
        weight = compact.get(weight_key)
        if weight is None or tuple(weight.shape) != tuple(value.shape):
            continue

        indices = value.to(dtype=torch.long)
        if indices.numel() and bool((indices < 0).any()):
            continue
        sorted_indices, order = indices.sort(dim=1)
        if sorted_indices.shape[1] > 1 and bool(
            (sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()
        ):
            raise ValueError(f"Topology indices must be unique within each row: {key}")

        n_rows, k = sorted_indices.shape
        n_inputs = int(sorted_indices.max().item()) + 1 if k else 0
        n_bytes = (n_inputs + 7) // 8
        packed = torch.zeros(
            (n_rows, n_bytes),
            dtype=torch.int16,
            device=sorted_indices.device,
        )
        if k:
            byte_indices = torch.div(sorted_indices, 8, rounding_mode="floor")
            bit_values = torch.bitwise_left_shift(
                torch.ones_like(sorted_indices, dtype=torch.int16),
                torch.remainder(sorted_indices, 8),
            ).to(dtype=torch.int16)
            packed.scatter_add_(1, byte_indices, bit_values)

        compact[weight_key] = weight.gather(1, order.to(weight.device))
        del compact[key]
        compact[f"{key}{_BITMASK_SUFFIX}"] = packed.to(dtype=torch.uint8)
        compact[f"{key}{_BITMASK_METADATA_SUFFIX}"] = torch.tensor(
            [n_rows, k, n_inputs],
            dtype=torch.int64,
            device=indices.device,
        )
        encoding[key] = "bitmask"
    return compact, encoding


def _decode_sparse_bitmask_state_dict(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Restore row-wise indices from `_compact_sparse_bitmask_state_dict`."""
    decoded: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.endswith(_BITMASK_METADATA_SUFFIX):
            continue
        if not key.endswith(_BITMASK_SUFFIX):
            decoded[key] = value
            continue

        original_key = key.removesuffix(_BITMASK_SUFFIX)
        metadata_key = f"{original_key}{_BITMASK_METADATA_SUFFIX}"
        if metadata_key not in state_dict:
            raise RuntimeError(f"Missing bitmask metadata for {original_key}")
        metadata = state_dict[metadata_key].to(dtype=torch.long).reshape(-1)
        if metadata.numel() != 3:
            raise RuntimeError(f"Invalid bitmask metadata for {original_key}")
        n_rows, k, n_inputs = (int(item) for item in metadata.tolist())
        if tuple(value.shape) != (n_rows, (n_inputs + 7) // 8):
            raise RuntimeError(f"Invalid packed bitmask shape for {original_key}")

        bit_offsets = torch.arange(8, device=value.device, dtype=torch.uint8)
        present = torch.bitwise_and(
            torch.bitwise_right_shift(value.unsqueeze(-1), bit_offsets),
            1,
        ).reshape(n_rows, -1)[:, :n_inputs]
        counts = present.sum(dim=1)
        if bool((counts != k).any()):
            raise RuntimeError(
                f"Packed bitmask row counts do not match K for {original_key}"
            )
        coordinates = torch.nonzero(present, as_tuple=False)
        decoded[original_key] = coordinates[:, 1].reshape(n_rows, k).to(torch.long)
    return decoded


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
                remapped[f"{new_prefix}{key[len(old_prefix) :]}"] = value
                break

    return remapped or None


def _load_replacement_state_dict(
    replacement: nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    """Load a replacement checkpoint, including layerwise EI-stack slots."""
    # Older identity-transform checkpoints could register this non-persistent
    # forward cache as a duplicate alias of ``pre_w``. It is never model state.
    state_dict = _decode_sparse_bitmask_state_dict(
        {
            key: value
            for key, value in state_dict.items()
            if not key.endswith("._last_forward_param_tensor")
            and key != "_last_forward_param_tensor"
        }
    )
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
            f"EI-stack checkpoint keys do not match the replacement slot: {missing[:5]}"
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


def _worker_checkpoint_path(path: str, rank: int) -> str:
    root, extension = os.path.splitext(path)
    return f"{root}.rank{int(rank)}{extension or '.pt'}"


def _atomic_torch_save(payload: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _save_layerwise_training_checkpoint(
    path: str,
    units: Sequence[Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    history: ReplacementTrainingHistory,
    *,
    step: int,
    device: torch.device,
) -> None:
    """Save common model state plus rank-local RNG state at one DDP barrier."""
    rank = transformer_process_rank()
    worker_payload: dict[str, Any] = {
        "schema_version": 1,
        "rank": rank,
        "step": int(step),
        "torch_rng_state": torch.get_rng_state(),
    }
    if device.type == "cuda":
        worker_payload["cuda_rng_state"] = torch.cuda.get_rng_state(device)
    _atomic_torch_save(worker_payload, _worker_checkpoint_path(path, rank))

    if is_transformer_main_process():
        _atomic_torch_save(
            {
                "schema_version": 1,
                "step": int(step),
                "replacement_state_dicts": [
                    _compact_sparse_index_state_dict(unit.replacement.state_dict())[0]
                    for unit in units
                ],
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "history": dataclasses.asdict(history),
            },
            path,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _load_layerwise_training_checkpoint(
    path: str,
    units: Sequence[Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    *,
    device: torch.device,
) -> tuple[int, ReplacementTrainingHistory]:
    """Restore a checkpoint produced by `_save_layerwise_training_checkpoint`."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location=device, weights_only=False)
    states = payload.get("replacement_state_dicts", [])
    if len(states) != len(units):
        raise RuntimeError(
            "Replacement checkpoint unit count does not match the current config"
        )
    for unit, state_dict in zip(units, states):
        _load_replacement_state_dict(unit.replacement, state_dict)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler.load_state_dict(payload.get("scaler_state_dict", {}))
    history = ReplacementTrainingHistory(**payload["history"])

    worker_path = _worker_checkpoint_path(path, transformer_process_rank())
    if os.path.isfile(worker_path):
        worker = torch.load(worker_path, map_location="cpu", weights_only=False)
        torch.set_rng_state(worker["torch_rng_state"])
        if device.type == "cuda" and "cuda_rng_state" in worker:
            torch.cuda.set_rng_state(worker["cuda_rng_state"], device)
    else:
        logger.warning("Rank-local checkpoint not found: %s", worker_path)
    return int(payload["step"]), history


def _save_best_replacement_checkpoint(
    save_dir: str,
    units: Sequence[Any],
    *,
    step: int,
    valid_loss: float,
) -> str:
    """Persist a validation-best layerwise state, including the initial state."""
    path = os.path.join(save_dir, BEST_REPLACEMENT_CHECKPOINT)
    if is_transformer_main_process():
        _atomic_torch_save(
            {
                "schema_version": 1,
                "step": int(step),
                "valid_loss": float(valid_loss),
                "replacement_state_dicts": [
                    _compact_sparse_index_state_dict(unit.replacement.state_dict())[0]
                    for unit in units
                ],
            },
            path,
        )
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    return path


def _restore_best_replacement_checkpoint(
    save_dir: str,
    units: Sequence[Any],
    *,
    device: torch.device,
) -> tuple[int, float] | None:
    """Restore the exported layerwise modules to their best validation state."""
    path = os.path.join(save_dir, BEST_REPLACEMENT_CHECKPOINT)
    if not os.path.isfile(path):
        logger.warning("Best replacement checkpoint not found: %s", path)
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    states = payload.get("replacement_state_dicts", [])
    if len(states) != len(units):
        raise RuntimeError("Best replacement checkpoint unit count mismatch")
    for unit, state_dict in zip(units, states):
        _load_replacement_state_dict(unit.replacement, state_dict)
    return int(payload["step"]), float(payload["valid_loss"])


__all__ = [
    "_compact_sparse_bitmask_state_dict",
    "_compact_sparse_index_state_dict",
    "_copy_replacement_state_dict",
    "_decode_sparse_bitmask_state_dict",
    "_load_layerwise_training_checkpoint",
    "_load_replacement_record_checkpoints",
    "_load_replacement_state_dict",
    "_remap_single_layer_stack_state_dict",
    "_restore_best_replacement_checkpoint",
    "_save_best_replacement_checkpoint",
    "_save_layerwise_training_checkpoint",
]
