"""Shared scalar helpers for replacement-training modules."""

from __future__ import annotations

import json
import os
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch.amp import GradScaler

from dendritic_modeling.config.conversion import to_plain_dict

_DTYPE_ALIASES = {
    "float": torch.float32,
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}
_MSE_LOSS_NAMES = {"mse", "l2"}
_SMOOTH_L1_LOSS_NAMES = {"smooth_l1", "huber"}


def _to_plain_mapping(obj: Any) -> dict[str, Any]:
    return to_plain_dict(obj)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype:
    normalized = str(dtype).lower()
    if normalized not in _DTYPE_ALIASES:
        raise ValueError(
            f"Unknown dtype {dtype!r}; expected one of {sorted(_DTYPE_ALIASES)}"
        )
    return _DTYPE_ALIASES[normalized]


def _make_cuda_grad_scaler(use_amp: bool, device: torch.device) -> GradScaler:
    """Create the CUDA GradScaler used by replacement-training loops."""
    return GradScaler("cuda", enabled=bool(use_amp and device.type == "cuda"))


def _base_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    error_context: str,
) -> torch.Tensor:
    """Compute the base distillation loss before optional cosine regularization."""
    normalized = str(loss_name).lower()
    if normalized in _MSE_LOSS_NAMES:
        return F.mse_loss(prediction, target)
    if normalized in _SMOOTH_L1_LOSS_NAMES:
        return F.smooth_l1_loss(prediction, target)
    raise ValueError(f"{error_context} loss must be mse or smooth_l1")


def _flatten_for_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    cosine_flatten: Literal["batch", "last_dim"],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flatten prediction/target tensors according to a replacement domain."""
    if cosine_flatten == "batch":
        return prediction.reshape(prediction.shape[0], -1), target.reshape(
            target.shape[0], -1
        )
    if cosine_flatten == "last_dim":
        return prediction.reshape(-1, prediction.shape[-1]), target.reshape(
            -1, target.shape[-1]
        )
    raise ValueError("cosine_flatten must be 'batch' or 'last_dim'")


def _compute_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    cosine_weight: float = 0.0,
    cosine_flatten: Literal["batch", "last_dim"] = "last_dim",
    error_context: str = "replacement",
) -> torch.Tensor:
    """Compute the shared MSE/SmoothL1 plus optional cosine distillation loss."""
    loss = _base_distillation_loss(
        prediction,
        target,
        loss_name=loss_name,
        error_context=error_context,
    )

    if cosine_weight <= 0:
        return loss

    pred_flat, tgt_flat = _flatten_for_cosine(
        prediction,
        target,
        cosine_flatten=cosine_flatten,
    )
    cosine_loss = 1.0 - F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
    return loss + float(cosine_weight) * cosine_loss


def _compute_transformer_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    cosine_weight: float = 0.0,
) -> torch.Tensor:
    """Compute transformer replacement loss with token/hidden-dim flattening."""
    return _compute_distillation_loss(
        prediction,
        target,
        loss_name=loss_name,
        cosine_weight=cosine_weight,
        cosine_flatten="last_dim",
        error_context="transformer replacement",
    )


def _compute_vision_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    cosine_weight: float = 0.0,
) -> torch.Tensor:
    """Compute vision replacement loss with per-example flattening."""
    return _compute_distillation_loss(
        prediction,
        target,
        loss_name=loss_name,
        cosine_weight=cosine_weight,
        cosine_flatten="batch",
        error_context="vision replacement",
    )


def _infinite_loader(loader: Any):
    """Yield batches from a loader forever without caching an epoch."""
    while True:
        yield from loader


def _write_metrics_json(
    result: Any,
    *,
    output_dir: str | os.PathLike[str] | None = None,
    filename: str = "metrics.json",
) -> None:
    """Write a replacement-training result object to a metrics JSON file."""
    save_dir = output_dir if output_dir is not None else result.save_dir
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), "w", encoding="utf-8") as handle:
        json.dump(result.asdict(), handle, indent=2)


__all__ = [
    "_compute_distillation_loss",
    "_compute_transformer_distillation_loss",
    "_compute_vision_distillation_loss",
    "_infinite_loader",
    "_make_cuda_grad_scaler",
    "_resolve_device",
    "_resolve_dtype",
    "_to_plain_mapping",
    "_write_metrics_json",
]
