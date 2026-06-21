"""Shared scalar helpers for replacement-training modules."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn.functional as F

from dendritic_modeling.config.conversion import to_plain_dict


def _to_plain_mapping(obj: Any) -> dict[str, Any]:
    return to_plain_dict(obj)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _resolve_dtype(dtype: str) -> torch.dtype:
    aliases = {
        "float": torch.float32,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    normalized = str(dtype).lower()
    if normalized not in aliases:
        raise ValueError(f"Unknown dtype {dtype!r}; expected one of {sorted(aliases)}")
    return aliases[normalized]


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
    normalized = str(loss_name).lower()
    if normalized in {"mse", "l2"}:
        loss = F.mse_loss(prediction, target)
    elif normalized in {"smooth_l1", "huber"}:
        loss = F.smooth_l1_loss(prediction, target)
    else:
        raise ValueError(f"{error_context} loss must be mse or smooth_l1")

    if cosine_weight <= 0:
        return loss

    if cosine_flatten == "batch":
        pred_flat = prediction.reshape(prediction.shape[0], -1)
        tgt_flat = target.reshape(target.shape[0], -1)
    elif cosine_flatten == "last_dim":
        pred_flat = prediction.reshape(-1, prediction.shape[-1])
        tgt_flat = target.reshape(-1, target.shape[-1])
    else:
        raise ValueError("cosine_flatten must be 'batch' or 'last_dim'")

    cosine_loss = 1.0 - F.cosine_similarity(pred_flat, tgt_flat, dim=-1).mean()
    return loss + float(cosine_weight) * cosine_loss


__all__ = [
    "_compute_distillation_loss",
    "_resolve_device",
    "_resolve_dtype",
    "_to_plain_mapping",
]
