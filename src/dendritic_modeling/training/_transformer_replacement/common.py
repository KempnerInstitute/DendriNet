"""Shared types and scalar helpers for transformer replacement training."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.training.dataloader_utils import (
    resolve_dataloader_tuning_from_config,
)
from dendritic_modeling.training.replacement_common import (
    _compute_distillation_loss as _shared_compute_distillation_loss,
    _resolve_device,
    _resolve_dtype,
    _to_plain_mapping,
)

logger = logging.getLogger(__name__)


def _activation(name: str, x: torch.Tensor) -> torch.Tensor:
    normalized = str(name).lower()
    if normalized == "silu":
        return F.silu(x)
    if normalized == "gelu":
        return F.gelu(x)
    if normalized == "relu":
        return F.relu(x)
    if normalized in {"identity", "linear", "none"}:
        return x
    raise ValueError(f"Unsupported synthetic teacher activation {name!r}")


class SyntheticTransformerMLP(nn.Module):
    """Small DeepSeek/Llama-style teacher MLP for smoke distillation runs."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        *,
        kind: str = "gated_mlp",
        activation: str = "silu",
        bias: bool = False,
    ):
        super().__init__()
        self.kind = str(kind).lower()
        self.activation = activation
        if self.kind == "linear":
            self.proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        elif self.kind == "mlp":
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        elif self.kind == "gated_mlp":
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        else:
            raise ValueError("synthetic_teacher.kind must be linear, mlp, or gated_mlp")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "linear":
            return self.proj(x)
        if self.kind == "mlp":
            return self.down_proj(_activation(self.activation, self.up_proj(x)))
        return self.down_proj(
            _activation(self.activation, self.gate_proj(x)) * self.up_proj(x)
        )


@dataclass
class DistillationUnit:
    """One teacher/replacement pair."""

    layer_index: int
    replacement: nn.Module
    teacher_mlp: nn.Module | None = None


def _replacement_forward_modules(
    units: Sequence[DistillationUnit],
    train_cfg: Any,
    device: torch.device,
) -> list[nn.Module]:
    """Return modules used for forward passes, optionally DataParallel-wrapped."""
    replacements = [unit.replacement for unit in units]
    if not bool(getattr(train_cfg, "data_parallel", False)):
        return replacements
    if device.type != "cuda":
        logger.warning("data_parallel=True ignored because device is %s", device)
        return replacements

    available = torch.cuda.device_count()
    requested = [int(idx) for idx in getattr(train_cfg, "data_parallel_devices", [])]
    device_ids = requested if requested else list(range(available))
    device_ids = [idx for idx in device_ids if 0 <= idx < available]
    if len(device_ids) < 2:
        logger.warning(
            "data_parallel=True requested, but only %s CUDA device(s) are usable",
            available,
        )
        return replacements

    primary = torch.device(f"cuda:{device_ids[0]}")
    for module in replacements:
        module.to(primary)
    logger.info(
        "Using DataParallel for replacement modules on CUDA devices %s", device_ids
    )
    return [
        nn.DataParallel(module, device_ids=device_ids, output_device=device_ids[0])
        for module in replacements
    ]


@dataclass
class TransformerReplacementTrainingResult:
    """Summary returned by transformer replacement distillation."""

    train_losses: list[float]
    valid_losses: list[float]
    initial_valid_loss: float
    final_valid_loss: float
    best_valid_loss: float
    best_step: int
    save_dir: str
    layer_indices: list[int]

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransformerReplacementBenchmarkResult:
    """Post-training comparison for a saved transformer replacement."""

    teacher_source: str
    layer_indices: list[int]
    checkpoint_dir: str
    initial: dict[str, float]
    trained: dict[str, float]
    teacher_dense_params: int | None
    dendritic_stored_params: int
    dendritic_active_params: int
    stored_reduction_factor: float | None
    active_reduction_factor: float | None
    initial_label: str = "initial"
    trained_label: str = "trained_replacement"

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _compute_distillation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_name: str,
    cosine_weight: float = 0.0,
) -> torch.Tensor:
    return _shared_compute_distillation_loss(
        prediction,
        target,
        loss_name=loss_name,
        cosine_weight=cosine_weight,
        cosine_flatten="last_dim",
        error_context="transformer replacement",
    )


def _make_tensor_loader(
    dataset: TensorDataset,
    *,
    batch_size: int,
    shuffle: bool,
    train_cfg: Any | None = None,
    device: torch.device | str | None = None,
) -> DataLoader:
    """Build the tensor-cache loader used by transformer replacement paths."""
    loader_kwargs = (
        resolve_dataloader_tuning_from_config(
            dataset,
            train_cfg,
            device=device,
        ).as_kwargs()
        if train_cfg is not None
        else {}
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        **loader_kwargs,
    )


__all__ = [
    "DistillationUnit",
    "SyntheticTransformerMLP",
    "TransformerReplacementBenchmarkResult",
    "TransformerReplacementTrainingResult",
    "_activation",
    "_compute_distillation_loss",
    "_make_tensor_loader",
    "_replacement_forward_modules",
    "_resolve_device",
    "_resolve_dtype",
    "_to_plain_mapping",
]
