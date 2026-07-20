"""Small torchrun/DDP runtime for transformer replacement experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel


@dataclass
class TransformerDistributedContext:
    """Resolved process topology for one transformer replacement worker."""

    mode: str = "none"
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    device: torch.device | None = None
    initialized_here: bool = False

    @property
    def enabled(self) -> bool:
        return self.mode == "ddp"

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    def close(self) -> None:
        if self.initialized_here and dist.is_initialized():
            dist.destroy_process_group()
            self.initialized_here = False


def initialize_transformer_distributed(
    train_cfg: Any,
) -> TransformerDistributedContext:
    """Initialize a process group when ``distributed_mode: ddp`` is requested."""
    mode = str(getattr(train_cfg, "distributed_mode", "none") or "none").lower()
    if mode in {"", "none", "off", "false"}:
        return TransformerDistributedContext()
    if mode != "ddp":
        raise ValueError(
            "transformer replacement distributed_mode must be 'none' or 'ddp'; "
            "full-model FSDP is a separate scale gate"
        )
    required = [
        name for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE") if name not in os.environ
    ]
    if required:
        raise RuntimeError(
            "distributed_mode='ddp' requires torchrun environment variables; "
            f"missing {', '.join(required)}"
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size < 1 or rank < 0 or rank >= world_size:
        raise RuntimeError(f"Invalid torchrun rank/world size: {rank}/{world_size}")

    requested_device = str(getattr(train_cfg, "device", "auto") or "auto").lower()
    force_cpu = requested_device == "cpu"
    if torch.cuda.is_available() and not force_cpu:
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise RuntimeError(f"LOCAL_RANK={local_rank} is not a visible CUDA device")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        default_backend = "nccl"
    else:
        device = torch.device("cpu")
        default_backend = "gloo"

    initialized_here = False
    if not dist.is_initialized():
        backend = str(getattr(train_cfg, "distributed_backend", "") or default_backend)
        dist.init_process_group(backend=backend, init_method="env://")
        initialized_here = True
    actual_rank = dist.get_rank()
    actual_world_size = dist.get_world_size()
    if actual_rank != rank or actual_world_size != world_size:
        if initialized_here:
            dist.destroy_process_group()
        raise RuntimeError(
            "Initialized process-group topology does not match torchrun environment"
        )
    return TransformerDistributedContext(
        mode="ddp",
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
        initialized_here=initialized_here,
    )


def wrap_transformer_ddp(
    module: nn.Module,
    train_cfg: Any,
    device: torch.device,
) -> nn.Module:
    """Wrap a trainable module in DDP when configured."""
    mode = str(getattr(train_cfg, "distributed_mode", "none") or "none").lower()
    if mode != "ddp":
        return module
    if not dist.is_initialized():
        raise RuntimeError("DDP wrapping requested before process-group initialization")
    kwargs: dict[str, Any] = {
        "broadcast_buffers": False,
        "find_unused_parameters": bool(
            getattr(train_cfg, "ddp_find_unused_parameters", False)
        ),
    }
    if device.type == "cuda":
        kwargs.update(device_ids=[device.index], output_device=device.index)
    return DistributedDataParallel(module, **kwargs)


def transformer_process_rank() -> int:
    """Return the active distributed rank without requiring initialization."""
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return int(os.environ.get("RANK", 0))


def is_transformer_main_process() -> bool:
    return transformer_process_rank() == 0


def transformer_distributed_mean(value: float, device: torch.device) -> float:
    """Average a scalar across the active process group for rank-0 reporting."""
    if not dist.is_available() or not dist.is_initialized():
        return float(value)
    scalar = torch.tensor(float(value), device=device, dtype=torch.float64)
    dist.all_reduce(scalar, op=dist.ReduceOp.SUM)
    scalar /= dist.get_world_size()
    return float(scalar.item())


__all__ = [
    "TransformerDistributedContext",
    "initialize_transformer_distributed",
    "is_transformer_main_process",
    "transformer_distributed_mean",
    "transformer_process_rank",
    "wrap_transformer_ddp",
]
