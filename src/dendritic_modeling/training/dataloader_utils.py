"""Shared DataLoader tuning helpers for training entrypoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from dendritic_modeling.training.dataloader_profiles import (
    base_dataset,
    is_torchvision_image_dataset,
    prefers_main_process_loading,
)

_LOCAL_WORLD_SIZE_ENV_VARS = (
    "LOCAL_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_SIZE",
    "MV2_COMM_WORLD_LOCAL_SIZE",
    "SLURM_NTASKS_PER_NODE",
    "SLURM_STEP_TASKS_PER_NODE",
)
_IMAGE_DATASET_PREFETCH_FACTOR = 4


@dataclass(frozen=True)
class DataLoaderTuning:
    """Resolved DataLoader knobs after applying auto defaults and validation."""

    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    available_cpus: int
    auto_num_workers: bool

    def as_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def summary(self, dataset: Dataset) -> str:
        """Return a compact human-readable summary of resolved loader tuning."""
        resolved_dataset = base_dataset(dataset)
        dataset_name = (
            f"{type(resolved_dataset).__module__}.{type(resolved_dataset).__name__}"
        )
        return (
            f"dataset={dataset_name}, num_workers={self.num_workers}, "
            f"pin_memory={self.pin_memory}, "
            f"persistent_workers={self.persistent_workers}, "
            f"prefetch_factor={self.prefetch_factor}, "
            f"available_cpus={self.available_cpus}, "
            f"auto_num_workers={self.auto_num_workers}"
        )


def visible_cpu_count() -> int:
    """Return CPUs visible to the process, respecting scheduler affinity."""
    available_cpus = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except OSError:
            pass
    return max(1, available_cpus)


def _parse_local_process_count(raw_value: str) -> int | None:
    """Parse local process counts from simple integers or SLURM node specs."""
    values = []
    for chunk in raw_value.split(","):
        digits = []
        for char in chunk.strip():
            if not char.isdigit():
                break
            digits.append(char)
        if not digits:
            continue
        value = int("".join(digits))
        if value > 0:
            values.append(value)
    return max(values) if values else None


def _local_process_count() -> int:
    """Return the number of training processes sharing this node, if known."""
    for env_name in _LOCAL_WORLD_SIZE_ENV_VARS:
        raw_value = os.environ.get(env_name)
        if raw_value is None:
            continue
        value = _parse_local_process_count(raw_value)
        if value is not None:
            return value
    return 1


def _auto_worker_cpu_budget(available_cpus: int) -> int:
    """Return the CPU budget for auto worker selection on the current process."""
    local_processes = _local_process_count()
    if local_processes <= 1:
        return max(1, available_cpus)
    return max(0, available_cpus // local_processes)


def auto_num_workers_for_dataset(
    dataset: Dataset,
    *,
    cuda_available: bool | None = None,
    available_cpus: int | None = None,
) -> int:
    """Choose a conservative worker count for a training dataset."""
    if prefers_main_process_loading(dataset):
        return 0

    cpus = visible_cpu_count() if available_cpus is None else max(1, available_cpus)
    worker_cpu_budget = _auto_worker_cpu_budget(cpus)
    use_cuda = torch.cuda.is_available() if cuda_available is None else cuda_available
    if use_cuda:
        return min(16, worker_cpu_budget)
    return min(8, max(0, worker_cpu_budget - 1))


def _resolve_num_workers(
    dataset: Dataset,
    num_workers: int | None,
    *,
    available_cpus: int,
) -> tuple[int, bool]:
    auto_num_workers = num_workers is None
    if auto_num_workers:
        return (
            auto_num_workers_for_dataset(
                dataset,
                cuda_available=torch.cuda.is_available(),
                available_cpus=available_cpus,
            ),
            True,
        )

    resolved_workers = max(0, int(num_workers))
    return min(resolved_workers, available_cpus), False


def _resolve_pin_memory(
    *,
    device: torch.device | str | None,
    pin_memory: bool | None,
    num_workers: int,
) -> bool:
    device_type = (
        torch.device(device).type
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return (
        bool(pin_memory)
        if pin_memory is not None
        else device_type == "cuda" and num_workers > 0
    )


def _resolve_persistent_workers(
    persistent_workers: bool | None,
    *,
    num_workers: int,
) -> bool:
    if persistent_workers is None:
        return num_workers > 0

    resolved_persistent = bool(persistent_workers)
    if resolved_persistent and num_workers == 0:
        raise ValueError("persistent_workers=True requires num_workers > 0")
    return resolved_persistent


def _resolve_prefetch_factor(
    prefetch_factor: int | None,
    *,
    dataset: Dataset,
    num_workers: int,
) -> int | None:
    if prefetch_factor is None:
        if num_workers > 0 and is_torchvision_image_dataset(dataset):
            return _IMAGE_DATASET_PREFETCH_FACTOR
        return None
    if num_workers == 0:
        raise ValueError("prefetch_factor requires num_workers > 0")
    return max(1, int(prefetch_factor))


def resolve_dataloader_tuning(
    dataset: Dataset,
    *,
    device: torch.device | str | None = None,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    prefetch_factor: int | None = None,
) -> DataLoaderTuning:
    """Resolve DataLoader worker/pinning/prefetch settings.

    Tensor-backed datasets default to single-process loading because worker
    startup and IPC usually dominate. Lazy image/file-backed datasets, including
    ImageNet ``ImageFolder``/``ImageNet`` subsets, keep multi-worker loading.
    """
    available_cpus = visible_cpu_count()
    resolved_workers, auto_num_workers = _resolve_num_workers(
        dataset,
        num_workers,
        available_cpus=available_cpus,
    )
    resolved_pin_memory = _resolve_pin_memory(
        device=device,
        pin_memory=pin_memory,
        num_workers=resolved_workers,
    )
    resolved_persistent = _resolve_persistent_workers(
        persistent_workers,
        num_workers=resolved_workers,
    )
    resolved_prefetch = _resolve_prefetch_factor(
        prefetch_factor,
        dataset=dataset,
        num_workers=resolved_workers,
    )

    return DataLoaderTuning(
        num_workers=resolved_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=resolved_persistent,
        prefetch_factor=resolved_prefetch,
        available_cpus=available_cpus,
        auto_num_workers=auto_num_workers,
    )


def resolve_dataloader_tuning_from_config(
    dataset: Dataset,
    config: Any,
    *,
    device: torch.device | str | None = None,
) -> DataLoaderTuning:
    """Resolve DataLoader tuning from a config-like object."""
    return resolve_dataloader_tuning(
        dataset,
        device=device,
        num_workers=getattr(config, "num_workers", None),
        pin_memory=getattr(config, "pin_memory", None),
        persistent_workers=getattr(config, "persistent_workers", None),
        prefetch_factor=getattr(config, "prefetch_factor", None),
    )


def dataloader_kwargs_from_config(
    dataset: Dataset,
    config: Any,
    *,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Return validated DataLoader keyword arguments from a config-like object."""
    return resolve_dataloader_tuning_from_config(
        dataset,
        config,
        device=device,
    ).as_kwargs()


__all__ = [
    "DataLoaderTuning",
    "auto_num_workers_for_dataset",
    "dataloader_kwargs_from_config",
    "prefers_main_process_loading",
    "resolve_dataloader_tuning",
    "resolve_dataloader_tuning_from_config",
    "visible_cpu_count",
]
