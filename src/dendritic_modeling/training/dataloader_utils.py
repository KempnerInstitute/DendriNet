"""Shared DataLoader tuning helpers for training entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from dendritic_modeling.training.dataloader_profiles import (
    dataset_type_name,
    is_torchvision_image_dataset,
    prefers_main_process_loading,
)
from dendritic_modeling.training.dataloader_resources import (
    auto_worker_cpu_budget,
    visible_cpu_count,
)

_IMAGE_DATASET_PREFETCH_FACTOR = 4
_CUDA_WORKER_CAP = 16
_CPU_WORKER_CAP = 8
_CPU_RESERVED_MAIN_PROCESS_WORKERS = 1


@dataclass(frozen=True)
class DataLoaderTuning:
    """Resolved DataLoader knobs after applying auto defaults and validation."""

    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    in_order: bool | None
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
        if self.in_order is not None:
            kwargs["in_order"] = self.in_order
        return kwargs

    def summary(self, dataset: Dataset) -> str:
        """Return a compact human-readable summary of resolved loader tuning."""
        dataset_name = dataset_type_name(dataset)
        return (
            f"dataset={dataset_name}, num_workers={self.num_workers}, "
            f"pin_memory={self.pin_memory}, "
            f"persistent_workers={self.persistent_workers}, "
            f"prefetch_factor={self.prefetch_factor}, "
            f"in_order={self.in_order}, "
            f"available_cpus={self.available_cpus}, "
            f"auto_num_workers={self.auto_num_workers}"
        )


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
    worker_cpu_budget = auto_worker_cpu_budget(cpus)
    use_cuda = torch.cuda.is_available() if cuda_available is None else cuda_available
    if use_cuda:
        return min(_CUDA_WORKER_CAP, worker_cpu_budget)
    return min(
        _CPU_WORKER_CAP,
        max(0, worker_cpu_budget - _CPU_RESERVED_MAIN_PROCESS_WORKERS),
    )


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
    in_order: bool | None = None,
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
    resolved_in_order = bool(in_order) if in_order is not None else None

    return DataLoaderTuning(
        num_workers=resolved_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=resolved_persistent,
        prefetch_factor=resolved_prefetch,
        in_order=resolved_in_order,
        available_cpus=available_cpus,
        auto_num_workers=auto_num_workers,
    )


def resolve_dataloader_tuning_from_config(
    dataset: Dataset,
    config: Any,
    *,
    device: torch.device | str | None = None,
    default_num_workers: int | None = None,
) -> DataLoaderTuning:
    """Resolve DataLoader tuning from a config-like object."""
    num_workers = getattr(config, "num_workers", None)
    if num_workers is None:
        num_workers = default_num_workers
    return resolve_dataloader_tuning(
        dataset,
        device=device,
        num_workers=num_workers,
        pin_memory=getattr(config, "pin_memory", None),
        persistent_workers=getattr(config, "persistent_workers", None),
        prefetch_factor=getattr(config, "prefetch_factor", None),
        in_order=getattr(config, "in_order", None),
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
