"""Shared DataLoader tuning helpers for training entrypoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset, Subset, TensorDataset


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


def visible_cpu_count() -> int:
    """Return CPUs visible to the process, respecting scheduler affinity."""
    available_cpus = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            available_cpus = len(os.sched_getaffinity(0))
        except OSError:
            pass
    return max(1, available_cpus)


def _base_dataset(dataset: Dataset) -> Dataset:
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


def prefers_main_process_loading(dataset: Dataset) -> bool:
    """Return True for in-memory tensor datasets where workers add overhead."""
    return isinstance(_base_dataset(dataset), TensorDataset)


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
    use_cuda = torch.cuda.is_available() if cuda_available is None else cuda_available
    if use_cuda:
        return min(16, cpus)
    return min(8, max(0, cpus - 1))


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
    auto_num_workers = num_workers is None
    if auto_num_workers:
        resolved_workers = auto_num_workers_for_dataset(
            dataset,
            cuda_available=torch.cuda.is_available(),
            available_cpus=available_cpus,
        )
    else:
        resolved_workers = max(0, int(num_workers))
        resolved_workers = min(resolved_workers, available_cpus)

    device_type = (
        torch.device(device).type
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    resolved_pin_memory = (
        bool(pin_memory)
        if pin_memory is not None
        else device_type == "cuda" and resolved_workers > 0
    )

    if persistent_workers is None:
        resolved_persistent = resolved_workers > 0
    else:
        resolved_persistent = bool(persistent_workers)
        if resolved_persistent and resolved_workers == 0:
            raise ValueError("persistent_workers=True requires num_workers > 0")

    resolved_prefetch = None
    if prefetch_factor is not None:
        if resolved_workers == 0:
            raise ValueError("prefetch_factor requires num_workers > 0")
        resolved_prefetch = max(1, int(prefetch_factor))

    return DataLoaderTuning(
        num_workers=resolved_workers,
        pin_memory=resolved_pin_memory,
        persistent_workers=resolved_persistent,
        prefetch_factor=resolved_prefetch,
        available_cpus=available_cpus,
        auto_num_workers=auto_num_workers,
    )


__all__ = [
    "DataLoaderTuning",
    "auto_num_workers_for_dataset",
    "prefers_main_process_loading",
    "resolve_dataloader_tuning",
    "visible_cpu_count",
]
