"""Dataset-profile helpers for DataLoader tuning."""

from __future__ import annotations

from torch.utils.data import Dataset, Subset, TensorDataset

try:
    from torchvision.datasets.folder import DatasetFolder as TorchvisionDatasetFolder
except Exception:  # pragma: no cover - torchvision is an optional runtime dependency
    TorchvisionDatasetFolder = None

_TORCHVISION_IMAGE_DATASET_NAMES = {
    "DatasetFolder",
    "ImageFolder",
    "ImageNet",
}


def base_dataset(dataset: Dataset) -> Dataset:
    """Unwrap common dataset adapters to the storage-backed base dataset."""
    seen: set[int] = set()
    while id(dataset) not in seen:
        seen.add(id(dataset))
        if isinstance(dataset, Subset):
            dataset = dataset.dataset
            continue
        base = getattr(dataset, "base_dataset", None)
        if isinstance(base, Dataset):
            dataset = base
            continue
        wrapped = getattr(dataset, "dataset", None)
        if isinstance(wrapped, Dataset):
            dataset = wrapped
            continue
        break
    return dataset


def is_torchvision_image_dataset(dataset: Dataset) -> bool:
    """Return True for torchvision image datasets that are usually I/O-bound."""
    resolved_dataset = base_dataset(dataset)
    if TorchvisionDatasetFolder is not None and isinstance(
        resolved_dataset,
        TorchvisionDatasetFolder,
    ):
        return True

    dataset_cls = type(resolved_dataset)
    return (
        dataset_cls.__name__ in _TORCHVISION_IMAGE_DATASET_NAMES
        and dataset_cls.__module__.startswith("torchvision.datasets")
    )


def dataset_type_name(dataset: Dataset) -> str:
    """Return the module-qualified type name of the storage-backed dataset."""
    resolved_dataset = base_dataset(dataset)
    return f"{type(resolved_dataset).__module__}.{type(resolved_dataset).__name__}"


def prefers_main_process_loading(dataset: Dataset) -> bool:
    """Return True for in-memory tensor datasets where workers add overhead."""
    return isinstance(base_dataset(dataset), TensorDataset)


__all__ = [
    "base_dataset",
    "dataset_type_name",
    "is_torchvision_image_dataset",
    "prefers_main_process_loading",
]
