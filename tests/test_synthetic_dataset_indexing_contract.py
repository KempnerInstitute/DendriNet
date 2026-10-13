"""Datasets returned by the loaders support list, tensor and slice indexing."""

import torch
from torch.utils.data import Dataset

from dendritic_modeling.datasets.indexing import _ensure_slice_safe_triplet
from dendritic_modeling.datasets.standard_datasets import (
    SliceSafeDataset,
    get_unified_datasets as get_standard_unified_datasets,
)


def _manual_batch(dataset, indices):
    xs = []
    ys = []
    for idx in indices:
        x, y = dataset[int(idx)]
        xs.append(x if torch.is_tensor(x) else torch.as_tensor(x))
        ys.append(y if torch.is_tensor(y) else torch.as_tensor(y))
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0)


def _assert_indexing_contract(dataset):
    assert len(dataset) >= 3
    indices = [0, 1, 2]

    x_manual, y_manual = _manual_batch(dataset, indices)

    x_list, y_list = dataset[indices]
    torch.testing.assert_close(x_list, x_manual)
    torch.testing.assert_close(y_list, y_manual)

    x_tensor, y_tensor = dataset[torch.tensor(indices)]
    torch.testing.assert_close(x_tensor, x_manual)
    torch.testing.assert_close(y_tensor, y_manual)

    x_slice, y_slice = dataset[:3]
    torch.testing.assert_close(x_slice, x_manual)
    torch.testing.assert_close(y_slice, y_manual)


def test_slice_and_batch_indexing_for_library_synthetic_datasets():
    for dataset_name in ("linear_separable", "nonlinear_transformation"):
        task_cfg = {
            "dataset": dataset_name,
            "data_path": None,
            "train_valid_split": 0.8,
            "parameters": {"n_samples": 40, "seed": 7},
        }
        for dataset in get_standard_unified_datasets(task_cfg):
            _assert_indexing_contract(dataset)


class _ScalarOnlyDataset(Dataset):
    """Dataset that intentionally rejects non-scalar indexing."""

    def __init__(self, size: int = 10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError("scalar-only indexing")
        return torch.tensor([float(idx)]), torch.tensor(idx % 2, dtype=torch.long)


def test_scalar_only_datasets_are_wrapped_as_slice_safe():
    triplet = _ensure_slice_safe_triplet(
        _ScalarOnlyDataset(10), _ScalarOnlyDataset(6), _ScalarOnlyDataset(6)
    )
    for dataset in triplet:
        assert isinstance(dataset, SliceSafeDataset)
        _assert_indexing_contract(dataset)
