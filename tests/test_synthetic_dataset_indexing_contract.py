import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset

from dendritic_modeling.datasets.standard_datasets import (
    SliceSafeDataset,
    get_unified_datasets as get_standard_unified_datasets,
)

experiments_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"
)
if experiments_path not in sys.path:
    sys.path.insert(0, experiments_path)

try:
    from data_generation.synthetic_datasets import get_unified_datasets
except ImportError:  # pragma: no cover - integration-only path
    get_unified_datasets = None


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


def test_slice_and_batch_indexing_for_split_synthetic_datasets():
    if get_unified_datasets is None:
        return

    train_cfg = SimpleNamespace(train_valid_split=0.8, seed=7)

    dataset_settings = [
        ("line", {"train_size": 20, "valid_size": 6, "test_size": 6, "image_size": 8}),
        (
            "orthonet",
            {
                "train_size": 16,
                "valid_size": 4,
                "test_size": 4,
                "input_dim": 6,
                "n_layers": 2,
                "noise_std": 0.0,
            },
        ),
    ]

    for dataset_name, parameters in dataset_settings:
        task_cfg = {
            "dataset": dataset_name,
            "data_path": None,
            "parameters": parameters,
        }
        train_ds, valid_ds, test_ds = get_unified_datasets(task_cfg, train_cfg)

        _assert_indexing_contract(train_ds)
        _assert_indexing_contract(valid_ds)
        _assert_indexing_contract(test_ds)


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


def test_standard_loader_wraps_experimental_datasets_as_slice_safe():
    fake_module = SimpleNamespace(
        get_unified_datasets=lambda *_: (
            _ScalarOnlyDataset(10),
            _ScalarOnlyDataset(6),
            _ScalarOnlyDataset(6),
        )
    )

    with patch(
        "dendritic_modeling.datasets.standard_datasets.importlib.import_module",
        return_value=fake_module,
    ):
        task_cfg = {
            "dataset": "custom_experimental_dataset",
            "data_path": None,
            "train_valid_split": 0.8,
            "parameters": {},
        }
        train_ds, valid_ds, test_ds = get_standard_unified_datasets(task_cfg)

    assert isinstance(train_ds, SliceSafeDataset)
    assert isinstance(valid_ds, SliceSafeDataset)
    assert isinstance(test_ds, SliceSafeDataset)

    _assert_indexing_contract(train_ds)
