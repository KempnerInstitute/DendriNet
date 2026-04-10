import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.datasets.standard_datasets import (
    _build_paired_arithmetic_dataset,
)


def test_paired_sum_mod10_flattened_layout() -> None:
    inputs = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    labels = torch.arange(8, dtype=torch.long)
    base = TensorDataset(inputs, labels)

    dataset = _build_paired_arithmetic_dataset(
        base,
        shuffle_iterations=1,
        task_mode="sum_mod10",
        flatten=True,
        include_context=False,
        pair_seed=0,
    )

    x, y = dataset[0]
    assert x.ndim == 1
    assert x.shape[0] == 6
    assert 0 <= int(y.item()) <= 9


def test_paired_contextual_dataset_appends_context_and_switches_rule() -> None:
    inputs = torch.arange(8 * 1 * 2 * 2, dtype=torch.float32).reshape(8, 1, 2, 2)
    labels = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long)
    base = TensorDataset(inputs, labels)

    dataset = _build_paired_arithmetic_dataset(
        base,
        shuffle_iterations=1,
        task_mode="context_sum_diff",
        flatten=True,
        include_context=True,
        pair_seed=0,
        context_seed=0,
    )

    x0, y0 = dataset[0]
    x1, y1 = dataset[1]

    # Two flattened images (4 + 4) plus 2-d one-hot context.
    assert x0.shape[0] == 10
    assert torch.isclose(x0[-2:].sum(), torch.tensor(1.0))
    assert torch.isclose(x1[-2:].sum(), torch.tensor(1.0))
    assert 0 <= int(y0.item()) <= 9
    assert 0 <= int(y1.item()) <= 9
