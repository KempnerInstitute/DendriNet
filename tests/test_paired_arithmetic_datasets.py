import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.datasets.standard_datasets import (
    _build_iterative_modulo10_dataset,
    _build_paired_arithmetic_dataset,
    _build_paired_arithmetic_triplet,
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


def test_paired_arithmetic_triplet_matches_direct_split_seed_offsets() -> None:
    inputs = torch.arange(12 * 2, dtype=torch.float32).reshape(12, 2)
    labels = torch.arange(12, dtype=torch.long)
    train = TensorDataset(inputs, labels)
    valid = TensorDataset(inputs + 100, labels + 10)
    test = TensorDataset(inputs + 200, labels + 20)

    triplet = _build_paired_arithmetic_triplet(
        train,
        valid,
        test,
        shuffle_iterations=2,
        task_mode="context_sum_diff",
        flatten=True,
        include_context=True,
        pair_seed=5,
        context_seed=13,
    )
    expected = {
        "train": _build_paired_arithmetic_dataset(
            train,
            shuffle_iterations=2,
            task_mode="context_sum_diff",
            flatten=True,
            include_context=True,
            pair_seed=5,
            context_seed=13,
        ),
        "valid": _build_paired_arithmetic_dataset(
            valid,
            shuffle_iterations=2,
            task_mode="context_sum_diff",
            flatten=True,
            include_context=True,
            pair_seed=6,
            context_seed=14,
        ),
        "test": _build_paired_arithmetic_dataset(
            test,
            shuffle_iterations=2,
            task_mode="context_sum_diff",
            flatten=True,
            include_context=True,
            pair_seed=7,
            context_seed=15,
        ),
    }

    assert list(triplet) == ["train", "valid", "test"]
    for split_name, dataset in triplet.items():
        x, y = dataset[0]
        expected_x, expected_y = expected[split_name][0]
        assert torch.equal(x, expected_x)
        assert torch.equal(y, expected_y)


def test_iterative_modulo10_dataset_preserves_split_half_shuffle() -> None:
    inputs = torch.arange(8 * 2, dtype=torch.float32).reshape(8, 2)
    labels = torch.arange(8, dtype=torch.long)
    base = TensorDataset(inputs, labels)

    dataset = _build_iterative_modulo10_dataset(
        base,
        shuffle_iterations=2,
        pair_seed=11,
    )

    expected_inputs = torch.tensor(
        [
            [2.0, 3.0, 0.0, 1.0],
            [6.0, 7.0, 8.0, 9.0],
            [4.0, 5.0, 14.0, 15.0],
            [10.0, 11.0, 12.0, 13.0],
            [2.0, 3.0, 14.0, 15.0],
            [4.0, 5.0, 8.0, 9.0],
            [10.0, 11.0, 12.0, 13.0],
            [6.0, 7.0, 0.0, 1.0],
        ]
    )
    expected_labels = torch.tensor([1, 7, 9, 1, 8, 6, 1, 3], dtype=torch.long)

    torch.testing.assert_close(dataset.tensors[0], expected_inputs)
    torch.testing.assert_close(dataset.tensors[1], expected_labels)
