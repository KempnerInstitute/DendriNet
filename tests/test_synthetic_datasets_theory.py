"""Tests for synthetic dataset generators in datasets/synthetic_datasets.py."""

import pytest
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling.datasets.synthetic_datasets import (
    create_asymmetric_noise_dataset,
    create_balanced_information_dataset,
    create_block_correlated_dataset,
    create_contextual_stream_gain_shift_dataset,
    create_correlated_gaussian_dataset,
    create_heterogeneous_information_dataset,
    create_linear_separable_dataset,
    create_nonlinear_interaction_dataset,
    create_nonlinear_separable_dataset,
    create_nonlinear_transformation_dataset,
    create_population_code_dataset,
    get_synthetic_datasets,
)

# ---------------------------------------------------------------------------
# create_correlated_gaussian_dataset
# ---------------------------------------------------------------------------


def test_correlated_gaussian_returns_three_datasets():
    train, valid, test = create_correlated_gaussian_dataset(
        n_samples=30, input_dim=10, ne=5, ni=5, seed=42
    )
    assert isinstance(train, TensorDataset)
    assert isinstance(valid, TensorDataset)
    assert isinstance(test, TensorDataset)


def test_correlated_gaussian_shapes():
    n_samples = 40
    input_dim = 10
    n_classes = 2
    train, valid, test = create_correlated_gaussian_dataset(
        n_samples=n_samples,
        input_dim=input_dim,
        n_classes=n_classes,
        ne=5,
        ni=5,
        seed=0,
    )
    total = n_samples * n_classes
    n_train = int(0.6 * total)
    n_valid = int(0.2 * total)
    n_test = total - n_train - n_valid

    assert train.tensors[0].shape == (n_train, input_dim)
    assert valid.tensors[0].shape == (n_valid, input_dim)
    assert test.tensors[0].shape == (n_test, input_dim)


def test_correlated_gaussian_label_values():
    n_classes = 3
    train, _, _ = create_correlated_gaussian_dataset(
        n_samples=30, input_dim=12, n_classes=n_classes, ne=6, ni=6, seed=1
    )
    labels = train.tensors[1]
    assert labels.min() >= 0
    assert labels.max() < n_classes


def test_correlated_gaussian_data_clamped():
    train, valid, test = create_correlated_gaussian_dataset(
        n_samples=50, input_dim=10, ne=5, ni=5, seed=2
    )
    for ds in (train, valid, test):
        data = ds.tensors[0]
        assert (data >= 0).all()
        assert (data <= 1).all()


def test_correlated_gaussian_seed_reproducibility():
    ds1 = create_correlated_gaussian_dataset(
        n_samples=30, input_dim=10, ne=5, ni=5, seed=99
    )
    ds2 = create_correlated_gaussian_dataset(
        n_samples=30, input_dim=10, ne=5, ni=5, seed=99
    )
    torch.testing.assert_close(ds1[0].tensors[0], ds2[0].tensors[0])
    torch.testing.assert_close(ds1[0].tensors[1], ds2[0].tensors[1])


def test_correlated_gaussian_ne_ni_mismatch_raises():
    with pytest.raises(AssertionError):
        create_correlated_gaussian_dataset(
            n_samples=10, input_dim=10, ne=3, ni=5, seed=0
        )


def test_correlated_gaussian_split_ratios():
    n_samples = 50
    n_classes = 2
    total = n_samples * n_classes
    train, valid, test = create_correlated_gaussian_dataset(
        n_samples=n_samples, input_dim=10, ne=5, ni=5, seed=0
    )
    assert len(train) == int(0.6 * total)
    assert len(valid) == int(0.2 * total)
    assert len(test) == total - int(0.6 * total) - int(0.2 * total)


# ---------------------------------------------------------------------------
# create_nonlinear_interaction_dataset
# ---------------------------------------------------------------------------


def test_nonlinear_interaction_returns_three():
    train, valid, test = create_nonlinear_interaction_dataset(
        n_samples=40, input_dim=2, seed=0
    )
    assert isinstance(train, TensorDataset)
    assert isinstance(valid, TensorDataset)
    assert isinstance(test, TensorDataset)


def test_nonlinear_interaction_2d_labels():
    train, _, _ = create_nonlinear_interaction_dataset(
        n_samples=40, input_dim=2, seed=0
    )
    labels = train.tensors[1]
    assert set(labels.unique().tolist()).issubset({0, 1})


# ---------------------------------------------------------------------------
# create_linear_separable_dataset
# ---------------------------------------------------------------------------


def test_linear_separable_shapes():
    train, valid, test = create_linear_separable_dataset(
        n_samples=30, input_dim=10, ne=5, ni=5, seed=0
    )
    assert train.tensors[0].shape[1] == 10
    assert valid.tensors[0].shape[1] == 10
    assert test.tensors[0].shape[1] == 10


# ---------------------------------------------------------------------------
# create_nonlinear_separable_dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("nltype", ["quadratic", "xor", "multiplicative", "radial"])
def test_nonlinear_separable_all_types(nltype):
    train, _valid, _test = create_nonlinear_separable_dataset(
        n_samples=30, input_dim=10, ne=5, ni=5, nonlinearity_type=nltype, seed=0
    )
    assert isinstance(train, TensorDataset)
    assert len(train) > 0


def test_nonlinear_separable_multiplicative_stays_balanced_high_dimensional():
    datasets = create_nonlinear_separable_dataset(
        n_samples=300,
        input_dim=200,
        ne=100,
        ni=100,
        nonlinearity_type="multiplicative",
        noise_std=0.0,
        seed=0,
    )
    labels = torch.cat([dataset.tensors[1] for dataset in datasets])

    assert labels.float().mean().item() == pytest.approx(0.5, abs=0.03)


def test_nonlinear_separable_unknown_raises():
    with pytest.raises(ValueError, match="Unknown nonlinearity"):
        create_nonlinear_separable_dataset(
            n_samples=10, input_dim=10, ne=5, ni=5, nonlinearity_type="cubic", seed=0
        )


# ---------------------------------------------------------------------------
# create_block_correlated_dataset
# ---------------------------------------------------------------------------


def test_block_correlated_shapes():
    train, _valid, _test = create_block_correlated_dataset(
        n_samples=30, input_dim=20, ne=10, ni=10, n_blocks=2, seed=0
    )
    assert train.tensors[0].shape[1] == 20


# ---------------------------------------------------------------------------
# create_balanced_information_dataset
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["balanced", "concentrated", "gradient"])
def test_balanced_info_all_modes(mode):
    train, _valid, _test = create_balanced_information_dataset(
        n_samples=30, input_dim=10, ne=5, ni=5, info_per_dim=mode, seed=0
    )
    assert isinstance(train, TensorDataset)
    assert len(train) > 0


# ---------------------------------------------------------------------------
# get_synthetic_datasets (factory)
# ---------------------------------------------------------------------------


def test_get_synthetic_datasets_dispatch():
    config = {
        "type": "correlated_gaussian",
        "n_samples": 20,
        "input_dim": 10,
        "ne": 5,
        "ni": 5,
        "seed": 0,
    }
    train, _valid, _test = get_synthetic_datasets(config)
    assert isinstance(train, TensorDataset)


@pytest.mark.parametrize(
    ("dataset_type", "factory", "kwargs"),
    [
        (
            "correlated_gaussian",
            create_correlated_gaussian_dataset,
            {"n_samples": 12, "input_dim": 8, "ne": 4, "ni": 4, "seed": 11},
        ),
        (
            "nonlinear_interaction",
            create_nonlinear_interaction_dataset,
            {"n_samples": 24, "input_dim": 4, "seed": 12},
        ),
        (
            "linear_separable",
            create_linear_separable_dataset,
            {"n_samples": 12, "input_dim": 8, "ne": 4, "ni": 4, "seed": 13},
        ),
        (
            "nonlinear_separable",
            create_nonlinear_separable_dataset,
            {
                "n_samples": 24,
                "input_dim": 8,
                "ne": 4,
                "ni": 4,
                "nonlinearity_type": "quadratic",
                "seed": 14,
            },
        ),
        (
            "block_correlated",
            create_block_correlated_dataset,
            {
                "n_samples": 12,
                "input_dim": 8,
                "ne": 4,
                "ni": 4,
                "n_blocks": 2,
                "block_rho_ee": [0.0, 0.1],
                "block_rho_ii": [0.0, 0.1],
                "block_rho_ei": [0.0, 0.1],
                "seed": 15,
            },
        ),
        (
            "balanced_information",
            create_balanced_information_dataset,
            {
                "n_samples": 12,
                "input_dim": 8,
                "ne": 4,
                "ni": 4,
                "info_per_dim": "gradient",
                "seed": 16,
            },
        ),
        (
            "heterogeneous_information",
            create_heterogeneous_information_dataset,
            {
                "n_samples": 8,
                "input_dim": 8,
                "n_classes": 3,
                "ne": 4,
                "ni": 4,
                "n_high_info_dims": 2,
                "seed": 17,
            },
        ),
        (
            "nonlinear_transformation",
            create_nonlinear_transformation_dataset,
            {
                "n_samples": 12,
                "n_classes": 2,
                "ne": 2,
                "ni": 2,
                "noise_std": 0.05,
                "seed": 18,
            },
        ),
        (
            "asymmetric_noise",
            create_asymmetric_noise_dataset,
            {
                "n_samples": 8,
                "input_dim": 8,
                "n_classes": 3,
                "ne": 4,
                "ni": 4,
                "seed": 19,
            },
        ),
        (
            "population_code",
            create_population_code_dataset,
            {
                "n_samples": 12,
                "pop_dim": 16,
                "ne": 4,
                "ni": 4,
                "n_classes": 2,
                "seed": 20,
            },
        ),
        (
            "contextual_stream_gain_shift",
            create_contextual_stream_gain_shift_dataset,
            {"n_samples": 12, "stream_dim": 4, "n_classes": 2, "seed": 21},
        ),
    ],
)
def test_get_synthetic_datasets_matches_direct_constructor(
    dataset_type, factory, kwargs
):
    config = {"type": dataset_type, **kwargs}

    actual = get_synthetic_datasets(config)
    expected = factory(**kwargs)

    for actual_split, expected_split in zip(actual, expected):
        torch.testing.assert_close(actual_split.tensors[0], expected_split.tensors[0])
        torch.testing.assert_close(actual_split.tensors[1], expected_split.tensors[1])


def test_get_synthetic_datasets_unknown_raises():
    with pytest.raises(ValueError, match="Unknown synthetic dataset type"):
        get_synthetic_datasets({"type": "nonexistent"})
