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
    create_nested_multixor_dataset,
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
# create_nested_multixor_dataset
# ---------------------------------------------------------------------------


def test_nested_multixor_shapes_and_binary_labels():
    branch_factors = [2, 2]
    dims_per_leaf = 8
    latent_dims = 2
    train, valid, test = create_nested_multixor_dataset(
        n_samples=128,
        branch_factors=branch_factors,
        dims_per_leaf=dims_per_leaf,
        latent_dims=latent_dims,
        include_internal_latents=False,
        auxiliary_root=False,
        seed=0,
    )
    # Default: leaf blocks only
    input_dim = 4 * dims_per_leaf
    for split in (train, valid, test):
        assert split.tensors[0].shape[1] == input_dim
        assert set(split.tensors[1].unique().tolist()).issubset({0, 1})
    assert len(train) > 0 and len(valid) > 0 and len(test) > 0


def test_nested_multixor_shapes_with_internal_latents():
    dims_per_leaf = 8
    train, _, _ = create_nested_multixor_dataset(
        n_samples=64,
        branch_factors=[2, 2],
        dims_per_leaf=dims_per_leaf,
        latent_dims=2,
        include_internal_latents=True,
        auxiliary_root=False,
        seed=0,
    )
    # 4 leaves + 2 non-root internals
    assert train.tensors[0].shape[1] == 6 * dims_per_leaf


def test_nested_multixor_auxiliary_root_ignored_without_internal_latents():
    dims = 8
    train, _, _ = create_nested_multixor_dataset(
        n_samples=64,
        branch_factors=[2, 2],
        dims_per_leaf=dims,
        latent_dims=2,
        distractor_scale=0.0,
        include_internal_latents=False,
        auxiliary_root=True,
        seed=0,
    )
    assert train.tensors[0].shape[1] == 4 * dims


def test_nested_multixor_rejects_invalid_branch_factors():
    with pytest.raises(ValueError, match="branch factor"):
        create_nested_multixor_dataset(n_samples=16, branch_factors=[2, 1], seed=0)
    with pytest.raises(ValueError, match="non-empty"):
        create_nested_multixor_dataset(n_samples=16, branch_factors=[], seed=0)


def test_nested_multixor_rejects_dims_per_leaf_smaller_than_latent_dims():
    with pytest.raises(ValueError, match="dims_per_leaf"):
        create_nested_multixor_dataset(
            n_samples=16,
            branch_factors=[2, 2],
            dims_per_leaf=1,
            latent_dims=2,
            seed=0,
        )


def test_nested_multixor_schedule_shapes_leaves_only():
    dims = 8
    cases = [
        ([2, 2], 4),
        ([2, 3], 6),
        ([3, 3, 3], 27),
        ([2], 2),
    ]
    for branch_factors, n_blocks in cases:
        train, _, _ = create_nested_multixor_dataset(
            n_samples=64,
            branch_factors=branch_factors,
            dims_per_leaf=dims,
            latent_dims=2,
            distractor_scale=0.0,
            include_internal_latents=False,
            seed=0,
        )
        assert train.tensors[0].shape[1] == n_blocks * dims


def test_nested_multixor_schedule_shapes_auxiliary_root_false():
    dims = 8
    cases = [
        ([2, 2], 6),  # 4 leaves + 2 proximal
        ([2, 3], 8),  # 6 leaves + 2 proximal
        ([3, 3, 3], 39),  # 27 leaves + 12 non-root internals
        ([2], 2),  # leaves only when root aux excluded
    ]
    for branch_factors, n_blocks in cases:
        train, _, _ = create_nested_multixor_dataset(
            n_samples=64,
            branch_factors=branch_factors,
            dims_per_leaf=dims,
            latent_dims=2,
            distractor_scale=0.0,
            include_internal_latents=True,
            auxiliary_root=False,
            seed=0,
        )
        assert train.tensors[0].shape[1] == n_blocks * dims


def test_nested_multixor_schedule_shapes_auxiliary_root_true():
    dims = 8
    cases = [
        ([2, 2], 7),  # + root aux
        ([2], 3),  # 2 leaves + root
    ]
    for branch_factors, n_blocks in cases:
        train, _, _ = create_nested_multixor_dataset(
            n_samples=64,
            branch_factors=branch_factors,
            dims_per_leaf=dims,
            latent_dims=2,
            distractor_scale=0.0,
            include_internal_latents=True,
            auxiliary_root=True,
            seed=0,
        )
        assert train.tensors[0].shape[1] == n_blocks * dims


def test_nested_multixor_labels_ignore_within_block_distractors():
    """Labels depend on latents only; population distractor mix does not change them."""
    from dendritic_modeling.datasets.synthetic_datasets import (
        _leaf_values,
        _make_population_projection,
        _nested_reduce_with_internal_aux,
        _population_embed_leaf_blocks,
    )

    torch.manual_seed(0)
    n_leaves = 4
    n_internal = 2
    latent_dims = 2
    dims_per_leaf = 8
    n = 200
    leaf_latents = torch.rand(n, n_leaves, latent_dims)
    internal_latents = torch.rand(n, n_internal, latent_dims)
    node_latents = torch.cat([leaf_latents, internal_latents], dim=1)
    projection = _make_population_projection(latent_dims, dims_per_leaf)

    leaf_bits = _leaf_values(leaf_latents, leaf_op="and", threshold=0.25)
    aux_bits = _leaf_values(internal_latents, leaf_op="and", threshold=0.25)
    labels = _nested_reduce_with_internal_aux(
        leaf_bits,
        aux_bits,
        (2, 2),
        internal_op="xor",
        auxiliary_root=False,
        include_internal_latents=True,
    )

    blocks_a = _population_embed_leaf_blocks(
        node_latents, projection, distractor_scale=0.4
    )
    blocks_b = _population_embed_leaf_blocks(
        node_latents, projection, distractor_scale=0.4
    )
    assert not torch.equal(blocks_a, blocks_b)

    leaf_bits_again = _leaf_values(leaf_latents, leaf_op="and", threshold=0.25)
    aux_bits_again = _leaf_values(internal_latents, leaf_op="and", threshold=0.25)
    labels_again = _nested_reduce_with_internal_aux(
        leaf_bits_again,
        aux_bits_again,
        (2, 2),
        internal_op="xor",
        auxiliary_root=False,
        include_internal_latents=True,
    )
    assert torch.equal(labels, labels_again)


def test_nested_multixor_population_embedding_distributes_latents():
    """With mixing, observed block dims are not identical to raw latent coords."""
    from dendritic_modeling.datasets.synthetic_datasets import (
        _make_population_projection,
        _population_embed_leaf_blocks,
    )

    torch.manual_seed(1)
    latents = torch.rand(32, 6, 2)  # 4 leaves + 2 aux
    projection = _make_population_projection(2, 8)
    blocks = _population_embed_leaf_blocks(latents, projection, distractor_scale=0.0)
    assert blocks.shape == (32, 6, 8)
    assert not torch.allclose(blocks[..., :2], latents, atol=1e-5)
    assert (projection > 1e-6).all()


def test_nested_multixor_rejects_invalid_distractor_scale():
    with pytest.raises(ValueError, match="distractor_scale"):
        create_nested_multixor_dataset(
            n_samples=8,
            branch_factors=[2, 2],
            dims_per_leaf=4,
            latent_dims=2,
            distractor_scale=1.5,
            seed=0,
        )


def test_nested_multixor_reduce_respects_distal_then_proximal_order():
    """branch_factors are proximal→distal; aux-aware reduction applies reversed factors."""
    from dendritic_modeling.datasets.synthetic_datasets import (
        _nested_reduce_with_internal_aux,
    )

    # [2, 3]: 6 leaves, 2 proximal aux (no root). Distal groups of 3 + aux, then
    # proximal pair without root aux.
    # xor(1,0,1, a0=1)=1 ; xor(0,1,0, a1=0)=1 ; xor(1,1)=0
    leaves = torch.tensor([[1, 0, 1, 0, 1, 0]], dtype=torch.long)
    aux = torch.tensor([[1, 0]], dtype=torch.long)
    out = _nested_reduce_with_internal_aux(
        leaves,
        aux,
        (2, 3),
        internal_op="xor",
        auxiliary_root=False,
        include_internal_latents=True,
    )
    assert out.tolist() == [0]

    # Same with root aux: xor(1,1, a_root=1)=1
    aux_root = torch.tensor([[1, 0, 1]], dtype=torch.long)
    out_root = _nested_reduce_with_internal_aux(
        leaves,
        aux_root,
        (2, 3),
        internal_op="xor",
        auxiliary_root=True,
        include_internal_latents=True,
    )
    assert out_root.tolist() == [1]

    # Children-only flat schedule [6] with no aux (single factor, no root aux).
    flat_leaves = leaves
    flat_aux = torch.zeros(1, 0, dtype=torch.long)
    flat = _nested_reduce_with_internal_aux(
        flat_leaves,
        flat_aux,
        (6,),
        internal_op="xor",
        auxiliary_root=False,
        include_internal_latents=False,
    )
    assert flat.tolist() == [1]

    # Children-only nested [2,2]: xor(1,0)=1, xor(1,0)=1, xor(1,1)=0
    nested_leaves = torch.tensor([[1, 0, 1, 0]], dtype=torch.long)
    nested_empty = torch.zeros(1, 0, dtype=torch.long)
    nested_children = _nested_reduce_with_internal_aux(
        nested_leaves,
        nested_empty,
        (2, 2),
        internal_op="xor",
        auxiliary_root=False,
        include_internal_latents=False,
    )
    assert nested_children.tolist() == [0]

    with pytest.raises(ValueError, match="does not match"):
        _nested_reduce_with_internal_aux(
            leaves,
            aux,
            (2, 2),
            internal_op="xor",
            auxiliary_root=False,
            include_internal_latents=True,
        )


def test_nested_multixor_and_xor_differs_from_flat_parity():
    """Default and→xor nesting is not equivalent to XOR of all soft bits."""
    from dendritic_modeling.datasets.synthetic_datasets import (
        _leaf_values,
        _nested_reduce_with_internal_aux,
        _soft_bits,
    )

    torch.manual_seed(0)
    n_leaves = 4
    n_internal = 2
    latent_dims = 2
    leaf_latents = torch.rand(2000, n_leaves, latent_dims)
    internal_latents = torch.rand(2000, n_internal, latent_dims)
    and_leaves = _leaf_values(leaf_latents, leaf_op="and", threshold=0.25)
    and_aux = _leaf_values(internal_latents, leaf_op="and", threshold=0.25)
    nested = _nested_reduce_with_internal_aux(
        and_leaves,
        and_aux,
        (2, 2),
        internal_op="xor",
        auxiliary_root=False,
        include_internal_latents=True,
    )

    all_latents = torch.cat([leaf_latents, internal_latents], dim=1)
    flat_bits = _soft_bits(all_latents.reshape(2000, (n_leaves + n_internal) * latent_dims))
    parity = flat_bits[:, 0]
    for d in range(1, flat_bits.shape[1]):
        parity = parity ^ flat_bits[:, d]

    assert not torch.equal(nested, parity)


def test_nested_multixor_xor_xor_matches_node_parity():
    """xor→xor nesting collapses to XOR of leaf bits and included aux bits."""
    from dendritic_modeling.datasets.synthetic_datasets import (
        _leaf_values,
        _nested_reduce_with_internal_aux,
    )

    torch.manual_seed(1)
    n_leaves = 4
    n_internal = 2
    latent_dims = 2
    leaf_latents = torch.rand(500, n_leaves, latent_dims)
    internal_latents = torch.rand(500, n_internal, latent_dims)
    xor_leaves = _leaf_values(leaf_latents, leaf_op="xor", threshold=0.25)
    xor_aux = _leaf_values(internal_latents, leaf_op="xor", threshold=0.25)
    nested = _nested_reduce_with_internal_aux(
        xor_leaves,
        xor_aux,
        (2, 2),
        internal_op="xor",
        auxiliary_root=False,
        include_internal_latents=True,
    )

    parity = xor_leaves[:, 0]
    for i in range(1, n_leaves):
        parity = parity ^ xor_leaves[:, i]
    for i in range(n_internal):
        parity = parity ^ xor_aux[:, i]

    assert torch.equal(nested, parity)


def test_nested_multixor_shuffle_leaves_changes_labels():
    kwargs = {
        "n_samples": 256,
        "branch_factors": [2, 2],
        "dims_per_leaf": 8,
        "latent_dims": 2,
        "distractor_scale": 0.25,
        "include_internal_latents": True,
        "auxiliary_root": False,
        "seed": 7,
    }
    train_a, _, _ = create_nested_multixor_dataset(shuffle_leaves=False, **kwargs)
    train_b, _, _ = create_nested_multixor_dataset(shuffle_leaves=True, **kwargs)
    # Same generator seed regenerates the same raw latents before shuffle labeling,
    # but shuffled node assignment should change at least some labels after balance.
    assert not torch.equal(train_a.tensors[1], train_b.tensors[1]) or not torch.equal(
        train_a.tensors[0], train_b.tensors[0]
    )


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
        (
            "nested_multixor",
            create_nested_multixor_dataset,
            {
                "n_samples": 64,
                "branch_factors": [2, 2],
                "dims_per_leaf": 8,
                "latent_dims": 2,
                "distractor_scale": 0.25,
                "include_internal_latents": False,
                "auxiliary_root": False,
                "seed": 22,
            },
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
