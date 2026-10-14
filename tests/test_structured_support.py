"""Tests for hardware-structured fixed support (row groups x column blocks)."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    sample_structured_indices,
)


def test_structured_indices_share_support_within_row_groups() -> None:
    generator = torch.Generator().manual_seed(3)
    indices = sample_structured_indices(
        out_features=12,
        in_features=64,
        K=8,
        generator=generator,
        support_group_rows=4,
        support_col_block=2,
    )

    assert indices.shape == (12, 8)
    for group_start in range(0, 12, 4):
        group = indices[group_start : group_start + 4]
        assert (group == group[0]).all(), "rows in a group must share support"
    # Column-block structure: consecutive pairs differ by exactly one.
    pairs = indices.view(12, 4, 2)
    assert ((pairs[..., 1] - pairs[..., 0]) == 1).all()
    assert (pairs[..., 0] % 2 == 0).all()
    # Unique columns within each row.
    sorted_row = indices[0].sort().values
    assert (sorted_row[1:] != sorted_row[:-1]).all()


def test_structured_indices_fail_closed_on_indivisible_shapes() -> None:
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="divisible"):
        sample_structured_indices(
            out_features=4,
            in_features=64,
            K=6,
            generator=generator,
            support_col_block=4,
        )


def test_indexed_layer_defaults_reproduce_unstructured_topology() -> None:
    kwargs = {
        "in_features": 64,
        "out_features": 16,
        "K": 8,
        "seed": 11,
        "weight_transform": "softplus",
    }
    baseline = IndexedSparseLinear(**kwargs)
    explicit = IndexedSparseLinear(**kwargs, support_group_rows=1, support_col_block=1)

    torch.testing.assert_close(baseline.connection_indices, explicit.connection_indices)


def test_indexed_layer_structured_support_forward_and_index_sharing() -> None:
    layer = IndexedSparseLinear(
        in_features=64,
        out_features=16,
        K=8,
        seed=11,
        weight_transform="softplus",
        support_group_rows=4,
        support_col_block=2,
    )
    indices = layer.connection_indices
    for group_start in range(0, 16, 4):
        group = indices[group_start : group_start + 4]
        assert (group == group[0]).all()

    out = layer(torch.rand(3, 64))
    assert out.shape == (3, 16)
    assert torch.isfinite(out).all()


def test_structured_support_rejects_connection_mask() -> None:
    with pytest.raises(ValueError, match="does not compose"):
        IndexedSparseLinear(
            in_features=8,
            out_features=4,
            K=2,
            seed=1,
            connection_mask=torch.ones(4, 8, dtype=torch.bool),
            support_group_rows=2,
        )
