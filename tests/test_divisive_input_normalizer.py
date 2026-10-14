"""Tests for fixed global and grouped input-normalization controls."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform import (
    DivisiveInputNormalizer,
    TransferLayer,
)


def test_default_mode_is_exact_identity() -> None:
    value = torch.randn(3, 8, dtype=torch.float64)
    normalizer = DivisiveInputNormalizer(8)
    assert normalizer(value) is value
    assert normalizer.state_dict() == {}


def test_global_l1_removes_only_common_gain() -> None:
    normalizer = DivisiveInputNormalizer(4, mode="global_l1", epsilon=1e-12)
    value = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float64)
    common = normalizer(7.0 * value)
    torch.testing.assert_close(common, normalizer(value), atol=1e-11, rtol=1e-11)

    relative = value * torch.tensor([[2.0, 2.0, 0.5, 0.5]], dtype=torch.float64)
    assert not torch.allclose(normalizer(relative), normalizer(value))


def test_group_l1_removes_independent_group_gains() -> None:
    normalizer = DivisiveInputNormalizer(
        4,
        mode="group_l1",
        group_ranges=[[0, 2], [2, 4]],
        epsilon=1e-12,
    )
    value = torch.tensor([[1.0, 3.0, 2.0, 5.0]], dtype=torch.float64)
    independently_gained = value * torch.tensor(
        [[0.25, 0.25, 9.0, 9.0]], dtype=torch.float64
    )
    torch.testing.assert_close(
        normalizer(independently_gained),
        normalizer(value),
        atol=1e-11,
        rtol=1e-11,
    )


def test_group_partition_validation_fails_closed() -> None:
    with pytest.raises(ValueError, match="exact non-overlapping partition"):
        DivisiveInputNormalizer(
            4,
            mode="group_l1",
            group_ranges=[[0, 3], [2, 4]],
        )
    with pytest.raises(ValueError, match="requires non-empty"):
        DivisiveInputNormalizer(4, mode="group_l1")
    with pytest.raises(ValueError, match="only valid"):
        DivisiveInputNormalizer(4, mode="global_l1", group_ranges=[[0, 4]])


def test_transfer_layer_applies_configured_normalizer_before_duplication() -> None:
    transfer = TransferLayer(
        input_dim=4,
        transfer_params={
            "independent_pathways": False,
            "input_normalization": {
                "mode": "group_l1",
                "group_ranges": [[0, 2], [2, 4]],
                "epsilon": 1e-12,
            },
        },
    ).double()
    value = torch.tensor([[2.0, 2.0, 3.0, 1.0]], dtype=torch.float64)
    excitatory, inhibitory = transfer(value)
    expected = torch.tensor([[0.5, 0.5, 0.75, 0.25]], dtype=torch.float64)
    torch.testing.assert_close(excitatory, expected, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(inhibitory, expected, atol=1e-11, rtol=1e-11)
