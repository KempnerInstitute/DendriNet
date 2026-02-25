"""Unit tests for stabilized local_mismatch broadcast signals."""

from __future__ import annotations

import torch

from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


def test_local_mismatch_falls_back_to_scalar_when_parent_missing() -> None:
    delta_scalar = torch.tensor([[0.5], [-0.25]], dtype=torch.float32)
    v_n = torch.zeros(2, 3, dtype=torch.float32)

    out = LocalCreditAssignment._compute_local_mismatch_broadcast(
        delta_scalar=delta_scalar,
        v_n=v_n,
        parent=None,
    )
    expected = delta_scalar.expand(-1, v_n.size(1))
    torch.testing.assert_close(out, expected)


def test_local_mismatch_falls_back_to_scalar_when_shape_mismatch() -> None:
    delta_scalar = torch.tensor([[0.2], [0.1]], dtype=torch.float32)
    v_n = torch.zeros(2, 4, dtype=torch.float32)
    parent = torch.zeros(2, 3, dtype=torch.float32)

    out = LocalCreditAssignment._compute_local_mismatch_broadcast(
        delta_scalar=delta_scalar,
        v_n=v_n,
        parent=parent,
    )
    expected = delta_scalar.expand(-1, v_n.size(1))
    torch.testing.assert_close(out, expected)


def test_local_mismatch_residual_path_prevents_zero_signal_collapse() -> None:
    delta_scalar = torch.tensor([[0.5], [-0.3]], dtype=torch.float32)
    v_n = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.0, 0.2]], dtype=torch.float32)

    out = LocalCreditAssignment._compute_local_mismatch_broadcast(
        delta_scalar=delta_scalar,
        v_n=v_n,
        parent=v_n.clone(),
        residual_fraction=0.2,
    )

    # Exact match of parent and soma gives zero mismatch, so output should be
    # the residual scalar path only.
    expected = 0.2 * delta_scalar.expand_as(v_n)
    torch.testing.assert_close(out, expected)


def test_local_mismatch_clipping_limits_amplification() -> None:
    delta_scalar = torch.ones(2, 1, dtype=torch.float32)
    v_n = torch.zeros(2, 16, dtype=torch.float32)
    parent = torch.zeros(2, 16, dtype=torch.float32)
    parent[0, 0] = 1.0
    parent[1, 0] = -1.0

    out = LocalCreditAssignment._compute_local_mismatch_broadcast(
        delta_scalar=delta_scalar,
        v_n=v_n,
        parent=parent,
        residual_fraction=0.0,
        clip_value=2.0,
    )

    assert float(out.abs().max().item()) <= 2.0 + 1e-6
