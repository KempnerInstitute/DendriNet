"""Tests for operational local-learning feedback modes."""

from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_mixin import (
    LocalLearningBroadcastMixin,
)


class _BroadcastProbe(LocalLearningBroadcastMixin):
    def __init__(self):
        self.local_cfg = SimpleNamespace(
            error_noise_sigma=0.0,
            broadcast_bandwidth="full",
        )


def _compute(mode: str, delta: torch.Tensor, out_features: int) -> torch.Tensor:
    probe = _BroadcastProbe()
    v_n = torch.zeros(delta.size(0), out_features)
    return probe._compute_local_broadcast_error(
        rec={},
        layer_idx=0,
        out_features=out_features,
        v_n=v_n,
        delta=delta,
        delta_scalar=delta.mean(dim=1, keepdim=True),
        broadcast_state=SimpleNamespace(mode=mode),
    )


def test_legacy_per_soma_falls_back_to_scalar_at_wider_branch_layer():
    delta = torch.tensor([[1.0, 3.0]])

    out = _compute("per_soma", delta, out_features=4)

    torch.testing.assert_close(out, torch.full((1, 4), 2.0))


def test_per_soma_shared_repeats_each_error_over_descendant_compartments():
    delta = torch.tensor([[1.0, 3.0]])

    out = _compute("per_soma_shared", delta, out_features=4)

    torch.testing.assert_close(out, torch.tensor([[1.0, 1.0, 3.0, 3.0]]))


def test_per_soma_shared_preserves_exact_soma_width():
    delta = torch.tensor([[1.0, 3.0]])

    out = _compute("per_soma_shared", delta, out_features=2)

    torch.testing.assert_close(out, delta)
