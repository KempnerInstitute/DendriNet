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
            broadcast_seed=17,
        )
        self._broadcast_cache = {}


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
        broadcast_state=SimpleNamespace(mode=mode, readout_records=frozenset({0})),
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


def test_per_soma_shuffled_preserves_values_and_bandwidth_but_breaks_routing():
    probe = _BroadcastProbe()
    delta = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    v_n = torch.zeros(1, 8)
    kwargs = {
        "rec": {},
        "layer_idx": 0,
        "out_features": 8,
        "v_n": v_n,
        "delta": delta,
        "delta_scalar": delta.mean(dim=1, keepdim=True),
        "broadcast_state": SimpleNamespace(
            mode="per_soma_shuffled", readout_records=frozenset({0})
        ),
    }

    first = probe._compute_local_broadcast_error(**kwargs)
    second = probe._compute_local_broadcast_error(**kwargs)
    routed = first[:, ::2]

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first[:, 0::2], first[:, 1::2])
    torch.testing.assert_close(routed.sort(dim=1).values, delta.sort(dim=1).values)
    assert not bool((routed == delta).any())


def test_unset_broadcast_seed_follows_the_trainer_seed():
    def seed_of(trainer_seed, configured=None):
        probe = _BroadcastProbe()
        probe.local_cfg.broadcast_seed = configured
        probe.seed = trainer_seed
        return probe._broadcast_seed()

    assert seed_of(1, configured=17) == 17
    assert seed_of(1) == seed_of(1)
    assert seed_of(1) != seed_of(2)


def test_path_modes_reject_falling_back_to_the_scalar_error():
    import pytest

    probe = _BroadcastProbe()
    delta = torch.ones(2, 3)
    delta_scalar = delta.mean(dim=1, keepdim=True)
    soma = {"v_n": torch.zeros(2, 3)}
    for method in (
        probe._precompute_path_transport_errors,
        probe._precompute_feedback_seeds,
    ):
        with pytest.raises(ValueError, match="differs from the error width"):
            method(
                layer_records=[{"v_n": torch.zeros(2, 5)}],
                delta=delta,
                delta_scalar=delta_scalar,
            )
        with pytest.raises(ValueError, match="no parent branch block"):
            method(
                layer_records=[{"v_n": torch.zeros(2, 6)}, soma],
                delta=delta,
                delta_scalar=delta_scalar,
            )


def test_layers_outside_the_readout_tree_receive_the_scalar_error():
    probe = _BroadcastProbe()
    delta = torch.tensor([[1.0, 3.0]])
    for mode in ("per_soma", "per_soma_shared", "per_soma_shuffled"):
        out = probe._compute_local_broadcast_error(
            rec={},
            layer_idx=1,
            out_features=2,
            v_n=torch.zeros(1, 2),
            delta=delta,
            delta_scalar=delta.mean(dim=1, keepdim=True),
            broadcast_state=SimpleNamespace(mode=mode, readout_records=frozenset({0})),
        )
        torch.testing.assert_close(out, torch.full((1, 2), 2.0))


def test_path_modes_use_the_scalar_error_when_the_fallback_is_allowed():
    probe = _BroadcastProbe()
    probe.local_cfg.allow_scalar_fallback = True
    delta = torch.ones(2, 3)
    delta_scalar = delta.mean(dim=1, keepdim=True) * 2.0
    soma = {"v_n": torch.zeros(2, 3)}
    for method in (
        probe._precompute_path_transport_errors,
        probe._precompute_feedback_seeds,
    ):
        seeds = method(
            layer_records=[{"v_n": torch.zeros(2, 5)}],
            delta=delta,
            delta_scalar=delta_scalar,
        )
        torch.testing.assert_close(seeds[0], torch.full((2, 5), 2.0))
        seeds = method(
            layer_records=[{"v_n": torch.zeros(2, 6)}, soma],
            delta=delta,
            delta_scalar=delta_scalar,
        )
        torch.testing.assert_close(seeds[0], torch.full((2, 6), 2.0))
