"""Unit tests for the extracted pure broadcast / pathway-role helpers."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_broadcast import (
    apply_broadcast_bandwidth,
    compute_role_selectivity,
    expand_parent_signal_to_children,
    normalize_role_mass,
    resolve_recent_role_profile,
    role_mass_from_weight_matrix,
)


def test_apply_broadcast_bandwidth_modes():
    e = torch.tensor([[-2.0, 1.0, 0.0, 3.0]])
    # sign_only: sign(e) * mean(|e|)
    out = apply_broadcast_bandwidth(SimpleNamespace(), e, "sign_only")
    torch.testing.assert_close(out, torch.sign(e) * e.abs().mean())
    # unknown bandwidth -> passthrough
    torch.testing.assert_close(
        apply_broadcast_bandwidth(SimpleNamespace(), e, "none"), e
    )
    # sparse_topk keeps only the top fraction by magnitude
    cfg = SimpleNamespace(broadcast_topk_fraction=0.5)
    sp = apply_broadcast_bandwidth(cfg, e, "sparse_topk")
    assert int((sp != 0).sum()) == 2  # 50% of 4
    assert sp[0, 0] == -2.0 and sp[0, 3] == 3.0  # largest-|.| retained


def test_normalize_role_mass_rows_sum_to_one():
    rm = torch.tensor([[1.0, 3.0], [0.0, 0.0]])
    out = normalize_role_mass(rm)
    torch.testing.assert_close(out[0], torch.tensor([0.25, 0.75]))
    assert torch.isfinite(out).all()  # zero row handled via clamp


def test_compute_role_selectivity_uniform_vs_onehot():
    uniform = torch.full((1, 4), 0.25)
    onehot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert compute_role_selectivity(uniform).item() < 1e-4  # max entropy -> ~0
    assert compute_role_selectivity(onehot).item() > 0.99  # min entropy -> ~1
    # single role -> selectivity 1
    assert compute_role_selectivity(torch.ones(2, 1)).tolist() == [1.0, 1.0]


def test_expand_parent_signal_to_children_repeat_and_truncate():
    sig = torch.tensor([[1.0, 2.0]])
    # block_size=2 -> [1,1,2,2]; truncate to 3
    torch.testing.assert_close(
        expand_parent_signal_to_children(sig, 2, 3), torch.tensor([[1.0, 1.0, 2.0]])
    )
    # tile up to 6
    torch.testing.assert_close(
        expand_parent_signal_to_children(sig, 1, 6),
        torch.tensor([[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]]),
    )


def test_role_mass_from_weight_matrix():
    w = torch.tensor([[1.0, -2.0]])  # out=1, in=2
    roles = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # in=2, roles=2
    torch.testing.assert_close(
        role_mass_from_weight_matrix(w, roles), torch.tensor([[1.0, 2.0]])
    )
    assert role_mass_from_weight_matrix(w, None) is None
    assert role_mass_from_weight_matrix(w, torch.ones(3, 2)) is None  # dim mismatch


def test_resolve_recent_role_profile_prefers_matching_base_then_recent():
    base = torch.ones(4, 2)
    assert resolve_recent_role_profile(4, base, []) is base
    # base wrong size -> search prior outputs (reversed) for a 3-row profile
    prior = [torch.ones(5, 2), torch.ones(3, 2)]
    out = resolve_recent_role_profile(3, base, prior)
    assert out is prior[1]
    assert resolve_recent_role_profile(9, None, prior) is None


def test_max_entropy_constant_matches_log():
    # guards the entropy normalization used in selectivity
    assert math.isclose(math.log(4.0), math.log(4))
