"""Unit tests for the extracted pure STDP helpers (local credit assignment)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_pathways import (
    STDP_TOPK_PATHS,
)
from dendritic_modeling.training.strategies.local_learning_stdp import (
    stdp_activity,
    stdp_decay,
    stdp_enabled,
    stdp_pathway_enabled,
)


def test_stdp_decay_matches_exp_and_clamps_tau():
    assert stdp_decay(2.0) == math.exp(-1.0 / 2.0)
    # tau is clamped to >= 1e-6 (no div-by-zero / overflow)
    assert 0.0 <= stdp_decay(0.0) <= 1.0
    assert stdp_decay(1e9) > 0.999


def test_stdp_activity_modes_and_detach():
    vals = torch.tensor([[-1.0, 0.5, 2.0]], requires_grad=True)
    relu = stdp_activity(vals, threshold=0.0, mode="relu")
    assert not relu.requires_grad  # detached
    torch.testing.assert_close(relu, torch.tensor([[0.0, 0.5, 2.0]]))
    torch.testing.assert_close(
        stdp_activity(vals, 0.0, "identity"), torch.tensor([[-1.0, 0.5, 2.0]])
    )
    torch.testing.assert_close(
        stdp_activity(vals, 0.0, "binary"), torch.tensor([[0.0, 1.0, 1.0]])
    )
    torch.testing.assert_close(
        stdp_activity(vals, 0.0, "abs"), torch.tensor([[1.0, 0.5, 2.0]])
    )
    # unknown mode falls back to relu (with a warning)
    torch.testing.assert_close(
        stdp_activity(vals, 0.0, "bogus"), torch.tensor([[0.0, 0.5, 2.0]])
    )


def test_stdp_activity_reshapes_non_2d_and_centers_threshold():
    vals = torch.arange(8.0).reshape(2, 2, 2)
    out = stdp_activity(vals, threshold=1.0, mode="identity")
    assert out.shape == (2, 4)
    torch.testing.assert_close(out[0], torch.tensor([-1.0, 0.0, 1.0, 2.0]))


def test_stdp_enabled_predicate():
    assert stdp_enabled(
        SimpleNamespace(stdp=SimpleNamespace(enabled=True), rule_variant="x")
    )
    assert stdp_enabled(SimpleNamespace(stdp=None, rule_variant="STDP"))
    assert not stdp_enabled(
        SimpleNamespace(stdp=SimpleNamespace(enabled=False), rule_variant="local_ca")
    )


def test_stdp_pathway_enabled_predicate():
    cfg = SimpleNamespace(stdp=SimpleNamespace(apply_to=("exc", "rec-exc")))
    assert stdp_pathway_enabled(cfg, "exc")
    assert stdp_pathway_enabled(cfg, "rec_exc")  # hyphen normalized
    assert not stdp_pathway_enabled(cfg, "inh")
    assert stdp_pathway_enabled(
        SimpleNamespace(stdp=SimpleNamespace(apply_to="all")), "inh"
    )
    # default apply_to when unset
    assert stdp_pathway_enabled(SimpleNamespace(stdp=None), "exc")


def test_stdp_path_descriptors_preserve_dispatch_order():
    assert [
        (path.pathway, path.module_key, path.input_key, path.mask_key)
        for path in STDP_TOPK_PATHS
    ] == [
        ("exc", "exc_module", "x_exc", "exc_mask"),
        ("inh", "inh_module", "x_inh", "inh_mask"),
        ("rec_exc", "rec_exc_module", "x_rec_exc", "rec_exc_mask"),
        ("rec_inh", "rec_inh_module", "x_rec_inh", "rec_inh_mask"),
    ]
