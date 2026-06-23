"""Unit tests for the extracted pure STDP helpers (local credit assignment)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_pathways import (
    RECORDER_TOPK_PATHS,
    STDP_TOPK_PATHS,
    iter_topk_path_modules,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_stdp import (
    stdp_activity,
    stdp_apply_error_modulation,
    stdp_apply_pathway_sign,
    stdp_clamp_update,
    stdp_decay,
    stdp_enabled,
    stdp_error_scale,
    stdp_next_trace,
    stdp_pathway_enabled,
    stdp_trace_like,
    stdp_weight_delta,
    weight_transform_derivative,
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


def test_stdp_trace_like_preserves_compatible_or_initializes_zero():
    activity = torch.ones(2, 3, dtype=torch.float64)
    previous = torch.arange(3, dtype=torch.float64)

    assert stdp_trace_like(previous, features=3, activity=activity) is previous

    initialized = stdp_trace_like(
        torch.ones(2, dtype=torch.float64),
        features=3,
        activity=activity,
    )

    torch.testing.assert_close(initialized, torch.zeros(3, dtype=torch.float64))
    assert initialized.device == activity.device
    assert initialized.dtype == activity.dtype


def test_stdp_weight_delta_and_pathway_sign_match_formula():
    pre_batch = torch.tensor([2.0, 3.0])
    post_batch = torch.tensor([0.5, 1.5])
    prev_pre = torch.tensor([0.25, 0.75])
    prev_post = torch.tensor([1.0, 2.0])

    delta = stdp_weight_delta(
        pre_batch=pre_batch,
        post_batch=post_batch,
        prev_pre=prev_pre,
        prev_post=prev_post,
        a_plus=1.2,
        a_minus=0.4,
    )
    expected = (
        1.2 * post_batch[:, None] * prev_pre[None, :]
        - 0.4 * prev_post[:, None] * pre_batch[None, :]
    )

    torch.testing.assert_close(delta, expected)
    assert (
        stdp_apply_pathway_sign(delta, pathway="exc", inhibitory_update_sign=-2.0)
        is delta
    )
    torch.testing.assert_close(
        stdp_apply_pathway_sign(delta, pathway="inh", inhibitory_update_sign=-2.0),
        delta * -2.0,
    )


def test_stdp_error_modulation_and_clamp_match_formula():
    delta = torch.tensor([[2.0, -3.0]])
    error = torch.tensor([[1.0, -2.0]])

    torch.testing.assert_close(
        stdp_error_scale(error, mode="scalar_signed"),
        error.mean(),
    )
    torch.testing.assert_close(
        stdp_error_scale(error, mode="scalar_abs"),
        error.abs().mean(),
    )
    torch.testing.assert_close(
        stdp_apply_error_modulation(
            delta,
            error_signal=error,
            mode="scalar_signed",
        ),
        delta * error.mean(),
    )
    torch.testing.assert_close(
        stdp_apply_error_modulation(
            delta,
            error_signal=error,
            mode="scalar_abs",
        ),
        delta * error.abs().mean(),
    )
    assert (
        stdp_apply_error_modulation(delta, error_signal=None, mode="scalar_abs")
        is delta
    )
    torch.testing.assert_close(
        stdp_clamp_update(delta, 1.5),
        torch.tensor([[1.5, -1.5]]),
    )
    assert stdp_clamp_update(delta, 0.0) is delta


def test_stdp_next_trace_preserves_decay_and_detach_modes():
    previous = torch.tensor([1.0, 2.0], requires_grad=True)
    batch = torch.tensor([3.0, 4.0])
    decay = stdp_decay(2.0)

    actual = stdp_next_trace(previous, batch, tau=2.0, detach=False)
    expected = decay * previous + (1.0 - decay) * batch

    torch.testing.assert_close(actual, expected)
    assert actual.requires_grad

    detached = stdp_next_trace(previous, batch, tau=2.0, detach=True)
    torch.testing.assert_close(detached, expected.detach())
    assert not detached.requires_grad


def test_weight_transform_derivative_modes():
    raw = torch.tensor([[-1.0, 0.0, 2.0]])

    torch.testing.assert_close(weight_transform_derivative(raw, "exp"), raw.exp())
    torch.testing.assert_close(
        weight_transform_derivative(raw, "relu"),
        torch.tensor([[0.0, 0.0, 1.0]]),
    )
    torch.testing.assert_close(
        weight_transform_derivative(raw, "softplus"),
        torch.sigmoid(raw),
    )
    torch.testing.assert_close(
        weight_transform_derivative(raw, "identity"),
        torch.ones_like(raw),
    )


def test_weight_transform_derivative_unknown_falls_back_to_exp_and_warns():
    raw = torch.tensor([[-1.0, 0.0, 2.0]])
    warnings = []

    out = weight_transform_derivative(
        raw,
        "unknown",
        warn_unknown=warnings.append,
    )

    torch.testing.assert_close(out, raw.exp())
    assert warnings == ["unknown"]


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


def test_recorder_path_descriptors_preserve_record_keys():
    assert [
        (
            path.pathway,
            path.module_attr,
            path.module_key,
            path.input_key,
            path.output_key,
            path.mask_key,
        )
        for path in RECORDER_TOPK_PATHS
    ] == [
        (
            "exc",
            "branch_excitation",
            "exc_module",
            "x_exc",
            "exc_out",
            "exc_mask",
        ),
        (
            "inh",
            "branch_inhibition",
            "inh_module",
            "x_inh",
            "inh_out",
            "inh_mask",
        ),
        (
            "rec_exc",
            "branch_recurrent",
            "rec_exc_module",
            "x_rec_exc",
            "rec_exc_out",
            "rec_exc_mask",
        ),
        (
            "rec_inh",
            "branch_rec_inhibition",
            "rec_inh_module",
            "x_rec_inh",
            "rec_inh_out",
            "rec_inh_mask",
        ),
    ]


def test_iter_topk_path_modules_preserves_descriptor_order():
    exc = object()
    rec_exc = object()
    rec_inh = object()
    module = SimpleNamespace(
        branch_excitation=exc,
        branch_inhibition=None,
        branch_recurrent=rec_exc,
        branch_rec_inhibition=rec_inh,
    )

    assert [
        (path.pathway, topk_module)
        for path, topk_module in iter_topk_path_modules(module)
    ] == [
        ("exc", exc),
        ("rec_exc", rec_exc),
        ("rec_inh", rec_inh),
    ]
