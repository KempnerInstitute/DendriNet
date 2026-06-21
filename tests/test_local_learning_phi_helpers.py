"""Tests for phi helper functions used by local learning."""

from types import SimpleNamespace

import pytest
import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_phi_mixin import (
    LocalLearningPhiMixin,
    _center_for_conditional_phi,
    _conditional_phi_from_moments,
    _mean_scalar_pair,
    _resolve_parent_proxy,
    _update_conditional_phi_moments,
    _update_scalar_ema,
    _update_single_sample_conditional_moments,
    _update_variance_ratio_phi,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerStats,
)


class _PhiProbe(LocalLearningPhiMixin):
    """Minimal object that exposes the phi mixin state contract."""

    def __init__(self, *, ema_alpha: float = 0.2, estimator: str = "conditional_ema"):
        self._layer_stats = {}
        self.local_cfg = SimpleNamespace(
            four_factor=SimpleNamespace(ema_alpha=ema_alpha),
            five_factor=SimpleNamespace(
                phi_estimator=estimator,
                phi_ridge_lambda=0.5,
                phi_clamp_min=0.0,
                phi_clamp_max=100.0,
            ),
        )


def test_update_scalar_ema_preserves_initialization_and_update_rule():
    assert _update_scalar_ema(None, 2.0, alpha=0.25) == 2.0
    assert _update_scalar_ema(2.0, 6.0, alpha=0.25) == pytest.approx(3.0)


def test_update_variance_ratio_phi_preserves_ema_formula():
    stats = _LayerStats()

    first = _update_variance_ratio_phi(stats, var_batch=4.0, alpha=0.25)
    second = _update_variance_ratio_phi(stats, var_batch=8.0, alpha=0.25)

    assert first == pytest.approx(4.0 / (4.0 + 1e-8))
    expected_ema = 0.75 * 4.0 + 0.25 * 8.0
    assert stats.var_ema == pytest.approx(expected_ema)
    assert second == pytest.approx(8.0 / (expected_ema + 1e-8))


def test_resolve_parent_proxy_preserves_blk_exc_inh_priority():
    blk = torch.tensor([1.0])
    exc = torch.tensor([2.0])
    inh = torch.tensor([3.0])

    assert (
        _resolve_parent_proxy({"blk_out": blk, "exc_out": exc, "inh_out": inh}) is blk
    )
    assert _resolve_parent_proxy({"exc_out": exc, "inh_out": inh}) is exc
    assert _resolve_parent_proxy({"inh_out": inh}) is inh
    assert _resolve_parent_proxy({}) is None


def test_center_for_conditional_phi_preserves_batch_centering():
    v_n = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    parent = torch.tensor([[2.0, 4.0], [8.0, 10.0]])

    y, x = _center_for_conditional_phi(v_n, parent)

    torch.testing.assert_close(y, v_n - v_n.mean(dim=0, keepdim=True))
    torch.testing.assert_close(x, parent - parent.mean(dim=0, keepdim=True))


def test_conditional_phi_from_moments_preserves_variance_floor_and_formula():
    assert (
        _conditional_phi_from_moments(
            var_y=2.0,
            var_x=1e-13,
            cov_xy=1.0,
            ridge_lambda=0.5,
        )
        == 1.0
    )

    actual = _conditional_phi_from_moments(
        var_y=3.0,
        var_x=2.0,
        cov_xy=1.5,
        ridge_lambda=0.5,
    )
    beta = 1.5 / (2.0 + 0.5)
    residual_var = 3.0 - beta * 1.5
    expected = 3.0 / residual_var
    assert actual == pytest.approx(expected)

    floored = _conditional_phi_from_moments(
        var_y=1.0,
        var_x=1.0,
        cov_xy=2.0,
        ridge_lambda=0.0,
    )
    assert floored == pytest.approx(1.0 / 1e-12)


def test_update_conditional_phi_moments_preserves_ema_formula():
    stats = _LayerStats()
    stats.ema_var_y = 10.0
    stats.ema_var_x = 20.0
    stats.ema_cov_xy = 30.0
    y = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    x = torch.tensor([[3.0, -3.0], [4.0, -4.0]])

    var_y, var_x, cov_xy = _update_conditional_phi_moments(
        stats,
        y=y,
        x=x,
        alpha=0.25,
    )

    assert var_y == pytest.approx(0.75 * 10.0 + 0.25 * y.pow(2).mean().item())
    assert var_x == pytest.approx(0.75 * 20.0 + 0.25 * x.pow(2).mean().item())
    assert cov_xy == pytest.approx(0.75 * 30.0 + 0.25 * (y * x).mean().item())
    assert stats.ema_var_y == var_y
    assert stats.ema_var_x == var_x
    assert stats.ema_cov_xy == cov_xy


def test_mean_scalar_pair_uses_global_means():
    v_n = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    parent = torch.tensor([[2.0, 4.0], [8.0, 10.0]])

    assert _mean_scalar_pair(v_n, parent) == (v_n.mean().item(), parent.mean().item())


def test_update_single_sample_conditional_moments_preserves_initial_then_update():
    stats = _LayerStats()

    initialized = _update_single_sample_conditional_moments(
        stats,
        y_sample=1.0,
        x_sample=2.0,
        alpha=0.25,
    )
    assert initialized is False
    assert stats.ema_mean_y == 1.0
    assert stats.ema_mean_x == 2.0
    assert stats.ema_var_y == 0.0
    assert stats.ema_var_x == 0.0
    assert stats.ema_cov_xy == 0.0

    updated = _update_single_sample_conditional_moments(
        stats,
        y_sample=5.0,
        x_sample=8.0,
        alpha=0.25,
    )
    assert updated is True
    assert stats.ema_mean_y == pytest.approx(0.75 * 1.0 + 0.25 * 5.0)
    assert stats.ema_mean_x == pytest.approx(0.75 * 2.0 + 0.25 * 8.0)
    assert stats.ema_var_y == pytest.approx(0.25 * (5.0 - 1.0) ** 2)
    assert stats.ema_var_x == pytest.approx(0.25 * (8.0 - 2.0) ** 2)
    assert stats.ema_cov_xy == pytest.approx(0.25 * (5.0 - 1.0) * (8.0 - 2.0))


def test_compute_layer_phi_preserves_variance_ratio_ema():
    probe = _PhiProbe(ema_alpha=0.25)
    rec = {"layer": object(), "v_n": torch.tensor([[1.0, 3.0], [5.0, 7.0]])}
    var_batch = float(rec["v_n"].var(unbiased=False).item())

    first = probe._compute_layer_phi(rec)
    second = probe._compute_layer_phi({"layer": rec["layer"], "v_n": rec["v_n"] * 2.0})

    var_batch_2 = float((rec["v_n"] * 2.0).var(unbiased=False).item())
    expected_ema_2 = 0.75 * var_batch + 0.25 * var_batch_2
    assert first == pytest.approx(var_batch / (var_batch + 1e-8))
    assert second == pytest.approx(var_batch_2 / (expected_ema_2 + 1e-8))


def test_compute_conditional_phi_preserves_parent_moment_formula():
    probe = _PhiProbe(ema_alpha=0.25)
    v_n = torch.tensor([[1.0, 3.0], [5.0, 7.0]])
    parent = torch.tensor([[2.0, 4.0], [8.0, 10.0]])
    rec = {"layer": object(), "v_n": v_n, "blk_out": parent}
    y, x = _center_for_conditional_phi(v_n, parent)

    actual = probe._compute_layer_phi_conditional(rec)

    var_y = 0.25 * float(y.pow(2).mean().item())
    var_x = 0.25 * float(x.pow(2).mean().item())
    cov_xy = 0.25 * float((y * x).mean().item())
    expected = _conditional_phi_from_moments(
        var_y=var_y,
        var_x=var_x,
        cov_xy=cov_xy,
        ridge_lambda=0.5,
    )
    assert actual == pytest.approx(expected)


def test_compute_conditional_phi_ema_preserves_first_sample_neutrality():
    probe = _PhiProbe(ema_alpha=0.25)
    layer = object()
    first = {
        "layer": layer,
        "v_n": torch.tensor([[1.0, 3.0]]),
        "blk_out": torch.tensor([[2.0, 4.0]]),
    }
    second = {
        "layer": layer,
        "v_n": torch.tensor([[5.0, 7.0]]),
        "blk_out": torch.tensor([[8.0, 10.0]]),
    }

    assert probe._compute_layer_phi_conditional(first) == 1.0
    actual = probe._compute_layer_phi_conditional(second)

    stats = probe._layer_stats[id(layer)]
    expected = _conditional_phi_from_moments(
        var_y=stats.ema_var_y,
        var_x=stats.ema_var_x,
        cov_xy=stats.ema_cov_xy,
        ridge_lambda=0.5,
    )
    assert actual == pytest.approx(expected)
