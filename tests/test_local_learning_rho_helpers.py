"""Tests for rho helper functions used by local learning."""

from types import SimpleNamespace

import pytest
import torch

from dendritic_modeling.training.strategies.local_learning_rho_mixin import (
    LocalLearningRhoMixin,
    _augmented_rho_sample,
    _clamp_rho,
    _ema_update,
    _mean_activity_pair,
    _mean_augmented_activity_pair,
    _noisy_mean_samples,
    _pearson_proxy_from_samples,
    _update_single_sample_ema_rho,
)
from dendritic_modeling.training.strategies.local_learning_state import _LayerStats


class _RhoProbe(LocalLearningRhoMixin):
    """Minimal object that exposes the rho mixin state contract."""

    def __init__(self, four_factor):
        self.local_cfg = SimpleNamespace(four_factor=four_factor)
        self._layer_stats = {}


def _four_factor_cfg(**kwargs):
    defaults = {
        "rho_mode": "pearson",
        "rho_estimator": "ema",
        "ema_alpha": 0.2,
        "augment_k": 4,
        "augment_noise_sigma": 0.05,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_clamp_rho_preserves_legacy_stability_range():
    assert _clamp_rho(-1.0) == 0.1
    assert _clamp_rho(0.1) == 0.1
    assert _clamp_rho(1.25) == 1.25
    assert _clamp_rho(2.0) == 2.0
    assert _clamp_rho(5.0) == 2.0


def test_ema_update_preserves_initialization_and_update_rule():
    assert _ema_update(None, 1.5, alpha=0.2, decay=0.8) == 1.5
    assert _ema_update(1.0, 2.0, alpha=0.2, decay=0.8) == pytest.approx(1.2)


def test_mean_activity_pair_uses_per_sample_neuron_means():
    v_n = torch.tensor([[1.0, 3.0], [2.0, 6.0]])
    v0 = torch.tensor([[4.0, 8.0], [1.0, 5.0]])

    a, b = _mean_activity_pair(v_n, v0)

    torch.testing.assert_close(a, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(b, torch.tensor([6.0, 3.0]))


def test_pearson_proxy_from_samples_preserves_batch_formula():
    a = torch.tensor([2.0, 4.0, 6.0])
    b = torch.tensor([1.0, 5.0, 7.0])
    a_centered = a - a.mean()
    b_centered = b - b.mean()
    denom = a_centered.std(unbiased=False) * b_centered.std(unbiased=False) + 1e-8
    expected = float((a_centered * b_centered).mean().item() / denom.item())

    assert _pearson_proxy_from_samples(a, b) == expected


def test_noisy_mean_samples_preserves_rng_order():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    n_samples = 3
    noise_sigma = 0.1

    torch.manual_seed(7)
    actual_a, actual_b = _noisy_mean_samples(
        a,
        b,
        n_samples=n_samples,
        noise_sigma=noise_sigma,
    )

    torch.manual_seed(7)
    expected_a = []
    expected_b = []
    a_orig = a.clone()
    b_orig = b.clone()
    for _ in range(n_samples):
        noise_a = torch.randn_like(a_orig) * noise_sigma
        noise_b = torch.randn_like(b_orig) * noise_sigma
        expected_a.append((a_orig + noise_a).mean().item())
        expected_b.append((b_orig + noise_b).mean().item())

    assert actual_a == expected_a
    assert actual_b == expected_b


def test_mean_augmented_activity_pair_preserves_sample_average():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])

    torch.manual_seed(13)
    actual = _mean_augmented_activity_pair(
        a,
        b,
        n_samples=4,
        noise_sigma=0.05,
    )

    torch.manual_seed(13)
    a_samples, b_samples = _noisy_mean_samples(
        a,
        b,
        n_samples=4,
        noise_sigma=0.05,
    )

    assert actual == (sum(a_samples) / 4, sum(b_samples) / 4)


def test_augmented_rho_sample_preserves_micro_ensemble_formula():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])

    torch.manual_seed(23)
    actual = _augmented_rho_sample(
        a,
        b,
        n_samples=5,
        noise_sigma=0.1,
    )

    torch.manual_seed(23)
    a_samples, b_samples = _noisy_mean_samples(
        a,
        b,
        n_samples=5,
        noise_sigma=0.1,
    )
    expected = _pearson_proxy_from_samples(
        torch.tensor(a_samples),
        torch.tensor(b_samples),
    )

    assert actual == expected


def test_update_single_sample_ema_rho_preserves_initial_and_second_update():
    stats = _LayerStats()

    first = _update_single_sample_ema_rho(
        stats,
        a_sample=1.0,
        b_sample=2.0,
        alpha=0.2,
        decay=0.8,
    )
    assert first == 0.0
    assert stats.rho_ema == 0.0

    second = _update_single_sample_ema_rho(
        stats,
        a_sample=3.0,
        b_sample=5.0,
        alpha=0.2,
        decay=0.8,
    )
    expected_var_y = 0.2 * (3.0 - 1.0) ** 2
    expected_var_x = 0.2 * (5.0 - 2.0) ** 2
    expected_cov = 0.2 * (3.0 - 1.0) * (5.0 - 2.0)
    expected = expected_cov / ((expected_var_y * expected_var_x) ** 0.5 + 1e-8)

    assert second == pytest.approx(expected)
    assert stats.rho_ema == pytest.approx(expected)


def test_compute_layer_rho_dot_mode_preserves_ema_clamp():
    probe = _RhoProbe(_four_factor_cfg(rho_mode="dot", ema_alpha=0.25))
    rec = {"layer": object(), "v_n": torch.tensor([[1.0, 3.0], [2.0, 4.0]])}
    v0 = torch.tensor([[2.0, 4.0], [3.0, 5.0]])

    first = probe._compute_layer_rho(rec, v0)
    second = probe._compute_layer_rho(rec, v0)

    expected_sample = float((rec["v_n"].mean(dim=1) * v0.mean(dim=1)).mean().item())
    assert first == _clamp_rho(expected_sample)
    assert second == _clamp_rho(expected_sample)


def test_compute_layer_rho_batch_pearson_preserves_ema_update():
    probe = _RhoProbe(_four_factor_cfg(rho_mode="pearson", ema_alpha=0.25))
    rec = {"layer": object(), "v_n": torch.tensor([[1.0, 3.0], [2.0, 8.0]])}
    v0 = torch.tensor([[2.0, 4.0], [3.0, 9.0]])
    a, b = _mean_activity_pair(rec["v_n"], v0)
    rho_sample = _pearson_proxy_from_samples(a, b)

    first = probe._compute_layer_rho(rec, v0)
    second = probe._compute_layer_rho(rec, v0 * 2.0)

    a2, b2 = _mean_activity_pair(rec["v_n"], v0 * 2.0)
    rho_sample_2 = _pearson_proxy_from_samples(a2, b2)
    expected_second = 0.75 * rho_sample + 0.25 * rho_sample_2

    assert first == _clamp_rho(rho_sample)
    assert second == _clamp_rho(expected_second)
