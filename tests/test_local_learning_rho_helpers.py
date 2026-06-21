"""Tests for rho helper functions used by local learning."""

from dendritic_modeling.training.strategies.local_learning_rho_mixin import (
    _clamp_rho,
)


def test_clamp_rho_preserves_legacy_stability_range():
    assert _clamp_rho(-1.0) == 0.1
    assert _clamp_rho(0.1) == 0.1
    assert _clamp_rho(1.25) == 1.25
    assert _clamp_rho(2.0) == 2.0
    assert _clamp_rho(5.0) == 2.0
