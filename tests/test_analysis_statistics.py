"""Tests for shared seed-level inferential summaries."""

from __future__ import annotations

import numpy as np
import pytest

from dendritic_modeling.analysis.statistics import bootstrap_mean_interval, stable_seed


def test_bootstrap_mean_interval_matches_explicit_numpy_resampling() -> None:
    values = np.asarray([0.1, 0.2, 0.4, 0.8, 0.9], dtype=float)
    draws = 2_000
    seed = 713
    rng = np.random.default_rng(seed)
    expected = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(
        axis=1
    )

    summary = bootstrap_mean_interval(
        values,
        draws=draws,
        seed=seed,
        require_n=5,
    )

    assert summary.mean == pytest.approx(values.mean())
    assert summary.standard_deviation == pytest.approx(values.std(ddof=1))
    assert summary.lower == pytest.approx(np.quantile(expected, 0.025))
    assert summary.upper == pytest.approx(np.quantile(expected, 0.975))
    assert summary.n_units == 5


def test_bootstrap_mean_interval_fails_closed_on_unit_count_and_nonfinite() -> None:
    with pytest.raises(ValueError, match="Expected 10"):
        bootstrap_mean_interval([1.0, 2.0], draws=100, seed=1, require_n=10)
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean_interval([1.0, np.nan], draws=100, seed=1)


def test_stable_seed_is_identity_sensitive_and_supports_legacy_rules() -> None:
    first = stable_seed(42, ("shunting", 3), combine="xor")
    assert first == stable_seed(42, ("shunting", 3), combine="xor")
    assert first != stable_seed(42, ("additive", 3), combine="xor")
    assert (
        stable_seed(
            42,
            "shunting",
            3,
            combine="add",
            representation="pipe",
            digest_bytes=4,
        )
        != first
    )
