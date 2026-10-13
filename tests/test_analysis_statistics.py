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


def test_stable_seed_prints_numpy_scalars_as_numpy_1_does() -> None:
    # Grouped pandas keys are NumPy scalars, which NumPy 2 prints with their
    # type (np.int64(0)); the seed must not depend on the NumPy version.
    python_key = ("additive", 0, 1, True, 0.1, 0.1)
    numpy_key = (
        "additive",
        np.int64(0),
        np.int32(1),
        np.bool_(True),
        np.float64(0.1),
        np.float32(0.1),
    )
    for representation in ("repr", "pipe"):
        assert stable_seed(7, *numpy_key, representation=representation) == stable_seed(
            7, *python_key, representation=representation
        )

    # A bootstrap seed of a published summary (the empirical manuscript's
    # source-activity group_summary.csv of 2026-08-12, written under NumPy 1).
    key = ("additive_occupancy_2x2", "additive", 0, 1)
    key += ("ff_excitatory", "excitatory", "image_pixels")
    numpy_key = tuple(np.int64(v) if isinstance(v, int) else v for v in key)
    for identity in (key, numpy_key):
        seed = stable_seed(20260812, "checkpoint-metric-summary", *identity)
        assert seed == 5247841351293295840


def test_stable_seed_ignores_print_options() -> None:
    identity = ("shunting", np.int64(3), np.float64(0.1))
    expected = stable_seed(42, *identity)
    with np.printoptions(legacy="1.13", precision=2, sign="+"):
        assert stable_seed(42, *identity) == expected
        assert repr(np.float64(0.1)) != "0.1"  # the caller's options still apply
