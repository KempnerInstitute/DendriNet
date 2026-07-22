from __future__ import annotations

import numpy as np
import pytest

from dendritic_modeling.scripts.analysis.analyze_dendritic_rnn_causal_current import (
    CausalCurrentAnalysisError,
    configured_depth_selectivity,
    paired_bootstrap_interval,
)


def test_configured_depth_selectivity_uses_paired_extremes() -> None:
    profile = {
        0: np.array([3.0, 5.0]),
        1: np.array([100.0, 100.0]),
        2: np.array([1.0, 8.0]),
    }
    assert np.array_equal(
        configured_depth_selectivity(profile, placement="proximal"),
        np.array([2.0, -3.0]),
    )
    assert np.array_equal(
        configured_depth_selectivity(profile, placement="distal"),
        np.array([-2.0, 3.0]),
    )


def test_bootstrap_is_seed_deterministic_and_order_sensitive() -> None:
    values = np.array([-1.0, 0.0, 2.0, 4.0])
    first = paired_bootstrap_interval(
        values, draws=1000, confidence_level=0.95, seed=11
    )
    second = paired_bootstrap_interval(
        values, draws=1000, confidence_level=0.95, seed=11
    )
    assert first == second
    assert first[0] <= values.mean() <= first[1]


def test_selectivity_rejects_unknown_placement() -> None:
    profile = {depth: np.ones(2) for depth in (0, 1, 2)}
    with pytest.raises(CausalCurrentAnalysisError, match="Unknown placement"):
        configured_depth_selectivity(profile, placement="middle")
