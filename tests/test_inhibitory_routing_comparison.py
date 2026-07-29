"""Tests for fixed-budget inhibitory-routing comparisons."""

import pandas as pd
import pytest

from dendritic_modeling.analysis.tools.inhibitory_routing_comparison import (
    fixed_budget_inhibitory_routing_comparison,
)


def _records() -> pd.DataFrame:
    rows = []
    for route in ("direct", "population"):
        for use_shunting in (False, True):
            for n_e, n_i in ((2, 8), (8, 2)):
                for seed in range(3):
                    base = 0.40 + 0.001 * seed
                    if route == "population":
                        base += 0.05 if not use_shunting else 0.02
                    if use_shunting:
                        base += 0.10
                    rows.append(
                        {
                            "dataset": "toy",
                            "seed": seed,
                            "inhibitory_routing": route,
                            "use_shunting": use_shunting,
                            "ee_value": n_e,
                            "ie_value": n_i,
                            "test_accuracy": base,
                        }
                    )
    return pd.DataFrame(rows)


def test_fixed_budget_routing_comparison_pairs_seed_level_effects():
    result = fixed_budget_inhibitory_routing_comparison(
        _records(),
        budget=10,
        expected_allocations=[(2, 8), (8, 2)],
        expected_routes=["direct", "population"],
        route_labels={"direct": "Direct", "population": "Population"},
        expected_seeds_per_cell=3,
        draws=500,
        bootstrap_seed=4,
        dataset="toy",
    )

    assert len(result.per_seed) == 24
    assert len(result.accuracy_summary) == 8
    mechanism = result.mechanism_paired_summary.set_index(["inhibitory_routing", "n_e"])
    assert mechanism.loc[("direct", 2), "mean"] == pytest.approx(0.10)
    assert mechanism.loc[("population", 2), "mean"] == pytest.approx(0.07)
    routing = result.routing_paired_summary.set_index(["use_shunting", "n_e"])
    assert routing.loc[(False, 2), "mean"] == pytest.approx(0.05)
    assert routing.loc[(True, 2), "mean"] == pytest.approx(0.02)


def test_fixed_budget_routing_comparison_rejects_incomplete_factorial():
    with pytest.raises(ValueError, match="Incomplete routing factorial"):
        fixed_budget_inhibitory_routing_comparison(
            _records().iloc[:-1],
            budget=10,
            expected_allocations=[(2, 8), (8, 2)],
            expected_routes=["direct", "population"],
            route_labels={"direct": "Direct", "population": "Population"},
            expected_seeds_per_cell=3,
            draws=100,
            bootstrap_seed=4,
            dataset="toy",
        )
