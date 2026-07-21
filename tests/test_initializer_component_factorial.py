from __future__ import annotations

import pandas as pd
import pytest

from dendritic_modeling.scripts.sweeps import (
    summarize_initializer_components as summary,
)


def test_initializer_policies_define_a_complete_two_by_two_factorial():
    assert summary.POLICY_COMPONENTS == {
        "analytical": ("analytical", "analytical"),
        "analytical_slope_occupancy_center": ("analytical", "occupancy"),
        "occupancy_slope_analytical_center": ("occupancy", "analytical"),
        "occupancy_quantile": ("occupancy", "occupancy"),
    }


def _synthetic_runs() -> pd.DataFrame:
    values = {
        ("additive", "analytical"): 0.70,
        ("additive", "analytical_slope_occupancy_center"): 0.86,
        ("additive", "occupancy_slope_analytical_center"): 0.74,
        ("additive", "occupancy_quantile"): 0.90,
        ("shunting", "analytical"): 0.93,
        ("shunting", "analytical_slope_occupancy_center"): 0.92,
        ("shunting", "occupancy_slope_analytical_center"): 0.88,
        ("shunting", "occupancy_quantile"): 0.87,
    }
    rows = []
    config_id = 0
    for (mechanism, policy), value in values.items():
        slope, center = summary.POLICY_COMPONENTS[policy]
        for seed in (1, 2):
            rows.append(
                {
                    "config_id": config_id,
                    "mechanism": mechanism,
                    "policy": policy,
                    "slope_source": slope,
                    "center_source": center,
                    "valid_accuracy": value,
                    "valid_categorical_loglikelihood": value - 1.0,
                    "seed": seed,
                    "condition": "c1",
                    "calibration_converged": True,
                    "calibration_iterations": 2,
                    "calibration_reverted_layers": 0,
                    "occupancy_fit_reverted_layers": 0,
                }
            )
            config_id += 1
    return pd.DataFrame(rows)


def test_component_selector_and_contrasts_are_comparator_explicit():
    recipes = summary.summarize(_synthetic_runs())
    selected = summary.select_per_mechanism(recipes).set_index("mechanism")
    shared = summary.select_shared_policy(recipes).iloc[0]
    effects = summary.component_effects(recipes).set_index(["mechanism", "metric"])

    assert selected.loc["additive", "policy"] == "occupancy_quantile"
    assert selected.loc["shunting", "policy"] == "analytical"
    assert shared["policy"] == "analytical_slope_occupancy_center"
    additive = effects.loc[("additive", "valid_accuracy")]
    assert additive["center_effect_at_analytical_slope"] == pytest.approx(0.16)
    assert additive["slope_effect_at_analytical_center"] == pytest.approx(0.04)
    assert additive["center_by_slope_interaction"] == pytest.approx(0.0)
