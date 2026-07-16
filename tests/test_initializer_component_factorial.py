from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from dendritic_modeling.scripts.sweeps import (
    summarize_initializer_components as summary,
)

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    ("relative_path", "expected_count", "expected_seeds"),
    [
        (
            "drafts/dendritic-information-processing/empirical/configs/"
            "publication_v3/confirmatory/initialization/"
            "empirical_initializer_component_factorial_resource_matched_cifar10.yaml",
            96,
            3,
        ),
        (
            "drafts/dendritic-information-processing/neurips/configs/sweeps/"
            "neurips_v1_initializer_component_factorial_revision.yaml",
            96,
            2,
        ),
    ],
)
def test_component_sweeps_use_the_complete_symmetric_policy_set(
    relative_path, expected_count, expected_seeds
):
    document = yaml.safe_load((ROOT / relative_path).read_text(encoding="utf-8"))
    sweep = document["sweep_config"]
    policies = sweep[
        "model.core.population_network.layers.0.population_defaults.reactivation_init_policy"
    ]
    mechanisms = sweep[
        "model.core.population_network.layers.0.population_defaults.use_shunting"
    ]
    common = document["base_config"]["training"]["main"]["common"]

    assert document["sweep_contract"]["expected_config_count"] == expected_count
    assert document["experiment_settings"]["seeds_per_condition"] == expected_seeds
    assert set(policies) == set(summary.POLICY_COMPONENTS)
    assert mechanisms == [True, False]
    assert common["weight_decay_rate"] == pytest.approx(0.01)
    assert common["weight_boosting"] is True
    assert document["base_config"]["training"]["main"]["optimizer"][
        "weight_decay"
    ] == pytest.approx(0.0)


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
