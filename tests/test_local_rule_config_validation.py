"""LocalRuleConfig accepts only the options the local-rule trainer implements."""

from __future__ import annotations

import dataclasses

import pytest

from dendritic_modeling.config.local_learning import build_local_rule_config
from dendritic_modeling.config.training import (
    LOCAL_BROADCAST_BANDWIDTHS,
    LOCAL_DECODER_UPDATE_MODES,
    LOCAL_ENCODER_UPDATE_MODES,
    LOCAL_ERROR_BROADCAST_MODES,
    LOCAL_ERROR_MODES,
    LocalRuleConfig,
)

NESTED_CHOICES = {
    ("three_factor", "dynamics_mode"): ("auto", "conductance", "additive"),
    ("three_factor", "additive_gain_mode"): (
        "none",
        "input_dependent",
        "running_stats",
    ),
    ("four_factor", "rho_mode"): ("pearson", "dot", "none"),
    ("four_factor", "rho_estimator"): ("ema", "augment", "ema+augment"),
    ("five_factor", "phi_mode"): ("conditional", "variance"),
    ("five_factor", "phi_estimator"): ("conditional_ema", "variance_ema"),
    ("morphology_aware", "path_factor_mode"): ("per_branch", "scalar_mean"),
    ("morphology_aware", "morphology_modulator_mode"): ("none", "depth", "centrality"),
    ("hsic", "target_source"): ("labels", "logits"),
    ("hsic", "kernel"): ("linear", "rbf", "polynomial"),
    ("stdp", "activity_mode"): ("relu", "identity", "binary", "abs"),
    ("stdp", "error_modulation_mode"): ("scalar_abs", "scalar_signed"),
    ("inhibitory_homeostasis", "mode"): ("r_tot", "voltage"),
}
TOP_CHOICES = {
    "rule_variant": ("3f", "4f", "5f", "stdp"),
    "error_mode": LOCAL_ERROR_MODES,
    "error_broadcast_mode": LOCAL_ERROR_BROADCAST_MODES,
    "broadcast_bandwidth": LOCAL_BROADCAST_BANDWIDTHS,
    "encoder_update_mode": LOCAL_ENCODER_UPDATE_MODES,
    "decoder_update_mode": LOCAL_DECODER_UPDATE_MODES,
}


@pytest.mark.parametrize(
    "field, value",
    [(field, value) for field, values in TOP_CHOICES.items() for value in values],
)
def test_every_documented_top_level_value_is_accepted(field, value):
    config = build_local_rule_config({field: value.upper()})
    assert getattr(config, field) == value


@pytest.mark.parametrize(
    "section, field, value",
    [
        (section, field, value)
        for (section, field), values in NESTED_CHOICES.items()
        for value in values
    ],
)
def test_every_documented_nested_value_is_accepted(section, field, value):
    config = build_local_rule_config({section: {field: value}})
    assert getattr(getattr(config, section), field) == value


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"rule_variant": "4fac"}, "rule_variant='4fac' is not supported"),
        ({"rule_variant": "hebbian"}, "rule_variant='hebbian' is not supported"),
        ({"error_mode": "cross_entropy"}, "error_mode="),
        ({"error_broadcast_mode": "per_som"}, "did you mean 'per_soma'"),
        ({"broadcast_bandwidth": "sign"}, "broadcast_bandwidth="),
        ({"decoder_update_mode": "back_prop"}, "did you mean 'backprop'"),
        ({"encoder_update_mode": "local"}, "encoder_update_mode="),
        ({"four_factor": {"rho_mode": "pearsn"}}, "did you mean 'pearson'"),
        ({"five_factor": {"phi_estimator": "conditional"}}, "phi_estimator="),
        ({"hsic": {"kernel": "laplace"}}, "hsic.kernel="),
        ({"stdp": {"apply_to": ["exc", "dendrite"]}}, "stdp.apply_to="),
        ({"inhibitory_homeostasis": {"mode": "rtot"}}, "did you mean 'r_tot'"),
    ],
)
def test_unknown_values_raise(payload, message):
    with pytest.raises(ValueError, match=message):
        build_local_rule_config(payload)


def test_aliases_map_to_canonical_values():
    config = build_local_rule_config(
        {
            "rule_variant": "trace-stdp",
            "three_factor": {"dynamics_mode": "legacy"},
            "hsic": {"kernel": "gaussian"},
            "stdp": {"activity_mode": "absolute"},
        }
    )
    assert config.rule_variant == "stdp" and config.stdp.enabled
    assert config.three_factor.dynamics_mode == "conductance"
    assert config.hsic.kernel == "rbf"
    assert config.stdp.activity_mode == "abs"


@pytest.mark.parametrize("alias", ["3f_vh", "4f+vh", "5f-vh"])
def test_voltage_homeostasis_alias_is_the_same_on_both_paths(alias):
    direct = LocalRuleConfig(rule_variant=alias)
    loaded = build_local_rule_config({"rule_variant": alias})
    assert dataclasses.asdict(direct) == dataclasses.asdict(loaded)
    assert direct.rule_variant == alias[:2]
    assert direct.voltage_homeostasis.enabled
    assert direct.voltage_homeostasis.weight == 0.02


def test_explicit_voltage_homeostasis_settings_survive_the_alias():
    config = build_local_rule_config(
        {"rule_variant": "5f_vh", "voltage_homeostasis": {"weight": 0.1}}
    )
    assert config.voltage_homeostasis.enabled
    assert config.voltage_homeostasis.weight == 0.1


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"broadcast_bits": 0}, "broadcast_bits"),
        ({"broadcast_topk_fraction": 1.5}, "broadcast_topk_fraction"),
        ({"broadcast_topk_fraction": 0.0}, "broadcast_topk_fraction"),
        ({"broadcast_rank": 0}, "broadcast_rank"),
    ],
)
def test_numeric_broadcast_options_are_range_checked(payload, message):
    with pytest.raises(ValueError, match=message):
        build_local_rule_config(payload)


def test_training_schedule_entries_are_validated():
    config = build_local_rule_config(
        {
            "training_schedule": [
                {"start_epoch": 1, "end_epoch": 3, "decoder_update_mode": "Local"},
                {"start_epoch": 4, "encoder_update_mode": "BACKPROP"},
            ]
        }
    )
    assert config.training_schedule[0]["decoder_update_mode"] == "local"
    assert config.training_schedule[1]["encoder_update_mode"] == "backprop"

    for entry, message in (
        ({"start_epoch": 1, "decoder_mode": "local"}, "decoder_mode"),
        ({"start_epoch": "first"}, "must be an integer"),
        ({"start_epoch": 5, "end_epoch": 2}, "before start_epoch"),
        ({"decoder_update_mode": "frozen"}, "decoder_update_mode="),
    ):
        with pytest.raises(ValueError, match=message):
            build_local_rule_config({"training_schedule": [entry]})


def test_round_trip_through_asdict_is_identical():
    config = build_local_rule_config(
        {
            "rule_variant": "4f_vh",
            "error_broadcast_mode": "Per_Soma_Shared",
            "four_factor": {"rho_estimator": "ema+augment"},
        }
    )
    replayed = build_local_rule_config(dataclasses.asdict(config))
    assert dataclasses.asdict(replayed) == dataclasses.asdict(config)


def test_learned_gain_is_retired():
    with pytest.raises(ValueError, match="'learned_gain' was retired"):
        build_local_rule_config(
            {"three_factor": {"additive_gain_mode": "learned_gain"}}
        )
