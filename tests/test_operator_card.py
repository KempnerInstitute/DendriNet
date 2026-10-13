"""operator_card.json describes the operator the constructed model used."""

from __future__ import annotations

from copy import deepcopy

import torch
from test_operator_config_contract import _config

from dendritic_modeling.config.operator_card import build_operator_card
from dendritic_modeling.networks.architectures.factory import get_architecture


def _built(family, rule, gate="param_tanh"):
    config = _config(family, rule, gate)
    torch.manual_seed(0)
    model = get_architecture(family, deepcopy(config), input_dim=6)
    return config, model


def test_population_card_reads_the_populations_that_were_built():
    config, model = _built("population_network", "raw")
    card = build_operator_card({"type": "population_network", **config}, model=model)
    assert card["source"] == "constructed_model"
    assert card["integration_rule"] == "additive_raw"
    assert card["dendritic_layers"] == 3
    assert card["biological_neuron"]["declared"] is False
    assert isinstance(card["biological_neuron"]["violations"], list)


def test_zscored_additive_is_not_reported_as_raw():
    config, model = _built("einet", "legacy_zscore")
    card = build_operator_card(config, model=model)
    assert card["integration_rule"] == "additive_raw_feature_zscore"
    assert build_operator_card(config)["integration_rule"] == (
        "additive_raw_feature_zscore"
    )


def test_config_card_applies_core_type_aliases():
    card = build_operator_card({"type": "dendritic_additive"})
    assert card["source"] == "config"
    assert card["integration_rule"] == "additive_raw"


def test_cards_without_a_dendritic_model_are_marked():
    assert build_operator_card({"type": "population_network"})["source"] == (
        "unresolved"
    )
    mlp = torch.nn.Linear(2, 2)
    assert build_operator_card({"type": "point_mlp"}, model=mlp)["source"] == (
        "not_applicable"
    )
