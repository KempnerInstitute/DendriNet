"""Adoption is explicit, versioned and separate from historical upgrades."""

import copy

import pytest
import torch
from test_biological_constraints import _einet, _population

from dendritic_modeling.config import CONFIG_VERSION, config_from_mapping
from dendritic_modeling.config.biological_adoption import prepare_biological_config
from dendritic_modeling.networks.architectures.factory import get_architecture


@pytest.mark.parametrize(
    "kind, payload", [("einet", _einet()), ("population_network", _population())]
)
def test_adoption_builds_without_mutating_historical_recipe(kind, payload):
    original = {"config_version": 3, "model": {"core": {"type": kind, **payload}}}
    saved = copy.deepcopy(original)
    prepared = prepare_biological_config(original)
    assert original == saved
    assert prepared["config_version"] == CONFIG_VERSION
    cfg = config_from_mapping(prepared)
    assert cfg.model.core.biological_neuron
    model = get_architecture(kind, cfg.model.core, input_dim=8)
    assert torch.isfinite(model(torch.rand(2, 8))).all()


def test_checked_identity_and_paired_external_sources_are_explicit():
    original = {"model": {"core": {"type": "einet", **_einet()}}}
    prepared = prepare_biological_config(
        original,
        input_activation="nonnegative",
        input_source_identity="paired_afferents",
    )
    assert (
        prepared["model"]["core"]["transfer"]["input_source_identity"]
        == "paired_afferents"
    )
    cfg = config_from_mapping(prepared)
    model = get_architecture("einet", cfg.model.core, input_dim=8)
    with pytest.raises(ValueError, match="finite values"):
        model(-torch.ones(2, 8))


def test_positive_projection_is_not_added_implicitly():
    data = {
        "model": {
            "core": {
                "type": "population_network",
                **_population(input_projection_dims=[7]),
            }
        }
    }
    prepared = prepare_biological_config(
        data, input_activation="nonnegative", positive_input_projection=True
    )
    cfg = config_from_mapping(prepared)
    assert get_architecture(cfg.model.core.type, cfg.model.core, input_dim=8)
    cfg = config_from_mapping(prepare_biological_config(data))
    with pytest.raises(ValueError, match="input projection"):
        get_architecture(cfg.model.core.type, cfg.model.core, input_dim=8)


def test_explicit_nonbiological_control():
    prepared = prepare_biological_config(
        {"model": {"core": {"type": "MLP"}}}, biological=False
    )
    assert prepared["model"]["core"]["biological_neuron"] is False


@pytest.mark.parametrize(
    "options",
    [
        {"biological": "false"},
        {"biological": False, "input_activation": "relu"},
        {"positive_input_projection": True},
        {"input_source_identity": "typo"},
    ],
)
def test_invalid_adoption_raises(options):
    with pytest.raises(ValueError):
        prepare_biological_config(
            {"model": {"core": {"type": "einet", **_einet()}}}, **options
        )


def test_requires_expanded_experiment():
    with pytest.raises(ValueError, match="Expand"):
        prepare_biological_config({"base_config": {"model": {}}})
