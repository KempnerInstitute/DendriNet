"""Config conversion does not mutate inputs; aliases and partial runtimes resolve once."""

from __future__ import annotations

import copy

import pytest
import yaml
from omegaconf import OmegaConf

from dendritic_modeling.config import load_config
from dendritic_modeling.config.model import ReactivationConfig
from dendritic_modeling.scripts.script_utils.config_utils import (
    prepare_ei_network_params,
)

CORE = {
    "type": "einet",
    "architecture": {"excitatory_layer_sizes": [3], "inhibitory_layer_sizes": []},
    "connectivity": {
        "ee_synapses_per_branch_per_layer": [2],
        "ie_synapses_per_branch_per_layer": [0],
    },
    "reactivation": {"enabled": True, "type": "param_tanh"},
}


@pytest.mark.parametrize("wrap", [lambda d: d, OmegaConf.create])
def test_prepare_ei_network_params_is_repeatable_and_leaves_input_unchanged(wrap):
    original = copy.deepcopy(CORE)
    config = wrap(copy.deepcopy(CORE))
    first = prepare_ei_network_params(config, input_dim=4)
    second = prepare_ei_network_params(config, input_dim=5)
    assert first["reactivate"] is True and second["reactivate"] is True
    assert {k: v for k, v in first.items() if k != "input_dim"} == {
        k: v for k, v in second.items() if k != "input_dim"
    }
    assert (first["input_dim"], second["input_dim"]) == (4, 5)
    plain = OmegaConf.to_container(config) if OmegaConf.is_config(config) else config
    assert plain == original


def test_dendritic_activation_alias_resolves_into_type():
    config = ReactivationConfig(dendritic_activation="param_relu")
    assert (config.enabled, config.type, config.dendritic_activation) == (
        True,
        "param_relu",
        None,
    )
    assert ReactivationConfig(dendritic_activation="none").enabled is False


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"enabled": False, "dendritic_activation": "param_relu"}, "enabled: false"),
        ({"type": "param_relu", "dendritic_activation": "tanh"}, "set only type"),
    ],
)
def test_dendritic_activation_alias_conflicts_raise(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ReactivationConfig(**kwargs)


def test_partial_analysis_runtime_keeps_the_implicit_mode(tmp_path):
    payload = {
        "data": {"dataset_name": "mnist"},
        "analysis": {"runtime": {"training": {"batch_size": 64}}},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    runtime = load_config(str(path)).analysis.runtime
    assert runtime.training.mode == "materialize"
    assert runtime.training.batch_size == 64
    assert runtime.final.mode == "materialize"
