"""Structured-EINet feature blocks are read from one place on every path."""

from __future__ import annotations

import copy
import re
import warnings

import pytest
import torch
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.config.model import (
    ArchitectureConfig,
    ConnectivityConfig,
    CoreConfig,
    DecoderConfig,
    DecoderParamsConfig,
    EncoderConfig,
    EncoderParamsConfig,
    ModelConfig,
    MorphologyConfig,
    ReactivationConfig,
    TransferFunctionConfig,
)
from dendritic_modeling.config.recurrence import (
    is_recurrent_core_config,
    resolve_feature_block,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    _consume_recurrent_only_options,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

SPIKES = {"enabled": True, "levels": "distal", "threshold": 0.6}


def _core(**blocks) -> CoreConfig:
    return CoreConfig(
        type="dendritic_shunting",
        architecture=ArchitectureConfig(
            excitatory_layer_sizes=[4],
            inhibitory_layer_sizes=[],
            excitatory_branch_factors=[2],
        ),
        connectivity=ConnectivityConfig(
            ee_synapses_per_branch_per_layer=[2],
            ie_synapses_per_branch_per_layer=[0],
            ei_synapses_per_branch_per_layer=[0],
            ii_synapses_per_branch_per_layer=[0],
            rec_ee_synapses_per_branch=[0],
            rec_ie_synapses_per_branch=[0],
            rec_ei_synapses_per_branch=[0],
            rec_ii_synapses_per_branch=[0],
        ),
        transfer=TransferFunctionConfig(),
        morphology=MorphologyConfig(use_shunting=True, somatic_synapses=True),
        reactivation=ReactivationConfig(enabled=False),
        **blocks,
    )


def _model(core: CoreConfig) -> ModelConfig:
    return ModelConfig(
        task="classification",
        encoder=EncoderConfig(type="identity", params=EncoderParamsConfig(input_dim=5)),
        core=core,
        decoder=DecoderConfig(
            type="MLP",
            params=DecoderParamsConfig(hidden_dims=[], output_dim=2),
        ),
    )


def _spiking_modules(model):
    return [
        module
        for module in model.modules()
        if getattr(module, "dendritic_spikes_enabled", False)
    ]


def test_top_level_block_builds_recurrent_einet_with_spikes():
    core = _core(dendritic_spikes=dict(SPIKES))
    assert is_recurrent_core_config(core)
    torch.manual_seed(0)
    model, _ = initialize_model(_model(core))
    spiking = _spiking_modules(model)
    assert spiking
    assert all(module.dendritic_spike_threshold == 0.6 for module in spiking)


def test_nested_block_warns_and_builds_the_same_model():
    top = _core(dendritic_spikes=dict(SPIKES))
    nested = _core(dynamics={"dendritic_spikes": dict(SPIKES)})
    torch.manual_seed(0)
    reference, _ = initialize_model(_model(top))
    with pytest.warns(
        DeprecationWarning, match=re.escape("model.core.dendritic_spikes")
    ):
        assert is_recurrent_core_config(nested)
        torch.manual_seed(0)
        model, _ = initialize_model(_model(nested))
    expected, observed = reference.state_dict(), model.state_dict()
    assert expected.keys() == observed.keys()
    for key in expected:
        assert torch.equal(expected[key], observed[key]), key


def test_conflicting_placements_raise():
    core = _core(
        dendritic_spikes=dict(SPIKES),
        dynamics={"dendritic_spikes": {**SPIKES, "threshold": 0.9}},
    )
    with pytest.raises(
        ValueError, match=re.escape("conflicts with model.core.dendritic_spikes")
    ):
        resolve_feature_block(core, "dendritic_spikes")


def test_matching_placements_are_accepted_without_warning():
    core = {"dendritic_spikes": dict(SPIKES), "dynamics": {"dendritic_spikes": SPIKES}}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_feature_block(core, "dendritic_spikes") == SPIKES


@pytest.mark.parametrize(
    "core",
    [
        {"soma_feedback": {"enabled": True}},
        {"dynamics": {"soma_feedback": {"enabled": True}}},
        {"recurrent_ei": {"soma_feedback": {"enabled": True}}},
        {"spiking": {"soma_feedback": {"enabled": True}}},
    ],
)
def test_soma_feedback_placements_resolve_to_one_block(core):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert resolve_feature_block(core, "soma_feedback") == {"enabled": True}


def test_absent_blocks_resolve_to_empty_mapping():
    assert resolve_feature_block(_core(), "dendritic_spikes") == {}
    assert not is_recurrent_core_config(_core())


def test_load_config_rejects_the_nested_placement(tmp_path):
    payload = yaml.safe_load(
        """
data: {dataset_name: mnist}
model:
  encoder: {type: identity, params: {input_dim: 5}}
  core:
    type: dendritic_shunting
    dynamics:
      dendritic_spikes: {enabled: true}
"""
    )
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(
        ValueError,
        match=re.escape(
            "model.core.dynamics.dendritic_spikes -> model.core.dendritic_spikes"
        ),
    ):
        load_config(str(path))

    moved = copy.deepcopy(payload)
    moved["model"]["core"]["dendritic_spikes"] = moved["model"]["core"].pop("dynamics")[
        "dendritic_spikes"
    ]
    path.write_text(yaml.safe_dump(moved))
    assert load_config(str(path)).model.core.dendritic_spikes == {"enabled": True}


def test_feedforward_einet_rejects_enabled_nested_blocks():
    with pytest.raises(
        ValueError, match=re.escape("dynamics.dendritic_spikes.enabled=true")
    ):
        _consume_recurrent_only_options(
            {"dynamics": {"dendritic_spikes": {"enabled": True}}}
        )


@pytest.mark.parametrize(
    "core, message",
    [
        ({"dynamics": {"mode": "spike", "treshold": 0.6}}, "did you mean 'threshold'"),
        ({"dynamics": {"mode": "spike", "dynamics_mode": "spike"}}, "name the same"),
        ({"dendritic_spikes": {"enabeld": True}}, "did you mean 'enabled'"),
        ({"soma_feedback": {"strenght": 0.2}}, "did you mean 'strength'"),
        (
            {
                "dendritic_spikes": {"threshold": 0.5},
                "dynamics": {"dendritic_spike_threshold": 0.6},
            },
            "keep model.core.dendritic_spikes",
        ),
        ({"recurrent_ei": {"learn_taus": True}}, "did you mean 'learnable_tau'"),
        (
            {"recurrent_ei": {"recurrent_per_layer": [1]}},
            "no longer read",
        ),
    ],
)
def test_recurrent_payload_typos_and_duplicates_raise(tmp_path, core, message):
    payload = {
        "data": {"dataset_name": "mnist"},
        "model": {
            "encoder": {"type": "identity", "params": {"input_dim": 5}},
            "core": {"type": "dendritic_shunting", **core},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match=message):
        load_config(str(path))
