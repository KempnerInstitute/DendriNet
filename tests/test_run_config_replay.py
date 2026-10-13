"""A saved config.json reloads into the configuration that produced it."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.config_utils import prepare_trainer_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.scripts.training.train_experiments import _save_run_config

FEEDFORWARD = """
experiment: {seed: 3}
data: {dataset_name: mnist}
model:
  encoder: {type: identity, params: {input_dim: 6}}
  core:
    type: einet
    architecture: {excitatory_layer_sizes: [4], inhibitory_layer_sizes: []}
    connectivity:
      ee_synapses_per_branch_per_layer: [2]
      ie_synapses_per_branch_per_layer: [0]
  decoder: {type: identity, params: {hidden_dims: []}}
training:
  main:
    strategy: standard
    common: {epochs: 1, grad_clip_value: 3.0}
"""

RECURRENT = """
experiment: {seed: 3}
data: {dataset_name: mnist}
model:
  output_scale_mode: fixed
  encoder: {type: identity, params: {input_dim: 6}}
  core:
    type: einet
    architecture: {excitatory_layer_sizes: [4], inhibitory_layer_sizes: []}
    connectivity:
      ee_synapses_per_branch_per_layer: [2]
      ie_synapses_per_branch_per_layer: [0]
      rec_ee_synapses_per_branch: [1]
  decoder: {type: MLP, params: {hidden_dims: [], output_dim: 3}}
training:
  main:
    strategy: recurrent
    common:
      epochs: 1
      grad_clip_value: 3.0
      use_amp: false
      lr_schedule: cosine
      lr_warmup_epochs: 2
"""

REGRESSION = FEEDFORWARD.replace(
    "  encoder: {type: identity", "  task: regression\n  encoder: {type: identity"
)


def _replay(tmp_path, text):
    path = tmp_path / "config.yaml"
    path.write_text(text)
    original = load_config(str(path))
    run_dir = tmp_path / "run"
    _save_run_config(original, str(run_dir))
    replayed = load_config(str(run_dir / "config.json"))
    return original, replayed


def _trainer_settings(config, tmp_path):
    return prepare_trainer_config(
        config.training.main, config.experiment, str(tmp_path / "out")
    )


@pytest.mark.parametrize("text", [FEEDFORWARD, RECURRENT, REGRESSION])
def test_saved_config_replays_trainer_settings_and_model(tmp_path, text):
    original, replayed = _replay(tmp_path, text)
    assert _trainer_settings(replayed, tmp_path) == _trainer_settings(
        original, tmp_path
    )
    torch.manual_seed(0)
    reference, _ = initialize_model(original.model)
    torch.manual_seed(0)
    model, _ = initialize_model(replayed.model)
    assert reference.state_dict().keys() == model.state_dict().keys()


def test_recurrent_replay_keeps_common_clipping_amp_and_schedule(tmp_path):
    original, replayed = _replay(tmp_path, RECURRENT)
    for config in (original, replayed):
        settings = _trainer_settings(config, tmp_path)
        assert "recurrent_config" not in settings
        assert settings["grad_clip_value"] == 3.0
        assert settings["lr_schedule"] == "cosine"
        assert settings["lr_warmup_epochs"] == 2


def test_explicit_recurrent_overrides_survive_the_round_trip(tmp_path):
    payload = yaml.safe_load(RECURRENT)
    payload["training"]["main"]["strategies"] = {"recurrent": {"grad_clip_value": 0.5}}
    original, replayed = _replay(tmp_path, yaml.safe_dump(payload))
    for config in (original, replayed):
        overrides = _trainer_settings(config, tmp_path)["recurrent_config"]
        assert overrides["grad_clip_value"] == 0.5
    assert dataclasses.asdict(replayed.training.main.strategies.recurrent) == (
        dataclasses.asdict(original.training.main.strategies.recurrent)
    )


def test_explicit_learned_scale_on_a_recurrent_core_is_rejected(tmp_path):
    payload = yaml.safe_load(RECURRENT)
    payload["model"].pop("output_scale_mode")
    payload["model"]["learned_output_scale"] = True
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="learned_output_scale"):
        load_config(str(path))


def test_a_saved_config_reloads_without_gate_start_value_warnings(
    tmp_path, caplog, monkeypatch
):
    import json
    import logging

    from dendritic_modeling.config import reactivation

    monkeypatch.setattr(reactivation, "_WARNED_UNUSED_GATE_INIT", set())
    caplog.set_level(logging.WARNING, logger="dendritic_modeling.config.reactivation")
    _replay(tmp_path, FEEDFORWARD)
    # config.json lists the default start values under the analytical policy.
    assert "no effect" not in caplog.text
    saved = json.loads((tmp_path / "run" / "config.json").read_text())
    saved["model"]["core"]["reactivation"]["init_m"] = 2.0
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(saved))
    load_config(str(edited))
    assert "init_m in model.core.reactivation has no effect" in caplog.text
