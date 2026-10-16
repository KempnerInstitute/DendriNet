"""Encoder and decoder params are checked against what each type reads."""

from __future__ import annotations

import dataclasses
import logging

import pytest
import torch
import yaml
from torch.utils.data import Subset, TensorDataset

from dendritic_modeling.config import load_config
from dendritic_modeling.config.model import EncoderParamsConfig
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

BASE = yaml.safe_load(
    """
data: {dataset_name: mnist}
model:
  encoder: {type: identity, params: {input_dim: 6}}
  core:
    biological_neuron: false
    type: einet
    architecture: {excitatory_layer_sizes: [4], inhibitory_layer_sizes: []}
    connectivity:
      ee_synapses_per_branch_per_layer: [2]
      ie_synapses_per_branch_per_layer: [0]
  decoder: {type: identity, params: {hidden_dims: []}}
"""
)


def _load(tmp_path, *, encoder=None, decoder=None):
    payload = yaml.safe_load(yaml.safe_dump(BASE))
    if encoder is not None:
        payload["model"]["encoder"] = encoder
    if decoder is not None:
        payload["model"]["decoder"] = decoder
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return load_config(str(path))


@pytest.mark.parametrize(
    "decoder, field",
    [
        ({"type": "identity", "params": {"hidden_dims": [32, 16]}}, "hidden_dims"),
        ({"type": "identity", "params": {"activation": "tanh"}}, "activation"),
    ],
)
def test_identity_decoder_rejects_params_it_ignores(tmp_path, decoder, field):
    with pytest.raises(ValueError, match=rf"model\.decoder\.params\.{field}"):
        _load(tmp_path, decoder=decoder)


def test_identity_encoder_rejects_router_params(tmp_path):
    encoder = {"type": "identity", "params": {"input_dim": 6, "router_mode": "learned"}}
    with pytest.raises(ValueError, match=r"model\.encoder\.params\.router_mode"):
        _load(tmp_path, encoder=encoder)


def test_params_at_their_default_and_read_params_are_accepted(tmp_path):
    config = _load(
        tmp_path,
        encoder={
            "type": "pathway_router",
            "params": {"input_dim": 6, "router_mode": "fixed", "n_pathways": 2},
        },
        decoder={"type": "MLP", "params": {"hidden_dims": [], "output_dim": 3}},
    )
    assert config.model.decoder.params.output_dim == 3


def test_saved_record_with_unread_params_warns(tmp_path, caplog):
    params = dataclasses.asdict(EncoderParamsConfig())
    params.update(input_dim=6, router_mode="learned")
    with caplog.at_level(logging.WARNING):
        _load(tmp_path, encoder={"type": "identity", "params": params})
    assert "Ignoring params of a saved config" in caplog.text


@pytest.mark.parametrize("output_dim, raises", [(None, False), (4, False), (3, True)])
def test_identity_decoder_output_dim_must_match_the_core(tmp_path, output_dim, raises):
    params = {} if output_dim is None else {"output_dim": output_dim}
    config = _load(tmp_path, decoder={"type": "identity", "params": params})
    if raises:
        with pytest.raises(ValueError, match="differs from the core output"):
            initialize_model(config.model)
    else:
        model, _ = initialize_model(config.model)
        assert model.core_network.output_dim == 4


def _tensor_data(labels):
    labels = torch.as_tensor(labels)
    return TensorDataset(torch.zeros(len(labels), 3), labels)


def test_training_class_count_reads_stored_labels():
    from dendritic_modeling.scripts.script_utils.setup_utils import (
        _training_class_count,
    )

    data = _tensor_data([0, 2, 1, 2])
    assert _training_class_count(data) == 3
    assert _training_class_count(Subset(data, [0, 2])) == 2
    assert _training_class_count(_tensor_data([0.5, 1.0])) is None
    assert _training_class_count(_tensor_data([-100, 1])) is None

    class Listed(torch.utils.data.Dataset):
        def __init__(self):
            self.classes = ["a", "b", "c", "d"]
            self.targets = [0, 1]

    assert _training_class_count(Listed()) == 4


@pytest.mark.parametrize(
    "task, labels, width, outcome",
    [
        ("classification", [0, 4], 3, "raise"),
        ("classification", [0, 1], 10, "warn"),
        ("classification", [0, 2], 3, "quiet"),
        ("regression", [0, 4], 3, "quiet"),
    ],
)
def test_classifier_width_is_checked_against_the_labels(
    caplog, task, labels, width, outcome
):
    from types import SimpleNamespace

    from dendritic_modeling.scripts.script_utils.setup_utils import (
        _check_classification_output_width,
    )

    config = SimpleNamespace(task=task)
    model = SimpleNamespace(decoder_network=SimpleNamespace(output_dim=width))
    data = _tensor_data(labels)
    if outcome == "raise":
        with pytest.raises(ValueError, match="training labels span"):
            _check_classification_output_width(config, model, data)
        return
    with caplog.at_level(logging.WARNING):
        _check_classification_output_width(config, model, data)
    assert ("extra outputs" in caplog.text) is (outcome == "warn")


@pytest.mark.parametrize(
    "valid_size, load_best, early_stopping, raises",
    [
        (0, True, False, True),
        (0, False, True, True),
        (0, False, False, False),
        (2, True, True, False),
    ],
)
def test_empty_validation_cannot_drive_selection(
    valid_size, load_best, early_stopping, raises
):
    from types import SimpleNamespace

    from dendritic_modeling.scripts.script_utils.setup_utils import (
        _reject_empty_validation_selection,
    )

    valid = _tensor_data(list(range(valid_size)))
    training = SimpleNamespace(
        main=SimpleNamespace(
            common=SimpleNamespace(
                load_best_state_dict=load_best, early_stopping=early_stopping
            )
        )
    )
    experiment = SimpleNamespace(train_valid_split=1.0)
    if raises:
        with pytest.raises(ValueError, match="no validation samples"):
            _reject_empty_validation_selection(valid, training, experiment)
    else:
        _reject_empty_validation_selection(valid, training, experiment)


def test_test_split_options_take_effect_or_raise():
    from types import SimpleNamespace

    from dendritic_modeling.config.config import ExperimentConfig
    from dendritic_modeling.scripts.script_utils.setup_utils import (
        _validate_sealed_test_dataset,
    )

    with pytest.raises(ValueError, match="only with sealed_test: true"):
        ExperimentConfig(allow_test_data=False)
    ExperimentConfig(sealed_test=True, allow_test_data=False, max_test_samples=5)

    test = _tensor_data([0, 1, 2])
    unsealed = SimpleNamespace(sealed_test=False, max_test_samples=2)
    with pytest.raises(ValueError, match=r"exceeds experiment\.max_test_samples"):
        _validate_sealed_test_dataset(unsealed, test)
    _validate_sealed_test_dataset(
        SimpleNamespace(sealed_test=False, max_test_samples=3), test
    )
