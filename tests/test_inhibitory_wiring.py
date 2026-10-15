"""Feedforward and recurrent structured EINets share one inhibitory wiring rule."""

from __future__ import annotations

import copy
import logging

import pytest
import torch
import yaml

from dendritic_modeling.config import keys, load_config
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.transfer import (
    TransferLayer,
)
from dendritic_modeling.networks.architectures.factory import get_architecture

INPUT_DIM = 12


def _core(layers=1, inhibitory=(), ei=0, ie=3, ii=0, **transfer):
    return {
        "biological_neuron": False,
        "architecture": {
            "excitatory_layer_sizes": [6] * layers,
            "inhibitory_layer_sizes": list(inhibitory),
            "excitatory_branch_factors": [2],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [3],
            "ei_synapses_per_branch_per_layer": ei if isinstance(ei, list) else [ei],
            "ie_synapses_per_branch_per_layer": ie if isinstance(ie, list) else [ie],
            "ii_synapses_per_branch_per_layer": ii if isinstance(ii, list) else [ii],
        },
        "transfer": {"independent_pathways": False, **transfer},
        "morphology": {
            "use_shunting": True,
            "weight_transform": "exp",
            "somatic_synapses": False,
            "dbl_init_method": "analytical_expectation",
        },
        "sparsity": {"init_method": "xavier_normal", "type": "standard"},
        "reactivation": {"enabled": True, "type": "param_tanh"},
        "implementation": {"adaptive_initialization": False},
    }


def _recurrent(core, **recurrent_ei):
    core = copy.deepcopy(core)
    core["connectivity"]["rec_ee_synapses_per_branch"] = [1]
    core["recurrent_ei"] = {"dt": 1.0, "output_mode": "last", **recurrent_ei}
    return core


def _build(core):
    torch.manual_seed(0)
    return get_architecture("einet", copy.deepcopy(core), input_dim=INPUT_DIM)


def _excitatory_inhibition(network):
    """The first layer's excitatory ff-inhibitory synapse bank."""
    layer = network.layers[0]
    cells = getattr(layer, "excitatory_cells", None) or layer.e_population
    return cells.branch_layers[0].branch_inhibition


@pytest.mark.parametrize("recurrent", [False, True])
def test_first_layer_without_inhibitory_cells_reads_the_transferred_stream(recurrent):
    core = _core()
    network = _build(_recurrent(core) if recurrent else core)
    bank = _excitatory_inhibition(network)
    assert bank is not None
    assert bank.in_features == network.transfer_fn.inhibitory_dim


@pytest.mark.parametrize("recurrent", [False, True])
def test_excitatory_cells_read_their_own_inhibitory_cells(recurrent):
    core = _core(inhibitory=[4], ei=3)
    network = _build(_recurrent(core) if recurrent else core)
    assert _excitatory_inhibition(network).in_features == 4


@pytest.mark.parametrize("recurrent", [False, True])
@pytest.mark.parametrize(
    "core, message",
    [
        (_core(layers=2), "only the first layer reads the transferred"),
        (_core(layers=2, inhibitory=[4, 0], ei=3), "only the first layer reads"),
        (_core(inhibitory=[4]), "gives them no input synapses"),
        (_core(ei=3), "has no inhibitory cells, but sets ei=3"),
        (_core(layers=2, inhibitory=[4]), "one size per layer"),
        (
            _core(layers=2, inhibitory=[0, 4], ei=[0, 3], ie=[0, 3], ii=[0, 2]),
            "layer 1 inhibitory cells have ii=2 synapses but no feedforward",
        ),
    ],
)
def test_synapses_without_a_source_raise(recurrent, core, message):
    with pytest.raises(ValueError, match=message):
        _build(_recurrent(core) if recurrent else core)


def test_per_layer_schedule_limits_inhibition_to_the_first_layer():
    network = _build(_core(layers=2, ie=[3, 0]))
    assert network.layers[1].excitatory_cells.branch_layers[0].branch_inhibition is None


def test_recurrent_core_without_transfer_has_no_inhibitory_stream():
    with pytest.raises(ValueError, match="no transfer layer"):
        _build(_recurrent(_core(), use_transfer=False))
    _build(_recurrent(_core(ie=0), use_transfer=False))


def test_recurrent_inhibition_needs_inhibitory_cells():
    core = _recurrent(_core(ie=0))
    core["connectivity"]["rec_ie_synapses_per_branch"] = [2]
    with pytest.raises(ValueError, match="rec_ie=2 recurrent inhibitory synapses"):
        _build(core)


@pytest.mark.parametrize("recurrent", [False, True])
def test_retired_routing_keys_build_the_same_network(recurrent, caplog, monkeypatch):
    monkeypatch.setattr(keys, "_WARNED_RETIRED", set())
    core = _core()
    old = _core(input_mode=1, allow_direct_inhibitory_stream=True)
    if recurrent:
        core, old = _recurrent(core), _recurrent(old)
    with caplog.at_level(logging.WARNING):
        old_network = _build(old)
    assert "transfer.input_mode" in caplog.text
    new_network = _build(core)
    old_state, new_state = old_network.state_dict(), new_network.state_dict()
    assert old_state.keys() == new_state.keys()
    assert all(torch.equal(old_state[k], new_state[k]) for k in old_state)


@pytest.mark.parametrize(
    "core, message",
    [
        (
            _core(inhibitory=[4], ei=3, input_mode=1),
            "without the inhibitory cells it declares",
        ),
        (_core(inhibitory_mode="all"), "applied only to recurrent EINets"),
        (_recurrent(_core(input_mode=0)), "without their ie synapses"),
        (
            _recurrent(
                _core(
                    layers=2,
                    inhibitory=[4, 4],
                    ei=3,
                    ii=2,
                    input_mode=1,
                    inhibitory_mode="all",
                )
            ),
            "fed the transferred inhibitory stream to the inhibitory cells",
        ),
    ],
)
def test_retired_routing_that_changed_wiring_raises(core, message):
    with pytest.raises(ValueError, match=message):
        _build(core)


def test_recurrent_ei_transfer_inhibitory_mode_is_retired():
    with pytest.raises(ValueError, match="transfer_inhibitory_mode was removed"):
        _build(_recurrent(_core(), transfer_inhibitory_mode="first"))


def _load(tmp_path, core):
    payload = {
        "data": {"dataset_name": "mnist"},
        "model": {
            "encoder": {"type": "identity", "params": {"input_dim": INPUT_DIM}},
            "core": {"type": "einet", **core},
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return load_config(str(path))


def test_loader_accepts_reproducible_retired_routing(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        config = _load(tmp_path, _core(input_mode=1, inhibitory_mode="first"))
    assert "Ignoring model.core.transfer.input_mode" in caplog.text
    assert not hasattr(config.model.core.transfer, "input_mode")


def test_loader_rejects_retired_routing_that_changed_wiring(tmp_path):
    with pytest.raises(ValueError, match="without the inhibitory cells"):
        _load(tmp_path, _core(inhibitory=[4], ei=3, input_mode=1))


def test_transfer_layer_rejects_unknown_options():
    with pytest.raises(ValueError, match="did you mean independent_pathways"):
        TransferLayer(8, {"independant_pathways": True})


def test_transfer_layer_derives_the_missing_split_size():
    layer = TransferLayer(8, {"independent_pathways": True, "inhibitory_dim": 3})
    assert (layer.excitatory_dim, layer.inhibitory_dim) == (5, 3)
    halves = TransferLayer(8, {"independent_pathways": True})
    assert (halves.excitatory_dim, halves.inhibitory_dim) == (4, 4)
    with pytest.raises(ValueError, match="only with independent_pathways"):
        TransferLayer(8, {"excitatory_dim": 4})
