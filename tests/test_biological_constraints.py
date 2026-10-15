"""model.core.biological_neuron: the constraint set is checked on the built model."""

from __future__ import annotations

import copy

import pytest
import torch
import yaml

from dendritic_modeling.config import load_config
from dendritic_modeling.networks.architectures.biological_constraints import (
    biological_violations,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


def _einet(**changes):
    """A feedforward EINet that satisfies the biological constraint set."""
    core = {
        "architecture": {
            "excitatory_layer_sizes": [6],
            "inhibitory_layer_sizes": [4],
            "excitatory_branch_factors": [2],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [3],
            "ei_synapses_per_branch_per_layer": [3],
            "ie_synapses_per_branch_per_layer": [3],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {"independent_pathways": False, "output_activation": "relu"},
        "morphology": {
            "use_shunting": True,
            "weight_transform": "softplus",
            "somatic_synapses": True,
            "dbl_init_method": "analytical_expectation",
        },
        "sparsity": {"init_method": "xavier_normal", "type": "standard"},
        "reactivation": {"enabled": True, "type": "param_tanh"},
        "implementation": {"adaptive_initialization": False},
    }
    for path, value in changes.items():
        section, key = path.split(".")
        core[section][key] = value
    return core


def _recurrent(core):
    core = copy.deepcopy(core)
    core["connectivity"].update(
        {
            "rec_ee_synapses_per_branch": [1],
            "rec_ie_synapses_per_branch": [1],
            "rec_ei_synapses_per_branch": [1],
            "rec_ii_synapses_per_branch": [0],
        }
    )
    core["recurrent_ei"] = {"dt": 1.0, "output_mode": "last"}
    return core


def _population(defaults=None, connections=None, **network):
    """A population network that satisfies the biological constraint set."""
    population_defaults = {
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 1,
        "rec_excitatory_synapses": 0,
        "rec_inhibitory_synapses": 0,
        "weight_transform": "softplus",
        "use_shunting": True,
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "dbl_init_method": "default",
        **(defaults or {}),
    }
    payload = {
        "input_transform": "relu",
        "layers": [
            {
                "name": "layer0",
                "readout_population": "e",
                "population_defaults": population_defaults,
                "populations": [
                    {
                        "name": "e",
                        "polarity": "excitatory",
                        "n_neurons": 5,
                        "branch_factors": [2],
                    },
                    {
                        "name": "i",
                        "polarity": "inhibitory",
                        "n_neurons": 3,
                        "branch_factors": [1],
                    },
                ],
                "connections": connections
                or [
                    {"source": "input", "target": "e"},
                    {"source": "input", "target": "i"},
                    {"source": "i", "target": "e"},
                ],
            }
        ],
        **network,
    }
    return {"population_network": payload}


def _violations(core_type, core):
    torch.manual_seed(0)
    return biological_violations(
        get_architecture(core_type, {**core, "biological_neuron": False}, input_dim=8)
    )


def _single(core_type, core):
    violations = _violations(core_type, core)
    assert len(violations) == 1, violations
    return violations[0]


@pytest.mark.parametrize(
    "core_type, core",
    [
        ("einet", _einet()),
        ("dendritic_additive", _einet(**{"morphology.use_shunting": False})),
        ("einet", _recurrent(_einet())),
        ("population_network", _population()),
        (
            "population_network",
            _population(defaults={"reactivate": False}),
        ),
    ],
)
def test_compliant_cores_build_with_the_declaration(core_type, core):
    assert _violations(core_type, core) == []
    torch.manual_seed(0)
    network = get_architecture(
        core_type, {**core, "biological_neuron": True}, input_dim=8
    )
    recurrent = getattr(network, "is_recurrent", False)
    sequence = torch.rand(2, 3, 8) if recurrent else torch.rand(2, 8)
    with torch.no_grad():
        assert torch.isfinite(network(sequence)).all()


def test_signed_weight_transform_is_reported():
    message = _single("einet", _einet(**{"morphology.weight_transform": "identity"}))
    assert "signed weight transform 'identity'" in message


def test_signed_alias_types_are_reported_from_the_built_model():
    for core_type in ("dendritic_signed", "flat_signed"):
        assert any(
            "'identity'" in message for message in _violations(core_type, _einet())
        )
    messages = _violations("dendritic_mlp", _einet())
    assert any("merges excitatory and inhibitory" in m for m in messages)


def test_mlp_inhibitory_network_is_reported():
    message = _single(
        "einet", _einet(**{"architecture.inhibitory_network_type": "mlp"})
    )
    assert "MLP inhibitory network" in message


def test_direct_inhibition_from_a_duplicated_input_is_reported():
    core = _einet(
        **{
            "architecture.inhibitory_layer_sizes": [],
            "connectivity.ei_synapses_per_branch_per_layer": [0],
        }
    )
    message = _single("einet", core)
    assert "first-layer excitatory cells" in message
    assert "independent_pathways" in message
    core["transfer"]["independent_pathways"] = True
    assert _violations("einet", core) == []


def test_inhibitory_cells_reading_a_duplicated_input_are_reported():
    core = _einet(**{"connectivity.ii_synapses_per_branch_per_layer": [2]})
    assert "first-layer inhibitory cells" in _single("einet", core)
    assert "layer 0 inhibitory cells" in _single("einet", _recurrent(core))


def test_unrectified_input_is_reported():
    # Positive-weight EINets already reject a signed transfer output (tanh).
    core = _einet(**{"transfer.output_activation": None})
    assert "unrectified" in _single("einet", core)


def test_recurrent_core_without_transfer_is_reported():
    core = _recurrent(_einet())
    core["recurrent_ei"]["use_transfer"] = False
    assert "no transfer layer" in _single("einet", core)


def test_signed_soma_gate_is_reported():
    messages = _violations("einet", _einet(**{"reactivation.type": "tanh"}))
    assert len(messages) == 2  # excitatory and inhibitory cells
    assert all("soma gate Tanh can emit negative values" in m for m in messages)


def test_ungated_additive_population_is_reported():
    core = _einet(**{"morphology.use_shunting": False, "reactivation.enabled": False})
    messages = _violations("dendritic_additive", core)
    assert messages and all("additive integration" in m for m in messages)


def test_population_unrectified_input_is_reported():
    message = _single("population_network", _population(input_transform="identity"))
    assert "input_transform 'identity'" in message


def test_population_reading_both_streams_of_an_unsplit_input_is_reported():
    connections = [
        {"source": "input", "target": "e"},
        {"source": "input", "target": "i"},
        {"source": "input_i", "target": "e"},
        {"source": "i", "target": "e"},
    ]
    message = _single("population_network", _population(connections=connections))
    assert "both excites and inhibits" in message


def test_population_pathway_against_source_polarity_is_reported():
    connections = [
        {"source": "input", "target": "e"},
        {"source": "input", "target": "i"},
        {"source": "i", "target": "e", "pathway": "ff_excitatory"},
    ]
    message = _single("population_network", _population(connections=connections))
    assert "uses pathway 'ff_excitatory', but its source is inhibitory" in message


def test_population_signed_projections_are_reported():
    messages = _violations("population_network", _population(input_projection_dims=[7]))
    assert any("input projection" in m for m in messages)
    messages = _violations(
        "population_network",
        _population(defaults={"soma_feedback_enabled": True}),
    )
    assert any("soma feedback" in m for m in messages)


def test_population_membrane_readout_is_reported():
    core = _population(defaults={"dynamics_mode": "spike", "spike_readout": "membrane"})
    messages = _violations("population_network", core)
    assert messages and all("membrane potential" in m for m in messages)
    core = _population(defaults={"dynamics_mode": "spike", "spike_readout": "rate"})
    assert _violations("population_network", core) == []


def test_declared_core_that_breaks_the_set_fails_to_build():
    core = {**_einet(**{"transfer.output_activation": None}), "biological_neuron": True}
    with pytest.raises(ValueError, match="breaks the biological constraint set"):
        get_architecture("einet", core, input_dim=8)
    with pytest.raises(ValueError, match="breaks the biological constraint set"):
        get_architecture(
            "population_network",
            {**_population(input_transform="identity"), "biological_neuron": True},
            input_dim=8,
        )


def test_declaration_is_rejected_for_cores_outside_the_set():
    with pytest.raises(ValueError, match="not available for core type 'gru'"):
        get_architecture(
            "gru", {"baseline_rnn": {"hidden_dim": 4}, "biological_neuron": True}, 4
        )


def _load(tmp_path, core):
    payload = {
        "data": {"dataset_name": "mnist"},
        "model": {
            "encoder": {"type": "identity", "params": {"input_dim": 8}},
            "core": core,
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    return load_config(str(path))


def test_loader_validates_the_declaration(tmp_path):
    population = {"type": "population_network", **_population()}
    assert _load(
        tmp_path, {**population, "biological_neuron": True}
    ).model.core.biological_neuron
    assert _load(tmp_path, population).model.core.biological_neuron is True
    assert (
        _load(
            tmp_path, {**population, "biological_neuron": None}
        ).model.core.biological_neuron
        is True
    )
    with pytest.raises(ValueError, match="must be true or false"):
        _load(tmp_path, {**population, "biological_neuron": "yes"})
    gru = {"type": "gru", "baseline_rnn": {"hidden_dim": 4}}
    assert (
        _load(
            tmp_path, {**gru, "biological_neuron": False}
        ).model.core.biological_neuron
        is False
    )
    with pytest.raises(ValueError, match="not available for core type 'gru'"):
        _load(tmp_path, {**gru, "biological_neuron": True})
