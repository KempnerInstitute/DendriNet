import pytest
import torch

from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent import MultiPopulationNetwork
from dendritic_modeling.training.strategies.local_learning import LocalCreditAssignment


def _base_population(overrides=None):
    payload = {
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 1,
        "rec_excitatory_synapses": 1,
        "rec_inhibitory_synapses": 1,
        "somatic_synapses": True,
        "reactivate": True,
        "dbl_init_method": "default",
        "weight_transform": "exp",
    }
    if overrides:
        payload.update(overrides)
    return payload


def _single_layer_temporal_population_config(population_overrides):
    return {
        "multi_population": {
            "output_mode": "last",
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                            "use_shunting": True,
                            **population_overrides,
                        }
                    ),
                    "populations": [
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                            "population": {"ff_inhibitory_synapses": 0},
                        },
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [2],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i"},
                        {"source": "input", "target": "e"},
                        {"source": "i", "target": "e"},
                    ],
                }
            ],
        }
    }


def test_multi_population_two_pop_feedforward_matches_current_ei_pattern():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                        },
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [2],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i"},
                        {"source": "input", "target": "e"},
                        {"source": "i", "target": "e"},
                    ],
                }
            ]
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=4)
    assert isinstance(net, MultiPopulationNetwork)
    assert net.output_dim == 5
    assert net.is_recurrent is False

    y = net(torch.randn(2, 4))
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize(
    "population_overrides",
    [
        {"dynamics_mode": "spike", "spike_readout": "rate"},
        {
            "synapse_types": {
                "enabled": True,
                "excitatory": [{"name": "AMPA", "fraction": 1.0}],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0}],
            }
        },
        {"dendritic_spikes_enabled": True, "dendritic_spike_levels": "distal"},
        {"soma_feedback_enabled": True, "soma_feedback_levels": "distal"},
    ],
)
def test_multi_population_temporal_features_force_recurrent_sequence_path(
    population_overrides,
):
    cfg = _single_layer_temporal_population_config(population_overrides)

    net = get_architecture("multi_population_ei", cfg, input_dim=4)

    assert net.layers[0].is_recurrent is True
    assert net.is_recurrent is True
    y, hidden = net(torch.randn(2, 3, 4), return_hidden=True)
    assert y.shape == (2, 5)
    assert hidden is not None
    assert set(hidden[0].outputs) == {"i", "e"}
    assert torch.isfinite(y).all()


def test_multi_population_four_pop_feedforward_with_distinct_morphologies():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "cortical",
                    "readout_population": "pyr",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "pv",
                            "polarity": "inhibitory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        },
                        {
                            "name": "vip",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                        },
                        {
                            "name": "sst",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [2],
                            "population": {"tau_base": 80.0},
                        },
                        {
                            "name": "pyr",
                            "polarity": "excitatory",
                            "n_neurons": 6,
                            "branch_factors": [3, 2],
                            "population": {"tau_base": 60.0},
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "pv"},
                        {"source": "input", "target": "vip"},
                        {"source": "input", "target": "sst"},
                        {"source": "input", "target": "pyr"},
                        {"source": "vip", "target": "sst"},
                        {"source": "pv", "target": "pyr"},
                        {"source": "sst", "target": "pyr"},
                    ],
                }
            ]
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=5)
    layer = net.layers[0]
    assert layer.population_configs["pyr"].branch_factors == [3, 2]
    assert layer.population_configs["sst"].branch_factors == [2]

    y = net(torch.randn(3, 5))
    assert y.shape == (3, 6)
    assert torch.isfinite(y).all()


def test_multi_population_two_feedforward_layers_propagate_readout_dims():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e0",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "i0",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                        },
                        {
                            "name": "e0",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [2],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i0"},
                        {"source": "input", "target": "e0"},
                        {"source": "i0", "target": "e0"},
                    ],
                },
                {
                    "name": "layer1",
                    "readout_population": "e1",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "i1",
                            "polarity": "inhibitory",
                            "n_neurons": 2,
                            "branch_factors": [1],
                        },
                        {
                            "name": "e1",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i1"},
                        {"source": "input", "target": "e1"},
                        {"source": "i1", "target": "e1"},
                    ],
                },
            ]
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=4)
    assert net.layers[0].input_dim == 4
    assert net.layers[1].input_dim == 5
    assert net.output_dim == 4

    y = net(torch.randn(3, 4))
    assert y.shape == (3, 4)
    assert torch.isfinite(y).all()


def test_multi_population_network_level_readout_population_fills_selected_layer():
    cfg = {
        "multi_population": {
            "readout_layer": "layer0",
            "readout_population": "e",
            "layers": [
                {
                    "name": "layer0",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                        },
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [2],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i"},
                        {"source": "input", "target": "e"},
                        {"source": "i", "target": "e"},
                    ],
                }
            ],
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=4)
    assert net.layers[0].readout_population == "e"
    assert net.output_dim == 5


def test_multi_population_readout_layer_must_exist():
    cfg = {
        "multi_population": {
            "readout_layer": "missing",
            "layers": [
                {
                    "name": "layer0",
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [{"source": "input", "target": "e"}],
                }
            ],
        }
    }
    with pytest.raises(ValueError, match="readout_layer"):
        get_architecture("multi_population_ei", cfg, input_dim=3)


def test_multi_population_network_readout_population_conflict_errors():
    cfg = {
        "multi_population": {
            "readout_layer": "layer0",
            "readout_population": "e",
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "other",
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        },
                        {
                            "name": "other",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [1],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "e"},
                        {"source": "input", "target": "other"},
                    ],
                }
            ],
        }
    }
    with pytest.raises(ValueError, match="conflicts"):
        get_architecture("multi_population_ei", cfg, input_dim=3)


def test_multi_population_missing_required_drive_errors_at_build():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "bad",
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [],
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="ff_excitatory"):
        get_architecture("multi_population_ei", cfg, input_dim=3)


def test_multi_population_rejects_runtime_input_dim_mismatch():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "layer0",
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [{"source": "input", "target": "e"}],
                }
            ]
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=3)
    with pytest.raises(ValueError, match="expected last input dimension 3"):
        net(torch.randn(2, 4))


def test_multi_population_input_projection_dims_must_be_positive():
    cfg = {
        "multi_population": {
            "input_projection_dims": [0],
            "layers": [
                {
                    "name": "layer0",
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [{"source": "input", "target": "e"}],
                }
            ],
        }
    }
    with pytest.raises(ValueError, match="input_projection_dims"):
        get_architecture("multi_population_ei", cfg, input_dim=3)


def test_multi_population_transfer_named_input_streams_split_first_layer():
    cfg = {
        "multi_population": {
            "use_transfer": True,
            "transfer_params": {
                "input_mode": 0,
                "independent_pathways": True,
                "excitatory_dim": 3,
                "inhibitory_dim": 2,
            },
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [
                        {"source": "input_e", "target": "e"},
                        {"source": "input_i", "target": "e"},
                    ],
                }
            ],
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=5)
    layer = net.layers[0]
    assert layer.input_dim == 3
    assert layer.external_input_dims["input_e"] == 3
    assert layer.external_input_dims["input_i"] == 2
    assert layer._stream_dim("e", "same_step", "ff_excitatory") == 3
    assert layer._stream_dim("e", "same_step", "ff_inhibitory") == 2

    y = net(torch.randn(2, 5))
    assert y.shape == (2, 4)
    assert torch.isfinite(y).all()


def test_multi_population_input_i_defaults_to_inhibitory_pathway():
    cfg = {
        "multi_population": {
            "use_transfer": True,
            "transfer_params": {
                "input_mode": 1,
                "independent_pathways": True,
                "excitatory_dim": 3,
                "inhibitory_dim": 2,
            },
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [
                        {"source": "input_e", "target": "e"},
                        {"source": "input_i", "target": "e"},
                    ],
                }
            ],
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=5)
    layer = net.layers[0]
    assert layer._stream_sources("e", "same_step", "ff_excitatory") == ["input_e"]
    assert layer._stream_sources("e", "same_step", "ff_inhibitory") == ["input_i"]


def test_multi_population_input_aliases_share_stream_without_transfer():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "layer0",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [
                        {"source": "input_excitatory", "target": "e"},
                        {"source": "input_inhibitory", "target": "e"},
                    ],
                }
            ]
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=3)
    layer = net.layers[0]
    assert layer.external_input_dims["input"] == 3
    assert layer.external_input_dims["input_e"] == 3
    assert layer.external_input_dims["input_i"] == 3

    y = net(torch.randn(2, 3))
    assert y.shape == (2, 4)
    assert torch.isfinite(y).all()


def test_multi_population_delayed_external_input_sources_error():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "bad",
                    "recurrent": True,
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [
                        {"source": "input", "target": "e"},
                        {
                            "source": "input_i",
                            "target": "e",
                            "timing": "delayed",
                        },
                    ],
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="cannot use timing='delayed'"):
        get_architecture("multi_population_ei", cfg, input_dim=3)


def test_multi_population_second_layer_input_streams_are_previous_readout():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e0",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "e0",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [{"source": "input", "target": "e0"}],
                },
                {
                    "name": "layer1",
                    "readout_population": "e1",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "e1",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [
                        {"source": "input_e", "target": "e1"},
                        {"source": "input_i", "target": "e1"},
                    ],
                },
            ]
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=3)
    layer1 = net.layers[1]
    assert layer1.input_dim == 5
    assert layer1.external_input_dims["input_e"] == 5
    assert layer1.external_input_dims["input_i"] == 5

    y = net(torch.randn(2, 3))
    assert y.shape == (2, 4)
    assert torch.isfinite(y).all()


def test_multi_population_qualified_cross_layer_sources_and_gradients():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e0",
                    "population_defaults": _base_population(
                        {
                            "ff_excitatory_synapses": 2,
                            "ff_inhibitory_synapses": 0,
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "i0",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                        },
                        {
                            "name": "e0",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [1],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i0"},
                        {"source": "input", "target": "e0"},
                    ],
                },
                {
                    "name": "layer1",
                    "readout_population": "e1",
                    "population_defaults": _base_population(
                        {
                            "ff_excitatory_synapses": 2,
                            "ff_inhibitory_synapses": 1,
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "e1",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [
                        {"source": "layer0.e0", "target": "e1"},
                        {"source": "layer0.i0", "target": "e1"},
                    ],
                },
            ]
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=4)
    layer1 = net.layers[1]
    assert layer1._stream_sources("e1", "same_step", "ff_excitatory") == ["layer0.e0"]
    assert layer1._stream_sources("e1", "same_step", "ff_inhibitory") == ["layer0.i0"]
    assert layer1._stream_dim("e1", "same_step", "ff_excitatory") == 5
    assert layer1._stream_dim("e1", "same_step", "ff_inhibitory") == 3

    x = torch.randn(3, 4, requires_grad=True)
    y = net(x)
    assert y.shape == (3, 4)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    i0_grad = sum(
        (
            param.grad.detach().abs().sum()
            if param.grad is not None
            else torch.tensor(0.0)
        )
        for param in net.layers[0].populations["i0"].parameters()
    )
    assert i0_grad.item() > 0.0


def test_multi_population_layer_state_has_recurrent_state_helpers():
    cfg = _single_layer_temporal_population_config({"soma_feedback_enabled": True})
    net = get_architecture("multi_population_ei", cfg, input_dim=4)
    state = net.init_state(batch_size=2, device=torch.device("cpu"))[0]

    cloned = state.clone()
    detached = state.detach()
    moved = state.to(torch.device("cpu"))

    assert cloned.outputs["e"] is not state.outputs["e"]
    assert torch.equal(cloned.outputs["e"], state.outputs["e"])
    assert detached.outputs["e"].requires_grad is False
    assert moved.outputs["e"].device.type == "cpu"
    assert set(moved.population_states) == set(state.population_states)


def test_multi_population_probability_masks_include_pathway_in_seed():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "connection_seed": 7,
                    "population_defaults": _base_population(
                        {
                            "ff_excitatory_synapses": 2,
                            "ff_inhibitory_synapses": 2,
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 8,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [
                        {
                            "source": "input",
                            "target": "e",
                            "pathway": "ff_excitatory",
                            "probability": 0.5,
                        },
                        {
                            "source": "input",
                            "target": "e",
                            "pathway": "ff_inhibitory",
                            "probability": 0.5,
                        },
                    ],
                }
            ]
        }
    }

    net = get_architecture("multi_population_ei", cfg, input_dim=8)
    branch = net.layers[0].populations["e"].branch_layers[0]
    exc_mask = branch.branch_excitation.connection_mask
    inh_mask = branch.branch_inhibition.connection_mask

    assert exc_mask.shape == inh_mask.shape
    assert not torch.equal(exc_mask, inh_mask)


def test_multi_population_qualified_sources_validate_layer_order_and_names():
    future_cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "layer0",
                    "populations": [
                        {
                            "name": "e0",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [
                        {"source": "input", "target": "e0"},
                        {"source": "layer1.e1", "target": "e0"},
                    ],
                },
                {
                    "name": "layer1",
                    "populations": [
                        {
                            "name": "e1",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [{"source": "input", "target": "e1"}],
                },
            ]
        }
    }
    with pytest.raises(ValueError, match="earlier layers"):
        get_architecture("multi_population_ei", future_cfg, input_dim=3)

    missing_cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "layer0",
                    "populations": [
                        {
                            "name": "e0",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [{"source": "input", "target": "e0"}],
                },
                {
                    "name": "layer1",
                    "populations": [
                        {
                            "name": "e1",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [
                        {"source": "input", "target": "e1"},
                        {"source": "layer0.missing", "target": "e1"},
                    ],
                },
            ]
        }
    }
    with pytest.raises(ValueError, match="unknown population"):
        get_architecture("multi_population_ei", missing_cfg, input_dim=3)


def test_multi_population_delayed_qualified_future_source_is_recurrent_safe():
    cfg = {
        "multi_population": {
            "output_mode": "last",
            "layers": [
                {
                    "name": "layer0",
                    "recurrent": True,
                    "readout_population": "e0",
                    "population_defaults": _base_population(),
                    "populations": [
                        {
                            "name": "e0",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [
                        {"source": "input", "target": "e0"},
                        {
                            "source": "layer1.e1",
                            "target": "e0",
                            "timing": "delayed",
                        },
                    ],
                },
                {
                    "name": "layer1",
                    "readout_population": "e1",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "e1",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [{"source": "layer0.e0", "target": "e1"}],
                },
            ],
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=3)
    assert net.is_recurrent is True
    assert net.layers[0]._stream_sources("e0", "delayed", "rec_excitatory") == [
        "layer1.e1"
    ]
    assert net.layers[0]._stream_dim("e0", "delayed", "rec_excitatory") == 4

    x = torch.randn(2, 5, 3, requires_grad=True)
    y = net(x)
    assert y.shape == (2, 4)
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_multi_population_local_learning_recorders_see_qualified_streams():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e0",
                    "population_defaults": _base_population(
                        {
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "i0",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                        },
                        {
                            "name": "e0",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [1],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "i0"},
                        {"source": "input", "target": "e0"},
                    ],
                },
                {
                    "name": "layer1",
                    "readout_population": "e1",
                    "population_defaults": _base_population(
                        {
                            "ff_excitatory_synapses": 2,
                            "ff_inhibitory_synapses": 1,
                            "rec_excitatory_synapses": 0,
                            "rec_inhibitory_synapses": 0,
                        }
                    ),
                    "populations": [
                        {
                            "name": "e1",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        }
                    ],
                    "connections": [
                        {"source": "layer0.e0", "target": "e1"},
                        {"source": "layer0.i0", "target": "e1"},
                    ],
                },
            ]
        }
    }
    core = get_architecture("multi_population_ei", cfg, input_dim=4)
    model = Classifier(
        encoder_network=Identity(input_dim=4),
        core_network=core,
        decoder_network=Identity(input_dim=4),
    )
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer.local_cfg = trainer._build_local_rule_config({})
    trainer._layer_stats = {}
    trainer._decoder_cache = {}
    trainer._broadcast_cache = {}
    trainer._stdp_traces = {}

    records, handles = trainer._attach_local_recorders(model)
    try:
        y = model(torch.randn(2, 4))
    finally:
        for handle in handles:
            handle.remove()

    assert y.shape == (2, 4)
    assert any(
        rec.get("x_exc") is not None and rec["x_exc"].shape[-1] == 5 for rec in records
    )
    assert any(
        rec.get("x_inh") is not None and rec["x_inh"].shape[-1] == 3 for rec in records
    )


def test_multi_population_same_step_bidirectional_cycle_errors():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "bad",
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        },
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 2,
                            "branch_factors": [1],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "e"},
                        {"source": "input", "target": "i"},
                        {"source": "e", "target": "i"},
                        {"source": "i", "target": "e"},
                    ],
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="acyclic"):
        get_architecture("multi_population_ei", cfg, input_dim=3)


def test_multi_population_recurrent_delayed_bidirectional_edges_are_allowed():
    cfg = {
        "multi_population": {
            "output_mode": "last",
            "layers": [
                {
                    "name": "recurrent",
                    "recurrent": True,
                    "readout_population": "e",
                    "population_defaults": _base_population(),
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 5,
                            "branch_factors": [1],
                        },
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 3,
                            "branch_factors": [1],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "e"},
                        {"source": "input", "target": "i"},
                        {
                            "source": "e",
                            "target": "i",
                            "timing": "delayed",
                        },
                        {
                            "source": "i",
                            "target": "e",
                            "timing": "delayed",
                        },
                    ],
                }
            ],
        }
    }
    net = get_architecture("multi_population_ei", cfg, input_dim=4)
    assert net.is_recurrent is True

    y = net(torch.randn(2, 6, 4))
    assert y.shape == (2, 5)
    assert torch.isfinite(y).all()


def test_multi_population_delayed_edges_require_recurrent_layer():
    cfg = {
        "multi_population": {
            "layers": [
                {
                    "name": "bad",
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 4,
                            "branch_factors": [1],
                        },
                        {
                            "name": "i",
                            "polarity": "inhibitory",
                            "n_neurons": 2,
                            "branch_factors": [1],
                        },
                    ],
                    "connections": [
                        {"source": "input", "target": "e"},
                        {"source": "input", "target": "i"},
                        {"source": "i", "target": "e", "timing": "delayed"},
                    ],
                }
            ]
        }
    }
    with pytest.raises(ValueError, match="recurrent=true"):
        get_architecture("multi_population_ei", cfg, input_dim=3)
