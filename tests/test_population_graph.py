"""Tests for named cell-type population graph recurrent core."""

import pytest
import torch

from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent import (
    CellTypeConfig,
    PopulationConnectionConfig,
    PopulationGraphConfig,
    PopulationGraphNetwork,
    PopulationNetwork,
)
from dendritic_modeling.networks.architectures.recurrent.population_graph import (
    _lower_population_graph_config,
)


def _small_population_graph_config():
    return PopulationGraphConfig(
        input_dim=6,
        readout_population="pyr",
        output_mode="last",
        dt=1.0,
        population_defaults={
            "ff_excitatory_synapses": 3,
            "ff_inhibitory_synapses": 0,
            "rec_excitatory_synapses": 2,
            "rec_inhibitory_synapses": 1,
            "reactivate": False,
            "somatic_synapses": True,
            "dbl_init_method": "default",
            "weight_transform": "exp",
        },
        cell_types=[
            CellTypeConfig(
                name="pyr",
                polarity="excitatory",
                n_neurons=8,
                branch_factors=[1],
            ),
            CellTypeConfig(
                name="pv",
                polarity="inhibitory",
                n_neurons=3,
                branch_factors=[1],
            ),
            CellTypeConfig(
                name="sst",
                polarity="inhibitory",
                n_neurons=2,
                branch_factors=[1],
            ),
        ],
    )


def _feedforward_population_graph_config():
    return PopulationGraphConfig(
        input_dim=6,
        readout_population="pyr",
        output_mode="last",
        dt=1.0,
        population_defaults={
            "ff_excitatory_synapses": 5,
            "ff_inhibitory_synapses": 0,
            "rec_excitatory_synapses": 0,
            "rec_inhibitory_synapses": 0,
            "reactivate": False,
            "somatic_synapses": True,
            "dbl_init_method": "default",
            "weight_transform": "exp",
            "synapse_types": {
                "enabled": True,
                "excitatory": [{"name": "AMPA", "fraction": 1.0, "tau": None}],
                "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
            },
            "structured_connectivity": {
                "enabled": True,
                "seed": 17,
                "pathways": {
                    "ee": {"method": "fixed_indegree", "indegree": 3},
                },
            },
        },
        cell_types=[
            CellTypeConfig(
                name="pyr",
                polarity="excitatory",
                n_neurons=8,
                branch_factors=[1],
            ),
        ],
    )


def test_population_graph_forward_shape_and_gradients():
    torch.manual_seed(0)
    net = PopulationGraphNetwork(_small_population_graph_config())
    x = torch.rand(4, 5, 6)

    y = net(x)
    loss = y.sum()
    loss.backward()

    assert y.shape == (4, 8)
    assert torch.isfinite(y).all()
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in net.parameters()
    )


def test_population_graph_output_mode_all():
    cfg = _small_population_graph_config()
    cfg.output_mode = "all"
    net = PopulationGraphNetwork(cfg)

    y = net(torch.rand(2, 4, 6))

    assert y.shape == (2, 4, 8)


def test_population_graph_factory_builds_core():
    params = {
        "population_graph": {
            "readout_population": "pyr",
            "output_mode": "last",
            "population_defaults": {
                "ff_excitatory_synapses": 2,
                "rec_excitatory_synapses": 1,
                "rec_inhibitory_synapses": 1,
                "reactivate": False,
                "somatic_synapses": True,
                "dbl_init_method": "default",
            },
            "cell_types": [
                {
                    "name": "pyr",
                    "polarity": "excitatory",
                    "n_neurons": 6,
                    "branch_factors": [1],
                },
                {
                    "name": "pv",
                    "polarity": "inhibitory",
                    "n_neurons": 2,
                    "branch_factors": [1],
                },
            ],
        }
    }

    net = get_architecture("population_graph", params, input_dim=5)
    y = net(torch.rand(3, 4, 5))

    assert isinstance(net, PopulationGraphNetwork)
    assert net.is_recurrent
    assert net.output_dim == 6
    assert y.shape == (3, 6)


def test_population_graph_lowers_to_population_network_equivalent_forward():
    cfg = _small_population_graph_config()
    cfg.connection_seed = 17
    cfg.connections = [
        PopulationConnectionConfig(source="pv", target="pyr", probability=0.5),
        PopulationConnectionConfig(source="sst", target="pyr", enabled=False),
    ]
    x = torch.rand(2, 4, 6)

    torch.manual_seed(123)
    graph = PopulationGraphNetwork(cfg)
    torch.manual_seed(123)
    direct = PopulationNetwork(
        _lower_population_graph_config(
            cfg,
            graph.cell_types,
            graph.connection_specs,
            graph.readout_population,
        )
    )

    torch.testing.assert_close(graph(x), direct(x))
    assert graph.is_recurrent == direct.is_recurrent
    assert graph.output_dim == direct.output_dim


def test_population_graph_without_recurrent_synapses_is_static_feedforward():
    torch.manual_seed(0)
    cfg = _feedforward_population_graph_config()
    cfg.population_defaults.pop("synapse_types")
    net = PopulationGraphNetwork(cfg)
    x = torch.rand(4, 6)

    y = net(x)
    y.sum().backward()

    assert not net.is_recurrent
    assert y.shape == (4, 8)
    assert torch.isfinite(y).all()
    assert not net.populations["pyr"].synapse_types_enabled
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0 for p in net.parameters()
    )
    for branch_layer in net.populations["pyr"].branch_layers:
        assert branch_layer.branch_recurrent is None
        assert branch_layer.branch_rec_inhibition is None
        mask = branch_layer.branch_excitation.weight_mask()
        assert torch.equal(mask.sum(dim=1), torch.full((mask.shape[0],), 3.0))


def test_population_graph_synapse_types_mark_core_recurrent_without_rec_synapses():
    net = PopulationGraphNetwork(_feedforward_population_graph_config())

    y = net(torch.rand(4, 6))

    assert net.is_recurrent
    assert y.shape == (4, 8)
    assert torch.isfinite(y).all()
    assert net.populations["pyr"].synapse_types_enabled


def test_feedforward_population_graph_supports_spiking_soma():
    cfg = _feedforward_population_graph_config()
    cfg.population_defaults.update(
        {
            "dynamics_mode": "spike",
            "spike_readout": "membrane",
            "spike_threshold": 0.5,
        }
    )
    net = PopulationGraphNetwork(cfg)

    y = net(torch.rand(2, 6))

    assert net.is_recurrent
    assert net.populations["pyr"].spiking_soma is not None
    assert y.shape == (2, 8)
    assert torch.isfinite(y).all()


def test_population_graph_soma_feedback_marks_core_recurrent_without_rec_synapses():
    cfg = _feedforward_population_graph_config()
    cfg.population_defaults.update(
        {
            "soma_feedback_enabled": True,
            "soma_feedback_levels": "distal",
        }
    )
    net = PopulationGraphNetwork(cfg)

    assert net.is_recurrent


def test_population_graph_factory_builds_feedforward_core_without_rec_synapses():
    params = {
        "population_graph": {
            "readout_population": "pyr",
            "output_mode": "last",
            "population_defaults": {
                "ff_excitatory_synapses": 2,
                "ff_inhibitory_synapses": 0,
                "rec_excitatory_synapses": 0,
                "rec_inhibitory_synapses": 0,
                "reactivate": False,
                "somatic_synapses": True,
                "dbl_init_method": "default",
            },
            "cell_types": [
                {
                    "name": "pyr",
                    "polarity": "excitatory",
                    "n_neurons": 6,
                    "branch_factors": [1],
                },
            ],
        }
    }

    net = get_architecture("population_graph", params, input_dim=5)
    y = net(torch.rand(3, 5))

    assert isinstance(net, PopulationGraphNetwork)
    assert not net.is_recurrent
    assert net.output_dim == 6
    assert y.shape == (3, 6)


def test_population_graph_source_target_connection_mask_blocks_pair():
    cfg = _small_population_graph_config()
    cfg.connections = [
        PopulationConnectionConfig(source="sst", target="pyr", enabled=False)
    ]

    net = PopulationGraphNetwork(cfg)
    pyr = net.populations["pyr"]

    for branch_layer in pyr.branch_layers:
        mask = branch_layer.branch_rec_inhibition.connection_mask
        pv_width = net._cell_dims["pv"]
        assert mask.shape[1] == net._cell_dims["pv"] + net._cell_dims["sst"]
        assert bool(mask[:, :pv_width].all())
        assert not bool(mask[:, pv_width:].any())


def test_population_graph_connection_probability_is_seeded():
    cfg_a = _small_population_graph_config()
    cfg_a.connection_seed = 7
    cfg_a.connections = [
        PopulationConnectionConfig(source="pv", target="pyr", probability=0.5)
    ]
    cfg_b = _small_population_graph_config()
    cfg_b.connection_seed = 7
    cfg_b.connections = [
        PopulationConnectionConfig(source="pv", target="pyr", probability=0.5)
    ]

    net_a = PopulationGraphNetwork(cfg_a)
    net_b = PopulationGraphNetwork(cfg_b)
    mask_a = (
        net_a.populations["pyr"].branch_layers[0].branch_rec_inhibition.connection_mask
    )
    mask_b = (
        net_b.populations["pyr"].branch_layers[0].branch_rec_inhibition.connection_mask
    )

    assert torch.equal(mask_a, mask_b)


def test_population_graph_connection_probability_is_deterministic_without_seed():
    cfg_a = _small_population_graph_config()
    cfg_a.connection_seed = None
    cfg_a.connections = [
        PopulationConnectionConfig(source="pv", target="pyr", probability=0.5)
    ]
    cfg_b = _small_population_graph_config()
    cfg_b.connection_seed = None
    cfg_b.connections = [
        PopulationConnectionConfig(source="pv", target="pyr", probability=0.5)
    ]

    torch.manual_seed(1)
    net_a = PopulationGraphNetwork(cfg_a)
    torch.manual_seed(999)
    net_b = PopulationGraphNetwork(cfg_b)
    mask_a = (
        net_a.populations["pyr"].branch_layers[0].branch_rec_inhibition.connection_mask
    )
    mask_b = (
        net_b.populations["pyr"].branch_layers[0].branch_rec_inhibition.connection_mask
    )

    assert torch.equal(mask_a, mask_b)


def test_population_graph_probability_masks_repair_empty_rows():
    cfg = _small_population_graph_config()
    cfg.connection_seed = 23
    cfg.connections = [
        PopulationConnectionConfig(source="pv", target="pyr", probability=1e-6),
        PopulationConnectionConfig(source="sst", target="pyr", enabled=False),
    ]

    net = PopulationGraphNetwork(cfg)
    mask = net.populations["pyr"].branch_layers[0].branch_rec_inhibition.connection_mask

    assert bool((mask.sum(dim=1) > 0).all())


def test_population_graph_population_override_deep_merges_nested_defaults():
    cfg = _small_population_graph_config()
    cfg.population_defaults["synapse_types"] = {
        "enabled": True,
        "excitatory": [{"name": "AMPA", "fraction": 1.0, "tau": None}],
        "inhibitory": [{"name": "GABA_A", "fraction": 1.0, "tau": None}],
    }
    cfg.cell_types[0] = CellTypeConfig(
        name="pyr",
        polarity="excitatory",
        n_neurons=8,
        branch_factors=[1],
        population={"synapse_types": {"excitatory": [{"name": "AMPA", "tau": 5.0}]}},
    )

    net = PopulationGraphNetwork(cfg)
    pyr_cfg = net.population_configs["pyr"]

    assert pyr_cfg.synapse_types["enabled"] is True
    assert pyr_cfg.synapse_types["excitatory"][0]["tau"] == 5.0
    assert pyr_cfg.synapse_types["inhibitory"][0]["name"] == "GABA_A"


def test_population_graph_population_defaults_survive_cell_size_override():
    cfg = _small_population_graph_config()
    cfg.population_defaults.update(
        {
            "use_shunting": False,
            "synapse_types": {
                "enabled": True,
                "excitatory": [
                    {
                        "name": "AMPA",
                        "fraction": 1.0,
                        "tau": None,
                        "reversal_potential": 1.0,
                    }
                ],
                "inhibitory": [
                    {
                        "name": "GABA_A",
                        "fraction": 1.0,
                        "tau": None,
                        "reversal_potential": 0.0,
                    }
                ],
            },
        }
    )
    cfg.cell_types[0] = CellTypeConfig(
        name="pyr",
        polarity="excitatory",
        n_neurons=10,
        branch_factors=[1],
    )

    net = PopulationGraphNetwork(cfg)
    pyr_cfg = net.population_configs["pyr"]
    pyr = net.populations["pyr"]

    assert pyr_cfg.n_neurons == 10
    assert pyr_cfg.use_shunting is False
    assert pyr_cfg.synapse_types["enabled"] is True
    assert pyr.synapse_types_enabled is True
    assert pyr.pop_config.use_shunting is False


def test_population_graph_rejects_duplicate_cell_type_names():
    cfg = _small_population_graph_config()
    cfg.cell_types[1] = CellTypeConfig(
        name="pyr",
        polarity="inhibitory",
        n_neurons=2,
        branch_factors=[1],
    )

    with pytest.raises(ValueError, match="unique"):
        PopulationGraphNetwork(cfg)


def test_population_graph_rejects_unknown_connection_endpoint():
    cfg = _small_population_graph_config()
    cfg.connections = [
        PopulationConnectionConfig(source="vip", target="pyr", enabled=False)
    ]

    with pytest.raises(ValueError, match="source"):
        PopulationGraphNetwork(cfg)
