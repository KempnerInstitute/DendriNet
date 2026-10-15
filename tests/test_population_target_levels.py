"""Projection target_levels: inhibitory classes restricted to dendritic depths."""

from pathlib import Path

import pytest
import torch

from dendritic_modeling.config import load_config
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.impact_rewiring import (
    build_population_impact_rewiring,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent.population_configs import (
    PopulationProjectionConfig,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "examples"


def test_target_levels_validate_and_map_to_depths():
    conn = PopulationProjectionConfig(
        source="p", target="e", target_levels=["soma", "primary"]
    )
    assert conn.target_depths(3) == {0, 1}
    distal = PopulationProjectionConfig(
        source="t", target="e", target_levels=["distal", 1]
    )
    assert distal.target_depths(3) == {1, 2}
    assert PopulationProjectionConfig(source="p", target="e").target_depths(3) is None
    for bad in ([], ["apical"], [-1], [True]):
        with pytest.raises(ValueError):
            PopulationProjectionConfig(source="p", target="e", target_levels=bad)
    with pytest.raises(ValueError, match="does not exist"):
        PopulationProjectionConfig(
            source="p", target="e", target_levels=[3]
        ).target_depths(3)


def _population(name, n_neurons, polarity, inhibitory):
    return {
        "name": name,
        "n_neurons": n_neurons,
        "polarity": polarity,
        "branch_factors": [2, 2],
        "population": {"ff_inhibitory_synapses_by_level": list(inhibitory)},
    }


def _network(connections, *, e_inhibitory=(2, 2, 2), seed=3):
    defaults = {
        "ff_excitatory_synapses": 3,
        "ff_inhibitory_synapses": 2,
        "rec_excitatory_synapses": 0,
        "rec_inhibitory_synapses": 0,
        "topk_type": "indexed_release",
        "weight_transform": "exp",
        "use_shunting": True,
        "learnable_leak": True,
        "leak_conductance": 0.03,
        "reactivation_type": "endpoint_tanh",
        "reactivation_init_policy": "fixed",
        "dbl_init_method": "fan_in_lognormal",
        "indexed_seed": seed,
    }
    layer = {
        "name": "layer0",
        "readout_population": "e",
        "population_defaults": defaults,
        "populations": [
            _population("p", 3, "inhibitory", (0, 0, 0)),
            _population("t", 3, "inhibitory", (0, 0, 0)),
            _population("e", 4, "excitatory", e_inhibitory),
        ],
        "connections": connections,
    }
    torch.manual_seed(seed)
    return get_architecture(
        "population_network",
        {
            "biological_neuron": False,
            "population_network": {"input_dim": 8, "layers": [layer]},
        },
        input_dim=8,
    )


P_T = [
    {"source": "input", "target": "p"},
    {"source": "input", "target": "t"},
    {"source": "input", "target": "e"},
    {"source": "p", "target": "e", "target_levels": ["soma", "primary"]},
    {"source": "t", "target": "e", "target_levels": ["distal"]},
]


def _legal(level):
    return set(range(3, 6)) if level == 0 else set(range(3))  # columns: p then t


def test_inhibitory_contacts_use_only_permitted_sources_per_depth():
    network = _network(P_T)
    e = network.layers[0].populations["e"]
    for level, branch in enumerate(e.branch_layers):  # distal, primary, soma
        indices = set(branch.branch_inhibition.connection_indices.flatten().tolist())
        assert indices <= _legal(level), (level, indices)
    output = network(torch.rand(5, 8))
    assert torch.isfinite(output).all()


def test_unreachable_depth_with_contacts_fails_explicitly():
    connections = [dict(c) for c in P_T]
    # distal keeps contacts but no projection may reach it
    connections[-1]["target_levels"] = ["primary"]
    with pytest.raises(ValueError, match="target_levels"):
        _network(connections)
    # A depth without contacts may be left without sources.
    assert _network(connections, e_inhibitory=(0, 2, 2)) is not None


def test_regrowth_stays_within_permitted_sources():
    network = _network(P_T)
    controller = build_population_impact_rewiring(
        network,
        {
            "frequency": 1,
            "maturity_steps": 1,
            "threshold": 1.0,
            "eligible_fraction": 1.0,
            "segment_fraction": 1.0,
        },
        steps_per_epoch=10,
        epochs=1,
        seed=0,
    )
    optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
    replaced = 0
    for _ in range(4):
        optimizer.zero_grad()
        network(torch.rand(6, 8)).sum().backward()
        optimizer.step()
        replaced += controller.step(optimizer)
    assert replaced > 0
    e = network.layers[0].populations["e"]
    for level, branch in enumerate(e.branch_layers):
        indices = set(branch.branch_inhibition.connection_indices.flatten().tolist())
        assert indices <= _legal(level)


def test_target_level_example_orders_populations_and_keeps_default_masks():
    config = load_config(str(CONFIGS / "population_network_target_levels.yaml"))
    config.model.encoder.params.input_dim = 816
    config.model.encoder.params.input_shape = [816]
    torch.manual_seed(0)
    model, _ = initialize_model(config.model)
    assert [layer._same_step_order for layer in model.core_network.layers] == [
        ["p", "t", "e"]
    ] * 2
    one_i = _network(
        [
            {"source": "input", "target": "p"},
            {"source": "input", "target": "t"},
            {"source": "input", "target": "e"},
            {"source": "p", "target": "e"},
            {"source": "t", "target": "e"},
        ]
    )
    layer = one_i.layers[0]
    population = layer._population_by_name["e"]
    assert (
        layer._build_connection_masks_for_population(
            population, layer.population_configs["e"]
        )
        == {}
    )


@pytest.mark.parametrize(
    "name, bank_type",
    [("population_network_target_levels", "VarianceTopKLinear")],
)
def test_target_level_example_selects_only_legal_inhibitory_sources(name, bank_type):
    config = load_config(str(CONFIGS / f"{name}.yaml"))
    config.model.encoder.params.input_dim = 816
    config.model.encoder.params.input_shape = [816]
    torch.manual_seed(0)
    model, _ = initialize_model(config.model)
    banks = [m for m in model.modules() if type(m).__name__ == bank_type]
    stored = sum(p.numel() for p in model.parameters())
    candidates = sum(m.pre_w.numel() for m in banks)
    active = sum(int(m.weight_mask().bool().sum()) for m in banks)
    assert stored == 2891231 and stored - candidates + active == 238367
    assert [layer._same_step_order for layer in model.core_network.layers] == [
        ["p", "t", "e"]
    ] * 2
    inhibitory = [
        branch.branch_inhibition
        for layer in model.core_network.layers
        for branch in layer.populations["e"].branch_layers
    ]
    for scramble in (False, True):  # selection follows the weights; legality must not
        for bank in inhibitory:
            if scramble:
                with torch.no_grad():
                    bank.pre_w.normal_()
            legal = bank.connection_mask.bool()
            assert not legal.all()
            assert not bool((bank.weight_mask().bool() & ~legal).any())
