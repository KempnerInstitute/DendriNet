"""Population containers forward sparse maintenance and rewiring to branch layers."""

import math
from copy import deepcopy

import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_rewire import (
    IndexedRewireLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent import (
    EILayerConfig,
    EINetwork,
    EINetworkConfig,
    PopulationConfig,
)
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer


def population_network(topk_type="standard", **options):
    config = _config("population_network", "shunting", "param_tanh")
    defaults = config["population_network"]["layers"][0]["population_defaults"]
    defaults.update(
        topk_type=topk_type,
        weight_transform="exp",
        ff_excitatory_synapses=3,
        indexed_seed=7,
        **options,
    )
    torch.manual_seed(13)
    return get_architecture("population_network", config, input_dim=6)


def ei_network():
    population = {
        "branch_factors": [2],
        "ff_excitatory_synapses": 3,
        "topk_type": "standard",
    }
    cfg = EINetworkConfig(
        input_dim=8,
        layers=[
            EILayerConfig(
                excitatory=PopulationConfig(n_neurons=4, **population),
                inhibitory=PopulationConfig(n_neurons=2, **population),
                recurrent=False,
                dt=1.0,
            )
        ],
    )
    torch.manual_seed(17)
    return EINetwork(cfg)


def branch_layers(model):
    return [m for m in model.modules() if isinstance(m, DendriticBranchLayer)]


def synapses(model, kind):
    return [m for m in model.modules() if isinstance(m, kind)]


@pytest.mark.parametrize("build", [population_network, ei_network])
def test_maintenance_reaches_every_population_branch_layer(build):
    model = build()
    reference = deepcopy(model)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    masks = {id(b): b.weight_mask().bool() for b in synapses(model, TopKLinear)}
    model.decay_weights(0.1, weight_boosting=True)
    for layer in branch_layers(reference):
        layer.decay_weights(0.1, True)
    after = model.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in after)
    for key, value in reference.state_dict().items():
        assert torch.equal(after[key], value), key
    for bank in synapses(model, TopKLinear):
        name = next(k for k, v in model.named_modules() if v is bank)
        old = before[name + ".pre_w"]
        mask = masks[id(bank)]  # selection before the maintenance step
        torch.testing.assert_close(bank.pre_w[mask], old[mask] + math.log(0.9))
        torch.testing.assert_close(bank.pre_w[~mask], old[~mask] + math.log(1.1))


def test_trainer_wrapper_applies_maintenance_before_the_update():
    model = population_network()
    reference = deepcopy(model)
    x = torch.rand(5, 6)
    wrapped = CustomWeightDecayOptimizer(
        model, torch.optim.Adam(model.parameters(), lr=0.01), 0.01, True
    )
    plain = torch.optim.Adam(reference.parameters(), lr=0.01)
    for net, optimizer in ((model, wrapped), (reference, plain)):
        optimizer.zero_grad()
        net(x).square().sum().backward()
    wrapped.step()
    for layer in branch_layers(reference):
        layer.decay_weights(0.01 * 0.01, True)
    plain.step()
    for (name, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        assert torch.equal(p, q), name


def test_zero_maintenance_and_non_rewiring_synapses_keep_plain_updates():
    model = population_network()
    reference = deepcopy(model)
    x = torch.rand(5, 6)
    wrapped = CustomWeightDecayOptimizer(
        model, torch.optim.Adam(model.parameters(), lr=0.01), 0.0, False
    )
    plain = torch.optim.Adam(reference.parameters(), lr=0.01)
    for net, optimizer in ((model, wrapped), (reference, plain)):
        optimizer.zero_grad()
        net(x).square().sum().backward()
        optimizer.step()
    for key, value in reference.state_dict().items():
        assert torch.equal(model.state_dict()[key], value), key


def test_rewiring_reaches_indexed_rewire_populations():
    model = population_network(
        "indexed_rewire", indexed_rewire_frequency=1, indexed_rewire_quantile=0.5
    )
    banks = synapses(model, IndexedRewireLinear)
    assert banks
    optimizer = CustomWeightDecayOptimizer(
        model, torch.optim.Adam(model.parameters(), lr=0.01), 0.0, False
    )
    before = [bank.connection_indices.clone() for bank in banks]
    for _ in range(3):
        optimizer.zero_grad()
        model(torch.rand(5, 6)).square().sum().backward()
        optimizer.step()
    changed = [
        not torch.equal(old, bank.connection_indices)
        for old, bank in zip(before, banks)
    ]
    assert any(changed)
