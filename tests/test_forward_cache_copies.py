"""Models deep-copy after a training forward and then train on identically.

Modules keep tensors of their last forward pass for analyses (population
outputs, routing information, branch currents, the pruned weights of a
gradient-scaling hook). Copies and pickles hold them without their autograd
graph; the module itself and what analyses read from it are unchanged.
"""

from __future__ import annotations

import copy
import pickle
from collections import defaultdict
from typing import NamedTuple

import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.utils.forward_caches import detach_graph_tensors


def _population(*, recurrent=False, **defaults):
    config = copy.deepcopy(_config("population_network", "shunting", "param_tanh"))
    layer = config["population_network"]["layers"][0]
    layer["population_defaults"].update(defaults)
    if recurrent:
        layer["recurrent"] = True
        layer["population_defaults"]["rec_excitatory_synapses"] = 1
        layer["connections"].append({"source": "e", "target": "e", "timing": "delayed"})
    return "population_network", config, (5, 6) if not recurrent else (3, 4, 6)


def _einet():
    config = copy.deepcopy(_config("einet", "shunting", "param_tanh"))
    config["sparsity"] = {"gradient_scaling": "conductance_dynamic"}
    config["blocklinear"] = {"gradient_scaling": "conductance_dynamic"}
    return "einet", config, (5, 6)


def _recurrent_ei():
    config = {
        "biological_neuron": False,
        "architecture": {
            "excitatory_layer_sizes": [4],
            "inhibitory_layer_sizes": [],
            "excitatory_branch_factors": [2],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [0],
            "rec_ee_synapses_per_branch": [1],
        },
        "morphology": {"use_shunting": True, "somatic_synapses": True},
        "blocklinear": {"gradient_scaling": "conductance_dynamic"},
        "reactivation": {"enabled": False},
    }
    return "dendritic_shunting", config, (2, 3, 5)


CASES = {
    "population_feedforward": lambda: _population(),
    "population_recurrent_routing": lambda: _population(recurrent=True),
    "population_release": lambda: _population(
        topk_type="indexed_release", indexed_seed=3, weight_transform="exp"
    ),
    "population_deepst_scaled": lambda: _population(
        topk_type="deepst",
        excitatory_target_density=0.5,
        topk_gradient_scaling="conductance_dynamic",
    ),
    "einet_analysis_currents": _einet,
    "recurrent_ei_routing": _recurrent_ei,
}


def _build(name):
    family, config, shape = CASES[name]()
    torch.manual_seed(1)
    model = get_architecture(family, config, input_dim=shape[-1])
    if hasattr(model, "store_routing"):
        model.store_routing = True
    for module in model.modules():
        if isinstance(module, DendriticBranchLayer):
            module.set_branch_diagnostics(True)
            # Stateful steps bypass branch.forward and reject graph capture.
            # They still populate routing/state caches, tested below.
            if name not in {"population_recurrent_routing", "recurrent_ei_routing"}:
                module.set_analysis_currents(True)
    return model, shape


def _train_step(model, optimizer, shape, step):
    x = torch.rand(shape, generator=torch.Generator().manual_seed(100 + step))
    torch.manual_seed(200 + step)  # release draws, noise
    model.train()
    optimizer.zero_grad(set_to_none=True)
    out = model(x)
    out = out[0] if isinstance(out, tuple) else out
    weights = torch.randn(out.shape, generator=torch.Generator().manual_seed(7))
    (out * weights).sum().backward()
    optimizer.step()


def _cached_tensors(model):
    """Every tensor in the forward caches of every module, by path."""
    found = {}

    def collect(value, path):
        if torch.is_tensor(value):
            found[path] = value
        elif isinstance(value, dict):
            for key, item in value.items():
                collect(item, f"{path}[{key!r}]")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                collect(item, f"{path}[{index}]")

    for name, module in model.named_modules():
        for attribute in getattr(module, "_FORWARD_CACHES", ()):
            collect(getattr(module, attribute, None), f"{name}.{attribute}")
    return found


@pytest.mark.parametrize("name", sorted(CASES))
def test_deepcopy_after_a_training_forward_trains_on_identically(name):
    model, shape = _build(name)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    _train_step(model, optimizer, shape, 0)
    cached = _cached_tensors(model)
    assert any(value.grad_fn is not None for value in cached.values())

    duplicate = copy.deepcopy(model)

    # The original keeps its caches; the copy holds their values detached.
    copied = _cached_tensors(duplicate)
    assert copied.keys() == cached.keys()
    for path, value in cached.items():
        assert torch.equal(copied[path], value.detach()), path
        assert copied[path].grad_fn is None, path
    assert any(value.grad_fn is not None for value in _cached_tensors(model).values())

    duplicate_optimizer = torch.optim.SGD(duplicate.parameters(), lr=0.05, momentum=0.9)
    # load_state_dict keeps the given momentum tensors, so copy them.
    duplicate_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    for step in range(1, 4):
        _train_step(model, optimizer, shape, step)
        _train_step(duplicate, duplicate_optimizer, shape, step)
    for (key, a), (_, b) in zip(
        model.state_dict().items(), duplicate.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), key


def test_pickled_models_keep_detached_forward_caches():
    model, shape = _build("population_recurrent_routing")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    _train_step(model, optimizer, shape, 0)
    restored = pickle.loads(pickle.dumps(model))
    cached = _cached_tensors(model)
    assert cached
    for path, value in _cached_tensors(restored).items():
        assert torch.equal(value, cached[path].detach()), path
        assert value.grad_fn is None and not value.requires_grad, path


def test_detach_graph_tensors_rebuilds_containers_only():
    leaf = torch.ones(2, requires_grad=True)
    graph = leaf * 2

    class Pair(NamedTuple):
        first: torch.Tensor
        second: int

    counts = defaultdict(list, {"a": [graph]})
    value = {"graph": graph, "leaf": leaf, "pair": Pair(graph, 3), "counts": counts}

    detached = detach_graph_tensors(value)

    assert detached["graph"].grad_fn is None
    assert torch.equal(detached["graph"], graph)
    assert detached["leaf"] is leaf
    assert isinstance(detached["pair"], Pair) and detached["pair"].second == 3
    assert detached["pair"].first.grad_fn is None
    assert isinstance(detached["counts"], defaultdict)
    assert detached["counts"]["a"][0].grad_fn is None
    # The owner's containers keep their graph tensors.
    assert value["graph"].grad_fn is not None
    assert counts["a"][0].grad_fn is not None
