"""Conductance-unit changes preserve initial functions and log gradients."""

from copy import deepcopy

import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


def branch(scale=1.0, **overrides):
    kwargs = {
        "output_dim": 3,
        "excitatory_input_dim": 8,
        "inhibitory_input_dim": 4,
        "excitatory_synapses_per_branch": 3,
        "inhibitory_synapses_per_branch": 2,
        "input_branch_factor": 2,
        "topk_type": "indexed_release",
        "indexed_seed": 71,
        "weight_transform": "exp",
        "dbl_init_method": "fan_in_lognormal",
        "reactivation_type": "endpoint_tanh",
        "reactivation_init_policy": "fixed",
        "reactivation_init_m": 3.0,
        "reactivation_init_b": 1.5,
        "adaptive_initialization": False,
        "initial_child_conductance": 0.3,
        "initial_conductance_scale": scale,
        "leak_conductance": 0.03 * scale,
        "epsilon": 1e-08 * scale,
        "learnable_leak": True,
    }
    kwargs.update(overrides)
    torch.manual_seed(87)
    return DendriticBranchLayer(**kwargs)


@pytest.mark.parametrize("training", [False, True])
def test_scale_preserves_voltage_release_and_log_parameter_gradients(training):
    a, b = branch(), branch(1 / 0.03)
    a.train(training)
    b.train(training)
    x, i, c = torch.rand(4, 8), torch.rand(4, 4), torch.rand(4, 6)
    torch.manual_seed(73)
    y = a(x, inhibitory_input=i, branch_input=c)
    torch.manual_seed(73)
    z = b(x, inhibitory_input=i, branch_input=c)
    torch.testing.assert_close(y, z, rtol=3e-6, atol=3e-7)
    y.square().sum().backward()
    z.square().sum().backward()
    for (name, p), (other, q) in zip(a.named_parameters(), b.named_parameters()):
        assert name == other
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=3e-7)
    for pathway in ("branch_excitation", "branch_inhibition"):
        left, right = getattr(a, pathway), getattr(b, pathway)
        torch.testing.assert_close(
            left.release_probability(left.sparse_weight()),
            right.release_probability(right.sparse_weight()),
        )
    assert b.synapse_config.initialization.initial_conductance_scale == pytest.approx(
        1 / 0.03
    )
    replay = branch(1 / 0.03)
    replay.load_state_dict(b.state_dict(), strict=True)
    b.eval()
    replay.eval()
    torch.testing.assert_close(
        b(x, inhibitory_input=i, branch_input=c),
        replay(x, inhibitory_input=i, branch_input=c),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("family", ["einet", "population_network"])
def test_scale_routes_through_each_supported_builder(family):
    config = _config(family, "shunting", "endpoint_tanh")
    options = (
        config["population_network"]["layers"][0]["population_defaults"]
        if family == "population_network"
        else config["morphology"]
    )
    options.update(
        weight_transform="exp",
        dbl_init_method="fan_in_lognormal",
        leak_conductance=0.03,
    )
    if family == "population_network":
        options.update(topk_type="indexed_release")
    else:
        config["sparsity"] = {"type": "indexed_release"}
    torch.manual_seed(55)
    a = get_architecture(family, deepcopy(config), input_dim=6).eval()
    before_rng = torch.get_rng_state().clone()
    options["leak_conductance"] = 1.0
    (options if family == "population_network" else config["implementation"])[
        "initial_conductance_scale"
    ] = (1 / 0.03)
    torch.manual_seed(55)
    b = get_architecture(family, deepcopy(config), input_dim=6).eval()
    assert torch.equal(torch.get_rng_state(), before_rng)
    for layer in b.modules():
        if isinstance(layer, DendriticBranchLayer):
            assert layer.initial_conductance_scale == pytest.approx(1 / 0.03)
    x = torch.rand(5, 6)
    torch.testing.assert_close(a(x), b(x), rtol=2e-5, atol=3e-7)


def test_default_identity_explicit_scale_and_config_roundtrip():
    a = branch()
    before_rng = torch.get_rng_state().clone()
    b = branch(1.0)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    cfg = DendriticSynapseConfig.from_kwargs(
        {"dbl_init_method": "fan_in_lognormal", "initial_conductance_scale": 7.0}
    )
    assert cfg.to_kwargs()["initial_conductance_scale"] == 7.0
    b = branch(7.0, release_weight_scale=0.12)
    assert b.branch_excitation.release_weight_scale == 0.12


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf")])
def test_invalid_scale_rejected(scale):
    with pytest.raises(ValueError, match="finite and positive"):
        branch(scale, leak_conductance=0.03)


def test_other_initializer_rejects_scale():
    with pytest.raises(ValueError, match="requires fan_in_lognormal"):
        branch(2.0, dbl_init_method="default")
