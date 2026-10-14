from copy import deepcopy

import pytest
import torch
from test_operator_config_contract import _config

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.activations.endpoint_tanh import EndpointNormalizedTanh
from dendritic_modeling.networks.architectures.factory import get_architecture


def test_endpoint_equation_gradients_and_saturation():
    gate = EndpointNormalizedTanh(3, 3, 1.5).double()
    x = torch.linspace(0, 1, 15).reshape(5, 3).double().requires_grad_()
    m, b = gate.log_m.exp(), gate.b
    expected = (torch.tanh(m * x - b) - torch.tanh(-b)) / (
        torch.tanh(m - b) - torch.tanh(-b)
    )
    torch.testing.assert_close(gate(x), expected)
    for a, r in zip(
        torch.autograd.grad(gate(x).sum(), (x, gate.log_m, gate.b)),
        torch.autograd.grad(expected.sum(), (x, gate.log_m, gate.b)),
    ):
        torch.testing.assert_close(a, r)
    for slope, offset in [(0.00001, 30), (3, -40), (20, 40)]:
        gate.initialize(slope, offset)
        y = gate(torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double))
        torch.testing.assert_close(
            y, torch.tensor([[0.0] * 3, [1.0] * 3], dtype=torch.double)
        )
        assert torch.isfinite(gate(x)).all()


@pytest.mark.parametrize("family", ["population_network", "einet"])
def test_leak_and_gate_reach_every_branch_and_checkpoint(family):
    config = _config(family, "shunting", "endpoint_tanh")
    options = (
        config["morphology"]
        if family == "einet"
        else config["population_network"]["layers"][0]["population_defaults"]
    )
    options.update(leak_conductance=0.03, learnable_leak=True)
    model = get_architecture(family, deepcopy(config), input_dim=6)
    branches = [m for m in model.modules() if isinstance(m, DendriticBranchLayer)]
    for branch in branches:
        assert branch.learnable_leak
        assert isinstance(branch.reactivation, EndpointNormalizedTanh)
        assert branch.log_leak.numel() == branch.n_branches
        E = torch.full((2, branch.n_branches), 0.4, requires_grad=True)
        inhibitory = torch.full_like(E, 0.2, requires_grad=True)
        actual, _ = branch.voltage_from_currents(E, 0, inhibitory)
        expected = E / (E + inhibitory + branch.log_leak.exp() + branch.epsilon)
        torch.testing.assert_close(actual, expected)
        a = torch.autograd.grad(
            actual.sum(), (E, inhibitory, branch.log_leak), retain_graph=True
        )
        r = torch.autograd.grad(expected.sum(), (E, inhibitory, branch.log_leak))
        for x, y in zip(a, r):
            torch.testing.assert_close(x, y)
    x = torch.rand(2, 6)
    output = model(x)
    output.sum().backward()
    assert all(
        b.log_leak.grad is not None and torch.isfinite(b.log_leak.grad).all()
        for b in branches
    )
    replay = get_architecture(family, deepcopy(config), input_dim=6)
    replay.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(replay(x), output, rtol=0, atol=0)


def test_default_leak_does_not_change_state_output_or_rng():
    kwargs = {
        "output_dim": 3,
        "excitatory_input_dim": 6,
        "excitatory_synapses_per_branch": 2,
    }
    torch.manual_seed(32)
    a = DendriticBranchLayer(**kwargs)
    state = torch.get_rng_state()
    torch.manual_seed(32)
    b = DendriticBranchLayer(**kwargs, leak_conductance=1.0, learnable_leak=False)
    assert torch.equal(state, torch.get_rng_state())
    assert list(a.state_dict()) == list(b.state_dict())
    assert not any("leak" in key for key in a.state_dict())
    x = torch.rand(2, 6)
    torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)


@pytest.mark.parametrize("family", ["population_network", "einet"])
def test_release_config_reaches_both_builders(family):
    config = _config(family, "shunting", "endpoint_tanh")
    if family == "population_network":
        options = config["population_network"]["layers"][0]["population_defaults"]
        options.update(
            topk_type="indexed_release",
            weight_transform="exp",
            release_p_min=0.75,
            release_weight_scale=0.123,
            dbl_init_method="fan_in_lognormal",
        )
    else:
        config["morphology"].update(
            weight_transform="exp", dbl_init_method="fan_in_lognormal"
        )
        config["sparsity"] = {
            "type": "indexed_release",
            "indexed": {"release_p_min": 0.75, "release_weight_scale": 0.123},
        }
    model = get_architecture(family, config, input_dim=6)
    branches = [m for m in model.modules() if isinstance(m, DendriticBranchLayer)]
    for branch in branches:
        if branch.branch_excitation is not None:
            assert branch.branch_excitation.release_p_min == 0.75
            assert branch.branch_excitation.release_weight_scale == 0.123


def test_analytic_local_training_rejects_new_compartment_options():
    from dendritic_modeling.training.strategies.local_learning import (
        LocalCreditAssignment,
    )

    model = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=4,
        excitatory_synapses_per_branch=2,
        learnable_leak=True,
    )
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer._decoder_cache = {}
    with pytest.raises(ValueError, match="analytic local"):
        trainer._attach_local_recorders(model)
