"""Signal delivery for local rules with autograd eligibilities."""

from __future__ import annotations

import pytest
import torch
from test_dendrinet_level_overrides import (
    BASE,
    DIM,
    LEAVES,
    RECIPE,
    SOMA,
    STREAMS,
    _inputs,
    _typed,
)
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)
from dendritic_modeling.training import local_delivery as ld


def _network(seed=21):
    net, _ = _typed(seed)
    net.reinitialize(RECIPE)
    net = net.double()
    for layer in net.branch_layers:
        layer.set_branch_diagnostics(True)
    return net


def _loss(soma, target):
    return 0.5 * (soma.sum(-1) - target).square().mean()


def _full_chain_gains(net, inhibition):
    _, proximal, soma = net.branch_layers
    resistance = ld.expand_to_children(ld.relative_resistance(proximal, inhibition), 2)
    slope = ld.expand_to_children(ld.activation_slope(proximal), 2)
    soma_ratio = ld.coupling_ratios(soma).reshape(-1)
    terminal = resistance * slope * ld.coupling_ratios(proximal).reshape(-1)
    terminal = terminal * ld.expand_to_children(soma_ratio, 2)
    return [terminal, soma_ratio.expand(inhibition.shape[0], -1)]


def test_full_chain_delivery_equals_backpropagation():
    net = _network()
    excitation, inhibition = _inputs(3, batch=7)
    target = torch.linspace(-1, 1, 7, dtype=torch.float64)
    parameters = dict(net.named_parameters())
    loss = _loss(net(excitation, inhibition), target)
    exact = torch.autograd.grad(loss, list(parameters.values()))
    outputs = net.forward_levels(excitation, inhibition, detach_child_inputs=True)
    loss = _loss(outputs[-1], target)
    error = ld.soma_error(loss, outputs[-1])
    objective = loss + ld.delivered_error_objective(
        outputs[:-1], error, _full_chain_gains(net, inhibition)
    )
    local = torch.autograd.grad(objective, list(parameters.values()))
    for name, a, b in zip(parameters, exact, local):
        assert torch.allclose(a, b, rtol=1e-12, atol=1e-15), name
    # Without the coupling ratios the delivered errors are no longer exact.
    outputs = net.forward_levels(excitation, inhibition, detach_child_inputs=True)
    loss = _loss(outputs[-1], target)
    error = ld.soma_error(loss, outputs[-1])
    objective = loss + ld.delivered_error_objective(outputs[:-1], error, [None, None])
    broadcast = torch.autograd.grad(objective, list(parameters.values()))
    assert not torch.allclose(exact[0], broadcast[0], rtol=1e-3, atol=0)


def test_gain_quantities_follow_their_definitions():
    net = _network(4)
    terminal, proximal, soma = net.branch_layers
    excitation, inhibition = _inputs(5)
    net(excitation, inhibition)
    base = 1 + proximal.branches_to_output.sum_conductances()
    inhibitory = proximal.branch_inhibition(inhibition)
    assert torch.equal(
        ld.relative_resistance(proximal, inhibition), base / (base + inhibitory)
    )
    voltage = proximal.get_branch_diagnostics()["V"]
    assert torch.equal(ld.activation_slope(proximal), 1 - voltage.tanh().square())
    assert torch.equal(
        ld.activation_slope(soma), torch.ones_like(soma.get_branch_diagnostics()["V"])
    )
    soma_base = 1 + soma.branches_to_output.sum_conductances()
    assert torch.equal(
        ld.coupling_ratios(soma), soma.branches_to_output.weight() / soma_base[:, None]
    )
    assert not ld.relative_resistance(proximal, inhibition).requires_grad
    with pytest.raises(ValueError, match="no inhibitory pathway"):
        ld.relative_resistance(soma, inhibition)
    with pytest.raises(ValueError, match="no child branches"):
        ld.coupling_ratios(terminal)


def test_excitatory_levels_include_their_conductance():
    torch.manual_seed(2)
    net = DendriNet(**BASE, epsilon=0.0).double()
    proximal = net.branch_layers[1]
    excitation, inhibition = _inputs(6)
    with pytest.raises(ValueError, match="pass excitatory_input"):
        ld.relative_resistance(proximal, inhibition)
    with pytest.raises(ValueError, match="pass excitatory_input"):
        ld.coupling_ratios(proximal)
    base = (
        1
        + proximal.branch_excitation(excitation)
        + proximal.branches_to_output.sum_conductances()
    )
    total = base + proximal.branch_inhibition(inhibition)
    assert torch.equal(
        ld.relative_resistance(proximal, inhibition, excitation), base / total
    )
    ratio = ld.coupling_ratios(proximal, excitation) * ld.relative_resistance(
        proximal, inhibition, excitation
    ).unsqueeze(-1)
    assert torch.allclose(
        ratio,
        proximal.branches_to_output.weight() / total.unsqueeze(-1),
        rtol=1e-14,
        atol=0,
    )


def test_activation_slope_needs_a_stored_supported_voltage():
    net, _ = _typed(1)
    net = net.double()
    proximal = net.branch_layers[1]
    with pytest.raises(ValueError, match="enable branch diagnostics"):
        ld.activation_slope(proximal)
    proximal.set_branch_diagnostics(True)
    proximal.reactivation = nn.Sigmoid()
    net(*_inputs())
    with pytest.raises(NotImplementedError, match="identity and tanh"):
        ld.activation_slope(proximal)


def test_delivered_objective_differentiates_to_the_delivered_error():
    outputs = [
        torch.randn(2, SOMA * LEAVES, dtype=torch.float64, requires_grad=True),
        torch.randn(2, SOMA * STREAMS, dtype=torch.float64, requires_grad=True),
    ]
    error = torch.randn(2, SOMA, dtype=torch.float64)
    gain = torch.rand(2, SOMA * LEAVES, dtype=torch.float64)
    total = ld.delivered_error_objective(outputs, error, [gain, None])
    first, second = torch.autograd.grad(total, outputs)
    assert torch.equal(first, error.repeat_interleave(LEAVES, -1) * gain)
    assert torch.equal(second, error.repeat_interleave(STREAMS, -1))
    with pytest.raises(ValueError, match="one gain"):
        ld.delivered_error_objective(outputs, error, [None])
    with pytest.raises(ValueError, match="does not divide"):
        ld.delivered_error_objective([torch.zeros(2, DIM + 1)], error, [None])


def test_soma_error_is_the_detached_loss_gradient():
    soma = torch.randn(3, SOMA, dtype=torch.float64, requires_grad=True)
    loss = (soma.square() * torch.arange(1.0, SOMA + 1, dtype=torch.float64)).sum()
    error = ld.soma_error(loss, soma)
    assert torch.equal(
        error, 2 * soma.detach() * torch.arange(1.0, SOMA + 1, dtype=torch.float64)
    )
    assert not error.requires_grad
    loss.backward()  # the graph is retained for the caller
