"""Additive compartment perturbations of branch layers (``perturb_compartments``)."""

from __future__ import annotations

import pytest
import torch

from dendritic_modeling.networks import (
    CompartmentPerturbation,
    capture_analysis_currents,
    perturb_compartments,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)

SOMA, STREAMS = 3, 2
LEAVES = 2 * STREAMS
DIM = 2 * LEAVES + STREAMS
BASE = {
    "n_soma": SOMA,
    "branch_factors": [STREAMS, 2],
    "excitatory_input_dim": DIM,
    "excitatory_synapses_per_branch": DIM,
    "inhibitory_input_dim": DIM,
    "inhibitory_synapses_per_branch": DIM,
    "reactivate": False,
    "somatic_synapses": False,
    "weight_transform": "exp",
    "use_noise": False,
    "topk_noise_level": 0.0,
    "topk_strategy": "none",
    "blocklinear_strategy": "none",
}
# A tanh proximal level, so voltage and output coordinates differ.
OVERRIDES = [{"depth": 1, "activation": "tanh"}]


def _net(shunting=True, seed=3):
    torch.manual_seed(seed)
    return (
        DendriNet(**BASE, use_shunting=shunting, level_overrides=OVERRIDES)
        .double()
        .eval()
    )


def _inputs(batch=5, seed=0):
    generator = torch.Generator().manual_seed(seed)
    excitation = 2 * torch.rand(batch, DIM, generator=generator, dtype=torch.float64)
    inhibition = 2 * torch.rand(batch, DIM, generator=generator, dtype=torch.float64)
    return excitation, inhibition


def _width(layer):
    return int(layer.branch_config.output_dim)


def _coordinates(shunting):
    return ("output", "voltage", "current") if shunting else ("output", "voltage")


@pytest.mark.parametrize("shunting", [True, False])
def test_a_zero_offset_leaves_every_level_unchanged_and_runs_it_once(shunting):
    net = _net(shunting)
    x, inhibition = _inputs()
    with torch.no_grad():
        reference = net.forward_levels(x, inhibition)
    for layer in net.branch_layers:
        calls = []
        handle = layer.register_forward_hook(lambda *_, calls=calls: calls.append(1))
        for coordinate in _coordinates(shunting):
            calls.clear()
            zero = torch.zeros(len(x), _width(layer), dtype=x.dtype)
            with (
                torch.no_grad(),
                perturb_compartments(
                    {layer: CompartmentPerturbation(zero, coordinate)}
                ),
            ):
                levels = net.forward_levels(x, inhibition)
            assert all(torch.equal(a, b) for a, b in zip(levels, reference)), coordinate
            assert len(calls) == 1
        handle.remove()


def test_an_output_offset_adds_to_the_level_output_once():
    net = _net()
    x, inhibition = _inputs()
    proximal = net.branch_layers[1]
    offset = torch.randn(
        len(x),
        _width(proximal),
        dtype=x.dtype,
        generator=torch.Generator().manual_seed(1),
    )
    with torch.no_grad():
        reference = net.forward_levels(x, inhibition)
        with perturb_compartments(
            {proximal: CompartmentPerturbation(offset, "output")}
        ):
            perturbed = net.forward_levels(x, inhibition)
    assert torch.equal(perturbed[0], reference[0])
    assert torch.allclose(perturbed[1], reference[1] + offset, rtol=0, atol=1e-15)
    assert not torch.allclose(perturbed[2], reference[2])


def test_a_voltage_offset_passes_through_the_activation():
    net = _net()
    x, inhibition = _inputs()
    proximal = net.branch_layers[1]
    offset = torch.full((len(x), _width(proximal)), 0.3, dtype=x.dtype)
    with torch.no_grad(), capture_analysis_currents(proximal):
        net.forward_levels(x, inhibition)
        voltage = proximal.get_analysis_currents()["pre_gate_voltage"]
    with (
        torch.no_grad(),
        perturb_compartments({proximal: CompartmentPerturbation(offset, "voltage")}),
    ):
        perturbed = net.forward_levels(x, inhibition)[1]
    assert torch.allclose(perturbed, torch.tanh(voltage + offset), rtol=0, atol=1e-15)


def test_an_injected_current_is_divided_by_the_total_conductance():
    net = _net()
    x, inhibition = _inputs()
    terminal = net.branch_layers[0]  # identity activation: the output is the voltage
    terminal.set_branch_diagnostics(True)
    with torch.no_grad():
        reference = net.forward_levels(x, inhibition)[0]
        diagnostics = terminal.get_branch_diagnostics()
        current = torch.full((len(x), _width(terminal)), 0.5, dtype=x.dtype)
        with perturb_compartments(
            {terminal: CompartmentPerturbation(current, "current")}
        ):
            perturbed = net.forward_levels(x, inhibition)[0]
    terminal.set_branch_diagnostics(False)
    conductance = diagnostics["T"] + diagnostics["leak"] + terminal.epsilon
    assert torch.allclose(
        perturbed - reference, current / conductance, rtol=1e-12, atol=0
    )


@pytest.mark.parametrize("coordinate", ["output", "voltage", "current"])
def test_gradients_with_respect_to_a_zero_offset_are_the_exact_errors(coordinate):
    net = _net()
    x, inhibition = _inputs()
    proximal = net.branch_layers[1]
    zero = torch.zeros(len(x), _width(proximal), dtype=x.dtype, requires_grad=True)
    with perturb_compartments({proximal: CompartmentPerturbation(zero, coordinate)}):
        loss = net(x, inhibition).square().sum()
    found = torch.autograd.grad(loss, zero)[0]
    with capture_analysis_currents(proximal):
        levels = net.forward_levels(x, inhibition)
        voltage = proximal.get_analysis_currents()["pre_gate_voltage"]
        output_error, voltage_error = torch.autograd.grad(
            levels[-1].square().sum(), [levels[1], voltage]
        )
    if coordinate == "output":
        expected = output_error
    elif coordinate == "voltage":
        expected = voltage_error
    else:
        proximal.set_branch_diagnostics(True)
        with torch.no_grad():
            net.forward_levels(x, inhibition)
        diagnostics = proximal.get_branch_diagnostics()
        proximal.set_branch_diagnostics(False)
        expected = voltage_error / (
            diagnostics["T"] + diagnostics["leak"] + proximal.epsilon
        )
    assert torch.allclose(found, expected, rtol=1e-12, atol=1e-15)


def test_offsets_must_fit_the_level():
    net = _net()
    x, inhibition = _inputs()
    proximal = net.branch_layers[1]
    for offset, message in (
        (
            torch.zeros(len(x), _width(proximal) + 1, dtype=x.dtype),
            "does not broadcast",
        ),
        (torch.zeros(len(x), _width(proximal), dtype=torch.float32), "dtype"),
    ):
        with (
            pytest.raises(ValueError, match=message),
            perturb_compartments({proximal: CompartmentPerturbation(offset, "output")}),
        ):
            net(x, inhibition)
    # Broadcasting over the batch is allowed: one offset per compartment.
    with (
        torch.no_grad(),
        perturb_compartments(
            {
                proximal: CompartmentPerturbation(
                    torch.zeros(_width(proximal), dtype=x.dtype), "output"
                )
            }
        ),
    ):
        net(x, inhibition)


@pytest.mark.requires_gpu
def test_an_offset_on_another_device_is_refused():
    net = _net().cuda()
    x, inhibition = (value.cuda() for value in _inputs())
    proximal = net.branch_layers[1]
    with (
        pytest.raises(ValueError, match="layer computes on"),
        perturb_compartments(
            {
                proximal: CompartmentPerturbation(
                    torch.zeros(len(x), _width(proximal), dtype=x.dtype), "output"
                )
            }
        ),
    ):
        net(x, inhibition)


def test_unknown_coordinates_and_currents_of_additive_layers_are_refused():
    with pytest.raises(ValueError, match="unknown perturbation coordinate"):
        CompartmentPerturbation(torch.zeros(1), "conductance")
    with pytest.raises(TypeError, match="must be a tensor"):
        CompartmentPerturbation([0.0])
    layer = _net(shunting=False).branch_layers[0]
    with pytest.raises(ValueError, match="needs a shunting layer"):
        layer.set_compartment_perturbation(
            CompartmentPerturbation(torch.zeros(1, dtype=torch.float64), "current")
        )


def test_the_block_restores_the_previous_perturbation_after_an_error():
    net = _net()
    terminal, proximal, _ = net.branch_layers
    previous = CompartmentPerturbation(
        torch.zeros(_width(terminal), dtype=torch.float64)
    )
    terminal.set_compartment_perturbation(previous)
    with pytest.raises(RuntimeError, match="inside"):
        with perturb_compartments(
            {
                terminal: CompartmentPerturbation(
                    torch.ones(_width(terminal), dtype=torch.float64)
                ),
                proximal: CompartmentPerturbation(
                    torch.ones(_width(proximal), dtype=torch.float64)
                ),
            }
        ):
            raise RuntimeError("inside")
    assert terminal.compartment_perturbation is previous
    assert proximal.compartment_perturbation is None
    with pytest.raises(TypeError, match="DendriticBranchLayer"):
        with perturb_compartments({net: previous}):
            pass


def test_perturbations_change_no_state_and_are_not_saved():
    net = _net()
    state = {key: value.clone() for key, value in net.state_dict().items()}
    x, inhibition = _inputs()
    proximal = net.branch_layers[1]
    with (
        torch.no_grad(),
        perturb_compartments(
            {
                proximal: CompartmentPerturbation(
                    torch.ones(len(x), _width(proximal), dtype=x.dtype)
                )
            }
        ),
    ):
        net(x, inhibition)
        assert list(net.state_dict()) == list(state)
    assert all(
        torch.equal(net.state_dict()[key], value) for key, value in state.items()
    )


def test_the_stateful_current_path_refuses_a_perturbation():
    layer = _net().branch_layers[0]
    x, inhibition = _inputs()
    layer.set_compartment_perturbation(
        CompartmentPerturbation(torch.zeros(1, dtype=torch.float64))
    )
    with pytest.raises(ValueError, match="stateful current path"):
        layer.compute_raw_currents(x, inhibition)
