"""Graph-connected branch currents for analyses (``set_analysis_currents``)."""

from __future__ import annotations

import copy

import pytest
import torch
from test_activation_lifecycle import (
    _build_einet,
    _count_fused_shunt_calls,
    _einet_config,
    _inputs,
)

from dendritic_modeling.networks import DendriticBranchLayer, capture_analysis_currents

CANONICAL = (
    "pre_gate_excitation_current",
    "pre_gate_inhibition_current",
    "pre_gate_upstream_current",
    "pre_gate_voltage",
    "post_gate_output",
)


def _model(memory_efficient=False):
    return _build_einet(
        _einet_config(reactivation={"memory_efficient": memory_efficient})
    )


def _branch_layers(model):
    return [m for m in model.modules() if isinstance(m, DendriticBranchLayer)]


def test_capture_is_off_by_default():
    model = _model()
    model(_inputs(0))
    for layer in _branch_layers(model):
        assert layer.analysis_currents_enabled is False
        assert layer.get_analysis_currents() == {}


def test_captured_tensors_are_the_forward_values_with_their_graph():
    model = _model()
    layers = _branch_layers(model)
    outputs = {}
    handles = [
        layer.register_forward_hook(
            lambda module, _args, output: outputs.__setitem__(module, output)
        )
        for layer in layers
    ]
    for layer in layers:
        layer.set_branch_diagnostics(True)
    with capture_analysis_currents(layers):
        y = model(_inputs(1))
        captured = {layer: layer.get_analysis_currents() for layer in layers}
        voltages = [captured[layer]["pre_gate_voltage"] for layer in layers]
        gradients = torch.autograd.grad(y.square().sum(), voltages, allow_unused=True)
    for handle in handles:
        handle.remove()

    assert any(gradient is not None for gradient in gradients)
    for layer in layers:
        currents = captured[layer]
        assert set(CANONICAL) <= currents.keys()
        assert torch.equal(currents["post_gate_output"], outputs[layer])
        assert currents["pre_gate_voltage"].grad_fn is not None
        torch.testing.assert_close(
            currents["pre_gate_voltage"].detach(),
            layer.get_branch_diagnostics()["V"],
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize("memory_efficient", [False, True])
def test_capture_leaves_the_forward_output_and_gradients_unchanged(
    monkeypatch, memory_efficient
):
    calls = _count_fused_shunt_calls(monkeypatch)
    model = _model(memory_efficient)
    x = _inputs(2)

    def run():
        model.zero_grad(set_to_none=True)
        y = model(x)
        y.square().mean().backward()
        return y.detach(), {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

    reference, reference_grads = run()
    assert bool(calls) == memory_efficient  # the fused gate runs by default
    fused_calls = len(calls)
    with capture_analysis_currents(_branch_layers(model)):
        captured, captured_grads = run()
    assert len(calls) == fused_calls  # capture forms the voltage explicitly
    assert torch.equal(captured, reference)
    assert captured_grads.keys() == reference_grads.keys()
    for name, gradient in reference_grads.items():
        if memory_efficient:  # the fused gate has its own backward formula
            torch.testing.assert_close(captured_grads[name], gradient)
        else:
            assert torch.equal(captured_grads[name], gradient), name


def test_the_context_restores_each_setting_and_discards_tensors():
    model = _model()
    layers = _branch_layers(model)
    layers[0].set_analysis_currents(True)
    with pytest.raises(RuntimeError, match="inside"):
        with capture_analysis_currents(layers):
            model(_inputs(3))
            assert all(layer.get_analysis_currents() for layer in layers)
            raise RuntimeError("inside the block")
    assert layers[0].analysis_currents_enabled is True
    assert all(not layer.analysis_currents_enabled for layer in layers[1:])
    assert all(layer.get_analysis_currents() == {} for layer in layers)


def test_one_layer_or_an_iterable_and_only_branch_layers():
    model = _model()
    layer = _branch_layers(model)[0]
    with capture_analysis_currents(layer) as captured:
        assert captured == (layer,)
        assert layer.analysis_currents_enabled
    with pytest.raises(TypeError, match="DendriticBranchLayer"):
        with capture_analysis_currents([layer, torch.nn.Linear(2, 2)]):
            pass
    assert not layer.analysis_currents_enabled


def test_disabling_discards_the_captured_tensors():
    model = _model()
    layer = _branch_layers(model)[0]
    layer.set_analysis_currents(True)
    model(_inputs(4))
    assert layer.get_analysis_currents()
    layer.set_analysis_currents(False)
    assert layer.get_analysis_currents() == {}


def test_copies_of_a_capturing_model_hold_detached_currents():
    model = _model()
    layers = _branch_layers(model)
    for layer in layers:
        layer.set_analysis_currents(True)
    model(_inputs(5))
    copied = copy.deepcopy(model)
    for layer in _branch_layers(copied):
        currents = layer.get_analysis_currents()
        assert currents
        assert all(
            value is None or value.grad_fn is None for value in currents.values()
        )
