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
    "pre_gate_recurrent_excitation_current",
    "pre_gate_recurrent_inhibition_current",
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


def test_capture_leaves_unfused_forward_output_and_gradients_unchanged():
    model = _model()
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
    with capture_analysis_currents(_branch_layers(model)):
        captured, captured_grads = run()
    assert torch.equal(captured, reference)
    assert captured_grads.keys() == reference_grads.keys()
    for name, gradient in reference_grads.items():
        assert torch.equal(captured_grads[name], gradient), name


def test_fused_capture_requires_no_grad_and_preserves_forward(monkeypatch):
    calls = _count_fused_shunt_calls(monkeypatch)
    model = _model(memory_efficient=True)
    layers = _branch_layers(model)
    x = _inputs(2)
    with torch.no_grad():
        # Exercise the saturated regime where the fused backward's clamped
        # logit gives a different voltage from the explicit path.
        for layer in layers:
            layer.reactivation.log_m.fill_(torch.tensor(1.5).log().item())
            layer.reactivation.b.fill_(6.0)
        reference = model(x)
    assert calls
    fused_calls = len(calls)
    with torch.no_grad(), capture_analysis_currents(layers):
        captured = model(x)
        assert all(layer.get_analysis_currents() for layer in layers)
    assert torch.equal(captured, reference)
    assert len(calls) == fused_calls

    with pytest.raises(ValueError, match="bypass the fused shunting backward"):
        with capture_analysis_currents(layers):
            model(x)
    assert all(not layer.analysis_currents_enabled for layer in layers)
    assert all(layer.get_analysis_currents() == {} for layer in layers)


@pytest.mark.parametrize("feedforward", [False, True])
def test_recurrent_currents_are_separate_forward_tensors_with_exact_credit(feedforward):
    torch.manual_seed(5)
    layer = (
        DendriticBranchLayer(
            output_dim=2,
            excitatory_input_dim=3 if feedforward else None,
            excitatory_synapses_per_branch=3 if feedforward else 0,
            recurrent_input_dim=3,
            recurrent_synapses_per_branch=3,
            rec_inhibitory_input_dim=3,
            rec_inhibitory_synapses_per_branch=3,
            topk_type="standard",
            reactivate=False,
            weight_transform="exp",
            use_noise=False,
            adaptive_initialization=False,
        )
        .double()
        .eval()
    )
    x = torch.rand(4, 3, dtype=torch.float64)
    inhibitory_input = torch.rand_like(x)
    with capture_analysis_currents(layer):
        output = layer(x, recurrent_input=x, rec_inhibitory_input=inhibitory_input)
        currents = layer.get_analysis_currents()
        recurrent_e = currents["pre_gate_recurrent_excitation_current"]
        recurrent_i = currents["pre_gate_recurrent_inhibition_current"]
        assert recurrent_e.grad_fn is not None
        assert recurrent_i.grad_fn is not None
        assert currents["excitation"] is currents["pre_gate_excitation_current"]
        assert currents["inhibition"] is currents["pre_gate_inhibition_current"]
        assert currents["branch_input"] is currents["pre_gate_upstream_current"]
        excitation = recurrent_e
        if feedforward:
            excitation = excitation + currents["pre_gate_excitation_current"]
        else:
            assert currents["pre_gate_excitation_current"] is None
        denominator = layer.effective_leak + excitation + recurrent_i + layer.epsilon
        torch.testing.assert_close(output, excitation / denominator)
        grad_e, grad_i = torch.autograd.grad(output.sum(), (recurrent_e, recurrent_i))
        torch.testing.assert_close(
            grad_e, (denominator - excitation) / denominator.square()
        )
        torch.testing.assert_close(grad_i, -excitation / denominator.square())
        layer(x, recurrent_input=x)
        assert (
            layer.get_analysis_currents()["pre_gate_recurrent_inhibition_current"]
            is None
        )


@pytest.mark.parametrize(
    "mode", ["parallel_readout", "all_active_star", "stateful_step"]
)
def test_population_paths_that_bypass_layer_forward_reject_capture(mode):
    from test_parallel_multiscale_control import (
        _small_feedforward_population,
        _small_population,
    )

    model = (
        _small_population()
        if mode == "stateful_step"
        else _small_feedforward_population(mode)
    )
    layers = _branch_layers(model)
    x = torch.ones(2, 6)
    with pytest.raises(
        ValueError, match=r"capture requires DendriticBranchLayer\.forward"
    ):
        with capture_analysis_currents(layers):
            if mode == "stateful_step":
                model.step(x, recurrent_input=torch.ones(2, 4))
            else:
                model(x)
    assert all(layer.get_analysis_currents() == {} for layer in layers)


@pytest.mark.parametrize(
    "method", ["compute_raw_currents", "voltage_from_currents", "output_from_currents"]
)
def test_current_substitution_rejects_capture_and_clears_previous_records(method):
    model = _model()
    layer = _branch_layers(model)[0]
    with capture_analysis_currents(layer):
        model(_inputs(2))
        assert layer.get_analysis_currents()
        with pytest.raises(
            ValueError, match=r"capture requires DendriticBranchLayer\.forward"
        ):
            if method == "compute_raw_currents":
                layer.compute_raw_currents(torch.ones(2, 3))
            elif method == "voltage_from_currents":
                layer.voltage_from_currents(1.0, 0.0, 0.0)
            else:
                layer.output_from_currents(1.0, 0.0)
        assert layer.get_analysis_currents() == {}


def test_intervened_levels_reject_capture_before_any_forward_and_clear_records():
    from test_dendrinet_level_overrides import _inputs as tree_inputs, _typed

    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.level_interventions import (
        CurrentInjection,
    )

    model, _ = _typed(10)
    model = model.double()
    x, inhibition = tree_inputs()
    layers = model.branch_layers[1:]
    with capture_analysis_currents(layers):
        model.forward_levels(x, inhibition)
        assert all(layer.get_analysis_currents() for layer in layers)
        calls = []
        handles = [
            layer.register_forward_hook(lambda *args: calls.append(1))
            for layer in model.branch_layers
        ]
        try:
            interventions = {
                layer.layer_idx: CurrentInjection(inhibition, 0.1) for layer in layers
            }
            with pytest.raises(
                ValueError, match="capture is unsupported on intervened levels"
            ):
                model.forward_levels(x, inhibition, interventions=interventions)
            assert not calls
            assert all(layer.get_analysis_currents() == {} for layer in layers)
        finally:
            for handle in handles:
                handle.remove()


def test_capture_on_other_levels_follows_the_intervened_forward():
    from test_dendrinet_level_overrides import _inputs as tree_inputs, _typed

    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.level_interventions import (
        CurrentInjection,
    )

    model, _ = _typed(10)
    model = model.double()
    x, inhibition = tree_inputs()
    soma = model.branch_layers[-1]
    with capture_analysis_currents(soma):
        outputs = model.forward_levels(
            x, inhibition, interventions={1: CurrentInjection(inhibition * 0.5, 0.1)}
        )
        assert soma.get_analysis_currents()["post_gate_output"] is outputs[-1]
        (gradient,) = torch.autograd.grad(
            outputs[-1].sum(), soma.get_analysis_currents()["pre_gate_voltage"]
        )
        assert torch.isfinite(gradient).all()


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
