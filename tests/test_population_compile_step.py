"""Opt-in compiled layer steps of population networks."""

from __future__ import annotations

import copy

import pytest
import torch
from golden_cases import _full_model, _recurrent_population_core

from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationNetwork,
)


def _network(compile_step: bool) -> PopulationNetwork:
    core = _recurrent_population_core()
    core["population_network"]["compile_step"] = compile_step
    torch.manual_seed(3)
    model = _full_model(core)
    (network,) = [m for m in model.modules() if isinstance(m, PopulationNetwork)]
    return model, network


def test_layers_are_compiled_only_on_request():
    _, eager = _network(False)
    assert all(layer._compiled_call_impl is None for layer in eager.layers)

    _, compiled = _network(True)
    assert compiled.compile_step
    assert all(layer._compiled_call_impl is not None for layer in compiled.layers)
    # Copies run their own eager layers.
    duplicate = copy.deepcopy(compiled)
    assert all(layer._compiled_call_impl is None for layer in duplicate.layers)


@pytest.mark.requires_gpu
def test_compiled_steps_match_the_eager_network():
    eager_model, _ = _network(False)
    compiled_model, _ = _network(True)
    compiled_model.load_state_dict(eager_model.state_dict())
    eager_model, compiled_model = eager_model.cuda(), compiled_model.cuda()
    x = torch.rand(3, 5, 8, device="cuda", generator=None)

    outputs, gradients = [], []
    for model in (eager_model, compiled_model):
        y = model(x)
        y = y[0] if isinstance(y, tuple) else y
        y.square().sum().backward()
        outputs.append(y.detach())
        gradients.append({n: p.grad for n, p in model.named_parameters()})

    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-5, atol=1e-6)
    for name, grad in gradients[0].items():
        if grad is not None:
            torch.testing.assert_close(
                gradients[1][name], grad, rtol=1e-4, atol=1e-6, msg=name
            )


def _scaled_population(rule: str, option: str):
    from test_operator_config_contract import _config

    from dendritic_modeling.networks.architectures.factory import get_architecture

    config = copy.deepcopy(_config("population_network", rule, "param_tanh"))
    layer = config["population_network"]["layers"][0]
    layer["population_defaults"][option] = "conductance_dynamic"
    layer["recurrent"] = True
    layer["population_defaults"]["rec_excitatory_synapses"] = 1
    layer["connections"].append({"source": "e", "target": "e", "timing": "delayed"})
    config["population_network"]["compile_step"] = True
    torch.manual_seed(3)
    return get_architecture("population_network", config, input_dim=6)


@pytest.mark.parametrize(
    "option", ["topk_gradient_scaling", "blocklinear_gradient_scaling"]
)
def test_compiled_steps_reject_conductance_scaling_of_shunting_layers(option):
    # The compiled step fails on the first training backward otherwise.
    with pytest.raises(ValueError, match=f"compile_step does not support .*{option}"):
        _scaled_population("shunting", option)


def test_compiled_steps_accept_additive_blocklinear_scaling():
    model = _scaled_population("additive", "blocklinear_gradient_scaling")
    (network,) = [m for m in model.modules() if isinstance(m, PopulationNetwork)]
    assert all(layer._compiled_call_impl is not None for layer in network.layers)


@pytest.mark.requires_gpu
def test_compiled_steps_apply_additive_blocklinear_scaling_like_eager_steps():
    compiled = _scaled_population("additive", "blocklinear_gradient_scaling")
    from test_operator_config_contract import _config

    from dendritic_modeling.networks.architectures.factory import get_architecture

    config = copy.deepcopy(_config("population_network", "additive", "param_tanh"))
    layer = config["population_network"]["layers"][0]
    layer["population_defaults"]["blocklinear_gradient_scaling"] = "conductance_dynamic"
    layer["recurrent"] = True
    layer["population_defaults"]["rec_excitatory_synapses"] = 1
    layer["connections"].append({"source": "e", "target": "e", "timing": "delayed"})
    eager = get_architecture("population_network", config, input_dim=6)
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for name, parameter in eager.named_parameters():
            if name.endswith("branches_to_output.log_weight"):
                parameter.copy_(0.5 * torch.randn(parameter.shape, generator=generator))
    compiled.load_state_dict(eager.state_dict())
    x = torch.rand(3, 5, 6, generator=torch.Generator().manual_seed(1)).cuda()

    outputs, gradients = [], []
    for model in (eager.cuda().train(), compiled.cuda().train()):
        y = model(x)
        y = y[0] if isinstance(y, tuple) else y
        y.square().sum().backward()
        outputs.append(y.detach())
        gradients.append({n: p.grad for n, p in model.named_parameters()})

    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-5, atol=1e-6)
    for name, grad in gradients[0].items():
        if grad is not None:
            torch.testing.assert_close(
                gradients[1][name], grad, rtol=1e-4, atol=1e-6, msg=name
            )
