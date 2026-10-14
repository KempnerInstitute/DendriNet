"""Compilation options act on their own model, never on the whole process."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.networks.architectures.factory import get_architecture

SRC = Path(__file__).resolve().parents[1] / "src"


def test_importing_the_analysis_metrics_leaves_torch_compile_enabled():
    code = (
        "import os, sys; sys.path.insert(0, sys.argv[1]); "
        "import dendritic_modeling.analysis.metrics; "
        "assert 'TORCHDYNAMO_DISABLE' not in os.environ"
    )
    subprocess.run([sys.executable, "-c", code, str(SRC)], check=True)


def _einet(compile_forward: bool):
    core = config_from_mapping(
        {
            "model": {
                "core": {
                    "type": "einet",
                    "architecture": {
                        "excitatory_layer_sizes": [4],
                        "excitatory_branch_factors": [2],
                    },
                    "connectivity": {"ee_synapses_per_branch_per_layer": [3]},
                    "implementation": {"compile_forward": compile_forward},
                }
            }
        }
    ).model.core
    return get_architecture(core.type, core, input_dim=6)


def test_compile_forward_compiles_this_module_only():
    limit = torch._dynamo.config.recompile_limit
    static = torch._dynamo.config.force_parameter_static_shapes
    network = _einet(compile_forward=True)

    assert network._compiled_call_impl is not None
    assert "forward" not in vars(network)  # no compiled bound method on the instance
    assert torch._dynamo.config.recompile_limit == limit
    assert torch._dynamo.config.force_parameter_static_shapes == static
    duplicate = copy.deepcopy(network)
    assert duplicate._compiled_call_impl is None  # runs its own forward
    assert _einet(compile_forward=False)._compiled_call_impl is None


def _scaled_einet(rule: str, compile_forward: bool, **sections):
    from test_operator_config_contract import _config

    config = copy.deepcopy(_config("einet", rule, "param_tanh"))
    for section, values in sections.items():
        config.setdefault(section, {}).update(values)
    config["implementation"]["compile_forward"] = compile_forward
    torch.manual_seed(3)
    return get_architecture("einet", config, input_dim=6)


def test_compile_forward_rejects_conductance_scaled_couplings_of_shunting_layers():
    # Inductor fails on these gradient hooks at the first training step.
    scaled = {"blocklinear": {"gradient_scaling": "conductance_dynamic"}}
    with pytest.raises(ValueError, match="compile_forward does not support"):
        _scaled_einet("shunting", True, **scaled)
    _scaled_einet("shunting", False, **scaled)  # uncompiled: unchanged


def _training_gradients(network, steps=2):
    x = torch.rand(4, 6, generator=torch.Generator().manual_seed(2))
    network.train()
    for _ in range(steps):
        network.zero_grad(set_to_none=True)
        network(x).square().sum().backward()
    return {
        n: p.grad.clone() for n, p in network.named_parameters() if p.grad is not None
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "rule, sections",
    [
        ("shunting", {"sparsity": {"gradient_scaling": "conductance_dynamic"}}),
        ("additive", {"blocklinear": {"gradient_scaling": "conductance_dynamic"}}),
    ],
)
def test_compile_forward_keeps_the_conductance_scaling_it_compiles(rule, sections):
    eager = _scaled_einet(rule, False, **sections)
    generator = torch.Generator().manual_seed(23)
    with torch.no_grad():
        for name, parameter in eager.named_parameters():
            if name.endswith("branches_to_output.log_weight"):
                parameter.copy_(0.5 * torch.randn(parameter.shape, generator=generator))
    torch._dynamo.reset()
    compiled = _scaled_einet(rule, True, **sections)
    compiled.load_state_dict(eager.state_dict())
    assert compiled._compiled_call_impl is not None
    expected = _training_gradients(eager)
    observed = _training_gradients(compiled)
    assert observed.keys() == expected.keys()
    for name, gradient in expected.items():
        torch.testing.assert_close(observed[name], gradient, rtol=1e-4, atol=1e-6)
