"""Tests for local-learning activation derivative helpers."""

import contextlib

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.networks.activations.factory import (
    ActivationFactory,
    get_registered_activation_types,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_activation_derivatives import (
    compute_reactivation_derivative,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_reactivation_mixin import (
    reactivation_parameter_grads,
)


def test_compute_reactivation_derivative_preserves_builtin_activation_formulas():
    v_n = torch.tensor([[-1.0, 0.0, 2.0]], dtype=torch.float64)

    torch.testing.assert_close(
        compute_reactivation_derivative(None, v_n),
        torch.ones_like(v_n),
    )
    torch.testing.assert_close(
        compute_reactivation_derivative(nn.Identity(), v_n),
        torch.ones_like(v_n),
    )
    torch.testing.assert_close(
        compute_reactivation_derivative(nn.ReLU(), v_n),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
    )

    sigmoid_out = torch.sigmoid(v_n).float()
    torch.testing.assert_close(
        compute_reactivation_derivative(nn.Sigmoid(), v_n, sigmoid_out),
        sigmoid_out.to(dtype=v_n.dtype) * (1.0 - sigmoid_out.to(dtype=v_n.dtype)),
    )

    tanh_out = torch.tanh(v_n).float()
    torch.testing.assert_close(
        compute_reactivation_derivative(nn.Tanh(), v_n, tanh_out),
        1.0 - tanh_out.to(dtype=v_n.dtype).square(),
    )


def test_compute_reactivation_derivative_preserves_generic_autograd_branch():
    class Cubic(nn.Module):
        def forward(self, x):
            return x.pow(3)

    v_n = torch.tensor([[-2.0, 0.5, 1.5]], dtype=torch.float64)

    derivative = compute_reactivation_derivative(Cubic(), v_n)

    torch.testing.assert_close(derivative, 3.0 * v_n.square())
    assert derivative.requires_grad is False


def _gate(act_type):
    torch.manual_seed(0)
    gate = ActivationFactory.create(
        act_type, output_dim=3, init_m=1.7, init_b=0.4, fixed_b=0.3
    ).double()
    for param in gate.parameters():  # move off the initial values
        param.data.add_(0.1 * torch.randn_like(param))
    return gate


def _autograd_slope(gate, v_n):
    voltage = v_n.clone().requires_grad_(True)
    return torch.autograd.grad(gate(voltage).sum(), voltage)[0]


@pytest.mark.parametrize("act_type", get_registered_activation_types())
@pytest.mark.parametrize("context", ["grad", "no_grad", "inference_mode"])
def test_reactivation_derivative_matches_autograd_for_every_gate(act_type, context):
    gate = _gate(act_type)
    v_n = torch.linspace(-1.0, 2.0, 12, dtype=torch.float64).view(4, 3)
    expected = _autograd_slope(gate, v_n)
    scope = {
        "grad": contextlib.nullcontext,
        "no_grad": torch.no_grad,
        "inference_mode": torch.inference_mode,
    }[context]
    with scope():
        derivative = compute_reactivation_derivative(gate, v_n, gate(v_n))
    torch.testing.assert_close(derivative, expected)


def test_reactivation_derivative_raises_when_the_gate_cannot_be_differentiated():
    class Broken(nn.Module):
        def forward(self, x):
            raise RuntimeError("cannot differentiate")

    with pytest.raises(RuntimeError, match="cannot differentiate"):
        compute_reactivation_derivative(Broken(), torch.ones(1, 2))


@pytest.mark.parametrize("act_type", get_registered_activation_types())
def test_local_gate_gradients_are_the_exact_parameter_gradients(act_type):
    gate = _gate(act_type)
    params = [param for param in gate.parameters() if param.requires_grad]
    v_n = torch.linspace(-1.0, 2.0, 12, dtype=torch.float64).view(4, 3)
    error = torch.linspace(-0.5, 0.7, 12, dtype=torch.float64).view(4, 3)

    grads = {
        id(param): grad
        for param, grad in reactivation_parameter_grads(gate, v_n, error)
    }

    assert set(grads) == {id(param) for param in params}
    if params:
        expected = torch.autograd.grad((gate(v_n) * error).sum() / 4, params)
        for param, grad in zip(params, expected):
            torch.testing.assert_close(grads[id(param)], grad)
