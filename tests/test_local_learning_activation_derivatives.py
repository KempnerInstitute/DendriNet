"""Tests for local-learning activation derivative helpers."""

import logging

import torch
import torch.nn as nn

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_activation_derivatives import (
    compute_reactivation_derivative,
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


def test_compute_reactivation_derivative_warning_fallback_returns_unit_slope(caplog):
    class Broken(nn.Module):
        def forward(self, x):
            raise RuntimeError("cannot differentiate")

    logger = logging.getLogger("tests.local_learning_activation_derivatives")
    v_n = torch.tensor([[1.0, 2.0]], dtype=torch.float64)

    with caplog.at_level(logging.WARNING, logger=logger.name):
        derivative = compute_reactivation_derivative(Broken(), v_n, logger=logger)

    torch.testing.assert_close(derivative, torch.ones_like(v_n))
    assert (
        "Falling back to unit reactivation derivative for module Broken." in caplog.text
    )
