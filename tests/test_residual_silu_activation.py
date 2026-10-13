from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from dendritic_modeling.networks.activations.factory import (
    ActivationFactory,
    get_registered_activation_types,
)
from dendritic_modeling.networks.activations.parametric import ResidualSiLU


def test_residual_silu_is_exact_identity_at_construction():
    act = ResidualSiLU(7)
    v = torch.randn(5, 7) * 3
    assert torch.equal(act(v), v)


def test_residual_silu_at_mix_one_is_silu():
    act = ResidualSiLU(4, init_mix=1.0)
    v = torch.randn(3, 4)
    assert torch.allclose(act(v), F.silu(v))


def test_residual_silu_mix_receives_gradient_at_zero():
    act = ResidualSiLU(6)
    v = torch.randn(8, 6)
    act(v).pow(2).sum().backward()
    assert act.mix.grad is not None
    assert float(act.mix.grad.abs().sum()) > 0


def test_factory_builds_residual_silu_and_ignores_tanh_kwargs():
    act = ActivationFactory.create(
        "residual_silu", output_dim=3, init_m=1.5, init_b=1.0, memory_efficient=True
    )
    assert isinstance(act, ResidualSiLU)
    assert "residual_silu" in get_registered_activation_types()
    v = torch.randn(2, 3)
    assert torch.equal(act(v), v)


def test_residual_silu_rejects_slope_midpoint_calibration():
    act = ResidualSiLU(3)
    with pytest.raises(ValueError):
        act.initialize(1.5, 0.5)
    act.initialize(mix=0.25)
    assert torch.allclose(act.mix, torch.full((3,), 0.25))


def test_existing_activation_types_are_unchanged():
    for name in ("param_tanh", "silu", "none", "relu", "softplus"):
        assert name in get_registered_activation_types()
    assert isinstance(ActivationFactory.create("silu"), torch.nn.SiLU)
