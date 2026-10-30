"""Routed affine and initialization scalars are checked in their tensor dtype."""

import pytest
import torch

from dendritic_modeling.config import RoutedEIConfig
from dendritic_modeling.networks.architectures.excitation_inhibition.routed_ei import (
    build_routed_ei,
)


@pytest.mark.parametrize("mechanism", ["shunting", "additive"])
@pytest.mark.parametrize(
    "dtype, tiny, large",
    [
        (torch.float16, 1e-8, 1e5),
        (torch.bfloat16, 1e-42, 1e40),
        (torch.float32, 1e-50, 1e40),
    ],
)
@pytest.mark.parametrize(
    "name, magnitude",
    [
        ("branch_scale", "tiny"),
        ("branch_scale", "large"),
        ("branch_divisor", "tiny"),
        ("branch_divisor", "large"),
        ("branch_center", "large"),
        ("branch_center", "negative_large"),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
)
def test_affine_constants_must_fit_voltage_dtype(
    mechanism, dtype, tiny, large, name, magnitude, device
):
    value = {"tiny": tiny, "large": large, "negative_large": -large}[magnitude]
    torch.manual_seed(27)
    model = build_routed_ei(
        RoutedEIConfig(features=4, branches=2, mechanism=mechanism, **{name: value})
    ).to(device=device, dtype=torch.float64)
    x = torch.zeros(2, 8, dtype=torch.float64, device=device)
    assert torch.isfinite(model(x)).all()
    model.to(dtype)
    for call in (model, model.branch_values, model.branch_response):
        with pytest.raises(ValueError, match=name):
            call(x.to(dtype))


@pytest.mark.parametrize("mechanism", ["shunting", "additive"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
)
def test_representable_signed_affine_keeps_values_and_input_gradients(
    mechanism, dtype, device
):
    torch.manual_seed(27)
    model = build_routed_ei(
        RoutedEIConfig(
            features=4,
            branches=2,
            mechanism=mechanism,
            branch_scale=1.5,
            branch_center=-0.125,
            branch_divisor=0.75,
        )
    ).to(dtype=dtype, device=device)
    x = (
        torch.linspace(0.1, 0.8, 16, dtype=dtype, device=device)
        .reshape(2, 8)
        .requires_grad_()
    )
    e, i = model.drives(x)
    voltage = e - i if mechanism == "additive" else e / (1.0 + e + i + 1e-8)
    expected = 1.5 * (voltage - (-0.125)) / 0.75
    actual = model.branch_values(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    a = torch.autograd.grad(actual.sum(), x)[0]
    b = torch.autograd.grad(expected.sum(), x)[0]
    torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize(
    "name",
    ["route_init_std", "weight_init_std", "activation_slope_raw", "activation_offset"],
)
@pytest.mark.parametrize("dtype, value", [(torch.float16, 1e5), (torch.float32, 1e40)])
def test_initial_constants_must_fit_dtype_before_random_draws(name, dtype, value):
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        config = RoutedEIConfig(
            features=4, branches=2, branch_activation="tanh", **{name: value}
        )
        rng = torch.get_rng_state().clone()
        with pytest.raises(ValueError, match=name):
            build_routed_ei(config)
        assert torch.equal(rng, torch.get_rng_state())
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.requires_gpu
def test_invalid_initial_constants_preserve_cuda_rng():
    with torch.device("cuda"):
        for name in (
            "route_init_std",
            "weight_init_std",
            "activation_slope_raw",
            "activation_offset",
        ):
            config = RoutedEIConfig(
                features=4, branches=2, branch_activation="tanh", **{name: 1e40}
            )
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = torch.cuda.get_rng_state().clone()
            with pytest.raises(ValueError, match=name):
                build_routed_ei(config)
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
