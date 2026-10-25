"""Public current operators retain native arithmetic without layer stand-ins."""

import pytest
import torch

from dendritic_modeling.networks import (
    DendriticBranchLayer,
    additive_voltage,
    shunting_voltage,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shunting", [False, True])
@pytest.mark.parametrize("children", [False, True])
@pytest.mark.parametrize("recurrent", [False, True])
def test_current_api_matches_real_layer_exactly(dtype, shunting, children, recurrent):
    torch.manual_seed(273)
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=4,
        excitatory_synapses_per_branch=2,
        use_shunting=shunting,
        leak_conductance=0.25 if shunting else 1.0,
    ).to(dtype=dtype)
    e, i = [torch.rand(5, 3, dtype=dtype, requires_grad=True) for _ in range(2)]
    er = torch.rand_like(e, requires_grad=True) if recurrent else 0
    ir = torch.rand_like(i, requires_grad=True) if recurrent else 0
    c = torch.randn_like(e, requires_grad=True) if children else 0
    g = torch.rand(3, dtype=dtype, requires_grad=True) if children else None
    options = {
        "recurrent_excitation": er,
        "recurrent_inhibition": ir,
        "child_current": c,
    }
    if shunting:
        actual = shunting_voltage(
            e,
            i,
            child_conductance=g,
            leak=layer.effective_leak,
            epsilon=layer.epsilon,
            **options,
        )
    else:
        actual = additive_voltage(e, i, **options)
    expected, _ = layer.voltage_from_currents(e, er, i, ir, c, g)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    variables = [
        x
        for x in (e, i, er, ir, c, g if shunting else None)
        if isinstance(x, torch.Tensor)
    ]
    got = torch.autograd.grad(actual.square().sum(), variables, retain_graph=True)
    want = torch.autograd.grad(expected.square().sum(), variables)
    for a, b in zip(got, want, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_shunting_gradcheck_includes_shared_sensor_leak_and_coupling():
    generator = torch.Generator().manual_seed(27)
    values = [
        torch.rand(shape, generator=generator, dtype=torch.float64)
        .add(0.2)
        .requires_grad_()
        for shape in ((2, 3), (2, 1), (3,), (2, 3), (3,))
    ]

    def voltage(e, i, leak, c, g):
        return shunting_voltage(e, i, leak=leak, child_current=c, child_conductance=g)

    assert torch.autograd.gradcheck(voltage, tuple(values))
    assert torch.autograd.gradgradcheck(voltage, tuple(values))


def test_terminal_and_soma_equations_and_raw_additive_units():
    e = torch.tensor([[0.3, 0.7], [1.2, 0.1]], dtype=torch.float64)
    i = torch.tensor([[0.1, 0.9], [0.8, 0.4]], dtype=e.dtype)
    v = shunting_voltage(e, i, leak=0.25, epsilon=0)
    torch.testing.assert_close(v, e / (0.25 + e + i), rtol=0, atol=0)
    soma = shunting_voltage(
        torch.zeros(2, dtype=e.dtype),
        child_current=2 * v.sum(-1),
        child_conductance=4.0,
        leak=0.5,
        epsilon=0,
    )
    torch.testing.assert_close(soma, 2 * v.sum(-1) / 4.5, rtol=0, atol=0)
    torch.testing.assert_close(additive_voltage(e, i), e - i, rtol=0, atol=0)
    assert bool((additive_voltage(e, i) < 0).any())
    # Explicit dimensionless inputs retain the historical rate-model convention.
    torch.testing.assert_close(
        additive_voltage(e / 0.25, i / 0.25), (e - i) / 0.25, rtol=0, atol=0
    )


def test_api_has_no_parameters_state_or_random_draws():
    e = torch.ones(3, requires_grad=True)
    state = torch.get_rng_state().clone()
    for operator in (shunting_voltage, additive_voltage):
        result = operator(e, 0.5)
        assert result.requires_grad
        assert result.dtype == e.dtype and result.device == e.device
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"leak": 0.0},
        {"leak": -1.0},
        {"leak": float("nan")},
        {"inhibition": -1.0},
        {"inhibition": float("inf")},
        {"epsilon": -1.0},
        {"child_current": 1.0},
        {"child_conductance": -1.0},
        {"inhibition": torch.ones(4)},
        {"inhibition": torch.ones(2, 1, 3)},
        {"inhibition": torch.ones(3, dtype=torch.float64)},
    ],
)
def test_invalid_shunting_inputs_fail_explicitly(kwargs):
    with pytest.raises((ValueError, TypeError)):
        shunting_voltage(torch.ones(2, 3), **kwargs)


@pytest.mark.parametrize("operator", [shunting_voltage, additive_voltage])
@pytest.mark.parametrize(
    "excitation",
    [
        torch.ones(2, dtype=torch.int64),
        torch.tensor([-1.0]),
        torch.tensor([float("nan")]),
        torch.empty(0),
        [1.0, 2.0],
    ],
)
def test_invalid_reference_is_rejected(operator, excitation):
    with pytest.raises((ValueError, TypeError)):
        operator(excitation)


@pytest.mark.requires_gpu
def test_public_operators_on_cuda():
    e = torch.tensor([[0.2, 0.5]], device="cuda", requires_grad=True)
    i = torch.tensor([[0.1, 0.4]], device="cuda", requires_grad=True)
    output = shunting_voltage(e, i, leak=0.25, epsilon=0)
    torch.testing.assert_close(output, e / (0.25 + e + i), rtol=0, atol=0)
    gradients = torch.autograd.grad(output.sum(), (e, i))
    assert all(bool(torch.isfinite(value).all()) for value in gradients)
    with pytest.raises(ValueError, match="device"):
        additive_voltage(e, i.cpu())
