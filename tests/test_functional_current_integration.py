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


def _literal_shunting(e, er, i, ir, c, g, leak, epsilon):
    # The layer expression before the shared helper existed (63f0d6f8c),
    # copied rather than imported so a regrouped sum in the helper is caught.
    numerator = e + er
    if torch.is_tensor(c) or c:
        numerator = numerator + c
    denominator = leak + e
    for term in (er, i, ir):
        if torch.is_tensor(term) or term:
            denominator = denominator + term
    if g is not None:
        denominator = denominator + g
    return numerator / (denominator + epsilon)


def _literal_additive(e, er, i, ir, c):
    voltage = e + er - i - ir
    if torch.is_tensor(c) or c:
        voltage = voltage + c
    return voltage


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shunting", [False, True])
@pytest.mark.parametrize("children", [False, True])
@pytest.mark.parametrize("recurrent", [False, True])
def test_layer_and_api_match_the_literal_expression_with_a_learnable_leak(
    dtype, shunting, children, recurrent
):
    torch.manual_seed(274)
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=4,
        excitatory_synapses_per_branch=2,
        use_shunting=shunting,
        leak_conductance=0.25 if shunting else 1.0,
        learnable_leak=shunting,
    ).to(dtype=dtype)
    e, i = [torch.rand(5, 3, dtype=dtype, requires_grad=True) for _ in range(2)]
    er = torch.rand_like(e, requires_grad=True) if recurrent else 0
    ir = torch.rand_like(i, requires_grad=True) if recurrent else 0
    c = torch.randn_like(e, requires_grad=True) if children else 0
    g = torch.rand(3, dtype=dtype, requires_grad=True) if children else None
    if shunting:
        computations = [
            lambda: _literal_shunting(
                e, er, i, ir, c, g, layer.effective_leak, layer.epsilon
            ),
            lambda: layer.voltage_from_currents(e, er, i, ir, c, g)[0],
            lambda: shunting_voltage(
                e,
                i,
                recurrent_excitation=er,
                recurrent_inhibition=ir,
                child_current=c,
                child_conductance=g,
                leak=layer.effective_leak,
                epsilon=layer.epsilon,
            ),
        ]
    else:
        computations = [
            lambda: _literal_additive(e, er, i, ir, c),
            lambda: layer.voltage_from_currents(e, er, i, ir, c, g)[0],
            lambda: additive_voltage(
                e, i, recurrent_excitation=er, recurrent_inhibition=ir, child_current=c
            ),
        ]
    variables = [
        x
        for x in (e, i, er, ir, c, g if shunting else None, layer.log_leak)
        if isinstance(x, torch.Tensor)
    ]
    assert (layer.log_leak is not None) is shunting
    results = []
    for compute in computations:
        voltage = compute()
        results.append(
            (voltage, torch.autograd.grad(voltage.square().sum(), variables))
        )
    expected, expected_gradients = results[0]
    for voltage, gradients in results[1:]:
        torch.testing.assert_close(voltage, expected, rtol=0, atol=0)
        for actual, wanted in zip(gradients, expected_gradients, strict=True):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inhibition": -torch.ones(2, 3)},
        {"inhibition": torch.full((2, 3), float("nan"))},
        {"recurrent_excitation": -torch.ones(2, 3)},
        {"recurrent_inhibition": torch.full((2, 3), float("inf"))},
        {"leak": torch.tensor([1.0, 0.0, 1.0])},
        {"leak": torch.tensor([1.0, -1.0, 1.0])},
        {"child_current": torch.ones(2, 3)},
        {
            "child_current": torch.full((2, 3), float("nan")),
            "child_conductance": torch.ones(3),
        },
        {"child_conductance": -torch.ones(3)},
        {"child_conductance": torch.full((3,), float("nan"))},
    ],
)
def test_invalid_shunting_tensor_operands_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        shunting_voltage(torch.ones(2, 3), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inhibition": -torch.ones(2, 3)},
        {"inhibition": torch.full((2, 3), float("nan"))},
        {"recurrent_excitation": -1.0},
        {"recurrent_inhibition": torch.full((2, 3), float("inf"))},
        {"child_current": torch.full((2, 3), float("nan"))},
        {"inhibition": torch.ones(4)},
        {"inhibition": torch.ones(2, 1, 3)},
        {"inhibition": torch.ones(3, dtype=torch.float64)},
    ],
)
def test_invalid_additive_operands_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        additive_voltage(torch.ones(2, 3), **kwargs)


@pytest.mark.parametrize(
    "operator, name, sign",
    [
        (shunting_voltage, "inhibition", 1),
        (shunting_voltage, "recurrent_excitation", 1),
        (shunting_voltage, "recurrent_inhibition", 1),
        (shunting_voltage, "child_current", 1),
        (shunting_voltage, "child_current", -1),
        (shunting_voltage, "child_conductance", 1),
        (shunting_voltage, "leak", 1),
        (shunting_voltage, "epsilon", 1),
        (additive_voltage, "inhibition", 1),
        (additive_voltage, "recurrent_excitation", 1),
        (additive_voltage, "recurrent_inhibition", 1),
        (additive_voltage, "child_current", 1),
        (additive_voltage, "child_current", -1),
    ],
)
@pytest.mark.parametrize(
    "dtype, value",
    [(torch.float16, 1e5), (torch.bfloat16, 1e40), (torch.float32, 1e40)],
)
def test_scalar_operands_must_be_finite_in_the_excitation_dtype(
    operator, name, sign, dtype, value
):
    e = torch.ones(2, dtype=dtype)
    kwargs = {name: sign * value}
    if name == "child_current" and operator is shunting_voltage:
        kwargs["child_conductance"] = 1.0
    with pytest.raises(ValueError, match=f"{name} must be finite"):
        operator(e, **kwargs)
    # The same scalar is valid in a dtype that can represent it.
    expected = (
        _literal_shunting(
            e.double(),
            kwargs.get("recurrent_excitation", 0),
            kwargs.get("inhibition", 0),
            kwargs.get("recurrent_inhibition", 0),
            kwargs.get("child_current", 0),
            kwargs.get("child_conductance"),
            kwargs.get("leak", 1),
            kwargs.get("epsilon", 1e-8),
        )
        if operator is shunting_voltage
        else _literal_additive(
            e.double(),
            kwargs.get("recurrent_excitation", 0),
            kwargs.get("inhibition", 0),
            kwargs.get("recurrent_inhibition", 0),
            kwargs.get("child_current", 0),
        )
    )
    torch.testing.assert_close(operator(e.double(), **kwargs), expected, rtol=0, atol=0)
    assert torch.isfinite(expected).all()


@pytest.mark.parametrize(
    "dtype, tiny_leak",
    [(torch.float16, 1e-8), (torch.bfloat16, 1e-42), (torch.float32, 1e-50)],
)
def test_scalar_leak_must_stay_positive_in_the_excitation_dtype(dtype, tiny_leak):
    e = torch.ones(2, dtype=dtype)
    with pytest.raises(ValueError, match="rounds to zero"):
        shunting_voltage(e, leak=tiny_leak)
    assert torch.isfinite(shunting_voltage(e, leak=1e-4)).all()
    assert torch.isfinite(shunting_voltage(e.double(), leak=tiny_leak)).all()


def test_signed_child_current_and_zero_current_without_coupling_are_valid():
    e = torch.ones(2, 3)
    torch.testing.assert_close(
        additive_voltage(e, child_current=-torch.ones(2, 3)), torch.zeros(2, 3)
    )
    torch.testing.assert_close(
        shunting_voltage(e, child_current=torch.zeros(2, 3), epsilon=0),
        torch.full((2, 3), 0.5),
    )


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
