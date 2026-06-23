"""Tests for activation factory and parametric activation functions."""

import math

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.networks.activations.factory import (
    ActivationFactory,
    get_activation_builder,
    get_registered_activation_types,
    register_activation_builder,
    unregister_activation_builder,
)
from dendritic_modeling.networks.activations.parametric import (
    ParametricLinearSigmoid,
    ParametricLinearTanh,
    ParametricLinearTanhTransition,
    ParametricTanh,
    ParametricTanhOnlyM,
)

# ---------------------------------------------------------------------------
# Factory dispatch tests
# ---------------------------------------------------------------------------


def test_create_none():
    act = ActivationFactory.create("none")
    assert isinstance(act, nn.Identity)


def test_create_relu():
    act = ActivationFactory.create("relu")
    assert isinstance(act, nn.ReLU)


def test_create_sigmoid():
    act = ActivationFactory.create("sigmoid")
    assert isinstance(act, nn.Sigmoid)


def test_create_tanh():
    act = ActivationFactory.create("tanh")
    assert isinstance(act, nn.Tanh)


def test_create_param_tanh():
    act = ActivationFactory.create("param_tanh", output_dim=8)
    assert isinstance(act, ParametricTanh)
    assert act.log_m.shape == (8,)


def test_create_param_tanh_no_dim_raises():
    with pytest.raises(ValueError, match="requires output_dim"):
        ActivationFactory.create("param_tanh")


def test_create_param_tanh_only_m():
    act = ActivationFactory.create("param_tanh_only_m", output_dim=5)
    assert isinstance(act, ParametricTanhOnlyM)


def test_create_param_tanh_only_m_no_dim_raises():
    with pytest.raises(ValueError, match="requires output_dim"):
        ActivationFactory.create("param_tanh_only_m")


def test_create_param_linear_sigmoid():
    act = ActivationFactory.create("param_linear_sigmoid", output_dim=4)
    assert isinstance(act, ParametricLinearSigmoid)
    assert act.log_m.requires_grad is True


def test_create_linear_sigmoid_not_trainable():
    act = ActivationFactory.create("linear_sigmoid", output_dim=4)
    assert isinstance(act, ParametricLinearSigmoid)
    assert act.log_m.requires_grad is False


def test_create_param_linear_tanh():
    act = ActivationFactory.create("param_linear_tanh", output_dim=4)
    assert isinstance(act, ParametricLinearTanh)
    assert act.log_m.requires_grad is True


def test_create_linear_tanh_not_trainable():
    act = ActivationFactory.create("linear_tanh", output_dim=4)
    assert isinstance(act, ParametricLinearTanh)
    assert act.log_m.requires_grad is False


def test_create_param_linear_tanh_transition():
    act = ActivationFactory.create("param_linear_tanh_transition", output_dim=6)
    assert isinstance(act, ParametricLinearTanhTransition)
    assert isinstance(act.threshold, nn.Parameter)
    assert isinstance(act.transition_slope, nn.Parameter)


def test_create_linear_tanh_transition_not_trainable():
    act = ActivationFactory.create("linear_tanh_transition", output_dim=6)
    assert isinstance(act, ParametricLinearTanhTransition)
    assert not isinstance(act.threshold, nn.Parameter)
    assert not isinstance(act.transition_slope, nn.Parameter)


def test_create_unknown_raises():
    with pytest.raises(ValueError, match="Unknown activation type"):
        ActivationFactory.create("leaky_relu")


def test_create_unknown_error_uses_normalized_name():
    with pytest.raises(ValueError, match="Unknown activation type: leaky_relu"):
        ActivationFactory.create("LEAKY_RELU")


def test_create_none_value_raises_clear_error():
    with pytest.raises(ValueError, match="got None"):
        ActivationFactory.create(None)


def test_create_case_insensitive():
    for name in ("ReLU", "RELU", "relu", "Relu"):
        act = ActivationFactory.create(name)
        assert isinstance(act, nn.ReLU)


def test_activation_registry_exposes_builtin_names():
    names = get_registered_activation_types()

    assert "param_tanh" in names
    assert "relu" in names
    assert get_activation_builder("ReLU") is get_activation_builder("relu")


def test_activation_registry_adds_custom_builder():
    def build_elu(**_kwargs):
        return nn.ELU(alpha=0.25)

    register_activation_builder("unit_test_elu", build_elu)
    try:
        act = ActivationFactory.create("UNIT_TEST_ELU")
        assert isinstance(act, nn.ELU)
        assert act.alpha == 0.25
        assert "unit_test_elu" in get_registered_activation_types()

        with pytest.raises(ValueError, match="already registered"):
            register_activation_builder("unit_test_elu", build_elu)
    finally:
        unregister_activation_builder("unit_test_elu")


def test_activation_registry_rejects_non_callable_builder():
    with pytest.raises(TypeError, match="must be callable"):
        register_activation_builder("unit_test_bad", object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# ParametricTanh tests
# ---------------------------------------------------------------------------


def test_parametric_tanh_output_range():
    act = ParametricTanh(output_dim=4, init_m=1.5, init_b=1.0)
    x = torch.randn(10, 4)
    out = act(x)
    assert (out >= 0).all() and (out <= 1).all()


def test_parametric_tanh_parameters_trainable():
    act = ParametricTanh(output_dim=3)
    assert isinstance(act.log_m, nn.Parameter)
    assert act.log_m.requires_grad
    assert isinstance(act.b, nn.Parameter)
    assert act.b.requires_grad


def test_parametric_tanh_initialize():
    act = ParametricTanh(output_dim=4)
    act.initialize(2.0, 0.5)
    torch.testing.assert_close(act.log_m.data, torch.full((4,), math.log(2.0)))
    torch.testing.assert_close(act.b.data, torch.full((4,), 0.5))


def test_parametric_tanh_gradient_flow():
    act = ParametricTanh(output_dim=3)
    x = torch.randn(5, 3)
    out = act(x)
    out.sum().backward()
    assert act.log_m.grad is not None
    assert act.b.grad is not None


# ---------------------------------------------------------------------------
# ParametricTanhOnlyM tests
# ---------------------------------------------------------------------------


def test_parametric_tanh_only_m_output_range():
    act = ParametricTanhOnlyM(output_dim=4)
    x = torch.randn(8, 4)
    out = act(x)
    assert (out >= 0).all() and (out <= 1).all()


def test_parametric_tanh_only_m_fixed_b():
    act = ParametricTanhOnlyM(output_dim=3, fixed_b=0.7)
    assert isinstance(act.fixed_b, float)
    assert act.fixed_b == 0.7
    assert isinstance(act.log_m, nn.Parameter)
    assert act.log_m.requires_grad


def test_parametric_tanh_only_m_initialize():
    act = ParametricTanhOnlyM(output_dim=3, fixed_b=0.7)
    act.initialize(2.5, 0.4)
    torch.testing.assert_close(act.log_m.data, torch.full((3,), math.log(2.5)))
    assert act.fixed_b == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# ParametricLinearSigmoid tests
# ---------------------------------------------------------------------------


def test_linear_sigmoid_init_m_zero():
    act = ParametricLinearSigmoid(output_dim=3, init_m=0)
    # init_m=0 is clamped to 1e-8 internally; log_m should be finite
    assert torch.isfinite(act.log_m).all()


def test_linear_sigmoid_forward_shape():
    act = ParametricLinearSigmoid(output_dim=5)
    x = torch.randn(4, 5)
    out = act(x)
    assert out.shape == (4, 5)


@pytest.mark.parametrize(
    "activation_cls", [ParametricLinearSigmoid, ParametricLinearTanh]
)
def test_linear_blend_activations_remain_finite_for_extreme_negative_inputs(
    activation_cls,
):
    act = activation_cls(output_dim=3)
    x = torch.full((2, 3), -1000.0, requires_grad=True)
    out = act(x)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_linear_sigmoid_initialize():
    act = ParametricLinearSigmoid(output_dim=4)
    act.initialize(3.0, -0.2)
    torch.testing.assert_close(act.log_m.data, torch.full((4,), math.log(3.0)))
    assert act.fixed_b == pytest.approx(-0.2)


# ---------------------------------------------------------------------------
# ParametricLinearTanhTransition tests
# ---------------------------------------------------------------------------


def test_transition_output_dim_validation():
    with pytest.raises(ValueError, match="positive integer"):
        ParametricLinearTanhTransition(output_dim=0)
    with pytest.raises(ValueError, match="positive integer"):
        ParametricLinearTanhTransition(output_dim=-1)


def test_transition_linear_slope_validation():
    with pytest.raises(ValueError, match="linear_slope must be positive"):
        ParametricLinearTanhTransition(output_dim=3, linear_slope=-1.0)


def test_transition_linear_region():
    act = ParametricLinearTanhTransition(
        output_dim=1, init_threshold=0.5, linear_slope=2.0
    )
    x = torch.tensor([[0.1]])  # well below threshold
    out = act(x)
    expected = 2.0 * 0.1
    assert abs(out.item() - expected) < 1e-5


def test_transition_saturation_region():
    act = ParametricLinearTanhTransition(
        output_dim=1,
        init_threshold=0.0,
        linear_slope=1.0,
        init_transition_slope=10.0,
    )
    x = torch.tensor([[10.0]])  # far above threshold
    out = act(x)
    assert out.item() > 0.95  # should approach 1.0


def test_transition_from_list():
    act = ParametricLinearTanhTransition(output_dim=3, init_threshold=[0.1, 0.2, 0.3])
    assert act.threshold.shape == (3,)
    torch.testing.assert_close(act.threshold.data, torch.tensor([0.1, 0.2, 0.3]))


def test_transition_wrong_shape_raises():
    with pytest.raises(ValueError, match="wrong length"):
        ParametricLinearTanhTransition(output_dim=3, init_threshold=[0.1, 0.2])


def test_transition_non_trainable_buffers():
    act = ParametricLinearTanhTransition(
        output_dim=4, trainable_threshold=False, trainable_transition_slope=False
    )
    assert not isinstance(act.threshold, nn.Parameter)
    assert not isinstance(act.transition_slope, nn.Parameter)
    # Buffers should still be tensors
    assert isinstance(act.threshold, torch.Tensor)
    assert isinstance(act.transition_slope, torch.Tensor)


def test_linear_tanh_initialize():
    act = ParametricLinearTanh(output_dim=4)
    act.initialize(1.7, 0.25)
    torch.testing.assert_close(act.log_m.data, torch.full((4,), math.log(1.7)))
    assert act.fixed_b == pytest.approx(0.25)


def test_transition_initialize():
    act = ParametricLinearTanhTransition(output_dim=3, linear_slope=1.0)
    act.initialize(2.2, 0.35)
    torch.testing.assert_close(act.transition_slope.data, torch.full((3,), 2.2))
    torch.testing.assert_close(act.threshold.data, torch.full((3,), 0.35))
