"""Tests for weight transformation utilities."""

import pytest
import torch

from dendritic_modeling.networks.utils.weight_transforms import (
    INVERSE_EXP_EPS,
    NONNEGATIVE_TRANSFER_ACTIVATIONS,
    POSITIVE_WEIGHT_TRANSFORMS,
    apply_weight_transform,
    get_weight_transform_fn,
    inverse_weight_transform,
    safe_exp,
)

# ---------------------------------------------------------------------------
# apply_weight_transform
# ---------------------------------------------------------------------------


def test_apply_exp_positive_output():
    w = torch.randn(5, 3)
    out = apply_weight_transform(w, "exp")
    assert (out > 0).all()


def test_apply_exp_clamps_before_overflow():
    w = torch.tensor([100.0], dtype=torch.float32)
    out = apply_weight_transform(w, "exp")
    assert torch.isfinite(out).all()
    assert (out > 0).all()


def test_apply_relu_zeroes_negatives():
    w = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    out = apply_weight_transform(w, "relu")
    assert (out[:2] == 0).all()
    assert (out[3:] == w[3:]).all()


def test_apply_softplus_positive_output():
    w = torch.randn(4, 6)
    out = apply_weight_transform(w, "softplus")
    assert (out > 0).all()
    assert out.shape == w.shape


def test_apply_identity_passthrough():
    w = torch.randn(3, 4)
    out = apply_weight_transform(w, "identity")
    torch.testing.assert_close(out, w)


def test_apply_invalid_type_raises():
    w = torch.randn(2)
    with pytest.raises(ValueError, match="Unsupported weight transform"):
        apply_weight_transform(w, "nonexistent")


def test_invalid_type_error_lists_supported_transforms():
    w = torch.randn(2)

    with pytest.raises(ValueError) as exc_info:
        apply_weight_transform(w, "nonexistent")

    message = str(exc_info.value)
    for transform in ("identity", "exp", "relu", "softplus"):
        assert f"'{transform}'" in message


def test_shunting_guard_constants_match_legacy_import_sites():
    from dendritic_modeling.networks.architectures.excitation_inhibition import (
        ei_network,
    )
    from dendritic_modeling.utils import training_stability_validator

    assert ei_network.POSITIVE_WEIGHT_TRANSFORMS is POSITIVE_WEIGHT_TRANSFORMS
    assert (
        training_stability_validator.POSITIVE_WEIGHT_TRANSFORMS
        is POSITIVE_WEIGHT_TRANSFORMS
    )
    assert (
        ei_network.NONNEGATIVE_TRANSFER_ACTIVATIONS is NONNEGATIVE_TRANSFER_ACTIVATIONS
    )
    assert (
        training_stability_validator.NONNEGATIVE_TRANSFER_ACTIVATIONS
        is NONNEGATIVE_TRANSFER_ACTIVATIONS
    )


# ---------------------------------------------------------------------------
# get_weight_transform_fn
# ---------------------------------------------------------------------------


def test_get_fn_returns_callable():
    w = torch.randn(3, 4)
    for t in ("exp", "relu", "softplus", "identity"):
        fn = get_weight_transform_fn(t)
        assert callable(fn)
        expected = apply_weight_transform(w, t)
        torch.testing.assert_close(fn(w), expected)


def test_get_fn_invalid_type_raises():
    with pytest.raises(ValueError, match="Unsupported weight transform"):
        get_weight_transform_fn("bad")


# ---------------------------------------------------------------------------
# inverse_weight_transform  (roundtrip tests)
# ---------------------------------------------------------------------------


def test_inverse_exp_roundtrip():
    w = torch.randn(5, 3)
    forward = apply_weight_transform(w, "exp")
    recovered = inverse_weight_transform(forward, "exp")
    torch.testing.assert_close(recovered, w, atol=1e-5, rtol=1e-5)


def test_inverse_exp_uses_stable_epsilon_for_zero_input():
    zero = torch.tensor([0.0])

    recovered = inverse_weight_transform(zero, "exp")

    torch.testing.assert_close(recovered, torch.log(zero + INVERSE_EXP_EPS))


def test_inverse_softplus_roundtrip():
    w = torch.randn(4, 4)
    forward = apply_weight_transform(w, "softplus")
    recovered = inverse_weight_transform(forward, "softplus")
    torch.testing.assert_close(recovered, w, atol=1e-4, rtol=1e-4)


def test_inverse_identity_roundtrip():
    w = torch.randn(3)
    recovered = inverse_weight_transform(w, "identity")
    torch.testing.assert_close(recovered, w)


def test_inverse_relu_positive_input():
    w = torch.rand(5) + 0.1  # strictly positive
    recovered = inverse_weight_transform(w, "relu")
    torch.testing.assert_close(recovered, w)


def test_inverse_invalid_type_raises():
    with pytest.raises(ValueError, match="Unsupported weight transform"):
        inverse_weight_transform(torch.ones(2), "bad")


# ---------------------------------------------------------------------------
# Shape and gradient tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("transform", ["exp", "relu", "softplus", "identity"])
def test_preserves_shape(transform):
    for shape in [(10,), (4, 6), (2, 3, 5)]:
        w = torch.randn(shape)
        out = apply_weight_transform(w, transform)
        assert out.shape == shape


@pytest.mark.parametrize("transform", ["exp", "softplus", "relu"])
def test_gradient_flow(transform):
    w = torch.randn(3, 4, requires_grad=True)
    out = apply_weight_transform(w, transform)
    out.sum().backward()
    assert w.grad is not None
    assert w.grad.shape == w.shape


def test_safe_exp_respects_explicit_max_exponent():
    w = torch.tensor([0.0, 10.0], dtype=torch.float32)

    out = safe_exp(w, max_exponent=2.0)

    torch.testing.assert_close(out, torch.exp(torch.tensor([0.0, 2.0])))
