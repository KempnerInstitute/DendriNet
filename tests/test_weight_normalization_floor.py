"""Clamped-sum normalization rejects unrepresentable scalar denominators."""

import pytest
import torch

from dendritic_modeling.networks.utils.weight_normalization import (
    divide_rows_by_sum,
    normalize_weight_rows,
)


def _normalize(weight, floor):
    return normalize_weight_rows(weight, 1, 1.0, sum_floor=floor)


@pytest.mark.parametrize("operator", [divide_rows_by_sum, _normalize])
@pytest.mark.parametrize(
    "dtype, floor",
    [
        (torch.float16, 1e-8),
        (torch.float16, 1e5),
        (torch.bfloat16, 1e-42),
        (torch.bfloat16, 1e40),
        (torch.float32, 1e-50),
        (torch.float32, 1e40),
    ],
)
def test_clamped_sum_rejects_floor_outside_weight_dtype(operator, dtype, floor):
    weight = torch.tensor([[0.0, 0.0], [1.0, 2.0]], dtype=dtype)
    with pytest.raises(ValueError, match=r"floor.*finite and positive"):
        operator(weight, floor)
    # These magnitudes are representable in float64 and remain valid there.
    wide = weight.double()
    expected = wide / wide.sum(-1, keepdim=True).clamp_min(floor)
    torch.testing.assert_close(operator(wide, floor), expected, rtol=0, atol=0)
    assert torch.isfinite(expected).all()


@pytest.mark.parametrize("floor", [False, True, 0.0, -1.0, float("nan"), float("inf")])
def test_row_sum_helper_rejects_invalid_scalar_floors(floor):
    with pytest.raises(ValueError, match=r"floor.*finite and positive"):
        divide_rows_by_sum(torch.zeros(2, 3), floor)


@pytest.mark.parametrize("operator", [divide_rows_by_sum, _normalize])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("floor", [1e-3, 2.0])
def test_representable_floor_keeps_literal_values_and_gradients(operator, dtype, floor):
    weight = torch.tensor([[0.0, 0.0], [0.25, 0.75]], dtype=dtype, requires_grad=True)
    upstream = torch.tensor([[0.25, -0.25], [0.5, 0.25]], dtype=dtype)
    actual = operator(weight, floor)
    expected = weight / weight.sum(-1, keepdim=True).clamp_min(floor)
    actual_grad = torch.autograd.grad((actual * upstream).sum(), weight)[0]
    expected_grad = torch.autograd.grad((expected * upstream).sum(), weight)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
    assert torch.isfinite(actual).all() and torch.isfinite(actual_grad).all()
