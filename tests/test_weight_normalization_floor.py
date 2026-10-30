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


@pytest.mark.parametrize(
    "operator, floor",
    [(divide_rows_by_sum, None), (divide_rows_by_sum, 1.0), (_normalize, 1.0)],
)
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
)
def test_finite_weights_with_overflowing_sums_are_rejected(
    operator, floor, dtype, device
):
    large = torch.finfo(dtype).max * 0.75
    weight = torch.tensor([[0.0, 0.0], [large, large]], dtype=dtype, device=device)
    assert torch.isfinite(weight).all()
    assert not torch.isfinite(weight.sum(-1)).all()
    with pytest.raises(ValueError, match="row sums must be finite"):
        operator(weight, floor)


@pytest.mark.parametrize("floor", [None, 1.0])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
)
def test_routed_drives_reject_overflowing_weight_sums(floor, device):
    from dendritic_modeling.config import RoutedEIConfig
    from dendritic_modeling.networks.architectures.excitation_inhibition.routed_ei import (
        build_routed_ei,
    )

    torch.manual_seed(29)
    model = build_routed_ei(
        RoutedEIConfig(features=2, branches=1, weight_normalization_floor=floor)
    ).to(device=device, dtype=torch.float16)
    with torch.no_grad():
        model.e_weight_raw.fill_(40000)
    x = torch.ones(2, 4, device=device, dtype=torch.float16)
    for call in (model, model.drives, model.branch_values, model.branch_response):
        with pytest.raises(ValueError, match="row sums must be finite"):
            call(x)
