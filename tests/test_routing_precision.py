"""Routing rejects unrepresentable temperatures and scaled-logit overflow."""

import pytest
import torch
from torch.nn import functional as F

from dendritic_modeling.config import RoutedEIConfig
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.routed_ei import (
    build_routed_ei,
)
from dendritic_modeling.networks.utils.routing import routing_assignments


@pytest.mark.parametrize(
    "hard, training", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize(
    "dtype, temperature",
    [
        (torch.float16, 1e-8),
        (torch.float16, 1e5),
        (torch.bfloat16, 1e-42),
        (torch.bfloat16, 1e40),
        (torch.float32, 1e-50),
        (torch.float32, 1e40),
    ],
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
)
def test_temperature_must_fit_logits_dtype(dtype, temperature, hard, training, device):
    logits = torch.tensor([[0.125, 0.0625]], dtype=dtype, device=device)
    with pytest.raises(ValueError, match="temperature"):
        routing_assignments(logits, temperature, hard=hard, training=training)
    assert torch.isfinite(
        routing_assignments(logits.double(), temperature, hard=hard, training=training)
    ).all()


@pytest.mark.parametrize(
    "hard, training", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
)
def test_scaled_logits_must_be_finite_before_softmax(hard, training, sign, device):
    logits = sign * torch.tensor([[0.125, 0.0625]], dtype=torch.float16, device=device)
    assert torch.tensor(1e-6, dtype=logits.dtype).item() > 0
    with pytest.raises(ValueError, match="scaled routing logits"):
        routing_assignments(logits, 1e-6, hard=hard, training=training)


@pytest.mark.parametrize(
    "hard, training", [(False, False), (True, False), (True, True)]
)
@pytest.mark.parametrize("dim", [0, -1])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.requires_gpu)]
)
def test_valid_routing_keeps_literal_values_gradients_and_rng(
    dtype, dim, hard, training, device
):
    logits = torch.tensor(
        [[0.125, 0.0625], [-0.25, 0.375]],
        dtype=dtype,
        device=device,
        requires_grad=True,
    )
    upstream = torch.tensor([[0.5, -0.25], [0.125, 0.25]], dtype=dtype, device=device)
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == "cuda" else None
    actual = routing_assignments(logits, 0.8, dim=dim, hard=hard, training=training)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng is not None:
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    soft = torch.softmax(logits / 0.8, dim=dim)
    expected = soft
    if hard:
        discrete = F.one_hot(soft.argmax(dim=dim), num_classes=logits.shape[dim])
        discrete = discrete.movedim(-1, dim).to(dtype)
        expected = discrete + soft - soft.detach() if training else discrete
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if actual.requires_grad:
        a = torch.autograd.grad((actual * upstream).sum(), logits)[0]
        b = torch.autograd.grad((expected * upstream).sum(), logits)[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert torch.isfinite(a).all()


@pytest.mark.parametrize(
    "dtype, temperature, message",
    [
        (torch.float32, 1e-50, "temperature"),
        (torch.float32, 1e40, "temperature"),
        (torch.float16, 1e-6, "scaled routing logits"),
    ],
)
def test_routed_entry_points_share_temperature_checks(dtype, temperature, message):
    torch.manual_seed(27)
    model = build_routed_ei(
        RoutedEIConfig(features=4, branches=2, temperature=temperature)
    ).to(dtype)
    with torch.no_grad():
        model.route_logits[0].fill_(0.125)
        model.route_logits[1].fill_(0.0625)
    x = torch.ones(2, 8, dtype=dtype)
    calls = [
        model.assignment,
        model.routing_statistics,
        model.inferred_assignment,
        lambda: model.routing_regularizer(entropy_weight=0.1, balance_weight=0.1),
        lambda: model(x),
        lambda: model.drives(x),
        lambda: model.branch_values(x),
        lambda: model.branch_response(x),
    ]
    for call in calls:
        with pytest.raises(ValueError, match=message):
            call()


def test_pathway_router_rejects_scaled_logit_overflow():
    torch.manual_seed(27)
    model = PathwayRouter(
        4, router_mode="learned", n_pathways=2, learned_router_temperature=1e-6
    ).half()
    with torch.no_grad():
        model.assignment_logits[:, 0].fill_(0.125)
        model.assignment_logits[:, 1].fill_(0.0625)
    for call in (
        model._learned_assignments,
        model.routing_statistics,
        lambda: model(torch.ones(2, 4).half()),
    ):
        with pytest.raises(ValueError, match="scaled routing logits"):
            call()
