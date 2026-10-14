"""Independent linear-system and gradient checks for passive stars."""

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.passive_equilibrium import (
    passive_star_soma_voltage,
)


@pytest.mark.parametrize("branches", [1, 2, 4])
def test_passive_star_matches_full_compartment_solve_and_gradients(branches):
    generator = torch.Generator().manual_seed(43)
    e = torch.rand(3, branches, generator=generator, dtype=torch.float64)
    h = 1 + torch.rand(3, branches, generator=generator, dtype=torch.float64)
    g = torch.rand(3, branches, generator=generator, dtype=torch.float64)
    e.requires_grad_()
    h.requires_grad_()
    g.requires_grad_()
    soma_leak = 1.01
    matrix = torch.diag_embed(
        torch.cat((h + g, soma_leak + g.sum(-1, keepdim=True)), -1)
    )
    matrix[:, :-1, -1] = -g
    matrix[:, -1, :-1] = -g
    rhs = torch.cat((e, torch.zeros_like(e[:, :1])), -1)
    expected = torch.linalg.solve(matrix, rhs)[..., -1]
    actual = passive_star_soma_voltage(e, h, g, soma_leak)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_grad = torch.autograd.grad(actual.sum(), (e, h, g), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (e, h, g))
    for got, want in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_zero_and_large_coupling_are_finite():
    e = torch.tensor([[2.0, 3.0]], dtype=torch.float64)
    h = torch.tensor([[4.0, 5.0]], dtype=torch.float64)
    assert torch.equal(
        passive_star_soma_voltage(e, h, torch.zeros_like(e)),
        torch.zeros(1, dtype=e.dtype),
    )
    torch.testing.assert_close(
        passive_star_soma_voltage(e, h, torch.full_like(e, 1e12)),
        e.sum(-1) / (1 + h.sum(-1)),
        rtol=1e-10,
        atol=1e-12,
    )
