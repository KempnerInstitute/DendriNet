"""Tests for the MuonH optimizer (Muon + hyperball, AdamW fallback).

Covers the Track B contract:
- Newton-Schulz orthogonalization correctness on random matrices (fp64),
- hyperball projection exactness,
- the 2D-vs-rest parameter-group split,
- rank consistency (two simulated ranks, identical averaged grads),
- default-AdamW byte-identity of the existing optimizer path,
- a tiny end-to-end fit whose loss decreases.
"""

import io
import math

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.config.training import OptimizerConfig
from dendritic_modeling.training.optimizers import MuonH, create_optimizer
from dendritic_modeling.training.optimizers.muonh import (
    hyperball_project_,
    newton_schulz_orthogonalize,
)


def _polar_factor(matrix: torch.Tensor) -> torch.Tensor:
    U, _, Vh = torch.linalg.svd(matrix, full_matrices=False)
    return U @ Vh


class TestNewtonSchulz:
    def test_orthogonalizes_random_matrices_fp64(self):
        torch.manual_seed(0)
        for shape in [(64, 32), (32, 64), (48, 48)]:
            G = torch.randn(*shape, dtype=torch.float64)
            orth = newton_schulz_orthogonalize(G, steps=5)
            assert orth.shape == G.shape
            assert orth.dtype == torch.float64
            singular_values = torch.linalg.svdvals(orth)
            # The tuned quintic drives singular values into an approximately
            # unit band rather than exactly 1 (well-conditioned rectangular
            # inputs land in ~[0.68, 1.13]; square inputs can have a few
            # slow-converging small directions).
            assert float(singular_values.max()) < 1.35
            if shape[0] != shape[1]:
                assert float(singular_values.min()) > 0.5
            # Direction: orth approximates the exact polar factor U V^T.
            polar = _polar_factor(G)
            cosine = float((orth * polar).sum() / (orth.norm() * polar.norm()))
            assert cosine > 0.95
            relative_error = float((orth - polar).norm() / polar.norm())
            assert relative_error < 0.3

    def test_approximate_orthogonality_of_gram_matrix(self):
        torch.manual_seed(1)
        G = torch.randn(64, 32, dtype=torch.float64)
        orth = newton_schulz_orthogonalize(G, steps=5)
        gram = orth.mT @ orth
        identity = torch.eye(32, dtype=torch.float64)
        # The tuned quintic's ~[0.68, 1.13] singular-value band bounds the
        # Gram deviation; 0.5 rejects unorthogonalized input by a wide margin.
        assert float((gram - identity).norm() / identity.norm()) < 0.5

    def test_rejects_non_2d(self):
        try:
            newton_schulz_orthogonalize(torch.randn(4))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for 1D input")

    def test_low_precision_input_promoted_to_float32(self):
        G = torch.randn(8, 4, dtype=torch.bfloat16)
        orth = newton_schulz_orthogonalize(G, steps=5)
        assert orth.dtype == torch.float32


class TestHyperballProjection:
    def test_projects_outside_point_to_exact_radius(self):
        torch.manual_seed(2)
        param = torch.randn(16, 8, dtype=torch.float64)
        radius = float(param.norm()) * 0.5
        hyperball_project_(param, radius)
        assert abs(float(param.norm()) - radius) / radius < 1e-12

    def test_inside_point_is_bitwise_untouched(self):
        torch.manual_seed(3)
        param = torch.randn(16, 8, dtype=torch.float64)
        before = param.clone()
        hyperball_project_(param, float(param.norm()) * 2.0)
        assert torch.equal(param, before)

    def test_optimizer_step_projects_back_to_initial_norm(self):
        torch.manual_seed(4)
        param = nn.Parameter(torch.randn(12, 6, dtype=torch.float64))
        initial_norm = float(param.detach().norm())
        optimizer = MuonH([param], lr=1.0, muon_lr_multiplier=1.0)
        for _ in range(3):
            optimizer.zero_grad()
            param.grad = torch.randn_like(param)
            optimizer.step()
            # A unit-scale orthogonalized update at lr=1 pushes far outside
            # the ball, so the projection must bind and restore the radius.
            assert abs(float(param.detach().norm()) - initial_norm) < 1e-9

    def test_hyperball_disabled_allows_norm_growth(self):
        torch.manual_seed(5)
        param = nn.Parameter(torch.zeros(12, 6, dtype=torch.float64))
        optimizer = MuonH([param], lr=1.0, muon_lr_multiplier=1.0, hyperball=False)
        param.grad = torch.randn_like(param)
        optimizer.step()
        assert float(param.detach().norm()) > 1.0


class TestParameterGroupSplit:
    def test_2d_vs_rest(self):
        matrix = nn.Parameter(torch.randn(8, 4))
        wide = nn.Parameter(torch.randn(2, 300))
        bias = nn.Parameter(torch.randn(8))
        scalar_gate = nn.Parameter(torch.randn(()))
        single_contact = nn.Parameter(torch.randn(5, 1))  # gate-like [N, 1]
        optimizer = MuonH([matrix, wide, bias, scalar_gate, single_contact], lr=1e-3)
        assert optimizer.muon_parameter_count() == 2
        assert optimizer.adamw_parameter_count() == 3
        muon_ids = {
            id(p) for g in optimizer.param_groups if g["use_muon"] for p in g["params"]
        }
        assert muon_ids == {id(matrix), id(wide)}

    def test_pure_vector_model_falls_back_to_adamw_only(self):
        bias = nn.Parameter(torch.randn(7))
        optimizer = MuonH([bias], lr=1e-3)
        assert optimizer.muon_parameter_count() == 0
        assert optimizer.adamw_parameter_count() == 1
        bias.grad = torch.randn_like(bias)
        optimizer.step()  # must not raise

    def test_indexed_sparse_pre_w_shape_is_muon_managed(self):
        # IndexedSparseLinear.pre_w is [out_features, K]: the canonical
        # Muon-managed matrix in the recovery trainer.
        pre_w = nn.Parameter(torch.randn(206, 86))
        optimizer = MuonH([pre_w], lr=1e-3)
        assert optimizer.muon_parameter_count() == 1


class TestRankConsistency:
    def test_two_simulated_ranks_produce_identical_updates(self):
        # DDP averages gradients before step(); simulate two ranks that both
        # hold the identical averaged gradient and assert the nonlinear Muon
        # update is bitwise identical.
        torch.manual_seed(6)
        initial = torch.randn(20, 10)
        params = [nn.Parameter(initial.clone()) for _ in range(2)]
        optimizers = [MuonH([p], lr=0.05, muon_lr_multiplier=10.0) for p in params]
        for step in range(5):
            averaged_grad = torch.randn(20, 10)  # same tensor on both "ranks"
            for p, opt in zip(params, optimizers):
                opt.zero_grad()
                p.grad = averaged_grad.clone()
                opt.step()
            assert torch.equal(params[0].detach(), params[1].detach()), step

    def test_mixed_groups_stay_consistent(self):
        torch.manual_seed(7)
        shapes = [(16, 8), (8,), ()]
        rank_params = [
            [nn.Parameter(torch.randn(shape)) for shape in shapes] for _ in range(2)
        ]
        for a, b in zip(*rank_params):
            b.data.copy_(a.data)
        optimizers = [MuonH(ps, lr=1e-2) for ps in rank_params]
        for _ in range(3):
            grads = [torch.randn(shape) for shape in shapes]
            for ps, opt in zip(rank_params, optimizers):
                opt.zero_grad()
                for p, g in zip(ps, grads):
                    p.grad = g.clone()
                opt.step()
            for a, b in zip(*rank_params):
                assert torch.equal(a.detach(), b.detach())


class TestDefaultAdamWIdentity:
    def test_recovery_config_optimizer_block_still_builds_torch_adamw(self):
        # The exact optimizer mapping from the 32B joint recovery config.
        config = OptimizerConfig(name="adamw", lr=3.0e-05, weight_decay=0.0)
        param = nn.Parameter(torch.randn(4, 3))
        optimizer = create_optimizer([param], config)
        assert type(optimizer) is torch.optim.AdamW
        assert not isinstance(optimizer, MuonH)

    def test_adamw_trajectory_is_bitwise_identical_to_plain_torch_adamw(self):
        torch.manual_seed(8)
        initial = torch.randn(10, 6)
        factory_param = nn.Parameter(initial.clone())
        reference_param = nn.Parameter(initial.clone())
        config = OptimizerConfig(name="adamw", lr=3.0e-05, weight_decay=0.0)
        factory_optimizer = create_optimizer([factory_param], config)
        reference_optimizer = torch.optim.AdamW(
            [reference_param],
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=config.betas,
            eps=config.eps,
        )
        for _ in range(10):
            grad = torch.randn(10, 6)
            for p, opt in (
                (factory_param, factory_optimizer),
                (reference_param, reference_optimizer),
            ):
                opt.zero_grad()
                p.grad = grad.clone()
                opt.step()
        assert torch.equal(factory_param.detach(), reference_param.detach())

    def test_muonh_constructed_only_under_muonh_name(self):
        param = nn.Parameter(torch.randn(4, 3))
        muonh_config = OptimizerConfig(name="muonh", lr=1e-3)
        optimizer = create_optimizer([param], muonh_config)
        assert isinstance(optimizer, MuonH)

    def test_muon_lr_multiplier_flows_from_config(self):
        param = nn.Parameter(torch.randn(4, 3))
        config = OptimizerConfig(name="muonh", lr=1e-3, muon_lr_multiplier=7.0)
        optimizer = create_optimizer([param], config)
        muon_group = next(g for g in optimizer.param_groups if g["use_muon"])
        assert muon_group["lr"] == 1e-3
        assert muon_group["muon_lr_multiplier"] == 7.0


class TestStateDictRoundTrip:
    def test_momentum_and_init_norm_survive_reload(self):
        torch.manual_seed(9)
        param = nn.Parameter(torch.randn(6, 4))
        optimizer = MuonH([param], lr=1e-2)
        param.grad = torch.randn_like(param)
        optimizer.step()
        # Serialize the way checkpoints.py does (torch.save of state_dict);
        # a raw in-process state_dict aliases the live momentum buffers.
        buffer = io.BytesIO()
        torch.save(optimizer.state_dict(), buffer)
        buffer.seek(0)
        payload = torch.load(buffer, weights_only=False)

        fresh_param = nn.Parameter(param.detach().clone())
        fresh = MuonH([fresh_param], lr=1e-2)
        fresh.load_state_dict(payload)
        state = fresh.state[fresh_param]
        assert "momentum_buffer" in state
        assert isinstance(state["init_frobenius_norm"], float)
        # Continued trajectories must match.
        grad = torch.randn(6, 4)
        param.grad = grad.clone()
        fresh_param.grad = grad.clone()
        optimizer.step()
        fresh.step()
        assert torch.equal(param.detach(), fresh_param.detach())


class TestEndToEndFit:
    def test_tiny_regression_loss_decreases(self):
        torch.manual_seed(10)
        model = nn.Sequential(nn.Linear(6, 16), nn.Tanh(), nn.Linear(16, 4))
        target_map = nn.Linear(6, 4)
        for p in target_map.parameters():
            p.requires_grad_(False)
        optimizer = MuonH(model.parameters(), lr=0.02, muon_lr_multiplier=1.0)
        inputs = torch.randn(256, 6)
        with torch.no_grad():
            targets = target_map(inputs)
        losses = []
        for _ in range(150):
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        assert losses[-1] < 0.5 * losses[0]
        assert all(math.isfinite(value) for value in losses)  # no blow-up


def test_a_zero_norm_matrix_trains_outside_the_hyperball():
    weight = torch.nn.Parameter(torch.zeros(4, 4))
    optimizer = MuonH([weight], lr=0.01, min_dim=2)
    weight.grad = torch.randn(4, 4, generator=torch.Generator().manual_seed(0))
    with pytest.warns(UserWarning, match="zero initial norm"):
        optimizer.step()
    assert float(weight.norm()) > 0.0
