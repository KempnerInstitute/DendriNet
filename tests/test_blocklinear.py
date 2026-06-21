"""Tests for BlockLinear and EfficientBlockLinear layers."""

import unittest

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
    EfficientBlockLinear,
)

# ---------------------------------------------------------------------------
# Existing tests (migrated from unittest)
# ---------------------------------------------------------------------------


class TestBlockLinearBasic(unittest.TestCase):
    def test_basic_functionality(self):
        in_features = 8
        out_features = 2
        batch_size = 4
        model = BlockLinear(in_features, out_features)
        x = torch.randn(batch_size, in_features)
        output = model(x)
        self.assertEqual(output.shape, (batch_size, out_features))
        self.assertIsInstance(output, torch.Tensor)

    def test_edge_case_small_input(self):
        in_features = 2
        out_features = 1
        batch_size = 4
        model = BlockLinear(in_features, out_features)
        x = torch.randn(batch_size, in_features)
        output = model(x)
        self.assertEqual(output.shape, (batch_size, out_features))

    def test_error_handling_in_features_not_divisible(self):
        with self.assertRaises(AssertionError):
            BlockLinear(7, 2)

    def test_input_shape_mismatch(self):
        model = BlockLinear(8, 2)
        x = torch.randn(4, 9)
        with self.assertRaises(ValueError):
            model(x)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestBlockLinearShape:
    def test_output_shape(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        x = torch.randn(3, 12)
        out = bl(x)
        assert out.shape == (3, 4)

    def test_output_shape_1d_input(self):
        bl = BlockLinear(in_features=9, out_features=3, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        x = torch.randn(9)
        out = bl(x)
        assert out.shape == (1, 3)


# ---------------------------------------------------------------------------
# Block diagonal structure
# ---------------------------------------------------------------------------


class TestBlockLinearStructure:
    def test_block_diagonal_zeros_off_diagonal(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        full = bl.block()
        # For each row i, only columns [i*3, i*3+1, i*3+2] should be nonzero
        for i in range(4):
            for j in range(12):
                if j // 3 == i:
                    assert full[i, j] > 0, f"Expected nonzero at ({i}, {j})"
                else:
                    assert full[i, j] == 0, f"Expected zero at ({i}, {j})"


# ---------------------------------------------------------------------------
# Weight transform positivity
# ---------------------------------------------------------------------------


class TestBlockLinearWeightTransform:
    def test_weight_positive_exp(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        assert (bl.weight() > 0).all()

    def test_weight_positive_softplus(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="softplus")
        bl.initialize(g_branch=1.0)
        assert (bl.weight() > 0).all()


# ---------------------------------------------------------------------------
# Sum conductances
# ---------------------------------------------------------------------------


class TestBlockLinearConductances:
    def test_sum_conductances_shape(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        sc = bl.sum_conductances()
        assert sc.shape == (4,)

    def test_sum_conductances_positive_exp(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        assert (bl.sum_conductances() > 0).all()


# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------


class TestBlockLinearInit:
    def test_initialize_sets_correct_conductance(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        target_g = 2.0
        bl.initialize(g_branch=target_g)
        torch.testing.assert_close(
            bl.weight(),
            torch.full_like(bl.weight(), target_g),
            atol=1e-4,
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestBlockLinearGradient:
    def test_gradient_flow(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        x = torch.randn(2, 12)
        out = bl(x)
        out.sum().backward()
        assert bl.log_weight.grad is not None
        assert (bl.log_weight.grad != 0).any()


# ---------------------------------------------------------------------------
# EfficientBlockLinear matches standard
# ---------------------------------------------------------------------------


class TestEfficientBlockLinear:
    def test_matches_standard_output(self):
        torch.manual_seed(0)
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.5)

        ebl = EfficientBlockLinear(
            in_features=12, out_features=4, weight_transform="exp"
        )
        ebl.log_weight.data = bl.log_weight.data.clone()

        x = torch.randn(3, 12)
        out_standard = bl(x)
        out_efficient = ebl(x)
        torch.testing.assert_close(out_standard, out_efficient, atol=1e-5, rtol=1e-5)

    def test_efficient_gradient_flow(self):
        ebl = EfficientBlockLinear(
            in_features=12, out_features=4, weight_transform="exp"
        )
        ebl.initialize(g_branch=1.0)
        x = torch.randn(2, 12)
        out = ebl(x)
        out.sum().backward()
        assert ebl.log_weight.grad is not None

    def test_input_shape_mismatch(self):
        ebl = EfficientBlockLinear(
            in_features=12, out_features=4, weight_transform="exp"
        )
        x = torch.randn(2, 11)
        with pytest.raises(ValueError):
            ebl(x)


# ---------------------------------------------------------------------------
# Weight decay
# ---------------------------------------------------------------------------


class TestBlockLinearDecay:
    def test_decay_exp_reduces_weights(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=2.0)
        w_before = bl.weight().clone()
        bl.decay_weights(weight_decay=0.1)
        w_after = bl.weight()
        assert (w_after < w_before).all()

    def test_decay_softplus_reduces_weights(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="softplus")
        bl.initialize(g_branch=2.0)
        w_before = bl.weight().clone()
        bl.decay_weights(weight_decay=0.1)
        w_after = bl.weight()
        assert (w_after < w_before).all()

    def test_decay_identity_reduces_magnitude(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="identity")
        bl.log_weight.data = torch.randn(4, 3)
        mag_before = bl.log_weight.data.abs().clone()
        bl.decay_weights(weight_decay=0.1)
        mag_after = bl.log_weight.data.abs()
        assert (mag_after <= mag_before + 1e-7).all()

    def test_boosting_exp_increases_weights(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        bl.decay_weights(weight_decay=0.1, weight_boosting=True)
        assert torch.isfinite(bl.weight()).all()

    def test_boosting_softplus(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="softplus")
        bl.initialize(g_branch=1.0)
        bl.decay_weights(weight_decay=0.1, weight_boosting=True)
        assert torch.isfinite(bl.weight()).all()

    def test_boosting_identity(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="identity")
        bl.log_weight.data = torch.randn(4, 3)
        bl.decay_weights(weight_decay=0.1, weight_boosting=True)
        assert torch.isfinite(bl.log_weight.data).all()


# ---------------------------------------------------------------------------
# log_block and grad_block
# ---------------------------------------------------------------------------


class TestBlockLinearAccessors:
    def test_log_block_shape(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        lb = bl.log_block()
        assert lb.shape == (4, 12)

    def test_log_block_off_diagonal_negative(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        lb = bl.log_block()
        # Off-diagonal blocks filled with -10
        for i in range(4):
            for j in range(12):
                if j // 3 != i:
                    assert lb[i, j] == -10

    def test_grad_block_shape(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        gb = bl.grad_block()
        assert gb.shape == (4, 12)

    def test_grad_block_after_backward(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        x = torch.randn(2, 12)
        out = bl(x)
        out.sum().backward()
        gb = bl.grad_block()
        # Should have nonzero gradients in block-diagonal positions
        assert (gb != 0).any()

    def test_grad_block_no_grad(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        # No backward called → grad_block should be all zeros
        gb = bl.grad_block()
        assert (gb == 0).all()


# ---------------------------------------------------------------------------
# 3D input handling
# ---------------------------------------------------------------------------


class TestBlockLinear3DInput:
    def test_3d_input_flattened(self):
        bl = BlockLinear(in_features=12, out_features=4, weight_transform="exp")
        bl.initialize(g_branch=1.0)
        x = torch.randn(2, 3, 12)  # 3D → flattened to (6, 12)
        out = bl(x)
        assert out.shape == (2, 3, 4)  # Restored to original batch dims
