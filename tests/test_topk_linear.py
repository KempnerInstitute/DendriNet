"""Tests for TopKLinear sparse synapse layer."""

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TOPK_INIT_METHODS,
    TopKLinear,
)

# ---------------------------------------------------------------------------
# Construction & output shape
# ---------------------------------------------------------------------------


class TestTopKLinearShape:
    def test_output_shape_basic(self):
        layer = TopKLinear(in_features=20, out_features=10, K=5)
        layer.initialize()
        x = torch.randn(4, 20)
        out = layer(x)
        assert out.shape == (4, 10)

    def test_output_shape_single_sample(self):
        layer = TopKLinear(in_features=16, out_features=8, K=4)
        layer.initialize()
        x = torch.randn(1, 16)
        out = layer(x)
        assert out.shape == (1, 8)

    def test_output_shape_1d_input(self):
        layer = TopKLinear(in_features=10, out_features=5, K=3)
        layer.initialize()
        x = torch.randn(10)  # 1D → unsqueezed to (1, 10)
        out = layer(x)
        assert out.shape == (1, 5)


# ---------------------------------------------------------------------------
# Weight positivity per transform
# ---------------------------------------------------------------------------


class TestTopKLinearWeightTransforms:
    def test_weight_positive_exp(self):
        layer = TopKLinear(in_features=16, out_features=8, K=4, weight_transform="exp")
        layer.initialize()
        assert (layer.weight() > 0).all()

    def test_weight_positive_softplus(self):
        layer = TopKLinear(
            in_features=16, out_features=8, K=4, weight_transform="softplus"
        )
        layer.initialize()
        assert (layer.weight() > 0).all()

    def test_weight_nonnegative_relu(self):
        layer = TopKLinear(in_features=16, out_features=8, K=4, weight_transform="relu")
        layer.initialize()
        assert (layer.weight() >= 0).all()

    def test_weight_identity_allows_negative(self):
        layer = TopKLinear(
            in_features=16, out_features=8, K=4, weight_transform="identity"
        )
        layer.initialize()
        # Identity: some weights should be negative (Xavier normal init)
        assert (layer.weight() < 0).any()


# ---------------------------------------------------------------------------
# Mask properties
# ---------------------------------------------------------------------------


class TestTopKLinearMask:
    def test_mask_exactly_k_per_row(self):
        layer = TopKLinear(in_features=20, out_features=10, K=5)
        layer.initialize()
        mask = layer.weight_mask()
        assert (mask.sum(dim=1) == 5).all()

    def test_mask_binary(self):
        layer = TopKLinear(in_features=20, out_features=10, K=5)
        layer.initialize()
        mask = layer.weight_mask()
        assert ((mask == 0) | (mask == 1)).all()

    def test_k_equals_in_features_dense(self):
        layer = TopKLinear(in_features=10, out_features=5, K=10)
        layer.initialize()
        mask = layer.weight_mask()
        assert (mask == 1).all()

    def test_k_equals_1_sparse(self):
        layer = TopKLinear(in_features=10, out_features=5, K=1)
        layer.initialize()
        mask = layer.weight_mask()
        assert (mask.sum(dim=1) == 1).all()

    def test_cache_mask_stores_last(self):
        layer = TopKLinear(in_features=10, out_features=5, K=3)
        layer.initialize()
        layer.cache_mask = True
        assert layer._last_forward_weight_mask is None
        x = torch.randn(2, 10)
        layer(x)
        assert layer._last_forward_weight_mask is not None
        assert layer._last_forward_weight_mask.shape == (5, 10)

    def test_noise_level_can_change_mask(self):
        layer = TopKLinear(in_features=20, out_features=10, K=5, noise_level=10.0)
        layer.initialize()
        masks = [layer.weight_mask() for _ in range(10)]
        # With high noise, masks should vary across calls
        not_all_same = any(not torch.equal(masks[0], m) for m in masks[1:])
        assert not_all_same


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestTopKLinearValidation:
    def test_k_less_than_1_raises(self):
        with pytest.raises(ValueError, match="K must be >= 1"):
            TopKLinear(in_features=10, out_features=5, K=0)

    def test_k_greater_than_in_features_raises(self):
        with pytest.raises(ValueError, match="K must be <="):
            TopKLinear(in_features=10, out_features=5, K=11)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestTopKLinearGradient:
    def test_gradient_flows_to_pre_w(self):
        layer = TopKLinear(in_features=16, out_features=8, K=4)
        layer.initialize()
        x = torch.randn(2, 16)
        out = layer(x)
        out.sum().backward()
        assert layer.pre_w.grad is not None
        assert (layer.pre_w.grad != 0).any()


# ---------------------------------------------------------------------------
# Weight normalization
# ---------------------------------------------------------------------------


class TestTopKLinearNorm:
    def test_weight_norm_order_normalizes_rows(self):
        layer = TopKLinear(
            in_features=16, out_features=8, K=4, weight_norm_order=2, gamma=1.0
        )
        layer.initialize()
        # Check that pruned_weight rows have unit L2 norm (scaled by gamma=1)
        pw = layer.pruned_weight()
        norms = torch.norm(pw, p=2, dim=-1)
        # Rows with nonzero weights should have norm = gamma = 1.0
        torch.testing.assert_close(norms, torch.ones(8), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Weight decay
# ---------------------------------------------------------------------------


class TestTopKLinearDecay:
    def test_decay_exp_reduces_active_weights(self):
        layer = TopKLinear(in_features=16, out_features=8, K=4, weight_transform="exp")
        layer.initialize()
        w_before = layer.weight().clone()
        mask = layer.weight_mask()
        layer.decay_weights(weight_decay=0.1)
        w_after = layer.weight()
        # Active weights should decrease
        active = mask > 0
        assert (w_after[active] < w_before[active]).all()

    def test_decay_softplus_reduces_active_weights(self):
        layer = TopKLinear(
            in_features=16, out_features=8, K=4, weight_transform="softplus"
        )
        layer.initialize()
        w_before = layer.weight().clone()
        mask = layer.weight_mask()
        layer.decay_weights(weight_decay=0.1)
        w_after = layer.weight()
        active = mask > 0
        assert (w_after[active] < w_before[active]).all()

    def test_decay_identity_preserves_sign(self):
        layer = TopKLinear(
            in_features=16, out_features=8, K=4, weight_transform="identity"
        )
        layer.initialize()
        signs_before = torch.sign(layer.pre_w.data.clone())
        layer.decay_weights(weight_decay=0.1)
        signs_after = torch.sign(layer.pre_w.data)
        # Signs should be preserved
        assert torch.equal(signs_before, signs_after)

    def test_decay_inactive_unchanged(self):
        layer = TopKLinear(in_features=16, out_features=8, K=4, weight_transform="exp")
        layer.initialize()
        pre_w_before = layer.pre_w.data.clone()
        mask = layer.weight_mask()
        layer.decay_weights(weight_decay=0.1)
        inactive = mask == 0
        torch.testing.assert_close(layer.pre_w.data[inactive], pre_w_before[inactive])

    def test_weight_boosting_increases_inactive(self):
        layer = TopKLinear(in_features=16, out_features=8, K=4, weight_transform="exp")
        layer.initialize()
        w_before = layer.weight().clone()
        mask = layer.weight_mask()
        layer.decay_weights(weight_decay=0.1, weight_boosting=True)
        w_after = layer.weight()
        inactive = mask == 0
        # Inactive weights should increase (boosted)
        assert (w_after[inactive] > w_before[inactive]).all()


# ---------------------------------------------------------------------------
# Initialization methods
# ---------------------------------------------------------------------------


class TestTopKLinearInit:
    @pytest.mark.parametrize("method", list(TOPK_INIT_METHODS.keys()))
    def test_init_method_produces_finite(self, method):
        layer = TopKLinear(in_features=16, out_features=8, K=4, init_method=method)
        layer.initialize()
        assert torch.isfinite(layer.pre_w).all()

    def test_invalid_init_method_raises(self):
        layer = TopKLinear(
            in_features=16, out_features=8, K=4, init_method="nonexistent"
        )
        with pytest.raises(ValueError, match="Invalid initialization method"):
            layer.initialize()
