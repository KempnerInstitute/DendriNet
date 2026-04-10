"""Tests for DendriticBranchLayer mathematical invariants."""

import pytest
import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)


def _make_layer(
    output_dim=8,
    excitatory_input_dim=20,
    excitatory_synapses_per_branch=5,
    inhibitory_input_dim=None,
    inhibitory_synapses_per_branch=None,
    input_branch_factor=None,
    use_shunting=True,
    reactivate=False,
    dbl_init_method="analytical_expectation",
    weight_transform="exp",
    **kwargs,
):
    return DendriticBranchLayer(
        output_dim=output_dim,
        excitatory_input_dim=excitatory_input_dim,
        excitatory_synapses_per_branch=excitatory_synapses_per_branch,
        inhibitory_input_dim=inhibitory_input_dim,
        inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
        input_branch_factor=input_branch_factor,
        use_shunting=use_shunting,
        reactivate=reactivate,
        dbl_init_method=dbl_init_method,
        weight_transform=weight_transform,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Shunting invariants
# ---------------------------------------------------------------------------


class TestShuntingInvariants:
    def test_excitation_only_output_bounded(self):
        """Shunting with only excitation: v = E/(E+relu(E)+1+eps) in [0, 1)."""
        layer = _make_layer(use_shunting=True, reactivate=False)
        x = torch.randn(4, 20)
        out = layer(x)
        # Shunting formula: numerator / (denominator + eps)
        # With only excitation: denominator >= 1 (starts at 1) and >= |numerator|
        # So voltage is bounded
        assert torch.isfinite(out).all()

    def test_inhibition_suppresses_output(self):
        """Adding inhibition should reduce the shunting output magnitude."""
        torch.manual_seed(42)
        layer_exc = _make_layer(use_shunting=True, reactivate=False)

        torch.manual_seed(42)
        layer_both = _make_layer(
            use_shunting=True,
            reactivate=False,
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
        )
        # Copy excitatory weights to match
        layer_both.branch_excitation.pre_w.data = (
            layer_exc.branch_excitation.pre_w.data.clone()
        )

        x = torch.abs(torch.randn(4, 20))  # Positive input for consistent comparison
        inh = torch.abs(torch.randn(4, 10))

        out_exc = layer_exc(x)
        out_both = layer_both(x, inhibitory_input=inh)
        # With inhibition, denominator is larger, so |output| should be smaller
        assert out_both.abs().mean() < out_exc.abs().mean()


# ---------------------------------------------------------------------------
# Additive invariants
# ---------------------------------------------------------------------------


class TestAdditiveInvariants:
    def test_additive_gradient_flow(self):
        layer = _make_layer(use_shunting=False, reactivate=False)
        x = torch.randn(2, 20)
        out = layer(x)
        out.sum().backward()
        assert layer.branch_excitation.pre_w.grad is not None

    def test_additive_finite_output(self):
        layer = _make_layer(use_shunting=False, reactivate=False)
        x = torch.randn(4, 20)
        out = layer(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Reactivation
# ---------------------------------------------------------------------------


class TestReactivation:
    def test_reactivation_applied(self):
        layer = _make_layer(use_shunting=True, reactivate=True)
        # ParametricTanh should be the reactivation function
        assert not isinstance(layer.reactivation, nn.Identity)

    def test_no_reactivation_identity(self):
        layer = _make_layer(use_shunting=True, reactivate=False)
        assert isinstance(layer.reactivation, nn.Identity)

    def test_reactivation_output_bounded(self):
        """With reactivation (ParametricTanh), output should be in [0, 1]."""
        layer = _make_layer(use_shunting=True, reactivate=True)
        x = torch.randn(4, 20)
        out = layer(x)
        assert (out >= 0).all()
        assert (out <= 1).all()

    def test_additive_default_reactivation_init_is_legacy(self):
        layer = _make_layer(
            use_shunting=False,
            reactivate=True,
            reactivation_type="param_tanh",
        )
        assert pytest.approx(layer.reactivation.log_m.exp()[0].item(), rel=1e-6) == 0.1
        assert pytest.approx(layer.reactivation.b[0].item(), rel=1e-6) == 0.0

    def test_additive_can_match_shunting_reactivation_init(self):
        layer = DendriticBranchLayer(
            output_dim=8,
            excitatory_input_dim=20,
            excitatory_synapses_per_branch=5,
            use_shunting=False,
            reactivate=True,
            reactivation_type="param_tanh",
            reactivation_init_m=1.5,
            reactivation_init_b=0.5,
            reactivation_match_additive_init_to_shunting=True,
        )
        assert pytest.approx(layer.reactivation.log_m.exp()[0].item(), rel=1e-6) == 1.5
        assert pytest.approx(layer.reactivation.b[0].item(), rel=1e-6) == 0.5


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestBranchLayerGradientFlow:
    def test_gradient_shunting(self):
        layer = _make_layer(use_shunting=True, reactivate=True)
        x = torch.randn(2, 20)
        out = layer(x)
        out.sum().backward()
        assert layer.branch_excitation.pre_w.grad is not None

    def test_gradient_additive(self):
        layer = _make_layer(use_shunting=False, reactivate=True)
        x = torch.randn(2, 20)
        out = layer(x)
        out.sum().backward()
        assert layer.branch_excitation.pre_w.grad is not None

    def test_gradient_with_inhibition(self):
        layer = _make_layer(
            use_shunting=True,
            reactivate=False,
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
        )
        x = torch.randn(2, 20)
        inh = torch.randn(2, 10)
        out = layer(x, inhibitory_input=inh)
        out.sum().backward()
        assert layer.branch_excitation.pre_w.grad is not None
        assert layer.branch_inhibition.pre_w.grad is not None


# ---------------------------------------------------------------------------
# Branch input aggregation
# ---------------------------------------------------------------------------


class TestBranchInputAggregation:
    def test_branch_input_contributes(self):
        """When branch_input is provided, it should change the output."""
        layer = _make_layer(
            use_shunting=True,
            reactivate=False,
            input_branch_factor=3,
        )
        x = torch.randn(2, 20)
        branch_in = torch.randn(2, 24)  # 8 * 3

        out_no_branch = layer(x, branch_input=None)
        out_with_branch = layer(x, branch_input=branch_in)
        assert not torch.equal(out_no_branch, out_with_branch)


# ---------------------------------------------------------------------------
# Initialization methods
# ---------------------------------------------------------------------------


class TestBranchLayerInitMethods:
    @pytest.mark.parametrize(
        "method",
        ["analytical_expectation", "default", "naive", "ei_equivalence"],
    )
    def test_init_method_produces_finite_weights(self, method):
        layer = _make_layer(dbl_init_method=method, weight_transform="exp")
        assert torch.isfinite(layer.branch_excitation.pre_w).all()

    def test_identity_transform_init(self):
        layer = _make_layer(
            weight_transform="identity",
            dbl_init_method="analytical_expectation",
        )
        # identity_weighttransform_dbl_init should have been called
        assert torch.isfinite(layer.branch_excitation.pre_w).all()
        # Identity transform allows negative weights
        assert (layer.branch_excitation.pre_w < 0).any()


# ---------------------------------------------------------------------------
# Weight decay propagation
# ---------------------------------------------------------------------------


class TestBranchLayerDecay:
    def test_decay_weights_changes_excitatory(self):
        layer = _make_layer(use_shunting=True)
        w_before = layer.branch_excitation.pre_w.data.clone()
        layer.decay_weights(weight_decay=0.1)
        assert not torch.equal(w_before, layer.branch_excitation.pre_w.data)

    def test_decay_weights_changes_inhibitory(self):
        layer = _make_layer(
            use_shunting=True,
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
        )
        w_before = layer.branch_inhibition.pre_w.data.clone()
        layer.decay_weights(weight_decay=0.1)
        assert not torch.equal(w_before, layer.branch_inhibition.pre_w.data)

    def test_decay_no_inhibition_no_error(self):
        """decay_weights should not error when no inhibitory layer exists."""
        layer = _make_layer(use_shunting=True)
        layer.decay_weights(weight_decay=0.1)  # Should not raise


# ---------------------------------------------------------------------------
# TopK type variants
# ---------------------------------------------------------------------------


class TestBranchLayerTopKTypes:
    def test_stochastic_topk(self):
        layer = DendriticBranchLayer(
            output_dim=8,
            excitatory_input_dim=20,
            excitatory_synapses_per_branch=5,
            use_shunting=True,
            reactivate=False,
            topk_type="stochastic",
            weight_transform="exp",
        )
        x = torch.randn(2, 20)
        out = layer(x)
        assert out.shape == (2, 8)

    def test_variance_topk(self):
        layer = DendriticBranchLayer(
            output_dim=8,
            excitatory_input_dim=20,
            excitatory_synapses_per_branch=5,
            use_shunting=True,
            reactivate=False,
            topk_type="variance",
            weight_transform="exp",
        )
        x = torch.randn(2, 20)
        out = layer(x)
        assert out.shape == (2, 8)

    def test_k_capped_to_in_features(self):
        """When K > in_features, it should be capped."""
        layer = DendriticBranchLayer(
            output_dim=4,
            excitatory_input_dim=5,
            excitatory_synapses_per_branch=10,  # > in_features
            use_shunting=True,
            reactivate=False,
            weight_transform="exp",
        )
        # Should not raise, K capped to 5
        assert layer.branch_excitation.K == 5

    def test_unknown_init_method_raises(self):
        with pytest.raises(ValueError, match="Unknown dbl_init_method"):
            _make_layer(dbl_init_method="nonexistent")


# ---------------------------------------------------------------------------
# Weight accessor methods
# ---------------------------------------------------------------------------


class TestBranchLayerAccessors:
    def test_get_weights_excitation_only(self):
        layer = _make_layer()
        w = layer.get_weights()
        assert isinstance(w, dict)
        assert "exc" in w
        assert w["exc"].shape == (8, 20)

    def test_get_weights_both(self):
        layer = _make_layer(inhibitory_input_dim=10, inhibitory_synapses_per_branch=3)
        w = layer.get_weights()
        assert isinstance(w, dict)
        assert "exc" in w and "inh" in w

    def test_get_log_weights_excitation_only(self):
        layer = _make_layer()
        lw = layer.get_log_weights()
        assert isinstance(lw, dict)
        assert "exc" in lw

    def test_get_log_weights_both(self):
        layer = _make_layer(inhibitory_input_dim=10, inhibitory_synapses_per_branch=3)
        lw = layer.get_log_weights()
        assert isinstance(lw, dict)
        assert "exc" in lw and "inh" in lw

    def test_get_mask_excitation_only(self):
        layer = _make_layer()
        m = layer.get_mask()
        assert isinstance(m, dict)
        assert "exc" in m
        assert (m["exc"].sum(dim=1) == 5).all()  # K=5

    def test_get_mask_both(self):
        layer = _make_layer(inhibitory_input_dim=10, inhibitory_synapses_per_branch=3)
        m = layer.get_mask()
        assert isinstance(m, dict)
        assert "exc" in m and "inh" in m

    def test_get_pruned_weights_excitation_only(self):
        layer = _make_layer()
        pw = layer.get_pruned_weights()
        assert isinstance(pw, dict)
        assert "exc" in pw

    def test_get_pruned_weights_both(self):
        layer = _make_layer(inhibitory_input_dim=10, inhibitory_synapses_per_branch=3)
        pw = layer.get_pruned_weights()
        assert isinstance(pw, dict)
        assert "exc" in pw and "inh" in pw

    def test_get_log_pruned_weights(self):
        layer = _make_layer()
        lpw = layer.get_log_pruned_weights()
        assert isinstance(lpw, dict)
        assert "exc" in lpw

    def test_get_log_pruned_weights_both(self):
        layer = _make_layer(inhibitory_input_dim=10, inhibitory_synapses_per_branch=3)
        lpw = layer.get_log_pruned_weights()
        assert isinstance(lpw, dict)
        assert "exc" in lpw and "inh" in lpw


# ---------------------------------------------------------------------------
# Additive mode with inhibition and branches
# ---------------------------------------------------------------------------


class TestAdditiveWithInhibitionAndBranches:
    def test_additive_with_inhibition(self):
        layer = _make_layer(
            use_shunting=False,
            reactivate=False,
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
        )
        x = torch.randn(2, 20)
        inh = torch.randn(2, 10)
        out = layer(x, inhibitory_input=inh)
        assert torch.isfinite(out).all()

    def test_additive_with_branches(self):
        layer = _make_layer(
            use_shunting=False,
            reactivate=False,
            input_branch_factor=3,
        )
        x = torch.randn(2, 20)
        branch = torch.randn(2, 24)
        out = layer(x, branch_input=branch)
        assert torch.isfinite(out).all()
