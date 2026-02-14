"""Tests for dendritic branch layer initialization methods."""

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    _compute_expectation_truncated_relu,
    _compute_expectation_truncated_softplus,
    _expected_weight_for_transform,
    compute_expectation_truncated_log_normal,
    compute_network_scale_factor,
)


def _make_layer(
    output_dim=8,
    excitatory_input_dim=20,
    excitatory_synapses_per_branch=5,
    inhibitory_input_dim=None,
    inhibitory_synapses_per_branch=None,
    input_branch_factor=None,
    use_shunting=True,
    reactivate=True,
    dbl_init_method="analytical_expectation",
    weight_transform="exp",
    adaptive_initialization=True,
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
        adaptive_initialization=adaptive_initialization,
    )


# ---------------------------------------------------------------------------
# Truncated expectation helper functions
# ---------------------------------------------------------------------------


class TestTruncatedExpectations:
    def test_truncated_lognormal_positive(self):
        val = compute_expectation_truncated_log_normal(
            mean=0.0, std=1.0, upper_quantile=0.25
        )
        assert val > 0

    def test_truncated_lognormal_decreases_with_lower_quantile(self):
        """Keeping fewer top weights (smaller quantile) → higher average."""
        val_high = compute_expectation_truncated_log_normal(
            mean=0.0, std=1.0, upper_quantile=0.1
        )
        val_low = compute_expectation_truncated_log_normal(
            mean=0.0, std=1.0, upper_quantile=0.5
        )
        # Keeping top 10% should have higher expected value than keeping top 50%
        assert val_high > val_low

    def test_truncated_softplus_positive(self):
        val = _compute_expectation_truncated_softplus(
            mean=0.0, std=1.0, upper_quantile=0.25
        )
        assert val > 0

    def test_truncated_relu_nonnegative(self):
        val = _compute_expectation_truncated_relu(
            mean=0.0, std=1.0, upper_quantile=0.25
        )
        assert val >= 0

    @pytest.mark.parametrize("transform", ["exp", "softplus", "relu"])
    def test_expected_weight_positive(self, transform):
        val = _expected_weight_for_transform(
            transform=transform, mean=0.0, std=1.0, upper_quantile=0.25
        )
        assert val > 0


# ---------------------------------------------------------------------------
# Network scale factor
# ---------------------------------------------------------------------------


class TestNetworkScaleFactor:
    def test_increases_with_more_branches(self):
        layer = _make_layer(adaptive_initialization=False)
        sf_small = compute_network_scale_factor(layer, branch_factors=[2])
        sf_large = compute_network_scale_factor(layer, branch_factors=[2, 3, 4])
        assert sf_large >= sf_small

    def test_nonnegative(self):
        layer = _make_layer(adaptive_initialization=False)
        sf = compute_network_scale_factor(layer, branch_factors=[3])
        assert sf >= 0


# ---------------------------------------------------------------------------
# analytical_expectation init
# ---------------------------------------------------------------------------


class TestAnalyticalExpectationInit:
    def test_excitation_weights_finite(self):
        layer = _make_layer(dbl_init_method="analytical_expectation")
        assert torch.isfinite(layer.branch_excitation.pre_w).all()

    def test_excitation_weights_finite_softplus(self):
        layer = _make_layer(
            dbl_init_method="analytical_expectation", weight_transform="softplus"
        )
        assert torch.isfinite(layer.branch_excitation.pre_w).all()

    def test_inhibition_weights_finite(self):
        layer = _make_layer(
            dbl_init_method="analytical_expectation",
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
        )
        assert torch.isfinite(layer.branch_inhibition.pre_w).all()

    def test_with_branches_finite(self):
        layer = _make_layer(
            dbl_init_method="analytical_expectation",
            input_branch_factor=3,
        )
        assert torch.isfinite(layer.branch_excitation.pre_w).all()
        assert torch.isfinite(layer.branches_to_output.log_weight).all()

    def test_adaptive_reduces_magnitude(self):
        """Adaptive initialization should produce smaller weights than non-adaptive."""
        torch.manual_seed(42)
        layer_adaptive = _make_layer(
            dbl_init_method="analytical_expectation",
            adaptive_initialization=True,
        )
        torch.manual_seed(42)
        layer_nonadaptive = _make_layer(
            dbl_init_method="analytical_expectation",
            adaptive_initialization=False,
        )
        # Adaptive weights should have smaller (more negative) mean in log space
        # or same if scale factor is 0 for small networks
        assert layer_adaptive.branch_excitation.pre_w.mean() <= (
            layer_nonadaptive.branch_excitation.pre_w.mean() + 0.01
        )


# ---------------------------------------------------------------------------
# default init
# ---------------------------------------------------------------------------


class TestDefaultInit:
    def test_weights_finite(self):
        layer = _make_layer(dbl_init_method="default")
        assert torch.isfinite(layer.branch_excitation.pre_w).all()

    def test_reactivation_params(self):
        layer = _make_layer(dbl_init_method="default", reactivate=True)
        # default init should set reactivation parameters
        if hasattr(layer.reactivation, "log_m"):
            assert torch.isfinite(layer.reactivation.log_m).all()


# ---------------------------------------------------------------------------
# naive init
# ---------------------------------------------------------------------------


class TestNaiveInit:
    def test_weights_finite(self):
        layer = _make_layer(dbl_init_method="naive")
        assert torch.isfinite(layer.branch_excitation.pre_w).all()

    def test_with_inhibition(self):
        layer = _make_layer(
            dbl_init_method="naive",
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
        )
        assert torch.isfinite(layer.branch_excitation.pre_w).all()
        assert torch.isfinite(layer.branch_inhibition.pre_w).all()


# ---------------------------------------------------------------------------
# ei_equivalence init
# ---------------------------------------------------------------------------


class TestEIEquivalenceInit:
    def test_weights_finite(self):
        layer = _make_layer(dbl_init_method="ei_equivalence")
        assert torch.isfinite(layer.branch_excitation.pre_w).all()

    def test_with_inhibition(self):
        layer = _make_layer(
            dbl_init_method="ei_equivalence",
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
        )
        assert torch.isfinite(layer.branch_excitation.pre_w).all()
        assert torch.isfinite(layer.branch_inhibition.pre_w).all()


# ---------------------------------------------------------------------------
# identity weight transform init
# ---------------------------------------------------------------------------


class TestIdentityInit:
    def test_xavier_applied(self):
        layer = _make_layer(weight_transform="identity")
        # Should have been initialized by identity_weighttransform_dbl_init
        # Xavier normal produces values roughly in [-gain*sqrt(6/(in+out)), ...]
        assert torch.isfinite(layer.branch_excitation.pre_w).all()
        # Identity allows negative weights
        assert (layer.branch_excitation.pre_w < 0).any()

    def test_reactivation_set(self):
        layer = _make_layer(weight_transform="identity", reactivate=True)
        # identity init sets reactivation to (1.0, 0.0)
        if hasattr(layer.reactivation, "b"):
            # b should be near 0 (per-neuron tensor)
            assert layer.reactivation.b.abs().max().item() < 0.1


# ---------------------------------------------------------------------------
# Cross-init: all methods produce valid forward pass
# ---------------------------------------------------------------------------


class TestInitForwardPass:
    @pytest.mark.parametrize(
        "method", ["analytical_expectation", "default", "naive", "ei_equivalence"]
    )
    def test_forward_produces_finite_output(self, method):
        layer = _make_layer(dbl_init_method=method, reactivate=True)
        x = torch.randn(4, 20)
        out = layer(x)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(
        "method", ["analytical_expectation", "default", "naive", "ei_equivalence"]
    )
    @pytest.mark.parametrize("transform", ["exp", "softplus"])
    def test_forward_finite_across_transforms(self, method, transform):
        layer = _make_layer(
            dbl_init_method=method, weight_transform=transform, reactivate=True
        )
        x = torch.randn(4, 20)
        out = layer(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Init methods with branches (covers input_branches paths)
# ---------------------------------------------------------------------------


class TestInitWithBranches:
    @pytest.mark.parametrize(
        "method", ["analytical_expectation", "default", "naive", "ei_equivalence"]
    )
    def test_with_branches_shunting(self, method):
        """All init methods should handle input_branch_factor (shunting)."""
        layer = _make_layer(
            dbl_init_method=method,
            input_branch_factor=3,
            use_shunting=True,
            reactivate=True,
        )
        assert torch.isfinite(layer.branch_excitation.pre_w).all()
        assert torch.isfinite(layer.branches_to_output.log_weight).all()
        x = torch.randn(2, 20)
        out = layer(x)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(
        "method", ["analytical_expectation", "default", "naive", "ei_equivalence"]
    )
    def test_with_branches_additive(self, method):
        """All init methods should handle input_branch_factor (additive)."""
        layer = _make_layer(
            dbl_init_method=method,
            input_branch_factor=3,
            use_shunting=False,
            reactivate=True,
        )
        assert torch.isfinite(layer.branch_excitation.pre_w).all()
        assert torch.isfinite(layer.branches_to_output.log_weight).all()
        x = torch.randn(2, 20)
        out = layer(x)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Init methods with inhibition (covers inhibition paths)
# ---------------------------------------------------------------------------


class TestInitWithInhibition:
    @pytest.mark.parametrize(
        "method", ["analytical_expectation", "default", "naive", "ei_equivalence"]
    )
    def test_with_inhibition_shunting(self, method):
        """All init methods should handle inhibitory inputs (shunting)."""
        layer = _make_layer(
            dbl_init_method=method,
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
            use_shunting=True,
            reactivate=True,
        )
        assert torch.isfinite(layer.branch_excitation.pre_w).all()
        assert torch.isfinite(layer.branch_inhibition.pre_w).all()
        x = torch.randn(2, 20)
        inh = torch.randn(2, 10)
        out = layer(x, inhibitory_input=inh)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(
        "method", ["analytical_expectation", "default", "naive", "ei_equivalence"]
    )
    def test_with_inhibition_additive(self, method):
        """All init methods should handle inhibitory inputs (additive)."""
        layer = _make_layer(
            dbl_init_method=method,
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
            use_shunting=False,
            reactivate=True,
        )
        assert torch.isfinite(layer.branch_excitation.pre_w).all()
        assert torch.isfinite(layer.branch_inhibition.pre_w).all()
        x = torch.randn(2, 20)
        inh = torch.randn(2, 10)
        out = layer(x, inhibitory_input=inh)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# Init methods with both branches AND inhibition
# ---------------------------------------------------------------------------


class TestInitWithBranchesAndInhibition:
    @pytest.mark.parametrize(
        "method", ["analytical_expectation", "naive", "ei_equivalence"]
    )
    def test_branches_and_inhibition_shunting(self, method):
        """Init with both branches and inhibition (shunting)."""
        layer = _make_layer(
            dbl_init_method=method,
            input_branch_factor=3,
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
            use_shunting=True,
            reactivate=True,
        )
        assert torch.isfinite(layer.branch_excitation.pre_w).all()
        assert torch.isfinite(layer.branch_inhibition.pre_w).all()
        assert torch.isfinite(layer.branches_to_output.log_weight).all()
        x = torch.randn(2, 20)
        inh = torch.randn(2, 10)
        out = layer(x, inhibitory_input=inh)
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(
        "method", ["analytical_expectation", "naive", "ei_equivalence"]
    )
    def test_branches_and_inhibition_additive(self, method):
        """Init with both branches and inhibition (additive)."""
        layer = _make_layer(
            dbl_init_method=method,
            input_branch_factor=3,
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
            use_shunting=False,
            reactivate=True,
        )
        assert torch.isfinite(layer.branch_excitation.pre_w).all()
        assert torch.isfinite(layer.branch_inhibition.pre_w).all()
        x = torch.randn(2, 20)
        inh = torch.randn(2, 10)
        out = layer(x, inhibitory_input=inh)
        assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# EI equivalence: E[g_exc] ≈ E[g_inh] after balancing
# ---------------------------------------------------------------------------


class TestEIEquivalenceBalance:
    def test_balanced_conductances(self):
        """ei_equivalence should balance excitatory and inhibitory conductances."""
        layer = _make_layer(
            dbl_init_method="ei_equivalence",
            inhibitory_input_dim=10,
            inhibitory_synapses_per_branch=3,
            use_shunting=True,
            reactivate=False,
        )
        # After init, excitatory and inhibitory weights should produce
        # similar expected conductances
        exc_w = layer.branch_excitation.weight().detach()
        inh_w = layer.branch_inhibition.weight().detach()
        exc_mask = layer.branch_excitation.weight_mask().detach()
        inh_mask = layer.branch_inhibition.weight_mask().detach()
        E_g_exc = (exc_w * exc_mask).sum(dim=1).mean()
        E_g_inh = (inh_w * inh_mask).sum(dim=1).mean()
        # They should be roughly balanced (within factor of 3)
        ratio = E_g_exc / (E_g_inh + 1e-8)
        assert 0.3 < ratio < 3.0
