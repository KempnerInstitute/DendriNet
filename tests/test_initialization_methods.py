"""Tests for dendritic branch layer initialization methods."""

from unittest.mock import patch

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    initialize as init_mod,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    _aggregate_quantiles_from_chunks,
    _compute_expectation_truncated_relu,
    _compute_expectation_truncated_softplus,
    _expected_weight_for_transform,
    calibrate_reactivation_from_data,
    compute_expectation_truncated_log_normal,
    compute_network_scale_factor,
)
from dendritic_modeling.scripts.training.train_experiments import (
    _maybe_calibrate_data_driven_reactivation,
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

    def test_additive_fixed_policy_respects_yaml(self):
        layer = DendriticBranchLayer(
            output_dim=8,
            excitatory_input_dim=20,
            excitatory_synapses_per_branch=5,
            use_shunting=False,
            reactivate=True,
            reactivation_type="param_tanh",
            reactivation_init_m=1.5,
            reactivation_init_b=0.5,
            reactivation_init_policy="fixed",
            dbl_init_method="analytical_expectation",
        )
        assert layer.reactivation.log_m.exp()[0].item() == pytest.approx(1.5, rel=1e-6)
        assert layer.reactivation.b[0].item() == pytest.approx(0.5, rel=1e-6)

    @pytest.mark.parametrize(
        ("policy", "canonical"),
        [("auto", "analytical"), ("manual", "fixed")],
    )
    def test_legacy_policy_aliases_normalize_to_canonical_names(
        self, policy, canonical
    ):
        layer = DendriticBranchLayer(
            output_dim=8,
            excitatory_input_dim=20,
            excitatory_synapses_per_branch=5,
            use_shunting=False,
            reactivate=True,
            reactivation_type="param_tanh",
            reactivation_init_policy=policy,
            dbl_init_method="analytical_expectation",
        )
        assert layer.reactivation_init_policy == canonical


# ---------------------------------------------------------------------------
# Empirical calibration (from-data)
# ---------------------------------------------------------------------------


class TestEmpiricalCalibration:
    def test_calibrate_updates_reactivation_mb(self):
        """Data-driven calibration sets (m, b) from measured voltages."""
        torch.manual_seed(0)
        layer = DendriticBranchLayer(
            output_dim=16,
            excitatory_input_dim=50,
            excitatory_synapses_per_branch=10,
            inhibitory_input_dim=50,
            inhibitory_synapses_per_branch=5,
            use_shunting=False,
            reactivate=True,
            reactivation_type="param_tanh",
            reactivation_init_policy="analytical",
            reactivation_sigma_aware_k=0.25,
            dbl_init_method="analytical_expectation",
            weight_transform="softplus",
        )

        m_before = float(layer.reactivation.log_m.exp()[0].item())
        b_before = float(layer.reactivation.b[0].item())

        # Wrap the layer in a tiny Module so the helper can traverse it.
        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x, inh):
                return self.inner(x, inh)

        model = Wrapper(layer)

        batches = [
            (torch.randn(128, 50), torch.relu(torch.randn(128, 50))) for _ in range(3)
        ]
        diag = calibrate_reactivation_from_data(model, batches, k=0.25)

        assert len(diag) == 1
        stats = next(iter(diag.values()))
        assert stats["n"] > 0
        assert stats["V_std"] > 0

        m_after = float(layer.reactivation.log_m.exp()[0].item())
        b_after = float(layer.reactivation.b[0].item())
        # Values must have changed
        assert m_after != pytest.approx(m_before, rel=1e-6)
        assert b_after != pytest.approx(b_before, rel=1e-6)
        # And should match what the helper reports
        assert m_after == pytest.approx(stats["m"], rel=1e-6)
        assert b_after == pytest.approx(stats["b"], rel=1e-6)
        # m = k / sigma ; sanity: value is positive and finite
        assert torch.isfinite(torch.tensor(m_after)).item()
        assert m_after > 0

    def test_shunting_vs_additive_parity_of_calibration(self):
        """Direct A/B: same seed, same architecture — assert that the calibrated
        (m, b) for shunting vs additive differ in a *predictable* way.

        This is the test the diagnosis is built around: the empirical
        calibration of a shunting layer produces a substantially steeper
        m than the additive counterpart, because shunting V is bounded in
        ``[0, 1)`` and therefore has a smaller natural MAD. If this
        invariant ever flips, either the calibration formula has changed
        or the morphology effect we care about has disappeared.
        """
        torch.manual_seed(0)

        def _build(use_shunting: bool) -> DendriticBranchLayer:
            torch.manual_seed(0)
            return DendriticBranchLayer(
                output_dim=32,
                excitatory_input_dim=50,
                excitatory_synapses_per_branch=10,
                inhibitory_input_dim=50,
                inhibitory_synapses_per_branch=5,
                use_shunting=use_shunting,
                reactivate=True,
                reactivation_type="param_tanh",
                reactivation_init_policy="analytical",
                reactivation_sigma_aware_k=0.25,
                dbl_init_method="analytical_expectation",
                weight_transform="softplus",
            )

        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x, inh):
                return self.inner(x, inh)

        layer_sh = _build(use_shunting=True)
        layer_ad = _build(use_shunting=False)

        # Use the SAME batches for both so the calibration is purely
        # a function of the downstream network (morphology), not the
        # random data draw.
        torch.manual_seed(42)
        batches = [(torch.rand(128, 50), torch.rand(128, 50)) for _ in range(2)]

        diag_sh = calibrate_reactivation_from_data(Wrapper(layer_sh), batches, k=0.25)
        diag_ad = calibrate_reactivation_from_data(Wrapper(layer_ad), batches, k=0.25)

        assert len(diag_sh) == 1 and len(diag_ad) == 1
        sh = next(iter(diag_sh.values()))
        ad = next(iter(diag_ad.values()))

        # Shunting V lives in [0, 1] by construction; its robust scale is
        # bounded above by ~0.5 and in practice much smaller. Additive V
        # is the sum of (K_E + K_I) signed contributions and has
        # substantially larger spread.
        assert (
            sh["V_mad_std"] < 0.5
        ), f"shunting mad_std unexpectedly large: {sh['V_mad_std']}"
        assert ad["V_mad_std"] > sh["V_mad_std"], (
            f"expected additive mad_std > shunting mad_std, "
            f"got additive={ad['V_mad_std']} shunting={sh['V_mad_std']}"
        )

        # With k/scale calibration and smaller shunting scale, the
        # calibrated shunting slope must be strictly larger than the
        # additive slope. This is the mechanism we want to track:
        # shunting gets a sharper gate than additive on identical input.
        assert sh["m"] > ad["m"], (
            f"expected shunting calibrated slope > additive, "
            f"got shunting m={sh['m']}, additive m={ad['m']}"
        )

    def test_calibrate_no_op_when_no_reactivation(self):
        """Layer with reactivate=False should be skipped silently."""
        torch.manual_seed(0)
        layer = DendriticBranchLayer(
            output_dim=8,
            excitatory_input_dim=20,
            excitatory_synapses_per_branch=5,
            use_shunting=True,
            reactivate=False,
        )

        class Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(x)

        model = Wrapper(layer)
        batches = [torch.randn(32, 20)]
        diag = calibrate_reactivation_from_data(model, batches)
        assert diag == {}

    def test_calibrate_mode_none_respects_layer_policy(self):
        """mode=None should calibrate only data-driven-policy layers."""
        torch.manual_seed(0)
        layer_quantile = DendriticBranchLayer(
            output_dim=8,
            excitatory_input_dim=20,
            excitatory_synapses_per_branch=5,
            use_shunting=True,
            reactivate=True,
            reactivation_type="param_tanh",
            reactivation_init_policy="quantile",
        )
        layer_auto = DendriticBranchLayer(
            output_dim=8,
            excitatory_input_dim=8,
            excitatory_synapses_per_branch=2,
            use_shunting=True,
            reactivate=True,
            reactivation_type="param_tanh",
            reactivation_init_policy="analytical",
        )

        m_before = float(layer_auto.reactivation.log_m.exp()[0].item())
        b_before = float(layer_auto.reactivation.b[0].item())

        class Wrapper(torch.nn.Module):
            def __init__(self, layer1, layer2):
                super().__init__()
                self.layer1 = layer1
                self.layer2 = layer2

            def forward(self, x):
                x = self.layer1(x)
                return self.layer2(x)

        model = Wrapper(layer_quantile, layer_auto)
        batches = [torch.rand(64, 20) for _ in range(2)]
        diag = calibrate_reactivation_from_data(model, batches, mode=None)

        assert "layer1" in diag
        assert "layer2" not in diag
        assert float(layer_auto.reactivation.log_m.exp()[0].item()) == pytest.approx(
            m_before, rel=1e-6
        )
        assert float(layer_auto.reactivation.b[0].item()) == pytest.approx(
            b_before, rel=1e-6
        )

    def test_maybe_calibrate_uses_per_layer_mode_when_top_level_missing(self):
        """Launcher helper should preserve per-layer policy boundaries."""

        class DummyCore:
            pass

        class DummyModelCfg:
            core = DummyCore()

        class DummyConfig:
            model = DummyModelCfg()

        class PolicyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.reactivation_init_policy = "quantile"
                self.weight = torch.nn.Parameter(torch.zeros(1))

        model = PolicyModule()
        dataset = torch.utils.data.TensorDataset(torch.randn(8, 4), torch.zeros(8))

        with patch(
            "dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize.calibrate_reactivation_from_data"
        ) as mock_calibrate:
            mock_calibrate.return_value = {"layer": {"m": 1.0, "b": 0.0, "V_std": 1.0}}
            _maybe_calibrate_data_driven_reactivation(
                config=DummyConfig(),
                model=model,
                train_dataset=dataset,
                is_main=False,
                n_batches=1,
                batch_size=4,
                n_iterations=1,
            )

        mock_calibrate.assert_called_once()
        assert mock_calibrate.call_args.kwargs["mode"] is None

    def test_maybe_calibrate_ignores_stale_config_when_layer_policies_exist(self):
        """Built module policies should override stale YAML-level init policy."""

        class DummyReactivation:
            init_policy = "quantile"

        class DummyCore:
            reactivation = DummyReactivation()

        class DummyModelCfg:
            core = DummyCore()

        class DummyConfig:
            model = DummyModelCfg()

        class FixedPolicyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.reactivation_init_policy = "fixed"
                self.weight = torch.nn.Parameter(torch.zeros(1))

        model = FixedPolicyModule()
        dataset = torch.utils.data.TensorDataset(torch.randn(8, 4), torch.zeros(8))

        with patch(
            "dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize.calibrate_reactivation_from_data"
        ) as mock_calibrate:
            _maybe_calibrate_data_driven_reactivation(
                config=DummyConfig(),
                model=model,
                train_dataset=dataset,
                is_main=False,
                n_batches=1,
                batch_size=4,
                n_iterations=1,
            )

        mock_calibrate.assert_not_called()

    def test_maybe_calibrate_preserves_recurrent_sequence_batches_and_seq_lengths(self):
        """Recurrent calibration should keep [B, T, D] inputs and pass seq_lengths."""

        class DummyCore:
            pass

        class DummyModelCfg:
            core = DummyCore()

        class DummyConfig:
            model = DummyModelCfg()

        class RecurrentPolicyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.reactivation_init_policy = "quantile"
                self._is_recurrent_core = True
                self.weight = torch.nn.Parameter(torch.zeros(1))

        model = RecurrentPolicyModule()
        x = torch.randn(8, 5, 4)
        y = torch.zeros(8, dtype=torch.long)
        seq_lengths = torch.tensor([5, 4, 5, 3, 5, 4, 2, 5])
        dataset = torch.utils.data.TensorDataset(x, y, seq_lengths)

        with patch(
            "dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize.calibrate_reactivation_from_data"
        ) as mock_calibrate:
            mock_calibrate.return_value = {"layer": {"m": 1.0, "b": 0.0, "V_std": 1.0}}
            _maybe_calibrate_data_driven_reactivation(
                config=DummyConfig(),
                model=model,
                train_dataset=dataset,
                is_main=False,
                n_batches=1,
                batch_size=4,
                n_iterations=1,
            )

        mock_calibrate.assert_called_once()
        batches = mock_calibrate.call_args.args[1]
        assert len(batches) == 1
        assert isinstance(batches[0], tuple)
        batch_x, batch_seq_lengths = batches[0]
        assert batch_x.ndim == 3
        assert batch_x.shape[1:] == (5, 4)
        assert batch_seq_lengths.shape == (4,)
        assert set(batch_seq_lengths.cpu().tolist()).issubset(set(seq_lengths.tolist()))
        assert mock_calibrate.call_args.kwargs["quantile_aggregation"] == "chunk_median"


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
        x = torch.rand(4, 20)  # non-negative (shunting + positive weights)
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
        x = torch.rand(4, 20)  # non-negative (shunting + positive weights)
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
        x = torch.rand(2, 20)  # non-negative for shunting
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
        x = torch.rand(2, 20)  # non-negative for shunting
        inh = torch.rand(2, 10)  # non-negative for shunting
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
        x = torch.rand(2, 20)  # non-negative for shunting
        inh = torch.rand(2, 10)  # non-negative for shunting
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


# ---------------------------------------------------------------------------
# Quantile-mode empirical calibration
# ---------------------------------------------------------------------------


def _build_calibration_fixture(use_shunting=True, seed=0, **layer_kwargs):
    """Build a single-layer model + batches for calibration tests."""
    torch.manual_seed(seed)
    base_kwargs = {
        "output_dim": 16,
        "excitatory_input_dim": 50,
        "excitatory_synapses_per_branch": 10,
        "inhibitory_input_dim": 50,
        "inhibitory_synapses_per_branch": 5,
        "use_shunting": use_shunting,
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "reactivation_init_policy": "analytical",
        "reactivation_sigma_aware_k": 0.25,
        "dbl_init_method": "analytical_expectation",
        "weight_transform": "softplus",
    }
    base_kwargs.update(layer_kwargs)
    layer = DendriticBranchLayer(**base_kwargs)

    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x, inh):
            return self.inner(x, inh)

    model = Wrapper(layer)
    torch.manual_seed(42)
    batches = [(torch.rand(128, 50), torch.rand(128, 50)) for _ in range(3)]
    return model, layer, batches


class TestQuantileModeCalibration:
    def test_quantile_mode_sets_m_from_interquartile_range(self):
        """m should equal atanh(0.8) / ((q90 - q10) / 2) for the quantile mode."""
        from math import atanh

        model, layer, batches = _build_calibration_fixture()
        diag = calibrate_reactivation_from_data(model, batches, k=0.25, mode="quantile")

        assert len(diag) == 1
        stats = next(iter(diag.values()))

        q10 = stats["V_q10"]
        q90 = stats["V_q90"]
        half_width = (q90 - q10) / 2.0
        expected_m = atanh(0.8) / half_width

        m_actual = float(layer.reactivation.log_m.exp()[0].item())
        assert m_actual == pytest.approx(expected_m, rel=1e-5)
        assert m_actual == pytest.approx(stats["m"], rel=1e-6)

    def test_quantile_mode_b_equals_midpoint_of_q10_q90(self):
        """b should equal (q10 + q90) / 2."""
        model, layer, batches = _build_calibration_fixture()
        diag = calibrate_reactivation_from_data(model, batches, k=0.25, mode="quantile")

        stats = next(iter(diag.values()))
        expected_b = (stats["V_q10"] + stats["V_q90"]) / 2.0

        b_actual = float(layer.reactivation.b[0].item())
        assert b_actual == pytest.approx(expected_b, rel=1e-5)
        assert b_actual == pytest.approx(stats["b"], rel=1e-6)

    def test_quantile_mode_diagnostics_include_quantiles(self):
        """Diagnostics dict must contain V_q10, V_q50, V_q90 keys."""
        model, _layer, batches = _build_calibration_fixture()
        diag = calibrate_reactivation_from_data(model, batches, mode="quantile")

        assert len(diag) == 1
        stats = next(iter(diag.values()))

        for key in ("V_q10", "V_q50", "V_q90"):
            assert key in stats, f"missing diagnostics key: {key}"
            assert isinstance(stats[key], float)

        # q10 <= q50 <= q90 by construction
        assert stats["V_q10"] <= stats["V_q50"] <= stats["V_q90"]
        # mode tag should be recorded
        assert stats["mode"] == "occupancy_quantile"

    def test_quantile_mode_ema_alpha_blends_calibration_with_current_params(self):
        """EMA-style refresh should blend the calibrated target with the current gate state."""
        model, layer, batches = _build_calibration_fixture(seed=11)
        initial_m = float(layer.reactivation.log_m.exp()[0].item())
        initial_b = float(layer.reactivation.b[0].item())

        diag = calibrate_reactivation_from_data(
            model,
            batches,
            mode="quantile",
            ema_alpha=0.25,
        )
        stats = next(iter(diag.values()))

        expected_m = (1.0 - 0.25) * initial_m + 0.25 * stats["m"]
        expected_b = (1.0 - 0.25) * initial_b + 0.25 * stats["b"]

        assert stats["ema_alpha"] == pytest.approx(0.25, rel=1e-8)
        assert stats["m_applied"] == pytest.approx(expected_m, rel=1e-5)
        assert stats["b_applied"] == pytest.approx(expected_b, rel=1e-5)
        assert float(layer.reactivation.log_m.exp()[0].item()) == pytest.approx(
            expected_m, rel=1e-5
        )
        assert float(layer.reactivation.b[0].item()) == pytest.approx(
            expected_b, rel=1e-5
        )

    def test_occupancy_quantile_clamps_extreme_slopes(self, monkeypatch):
        """Tiny quantile spans should trigger the slope clamp instead of exploding."""
        model, layer, batches = _build_calibration_fixture(
            seed=13,
            reactivation_init_policy="occupancy_quantile",
            reactivation_calibration_max_m=12.0,
            reactivation_calibration_min_quantile_width=1e-3,
            reactivation_calibration_revert_on_invalid=False,
        )
        original_aggregate = init_mod._aggregate_quantiles_from_chunks

        def fake_aggregate(chunks, probs, aggregation="global"):
            if probs.numel() == 2:
                return torch.tensor([1.0, 1.0], dtype=torch.double), 2
            return original_aggregate(chunks, probs, aggregation=aggregation)

        monkeypatch.setattr(
            init_mod, "_aggregate_quantiles_from_chunks", fake_aggregate
        )

        with pytest.warns(RuntimeWarning, match="tiny quantile span"):
            diag = calibrate_reactivation_from_data(model, batches, mode=None)

        stats = next(iter(diag.values()))
        assert stats["occupancy_m_was_clamped"] is True
        assert stats["calibration_reverted"] is False
        assert stats["occupancy_delta_q_raw"] == pytest.approx(0.0, abs=1e-12)
        assert stats["occupancy_delta_q_used"] == pytest.approx(1e-3, rel=1e-8)
        assert stats["m"] == pytest.approx(12.0, rel=1e-8)
        assert float(layer.reactivation.log_m.exp()[0].item()) == pytest.approx(
            12.0, rel=1e-8
        )

    def test_occupancy_quantile_reverts_on_invalid_candidate(self, monkeypatch):
        """Non-finite occupancy fits should keep the previous gate state."""
        model, layer, batches = _build_calibration_fixture(
            seed=17,
            reactivation_init_policy="occupancy_quantile",
            reactivation_calibration_revert_on_invalid=True,
        )
        initial_m = float(layer.reactivation.log_m.exp()[0].item())
        initial_b = float(layer.reactivation.b[0].item())
        original_aggregate = init_mod._aggregate_quantiles_from_chunks

        def fake_aggregate(chunks, probs, aggregation="global"):
            if probs.numel() == 2:
                return torch.tensor([float("nan"), float("nan")], dtype=torch.double), 2
            return original_aggregate(chunks, probs, aggregation=aggregation)

        monkeypatch.setattr(
            init_mod, "_aggregate_quantiles_from_chunks", fake_aggregate
        )

        with pytest.warns(RuntimeWarning, match="non-finite quantiles"):
            diag = calibrate_reactivation_from_data(model, batches, mode=None)

        stats = next(iter(diag.values()))
        assert stats["calibration_reverted"] is True
        assert stats["m"] == pytest.approx(initial_m, rel=1e-6)
        assert stats["b"] == pytest.approx(initial_b, rel=1e-6)
        assert stats["m_applied"] == pytest.approx(initial_m, rel=1e-6)
        assert stats["b_applied"] == pytest.approx(initial_b, rel=1e-6)
        assert float(layer.reactivation.log_m.exp()[0].item()) == pytest.approx(
            initial_m, rel=1e-6
        )
        assert float(layer.reactivation.b[0].item()) == pytest.approx(
            initial_b, rel=1e-6
        )

    def test_occupancy_quantile_respects_custom_quantiles_and_targets(self):
        """General occupancy_quantile should fit the configured quantile/target pairs."""
        from math import atanh

        model, layer, batches = _build_calibration_fixture(
            reactivation_init_policy="occupancy_quantile",
            reactivation_occupancy_quantile_low=0.2,
            reactivation_occupancy_quantile_high=0.8,
            reactivation_occupancy_target_low=0.25,
            reactivation_occupancy_target_high=0.75,
        )
        diag = calibrate_reactivation_from_data(model, batches, mode=None)
        stats = next(iter(diag.values()))

        # Recover the fitted quantiles directly from the collected voltages.
        with torch.no_grad():
            samples = []
            was_training = model.training
            model.eval()
            try:
                handle = layer.reactivation.register_forward_pre_hook(
                    lambda _mod, inputs: samples.append(inputs[0].detach().reshape(-1))
                )
                try:
                    for x, inh in batches:
                        _ = model(x, inh)
                finally:
                    handle.remove()
            finally:
                if was_training:
                    model.train()

        v = torch.cat(samples).double()
        q20, q80 = [
            float(x.item())
            for x in torch.quantile(v, torch.tensor([0.2, 0.8], dtype=v.dtype))
        ]
        z25 = atanh(2 * 0.25 - 1.0)
        z75 = atanh(2 * 0.75 - 1.0)
        expected_m = (z75 - z25) / max(q80 - q20, 1e-12)
        expected_b = q20 - z25 / expected_m

        assert stats["occupancy_q_low"] == pytest.approx(0.2, rel=1e-8)
        assert stats["occupancy_q_high"] == pytest.approx(0.8, rel=1e-8)
        assert stats["occupancy_r_low"] == pytest.approx(0.25, rel=1e-8)
        assert stats["occupancy_r_high"] == pytest.approx(0.75, rel=1e-8)
        assert stats["m"] == pytest.approx(expected_m, rel=1e-5)
        assert stats["b"] == pytest.approx(expected_b, rel=1e-5)
        assert float(layer.reactivation.log_m.exp()[0].item()) == pytest.approx(
            expected_m, rel=1e-5
        )
        assert float(layer.reactivation.b[0].item()) == pytest.approx(
            expected_b, rel=1e-5
        )

    def test_quantile_mode_subsamples_large_voltage_tensors(self, monkeypatch):
        """Large flattened voltage tensors should use a capped quantile sample."""
        model, _layer, batches = _build_calibration_fixture(seed=9)
        monkeypatch.setattr(init_mod, "MAX_CALIBRATION_QUANTILE_SAMPLES", 64)

        diag = calibrate_reactivation_from_data(model, batches, mode="quantile")
        stats = next(iter(diag.values()))

        assert stats["n"] > 64
        assert stats["n_quantile_sample"] == 64
        assert stats["V_q10"] <= stats["V_q50"] <= stats["V_q90"]

    def test_chunk_median_quantile_aggregation_is_temporally_robust(self):
        """Chunk-median aggregation should fit typical slices, not pooled extremes."""
        chunks = [
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            torch.tensor([10.0, 10.0, 11.0, 11.0]),
            torch.tensor([20.0, 20.0, 21.0, 21.0]),
        ]
        probs = torch.tensor([0.1, 0.5, 0.9], dtype=torch.double)

        pooled_q, _ = _aggregate_quantiles_from_chunks(
            chunks, probs, aggregation="global"
        )
        chunk_q, _ = _aggregate_quantiles_from_chunks(
            chunks, probs, aggregation="chunk_median"
        )

        # Global pooling spans the full multimodal temporal distribution,
        # while chunk-median tracks the typical chunk (the middle slice here).
        assert float(pooled_q[1]) == pytest.approx(10.5, rel=1e-6)
        assert float(chunk_q[1]) == pytest.approx(10.5, rel=1e-6)
        assert float(pooled_q[0]) < 2.0
        assert float(pooled_q[2]) > 19.0
        assert 9.5 < float(chunk_q[0]) < 10.5
        assert 10.5 < float(chunk_q[2]) < 11.5

    def test_quantile_vs_empirical_different_m_values(self):
        """Quantile and median_mad modes must produce different (m, b) on the
        same model and data — they use fundamentally different formulas."""
        model_q, _layer_q, batches = _build_calibration_fixture(seed=7)
        model_m, _layer_m, _ = _build_calibration_fixture(seed=7)

        diag_q = calibrate_reactivation_from_data(
            model_q, batches, k=0.25, mode="quantile"
        )
        diag_m = calibrate_reactivation_from_data(
            model_m, batches, k=0.25, mode="median_mad"
        )

        stats_q = next(iter(diag_q.values()))
        stats_m = next(iter(diag_m.values()))

        # The slopes should differ because the formulas are different
        assert stats_q["m"] != pytest.approx(
            stats_m["m"], rel=1e-3
        ), f"quantile m={stats_q['m']} should differ from median_mad m={stats_m['m']}"
        # The biases may also differ (median vs midpoint of q10/q90)
        assert stats_q["b"] != pytest.approx(
            stats_m["b"], rel=1e-3
        ), f"quantile b={stats_q['b']} should differ from median_mad b={stats_m['b']}"

    def test_iterative_calibration_converges(self):
        """Calling calibrate 3x sequentially should stabilize m: the change
        between iteration 2->3 should be smaller than the change 1->2."""
        model, _layer, batches = _build_calibration_fixture()

        m_values = []
        for _ in range(3):
            diag = calibrate_reactivation_from_data(
                model, batches, k=0.25, mode="quantile"
            )
            stats = next(iter(diag.values()))
            m_values.append(stats["m"])

        delta_12 = abs(m_values[1] - m_values[0])
        delta_23 = abs(m_values[2] - m_values[1])

        # Later iterations should show smaller (or equal) changes
        assert (
            delta_23 <= delta_12 + 1e-8
        ), f"calibration did not converge: delta_12={delta_12}, delta_23={delta_23}"
