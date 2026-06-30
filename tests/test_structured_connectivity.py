"""Tests for hard structured synaptic connectivity masks."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    DeepstLinear,
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
    StochasticTopKLinear,
    TopKLinear,
    distance_kernel_probabilities,
    sample_fixed_indegree_mask,
    sample_probability_mask,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent.structured_einet_factory import (
    _build_structured_inhibitory_population_config,
    _effective_layer_recurrence_enabled,
    _LayerPopulationSynapseCounts,
    _RecurrentFeatureFlags,
    _RecurrentReactivationOptions,
    _RecurrentSynapseCounts,
    _resolve_layer_excitatory_synapse_counts,
    _resolve_layer_inhibitory_synapse_counts,
    _resolve_layer_recurrent_timing,
    _resolve_recurrent_input_projection_dims,
    _resolve_recurrent_store_routing,
    _resolve_recurrent_transfer_params,
    _should_build_inhibitory_population,
    _structured_population_adaptive_init_kwargs,
    _structured_population_blocklinear_kwargs,
    _structured_population_deepst_kwargs,
    _structured_population_dendritic_spike_kwargs,
    _structured_population_dynamics_kwargs,
    _structured_population_morphology_kwargs,
    _structured_population_reactivation_kwargs,
    _structured_population_soma_feedback_kwargs,
    _structured_population_timing_kwargs,
)


@dataclass(frozen=True)
class _TinyPopulationConfig:
    n_neurons: int
    branch_factors: list[int]
    ff_excitatory_synapses: int
    ff_inhibitory_synapses: int
    rec_excitatory_synapses: int
    rec_inhibitory_synapses: int
    structured_layer_idx: int
    ff_excitatory_pathway: str
    ff_inhibitory_pathway: str
    rec_excitatory_pathway: str
    rec_inhibitory_pathway: str
    untouched: str = "preserved"


def test_topk_connection_mask_limits_active_synapses():
    connection_mask = torch.tensor(
        [
            [True, False, True, False, False],
            [False, True, False, True, False],
            [False, False, False, True, True],
        ]
    )
    layer = TopKLinear(
        in_features=5,
        out_features=3,
        K=5,
        connection_mask=connection_mask,
    )
    layer.initialize()

    active = layer.weight_mask().bool()

    assert torch.equal(active, connection_mask)
    assert not bool((active & ~connection_mask).any())


def test_topk_connection_mask_blocks_disallowed_gradients():
    connection_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    layer = TopKLinear(
        in_features=4,
        out_features=2,
        K=4,
        connection_mask=connection_mask,
    )
    layer.initialize()

    x = torch.randn(3, 4)
    layer(x).sum().backward()

    assert layer.pre_w.grad is not None
    assert torch.count_nonzero(layer.pre_w.grad[~connection_mask]) == 0
    assert torch.count_nonzero(layer.pre_w.grad[connection_mask]) > 0


def test_topk_unmasked_state_dict_stays_legacy_compatible():
    unmasked = TopKLinear(in_features=4, out_features=2, K=2)
    masked = TopKLinear(
        in_features=4,
        out_features=2,
        K=2,
        connection_mask=torch.ones(2, 4, dtype=torch.bool),
    )

    assert "connection_mask" not in unmasked.state_dict()
    assert "connection_mask" in masked.state_dict()


def test_stochastic_topk_connection_mask_is_enforced_in_both_modes():
    connection_mask = torch.tensor(
        [
            [True, False, True, False, False],
            [False, True, False, True, False],
        ]
    )
    for ultrafast in (False, True):
        layer = StochasticTopKLinear(
            in_features=5,
            out_features=2,
            K=5,
            connection_mask=connection_mask,
            ultrafast=ultrafast,
            temperature=0.5,
        )
        layer.initialize()

        active = layer.weight_mask().bool()

        assert not bool((active & ~connection_mask).any())
        assert bool(active.any())


def test_connection_mask_rejects_empty_output_rows():
    connection_mask = torch.tensor([[True, False], [False, False]])

    with pytest.raises(ValueError, match="at least one input"):
        TopKLinear(
            in_features=2,
            out_features=2,
            K=1,
            connection_mask=connection_mask,
        )


def test_deepst_initializes_only_allowed_connections():
    connection_mask = torch.tensor(
        [
            [True, False, True, False],
            [False, True, False, True],
        ]
    )
    layer = DeepstLinear(
        in_features=4,
        out_features=2,
        target_density=1.0,
        connection_mask=connection_mask,
    )

    active = layer.weight_mask().bool()
    stats = layer.get_connectivity_stats()

    assert torch.equal(active, connection_mask)
    assert stats["allowed_connections"] == int(connection_mask.sum())
    assert stats["allowed_density"] == pytest.approx(1.0)


def test_deepst_linear_decays_active_and_boosts_inactive_allowed_weights():
    layer = DeepstLinear(
        in_features=4,
        out_features=2,
        target_density=0.0,
        weight_transform="exp",
        weight_init_range=(0.0, 0.0),
        connection_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    with torch.no_grad():
        layer.mask.zero_()
        layer.mask[0, 0] = True
        layer.mask[1, 2] = True

    before = layer.weight().clone()
    active = layer.weight_mask().bool()
    inactive = ~active

    layer.decay_weights(weight_decay=0.25, weight_boosting=True)
    after = layer.weight()

    assert torch.allclose(after[active], before[active] * 0.75)
    assert torch.allclose(after[inactive], before[inactive] * 1.25)


def test_structured_population_deepst_kwargs_preserve_all_rewiring_fields():
    kwargs = _structured_population_deepst_kwargs(
        {
            "excitatory_target_density": 0.3,
            "inhibitory_target_density": 0.4,
            "use_noise": False,
            "sigma": 0.07,
            "rewiring_mode": "constant_branch",
            "rewire_frequency": 5,
            "synapses_per_branch": 6,
            "freeze_excitatory_connectivity": True,
            "freeze_inhibitory_connectivity": True,
            "init_method": "kaiming_uniform",
            "weight_threshold": 1e-4,
        }
    )

    assert kwargs == {
        "excitatory_target_density": 0.3,
        "inhibitory_target_density": 0.4,
        "use_noise": False,
        "sigma": 0.07,
        "rewiring_mode": "constant_branch",
        "rewire_frequency": 5,
        "synapses_per_branch": 6,
        "freeze_excitatory_connectivity": True,
        "freeze_inhibitory_connectivity": True,
        "init_method": "kaiming_uniform",
        "weight_threshold": 1e-4,
    }


def test_structured_population_adaptive_init_kwargs_preserve_defaults_and_overrides():
    assert _structured_population_adaptive_init_kwargs({}) == {
        "adaptive_initialization": True,
        "adaptive_initialization_policy": "preserve_shunting_center",
        "adaptive_target_conductance": 5.0,
    }

    assert _structured_population_adaptive_init_kwargs(
        {
            "adaptive_initialization": True,
            "adaptive_initialization_policy": "preserve_branch_mass",
            "adaptive_target_conductance": 7,
        }
    ) == {
        "adaptive_initialization": True,
        "adaptive_initialization_policy": "preserve_branch_mass",
        "adaptive_target_conductance": 7.0,
    }


def test_structured_population_morphology_kwargs_preserve_defaults_and_overrides():
    assert _structured_population_morphology_kwargs(
        morphology={},
        use_shunting=True,
        use_additive_normalization=False,
    ) == {
        "weight_transform": "softplus",
        "use_shunting": True,
        "use_additive_normalization": False,
        "allow_self_recurrence": True,
        "somatic_synapses": False,
        "dbl_init_method": "analytical_expectation",
    }

    assert _structured_population_morphology_kwargs(
        morphology={
            "weight_transform": "exp",
            "allow_self_recurrence": False,
            "somatic_synapses": True,
            "dbl_init_method": "legacy",
        },
        use_shunting=False,
        use_additive_normalization=True,
    ) == {
        "weight_transform": "exp",
        "use_shunting": False,
        "use_additive_normalization": True,
        "allow_self_recurrence": False,
        "somatic_synapses": True,
        "dbl_init_method": "legacy",
    }


def test_structured_population_reactivation_kwargs_preserve_calibration_fields():
    kwargs = _structured_population_reactivation_kwargs(
        reactivation={
            "occupancy_quantile_low": 0.1,
            "occupancy_quantile_high": 0.9,
            "occupancy_target_low": 0.2,
            "occupancy_target_high": 0.8,
            "calibration_min_quantile_width": 0.02,
            "calibration_max_m": 11,
            "calibration_revert_on_invalid": False,
            "sigma_aware_k": 0.35,
            "gradient_scaling": "conductance_dynamic",
        },
        reactivation_options=_RecurrentReactivationOptions(
            reactivate=True,
            reactivation_type="param_tanh",
            init_m=1.25,
            init_b=0.75,
            init_policy="analytical",
        ),
    )

    assert kwargs == {
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "reactivation_init_m": 1.25,
        "reactivation_init_b": 0.75,
        "reactivation_init_policy": "analytical",
        "reactivation_occupancy_quantile_low": 0.1,
        "reactivation_occupancy_quantile_high": 0.9,
        "reactivation_occupancy_target_low": 0.2,
        "reactivation_occupancy_target_high": 0.8,
        "reactivation_calibration_min_quantile_width": 0.02,
        "reactivation_calibration_max_m": 11.0,
        "reactivation_calibration_revert_on_invalid": False,
        "reactivation_sigma_aware_k": 0.35,
        "reactivation_strategy": "conductance_dynamic",
    }


def test_structured_population_blocklinear_kwargs_preserve_defaults_and_overrides():
    assert _structured_population_blocklinear_kwargs(
        blocklinear={},
        implementation={},
    ) == {
        "blocklinear_strategy": "none",
        "efficient_blocklinear": False,
        "print_hooks": False,
    }

    assert _structured_population_blocklinear_kwargs(
        blocklinear={"gradient_scaling": "distal_upweight_by_idx", "efficient": True},
        implementation={"print_hooks": True},
    ) == {
        "blocklinear_strategy": "distal_upweight_by_idx",
        "efficient_blocklinear": True,
        "print_hooks": True,
    }


def test_structured_population_timing_kwargs_preserve_defaults_and_overrides():
    assert _structured_population_timing_kwargs(
        recurrent_cfg={},
        tau_base=50.0,
        tau_ratio=3.0,
    ) == {
        "level_taus": [],
        "tau_base": 50.0,
        "tau_ratio": 3.0,
        "learnable_tau": False,
        "epsilon": 1e-8,
    }

    assert _structured_population_timing_kwargs(
        recurrent_cfg={
            "level_taus": (1, 2, 3),
            "learnable_tau": True,
            "epsilon": "1e-6",
        },
        tau_base=25,
        tau_ratio=2,
    ) == {
        "level_taus": [1, 2, 3],
        "tau_base": 25,
        "tau_ratio": 2,
        "learnable_tau": True,
        "epsilon": 1e-6,
    }


def test_layer_recurrent_timing_preserves_repeat_last_semantics():
    timing = _resolve_layer_recurrent_timing(
        recurrent_cfg={
            "dt": [0.5],
            "tau_base": [10, 20],
            "tau_ratio": 4,
        },
        layer_idx=3,
    )

    assert timing.dt == 0.5
    assert timing.tau_base == 20.0
    assert timing.tau_ratio == 4.0


def test_recurrent_transfer_params_preserve_override_order():
    params = _resolve_recurrent_transfer_params(
        transfer={
            "input_mode": 0,
            "activation": "relu",
            "hidden_dims": [8],
        },
        recurrent_cfg={
            "transfer": {"activation": "identity", "hidden_dims": [16]},
            "transfer_params": {"hidden_dims": [32], "output_activation": "relu"},
        },
    )

    assert params["input_mode"] == 0
    assert params["activation"] == "identity"
    assert params["hidden_dims"] == [32]
    assert params["output_activation"] == "relu"


def test_recurrent_input_projection_dims_preserve_list_only_legacy_behavior():
    assert _resolve_recurrent_input_projection_dims(
        architecture={"input_projection_dims": [4, 5]},
        recurrent_cfg={"input_projection_dims": [1, 2]},
    ) == [4, 5]

    assert _resolve_recurrent_input_projection_dims(
        architecture={},
        recurrent_cfg={"input_projection_dims": [1, 2]},
    ) == [1, 2]

    assert (
        _resolve_recurrent_input_projection_dims(
            architecture={"input_projection_dims": (4, 5)},
            recurrent_cfg={"input_projection_dims": [1, 2]},
        )
        == []
    )


def test_recurrent_store_routing_prefers_implementation_section():
    assert (
        _resolve_recurrent_store_routing(
            implementation={"store_routing": False},
            recurrent_cfg={"store_routing": True},
        )
        is False
    )
    assert (
        _resolve_recurrent_store_routing(
            implementation={},
            recurrent_cfg={"store_routing": True},
        )
        is True
    )


def test_structured_population_dynamics_kwargs_preserve_modern_and_legacy_keys():
    assert _structured_population_dynamics_kwargs(
        {
            "mode": "spike",
            "threshold": 0.8,
            "reset": -0.2,
            "tau": 11,
            "refractory_steps": 3,
            "surrogate_beta": 4.5,
            "readout": "voltage",
            "readout_tau": 12,
        }
    ) == {
        "dynamics_mode": "spike",
        "spike_threshold": 0.8,
        "spike_reset": -0.2,
        "spike_tau": 11.0,
        "spike_refractory_steps": 3,
        "spike_surrogate_beta": 4.5,
        "spike_readout": "voltage",
        "spike_readout_tau": 12.0,
    }

    assert _structured_population_dynamics_kwargs(
        {
            "dynamics_mode": "legacy_spike",
            "spike_threshold": 0.6,
            "spike_reset": -0.1,
            "spike_tau": 9,
            "spike_refractory_steps": 2,
            "spike_surrogate_beta": 5.5,
            "spike_readout": "rates",
            "spike_readout_tau": 13,
        }
    ) == {
        "dynamics_mode": "legacy_spike",
        "spike_threshold": 0.6,
        "spike_reset": -0.1,
        "spike_tau": 9.0,
        "spike_refractory_steps": 2,
        "spike_surrogate_beta": 5.5,
        "spike_readout": "rates",
        "spike_readout_tau": 13.0,
    }


def test_structured_population_dendritic_spike_kwargs_preserve_precedence():
    kwargs = _structured_population_dendritic_spike_kwargs(
        dendritic_spikes={
            "enabled": True,
            "mode": "burst",
            "levels": [1, 2],
            "threshold": 0.4,
            "plateau_amplitude": 0.9,
            "plateau_tau": 8,
            "refractory_steps": 2,
            "surrogate_beta": 3.5,
            "propagation": "gated",
        },
        dynamics={
            "dendritic_spikes_enabled": False,
            "dendritic_spike_mode": "legacy",
            "dendritic_spike_levels": "all",
            "dendritic_spike_threshold": 0.7,
            "dendritic_spike_plateau_amplitude": 1.0,
            "dendritic_spike_plateau_tau": 20.0,
            "dendritic_spike_refractory_steps": 0,
            "dendritic_spike_surrogate_beta": 10.0,
            "dendritic_spike_propagation": "additive",
        },
    )

    assert kwargs == {
        "dendritic_spikes_enabled": True,
        "dendritic_spike_mode": "burst",
        "dendritic_spike_levels": [1, 2],
        "dendritic_spike_threshold": 0.4,
        "dendritic_spike_plateau_amplitude": 0.9,
        "dendritic_spike_plateau_tau": 8.0,
        "dendritic_spike_refractory_steps": 2,
        "dendritic_spike_surrogate_beta": 3.5,
        "dendritic_spike_propagation": "gated",
    }

    assert (
        _structured_population_dendritic_spike_kwargs(
            dendritic_spikes={},
            dynamics={"dendritic_spike_mode": "legacy"},
        )["dendritic_spike_mode"]
        == "legacy"
    )


def test_structured_population_soma_feedback_kwargs_preserve_precedence():
    kwargs = _structured_population_soma_feedback_kwargs(
        soma_feedback={
            "enabled": True,
            "mode": "multiplicative",
            "source": "state",
            "levels": [0],
            "strength": 0.25,
            "learnable_strength": True,
            "per_level": False,
            "init_std": 0.01,
            "reversal": 1.5,
        },
        dynamics={
            "soma_feedback_enabled": False,
            "soma_feedback_mode": "legacy",
            "soma_feedback_source": "output",
            "soma_feedback_levels": "all",
            "soma_feedback_strength": 0.1,
            "soma_feedback_learnable_strength": False,
            "soma_feedback_per_level": True,
            "soma_feedback_init_std": 0.0,
            "soma_feedback_reversal": 1.0,
        },
    )

    assert kwargs == {
        "soma_feedback_enabled": True,
        "soma_feedback_mode": "multiplicative",
        "soma_feedback_source": "state",
        "soma_feedback_levels": [0],
        "soma_feedback_strength": 0.25,
        "soma_feedback_learnable_strength": True,
        "soma_feedback_per_level": False,
        "soma_feedback_init_std": 0.01,
        "soma_feedback_reversal": 1.5,
    }

    assert (
        _structured_population_soma_feedback_kwargs(
            soma_feedback={},
            dynamics={"soma_feedback_source": "legacy_output"},
        )["soma_feedback_source"]
        == "legacy_output"
    )


def test_layer_inhibitory_synapse_counts_preserve_repeat_last_semantics():
    synapses = _resolve_layer_inhibitory_synapse_counts(
        connectivity={
            "ei_synapses_per_branch_per_layer": [1],
            "ii_synapses_per_branch_per_layer": [0, 2],
        },
        counts=_RecurrentSynapseCounts(
            rec_ee=0,
            rec_ie=0,
            rec_ei=[3],
            rec_ii=4,
        ),
        layer_idx=2,
    )

    assert synapses.ff_excitatory == 1
    assert synapses.ff_inhibitory == 2
    assert synapses.rec_excitatory == 3
    assert synapses.rec_inhibitory == 4
    assert synapses.has_incoming_synapses


def test_layer_excitatory_synapse_counts_preserve_repeat_last_semantics():
    synapses = _resolve_layer_excitatory_synapse_counts(
        connectivity={
            "ee_synapses_per_branch_per_layer": [4],
            "ie_synapses_per_branch_per_layer": [0, 5],
        },
        counts=_RecurrentSynapseCounts(
            rec_ee=[6],
            rec_ie=7,
            rec_ei=0,
            rec_ii=0,
        ),
        layer_idx=2,
    )

    assert synapses.ff_excitatory == 4
    assert synapses.ff_inhibitory == 5
    assert synapses.rec_excitatory == 6
    assert synapses.rec_inhibitory == 7
    assert synapses.has_incoming_synapses


def test_inhibitory_population_requires_size_and_incoming_synapses():
    empty_synapses = _resolve_layer_inhibitory_synapse_counts(
        connectivity={},
        counts=_RecurrentSynapseCounts(rec_ee=0, rec_ie=0, rec_ei=0, rec_ii=0),
        layer_idx=0,
    )
    incoming_synapses = _resolve_layer_inhibitory_synapse_counts(
        connectivity={"ei_synapses_per_branch_per_layer": [1]},
        counts=_RecurrentSynapseCounts(rec_ee=0, rec_ie=0, rec_ei=0, rec_ii=0),
        layer_idx=0,
    )

    assert not _should_build_inhibitory_population(n_inh=5, synapses=empty_synapses)
    assert not _should_build_inhibitory_population(n_inh=0, synapses=incoming_synapses)
    assert _should_build_inhibitory_population(n_inh=5, synapses=incoming_synapses)


def test_build_structured_inhibitory_population_config_replaces_pathway_fields():
    exc_cfg = _TinyPopulationConfig(
        n_neurons=10,
        branch_factors=[2],
        ff_excitatory_synapses=11,
        ff_inhibitory_synapses=12,
        rec_excitatory_synapses=13,
        rec_inhibitory_synapses=14,
        structured_layer_idx=0,
        ff_excitatory_pathway="ee",
        ff_inhibitory_pathway="ie",
        rec_excitatory_pathway="rec_ee",
        rec_inhibitory_pathway="rec_ie",
    )

    inh_cfg = _build_structured_inhibitory_population_config(
        excitatory_config=exc_cfg,
        architecture={"inhibitory_branch_factors": [3, 1]},
        layer_idx=4,
        n_inh=7,
        synapses=_LayerPopulationSynapseCounts(
            ff_excitatory=1,
            ff_inhibitory=2,
            rec_excitatory=3,
            rec_inhibitory=4,
        ),
    )

    assert inh_cfg.n_neurons == 7
    assert inh_cfg.branch_factors == [3, 1]
    assert inh_cfg.ff_excitatory_synapses == 1
    assert inh_cfg.ff_inhibitory_synapses == 2
    assert inh_cfg.rec_excitatory_synapses == 3
    assert inh_cfg.rec_inhibitory_synapses == 4
    assert inh_cfg.structured_layer_idx == 4
    assert inh_cfg.ff_excitatory_pathway == "ei"
    assert inh_cfg.ff_inhibitory_pathway == "ii"
    assert inh_cfg.rec_excitatory_pathway == "rec_ei"
    assert inh_cfg.rec_inhibitory_pathway == "rec_ii"
    assert inh_cfg.untouched == "preserved"
    assert exc_cfg.ff_excitatory_pathway == "ee"


def test_effective_layer_recurrence_includes_stateful_feature_requirements():
    inactive_flags = _RecurrentFeatureFlags(
        has_rec_synapses=False,
        has_soma_feedback=False,
        has_spiking_dynamics=False,
        has_dendritic_spikes=False,
        has_synapse_types=False,
    )
    exc_inactive = SimpleNamespace(
        dendritic_spikes_enabled=False,
        soma_feedback_enabled=False,
    )
    inh_inactive = SimpleNamespace(
        dendritic_spikes_enabled=False,
        soma_feedback_enabled=False,
    )

    assert not _effective_layer_recurrence_enabled(
        base_recurrent=False,
        flags=inactive_flags,
        exc_cfg=exc_inactive,
        inh_cfg=None,
    )
    assert _effective_layer_recurrence_enabled(
        base_recurrent=True,
        flags=inactive_flags,
        exc_cfg=exc_inactive,
        inh_cfg=None,
    )
    assert _effective_layer_recurrence_enabled(
        base_recurrent=False,
        flags=_RecurrentFeatureFlags(
            has_rec_synapses=False,
            has_soma_feedback=False,
            has_spiking_dynamics=True,
            has_dendritic_spikes=False,
            has_synapse_types=False,
        ),
        exc_cfg=exc_inactive,
        inh_cfg=None,
    )
    assert _effective_layer_recurrence_enabled(
        base_recurrent=False,
        flags=inactive_flags,
        exc_cfg=SimpleNamespace(
            dendritic_spikes_enabled=True,
            soma_feedback_enabled=False,
        ),
        inh_cfg=None,
    )
    assert _effective_layer_recurrence_enabled(
        base_recurrent=False,
        flags=inactive_flags,
        exc_cfg=SimpleNamespace(
            dendritic_spikes_enabled=False,
            soma_feedback_enabled=True,
        ),
        inh_cfg=None,
    )
    assert _effective_layer_recurrence_enabled(
        base_recurrent=False,
        flags=_RecurrentFeatureFlags(
            has_rec_synapses=False,
            has_soma_feedback=False,
            has_spiking_dynamics=False,
            has_dendritic_spikes=False,
            has_synapse_types=True,
        ),
        exc_cfg=exc_inactive,
        inh_cfg=None,
    )
    assert _effective_layer_recurrence_enabled(
        base_recurrent=False,
        flags=inactive_flags,
        exc_cfg=exc_inactive,
        inh_cfg=SimpleNamespace(
            dendritic_spikes_enabled=False,
            soma_feedback_enabled=True,
        ),
    )
    assert not _effective_layer_recurrence_enabled(
        base_recurrent=False,
        flags=inactive_flags,
        exc_cfg=exc_inactive,
        inh_cfg=inh_inactive,
    )


@pytest.mark.parametrize(
    "layer",
    [
        TopKLinear(
            in_features=4,
            out_features=2,
            K=2,
            weight_transform="softplus",
        ),
        IndexedSparseLinear(
            in_features=4,
            out_features=2,
            K=2,
            weight_transform="softplus",
            seed=11,
        ),
        IndexedDynamicTopKLinear(
            in_features=4,
            out_features=2,
            K=2,
            candidate_size=3,
            weight_transform="softplus",
            seed=13,
        ),
        DeepstLinear(
            in_features=4,
            out_features=2,
            target_density=1.0,
            weight_transform="softplus",
            weight_init_range=(0.0, 0.0),
        ),
    ],
)
def test_softplus_decay_inverse_is_stable_for_large_weights(layer):
    with torch.no_grad():
        layer.pre_w.fill_(1000.0)

    layer.decay_weights(weight_decay=0.1, weight_boosting=True)

    assert torch.isfinite(layer.pre_w).all()


def test_indexed_sparse_linear_seed_none_follows_global_rng():
    torch.manual_seed(123)
    first = IndexedSparseLinear(in_features=7, out_features=5, K=3, seed=None)
    torch.manual_seed(123)
    second = IndexedSparseLinear(in_features=7, out_features=5, K=3, seed=None)
    torch.manual_seed(124)
    third = IndexedSparseLinear(in_features=7, out_features=5, K=3, seed=None)

    assert torch.equal(first.connection_indices, second.connection_indices)
    assert not torch.equal(first.connection_indices, third.connection_indices)


def test_indexed_dynamic_auto_candidate_size_respects_connection_mask():
    connection_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, False, False, False, False],
            [False, False, True, True, True, False],
        ]
    )

    layer = IndexedDynamicTopKLinear(
        in_features=6,
        out_features=3,
        K=2,
        candidate_size=None,
        connection_mask=connection_mask,
        seed=5,
    )

    row = torch.arange(layer.out_features)[:, None]
    assert layer.candidate_size == 2
    assert bool(connection_mask[row, layer.connection_indices].all())


def test_indexed_dynamic_explicit_candidate_size_remains_strict_with_mask():
    connection_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
        ]
    )

    with pytest.raises(ValueError, match="at least K allowed inputs"):
        IndexedDynamicTopKLinear(
            in_features=4,
            out_features=2,
            K=2,
            candidate_size=3,
            connection_mask=connection_mask,
            seed=5,
        )


def test_indexed_sparse_linear_forward_param_gradient_scale():
    layer = IndexedSparseLinear(
        in_features=4,
        out_features=3,
        K=2,
        seed=0,
        weight_transform="identity",
    )
    x = torch.ones(2, 4)
    out = layer(x).sum()
    layer.register_forward_param_gradient_scale(torch.zeros(3, 1))
    out.backward()

    assert layer.pre_w.grad is not None
    assert torch.allclose(layer.pre_w.grad, torch.zeros_like(layer.pre_w.grad))


def test_dendritic_branch_layer_passes_connection_mask_to_topk():
    connection_mask = sample_fixed_indegree_mask(
        out_features=3,
        in_features=5,
        indegree=2,
        generator=torch.Generator().manual_seed(7),
    )
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=5,
        excitatory_synapses_per_branch=5,
        excitatory_connection_mask=connection_mask,
        input_branch_factor=None,
        reactivate=False,
        dbl_init_method="default",
    )

    active = layer.branch_excitation.weight_mask().bool()

    assert torch.equal(active, connection_mask)


def test_sample_probability_mask_is_reproducible_and_calibrated():
    generator_a = torch.Generator().manual_seed(11)
    generator_b = torch.Generator().manual_seed(11)

    mask_a = sample_probability_mask(200, 100, 0.25, generator=generator_a)
    mask_b = sample_probability_mask(200, 100, 0.25, generator=generator_b)

    assert torch.equal(mask_a, mask_b)
    assert mask_a.float().mean().item() == pytest.approx(0.25, abs=0.02)


def test_sample_probability_mask_repairs_empty_rows_when_possible():
    generator = torch.Generator().manual_seed(23)

    mask = sample_probability_mask(50, 4, 0.05, generator=generator)

    assert bool((mask.sum(dim=1) > 0).all())


def test_sample_fixed_indegree_mask_sets_exact_row_counts():
    mask = sample_fixed_indegree_mask(
        out_features=8,
        in_features=20,
        indegree=4,
        generator=torch.Generator().manual_seed(13),
    )

    assert mask.dtype == torch.bool
    assert torch.equal(mask.sum(dim=1), torch.full((8,), 4))


def test_distance_kernel_probabilities_decay_with_distance():
    pre_positions = torch.tensor([[0.0], [1.0], [2.0]])
    post_positions = torch.tensor([[0.0]])

    probs = distance_kernel_probabilities(
        pre_positions,
        post_positions,
        kernel="gaussian",
        sigma=1.0,
    )

    assert probs.shape == (1, 3)
    assert probs[0, 0] > probs[0, 1] > probs[0, 2]


def test_factory_applies_structured_connectivity_to_feedforward_einet():
    config = {
        "type": "dendritic_shunting",
        "architecture": {
            "excitatory_layer_sizes": [4],
            "inhibitory_layer_sizes": [],
            "excitatory_branch_factors": [1],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [5],
            "ie_synapses_per_branch_per_layer": [0],
            "structured": {
                "enabled": True,
                "seed": 17,
                "pathways": {
                    "ee": {"method": "fixed_indegree", "indegree": 2},
                },
            },
        },
        "transfer": {"input_mode": 1},
        "morphology": {
            "somatic_synapses": True,
            "use_shunting": True,
            "weight_transform": "exp",
            "dbl_init_method": "default",
        },
        "sparsity": {"type": "standard", "init_method": "xavier_normal"},
        "reactivation": {"enabled": False},
        "blocklinear": {},
        "implementation": {"compile_forward": False},
    }

    net = get_architecture("dendritic_shunting", config, input_dim=6)
    branch_layers = net.layers[0].excitatory_cells.branch_layers

    assert branch_layers
    for branch_layer in branch_layers:
        mask = branch_layer.branch_excitation.weight_mask()
        assert torch.equal(mask.sum(dim=1), torch.full((mask.shape[0],), 2.0))


def test_factory_applies_structured_connectivity_to_recurrent_einet():
    config = {
        "type": "dendritic_shunting",
        "architecture": {
            "excitatory_layer_sizes": [5],
            "inhibitory_layer_sizes": [],
            "excitatory_branch_factors": [1],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [3],
            "ie_synapses_per_branch_per_layer": [0],
            "rec_ee_synapses_per_branch": [4],
            "structured": {
                "enabled": True,
                "seed": 19,
                "pathways": {
                    "rec_ee": {"method": "fixed_indegree", "indegree": 1},
                },
            },
        },
        "transfer": {"input_mode": 1},
        "morphology": {
            "somatic_synapses": True,
            "use_shunting": True,
            "allow_self_recurrence": True,
            "weight_transform": "exp",
            "dbl_init_method": "default",
        },
        "sparsity": {"type": "standard", "init_method": "xavier_normal"},
        "reactivation": {"enabled": False},
        "blocklinear": {},
        "implementation": {"compile_forward": False},
        "recurrent_ei": {"output_mode": "last"},
    }

    net = get_architecture("dendritic_shunting", config, input_dim=6)
    branch_layers = net.layers[0].e_population.branch_layers

    assert branch_layers
    for branch_layer in branch_layers:
        mask = branch_layer.branch_recurrent.weight_mask()
        assert torch.equal(mask.sum(dim=1), torch.ones(mask.shape[0]))
