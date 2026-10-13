import pytest
import torch

from dendritic_modeling.config import TransferFunctionConfig
from dendritic_modeling.networks import (
    AdaptiveInitConfig,
    DeepSTConfig,
    DendriticBranchConfig,
    DendriticSynapseConfig,
    DenseToSparseConfig,
    ExcitationInhibitionNetwork,
    IndexedSynapseConfig,
    MorphologyConfig,
    ReactivationConfig,
    TopKConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    DendriNet,
    DendriticBranchLayer,
    synapse_config as synapse_config_module,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_options import (
    resolve_branch_config,
    resolve_branch_layer_synapse_config,
    resolve_branch_options,
    resolve_synapse_config,
    resolve_synapse_options,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
    ExcitationInhibitionLayer,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)


def _assert_state_dict_exact(left, right):
    assert left.keys() == right.keys()
    for key in left:
        torch.testing.assert_close(left[key], right[key], rtol=0.0, atol=0.0)


def test_dendritic_synapse_section_configs_are_public_reexports():
    assert AdaptiveInitConfig is synapse_config_module.AdaptiveInitConfig
    assert DeepSTConfig is synapse_config_module.DeepSTConfig
    assert DenseToSparseConfig is synapse_config_module.DenseToSparseConfig
    assert IndexedSynapseConfig is synapse_config_module.IndexedSynapseConfig
    assert MorphologyConfig is synapse_config_module.MorphologyConfig
    assert ReactivationConfig is synapse_config_module.ReactivationConfig
    assert TopKConfig is synapse_config_module.TopKConfig


def test_dendritic_synapse_config_round_trips_legacy_kwargs():
    legacy_kwargs = {
        "topk_type": "indexed_rewire",
        "topk_init_method": "xavier_normal",
        "topk_gamma": 0.75,
        "indexed_seed": 17,
        "indexed_support_group_rows": 4,
        "indexed_support_col_block": 2,
        "indexed_output_chunk_size": 64,
        "indexed_rewire_frequency": 9,
        "indexed_rewire_quantile": 0.2,
        "dense_to_sparse_initial_density": 0.4,
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "reactivation_init_policy": "fixed",
        "reactivation_init_m": 1.2,
        "reactivation_init_b": 0.3,
        "use_shunting": False,
        "use_additive_normalization": True,
        "weight_transform": "softplus",
        "dbl_init_method": "default",
        "adaptive_initialization": False,
        "adaptive_initialization_policy": "preserve_shunting_center",
        "adaptive_target_conductance": 4.0,
        "initial_child_conductance": 3.0,
        "branch_factors": [2, 1],
        "unknown_activation_kwarg": "kept out of config",
    }

    config, extra = DendriticSynapseConfig.split_kwargs(legacy_kwargs)
    round_trip = config.to_kwargs()

    assert extra == {"unknown_activation_kwarg": "kept out of config"}
    for key, value in legacy_kwargs.items():
        if key in extra:
            continue
        assert round_trip[key] == value


def test_dendritic_synapse_config_round_trips_every_legacy_key():
    legacy_kwargs = {
        "topk_init_method": "kaiming_uniform",
        "topk_noise_level": 0.2,
        "topk_type": "indexed_dynamic",
        "topk_weight_norm_order": 2,
        "topk_gamma": 0.75,
        "topk_temperature": 0.3,
        "topk_ultrafast": True,
        "topk_strategy": "conductance_dynamic",
        "topk_variance_score": "relative",
        "dense_to_sparse_initial_density": 0.6,
        "dense_to_sparse_initial_k": 4,
        "dense_to_sparse_start_step": 3,
        "dense_to_sparse_end_step": 30,
        "dense_to_sparse_update_interval": 5,
        "dense_to_sparse_schedule": "linear",
        "dense_to_sparse_freeze_on_end": True,
        "dense_to_sparse_advance_on_forward": False,
        "dense_to_sparse_prune_metric": "magnitude",
        "release_p_min": 0.6,
        "release_p_max": 0.9,
        "release_weight_scale": 0.02,
        "release_impact_decay": 0.95,
        "release_impact_metric": "active_drive",
        "release_max_conductance": 30.0,
        "release_track_source_activity": True,
        "indexed_seed": 17,
        "indexed_output_chunk_size": 128,
        "indexed_candidate_size": 64,
        "indexed_selection": "stochastic",
        "indexed_rewire_frequency": 9,
        "indexed_rewire_quantile": 0.2,
        "indexed_rewire_until_step": 40,
        "indexed_index_dtype": "auto",
        "indexed_workspace_mb": 96,
        "indexed_cache_transformed_weights": True,
        "indexed_recompute_backward": True,
        "indexed_projection_backend": "recompute",
        "indexed_persistent_indices": False,
        "indexed_init_mode": "rank0_broadcast",
        "indexed_support_group_rows": 4,
        "indexed_support_col_block": 2,
        "use_shunting": False,
        "use_additive_normalization": True,
        "leak_conductance": 0.05,
        "learnable_leak": True,
        "signal_rounding": "float16",
        "additive_mode": "raw",
        "additive_tangent_n0": None,
        "additive_tangent_t0": None,
        "additive_options": {"pool": "input", "levels": [1, 2]},
        "epsilon": 1e-6,
        "weight_transform": "softplus",
        "efficient_blocklinear": True,
        "blocklinear_strategy": "conductance_dynamic",
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "dendritic_activation": "tanh",
        "reactivation_init_m": 1.2,
        "reactivation_init_b": 0.3,
        "reactivation_init_policy": "fixed",
        "reactivation_soma_type": "param_relu",
        "reactivation_soma_init_m": 3.0,
        "reactivation_soma_init_b": 0.25,
        "reactivation_occupancy_quantile_low": 0.1,
        "reactivation_occupancy_quantile_high": 0.9,
        "reactivation_occupancy_target_low": 0.2,
        "reactivation_occupancy_target_high": 0.8,
        "reactivation_calibration_min_quantile_width": 0.02,
        "reactivation_calibration_max_m": 25.0,
        "reactivation_calibration_revert_on_invalid": False,
        "reactivation_sigma_aware_k": 0.4,
        "reactivation_strategy": "straight_through",
        "reactivation_memory_efficient": True,
        "dbl_init_method": "default",
        "adaptive_initialization": False,
        "adaptive_initialization_policy": "preserve_shunting_center",
        "adaptive_target_conductance": 4.0,
        "initial_child_conductance": 3.0,
        "initial_conductance_scale": 2.0,
        "initial_child_conductance_log_sd": 0.3,
        "branch_factors": [2, 1],
        "initialization_seed": 123,
        "initialization_namespace": "layer0.excitatory",
        "use_noise": False,
        "rewiring_mode": "local",
        "rewire_frequency": 7,
        "synapses_per_branch": 5,
        "sigma": 0.02,
        "excitatory_target_density": 0.3,
        "inhibitory_target_density": 0.4,
        "freeze_excitatory_connectivity": True,
        "freeze_inhibitory_connectivity": True,
        "init_method": "xavier_uniform",
        "weight_threshold": 1e-4,
        "credit_trace_decay": 0.8,
        "credit_candidate_pool_size": 12,
        "credit_turnover_fraction": 0.2,
        "credit_weak_active_pool_fraction": 0.75,
        "credit_min_observations": 3,
        "credit_swap_margin": 0.01,
        "credit_force_turnover": True,
        "credit_selection": "shuffled_credit",
        "credit_warmup_steps": 9,
        "print_hooks": True,
    }

    round_trip = DendriticSynapseConfig.from_kwargs(legacy_kwargs).to_kwargs()

    assert set(round_trip) == set(legacy_kwargs)
    for key, value in legacy_kwargs.items():
        assert round_trip[key] == value


def test_dendritic_synapse_config_accepts_nested_section_mapping():
    config = DendriticSynapseConfig.from_config(
        {
            "topk": {"type": "indexed_rewire", "gamma": 0.5},
            "morphology": MorphologyConfig(
                use_shunting=False,
                weight_transform="softplus",
                efficient_blocklinear=True,
            ),
            "reactivation": ReactivationConfig(
                enabled=True,
                type="param_tanh",
                init_m=1.2,
                init_b=0.25,
            ),
            "print_hooks": True,
        }
    )

    kwargs = config.to_kwargs()
    assert kwargs["topk_type"] == "indexed_rewire"
    assert kwargs["topk_gamma"] == 0.5
    assert kwargs["use_shunting"] is False
    assert kwargs["weight_transform"] == "softplus"
    assert kwargs["efficient_blocklinear"] is True
    assert kwargs["reactivate"] is True
    assert kwargs["reactivation_init_m"] == 1.2
    assert kwargs["reactivation_init_b"] == 0.25
    assert kwargs["print_hooks"] is True


def test_dendritic_synapse_config_nested_sections_preserve_flat_defaults():
    config = DendriticSynapseConfig.from_sections(
        {
            "topk_type": "standard",
            "topk_gamma": 0.75,
            "topk": {"type": "indexed_rewire"},
            "morphology": {"weight_transform": "softplus"},
        }
    )

    kwargs = config.to_kwargs()
    assert kwargs["topk_type"] == "indexed_rewire"
    assert kwargs["topk_gamma"] == 0.75
    assert kwargs["weight_transform"] == "softplus"
    assert kwargs["use_shunting"] is True


def test_dendritic_synapse_config_accepts_nested_section_objects():
    config = DendriticSynapseConfig.from_sections(
        topk=TopKConfig(type="indexed_rewire", gamma=0.4),
        morphology={"use_shunting": False},
    )

    kwargs = config.to_kwargs()
    assert kwargs["topk_type"] == "indexed_rewire"
    assert kwargs["topk_gamma"] == 0.4
    assert kwargs["use_shunting"] is False


def test_dendritic_synapse_config_from_config_ignores_non_section_extras():
    config = DendriticSynapseConfig.from_config(
        {"topk": "not a section payload", "topk_type": "indexed_rewire"}
    )

    kwargs = config.to_kwargs()
    assert kwargs["topk_type"] == "indexed_rewire"


def test_synapse_options_resolve_typed_config_and_legacy_kwargs():
    formal_kwargs = {
        "topk_type": "indexed_rewire",
        "topk_gamma": 0.75,
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "weight_transform": "softplus",
    }
    extra_kwargs = {
        "indexed_seed": 123,
        "non_synapse_option": "keep",
    }

    config, remaining = resolve_synapse_config(
        formal_synapse_kwargs=formal_kwargs,
        extra_kwargs=extra_kwargs,
        synapse_config=None,
    )
    options, legacy_remaining = resolve_synapse_options(
        formal_synapse_kwargs=formal_kwargs,
        extra_kwargs=extra_kwargs,
        synapse_config=None,
    )

    assert options == config.to_kwargs()
    assert remaining == legacy_remaining == {"non_synapse_option": "keep"}
    assert config.to_kwargs()["indexed_seed"] == 123


def test_branch_layer_synapse_config_resolves_activation_alias_and_extras():
    formal_kwargs = {
        "topk_type": "standard",
        "dendritic_activation": "tanh",
        "reactivate": False,
        "reactivation_type": "param_tanh",
        "weight_transform": "identity",
    }
    extra_kwargs = {
        "indexed_seed": 7,
        "activation_extra": "keep",
    }

    config, options, remaining = resolve_branch_layer_synapse_config(
        formal_synapse_kwargs=formal_kwargs,
        extra_kwargs=extra_kwargs,
        synapse_config=None,
    )

    assert options == config.to_kwargs()
    assert remaining == {"activation_extra": "keep"}
    assert options["indexed_seed"] == 7
    assert options["dendritic_activation"] == "tanh"
    assert options["reactivate"] is True
    assert options["reactivation_type"] == "tanh"


def test_dendritic_branch_config_round_trips_legacy_kwargs():
    connection_mask = torch.ones(4, 7, dtype=torch.bool)
    legacy_kwargs = {
        "output_dim": 4,
        "excitatory_input_dim": 7,
        "excitatory_synapses_per_branch": 3,
        "inhibitory_input_dim": 5,
        "inhibitory_synapses_per_branch": 2,
        "input_branch_factor": 2,
        "layer_idx": 1,
        "excitatory_connection_mask": connection_mask,
    }

    config = DendriticBranchConfig.from_kwargs(legacy_kwargs)
    round_trip = config.to_kwargs()

    for key, value in legacy_kwargs.items():
        assert round_trip[key] is value or round_trip[key] == value


def test_dendritic_branch_config_requires_output_dim():
    with pytest.raises(ValueError, match="output_dim"):
        DendriticBranchConfig.from_kwargs(excitatory_input_dim=3)


def test_branch_options_resolve_typed_config_and_legacy_kwargs():
    config = DendriticBranchConfig(
        output_dim=4,
        excitatory_input_dim=7,
        excitatory_synapses_per_branch=3,
        layer_idx=2,
    )

    assert resolve_branch_config(None) is None
    assert resolve_branch_config(config) is config
    assert resolve_branch_options(config) == config.to_kwargs()
    assert resolve_branch_options(config.to_kwargs()) == config.to_kwargs()


def test_branch_layer_typed_branch_config_matches_legacy_kwargs_numerically():
    geometry_kwargs = {
        "output_dim": 4,
        "excitatory_input_dim": 7,
        "excitatory_synapses_per_branch": 3,
        "inhibitory_input_dim": 5,
        "inhibitory_synapses_per_branch": 2,
        "input_branch_factor": 2,
        "layer_idx": 1,
    }
    synapse_kwargs = {
        "topk_type": "standard",
        "topk_init_method": "xavier_normal",
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "reactivation_init_policy": "fixed",
        "reactivation_init_m": 1.3,
        "reactivation_init_b": 0.25,
        "use_shunting": True,
        "weight_transform": "exp",
        "dbl_init_method": "default",
        "adaptive_initialization": False,
        "branch_factors": [2],
    }

    torch.manual_seed(123)
    legacy_layer = DendriticBranchLayer(
        **geometry_kwargs,
        **synapse_kwargs,
    )
    torch.manual_seed(123)
    typed_layer = DendriticBranchLayer(
        branch_config=DendriticBranchConfig.from_kwargs(geometry_kwargs),
        **synapse_kwargs,
    )

    assert (
        legacy_layer.branch_config.to_kwargs() == typed_layer.branch_config.to_kwargs()
    )
    assert (
        legacy_layer.synapse_config.to_kwargs()
        == typed_layer.synapse_config.to_kwargs()
    )

    _assert_state_dict_exact(legacy_layer.state_dict(), typed_layer.state_dict())

    x = torch.rand(3, 7)
    inhibitory = torch.rand(3, 5)
    branch_input = torch.rand(3, 8)

    torch.testing.assert_close(
        legacy_layer(x, inhibitory, branch_input),
        typed_layer(x, inhibitory, branch_input),
        rtol=0.0,
        atol=0.0,
    )


def test_branch_layer_stored_synapse_config_reflects_activation_alias():
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=5,
        excitatory_synapses_per_branch=2,
        dendritic_activation="tanh",
        topk_type="standard",
        weight_transform="identity",
        dbl_init_method="default",
        adaptive_initialization=False,
    )

    kwargs = layer.synapse_config.to_kwargs()
    assert kwargs["dendritic_activation"] == "tanh"
    assert kwargs["reactivate"] is True
    assert kwargs["reactivation_type"] == "tanh"
    assert layer.dendritic_activation == "tanh"


def test_branch_layer_typed_config_matches_legacy_kwargs_numerically():
    geometry_kwargs = {
        "output_dim": 4,
        "excitatory_input_dim": 7,
        "excitatory_synapses_per_branch": 3,
        "inhibitory_input_dim": 5,
        "inhibitory_synapses_per_branch": 2,
        "input_branch_factor": 2,
        "layer_idx": 1,
    }
    legacy_synapse_kwargs = {
        "topk_type": "standard",
        "topk_init_method": "xavier_normal",
        "topk_gamma": 0.6,
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "reactivation_init_policy": "fixed",
        "reactivation_init_m": 1.3,
        "reactivation_init_b": 0.25,
        "use_shunting": True,
        "weight_transform": "exp",
        "dbl_init_method": "default",
        "adaptive_initialization": False,
        "branch_factors": [2],
    }

    torch.manual_seed(123)
    legacy_layer = DendriticBranchLayer(
        **geometry_kwargs,
        **legacy_synapse_kwargs,
    )
    torch.manual_seed(123)
    typed_layer = DendriticBranchLayer(
        **geometry_kwargs,
        synapse_config=DendriticSynapseConfig.from_kwargs(legacy_synapse_kwargs),
    )

    _assert_state_dict_exact(legacy_layer.state_dict(), typed_layer.state_dict())

    x = torch.rand(3, 7)
    inhibitory = torch.rand(3, 5)
    branch_input = torch.rand(3, 8)

    torch.testing.assert_close(
        legacy_layer(x, inhibitory, branch_input),
        typed_layer(x, inhibitory, branch_input),
        rtol=0.0,
        atol=0.0,
    )


def test_dendrinet_typed_config_matches_legacy_kwargs_numerically():
    geometry_kwargs = {
        "n_soma": 3,
        "branch_factors": [2],
        "excitatory_input_dim": 7,
        "excitatory_synapses_per_branch": 3,
        "inhibitory_input_dim": 5,
        "inhibitory_synapses_per_branch": 2,
        "somatic_synapses": True,
    }
    legacy_synapse_kwargs = {
        "topk_type": "standard",
        "topk_init_method": "xavier_normal",
        "topk_gamma": 0.7,
        "reactivate": True,
        "reactivation_type": "param_tanh",
        "reactivation_init_policy": "fixed",
        "reactivation_init_m": 1.1,
        "reactivation_init_b": 0.2,
        "use_shunting": True,
        "weight_transform": "exp",
        "dbl_init_method": "default",
        "adaptive_initialization": False,
    }

    torch.manual_seed(321)
    legacy_net = DendriNet(
        **geometry_kwargs,
        **legacy_synapse_kwargs,
    )
    torch.manual_seed(321)
    typed_net = DendriNet(
        **geometry_kwargs,
        synapse_config=DendriticSynapseConfig.from_kwargs(legacy_synapse_kwargs),
    )

    assert legacy_net.synapse_config.to_kwargs() == typed_net.synapse_config.to_kwargs()
    assert len(legacy_net.branch_configs) == len(legacy_net.branch_layers)
    assert len(typed_net.branch_configs) == len(typed_net.branch_layers)
    assert [cfg.to_kwargs() for cfg in typed_net.branch_configs] == [
        layer.branch_config.to_kwargs() for layer in typed_net.branch_layers
    ]

    _assert_state_dict_exact(legacy_net.state_dict(), typed_net.state_dict())

    x = torch.rand(4, 7)
    inhibitory = torch.rand(4, 5)

    torch.testing.assert_close(
        legacy_net(x, inhibitory),
        typed_net(x, inhibitory),
        rtol=0.0,
        atol=0.0,
    )


def test_population_config_builds_equivalent_synapse_config():
    population = PopulationConfig(
        n_neurons=5,
        branch_factors=[2, 1],
        topk_init_method="kaiming_normal",
        topk_noise_level=0.01,
        topk_type="indexed_rewire",
        topk_weight_norm_order=2,
        topk_gamma=0.8,
        topk_temperature=0.3,
        topk_ultrafast=True,
        topk_gradient_scaling="conductance_dynamic",
        indexed_seed=17,
        indexed_candidate_size=32,
        indexed_selection="softmax",
        indexed_output_chunk_size=128,
        indexed_rewire_frequency=11,
        indexed_rewire_quantile=0.2,
        indexed_rewire_until_step=400,
        indexed_index_dtype="auto",
        indexed_workspace_mb=96,
        indexed_cache_transformed_weights=True,
        indexed_recompute_backward=True,
        indexed_persistent_indices=True,
        indexed_init_mode="rank0_broadcast",
        dense_to_sparse_initial_density=0.7,
        dense_to_sparse_initial_k=4,
        dense_to_sparse_start_step=2,
        dense_to_sparse_end_step=30,
        dense_to_sparse_update_interval=3,
        dense_to_sparse_schedule="linear",
        dense_to_sparse_freeze_on_end=True,
        dense_to_sparse_advance_on_forward=False,
        dense_to_sparse_prune_metric="magnitude",
        weight_transform="softplus",
        use_shunting=False,
        use_additive_normalization=True,
        reactivate=False,
        reactivation_type="none",
        reactivation_init_m=1.4,
        reactivation_init_b=0.1,
        reactivation_init_policy="fixed",
        reactivation_occupancy_quantile_low=0.2,
        reactivation_occupancy_quantile_high=0.8,
        reactivation_occupancy_target_low=0.25,
        reactivation_occupancy_target_high=0.75,
        reactivation_calibration_min_quantile_width=0.02,
        reactivation_calibration_max_m=25.0,
        reactivation_calibration_revert_on_invalid=False,
        reactivation_sigma_aware_k=0.5,
        reactivation_gradient_scaling="straight_through",
        blocklinear_gradient_scaling="conductance_dynamic",
        epsilon=1e-6,
        dbl_init_method="default",
        efficient_blocklinear=True,
        print_hooks=True,
        excitatory_target_density=0.3,
        inhibitory_target_density=0.4,
        use_noise=False,
        sigma=0.02,
        rewiring_mode="local",
        rewire_frequency=5,
        synapses_per_branch=6,
        freeze_excitatory_connectivity=True,
        freeze_inhibitory_connectivity=True,
        init_method="xavier_uniform",
        weight_threshold=1e-4,
        adaptive_initialization=True,
        adaptive_initialization_policy="legacy_scale",
        adaptive_target_conductance=3.0,
        initial_child_conductance=2.5,
    )

    kwargs = DendriticSynapseConfig.from_population_config(population).to_kwargs()

    for key, value in kwargs.items():
        if hasattr(population, key):
            assert value == getattr(population, key)
    assert kwargs["dendritic_activation"] is None


def test_ei_layer_typed_config_matches_legacy_kwargs_numerically():
    geometry_kwargs = {
        "n_excitatory_cells": 4,
        "n_inhibitory_cells": 3,
        "excitatory_branch_factors": [1],
        "inhibitory_branch_factors": [1],
        "excitatory_input_dim": 6,
        "inhibitory_input_dim": 6,
        "ee_synapses_per_branch": 2,
        "ei_synapses_per_branch": 2,
        "ie_synapses_per_branch": 1,
        "ii_synapses_per_branch": 1,
        "build_inhibitory_cells": True,
        "inhibitory_network_type": "dendritic",
        "somatic_synapses": True,
    }
    legacy_synapse_kwargs = {
        "topk_type": "standard",
        "topk_init_method": "xavier_normal",
        "topk_gamma": 0.9,
        "reactivate": False,
        "use_shunting": False,
        "weight_transform": "softplus",
        "dbl_init_method": "default",
        "adaptive_initialization": False,
    }

    torch.manual_seed(456)
    legacy_layer = ExcitationInhibitionLayer(
        **geometry_kwargs,
        **legacy_synapse_kwargs,
    )
    provided_synapse_config = DendriticSynapseConfig.from_kwargs(legacy_synapse_kwargs)
    torch.manual_seed(456)
    typed_layer = ExcitationInhibitionLayer(
        **geometry_kwargs,
        synapse_config=provided_synapse_config,
    )

    assert typed_layer.synapse_config is provided_synapse_config
    assert (
        legacy_layer.excitatory_synapse_config.to_kwargs()
        == legacy_layer.excitatory_cells.synapse_config.to_kwargs()
    )
    assert (
        legacy_layer.inhibitory_synapse_config.to_kwargs()
        == legacy_layer.inhibitory_cells.synapse_config.to_kwargs()
    )
    assert (
        typed_layer.excitatory_synapse_config.to_kwargs()
        == typed_layer.excitatory_cells.synapse_config.to_kwargs()
    )
    assert (
        typed_layer.inhibitory_synapse_config.to_kwargs()
        == typed_layer.inhibitory_cells.synapse_config.to_kwargs()
    )
    assert [cfg.to_kwargs() for cfg in typed_layer.excitatory_branch_configs] == [
        layer.branch_config.to_kwargs()
        for layer in typed_layer.excitatory_cells.branch_layers
    ]
    assert [cfg.to_kwargs() for cfg in typed_layer.inhibitory_branch_configs] == [
        layer.branch_config.to_kwargs()
        for layer in typed_layer.inhibitory_cells.branch_layers
    ]

    _assert_state_dict_exact(legacy_layer.state_dict(), typed_layer.state_dict())

    x = torch.rand(5, 6)
    inhibitory = torch.rand(5, 6)
    legacy_exc, legacy_inh = legacy_layer(x, inhibitory)
    typed_exc, typed_inh = typed_layer(x, inhibitory)

    torch.testing.assert_close(legacy_exc, typed_exc, rtol=0.0, atol=0.0)
    torch.testing.assert_close(legacy_inh, typed_inh, rtol=0.0, atol=0.0)


def test_ei_network_typed_config_metadata_matches_layers_numerically():
    geometry_kwargs = {
        "input_dim": 6,
        "transfer_params": TransferFunctionConfig().asdict(),
        "excitatory_layer_sizes": [4, 3],
        "inhibitory_layer_sizes": [3, 2],
        "excitatory_branch_factors": [1],
        "inhibitory_branch_factors": [1],
        "ee_synapses_per_branch_per_layer": [2, 2],
        "ei_synapses_per_branch_per_layer": [2, 1],
        "ie_synapses_per_branch_per_layer": [1, 1],
        "ii_synapses_per_branch_per_layer": [1, 1],
        "somatic_synapses": True,
    }
    legacy_synapse_kwargs = {
        "topk_type": "standard",
        "topk_init_method": "xavier_normal",
        "topk_gamma": 0.8,
        "reactivate": False,
        "use_shunting": False,
        "weight_transform": "softplus",
        "dbl_init_method": "default",
        "adaptive_initialization": False,
    }

    torch.manual_seed(654)
    legacy_net = ExcitationInhibitionNetwork(
        **geometry_kwargs,
        **legacy_synapse_kwargs,
    )
    provided_synapse_config = DendriticSynapseConfig.from_kwargs(legacy_synapse_kwargs)
    torch.manual_seed(654)
    typed_net = ExcitationInhibitionNetwork(
        **geometry_kwargs,
        synapse_config=provided_synapse_config,
    )

    assert typed_net.synapse_configs == tuple(
        layer.synapse_config for layer in typed_net.layers
    )
    assert typed_net.excitatory_synapse_configs == tuple(
        layer.excitatory_synapse_config for layer in typed_net.layers
    )
    assert typed_net.inhibitory_synapse_configs == tuple(
        layer.inhibitory_synapse_config for layer in typed_net.layers
    )
    assert typed_net.excitatory_branch_configs == tuple(
        tuple(layer.excitatory_branch_configs) for layer in typed_net.layers
    )
    assert typed_net.inhibitory_branch_configs == tuple(
        tuple(layer.inhibitory_branch_configs) for layer in typed_net.layers
    )
    assert [cfg.to_kwargs() for cfg in typed_net.branch_configs] == [
        layer.branch_config.to_kwargs() for layer in typed_net.branch_layers
    ]

    _assert_state_dict_exact(legacy_net.state_dict(), typed_net.state_dict())

    x = torch.rand(5, 6)
    torch.testing.assert_close(
        legacy_net(x),
        typed_net(x),
        rtol=0.0,
        atol=0.0,
    )


def test_dendrinet_passes_legacy_synapse_kwargs_through_typed_config():
    net = DendriNet(
        n_soma=3,
        branch_factors=[2],
        excitatory_input_dim=6,
        excitatory_synapses_per_branch=2,
        inhibitory_input_dim=4,
        inhibitory_synapses_per_branch=1,
        somatic_synapses=True,
        topk_type="indexed_rewire",
        indexed_seed=31,
        indexed_output_chunk_size=16,
        indexed_rewire_frequency=7,
        indexed_rewire_quantile=0.15,
        reactivate=False,
        use_shunting=False,
        use_additive_normalization=True,
        weight_transform="softplus",
    )

    assert len(net.branch_layers) == 2
    for layer in net.branch_layers:
        assert layer.topk_type == "indexed_rewire"
        assert layer.indexed_seed == 31
        assert layer.indexed_output_chunk_size == 16
        assert layer.indexed_rewire_frequency == 7
        assert layer.indexed_rewire_quantile == 0.15
        assert layer.use_additive_normalization is True
        assert layer.weight_transform == "softplus"


def _typed(**overrides):
    return DendriticSynapseConfig.from_kwargs(
        {"use_shunting": True, "reactivate": False, **overrides}
    )


def _layer(**kwargs):
    return DendriticBranchLayer(
        output_dim=2, excitatory_input_dim=4, excitatory_synapses_per_branch=2, **kwargs
    )


def test_explicit_options_that_contradict_a_typed_synapse_config_raise():
    with pytest.raises(ValueError, match=r"contradict it: .*use_shunting"):
        _layer(synapse_config=_typed(), use_shunting=False)
    with pytest.raises(ValueError, match=r"contradict it: .*topk_type"):
        _layer(synapse_config=_typed(), topk_type="indexed")
    # Values that agree with the typed config are accepted.
    _layer(synapse_config=_typed(), use_shunting=True)


def test_geometry_arguments_that_contradict_a_branch_config_raise():
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_config import (
        DendriticBranchConfig,
    )

    config = DendriticBranchConfig(
        output_dim=2, excitatory_input_dim=4, excitatory_synapses_per_branch=2
    )
    with pytest.raises(ValueError, match=r"branch_config.*output_dim"):
        DendriticBranchLayer(branch_config=config, output_dim=3, reactivate=False)
    DendriticBranchLayer(branch_config=config, output_dim=2, reactivate=False)


def test_dendrinet_arguments_that_contradict_its_config_raise():
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet_config import (
        DendriNetConfig,
    )

    config = DendriNetConfig(
        n_soma=3,
        branch_factors=(2,),
        excitatory_input_dim=4,
        excitatory_synapses_per_branch=2,
        synapse_config=_typed(),
    )
    with pytest.raises(ValueError, match=r"config.*n_soma"):
        DendriNet(config=config, n_soma=5)
    with pytest.raises(ValueError, match="different synapse_config"):
        DendriNet(config=config, synapse_config=_typed(use_shunting=False))
    with pytest.raises(ValueError, match=r"contradict it: .*use_shunting"):
        DendriNet(config=config, use_shunting=False)
    with pytest.raises(ValueError, match="flat synapse keys"):
        DendriNetConfig.from_kwargs(
            {
                "n_soma": 3,
                "branch_factors": [2],
                "synapse_config": _typed(),
                "use_shunting": False,
            }
        )
    DendriNet(config=config)
