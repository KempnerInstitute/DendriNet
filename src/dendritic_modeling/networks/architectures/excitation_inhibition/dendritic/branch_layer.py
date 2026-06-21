"""
Dendritic Branch Layer for neural networks.

This module implements the DendriticBranchLayer which emulates dendritic
computation along a layer of branches, modeling the interaction between
excitatory and inhibitory inputs.
"""

from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.activations import (
    ActivationFactory,
    resolve_dendritic_activation,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
    EfficientBlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_config import (
    DendriticBranchConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.gradient_scaling import (
    GradientScaler,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    analytical_expectation_dbl_init,
    default_dbl_init,
    ei_equivalence_dbl_init,
    identity_weighttransform_dbl_init,
    naive_dbl_init,
    normalize_reactivation_init_policy,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)


class DendriticBranchLayer(nn.Module):
    """
    Emulates dendritic computation along a layer of branches (equally distant
    from soma).

    This layer models the interaction between excitatory and inhibitory inputs,
    simulating dendritic processing in biological neurons. It aggregates
    information from converging branches in the previous layer and optionally
    incorporates inhibitory modulation.

    Parameters
    ----------
    output_dim : int
        The number of output features.

    excitatory_input_dim : int
        The number of input features from excitatory cells.

    excitatory_synapses_per_branch : int
        The number of excitatory synapses per branch.

    inhibitory_input_dim : int, optional
        The number of input features from inhibitory cells. Defaults to `None`.

    inhibitory_synapses_per_branch : int, optional
        The number of inhibitory synapses per branch. Defaults to `None`.

    input_branch_factor : int, optional
        The number of branches converging onto a single branch. Defaults to
        `None`.
    """

    def __init__(
        self,
        output_dim=None,
        excitatory_input_dim=None,
        excitatory_synapses_per_branch=None,
        inhibitory_input_dim=None,
        inhibitory_synapses_per_branch=None,
        input_branch_factor=None,
        recurrent_input_dim=None,
        recurrent_synapses_per_branch=None,
        recurrent_forbidden_input_index_per_output=None,
        rec_inhibitory_input_dim=None,
        rec_inhibitory_synapses_per_branch=None,
        rec_inhibitory_forbidden_input_index_per_output=None,
        topk_init_method="xavier_normal",
        topk_noise_level=0.0,
        topk_type="standard",  # New parameter
        topk_weight_norm_order=None,
        topk_gamma=1.0,
        topk_temperature=0.5,  # For stochastic TopK
        topk_ultrafast=False,  # For much faster stochastic TopK
        topk_strategy="none",
        dense_to_sparse_initial_density=1.0,
        dense_to_sparse_initial_k=None,
        dense_to_sparse_start_step=0,
        dense_to_sparse_end_step=1000,
        dense_to_sparse_update_interval=1,
        dense_to_sparse_schedule="cubic",
        dense_to_sparse_freeze_on_end=False,
        dense_to_sparse_advance_on_forward=True,
        dense_to_sparse_prune_metric="weight",
        indexed_rewire_frequency=100,
        indexed_rewire_quantile=0.05,
        indexed_rewire_until_step=None,
        use_shunting=True,
        reactivate=False,
        reactivation_type="param_tanh",
        dendritic_activation=None,
        reactivation_init_m=1,
        reactivation_init_b=0.5,
        reactivation_init_policy="analytical",
        reactivation_occupancy_quantile_low=None,
        reactivation_occupancy_quantile_high=None,
        reactivation_occupancy_target_low=None,
        reactivation_occupancy_target_high=None,
        reactivation_calibration_min_quantile_width=1e-3,
        reactivation_calibration_max_m=50.0,
        reactivation_calibration_revert_on_invalid=True,
        reactivation_sigma_aware_k=0.25,
        dbl_init_method="analytical_expectation",
        reactivation_strategy="none",
        blocklinear_strategy="none",
        efficient_blocklinear: bool = False,
        layer_idx=0,
        print_hooks=False,
        # DEEPST specific parameters, clean up later
        use_noise=True,
        rewiring_mode="global",
        rewire_frequency=1,
        synapses_per_branch=None,
        sigma=0.05,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method=None,
        weight_threshold=1e-6,
        adaptive_initialization=True,
        adaptive_initialization_policy="legacy_scale",
        adaptive_target_conductance=5.0,
        branch_factors=None,  # Add branch_factors parameter
        epsilon=1e-8,  # Add configurable epsilon for numerical stability
        weight_transform="exp",  # Weight transformation for positive weights
        excitatory_connection_mask=None,
        inhibitory_connection_mask=None,
        recurrent_connection_mask=None,
        rec_inhibitory_connection_mask=None,
        branch_config=None,
        synapse_config=None,
        **kwargs,
    ):
        super().__init__()
        if branch_config is not None:
            branch_options = DendriticBranchConfig.from_config(
                branch_config
            ).to_kwargs()
            output_dim = branch_options["output_dim"]
            excitatory_input_dim = branch_options["excitatory_input_dim"]
            excitatory_synapses_per_branch = branch_options[
                "excitatory_synapses_per_branch"
            ]
            inhibitory_input_dim = branch_options["inhibitory_input_dim"]
            inhibitory_synapses_per_branch = branch_options[
                "inhibitory_synapses_per_branch"
            ]
            input_branch_factor = branch_options["input_branch_factor"]
            recurrent_input_dim = branch_options["recurrent_input_dim"]
            recurrent_synapses_per_branch = branch_options[
                "recurrent_synapses_per_branch"
            ]
            recurrent_forbidden_input_index_per_output = branch_options[
                "recurrent_forbidden_input_index_per_output"
            ]
            rec_inhibitory_input_dim = branch_options["rec_inhibitory_input_dim"]
            rec_inhibitory_synapses_per_branch = branch_options[
                "rec_inhibitory_synapses_per_branch"
            ]
            rec_inhibitory_forbidden_input_index_per_output = branch_options[
                "rec_inhibitory_forbidden_input_index_per_output"
            ]
            layer_idx = branch_options["layer_idx"]
            excitatory_connection_mask = branch_options["excitatory_connection_mask"]
            inhibitory_connection_mask = branch_options["inhibitory_connection_mask"]
            recurrent_connection_mask = branch_options["recurrent_connection_mask"]
            rec_inhibitory_connection_mask = branch_options[
                "rec_inhibitory_connection_mask"
            ]
        if output_dim is None:
            raise ValueError("DendriticBranchLayer requires output_dim")
        formal_synapse_kwargs = DendriticSynapseConfig.legacy_kwargs_from_mapping(
            locals()
        )
        if synapse_config is None:
            config_source = {**kwargs, **formal_synapse_kwargs}
            normalized_config = DendriticSynapseConfig.from_kwargs(config_source)
        else:
            normalized_config = DendriticSynapseConfig.from_config(synapse_config)
        synapse_options = normalized_config.to_kwargs()
        _, kwargs = DendriticSynapseConfig.split_kwargs(kwargs)

        topk_init_method = synapse_options["topk_init_method"]
        topk_noise_level = synapse_options["topk_noise_level"]
        topk_type = synapse_options["topk_type"]
        topk_weight_norm_order = synapse_options["topk_weight_norm_order"]
        topk_gamma = synapse_options["topk_gamma"]
        topk_temperature = synapse_options["topk_temperature"]
        topk_ultrafast = synapse_options["topk_ultrafast"]
        topk_strategy = synapse_options["topk_strategy"]
        dense_to_sparse_initial_density = synapse_options[
            "dense_to_sparse_initial_density"
        ]
        dense_to_sparse_initial_k = synapse_options["dense_to_sparse_initial_k"]
        dense_to_sparse_start_step = synapse_options["dense_to_sparse_start_step"]
        dense_to_sparse_end_step = synapse_options["dense_to_sparse_end_step"]
        dense_to_sparse_update_interval = synapse_options[
            "dense_to_sparse_update_interval"
        ]
        dense_to_sparse_schedule = synapse_options["dense_to_sparse_schedule"]
        dense_to_sparse_freeze_on_end = synapse_options["dense_to_sparse_freeze_on_end"]
        dense_to_sparse_advance_on_forward = synapse_options[
            "dense_to_sparse_advance_on_forward"
        ]
        dense_to_sparse_prune_metric = synapse_options["dense_to_sparse_prune_metric"]
        indexed_seed = synapse_options["indexed_seed"]
        indexed_output_chunk_size = synapse_options["indexed_output_chunk_size"]
        indexed_candidate_size = synapse_options["indexed_candidate_size"]
        indexed_selection = synapse_options["indexed_selection"]
        indexed_rewire_frequency = synapse_options["indexed_rewire_frequency"]
        indexed_rewire_quantile = synapse_options["indexed_rewire_quantile"]
        indexed_rewire_until_step = synapse_options["indexed_rewire_until_step"]
        use_shunting = synapse_options["use_shunting"]
        use_additive_normalization = synapse_options["use_additive_normalization"]
        reactivate = synapse_options["reactivate"]
        reactivation_type = synapse_options["reactivation_type"]
        dendritic_activation = synapse_options["dendritic_activation"]
        reactivation_init_m = synapse_options["reactivation_init_m"]
        reactivation_init_b = synapse_options["reactivation_init_b"]
        reactivation_init_policy = synapse_options["reactivation_init_policy"]
        reactivation_occupancy_quantile_low = synapse_options[
            "reactivation_occupancy_quantile_low"
        ]
        reactivation_occupancy_quantile_high = synapse_options[
            "reactivation_occupancy_quantile_high"
        ]
        reactivation_occupancy_target_low = synapse_options[
            "reactivation_occupancy_target_low"
        ]
        reactivation_occupancy_target_high = synapse_options[
            "reactivation_occupancy_target_high"
        ]
        reactivation_calibration_min_quantile_width = synapse_options[
            "reactivation_calibration_min_quantile_width"
        ]
        reactivation_calibration_max_m = synapse_options[
            "reactivation_calibration_max_m"
        ]
        reactivation_calibration_revert_on_invalid = synapse_options[
            "reactivation_calibration_revert_on_invalid"
        ]
        reactivation_sigma_aware_k = synapse_options["reactivation_sigma_aware_k"]
        dbl_init_method = synapse_options["dbl_init_method"]
        reactivation_strategy = synapse_options["reactivation_strategy"]
        blocklinear_strategy = synapse_options["blocklinear_strategy"]
        efficient_blocklinear = synapse_options["efficient_blocklinear"]
        print_hooks = synapse_options["print_hooks"]
        use_noise = synapse_options["use_noise"]
        rewiring_mode = synapse_options["rewiring_mode"]
        rewire_frequency = synapse_options["rewire_frequency"]
        synapses_per_branch = synapse_options["synapses_per_branch"]
        sigma = synapse_options["sigma"]
        excitatory_target_density = synapse_options["excitatory_target_density"]
        inhibitory_target_density = synapse_options["inhibitory_target_density"]
        freeze_excitatory_connectivity = synapse_options[
            "freeze_excitatory_connectivity"
        ]
        freeze_inhibitory_connectivity = synapse_options[
            "freeze_inhibitory_connectivity"
        ]
        init_method = synapse_options["init_method"]
        weight_threshold = synapse_options["weight_threshold"]
        adaptive_initialization = synapse_options["adaptive_initialization"]
        adaptive_initialization_policy = synapse_options[
            "adaptive_initialization_policy"
        ]
        adaptive_target_conductance = synapse_options["adaptive_target_conductance"]
        branch_factors = synapse_options["branch_factors"]
        epsilon = synapse_options["epsilon"]
        weight_transform = synapse_options["weight_transform"]

        reactivate, reactivation_type = resolve_dendritic_activation(
            dendritic_activation,
            reactivate,
            reactivation_type,
        )
        self.layer_idx = layer_idx
        self.n_branches = output_dim
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.efficient_blocklinear = efficient_blocklinear
        self.use_shunting = use_shunting
        self.use_additive_normalization = use_additive_normalization
        self.dbl_init_method = dbl_init_method
        self.epsilon = epsilon  # Store epsilon for numerical stability
        self.topk_type = topk_type  # Store the strategy
        self.topk_weight_norm_order = (
            topk_weight_norm_order  # Store the weight norm order
        )
        self.topk_gamma = topk_gamma  # Store the gamma
        self.topk_temperature = topk_temperature  # Store the temperature
        self.topk_ultrafast = topk_ultrafast  # Store the ultrafast flag
        self.topk_strategy = topk_strategy  # Store the strategy
        self.dense_to_sparse_initial_density = dense_to_sparse_initial_density
        self.dense_to_sparse_initial_k = dense_to_sparse_initial_k
        self.dense_to_sparse_start_step = dense_to_sparse_start_step
        self.dense_to_sparse_end_step = dense_to_sparse_end_step
        self.dense_to_sparse_update_interval = dense_to_sparse_update_interval
        self.dense_to_sparse_schedule = dense_to_sparse_schedule
        self.dense_to_sparse_freeze_on_end = dense_to_sparse_freeze_on_end
        self.dense_to_sparse_advance_on_forward = dense_to_sparse_advance_on_forward
        self.dense_to_sparse_prune_metric = dense_to_sparse_prune_metric
        self.indexed_seed = indexed_seed
        self.indexed_output_chunk_size = indexed_output_chunk_size
        self.indexed_candidate_size = indexed_candidate_size
        self.indexed_selection = indexed_selection
        self.indexed_rewire_frequency = int(indexed_rewire_frequency)
        self.indexed_rewire_quantile = float(indexed_rewire_quantile)
        self.indexed_rewire_until_step = indexed_rewire_until_step
        self.input_excitatory = excitatory_input_dim is not None
        self.print_hooks = print_hooks

        # DEEPST specific parameters, clean up later
        self.use_noise = use_noise
        self.rewiring_mode = rewiring_mode
        self.rewire_frequency = rewire_frequency
        self.synapses_per_branch = synapses_per_branch
        self.sigma = sigma
        self.excitatory_target_density = excitatory_target_density
        self.inhibitory_target_density = inhibitory_target_density
        self.freeze_excitatory_connectivity = freeze_excitatory_connectivity
        self.freeze_inhibitory_connectivity = freeze_inhibitory_connectivity
        self.init_method = init_method
        self.weight_threshold = weight_threshold
        self.adaptive_initialization = adaptive_initialization
        self.adaptive_initialization_policy = adaptive_initialization_policy
        self.adaptive_target_conductance = adaptive_target_conductance
        self.branch_factors = branch_factors  # Store branch_factors
        self.weight_transform = weight_transform
        self.dendritic_activation = reactivation_type if reactivate else "none"

        gradient_scaler = GradientScaler(
            topk_strategy=topk_strategy,
            reactivation_strategy=reactivation_strategy,
            blocklinear_strategy=blocklinear_strategy,
            layer_idx=layer_idx,
            print_hooks=print_hooks,
        )

        if self.input_excitatory:
            self.branch_excitation = self._create_topk_layer(
                in_features=excitatory_input_dim,
                out_features=output_dim,
                K=excitatory_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                synapse_type="exc",
                connection_mask=excitatory_connection_mask,
            )
            gradient_scaler.register_topk_dynamic(self.branch_excitation)
            self.excitatory_input_dim = excitatory_input_dim
        else:
            self.branch_excitation = None

        # Determine if this layer actually USES inhibitory synapses
        # Note: inhibitory_input_dim can be set even when inhibitory_synapses_per_branch=0
        # This happens when:
        #   - input_mode=1: external inhibitory input available but layer doesn't use it
        #   - Multi-layer networks: inhibitory input passes through but not all layers use it
        # The condition inhibitory_synapses_per_branch > 0 ensures we only create
        # branch_inhibition when this layer actively processes inhibitory input
        self.input_inhibitory = (
            inhibitory_input_dim is not None
            and inhibitory_synapses_per_branch is not None
            and inhibitory_synapses_per_branch > 0
        )
        if self.input_inhibitory:
            self.branch_inhibition = self._create_topk_layer(
                in_features=inhibitory_input_dim,
                out_features=output_dim,
                K=inhibitory_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                synapse_type="inh",
                connection_mask=inhibitory_connection_mask,
            )
            gradient_scaler.register_topk_dynamic(self.branch_inhibition)
            self.inhibitory_input_dim = inhibitory_input_dim
        else:
            self.branch_inhibition = None

        # Recurrent excitatory synapses (optional, for RNN usage).
        # When recurrent_input_dim is None (default for all feedforward code),
        # no layer is created and no extra computation occurs.
        self.input_recurrent = (
            recurrent_input_dim is not None
            and recurrent_synapses_per_branch is not None
            and recurrent_synapses_per_branch > 0
        )
        if self.input_recurrent:
            self.branch_recurrent = self._create_topk_layer(
                in_features=recurrent_input_dim,
                out_features=output_dim,
                K=recurrent_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                synapse_type="exc",
                forbidden_input_index_per_output=recurrent_forbidden_input_index_per_output,
                connection_mask=recurrent_connection_mask,
            )
            gradient_scaler.register_topk_dynamic(self.branch_recurrent)
            self.recurrent_input_dim = recurrent_input_dim
        else:
            self.branch_recurrent = None

        # Recurrent inhibitory synapses (optional, for RNN usage with local I pool).
        # When rec_inhibitory_input_dim is None (default for all feedforward code),
        # no layer is created and no extra computation occurs.
        self.input_rec_inhibitory = (
            rec_inhibitory_input_dim is not None
            and rec_inhibitory_synapses_per_branch is not None
            and rec_inhibitory_synapses_per_branch > 0
        )
        if self.input_rec_inhibitory:
            self.branch_rec_inhibition = self._create_topk_layer(
                in_features=rec_inhibitory_input_dim,
                out_features=output_dim,
                K=rec_inhibitory_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                synapse_type="inh",
                forbidden_input_index_per_output=rec_inhibitory_forbidden_input_index_per_output,
                connection_mask=rec_inhibitory_connection_mask,
            )
            gradient_scaler.register_topk_dynamic(self.branch_rec_inhibition)
            self.rec_inhibitory_input_dim = rec_inhibitory_input_dim
        else:
            self.branch_rec_inhibition = None

        self.input_branches = input_branch_factor is not None
        if self.input_branches:
            BlockCls = (
                EfficientBlockLinear if self.efficient_blocklinear else BlockLinear
            )
            self.branches_to_output = BlockCls(
                output_dim * input_branch_factor,
                output_dim,
                weight_transform=self.weight_transform,
            )
            gradient_scaler.register_block_linear_dynamic(self.branches_to_output)
        self.input_branch_factor = input_branch_factor
        self.reactivation_init_m = reactivation_init_m
        self.reactivation_init_b = reactivation_init_b
        self.reactivation_init_policy = normalize_reactivation_init_policy(
            reactivation_init_policy
        )
        self.reactivation_occupancy_quantile_low = reactivation_occupancy_quantile_low
        self.reactivation_occupancy_quantile_high = reactivation_occupancy_quantile_high
        self.reactivation_occupancy_target_low = reactivation_occupancy_target_low
        self.reactivation_occupancy_target_high = reactivation_occupancy_target_high
        self.reactivation_calibration_min_quantile_width = (
            reactivation_calibration_min_quantile_width
        )
        self.reactivation_calibration_max_m = reactivation_calibration_max_m
        self.reactivation_calibration_revert_on_invalid = (
            reactivation_calibration_revert_on_invalid
        )
        self.reactivation_sigma_aware_k = reactivation_sigma_aware_k

        if reactivate:
            # Construct with YAML-supplied values; initialize() will overwrite
            # them based on reactivation_init_policy and dbl_init_method.
            self.reactivation = ActivationFactory.create(
                act_type=reactivation_type,
                output_dim=output_dim,
                init_m=reactivation_init_m,
                init_b=reactivation_init_b,
                **kwargs,
            )
            if use_shunting:
                gradient_scaler.register_reactivation_inverse(self.reactivation)
            # gradient_scaler.register_reactivation_inverse(self.reactivation)
        else:
            self.reactivation = nn.Identity()
        self.reactivate = reactivate

        self.set_forward_dynamic_grad_scaling(False)
        self.initialize()

    def set_forward_dynamic_grad_scaling(self, enabled: bool) -> None:
        """Opt TopK sublayers into per-forward parameter-gradient hooks."""

        for attr in (
            "branch_excitation",
            "branch_inhibition",
            "branch_recurrent",
            "branch_rec_inhibition",
        ):
            layer = getattr(self, attr, None)
            if layer is not None and hasattr(
                layer, "_use_forward_dynamic_grad_scaling"
            ):
                layer._use_forward_dynamic_grad_scaling = bool(enabled)

    def _create_topk_layer(
        self,
        in_features,
        out_features,
        K,
        init_method,
        noise_level,
        synapse_type="exc",
        forbidden_input_index_per_output=None,
        connection_mask=None,
    ):
        """Helper function that picks which topk class to use based on strategy."""
        # Keep invalid sweep configs runnable by capping requested synapse count to
        # available presynaptic features for this layer.
        if K is not None and K > in_features:
            K = in_features

        from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.factory import (
            get_sparse_layer,
            normalize_sparse_layer_type,
        )

        layer_type = normalize_sparse_layer_type(self.topk_type)

        if layer_type == "standard":
            return get_sparse_layer(
                layer_type,
                in_features,
                out_features,
                K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
                weight_transform=self.weight_transform,
                weight_norm_order=self.topk_weight_norm_order,
                gamma=self.topk_gamma if synapse_type == "inh" else 1.0,
                forbidden_input_index_per_output=forbidden_input_index_per_output,
                connection_mask=connection_mask,
            )

        # Alternative strategies only incur overhead if explicitly requested
        elif layer_type == "stochastic":
            # Get temperature parameter if available
            temperature = getattr(self, "topk_temperature", 0.5)
            # Check if ultrafast mode is enabled
            ultrafast = getattr(self, "topk_ultrafast", False)

            return get_sparse_layer(
                layer_type,
                in_features,
                out_features,
                K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
                temperature=temperature,
                ultrafast=ultrafast,
                weight_transform=self.weight_transform,
                forbidden_input_index_per_output=forbidden_input_index_per_output,
                connection_mask=connection_mask,
            )
        elif layer_type == "variance":
            return get_sparse_layer(
                layer_type,
                in_features,
                out_features,
                K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
                weight_transform=self.weight_transform,
                forbidden_input_index_per_output=forbidden_input_index_per_output,
                connection_mask=connection_mask,
            )
        elif layer_type in {"dense_to_sparse", "annealed_topk"}:
            return get_sparse_layer(
                layer_type,
                in_features,
                out_features,
                K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
                weight_transform=self.weight_transform,
                weight_norm_order=self.topk_weight_norm_order,
                gamma=self.topk_gamma if synapse_type == "inh" else 1.0,
                forbidden_input_index_per_output=forbidden_input_index_per_output,
                connection_mask=connection_mask,
                initial_density=self.dense_to_sparse_initial_density,
                initial_k=self.dense_to_sparse_initial_k,
                start_step=self.dense_to_sparse_start_step,
                end_step=self.dense_to_sparse_end_step,
                update_interval=self.dense_to_sparse_update_interval,
                schedule=self.dense_to_sparse_schedule,
                freeze_on_end=self.dense_to_sparse_freeze_on_end,
                advance_on_forward=self.dense_to_sparse_advance_on_forward,
                prune_metric=self.dense_to_sparse_prune_metric,
            )
        elif layer_type == "deepst":
            self.step_counter = 0
            return get_sparse_layer(
                layer_type,
                in_features,
                out_features,
                K,
                target_density=(
                    self.excitatory_target_density
                    if synapse_type == "exc"
                    else self.inhibitory_target_density
                ),
                sigma=self.sigma,
                use_noise=self.use_noise,
                rewiring_mode=self.rewiring_mode,
                synapses_per_branch=self.synapses_per_branch,
                param_space="log",
                init_method=self.init_method,
                freeze_connectivity=(
                    self.freeze_excitatory_connectivity
                    if synapse_type == "exc"
                    else self.freeze_inhibitory_connectivity
                ),
                weight_threshold=self.weight_threshold,
                weight_transform=self.weight_transform,
                forbidden_input_index_per_output=forbidden_input_index_per_output,
                connection_mask=connection_mask,
            )
        elif layer_type in {"indexed", "indexed_rewire"}:
            seed = None
            if self.indexed_seed is not None:
                synapse_offset = {"exc": 0, "inh": 1}.get(synapse_type, 2)
                seed = (
                    int(self.indexed_seed)
                    + (1009 * int(self.layer_idx))
                    + synapse_offset
                )
            extra_kwargs = {}
            if layer_type == "indexed_rewire":
                extra_kwargs = {
                    "rewire_frequency": self.indexed_rewire_frequency,
                    "rewire_quantile": self.indexed_rewire_quantile,
                    "rewire_until_step": self.indexed_rewire_until_step,
                }
            return get_sparse_layer(
                layer_type,
                in_features,
                out_features,
                K,
                param_space="log",
                init_method=init_method,
                weight_transform=self.weight_transform,
                weight_norm_order=self.topk_weight_norm_order,
                gamma=self.topk_gamma if synapse_type == "inh" else 1.0,
                connection_mask=connection_mask,
                forbidden_input_index_per_output=forbidden_input_index_per_output,
                seed=seed,
                output_chunk_size=self.indexed_output_chunk_size,
                **extra_kwargs,
            )
        elif layer_type == "indexed_dynamic":
            seed = None
            if self.indexed_seed is not None:
                synapse_offset = {"exc": 0, "inh": 1}.get(synapse_type, 2)
                seed = (
                    int(self.indexed_seed)
                    + (1009 * int(self.layer_idx))
                    + synapse_offset
                )
            return get_sparse_layer(
                layer_type,
                in_features,
                out_features,
                K,
                candidate_size=self.indexed_candidate_size,
                selection=self.indexed_selection,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
                temperature=self.topk_temperature,
                ultrafast=self.topk_ultrafast,
                weight_transform=self.weight_transform,
                weight_norm_order=self.topk_weight_norm_order,
                gamma=self.topk_gamma if synapse_type == "inh" else 1.0,
                connection_mask=connection_mask,
                forbidden_input_index_per_output=forbidden_input_index_per_output,
                seed=seed,
                output_chunk_size=self.indexed_output_chunk_size,
            )
        return get_sparse_layer(
            layer_type,
            in_features,
            out_features,
            K,
            param_space="log",
            init_method=init_method,
            noise_level=noise_level,
            weight_transform=self.weight_transform,
            weight_norm_order=self.topk_weight_norm_order,
            gamma=self.topk_gamma if synapse_type == "inh" else 1.0,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
        )

    def initialize(self):
        if (self.weight_transform or "").lower() == "identity":
            identity_weighttransform_dbl_init(self)

        elif self.dbl_init_method == "analytical_expectation":
            analytical_expectation_dbl_init(
                self,
                adaptive=self.adaptive_initialization,
                branch_factors=self.branch_factors,
            )
        elif self.dbl_init_method == "ei_equivalence":
            ei_equivalence_dbl_init(self)
        elif self.dbl_init_method == "naive":
            naive_dbl_init(self)
        elif self.dbl_init_method == "default":
            default_dbl_init(self)
        else:
            raise ValueError(f"Unknown dbl_init_method: {self.dbl_init_method}")

    def decay_weights(self, weight_decay, weight_boosting=False):
        if self.branch_excitation is not None:
            self.branch_excitation.decay_weights(weight_decay, weight_boosting)
        if self.branch_inhibition is not None:
            self.branch_inhibition.decay_weights(weight_decay, weight_boosting)
        if self.branch_recurrent is not None:
            self.branch_recurrent.decay_weights(weight_decay, weight_boosting)
        if self.branch_rec_inhibition is not None:
            self.branch_rec_inhibition.decay_weights(weight_decay, weight_boosting)
        # if self.input_branches:
        #     self.branches_to_output.decay_weights(weight_decay)

    # DEEPST specific method
    def apply_rewiring(self):
        """
        Apply rewiring to all synaptic connections.
        """

        def _apply_synapse_rewiring(layer):
            if layer is None:
                return
            if rewire_condition and hasattr(
                layer, "apply_noise_and_enforce_constraints"
            ):
                layer.apply_noise_and_enforce_constraints()
            elif hasattr(layer, "apply_rewiring"):
                layer.apply_rewiring()

        if not hasattr(self, "step_counter"):
            self.step_counter = 0
        self.step_counter += 1
        rewire_condition = self.step_counter % self.rewire_frequency == 0
        _apply_synapse_rewiring(self.branch_excitation)
        _apply_synapse_rewiring(self.branch_inhibition)
        _apply_synapse_rewiring(self.branch_recurrent)
        _apply_synapse_rewiring(self.branch_rec_inhibition)

    def forward(
        self,
        x,
        inhibitory_input=None,
        branch_input=None,
        recurrent_input=None,
        rec_inhibitory_input=None,
    ):
        excitation = None
        inhibition = None
        branch_current = None

        if self.use_shunting:
            numerator = 0
            denominator = 1

            if self.input_excitatory and self.branch_excitation is not None:
                excitation = self.branch_excitation(x)
                numerator = numerator + excitation
                denominator = denominator + excitation

            if (
                self.input_recurrent
                and self.branch_recurrent is not None
                and recurrent_input is not None
            ):
                rec_excitation = self.branch_recurrent(recurrent_input)
                numerator = numerator + rec_excitation
                denominator = denominator + rec_excitation

            if self.input_branches and branch_input is not None:
                branch_current = self.branches_to_output(branch_input)
                numerator = numerator + branch_current
                conductance = self.branches_to_output.sum_conductances()
                denominator = denominator + conductance

            if (
                self.input_inhibitory
                and self.branch_inhibition is not None
                and inhibitory_input is not None
            ):
                inhibition = self.branch_inhibition(inhibitory_input)
                denominator = denominator + inhibition

            if (
                self.input_rec_inhibitory
                and self.branch_rec_inhibition is not None
                and rec_inhibitory_input is not None
            ):
                rec_inhibition = self.branch_rec_inhibition(rec_inhibitory_input)
                denominator = denominator + rec_inhibition

            voltage = numerator / (denominator + self.epsilon)

            # Store for diagnostic access (no-op during normal training)
            if getattr(self, "_store_diagnostics", False):
                self._diag_g_tot = denominator.detach()
                self._diag_numerator = (
                    numerator.detach() if torch.is_tensor(numerator) else None
                )

        else:
            voltage = 0

            if self.input_excitatory and self.branch_excitation is not None:
                excitation = self.branch_excitation(x)
                voltage = voltage + excitation

            if (
                self.input_recurrent
                and self.branch_recurrent is not None
                and recurrent_input is not None
            ):
                voltage = voltage + self.branch_recurrent(recurrent_input)

            if self.input_branches and branch_input is not None:
                branch_current = self.branches_to_output(branch_input)
                voltage = voltage + branch_current

            if (
                self.input_inhibitory
                and self.branch_inhibition is not None
                and inhibitory_input is not None
            ):
                inhibition = self.branch_inhibition(inhibitory_input)
                voltage = voltage - inhibition

            if (
                self.input_rec_inhibitory
                and self.branch_rec_inhibition is not None
                and rec_inhibitory_input is not None
            ):
                voltage = voltage - self.branch_rec_inhibition(rec_inhibitory_input)

            # Optional voltage normalization for additive mode
            # (control experiment: tests whether normalization alone recovers
            # the gradient fidelity benefits of shunting inhibition)
            voltage = self.normalize_additive_voltage(voltage)

        if self.training and "conductance_dynamic" in [
            self.topk_strategy,
            self.blocklinear_strategy,
        ]:
            self.compute_grad_scales(g_total=denominator if self.use_shunting else None)

        result = self.reactivation(voltage)
        if getattr(self, "_store_analysis_currents", False):
            self._last_analysis_currents = {
                "excitation": excitation,
                "inhibition": inhibition,
                "branch_input": branch_current,
            }
        return result

    def compute_raw_currents(
        self, x, inhibitory_input=None, recurrent_input=None, rec_inhibitory_input=None
    ):
        """Compute raw synaptic currents without shunting or reactivation.

        Used by the RNN cell to separate current computation from temporal
        integration and shunting. Feedforward code should use forward() instead.

        Returns:
            raw_E: FF excitatory current [batch, output_dim] or 0
            raw_E_rec: Recurrent excitatory current [batch, output_dim] or 0
            raw_I: FF inhibitory current [batch, output_dim] or 0
            raw_I_rec: Recurrent inhibitory current [batch, output_dim] or 0
        """
        raw_E = 0
        if self.input_excitatory and self.branch_excitation is not None:
            raw_E = self.branch_excitation(x)

        raw_E_rec = 0
        if (
            self.input_recurrent
            and self.branch_recurrent is not None
            and recurrent_input is not None
        ):
            raw_E_rec = self.branch_recurrent(recurrent_input)

        raw_I = 0
        if (
            self.input_inhibitory
            and self.branch_inhibition is not None
            and inhibitory_input is not None
        ):
            raw_I = self.branch_inhibition(inhibitory_input)

        raw_I_rec = 0
        if (
            self.input_rec_inhibitory
            and self.branch_rec_inhibition is not None
            and rec_inhibitory_input is not None
        ):
            raw_I_rec = self.branch_rec_inhibition(rec_inhibitory_input)

        return raw_E, raw_E_rec, raw_I, raw_I_rec

    def normalize_additive_voltage(self, voltage):
        """Normalize additive voltages safely across output features."""
        if not self.use_additive_normalization or not torch.is_tensor(voltage):
            return voltage
        centered = voltage - voltage.mean(dim=-1, keepdim=True)
        std = voltage.std(dim=-1, keepdim=True, unbiased=False)
        return centered / std.clamp_min(self.epsilon)

    def voltage_from_currents(
        self,
        trace_E,
        trace_E_rec,
        trace_I,
        trace_I_rec=0,
        trace_branch=0,
        branch_conductance=None,
    ):
        """Compute pre-reactivation voltage from integrated recurrent traces.

        Branch aggregation mirrors the feedforward shunting equation: branch
        current contributes to the numerator, while branch coupling conductance
        contributes separately to the denominator when provided.
        """
        denominator = None
        if self.use_shunting:
            numerator = trace_E + trace_E_rec
            if torch.is_tensor(trace_branch):
                numerator = numerator + trace_branch
            elif trace_branch:
                numerator = numerator + trace_branch

            denominator = 1 + trace_E
            if torch.is_tensor(trace_E_rec):
                denominator = denominator + trace_E_rec
            if torch.is_tensor(trace_I):
                denominator = denominator + trace_I
            if torch.is_tensor(trace_I_rec):
                denominator = denominator + trace_I_rec
            if branch_conductance is not None:
                denominator = denominator + branch_conductance
            voltage = numerator / (denominator + self.epsilon)
        else:
            voltage = trace_E + trace_E_rec - trace_I - trace_I_rec
            if torch.is_tensor(trace_branch):
                voltage = voltage + trace_branch
            elif trace_branch:
                voltage = voltage + trace_branch
            voltage = self.normalize_additive_voltage(voltage)
        return voltage, denominator

    def shunt_from_currents(
        self,
        trace_E,
        trace_E_rec,
        trace_I,
        trace_I_rec=0,
        trace_branch=0,
        branch_conductance=None,
    ):
        """Apply shunting (or additive) computation on pre-integrated currents.

        Used by the RNN cell after temporal integration of traces.
        The traces are the leaky-integrated versions of raw currents.

        Args:
            trace_E: Integrated FF excitatory trace [batch, output_dim]
            trace_E_rec: Integrated recurrent excitatory trace (can be 0)
            trace_I: Integrated FF inhibitory trace (can be 0)
            trace_I_rec: Integrated recurrent inhibitory trace (can be 0)

        Returns:
            voltage: Post-shunting, post-reactivation output [batch, output_dim]
        """
        voltage, _denominator = self.voltage_from_currents(
            trace_E=trace_E,
            trace_E_rec=trace_E_rec,
            trace_I=trace_I,
            trace_I_rec=trace_I_rec,
            trace_branch=trace_branch,
            branch_conductance=branch_conductance,
        )
        return self.reactivation(voltage)

    def compute_grad_scales(
        self,
        g_total: Optional[torch.Tensor] = None,
        *,
        use_forward_hooks: bool = False,
    ):
        """
        A conduction-based dynamic scale, returning a flattened vector so each weight
        can get a distinct multiplier. shape [out_features * block_size].
        """
        with torch.no_grad():
            if self.use_shunting:
                if g_total.dim() > 1:
                    g_total = g_total.mean(dim=tuple(range(g_total.dim() - 1)))

                if (
                    hasattr(self, "branches_to_output")
                    and self.blocklinear_strategy == "conductance_dynamic"
                ):
                    weight = self.branches_to_output.weight().clamp_min(self.epsilon)
                    input_scale = (g_total[:, None] / weight).flatten()
                    param_scale = g_total[:, None]
                    if use_forward_hooks:
                        self.branches_to_output.register_forward_gradient_scales(
                            param_scale=param_scale,
                            input_scale=input_scale,
                        )
                    else:
                        self.branches_to_output.input_scale_vec = input_scale
                        self.branches_to_output.param_scale_vec = param_scale

                if (
                    self.branch_excitation is not None
                    and self.topk_strategy == "conductance_dynamic"
                ):
                    param_scale = g_total[:, None]
                    if use_forward_hooks and hasattr(
                        self.branch_excitation, "register_forward_param_gradient_scale"
                    ):
                        self.branch_excitation.register_forward_param_gradient_scale(
                            param_scale
                        )
                    else:
                        self.branch_excitation.param_scale_vec = param_scale

                if (
                    self.branch_inhibition is not None
                    and self.topk_strategy == "conductance_dynamic"
                ):
                    param_scale = g_total[:, None]
                    if use_forward_hooks and hasattr(
                        self.branch_inhibition, "register_forward_param_gradient_scale"
                    ):
                        self.branch_inhibition.register_forward_param_gradient_scale(
                            param_scale
                        )
                    else:
                        self.branch_inhibition.param_scale_vec = param_scale

                if (
                    self.branch_recurrent is not None
                    and self.topk_strategy == "conductance_dynamic"
                ):
                    param_scale = g_total[:, None]
                    if use_forward_hooks and hasattr(
                        self.branch_recurrent, "register_forward_param_gradient_scale"
                    ):
                        self.branch_recurrent.register_forward_param_gradient_scale(
                            param_scale
                        )
                    else:
                        self.branch_recurrent.param_scale_vec = param_scale

                if (
                    self.branch_rec_inhibition is not None
                    and self.topk_strategy == "conductance_dynamic"
                ):
                    param_scale = g_total[:, None]
                    if use_forward_hooks and hasattr(
                        self.branch_rec_inhibition,
                        "register_forward_param_gradient_scale",
                    ):
                        self.branch_rec_inhibition.register_forward_param_gradient_scale(
                            param_scale
                        )
                    else:
                        self.branch_rec_inhibition.param_scale_vec = param_scale

            else:
                if (
                    hasattr(self, "branches_to_output")
                    and self.blocklinear_strategy == "conductance_dynamic"
                ):
                    # only apply input_scale_vec = 1 / branch_weights
                    w = self.branches_to_output.weight()

                    if (self.weight_transform or "").lower() == "identity":
                        w = w.abs() + self.epsilon

                    input_scale = torch.exp(-1 * w.log()).flatten()
                    if use_forward_hooks:
                        self.branches_to_output.register_forward_gradient_scales(
                            input_scale=input_scale
                        )
                    else:
                        self.branches_to_output.input_scale_vec = input_scale

    def _collect_from_layers(self, method_name):
        """Helper to collect a named attribute from all active TopK layers."""
        result = {}
        if self.branch_excitation is not None:
            result["exc"] = getattr(self.branch_excitation, method_name)()
        if self.branch_recurrent is not None:
            result["rec_exc"] = getattr(self.branch_recurrent, method_name)()
        if self.branch_inhibition is not None:
            result["inh"] = getattr(self.branch_inhibition, method_name)()
        if self.branch_rec_inhibition is not None:
            result["rec_inh"] = getattr(self.branch_rec_inhibition, method_name)()
        return result if result else None

    def get_weights(self):
        return self._collect_from_layers("weight")

    def get_log_weights(self):
        return self._collect_from_layers("log_weight")

    def get_mask(self):
        return self._collect_from_layers("weight_mask")

    def get_pruned_weights(self):
        return self._collect_from_layers("pruned_weight")

    def get_log_pruned_weights(self):
        return self._collect_from_layers("log_pruned_weight")


__all__ = ["DendriticBranchLayer"]
