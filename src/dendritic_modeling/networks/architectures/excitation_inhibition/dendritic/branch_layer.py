"""
Dendritic Branch Layer for neural networks.

This module implements the DendriticBranchLayer which emulates dendritic
computation along a layer of branches, modeling the interaction between
excitatory and inhibitory inputs.
"""

import torch
import torch.nn as nn

from dendritic_modeling.networks.activations import ActivationFactory
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
    EfficientBlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_config import (
    DendriticBranchConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_dynamics import (
    compute_branch_raw_currents,
    compute_branch_voltage_from_currents,
    forward_branch_dynamics,
    normalize_branch_additive_voltage,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_grad_scales import (
    compute_branch_grad_scales,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_maintenance import (
    apply_branch_rewiring,
    collect_from_branch_synapse_layers,
    decay_branch_synapse_weights,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_options import (
    resolve_branch_config,
    resolve_branch_layer_synapse_config,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_synapses import (
    create_branch_sparse_layer,
    create_registered_branch_sparse_layer,
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
        topk_type="standard",  # TopK implementation strategy.
        topk_weight_norm_order=None,
        topk_gamma=1.0,
        topk_temperature=0.5,  # Stochastic TopK temperature.
        topk_ultrafast=False,  # Fast stochastic TopK path.
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
        # DeepST-compatible sparse-rewiring parameters.
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
        branch_factors=None,  # Branch count per dendritic level.
        epsilon=1e-8,  # Numerical stability constant.
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
        provided_branch_config = resolve_branch_config(branch_config)
        if provided_branch_config is not None:
            output_dim = provided_branch_config.output_dim
            excitatory_input_dim = provided_branch_config.excitatory_input_dim
            excitatory_synapses_per_branch = (
                provided_branch_config.excitatory_synapses_per_branch
            )
            inhibitory_input_dim = provided_branch_config.inhibitory_input_dim
            inhibitory_synapses_per_branch = (
                provided_branch_config.inhibitory_synapses_per_branch
            )
            input_branch_factor = provided_branch_config.input_branch_factor
            recurrent_input_dim = provided_branch_config.recurrent_input_dim
            recurrent_synapses_per_branch = (
                provided_branch_config.recurrent_synapses_per_branch
            )
            recurrent_forbidden_input_index_per_output = (
                provided_branch_config.recurrent_forbidden_input_index_per_output
            )
            rec_inhibitory_input_dim = provided_branch_config.rec_inhibitory_input_dim
            rec_inhibitory_synapses_per_branch = (
                provided_branch_config.rec_inhibitory_synapses_per_branch
            )
            rec_inhibitory_forbidden_input_index_per_output = (
                provided_branch_config.rec_inhibitory_forbidden_input_index_per_output
            )
            layer_idx = provided_branch_config.layer_idx
            excitatory_connection_mask = (
                provided_branch_config.excitatory_connection_mask
            )
            inhibitory_connection_mask = (
                provided_branch_config.inhibitory_connection_mask
            )
            recurrent_connection_mask = provided_branch_config.recurrent_connection_mask
            rec_inhibitory_connection_mask = (
                provided_branch_config.rec_inhibitory_connection_mask
            )
        if output_dim is None:
            raise ValueError("DendriticBranchLayer requires output_dim")
        self.branch_config = DendriticBranchConfig.from_kwargs(
            output_dim=output_dim,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            input_branch_factor=input_branch_factor,
            recurrent_input_dim=recurrent_input_dim,
            recurrent_synapses_per_branch=recurrent_synapses_per_branch,
            recurrent_forbidden_input_index_per_output=(
                recurrent_forbidden_input_index_per_output
            ),
            rec_inhibitory_input_dim=rec_inhibitory_input_dim,
            rec_inhibitory_synapses_per_branch=rec_inhibitory_synapses_per_branch,
            rec_inhibitory_forbidden_input_index_per_output=(
                rec_inhibitory_forbidden_input_index_per_output
            ),
            layer_idx=layer_idx,
            excitatory_connection_mask=excitatory_connection_mask,
            inhibitory_connection_mask=inhibitory_connection_mask,
            recurrent_connection_mask=recurrent_connection_mask,
            rec_inhibitory_connection_mask=rec_inhibitory_connection_mask,
        )
        formal_synapse_kwargs = DendriticSynapseConfig.legacy_kwargs_from_mapping(
            locals()
        )
        self.synapse_config, synapse_options, kwargs = (
            resolve_branch_layer_synapse_config(
                formal_synapse_kwargs=formal_synapse_kwargs,
                extra_kwargs=kwargs,
                synapse_config=synapse_config,
            )
        )

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

        self.layer_idx = layer_idx
        self.n_branches = output_dim
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.efficient_blocklinear = efficient_blocklinear
        self.use_shunting = use_shunting
        self.use_additive_normalization = use_additive_normalization
        self.dbl_init_method = dbl_init_method
        self.epsilon = epsilon
        self.topk_type = topk_type
        self.topk_weight_norm_order = topk_weight_norm_order
        self.topk_gamma = topk_gamma
        self.topk_temperature = topk_temperature
        self.topk_ultrafast = topk_ultrafast
        self.topk_strategy = topk_strategy
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

        # DeepST-compatible sparse-rewiring parameters.
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
        self.branch_factors = branch_factors
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
            self.branch_excitation = self._create_registered_topk_layer(
                in_features=excitatory_input_dim,
                out_features=output_dim,
                K=excitatory_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                gradient_scaler=gradient_scaler,
                synapse_type="exc",
                connection_mask=excitatory_connection_mask,
            )
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
            self.branch_inhibition = self._create_registered_topk_layer(
                in_features=inhibitory_input_dim,
                out_features=output_dim,
                K=inhibitory_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                gradient_scaler=gradient_scaler,
                synapse_type="inh",
                connection_mask=inhibitory_connection_mask,
            )
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
            self.branch_recurrent = self._create_registered_topk_layer(
                in_features=recurrent_input_dim,
                out_features=output_dim,
                K=recurrent_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                gradient_scaler=gradient_scaler,
                synapse_type="exc",
                forbidden_input_index_per_output=recurrent_forbidden_input_index_per_output,
                connection_mask=recurrent_connection_mask,
            )
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
            self.branch_rec_inhibition = self._create_registered_topk_layer(
                in_features=rec_inhibitory_input_dim,
                out_features=output_dim,
                K=rec_inhibitory_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                gradient_scaler=gradient_scaler,
                synapse_type="inh",
                forbidden_input_index_per_output=rec_inhibitory_forbidden_input_index_per_output,
                connection_mask=rec_inhibitory_connection_mask,
            )
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
        return create_branch_sparse_layer(
            self,
            in_features=in_features,
            out_features=out_features,
            K=K,
            init_method=init_method,
            noise_level=noise_level,
            synapse_type=synapse_type,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
        )

    def _create_registered_topk_layer(
        self,
        in_features,
        out_features,
        K,
        init_method,
        noise_level,
        gradient_scaler,
        synapse_type="exc",
        forbidden_input_index_per_output=None,
        connection_mask=None,
    ):
        """Create a sparse synapse layer and register gradient scaling hooks."""
        return create_registered_branch_sparse_layer(
            self,
            gradient_scaler,
            in_features=in_features,
            out_features=out_features,
            K=K,
            init_method=init_method,
            noise_level=noise_level,
            synapse_type=synapse_type,
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
        decay_branch_synapse_weights(self, weight_decay, weight_boosting)

    # Sparse-rewiring maintenance API.
    def apply_rewiring(self):
        """
        Apply rewiring to all synaptic connections.
        """
        apply_branch_rewiring(self)

    def forward(
        self,
        x,
        inhibitory_input=None,
        branch_input=None,
        recurrent_input=None,
        rec_inhibitory_input=None,
    ):
        return forward_branch_dynamics(
            self,
            x,
            inhibitory_input=inhibitory_input,
            branch_input=branch_input,
            recurrent_input=recurrent_input,
            rec_inhibitory_input=rec_inhibitory_input,
        )

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
        return compute_branch_raw_currents(
            self,
            x,
            inhibitory_input=inhibitory_input,
            recurrent_input=recurrent_input,
            rec_inhibitory_input=rec_inhibitory_input,
        )

    def normalize_additive_voltage(self, voltage):
        """Normalize additive voltages safely across output features."""
        return normalize_branch_additive_voltage(self, voltage)

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
        return compute_branch_voltage_from_currents(
            self,
            trace_E=trace_E,
            trace_E_rec=trace_E_rec,
            trace_I=trace_I,
            trace_I_rec=trace_I_rec,
            trace_branch=trace_branch,
            branch_conductance=branch_conductance,
        )

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
        g_total: torch.Tensor | None = None,
        *,
        use_forward_hooks: bool = False,
    ):
        """
        A conduction-based dynamic scale, returning a flattened vector so each weight
        can get a distinct multiplier. shape [out_features * block_size].
        """
        return compute_branch_grad_scales(
            self,
            g_total=g_total,
            use_forward_hooks=use_forward_hooks,
        )

    def _collect_from_layers(self, method_name):
        """Helper to collect a named attribute from all active TopK layers."""
        return collect_from_branch_synapse_layers(self, method_name)

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
