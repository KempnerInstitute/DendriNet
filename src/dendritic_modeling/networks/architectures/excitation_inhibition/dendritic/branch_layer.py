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
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.gradient_scaling import (
    GradientScaler,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    initialize_dendritic_branch_layer,
)

# Import linear layer variants
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    DeepstLinear,
    StochasticTopKLinear,
    TopKLinear,
    VarianceTopKLinear,
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
        output_dim,
        excitatory_input_dim=None,
        excitatory_synapses_per_branch=None,
        inhibitory_input_dim=None,
        inhibitory_synapses_per_branch=None,
        input_branch_factor=None,
        topk_init_method="xavier_normal",
        topk_noise_level=0.0,
        topk_type="standard",  # New parameter
        topk_temperature=0.5,  # For stochastic TopK
        topk_ultrafast=False,  # For much faster stochastic TopK
        topk_strategy="none",
        use_shunting=True,
        reactivate=False,
        reactivation_type="param_tanh",
        reactivation_init_m=1,
        reactivation_init_b=0.5,
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
        branch_factors=None,  # Add branch_factors parameter
        **kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.efficient_blocklinear = efficient_blocklinear
        self.use_shunting = use_shunting
        self.topk_type = topk_type  # Store the strategy
        self.topk_temperature = topk_temperature  # Store the temperature
        self.topk_ultrafast = topk_ultrafast  # Store the ultrafast flag
        self.topk_strategy = topk_strategy  # Store the strategy
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
        self.branch_factors = branch_factors  # Store branch_factors

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
                cell_type="exc",
            )
            gradient_scaler.register_topk_dynamic(self.branch_excitation)
            self.excitatory_input_dim = excitatory_input_dim
        else:
            self.branch_excitation = None

        self.input_inhibitory = inhibitory_input_dim is not None
        if self.input_inhibitory:
            self.branch_inhibition = self._create_topk_layer(
                in_features=inhibitory_input_dim,
                out_features=output_dim,
                K=inhibitory_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                cell_type="inh",
            )
            gradient_scaler.register_topk_dynamic(self.branch_inhibition)
            self.inhibitory_input_dim = inhibitory_input_dim
        else:
            self.branch_inhibition = None

        self.input_branches = input_branch_factor is not None
        if self.input_branches:
            BlockCls = (
                EfficientBlockLinear if self.efficient_blocklinear else BlockLinear
            )
            self.branches_to_output = BlockCls(
                output_dim * input_branch_factor, output_dim
            )
            gradient_scaler.register_block_linear_dynamic(self.branches_to_output)
        self.input_branch_factor = input_branch_factor

        if reactivate:
            self.reactivation = ActivationFactory.create(
                act_type=reactivation_type,
                output_dim=output_dim,
                init_m=reactivation_init_m if use_shunting else 0.1,
                init_b=reactivation_init_b if use_shunting else 0.1,
                **kwargs,
            )
            if use_shunting:
                gradient_scaler.register_reactivation_inverse(self.reactivation)
        else:
            self.reactivation = nn.Identity()
        self.reactivate = reactivate

        # if topk_type == 'deepst':
        #     initialize_dendritic_branch_layer(self)
        # else:
        #     self.initialize()
        initialize_dendritic_branch_layer(
            self,
            adaptive=self.adaptive_initialization,
            branch_factors=self.branch_factors,
        )

    def _create_topk_layer(
        self,
        in_features,
        out_features,
        K,
        init_method,
        noise_level,
        cell_type="exc",
    ):
        """Helper function that picks which topk class to use based on strategy."""
        # For 'standard' strategy, use the original TopKLinear class exactly as before
        # This provides a direct path with zero overhead compared to the original code
        if self.topk_type is None or self.topk_type == "standard":
            # Create directly without any strategy-specific logic
            topk_layer = TopKLinear(
                in_features=in_features,
                out_features=out_features,
                K=K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
            )
            return topk_layer

        # Alternative strategies only incur overhead if explicitly requested
        elif self.topk_type == "stochastic":
            # Get temperature parameter if available
            temperature = getattr(self, "topk_temperature", 0.5)
            # Check if ultrafast mode is enabled
            ultrafast = getattr(self, "topk_ultrafast", False)

            return StochasticTopKLinear(
                in_features=in_features,
                out_features=out_features,
                K=K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
                temperature=temperature,
                ultrafast=ultrafast,
            )
        elif self.topk_type == "variance":
            return VarianceTopKLinear(
                in_features=in_features,
                out_features=out_features,
                K=K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
            )
        elif self.topk_type == "deepst":
            self.step_counter = 0
            return DeepstLinear(
                in_features=in_features,
                out_features=out_features,
                target_density=(
                    self.excitatory_target_density
                    if cell_type == "exc"
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
                    if cell_type == "exc"
                    else self.freeze_inhibitory_connectivity
                ),
                weight_threshold=self.weight_threshold,
            )
        else:
            # Fallback for any unknown strategy - use standard
            return TopKLinear(
                in_features=in_features,
                out_features=out_features,
                K=K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
            )

    def initialize(self):
        with torch.no_grad():
            if self.input_excitatory and self.branch_excitation is not None:
                self.branch_excitation.initialize()
            if self.input_inhibitory and self.branch_inhibition is not None:
                self.branch_inhibition.initialize()

            if self.input_branches:
                if (
                    self.input_excitatory or self.input_inhibitory
                ) and self.use_shunting:
                    sum_synapse_conductances = 0
                    if self.input_excitatory and self.branch_excitation is not None:
                        sum_synapse_conductances += self.branch_excitation.weight().sum(
                            dim=1
                        )
                    if self.input_inhibitory and self.branch_inhibition is not None:
                        sum_synapse_conductances += self.branch_inhibition.weight().sum(
                            dim=1
                        )
                else:
                    sum_synapse_conductances = None
                self.branches_to_output.initialize(sum_synapse_conductances)

    def decay_weights(self, weight_decay):
        if self.input_excitatory and self.branch_excitation is not None:
            self.branch_excitation.decay_weights(weight_decay, weight_boost=False)
        if self.input_inhibitory and self.branch_inhibition is not None:
            self.branch_inhibition.decay_weights(weight_decay, weight_boost=False)
        # if self.input_branches:
        #     self.branches_to_output.decay_weights(weight_decay)

    # DEEPST specific method
    def apply_rewiring(self):
        """
        Apply rewiring to all synaptic connections.
        """
        if hasattr(self, "step_counter"):
            self.step_counter += 1
            rewire_condition = self.step_counter % self.rewire_frequency == 0
            if rewire_condition and hasattr(
                self.branch_excitation, "apply_noise_and_enforce_constraints"
            ):
                self.branch_excitation.apply_noise_and_enforce_constraints()
            if rewire_condition and hasattr(
                self.branch_inhibition, "apply_noise_and_enforce_constraints"
            ):
                self.branch_inhibition.apply_noise_and_enforce_constraints()

    def forward(self, x, inhibitory_input=None, branch_input=None):
        if self.use_shunting:
            numerator = 0
            denominator = 1

            if self.input_excitatory and self.branch_excitation is not None:
                excitation = self.branch_excitation(x)
                numerator = numerator + excitation
                denominator = denominator + excitation

            if self.input_branches and branch_input is not None:
                current = self.branches_to_output(branch_input)
                numerator = numerator + current
                conductance = self.branches_to_output.sum_conductances()
                denominator = denominator + conductance

            if (
                self.input_inhibitory
                and self.branch_inhibition is not None
                and inhibitory_input is not None
            ):
                denominator = denominator + self.branch_inhibition(inhibitory_input)

            voltage = numerator / denominator

        else:
            voltage = 0

            if self.input_excitatory and self.branch_excitation is not None:
                voltage = voltage + self.branch_excitation(x)

            if self.input_branches and branch_input is not None:
                voltage = voltage + self.branches_to_output(branch_input)

            if (
                self.input_inhibitory
                and self.branch_inhibition is not None
                and inhibitory_input is not None
            ):
                voltage = voltage - self.branch_inhibition(inhibitory_input)

        if self.training and "conductance_dynamic" in [
            self.topk_strategy,
            self.blocklinear_strategy,
        ]:
            self.compute_grad_scales(g_total=denominator if self.use_shunting else None)

        return self.reactivation(voltage)

    def compute_grad_scales(self, g_total=None):
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
                    self.branches_to_output.input_scale_vec = (
                        g_total[:, None] / self.branches_to_output.log_weight.exp()
                    ).flatten()
                    self.branches_to_output.param_scale_vec = g_total[:, None]

                if (
                    self.branch_excitation is not None
                    and self.topk_strategy == "conductance_dynamic"
                ):
                    self.branch_excitation.input_scale_vec = (
                        g_total[:, None] / self.branch_excitation.weight()
                    ).mean(dim=0)
                    self.branch_excitation.param_scale_vec = g_total[:, None]

                if (
                    self.branch_inhibition is not None
                    and self.topk_strategy == "conductance_dynamic"
                ):
                    self.branch_inhibition.input_scale_vec = (
                        g_total[:, None] / self.branch_inhibition.weight()
                    ).mean(dim=0)
                    self.branch_inhibition.param_scale_vec = g_total[:, None]

    def get_weights(self):
        if self.input_inhibitory and self.input_excitatory:
            return (self.branch_excitation.weight(), self.branch_inhibition.weight())
        elif self.input_excitatory:
            return self.branch_excitation.weight()
        elif self.input_inhibitory:
            return self.branch_inhibition.weight()
        else:
            return None

    def get_log_weights(self):
        if self.input_inhibitory and self.input_excitatory:
            return (
                self.branch_excitation.log_weight(),
                self.branch_inhibition.log_weight(),
            )
        elif self.input_excitatory:
            return self.branch_excitation.log_weight()
        elif self.input_inhibitory:
            return self.branch_inhibition.log_weight()
        else:
            return None

    def get_mask(self):
        if self.input_inhibitory and self.input_excitatory:
            return (
                self.branch_excitation.weight_mask(),
                self.branch_inhibition.weight_mask(),
            )
        elif self.input_excitatory:
            return self.branch_excitation.weight_mask()
        elif self.input_inhibitory:
            return self.branch_inhibition.weight_mask()
        else:
            return None

    def get_pruned_weights(self):
        if self.input_inhibitory and self.input_excitatory:
            return (
                self.branch_excitation.pruned_weight(),
                self.branch_inhibition.pruned_weight(),
            )
        elif self.input_excitatory:
            return self.branch_excitation.pruned_weight()
        elif self.input_inhibitory:
            return self.branch_inhibition.pruned_weight()
        else:
            return None

    def get_log_pruned_weights(self):
        if self.input_inhibitory and self.input_excitatory:
            return (
                self.branch_excitation.log_pruned_weight(),
                self.branch_inhibition.log_pruned_weight(),
            )
        elif self.input_excitatory:
            return self.branch_excitation.log_pruned_weight()
        elif self.input_inhibitory:
            return self.branch_inhibition.log_pruned_weight()
        else:
            return None


__all__ = ["DendriticBranchLayer"]
