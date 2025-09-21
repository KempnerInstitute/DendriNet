"""
Dendritic Branch Layer for neural networks.

This module implements the DendriticBranchLayer which emulates dendritic
computation along a layer of branches, modeling the interaction between
excitatory and inhibitory inputs.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional

from dendritic_modeling.networks.activations import ActivationFactory
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
    EfficientBlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.gradient_scaling import (
    GradientScaler,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    analytical_expectation_dbl_init,
    default_dbl_init,
    ei_equivalence_dbl_init,
    naive_dbl_init,
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
        The number of branches in this layer.

    excitatory_input_dim : int, optional
        The dimension of the excitatory input. Defaults to `None`.

    inhibitory_input_dim : int, optional
        The dimension of the inhibitory input. Defaults to `None`.

    excitatory_synapses_per_branch : int, optional
        The number of excitatory synapses per branch. Defaults to `None`.

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
        weight_transform="exp",  # Weight transformation for positive weights
        topk_temperature=0.5,  # For stochastic TopK
        topk_ultrafast=False,  # For much faster stochastic TopK
        topk_strategy="none",
        use_shunting=True,
        reactivate=False,
        reactivation_type="param_tanh",
        reactivation_init_m=1,
        reactivation_init_b=0.5,
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
        branch_factors=None,  # Add branch_factors parameter
        epsilon=1e-8,  # Add configurable epsilon for numerical stability
        **kwargs,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.efficient_blocklinear = efficient_blocklinear
        self.dbl_init_method = dbl_init_method
        self.use_shunting = use_shunting
        self.print_hooks = print_hooks
        self.epsilon = epsilon  # Store epsilon for numerical stability
        self.weight_transform = weight_transform
        self.branch_factors = branch_factors
        self.adaptive_initialization = adaptive_initialization
        gradient_scaler = GradientScaler(
            topk_strategy=topk_strategy,
            reactivation_strategy=reactivation_strategy,
            blocklinear_strategy=blocklinear_strategy,
            layer_idx=layer_idx,
            print_hooks=print_hooks,
        )

        # TopK strategy-related configs
        self.topk_strategy = topk_strategy

        # DEEPST specific attributes
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

        # Add topk_type attribute
        self.topk_type = topk_type

        # get information from the constructor
        self.output_dim = output_dim
        self.excitatory_input_dim = excitatory_input_dim
        self.inhibitory_input_dim = inhibitory_input_dim
        self.excitatory_synapses_per_branch = excitatory_synapses_per_branch
        self.inhibitory_synapses_per_branch = inhibitory_synapses_per_branch

        self.input_excitatory = excitatory_input_dim is not None
        if self.input_excitatory:
            self.branch_excitation = self._create_topk_layer(
                in_features=excitatory_input_dim,
                out_features=output_dim,
                K=excitatory_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                temperature=topk_temperature,
                ultrafast=topk_ultrafast,
            )
            gradient_scaler.register_topk_dynamic(self.branch_excitation)
            self.excitatory_input_dim = excitatory_input_dim
        else:
            self.branch_excitation = None

        self.input_inhibitory = inhibitory_input_dim is not None
        self.input_inhibitory = self.input_inhibitory and (
            inhibitory_synapses_per_branch > 0
        )
        if self.input_inhibitory:
            self.branch_inhibition = self._create_topk_layer(
                in_features=inhibitory_input_dim,
                out_features=output_dim,
                K=inhibitory_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                temperature=topk_temperature,
                ultrafast=topk_ultrafast,
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
                output_dim * input_branch_factor,
                output_dim,
                weight_transform=self.weight_transform,
            )
            gradient_scaler.register_block_linear_dynamic(self.branches_to_output)
        self.input_branch_factor = input_branch_factor

        # Add activation for reactivation: instantiate whenever reactivate is True
        if reactivate:
            self.reactivation = ActivationFactory.create(
                act_type=reactivation_type,
                output_dim=output_dim,
                init_m=reactivation_init_m,
                init_b=reactivation_init_b,
                **kwargs,
            )
            if self.use_shunting:
                gradient_scaler.register_reactivation_inverse(self.reactivation)
        else:
            self.reactivation = nn.Identity()

        # Initialize parameters immediately to match volt behavior
        self.initialize()
        self.initialized = True

    def _compute_einet_scaling_factors(self):
        """
        Compute and set scaling factors for gradient normalization based on network configuration.
        Handles different scaling strategies for blocklinear layers.
        """
        with torch.no_grad():
            # Get conductances for all branches
            branch_conductances = torch.zeros(self.output_dim)

            # Compute excitatory conductances
            if self.input_excitatory and self.branch_excitation is not None:
                # TopK layer weights are always positive due to weight transform
                exc_weights = self.branch_excitation.weight()  # [out, in]
                # Sum conductances for each output branch
                branch_conductances += exc_weights.sum(dim=1)

            # Apply scaling for blocklinear layers if they exist
            if self.blocklinear_strategy != "none" and hasattr(
                self, "branches_to_output"
            ):
                if self.blocklinear_strategy == "branch_conductance":
                    # Scale by total branch conductance
                    g_branch = branch_conductances.mean()
                    self.branches_to_output.input_scale_vec = g_branch.expand(
                        self.branches_to_output.in_features
                    )
                    self.branches_to_output.param_scale_vec = g_branch.expand(
                        self.branches_to_output.out_features
                    )
                elif self.blocklinear_strategy == "branch_conductance_dynamic":
                    # Dynamic scaling per branch
                    # Each block weight gets scaled by the conductance of its corresponding output branch
                    g_per_branch = branch_conductances  # [out_features]
                    # Repeat for each block element
                    scale_vec = g_per_branch.repeat_interleave(
                        self.branches_to_output.block_size
                    )
                    self.branches_to_output.input_scale_vec = scale_vec
                    self.branches_to_output.param_scale_vec = g_per_branch

    def _create_topk_layer(
        self,
        in_features,
        out_features,
        K,
        init_method,
        noise_level,
        temperature=0.5,
        ultrafast=False,
    ):
        """Create a TopK layer based on the selected type."""
        if self.topk_type == "standard":
            return TopKLinear(
                in_features=in_features,
                out_features=out_features,
                K=K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
                weight_transform=self.weight_transform,
            )
        elif self.topk_type == "deepst":
            return DeepstLinear(
                in_features=in_features,
                out_features=out_features,
                K=K if self.synapses_per_branch is None else self.synapses_per_branch,
                use_noise=self.use_noise,
                rewiring_mode=self.rewiring_mode,
                rewire_frequency=self.rewire_frequency,
                init_method=self.init_method,
                init_gain=1.0,
                sigma=self.sigma,
                weight_threshold=self.weight_threshold,
            )
        elif self.topk_type == "stochastic":
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
                weight_transform=self.weight_transform,
            )
        else:
            raise ValueError(f"Unknown topk_type: {self.topk_type}")

    def initialize(self):
        # call the correct initialization function according to dbl_init_method
        if self.dbl_init_method == "analytical_expectation":
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
        # if self.input_branches:
        #     self.branches_to_output.decay_weights(weight_decay)

    # DEEPST specific method
    def apply_rewiring(self):
        """Apply rewiring to the synapse layers (DEEPST specific)."""
        if self.input_excitatory and hasattr(self.branch_excitation, "apply_rewiring"):
            self.branch_excitation.apply_rewiring()
        if self.input_inhibitory and hasattr(self.branch_inhibition, "apply_rewiring"):
            self.branch_inhibition.apply_rewiring()

    def forward(self, x, inhibitory_input=None, branch_input=None):
        """
        Forward pass through the dendritic branch layer.

        Args:
            x: Excitatory input tensor [batch, input_dim]
            inhibitory_input: Inhibitory input tensor [batch, input_dim]
            branch_input: Input from previous branches [batch, n_branches]

        Returns:
            Output tensor after dendritic processing
        """
        # Already initialized in __init__

        if self.use_shunting:
            numerator = 0
            denominator = 1

            if self.input_excitatory and self.branch_excitation is not None:
                excitation = self.branch_excitation(x)
                numerator = numerator + excitation
                denominator = denominator + torch.nn.functional.relu(excitation)

            if self.input_branches and branch_input is not None:
                current = self.branches_to_output(branch_input)
                numerator = numerator + current
                conductance = self.branches_to_output.sum_conductances()
                denominator = denominator + torch.nn.functional.relu(conductance)

            if (
                self.input_inhibitory
                and self.branch_inhibition is not None
                and inhibitory_input is not None
            ):
                inhibition = self.branch_inhibition(inhibitory_input)
                denominator = denominator + torch.nn.functional.relu(inhibition)

            voltage = numerator / (denominator + self.epsilon)

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

        result = self.reactivation(voltage)
        return result

    def compute_grad_scales(self, g_total: Optional[torch.Tensor] = None):
        """
        A conduction-based dynamic scale, returning a flattened vector so each weight
        can get a distinct multiplier. shape [out_features * block_size].
        """
        with torch.no_grad():
            if self.use_shunting and g_total is not None:
                if g_total.dim() > 1:
                    g_total = g_total.mean(dim=tuple(range(g_total.dim() - 1)))

                if (
                    hasattr(self, "branches_to_output")
                    and self.blocklinear_strategy == "conductance_dynamic"
                ):
                    self.branches_to_output.input_scale_vec = (
                        g_total[:, None] / self.branches_to_output.weight()
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

            else:
                if (
                    hasattr(self, "branches_to_output")
                    and self.blocklinear_strategy == "conductance_dynamic"
                ):
                    # only apply input_scale_vec = 1 / branch_weights
                    self.branches_to_output.input_scale_vec = torch.exp(
                        -1 * self.branches_to_output.weight().log()
                    ).flatten()

                    # No param scaling needed
                    self.branches_to_output.param_scale_vec = torch.ones(
                        self.branches_to_output.out_features
                    )
