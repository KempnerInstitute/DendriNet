"""
Excitation-Inhibition Network.

This module implements the ExcitationInhibitionNetwork which uses separate
excitatory and inhibitory pathways with dendritic computation.
"""

from copy import deepcopy
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from omegaconf import DictConfig

# Import dynamo config for JIT compilation optimization
try:
    import torch._dynamo.config
except ImportError:
    torch._dynamo = None

from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
    ExcitationInhibitionLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.transfer import (
    TransferLayer,
)


class ExcitationInhibitionNetwork(nn.Module):
    """Excitation-Inhibition network for dendritic modeling.

    Args:
        input_dim: Input dimension
        transfer_params: Parameters for the transfer function.
        excitatory_layer_sizes: List of integers specifying the number of excitatory cells in each layer.
        inhibitory_layer_sizes: List of integers specifying the number of inhibitory cells in each layer.
        excitatory_branch_factors: List of integers specifying the number of branches for excitatory cells.
        inhibitory_branch_factors: List of integers specifying the number of branches for inhibitory cells.
        reactivate: Whether to reactivate cells.
        reactivation_init_m: Initial m parameter for reactivation.
        reactivation_init_b: Initial b parameter for reactivation.
        dbl_init_method: Method for initializing dendritic branch layers.
        somatic_synapses: Whether to use somatic synapses.
        ee_synapses_per_branch_per_layer: List of integers specifying the number of excitatory-excitatory synapses per branch per layer.
        ei_synapses_per_branch_per_layer: List of integers specifying the number of excitatory-inhibitory synapses per branch per layer.
        ie_synapses_per_branch_per_layer: List of integers specifying the number of inhibitory-excitatory synapses per branch per layer.
        ii_synapses_per_branch_per_layer: List of integers specifying the number of inhibitory-inhibitory synapses per branch per layer.
        use_shunting: Whether to use shunting inhibition.
        reactivation_type: Type of reactivation function.
        reactivation_strategy: Strategy for reactivation.
        blocklinear_strategy: Strategy for blocklinear computation.
        topk_init_method: Method for initializing topk.
        topk_strategy: Strategy for topk.
        topk_type: Type of topk.
        print_hooks: Whether to print hooks.
        excitatory_target_density: Target density for excitatory cells.
        inhibitory_target_density: Target density for inhibitory cells.
        use_noise: Whether to use noise.
        sigma: Standard deviation for noise.
        rewiring_mode: Mode for rewiring.
        rewire_frequency: Frequency for rewiring.
        synapses_per_branch: Number of synapses per branch.
        efficient_blocklinear: Whether to use efficient blocklinear.
        freeze_excitatory_connectivity: Whether to freeze excitatory connectivity.
        freeze_inhibitory_connectivity: Whether to freeze inhibitory connectivity.
        freeze_all_connectivity: Whether to freeze all connectivity.
        init_method: Method for initializing weights.
        weight_threshold: Threshold for weight pruning.
        input_noise: Noise for input.
        adaptive_initialization: Whether to use adaptive initialization.
    """

    def __init__(
        self,
        input_dim: int,
        transfer_params: dict[str, Any],
        excitatory_layer_sizes: list[int],
        inhibitory_layer_sizes: list[int],
        excitatory_branch_factors: list[int],
        inhibitory_branch_factors: list[int],
        reactivate: bool = True,
        reactivation_init_m: float = 1.5,
        reactivation_init_b: float = 0.5,
        dbl_init_method: str = "analytical_expectation",
        somatic_synapses: bool = True,
        ee_synapses_per_branch_per_layer: Optional[list[int]] = None,
        ei_synapses_per_branch_per_layer: Optional[list[int]] = None,
        ie_synapses_per_branch_per_layer: Optional[list[int]] = None,
        ii_synapses_per_branch_per_layer: Optional[list[int]] = None,
        inhibitory_network_type: str = "dendritic",
        use_shunting: bool = True,
        reactivation_type: str = "param_tanh",
        reactivation_strategy: str = "none",
        blocklinear_strategy: str = "none",
        topk_init_method: str = "xavier_normal",
        topk_noise_level: float = 0.0,
        topk_strategy: str = "none",
        topk_type: str = "standard",
        weight_transform="exp",  # Weight transformation for positive weights
        print_hooks: bool = False,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode="global",
        rewire_frequency=1,
        synapses_per_branch=None,
        efficient_blocklinear: bool = False,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        freeze_ee_connectivity=False,
        freeze_ei_connectivity=False,
        freeze_ie_connectivity=False,
        freeze_ii_connectivity=False,
        freeze_all_connectivity=False,
        init_method="xavier_normal",
        weight_threshold=1e-6,
        weight_init=0.01,
        input_noise=0,
        adaptive_initialization=True,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.transfer_params = transfer_params
        self.output_dim = excitatory_layer_sizes[-1]
        self.excitatory_layer_sizes = excitatory_layer_sizes
        self.inhibitory_layer_sizes = inhibitory_layer_sizes
        self.excitatory_branch_factors = excitatory_branch_factors
        self.inhibitory_branch_factors = inhibitory_branch_factors
        self.reactivate = reactivate
        self.reactivation_init_m = reactivation_init_m
        self.reactivation_init_b = reactivation_init_b
        self.dbl_init_method = dbl_init_method
        self.somatic_synapses = somatic_synapses
        self.inhibitory_network_type = inhibitory_network_type
        self.ee_synapses_per_branch_per_layer = (
            ee_synapses_per_branch_per_layer if ee_synapses_per_branch_per_layer else []
        )
        self.ei_synapses_per_branch_per_layer = (
            ei_synapses_per_branch_per_layer if ei_synapses_per_branch_per_layer else []
        )
        self.ie_synapses_per_branch_per_layer = (
            ie_synapses_per_branch_per_layer if ie_synapses_per_branch_per_layer else []
        )
        self.ii_synapses_per_branch_per_layer = (
            ii_synapses_per_branch_per_layer if ii_synapses_per_branch_per_layer else []
        )
        self.use_shunting = use_shunting
        self.reactivation_type = reactivation_type
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.topk_init_method = topk_init_method
        self.topk_strategy = topk_strategy
        self.topk_type = topk_type
        self.weight_transform = weight_transform
        self.topk_noise_level = topk_noise_level
        self.print_hooks = print_hooks
        self.excitatory_target_density = excitatory_target_density
        self.inhibitory_target_density = inhibitory_target_density
        self.use_noise = use_noise
        self.sigma = sigma
        self.rewiring_mode = rewiring_mode
        self.rewire_frequency = rewire_frequency
        self.efficient_blocklinear = efficient_blocklinear
        self.synapses_per_branch = synapses_per_branch
        self.freeze_excitatory_connectivity = freeze_excitatory_connectivity
        self.freeze_inhibitory_connectivity = freeze_inhibitory_connectivity
        self.freeze_ee_connectivity = freeze_ee_connectivity
        self.freeze_ei_connectivity = freeze_ei_connectivity
        self.freeze_ie_connectivity = freeze_ie_connectivity
        self.freeze_ii_connectivity = freeze_ii_connectivity
        self.freeze_all_connectivity = freeze_all_connectivity
        self.init_method = init_method
        self.weight_threshold = weight_threshold
        self.weight_init = weight_init
        self.input_noise = input_noise
        self.adaptive_initialization = adaptive_initialization

        # Set up transfer function
        self.setup_transfer_function()

        # Set up network layers
        self.setup_network_layers(**kwargs)

    def setup_transfer_function(self):
        self.transfer_fn = TransferLayer(
            input_dim=self.input_dim, transfer_params=self.transfer_params
        )

    def setup_network_layers(self, **kwargs):
        """Set up the excitatory and inhibitory network layers"""
        # Create network layers using ExcitationInhibitionLayer
        layers = []

        # Setup the lists we need
        excitatory_layer_sizes = deepcopy(self.excitatory_layer_sizes)
        inhibitory_layer_sizes = deepcopy(self.inhibitory_layer_sizes)

        # Insert input_dim at the beginning
        excitatory_layer_sizes.insert(0, self.input_dim)

        # For input_mode=1, inhibitory_layer_sizes should remain empty or filled with None
        if self.transfer_params.get("input_mode", 0) == 1:
            # Ensure we have None for each layer if the list is empty
            if not inhibitory_layer_sizes:
                inhibitory_layer_sizes = [None] * len(self.excitatory_layer_sizes)
            inhibitory_layer_sizes.insert(0, None)
        else:
            inhibitory_layer_sizes.insert(0, None)

        # Set up synapses per branch per layer
        ee_syn = list(self.ee_synapses_per_branch_per_layer)
        ei_syn = list(self.ei_synapses_per_branch_per_layer)
        ie_syn = list(self.ie_synapses_per_branch_per_layer)
        ii_syn = list(self.ii_synapses_per_branch_per_layer)

        # Extend lists if needed - use exactly what's specified, no smart defaulting
        num_layers = len(excitatory_layer_sizes) - 1

        # Check if we have inhibitory neurons in the network
        has_inhibitory_neurons = any(
            size is not None and size > 0 for size in self.inhibitory_layer_sizes
        )

        # Only extend lists if they have values to repeat, otherwise error
        while len(ee_syn) < num_layers:
            if ee_syn:
                ee_syn.append(ee_syn[-1])
            else:
                raise ValueError("ee_synapses_per_branch_per_layer cannot be empty")

        # EI synapses: only required if inhibitory neurons exist
        while len(ei_syn) < num_layers:
            if ei_syn:
                ei_syn.append(ei_syn[-1])
            elif has_inhibitory_neurons:
                raise ValueError(
                    "ei_synapses_per_branch_per_layer cannot be empty when inhibitory neurons exist"
                )
            else:
                # No inhibitory neurons, so E->I connections are not needed
                ei_syn.append(0)

        # IE synapses: required for input_mode=1 or when inhibitory neurons exist
        while len(ie_syn) < num_layers:
            if ie_syn:
                ie_syn.append(ie_syn[-1])
            elif (
                self.transfer_params.get("input_mode", 0) == 1 or has_inhibitory_neurons
            ):
                raise ValueError("ie_synapses_per_branch_per_layer cannot be empty")
            else:
                # No inhibitory input available, so I->E connections are not needed
                ie_syn.append(0)

        # II synapses: only required if inhibitory neurons exist
        while len(ii_syn) < num_layers:
            if ii_syn:
                ii_syn.append(ii_syn[-1])
            elif has_inhibitory_neurons:
                raise ValueError(
                    "ii_synapses_per_branch_per_layer cannot be empty when inhibitory neurons exist"
                )
            else:
                # No inhibitory neurons, so I->I connections are not needed
                ii_syn.append(0)

        excitatory_input_dim = self.transfer_fn.excitatory_dim
        inhibitory_input_dim = self.transfer_fn.inhibitory_dim

        # Build each layer
        for i in range(num_layers):
            # For input_mode=1, don't build inhibitory cells
            # Also check if inhibitory layer size exists for this layer
            build_inhibitory_cells = (
                self.transfer_params.get("input_mode", 0) != 1
                and i + 1 < len(inhibitory_layer_sizes)
                and inhibitory_layer_sizes[i + 1] is not None
                and inhibitory_layer_sizes[i + 1] > 0
            )

            # For the first layer, determine correct dimensions
            if i == 0:
                # First layer receives input from transfer function
                layer_excitatory_input_dim = excitatory_input_dim
                layer_inhibitory_input_dim = inhibitory_input_dim
                # Excitatory cells receive inhibitory output from same layer's inhibitory cells
                # Check if inhibitory layer exists
                if (
                    i + 1 < len(inhibitory_layer_sizes)
                    and inhibitory_layer_sizes[i + 1] is not None
                ):
                    inhibitory_layer_sizes[i + 1]
                else:
                    # For input_mode=1, there are no inhibitory neurons
                    pass
            else:
                # Subsequent layers receive outputs from previous layer
                layer_excitatory_input_dim = excitatory_layer_sizes[i]
                layer_inhibitory_input_dim = (
                    inhibitory_layer_sizes[i]
                    if i < len(inhibitory_layer_sizes)
                    else None
                )
                # Excitatory cells receive inhibitory output from same layer's inhibitory cells
                # Check if inhibitory layer exists
                if (
                    i + 1 < len(inhibitory_layer_sizes)
                    and inhibitory_layer_sizes[i + 1] is not None
                ):
                    inhibitory_layer_sizes[i + 1]
                else:
                    # For input_mode=1, there are no inhibitory neurons
                    pass

            # Get n_inhibitory_cells for this layer
            n_inhibitory_cells = (
                inhibitory_layer_sizes[i + 1]
                if i + 1 < len(inhibitory_layer_sizes)
                else None
            )

            # Create ExcitationInhibitionLayer
            layer = ExcitationInhibitionLayer(
                n_excitatory_cells=excitatory_layer_sizes[i + 1],
                n_inhibitory_cells=n_inhibitory_cells,
                excitatory_branch_factors=self.excitatory_branch_factors,
                inhibitory_branch_factors=self.inhibitory_branch_factors,
                excitatory_input_dim=layer_excitatory_input_dim,
                ee_synapses_per_branch=ee_syn[i],
                ei_synapses_per_branch=ei_syn[i],
                build_inhibitory_cells=build_inhibitory_cells,
                inhibitory_network_type=self.inhibitory_network_type,
                inhibitory_input_dim=layer_inhibitory_input_dim,
                ie_synapses_per_branch=ie_syn[i],
                ii_synapses_per_branch=ii_syn[i],
                reactivate=self.reactivate,
                reactivation_init_m=self.reactivation_init_m,
                reactivation_init_b=self.reactivation_init_b,
                dbl_init_method=self.dbl_init_method,
                somatic_synapses=self.somatic_synapses,
                topk_init_method=self.topk_init_method,
                use_shunting=self.use_shunting,
                reactivation_strategy=self.reactivation_strategy,
                blocklinear_strategy=self.blocklinear_strategy,
                reactivation_type=self.reactivation_type,
                topk_noise_level=self.topk_noise_level,
                topk_strategy=self.topk_strategy,
                topk_type=self.topk_type,
                weight_transform=self.weight_transform,
                print_hooks=self.print_hooks,
                excitatory_target_density=self.excitatory_target_density,
                inhibitory_target_density=self.inhibitory_target_density,
                use_noise=self.use_noise,
                sigma=self.sigma,
                rewiring_mode=self.rewiring_mode,
                rewire_frequency=self.rewire_frequency,
                synapses_per_branch=self.synapses_per_branch,
                freeze_excitatory_connectivity=self.freeze_excitatory_connectivity,
                freeze_inhibitory_connectivity=self.freeze_inhibitory_connectivity,
                freeze_ee_connectivity=self.freeze_ee_connectivity,
                freeze_ei_connectivity=self.freeze_ei_connectivity,
                freeze_ie_connectivity=self.freeze_ie_connectivity,
                freeze_ii_connectivity=self.freeze_ii_connectivity,
                freeze_all_connectivity=self.freeze_all_connectivity,
                init_method=self.init_method,
                weight_threshold=self.weight_threshold,
                weight_init=self.weight_init,
                input_noise=self.input_noise,
                efficient_blocklinear=self.efficient_blocklinear,
                adaptive_initialization=self.adaptive_initialization,
                **kwargs,
            )
            layers.append(layer)

        # Create ModuleList with the layers
        self.layers = nn.ModuleList(layers)

        # Enable JIT compilation for better performance (PyTorch 2.0+)
        try:
            if torch._dynamo is not None:
                # Increase recompile limit to reduce warnings
                torch._dynamo.config.recompile_limit = 100
                torch._dynamo.config.force_parameter_static_shapes = (
                    False  # Allow dynamic params
                )
            self.forward = torch.compile(self.forward, dynamic=True)
        except (AttributeError, ImportError):
            # torch.compile not available in older PyTorch versions
            pass

    def decay_weights(self, weight_decay, weight_boosting=False):
        for layer in self.layers:
            layer: ExcitationInhibitionLayer
            layer.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self):
        for layer in self.layers:
            layer: ExcitationInhibitionLayer
            layer.apply_rewiring()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward logic:
        - TransferLayer splits/duplicates input based on input_mode and independent_pathways
        - All EI layers now build and process their own inhibitory cells
        - First layer's inhibitory cells receive the same input as excitatory cells
        - Subsequent layers receive outputs from previous layers
        """
        excitatory_x, inhibitory_x = self.transfer_fn(x)

        # Optimized: Use direct iteration over ModuleList
        for layer in self.layers:
            excitatory_x, inhibitory_x = layer(excitatory_x, inhibitory_x)

        return excitatory_x

    @property
    def branch_layers(self):
        """Get all layers that have branch structures"""
        branch_layers = []
        for layer in self.layers:
            layer: ExcitationInhibitionLayer
            if hasattr(layer, "excitatory_cells"):
                branch_layers.append(layer.excitatory_cells)
            if (
                hasattr(layer, "inhibitory_cells")
                and layer.inhibitory_cells is not None
            ):
                branch_layers.append(layer.inhibitory_cells)
        return branch_layers

    @property
    def n_branch_layers(self):
        """Get the number of branch layers"""
        return len(self.branch_layers)

    def get_effective_params(self) -> int:
        """Calculate the effective number of parameters after topk sparsity."""
        effective = 0

        for layer in self.layers:
            # Process excitatory cells
            if (
                hasattr(layer, "excitatory_cells")
                and layer.excitatory_cells is not None
            ):
                excitatory_cells = layer.excitatory_cells
                for branch_layer in excitatory_cells.branch_layers:
                    # Excitatory branch (TopKLinear)
                    if (
                        hasattr(branch_layer, "branch_excitation")
                        and branch_layer.branch_excitation is not None
                    ):
                        if hasattr(branch_layer.branch_excitation, "K"):
                            effective += (
                                branch_layer.branch_excitation.out_features
                                * branch_layer.branch_excitation.K
                            )

                    # Inhibitory branch (if present)
                    if (
                        hasattr(branch_layer, "branch_inhibition")
                        and branch_layer.branch_inhibition is not None
                    ):
                        if hasattr(branch_layer.branch_inhibition, "K"):
                            effective += (
                                branch_layer.branch_inhibition.out_features
                                * branch_layer.branch_inhibition.K
                            )

                    # Branches to output (BlockLinear - dense, so full params)
                    if (
                        hasattr(branch_layer, "branches_to_output")
                        and branch_layer.branches_to_output is not None
                    ):
                        effective += sum(
                            p.numel()
                            for p in branch_layer.branches_to_output.parameters()
                        )

                    # Reactivation (if present)
                    if hasattr(branch_layer, "reactivation") and not isinstance(
                        branch_layer.reactivation, nn.Identity
                    ):
                        effective += sum(
                            p.numel() for p in branch_layer.reactivation.parameters()
                        )

            # Process inhibitory cells
            if (
                hasattr(layer, "inhibitory_cells")
                and layer.inhibitory_cells is not None
            ):
                inhibitory_cells = layer.inhibitory_cells
                for branch_layer in inhibitory_cells.branch_layers:
                    # Excitatory branch (TopKLinear)
                    if (
                        hasattr(branch_layer, "branch_excitation")
                        and branch_layer.branch_excitation is not None
                    ):
                        if hasattr(branch_layer.branch_excitation, "K"):
                            effective += (
                                branch_layer.branch_excitation.out_features
                                * branch_layer.branch_excitation.K
                            )

                    # Inhibitory branch (if present)
                    if (
                        hasattr(branch_layer, "branch_inhibition")
                        and branch_layer.branch_inhibition is not None
                    ):
                        if hasattr(branch_layer.branch_inhibition, "K"):
                            effective += (
                                branch_layer.branch_inhibition.out_features
                                * branch_layer.branch_inhibition.K
                            )

                    # Branches to output (BlockLinear - dense, so full params)
                    if (
                        hasattr(branch_layer, "branches_to_output")
                        and branch_layer.branches_to_output is not None
                    ):
                        effective += sum(
                            p.numel()
                            for p in branch_layer.branches_to_output.parameters()
                        )

                    # Reactivation (if present)
                    if hasattr(branch_layer, "reactivation") and not isinstance(
                        branch_layer.reactivation, nn.Identity
                    ):
                        effective += sum(
                            p.numel() for p in branch_layer.reactivation.parameters()
                        )

        return effective


class ConfigurableEINetwork(ExcitationInhibitionNetwork):
    """
    ExcitationInhibitionNetwork that is initialized from structured config.

    This provides a clean way to create EINet instances from structured configuration,
    separating configuration structure from implementation details.
    """

    def __init__(self, config: Union[dict, DictConfig], input_dim: int):
        """
        Initialize from structured config.

        Args:
            config: Structured configuration (e.g., core_config from YAML)
                   Must contain 'architecture', 'connectivity', 'transfer', etc. sections
            input_dim: Input dimension from encoder
        """
        from dendritic_modeling.scripts.script_utils.config_utils import (
            prepare_ei_network_params,
        )

        params = prepare_ei_network_params(config, input_dim)
        super().__init__(**params)


__all__ = ["ConfigurableEINetwork", "ExcitationInhibitionNetwork"]
