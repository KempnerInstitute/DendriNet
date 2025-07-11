"""
Excitation-Inhibition Network.

This module implements the ExcitationInhibitionNetwork which uses separate
excitatory and inhibitory pathways with dendritic computation.
"""

from copy import deepcopy
from typing import Any, Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
    ExcitationInhibitionLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.transfer import (
    TransferLayer,
)
from dendritic_modeling.networks.base import BaseNetwork


class ExcitationInhibitionNetwork(BaseNetwork):
    """
    A neural network model that implements excitatory and inhibitory pathways.
    This is a placeholder implementation based on common patterns.
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
        somatic_synapses: bool = True,
        ee_synapses_per_branch_per_layer: Optional[list[int]] = None,
        ei_synapses_per_branch_per_layer: Optional[list[int]] = None,
        ie_synapses_per_branch_per_layer: Optional[list[int]] = None,
        ii_synapses_per_branch_per_layer: Optional[list[int]] = None,
        use_shunting: bool = True,
        reactivation_type: str = "param_tanh",
        reactivation_strategy: str = "inverse",
        blocklinear_strategy: str = "conductance_dynamic",
        topk_init_method: str = "xavier_normal",
        topk_strategy: str = "conductance_dynamic",
        topk_type: str = "standard",
        print_hooks: bool = False,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode="global",
        rewire_frequency=1,
        synapses_per_branch=None,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method="xavier_normal",
        weight_threshold=1e-6,
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
        self.somatic_synapses = somatic_synapses
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
        self.topk_noise_level = kwargs.get("topk_noise_level", 0.0)
        self.print_hooks = print_hooks
        self.excitatory_target_density = excitatory_target_density
        self.inhibitory_target_density = inhibitory_target_density
        self.use_noise = use_noise
        self.sigma = sigma
        self.rewiring_mode = rewiring_mode
        self.rewire_frequency = rewire_frequency
        self.synapses_per_branch = synapses_per_branch
        self.freeze_excitatory_connectivity = freeze_excitatory_connectivity
        self.freeze_inhibitory_connectivity = freeze_inhibitory_connectivity
        self.init_method = init_method
        self.weight_threshold = weight_threshold

        # Set up transfer function
        self.setup_transfer_function()

        # Set up network layers
        self.setup_network_layers()

    def setup_transfer_function(self):
        self.transfer_fn = TransferLayer(
            input_dim=self.input_dim,
            transfer_params=self.transfer_params,
            n_inhibitory_cells=self.inhibitory_layer_sizes[0],
            inhibitory_branch_factors=self.inhibitory_branch_factors,
            excitatory_synapses_per_branch=self.ee_synapses_per_branch_per_layer[0],
            reactivate=self.reactivate,
            reactivation_init_m=self.reactivation_init_m,
            reactivation_init_b=self.reactivation_init_b,
            somatic_synapses=self.somatic_synapses,
            topk_init_method=self.topk_init_method,
            use_shunting=self.use_shunting,
            reactivation_type=self.reactivation_type,
            reactivation_strategy=self.reactivation_strategy,
            blocklinear_strategy=self.blocklinear_strategy,
            topk_noise_level=self.topk_noise_level,
            topk_strategy=self.topk_strategy,
            topk_type=self.topk_type,
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
            init_method=self.init_method,
            weight_threshold=self.weight_threshold,
        )

    def setup_network_layers(self):
        """Set up the excitatory and inhibitory network layers"""
        # Create network layers using ExcitationInhibitionLayer
        layers = []

        # Setup the lists we need
        excitatory_layer_sizes = deepcopy(self.excitatory_layer_sizes)
        inhibitory_layer_sizes = deepcopy(self.inhibitory_layer_sizes)

        # Insert input_dim at the beginning
        excitatory_layer_sizes.insert(0, self.input_dim)
        inhibitory_layer_sizes.insert(0, None)

        # Set up synapses per branch per layer
        ee_syn = list(self.ee_synapses_per_branch_per_layer)
        ei_syn = list(self.ei_synapses_per_branch_per_layer)
        ie_syn = list(self.ie_synapses_per_branch_per_layer)
        ii_syn = list(self.ii_synapses_per_branch_per_layer)

        # Extend lists if needed
        num_layers = len(excitatory_layer_sizes) - 1
        while len(ee_syn) < num_layers:
            ee_syn.append(ee_syn[-1] if ee_syn else 1)
        while len(ei_syn) < num_layers:
            ei_syn.append(ei_syn[-1] if ei_syn else 1)
        while len(ie_syn) < num_layers:
            ie_syn.append(ie_syn[-1] if ie_syn else 1)
        while len(ii_syn) < num_layers:
            ii_syn.append(ii_syn[-1] if ii_syn else 1)

        excitatory_input_dim = self.transfer_fn.excitatory_dim
        inhibitory_input_dim = self.transfer_fn.inhibitory_dim

        # Build each layer
        for i in range(num_layers):
            build_inhibitory_cells = True
            if i == 0:
                build_inhibitory_cells = False

            # Create ExcitationInhibitionLayer
            layer = ExcitationInhibitionLayer(
                n_excitatory_cells=excitatory_layer_sizes[i + 1],
                n_inhibitory_cells=inhibitory_layer_sizes[i + 1],
                excitatory_branch_factors=self.excitatory_branch_factors,
                inhibitory_branch_factors=self.inhibitory_branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                ee_synapses_per_branch=ee_syn[i],
                ei_synapses_per_branch=ei_syn[i],
                build_inhibitory_cells=build_inhibitory_cells,
                inhibitory_input_dim=inhibitory_input_dim,
                ie_synapses_per_branch=ie_syn[i],
                ii_synapses_per_branch=ii_syn[i],
                reactivate=self.reactivate,
                reactivation_init_m=self.reactivation_init_m,
                reactivation_init_b=self.reactivation_init_b,
                somatic_synapses=self.somatic_synapses,
                topk_init_method=self.topk_init_method,
                use_shunting=self.use_shunting,
                reactivation_strategy=self.reactivation_strategy,
                blocklinear_strategy=self.blocklinear_strategy,
                reactivation_type=self.reactivation_type,
                topk_noise_level=self.topk_noise_level,
                topk_strategy=self.topk_strategy,
                topk_type=self.topk_type,
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
                init_method=self.init_method,
                weight_threshold=self.weight_threshold,
            )
            layers.append(layer)

            excitatory_input_dim = excitatory_layer_sizes[i + 1]
            inhibitory_input_dim = inhibitory_layer_sizes[i + 1]

        # Create ModuleList with the layers
        self.layers = nn.ModuleList(layers)

    def decay_weights(self, weight_decay):
        for layer in self.layers:
            layer: ExcitationInhibitionLayer
            layer.decay_weights(weight_decay)

    def apply_rewiring(self):
        for layer in self.layers:
            layer: ExcitationInhibitionLayer
            layer.apply_rewiring()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward logic depends on input_mode:
        - mode=0 => excit=x, inhibitory=None at the first layer
        - mode=1 => excit=x, inhibitory=x at the first layer, but we skip building that layer's inhib net => no separate output
        - mode=2 => pass x into self.input_net => produce (excit, inhib)
                    skip that layer's inhib net => no separate output
        Then subsequent layers do normal E-I if their inhib cells exist.
        """
        excitatory_x, inhibitory_x = self.transfer_fn(x)

        for i in range(len(self.layers)):
            # Each layer receives the previous layer's outputs
            excitatory_x, inhibitory_x = self.layers[i](excitatory_x, inhibitory_x)

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


__all__ = ["ExcitationInhibitionNetwork"]
