"""
Excitation-Inhibition Network.

This module implements the ExcitationInhibitionNetwork which uses separate
excitatory and inhibitory pathways with dendritic computation.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any
from copy import deepcopy

from dendritic_modeling.networks.layers.dendritic import ExcitationInhibitionLayer
from dendritic_modeling.networks.input_transforms import InputNetTransform


class ExcitationInhibitionNetwork(nn.Module):
    """
    A neural network model that implements excitatory and inhibitory pathways.
    This is a placeholder implementation based on common patterns.
    """

    def __init__(
        self,
        input_dim: int,
        excitatory_layer_sizes: List[int],
        inhibitory_layer_sizes: List[int],
        excitatory_branch_factors: List[int],
        inhibitory_branch_factors: List[int],
        output_dim: int = 10,
        reactivate: bool = True,
        reactivation_init_m: float = 1.5,
        reactivation_init_b: float = 0.5,
        somatic_synapses: bool = True,
        ee_synapses_per_branch_per_layer: List[int] = None,
        ei_synapses_per_branch_per_layer: List[int] = None,
        ie_synapses_per_branch_per_layer: List[int] = None,
        ii_synapses_per_branch_per_layer: List[int] = None,
        use_shunting: bool = True,
        reactivation_type: str = "param_tanh",
        reactivation_strategy: str = "inverse",
        blocklinear_strategy: str = "conductance_dynamic",
        enable_branch_outputs: bool = False,
        output_layer: bool = False,
        input_mode: int = 0,
        input_net_dict: Dict[str, Any] = None,
        learning_strategy: str = "mle",
        topk_init_method: str = "xavier_normal",
        topk_strategy: str = "conductance_dynamic",
        topk_type: str = "standard",
        print_hooks: bool = False,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode="global",
        synapses_per_branch=None,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method="xavier_normal",
        weight_threshold=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
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
        self.enable_branch_outputs = enable_branch_outputs
        self.output_layer = output_layer
        self.input_mode = input_mode
        self.input_net_dict = input_net_dict if input_net_dict else {}
        self.learning_strategy = learning_strategy
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
        self.synapses_per_branch = synapses_per_branch
        self.freeze_excitatory_connectivity = freeze_excitatory_connectivity
        self.freeze_inhibitory_connectivity = freeze_inhibitory_connectivity
        self.init_method = init_method
        self.weight_threshold = weight_threshold

        # Set up input transformation based on input_mode
        if input_mode == 0:
            # Original mode - raw input
            self.input_transform = nn.Identity()
        elif input_mode == 1:
            # Simple mode - input goes to both exc and inh pathways
            self.input_transform = lambda x: (x, x)
        elif input_mode == 2:
            # Network mode - transform input through a neural network
            if not input_net_dict:
                raise ValueError("input_net_dict must be provided when input_mode=2")
            self.setup_input_network()
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        # Set up network layers
        self.setup_network_layers()

        # Set up output layer if needed
        if self.output_layer:
            self.setup_output_layer()

        # Save branch activations if enabled
        self.branch_activations = [] if enable_branch_outputs else None

    def setup_input_network(self):
        """Set up the input transformation network based on input_net_dict"""
        # This would create the input network based on the configuration
        # Placeholder implementation
        self.input_net = InputNetTransform(
            input_dim=self.input_dim,
            hidden_dims=self.input_net_dict.get("hidden_dims", []),
            excitatory_dim=self.input_net_dict.get("excitatory_dim", 128),
            inhibitory_dim=self.input_net_dict.get("inhibitory_dim", 128),
            output_dim=self.input_net_dict.get("output_dim", 128),
            input_shape=self.input_net_dict.get("input_shape", None),
            excitatory_hidden_dims=self.input_net_dict.get(
                "excitatory_hidden_dims", []
            ),
            inhibitory_hidden_dims=self.input_net_dict.get(
                "inhibitory_hidden_dims", []
            ),
            excitatory_kernel_sizes=self.input_net_dict.get(
                "excitatory_kernel_sizes", []
            ),
            inhibitory_kernel_sizes=self.input_net_dict.get(
                "inhibitory_kernel_sizes", []
            ),
            excitatory_strides=self.input_net_dict.get("excitatory_strides", []),
            inhibitory_strides=self.input_net_dict.get("inhibitory_strides", []),
            network_type=self.input_net_dict.get("type", "MLPInputTransform"),
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

        # Initialize input_net if using input_mode=2
        if self.input_mode == 2 and self.input_net is not None:
            input_net_exc_dim = self.input_net.input_transform.excitatory_dim
            excitatory_layer_sizes[0] = input_net_exc_dim

        # Build each layer
        for i in range(num_layers):
            n_exc = excitatory_layer_sizes[i + 1]

            # Decide n_inh for layer i based on input_mode
            if i == 0:
                if self.input_mode == 0:
                    # input_mode=0 => normal approach => use config's inhibitory_layer_sizes[i+1]
                    n_inh = inhibitory_layer_sizes[i + 1]
                elif self.input_mode in (1, 2):
                    # skip building inhibitory net in the FIRST layer => set n_inh=0
                    n_inh = 0
                else:
                    raise ValueError(f"Unknown input_mode={self.input_mode}")
            else:
                # from 2nd layer onward, use the config's inhibitory_layer_sizes
                n_inh = inhibitory_layer_sizes[i + 1]

            # Determine inhibitory input size
            inhibitory_input_size = n_inh
            ii_connectivity = True
            if i == 0:
                ii_connectivity = False
                if self.input_mode == 1:
                    inhibitory_input_size = excitatory_layer_sizes[i]
                elif self.input_mode == 2:
                    # For input_mode=2, use net_out_dim as inhibitory_input_dim
                    inhibitory_input_size = (
                        self.input_net.input_transform.inhibitory_dim
                    )

            # Create ExcitationInhibitionLayer
            layer = ExcitationInhibitionLayer(
                n_excitatory_cells=n_exc,
                n_inhibitory_cells=n_inh,
                excitatory_branch_factors=self.excitatory_branch_factors,
                inhibitory_branch_factors=self.inhibitory_branch_factors,
                excitatory_input_dim=excitatory_layer_sizes[i],
                ee_synapses_per_branch=ee_syn[i],
                ei_synapses_per_branch=(
                    min(ei_syn[i], inhibitory_input_size)
                    if i == 0 and self.input_mode == 2 and inhibitory_input_size
                    else ei_syn[i]
                ),
                ii_connectivity=ii_connectivity,
                inhibitory_input_dim=inhibitory_input_size,
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
                enable_branch_outputs=self.enable_branch_outputs,
                topk_noise_level=self.topk_noise_level,
                topk_strategy=self.topk_strategy,
                topk_type=self.topk_type,
                print_hooks=self.print_hooks,
                excitatory_target_density=self.excitatory_target_density,
                inhibitory_target_density=self.inhibitory_target_density,
                use_noise=self.use_noise,
                sigma=self.sigma,
                rewiring_mode=self.rewiring_mode,
                synapses_per_branch=self.synapses_per_branch,
                freeze_excitatory_connectivity=self.freeze_excitatory_connectivity,
                freeze_inhibitory_connectivity=self.freeze_inhibitory_connectivity,
                init_method=self.init_method,
                weight_threshold=self.weight_threshold,
            )
            layers.append(layer)

        # Create ModuleList with the layers
        self.layers = nn.ModuleList(layers)

    def setup_output_layer(self):
        """Set up the output classification layer"""
        final_exc_size = (
            self.excitatory_layer_sizes[-1] if self.excitatory_layer_sizes else 0
        )
        self.output_classifier = nn.Linear(final_exc_size, self.output_dim)

    def decay_weights(self, weight_decay):
        for layer in self.layers:
            layer.decay_weights(weight_decay)

    def forward(self, x):
        """
        Forward logic depends on input_mode:
        - mode=0 => excit=x, inhibitory=None at the first layer
        - mode=1 => excit=x, inhibitory=x at the first layer, but we skip building that layer's inhib net => no separate output
        - mode=2 => pass x into self.input_net => produce (excit, inhib)
                    skip that layer's inhib net => no separate output
        Then subsequent layers do normal E–I if their inhib cells exist.
        """
        # Handle the input based on input_mode
        if self.input_mode == 0:
            excitatory_input = x
            inhibitory_input = None
        elif self.input_mode == 1:
            excitatory_input = x
            inhibitory_input = x
        elif self.input_mode == 2:
            if self.input_net is not None:
                excitatory_input, inhibitory_input = self.input_net(x)
            else:
                # Fallback if input_net is not initialized
                excitatory_input = x
                inhibitory_input = None
        else:
            raise ValueError(f"Unknown input_mode={self.input_mode}")

        # Forward through each layer
        excitatory_output, inhibitory_output = self.layers[0](
            excitatory_input, inhibitory_input
        )

        for i in range(1, len(self.layers)):
            # Each layer receives the previous layer's outputs
            excitatory_output, inhibitory_output = self.layers[i](
                excitatory_output, inhibitory_output
            )

        # Apply output layer if enabled
        if self.output_layer and hasattr(self, "output_classifier"):
            output = self.output_classifier(excitatory_output)
        else:
            output = excitatory_output

        # Store branch activations if enabled
        if self.enable_branch_outputs:
            self.branch_activations = {
                "excitatory": [excitatory_output],
                "inhibitory": [inhibitory_output]
                if inhibitory_output is not None
                else [],
            }

        return output

    def get_branch_activations(self):
        """Return branch activations if enabled"""
        if not self.enable_branch_outputs:
            raise ValueError(
                "Branch outputs not enabled. Set enable_branch_outputs=True to use this method."
            )
        return self.branch_activations

    def forward_with_branch_outputs(self, x):
        """Forward pass that extracts activations from all layers.

        Args:
            x: Input tensor

        Returns:
            tuple: (output, per_layer_activations)
                Where per_layer_activations is a list of activation tensors or lists
                from each layer in the network.
        """
        # Apply input transformations and create appropriate pathways
        if self.input_mode == 0:
            excitatory_input = x
            inhibitory_input = None
        elif self.input_mode == 1:
            excitatory_input = x
            inhibitory_input = x
        elif self.input_mode == 2:
            if self.input_net is not None:
                excitatory_input, inhibitory_input = self.input_net(x)
            else:
                # Fallback if input_net is not initialized
                excitatory_input = x
                inhibitory_input = None
        else:
            raise ValueError(f"Unknown input_mode={self.input_mode}")

        # Use forward_with_branch_outputs for each layer
        excitatory_output, inhibitory_output, inh_acts, exc_acts = self.layers[
            0
        ].forward_with_branch_outputs(excitatory_input, inhibitory_input)

        # Process the enhanced activations to create a list of activation tensors or lists
        # that contain information about both excitatory and inhibitory components
        all_acts = []
        for act in exc_acts:
            if isinstance(act, list) and len(act) > 1:
                # This is already an enhanced activation with [exc, inh, output] format
                all_acts.append(act)
            else:
                # Just use the excitatory activation as is
                all_acts.append(act)

        for i in range(1, len(self.layers)):
            (
                excitatory_output,
                inhibitory_output,
                layer_inh_acts,
                layer_exc_acts,
            ) = self.layers[i].forward_with_branch_outputs(
                excitatory_output, inhibitory_output
            )

            # Process this layer's activations similarly
            for act in layer_exc_acts:
                if isinstance(act, list) and len(act) > 1:
                    # Enhanced activation
                    all_acts.append(act)
                else:
                    # Just excitatory
                    all_acts.append(act)

        # Apply output layer if enabled - check if it's actually a callable object not a boolean
        output = excitatory_output
        if self.output_layer and hasattr(self, "output_classifier"):
            output = self.output_classifier(excitatory_output)

        return output, all_acts

    def compute_local_loss(self, x, y=None):
        # Placeholder for local loss computation
        return torch.tensor(0.0)

    @property
    def branch_layers(self):
        """Get all layers that have branch structures"""
        branch_layers = []
        for layer in self.layers:
            if hasattr(layer, "excitatory_cells"):
                branch_layers.append(layer.excitatory_cells)
            if hasattr(layer, "inhibitory_cells") and layer.inhibitory_cells is not None:
                branch_layers.append(layer.inhibitory_cells)
        return branch_layers

    @property
    def n_branch_layers(self):
        """Get the number of branch layers"""
        return len(self.branch_layers)

    def load_input_net_weights(self, input_net_weights):
        """Load weights for the input network"""
        if self.input_mode == 2 and hasattr(self, "input_net"):
            self.input_net.load_state_dict(input_net_weights)


__all__ = ["ExcitationInhibitionNetwork"] 