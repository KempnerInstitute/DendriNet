"""
DeepST Linear Layer for Dendritic Networks.

This module implements a dynamic sparse training (DeepST) linear layer
with Brownian motion-like weight evolution and dynamic rewiring.
"""

import math

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TOPK_INIT_METHODS,
)


class DeepstLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        target_density=0.1,
        sigma=0.05,  # Noise amplitude for the random walk
        use_noise=True,  # Whether to apply Gaussian noise
        rewiring_mode="global",  # 'local', 'global', or 'constant_branch'
        synapses_per_branch=None,  # Required for 'constant_branch' mode
        param_space="log",  # 'log' or 'presigmoid' for weight parameterization
        weight_init_range=None,
        init_method=None,  # Add initialization method parameter
        freeze_connectivity=False,  # New: flag to freeze synapse locations
        weight_threshold=1e-6,  # New: threshold for weight pruning
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.target_density = target_density
        self.sigma = sigma
        self.use_noise = use_noise
        self.rewiring_mode = rewiring_mode
        self.param_space = param_space
        self.init_method = init_method

        # Add K attribute for compatibility with initialize_dendritic_branch_layer
        self.K = int(target_density * in_features) if in_features else None

        # Default initialization ranges
        if weight_init_range is None:
            if param_space == "log":
                weight_init_range = (-2.1, -2.0)  # Default for log parameterization
            elif param_space == "presigmoid":
                weight_init_range = (-0.1, 0.1)  # Default for presigmoid
            elif param_space == "uniform":
                weight_init_range = (0.25, 0.75)  # Default for uniform
        # Store weight initialization range
        self.weight_init_range = weight_init_range

        # For constant_branch mode
        if rewiring_mode == "constant_branch":
            if synapses_per_branch is None:
                raise ValueError(
                    "synapses_per_branch must be specified for constant_branch mode"
                )
            if synapses_per_branch > in_features:
                raise ValueError(
                    f"synapses_per_branch ({synapses_per_branch}) cannot exceed in_features ({in_features})"
                )
            self.synapses_per_branch = synapses_per_branch

        # Total number of possible connections
        self.total_connections = out_features * in_features

        # Target number of active connections based on density
        if rewiring_mode == "constant_branch":
            self.target_connections = out_features * synapses_per_branch
        else:
            self.target_connections = int(self.total_connections * target_density)

        # Initialize weights parameter
        if self.init_method is not None and self.init_method in TOPK_INIT_METHODS:
            # Initialize using PyTorch's initialization methods
            self.pre_w = nn.Parameter(
                torch.zeros((out_features, in_features)), requires_grad=True
            )
            # Apply initialization to a temporary tensor (correct shape for initialization)
            temp_weight = torch.zeros((out_features, in_features))
            TOPK_INIT_METHODS[self.init_method](temp_weight)
            # Reshape to our desired shape
            self.pre_w.data = temp_weight.reshape(out_features, in_features)

            # If using log or presigmoid parameterization, transform the weights accordingly
            if param_space == "log":
                # For log parameterization, ensure weights are positive then take log
                self.pre_w.data = torch.abs(self.pre_w.data).clamp(min=1e-6)
                self.pre_w.data = torch.log(self.pre_w.data)
            elif param_space == "presigmoid":
                # For presigmoid, pass through inverse sigmoid (logit)
                self.pre_w.data = torch.clamp(self.pre_w.data, min=1e-6, max=1 - 1e-6)
                self.pre_w.data = torch.log(self.pre_w.data / (1 - self.pre_w.data))
            elif param_space == "uniform":
                # For uniform, pass through uniform initialization
                self.pre_w.data = torch.empty(out_features, in_features).uniform_(
                    self.weight_init_range[0], self.weight_init_range[1]
                )
        else:
            # Use uniform initialization with specified range
            self.pre_w = nn.Parameter(
                torch.zeros((out_features, in_features)).uniform_(
                    self.weight_init_range[0], self.weight_init_range[1]
                ),
                requires_grad=True,
            )

        # Initialize mask
        mask = torch.zeros_like(self.pre_w, dtype=torch.bool)
        # Register it as a buffer to ensure it moves with the model
        self.mask: torch.Tensor
        self.register_buffer("mask", mask)

        # Initialize connectivity
        self._initialize_connectivity()

        # Store freeze flag
        self.freeze_connectivity = freeze_connectivity

        # Store weight threshold for pruning
        self.weight_threshold = weight_threshold

    def _initialize_connectivity(self):
        if self.rewiring_mode == "constant_branch":
            # Initialize with constant number of connections per branch
            for neuron_idx in range(self.out_features):
                # Randomly select synapses_per_branch input connections
                input_indices = torch.randperm(self.in_features)[
                    : self.synapses_per_branch
                ]
                for input_idx in input_indices:
                    self.mask[neuron_idx, input_idx] = True
        else:
            # Standard random initialization for global or local modes
            connections_to_activate = self.target_connections

            # Flatten indices for uniform sampling
            all_indices = torch.nonzero(torch.ones_like(self.pre_w), as_tuple=False)

            # Randomly select indices to activate
            selected_indices = torch.randperm(len(all_indices))[
                :connections_to_activate
            ]
            selected_connections = all_indices[selected_indices]

            # Activate connections
            with torch.no_grad():
                for i, j in selected_connections:
                    self.mask[i, j] = True

    def weight(self):
        """Return the actual weight values (after transformation)"""
        if self.param_space == "log":
            return torch.exp(self.pre_w)
        elif self.param_space == "presigmoid":
            return torch.sigmoid(self.pre_w)
        elif self.param_space == "uniform":
            return self.pre_w

    def weight_mask(self):
        """Return the current mask"""
        return self.mask

    def pruned_weight(self):
        """Return masked weights"""
        return self.weight() * self.mask

    def forward(self, x):
        # Apply mask and sum across dendritic locations
        return torch.matmul(x, self.pruned_weight().t())

    def weighted_synapses(self, cell_weights, prune=False):
        """Apply cell weights to synapses (compatibility with DendriNet)"""
        if prune:
            synapse_weights = self.pruned_weight()
        else:
            synapse_weights = self.weight()

        weighted_synapses = cell_weights[:, None, None] * synapse_weights
        return weighted_synapses.sum(dim=0)

    def apply_noise_and_enforce_constraints(self):
        """
        Apply Brownian motion-like noise and enforce connectivity constraints.
        Modified to support freezing connectivity while maintaining stability.
        """
        with torch.no_grad():
            # Get current weights
            current_weights = self.weight()

            # Always apply basic constraints for numerical stability
            if self.param_space == "log":
                # Use a tighter clamping range for better stability
                self.pre_w.data = torch.clamp(self.pre_w.data, min=-10.0, max=10.0)
            elif self.param_space == "presigmoid":
                self.pre_w.data = torch.clamp(self.pre_w.data, min=-5.0, max=5.0)
            elif self.param_space == "uniform":
                pass

            # Skip rewiring if connectivity is frozen
            if self.freeze_connectivity:
                return

            # Continue with normal rewiring process for non-frozen connectivity

            # Apply Gaussian noise if enabled
            if self.use_noise:
                noise = torch.zeros_like(current_weights)
                noise[self.mask] = (
                    torch.randn(self.mask.sum(), device=self.mask.device) * self.sigma
                )
                current_weights = current_weights + noise

            # Find connections to prune (weights <= weight_threshold)
            to_prune = (current_weights <= self.weight_threshold) & self.mask

            # Count pruned connections
            num_pruned = to_prune.sum().item()

            # Update mask (remove pruned connections)
            self.mask[to_prune] = False

            # Create new random connections
            if num_pruned > 0:
                self._create_new_connections(to_prune)

    def _create_new_connections(self, pruned_mask):
        """
        Create new random connections based on the rewiring mode
        """
        if self.rewiring_mode == "constant_branch":
            # For each branch, maintain a constant number of synapses
            for neuron_idx in range(self.out_features):
                # Current active synapses at this branch
                current_active = self.mask[neuron_idx].sum().item()

                # How many synapses need to be added
                to_add = self.synapses_per_branch - current_active

                if to_add > 0:
                    # Get potential input neurons for new connections
                    potential_inputs = ~self.mask[neuron_idx]
                    potential_indices = torch.nonzero(
                        potential_inputs, as_tuple=False
                    ).squeeze(1)

                    if len(potential_indices) > 0:
                        # Randomly select input neurons to connect
                        num_to_select = min(to_add, len(potential_indices))
                        selected_indices = potential_indices[
                            torch.randperm(len(potential_indices))[:num_to_select]
                        ]

                        # Create new connections
                        for input_idx in selected_indices:
                            self.mask[neuron_idx, input_idx] = True
                            self.pre_w.data[neuron_idx, input_idx] = (
                                self._initialize_new_weight()
                            )

        elif self.rewiring_mode == "local":
            # Local rewiring - only create connections within the same neuron
            for neuron_idx in range(self.out_features):
                # Count pruned connections for this neuron
                pruned_for_neuron = pruned_mask[neuron_idx].sum().item()

                if pruned_for_neuron > 0:
                    # Get potential new connections for this neuron
                    potential_connections = ~self.mask[neuron_idx]

                    if potential_connections.sum() > 0:
                        # Get indices of potential connections
                        flat_indices = torch.nonzero(
                            potential_connections, as_tuple=False
                        )

                        # Randomly select indices to add
                        num_to_add = min(pruned_for_neuron, len(flat_indices))
                        if num_to_add > 0:
                            selected_indices = torch.randperm(len(flat_indices))[
                                :num_to_add
                            ]
                            selected_connections = flat_indices[selected_indices]

                            # Add new connections
                            for in_idx in selected_connections:
                                self.mask[neuron_idx, in_idx] = True
                                self.pre_w.data[neuron_idx, in_idx] = (
                                    self._initialize_new_weight()
                                )

        else:  # 'global' rewiring mode
            # Count total pruned connections
            num_pruned = pruned_mask.sum().item()

            # Get all potential connections globally
            potential_connections = ~self.mask

            if potential_connections.sum() > 0 and num_pruned > 0:
                flat_indices = torch.nonzero(potential_connections, as_tuple=False)

                # Randomly select indices to add
                num_to_add = min(num_pruned, len(flat_indices))
                if num_to_add > 0:
                    selected_indices = torch.randperm(len(flat_indices))[:num_to_add]
                    selected_connections = flat_indices[selected_indices]

                    # Add new connections
                    for i, j in selected_connections:
                        self.mask[i, j] = True
                        self.pre_w.data[i, j] = self._initialize_new_weight()

    def _initialize_new_weight(self):
        """Helper method to initialize a new weight according to the chosen method"""
        if self.init_method is not None and self.init_method in TOPK_INIT_METHODS:
            # Create a tensor with at least 2 dimensions for initialization
            temp = torch.zeros(2, 2, device=self.pre_w.device)  # Changed from 1D to 2D
            TOPK_INIT_METHODS[self.init_method](temp)

            # Just use one value from the initialized tensor
            value = temp[0, 0].item()

            # Apply transformations for log/presigmoid if needed
            if self.param_space == "log":
                #     value = abs(value)
                #     value = max(value, 1e-6)
                #     value = math.log(value)

                ### EXPERIMENTAL
                value = math.log(self.weight_threshold)
            elif self.param_space == "presigmoid":
                value = max(min(value, 1 - 1e-6), 1e-6)
                value = math.log(value / (1 - value))
            elif self.param_space == "uniform":
                pass
            return value
        else:
            # Default to uniform initialization with the specified range
            return (
                torch.empty(1, device=self.pre_w.device)
                .uniform_(self.weight_init_range[0], self.weight_init_range[1])
                .item()
            )

    def get_connectivity_stats(self):
        """
        Return statistics about the network connectivity
        """
        active_connections = self.mask.sum().item()
        density = active_connections / self.total_connections

        # Per-branch stats for constant_branch mode
        branch_stats = {}
        if self.rewiring_mode == "constant_branch":
            branch_counts = torch.sum(self.mask, dim=1)  # Sum over input dimension
            branch_stats = {
                "min_per_branch": branch_counts.min().item(),
                "max_per_branch": branch_counts.max().item(),
                "mean_per_branch": branch_counts.float().mean().item(),
                "target_per_branch": self.synapses_per_branch,
            }

        return {
            "active_connections": active_connections,
            "total_connections": self.total_connections,
            "density": density,
            "target_density": self.target_density,
            "branch_stats": branch_stats,
        }

    def decay_weights(self, weight_decay=0.1, weight_boost=False):
        """
        Apply weight decay to active synaptic connections.

        Args:
            weight_decay: float, weight decay rate (default: 0.1)
        """


__all__ = ["DeepstLinear"]
