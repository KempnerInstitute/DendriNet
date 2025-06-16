"""
Matched Parameter MLP.

This module implements MatchedParamMLP which creates an MLP with
at least as many parameters as an equivalent EINet configuration.
"""

import torch.nn as nn
import torchinfo

from dendritic_modeling.networks.architectures.excitation_inhibition import ExcitationInhibitionNetwork


class MatchedParamMLP(nn.Module):
    """
    Creates an MLP with at least as many parameters as an equivalent EINet configuration.
    Matches the number of layers to DendriticBranchLayers and minimizes parameters while
    exceeding EINet parameter count.
    """

    def __init__(
        self,
        input_dim,
        excitatory_layer_sizes,
        inhibitory_layer_sizes,
        excitatory_branch_factors,
        inhibitory_branch_factors,
        ee_synapses_per_branch_per_layer,
        ei_synapses_per_branch_per_layer,
        ie_synapses_per_branch_per_layer,
        ii_synapses_per_branch_per_layer=[],
        reactivate=False,
        somatic_synapses=True,
        output_layer=False,
        output_dim=10,
        **kwargs,  # Accept but ignore other EINet parameters
    ):
        super(MatchedParamMLP, self).__init__()

        # Create EINet to get parameter count and number of layers
        einet = ExcitationInhibitionNetwork(
            input_dim=input_dim,
            excitatory_layer_sizes=excitatory_layer_sizes,
            inhibitory_layer_sizes=inhibitory_layer_sizes,
            excitatory_branch_factors=excitatory_branch_factors,
            inhibitory_branch_factors=inhibitory_branch_factors,
            ee_synapses_per_branch_per_layer=ee_synapses_per_branch_per_layer,
            ei_synapses_per_branch_per_layer=ei_synapses_per_branch_per_layer,
            ie_synapses_per_branch_per_layer=ie_synapses_per_branch_per_layer,
            ii_synapses_per_branch_per_layer=ii_synapses_per_branch_per_layer,
            reactivate=reactivate,
            somatic_synapses=somatic_synapses,
            output_layer=output_layer,
            output_dim=output_dim,
            **kwargs,
        )

        summary = torchinfo.summary(einet, verbose=0)
        target_params = summary.trainable_params

        # Get number of branch layers
        n_branch_layers = einet.n_branch_layers
        if not somatic_synapses:
            n_branch_layers -= 2 * len(einet.layers)  # Account for somatic layer

        # Calculate required hidden layer widths
        # For n hidden layers (n+1 total layers), we need:
        # input_dim * h1 + h1 * h2 + h2 * h3 + ... + hn * output_dim ≥ target_params
        # Minimize total parameters while satisfying this constraint
        # We'll use equal width hidden layers for simplicity
        # Then: input_dim * h + h * h * (n-1) + h * output_dim ≥ target_params
        # where n is number of hidden layers

        def get_param_count(hidden_width):
            """Calculate total parameters for given hidden width"""
            params = input_dim * hidden_width  # First layer
            params += (
                hidden_width * hidden_width * (n_branch_layers - 1)
            )  # Hidden layers
            params += hidden_width * output_dim  # Output layer
            return params

        # Binary search for minimum hidden width that exceeds target params
        left, right = 1, 10000
        min_width = right

        while left <= right:
            mid = (left + right) // 2
            params = get_param_count(mid)

            if params >= target_params:
                min_width = mid
                right = mid - 1
            else:
                left = mid + 1

        hidden_width = min_width

        # Build MLP layers
        layers = []

        # Input layer
        layers.extend([nn.Linear(input_dim, hidden_width), nn.ReLU()])

        # Hidden layers
        for _ in range(n_branch_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.ReLU()])

        # Output layer
        layers.append(nn.Linear(hidden_width, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


__all__ = ["MatchedParamMLP"] 