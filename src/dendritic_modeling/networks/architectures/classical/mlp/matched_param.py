"""
Matched Parameter MLP.

This module implements MatchedParamMLP which creates an MLP with
at least as many parameters as an equivalent EINet configuration.
"""

import torch
from torch import nn

from typing import Union
from omegaconf import DictConfig

from dendritic_modeling.networks.activations.factory import ActivationFactory
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ConfigurableEINetwork,
    DendriticBranchLayer,
    DendriNet
)
from dendritic_modeling.networks.base import BaseNetwork


class MatchedTotalParamMLP(BaseNetwork):
    """
    Creates an MLP with at least as many parameters as an equivalent EINet configuration.
    Matches the number of layers to DendriticBranchLayers and minimizes parameters while
    exceeding EINet parameter count.
    """

    def __init__(
        self,
        config: Union[dict, DictConfig],
        input_dim
    ):
        super().__init__()
        # Create EINet to get parameter count and number of layers
        einet = ConfigurableEINetwork(config, input_dim)

        self.output_dim = einet.output_dim

        # summary_ = torchinfo.summary(einet, verbose=0)  # Summary for debugging
        target_params = sum(p.numel() for p in einet.parameters() if p.requires_grad)

        # Get number of branch layers
        n_branch_layers = einet.n_branch_layers

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
                hidden_width * hidden_width * (n_branch_layers - 2)
            )  # Hidden layers
            params += hidden_width * einet.output_dim  # Output layer
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
        # activation_factory = ActivationFactory()

        # Input layer
        fc_linear = nn.Linear(input_dim, hidden_width)
        # if activation == "relu":
        #     nn.init.kaiming_normal_(fc_linear.weight)
        #     nn.init.zeros_(fc_linear.bias)
        # else:
        #     nn.init.xavier_normal_(fc_linear.weight)
        #     nn.init.zeros_(fc_linear.bias)
        # layers.extend(
        #     [
        #         fc_linear,
        #         activation_factory.create(act_type=activation, output_dim=hidden_width),
        #     ]
        # )
        nn.init.xavier_normal_(fc_linear.weight)
        nn.init.zeros_(fc_linear.bias)
        layers.extend([fc_linear, nn.Sigmoid()])

        # Hidden layers
        for _ in range(n_branch_layers - 2):
            fc_linear = nn.Linear(hidden_width, hidden_width)
            # if activation == "relu":
            #     nn.init.kaiming_normal_(fc_linear.weight)
            #     nn.init.zeros_(fc_linear.bias)
            # else:
            #     nn.init.xavier_normal_(fc_linear.weight)
            #     nn.init.zeros_(fc_linear.bias)
            # layers.extend(
            #     [
            #         fc_linear,
            #         activation_factory.create(
            #             act_type=activation, output_dim=hidden_width
            #         ),
            #     ]
            # )
            nn.init.xavier_normal_(fc_linear.weight)
            nn.init.zeros_(fc_linear.bias)
            layers.extend([fc_linear, nn.Sigmoid()])

        # Output layer
        layers.extend([nn.Linear(hidden_width, einet.output_dim), nn.Sigmoid()])
        self.hidden = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)


class MatchedActiveParamMLP(BaseNetwork):
    """
    Creates an MLP with at least as many parameters as an equivalent EINet configuration.
    Matches the number of layers to DendriticBranchLayers and minimizes parameters while
    exceeding EINet parameter count.
    """

    def __init__(
        self,
        config: Union[dict, DictConfig],
        input_dim
    ):
        super().__init__()
        # Create EINet to get parameter count and number of layers
        einet = ConfigurableEINetwork(config, input_dim)

        self.output_dim = einet.output_dim

        target_params = 0
        with torch.no_grad():
            for layer in einet.layers:
                exc_dendrinet = layer.excitatory_cells
                for branch_layer in exc_dendrinet.branch_layers:
                    branch_layer: DendriticBranchLayer
                    if branch_layer.branch_excitation is not None:
                        target_params += branch_layer.branch_excitation.weight_mask().sum().item()
                    if branch_layer.branch_inhibition is not None:
                        target_params += branch_layer.branch_inhibition.weight_mask().sum().item()
                    if hasattr(branch_layer, "branches_to_output"):
                        target_params += branch_layer.branches_to_output.log_weight.numel()
                    target_params += sum(p.numel() for p in branch_layer.reactivation.parameters() if p.requires_grad)
                
                if layer.inhibitory_cells is not None:
                    if isinstance(layer.inhibitory_cells, DendriNet):
                        inh_dendrinet = layer.inhibitory_cells
                        for branch_layer in inh_dendrinet.branch_layers:
                            branch_layer: DendriticBranchLayer
                            if branch_layer.branch_excitation is not None:
                                target_params += branch_layer.branch_excitation.weight_mask().sum().item()
                            if branch_layer.branch_inhibition is not None:
                                target_params += branch_layer.branch_inhibition.weight_mask().sum().item()
                            if hasattr(branch_layer, "branches_to_output"):
                                target_params += branch_layer.branches_to_output.log_weight.numel()
                            target_params += sum(p.numel() for p in branch_layer.reactivation.parameters() if p.requires_grad)
                    else:
                        target_params += sum(p.numel() for p in layer.inhibitory_cells.parameters() if p.requires_grad)

        # Get number of branch layers
        n_branch_layers = einet.n_branch_layers

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
                hidden_width * hidden_width * (n_branch_layers - 2)
            )  # Hidden layers
            params += hidden_width * einet.output_dim  # Output layer
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
        # activation_factory = ActivationFactory()

        # Input layer
        fc_linear = nn.Linear(input_dim, hidden_width)
        # if activation == "relu":
        #     nn.init.kaiming_normal_(fc_linear.weight)
        #     nn.init.zeros_(fc_linear.bias)
        # else:
        #     nn.init.xavier_normal_(fc_linear.weight)
        #     nn.init.zeros_(fc_linear.bias)
        # layers.extend(
        #     [
        #         fc_linear,
        #         activation_factory.create(act_type=activation, output_dim=hidden_width),
        #     ]
        # )
        nn.init.xavier_normal_(fc_linear.weight)
        nn.init.zeros_(fc_linear.bias)
        layers.extend([fc_linear, nn.Sigmoid()])

        # Hidden layers
        for _ in range(n_branch_layers - 2):
            fc_linear = nn.Linear(hidden_width, hidden_width)
            # if activation == "relu":
            #     nn.init.kaiming_normal_(fc_linear.weight)
            #     nn.init.zeros_(fc_linear.bias)
            # else:
            #     nn.init.xavier_normal_(fc_linear.weight)
            #     nn.init.zeros_(fc_linear.bias)
            # layers.extend(
            #     [
            #         fc_linear,
            #         activation_factory.create(
            #             act_type=activation, output_dim=hidden_width
            #         ),
            #     ]
            # )
            nn.init.xavier_normal_(fc_linear.weight)
            nn.init.zeros_(fc_linear.bias)
            layers.extend([fc_linear, nn.Sigmoid()])

        # Output layer
        layers.extend([nn.Linear(hidden_width, einet.output_dim), nn.Sigmoid()])
        self.hidden = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.hidden(x)


__all__ = ["MatchedTotalParamMLP", "MatchedActiveParamMLP"]
