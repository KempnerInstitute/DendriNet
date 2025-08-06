import logging
from typing import Any, Optional

import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.identity import \
    IdentityInputTransform
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.index import \
    IndexInputTransform

logger = logging.getLogger(__name__)


class TransferLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        transfer_params: dict[str, Any],
        n_inhibitory_cells: Optional[int] = None,
        inhibitory_branch_factors: Optional[list[int]] = None,
        excitatory_synapses_per_branch: Optional[int] = None,
        reactivate: bool = True,
        reactivation_init_m: float = 1.5,
        reactivation_init_b: float = 0.5,
        somatic_synapses: bool = True,
        topk_init_method: str = "xavier_normal",
        use_shunting: bool = True,
        reactivation_type: str = "param_tanh",
        reactivation_strategy: str = "inverse",
        blocklinear_strategy: str = "conductance_dynamic",
        topk_noise_level: float = 0.0,
        topk_strategy: str = "conductance_dynamic",
        topk_type: str = "standard",
        print_hooks: bool = False,
        excitatory_target_density: float = 0.1,
        inhibitory_target_density: float = 0.1,
        use_noise: bool = True,
        sigma: float = 0.05,
        rewiring_mode: str = "global",
        rewire_frequency: int = 1,
        synapses_per_branch: Optional[list[int]] = None,
        freeze_excitatory_connectivity: bool = False,
        freeze_inhibitory_connectivity: bool = False,
        init_method: str = "xavier_normal",
        weight_threshold: float = 1e-6,
        efficient_blocklinear: bool = False,
        adaptive_initialization: bool = True,
    ):
        super().__init__()

        self.input_mode = input_mode = transfer_params.get("input_mode", 0)
        self.independent_pathways = independent_pathways = transfer_params.get(
            "independent_pathways", False
        )

        # Add output activation option
        self.output_activation = transfer_params.get("output_activation", None)

        # Create activation function if specified
        if self.output_activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif self.output_activation == "relu":
            self.activation_fn = nn.ReLU()
        elif self.output_activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif self.output_activation == "softplus":
            self.activation_fn = nn.Softplus()
        elif self.output_activation is None or self.output_activation == "none":
            self.activation_fn = nn.Identity()
        else:
            raise ValueError(f"Unknown output_activation: {self.output_activation}")

        if independent_pathways:
            # Get dimensions from transfer_params, checking for None, 0, or missing
            excitatory_dim = transfer_params.get("excitatory_dim", None)
            inhibitory_dim = transfer_params.get("inhibitory_dim", None)

            # If excitatory_dim is None, 0, or not provided, use input_dim / 2
            if not excitatory_dim:  # This covers None, 0, empty string, etc.
                excitatory_dim = int(input_dim / 2)
                logger.info(
                    f"excitatory_dim not provided or is 0/None. Setting to input_dim/2 = {excitatory_dim}"
                )

            # If inhibitory_dim is None, 0, or not provided, use remaining dimensions
            if not inhibitory_dim:  # This covers None, 0, empty string, etc.
                inhibitory_dim = int(input_dim - excitatory_dim)
                logger.info(
                    f"inhibitory_dim not provided or is 0/None. Setting to input_dim - excitatory_dim = {inhibitory_dim}"
                )

            excitatory_input_dim = excitatory_dim
            inhibitory_input_dim = inhibitory_dim

            self.pathway_split = IndexInputTransform(
                input_dim=input_dim,
                excitatory_dim=excitatory_input_dim,
                inhibitory_dim=inhibitory_input_dim,
            )
        else:
            # When independent_pathways is False, both use full input_dim
            excitatory_input_dim = input_dim
            inhibitory_input_dim = input_dim
            self.pathway_split = IdentityInputTransform()

        # Remove inhibitory cell creation - this will now be handled in the first EI layer
        # The TransferLayer only handles input splitting/duplication

        self.excitatory_dim = excitatory_input_dim

        if input_mode == 0:
            # When input_mode=0, inhibitory dimension is the same as excitatory
            # The first EI layer will build inhibitory cells
            self.inhibitory_dim = inhibitory_input_dim
        elif input_mode == 1 and independent_pathways:
            self.inhibitory_dim = int(input_dim - self.excitatory_dim)
        elif input_mode == 1 and not independent_pathways:
            self.inhibitory_dim = self.excitatory_dim
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = self.pathway_split(x)

        # Apply activation function if specified
        x1 = self.activation_fn(x1)
        x2 = self.activation_fn(x2)

        # No longer process inhibitory cells here - just return the split/duplicated input
        return x1, x2
