import logging
from typing import Any

import torch
from torch import nn

from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.identity import (
    IdentityInputTransform,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.index import (
    IndexInputTransform,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.prefix_split import (
    PrefixSplitInputTransform,
    resolve_e_stream_prefix_k,
)

logger = logging.getLogger(__name__)


class TransferLayer(nn.Module):
    def __init__(self, input_dim: int, transfer_params: dict[str, Any]):
        super().__init__()

        self.input_mode = input_mode = transfer_params.get("input_mode", 0)
        self.independent_pathways = independent_pathways = transfer_params.get(
            "independent_pathways", False
        )
        split_input_stream_flag = bool(
            transfer_params.get("split_input_stream", False)
        )
        e_stream_dims = transfer_params.get("e_stream_dims", None)

        if input_dim is None or int(input_dim) <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        input_dim = int(input_dim)
        self.input_dim = input_dim
        if input_mode not in {0, 1}:
            raise ValueError(f"Invalid input_mode: {input_mode}")

        if independent_pathways and split_input_stream_flag:
            raise ValueError(
                "independent_pathways and split_input_stream are mutually exclusive; "
                "only one may be True"
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

        # Effective split only when flag is on and e_stream_dims is provided.
        # Flag on with omitted dims reverts to identity (same as flag False).
        effective_split = split_input_stream_flag and e_stream_dims is not None
        self.split_input_stream = effective_split

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
                split_seed=transfer_params.get(
                    "split_seed", transfer_params.get("seed", None)
                ),
            )
        elif effective_split:
            # Contiguous prefix split: first k dims -> e_stream, rest -> i_stream.
            # Validate k eagerly so bad configs fail at construction.
            k = resolve_e_stream_prefix_k(e_stream_dims, input_dim=input_dim)
            self.pathway_split = PrefixSplitInputTransform(
                input_dim=input_dim,
                e_stream_dims=k,
            )
            excitatory_input_dim = k
            inhibitory_input_dim = input_dim - k
        else:
            # When independent_pathways / split_input_stream are inactive,
            # both streams use the full input_dim (identity copy).
            if split_input_stream_flag and e_stream_dims is None:
                logger.info(
                    "split_input_stream=True but e_stream_dims is omitted/None; "
                    "reverting to identity transfer (same as split_input_stream=False)"
                )
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
        elif input_mode == 1 and (independent_pathways or effective_split):
            self.inhibitory_dim = int(input_dim - self.excitatory_dim)
        elif input_mode == 1 and not independent_pathways and not effective_split:
            self.inhibitory_dim = self.excitatory_dim
        else:
            raise ValueError(f"Invalid input_mode: {input_mode}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"TransferLayer expected last input dimension {self.input_dim}, "
                f"got {x.shape[-1]}"
            )
        e_stream, i_stream = self.pathway_split(x)

        # Apply activation function if specified
        e_stream = self.activation_fn(e_stream)
        i_stream = self.activation_fn(i_stream)

        # No longer process inhibitory cells here - just return the split/duplicated input
        return e_stream, i_stream
