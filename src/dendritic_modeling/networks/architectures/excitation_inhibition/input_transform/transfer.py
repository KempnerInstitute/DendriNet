import difflib
import logging
from typing import Any

import torch
from torch import nn

from dendritic_modeling.config.inhibitory_wiring import strip_retired_routing_keys
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.divisive import (
    DivisiveInputNormalizer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.identity import (
    IdentityInputTransform,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.index import (
    IndexInputTransform,
)

logger = logging.getLogger(__name__)

# Options the transfer layer reads. ``seed`` and ``split_seed`` are filled in by
# training; ``split_seed`` wins when both are set.
TRANSFER_LAYER_KEYS = frozenset(
    {
        "input_transform",
        "independent_pathways",
        "excitatory_dim",
        "inhibitory_dim",
        "split_strategy",
        "split_seed",
        "seed",
        "output_activation",
        "input_normalization",
    }
)


class TransferLayer(nn.Module):
    """Turn the core input into excitatory and inhibitory input streams.

    ``independent_pathways: false`` passes the whole input to both streams;
    ``true`` splits the input channels between them. ``output_activation`` is
    applied to both streams.
    """

    def __init__(self, input_dim: int, transfer_params: dict[str, Any]):
        super().__init__()

        # Retired inhibitory-routing keys never changed this layer's output.
        transfer_params = strip_retired_routing_keys(transfer_params)
        unknown = sorted(set(transfer_params) - TRANSFER_LAYER_KEYS)
        if unknown:
            hints = []
            for key in unknown:
                match = difflib.get_close_matches(key, sorted(TRANSFER_LAYER_KEYS), n=1)
                hints.append(f"{key} (did you mean {match[0]}?)" if match else key)
            raise ValueError(
                "Unknown transfer option(s): "
                + ", ".join(hints)
                + ". Valid options: "
                + ", ".join(sorted(TRANSFER_LAYER_KEYS))
                + "."
            )
        input_transform = transfer_params.get("input_transform")
        if input_transform is not None and str(input_transform).lower() != "identity":
            raise ValueError(
                f"transfer.input_transform={input_transform!r} is not applied by "
                "TransferLayer. Use population_network.input_transform, "
                "spatial.input_transform or the transformer replacement "
                "input_transform for a nonnegative input adapter."
            )
        self.independent_pathways = independent_pathways = transfer_params.get(
            "independent_pathways", False
        )
        if input_dim is None or int(input_dim) <= 0:
            raise ValueError(f"input_dim must be a positive integer, got {input_dim}")
        input_dim = int(input_dim)
        self.input_dim = input_dim
        self.input_normalizer = DivisiveInputNormalizer.from_config(
            input_dim,
            transfer_params.get("input_normalization"),
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

        # A size of 0 means unset, as in configs written before 2026-10-08.
        excitatory_dim = transfer_params.get("excitatory_dim") or None
        inhibitory_dim = transfer_params.get("inhibitory_dim") or None
        if independent_pathways:
            # An unset size is the rest of the input; with both unset the input
            # is halved.
            self.pathway_split = IndexInputTransform(
                input_dim=input_dim,
                excitatory_dim=excitatory_dim,
                inhibitory_dim=inhibitory_dim,
                split_seed=transfer_params.get(
                    "split_seed", transfer_params.get("seed", None)
                ),
                split_strategy=transfer_params.get("split_strategy", "random"),
            )
            excitatory_input_dim = self.pathway_split.excitatory_dim
            inhibitory_input_dim = self.pathway_split.inhibitory_dim
        else:
            if excitatory_dim is not None or inhibitory_dim is not None:
                raise ValueError(
                    "transfer excitatory_dim and inhibitory_dim split the input "
                    "only with independent_pathways: true; set it, or remove them"
                )
            # Both streams carry the whole input.
            excitatory_input_dim = input_dim
            inhibitory_input_dim = input_dim
            self.pathway_split = IdentityInputTransform()

        self.excitatory_dim = excitatory_input_dim
        self.inhibitory_dim = inhibitory_input_dim

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"TransferLayer expected last input dimension {self.input_dim}, "
                f"got {x.shape[-1]}"
            )
        normalized = self.input_normalizer(x)
        x1, x2 = self.pathway_split(normalized)
        if x1 is x2 and self.output_activation in (None, "none", "relu"):
            # Both streams carry the same input: activate it once. For these
            # activations (derivative 0 or 1) the gradient is bitwise the same.
            activated = self.activation_fn(x1)
            return activated, activated
        return self.activation_fn(x1), self.activation_fn(x2)
