"""
Input networks that map raw inputs to excitatory and inhibitory streams.

``InputNetTransform`` produces the ``(excitatory_input, inhibitory_input)``
pair that E-I dendritic cores consume, either with a trainable MLP or by
splitting the input features by index.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.index import (
    IndexInputTransform,
)

logger = logging.getLogger(__name__)

NETWORK_TYPES = ("MLP", "IndexInputTransform")


class InputNetTransform(nn.Module):
    """Map a raw input ``x`` to ``(excitatory_input, inhibitory_input)``.

    ``network_type`` selects the mapping:

    * ``"MLP"`` (default): a shared trunk (``input_dim -> hidden_dims``,
      ``shared``) followed by an excitatory head (``-> excitatory_hidden_dims
      -> excitatory_dim``, ``head_exc``) and an inhibitory head
      (``-> inhibitory_hidden_dims -> inhibitory_dim``, ``head_inh``). With
      ``independent_heads=True`` there is no shared trunk and each head starts
      with its own ``hidden_dims`` layers. Hidden layers use ``activation``;
      the heads end in a linear layer, so a shunting core that needs
      nonnegative inputs needs a nonnegative activation after this transform.
    * ``"IndexInputTransform"``: splits the input features by index into
      ``excitatory_dim`` and ``inhibitory_dim`` features (``split_seed``
      fixes the split). It has no MLP options.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dims: Optional[list[int]] = None,
        independent_heads: bool = False,
        excitatory_dim: Optional[int] = None,
        inhibitory_dim: Optional[int] = None,
        excitatory_hidden_dims: Optional[list[int]] = None,
        inhibitory_hidden_dims: Optional[list[int]] = None,
        activation: str = "relu",
        network_type: str = "MLP",
        split_seed: Optional[int] = None,
    ):
        super().__init__()
        if network_type not in NETWORK_TYPES:
            raise ValueError(
                f"Unknown network type: {network_type!r} (expected one of "
                f"{', '.join(NETWORK_TYPES)})"
            )
        self.network_type = network_type

        if network_type == "IndexInputTransform":
            mlp_options = {
                "hidden_dims": hidden_dims,
                "independent_heads": independent_heads or None,
                "excitatory_hidden_dims": excitatory_hidden_dims,
                "inhibitory_hidden_dims": inhibitory_hidden_dims,
            }
            given = sorted(name for name, value in mlp_options.items() if value)
            if given:
                raise ValueError(
                    f"{', '.join(given)} configure the MLP input network; "
                    "network_type 'IndexInputTransform' splits the input by index"
                )
            self.input_transform = IndexInputTransform(
                input_dim=input_dim,
                excitatory_dim=excitatory_dim,
                inhibitory_dim=inhibitory_dim,
                split_seed=split_seed,
            )
            return

        if split_seed is not None:
            raise ValueError("split_seed applies to network_type 'IndexInputTransform'")
        missing = [
            name
            for name, value in (
                ("input_dim", input_dim),
                ("excitatory_dim", excitatory_dim),
                ("inhibitory_dim", inhibitory_dim),
            )
            if value is None
        ]
        if missing:
            raise ValueError(f"the MLP input network needs {', '.join(missing)}")
        trunk = list(hidden_dims or [])
        if independent_heads:
            self.shared = nn.Identity()
            head_input_dim = input_dim
            excitatory_hidden = trunk + list(excitatory_hidden_dims or [])
            inhibitory_hidden = trunk + list(inhibitory_hidden_dims or [])
        else:
            self.shared = MLP(input_dim, trunk, activation=activation)
            head_input_dim = self.shared.output_dim
            excitatory_hidden = list(excitatory_hidden_dims or [])
            inhibitory_hidden = list(inhibitory_hidden_dims or [])
        self.head_exc = MLP(
            head_input_dim,
            excitatory_hidden,
            activation=activation,
            output_dim=excitatory_dim,
        )
        self.head_inh = MLP(
            head_input_dim,
            inhibitory_hidden,
            activation=activation,
            output_dim=inhibitory_dim,
        )

    def forward(self, x):
        if self.network_type == "IndexInputTransform":
            return self.input_transform(x)
        features = self.shared(x)
        return self.head_exc(features), self.head_inh(features)

    def load_pretrained_weights(self, source, weights_path=None):
        """Load a state dict saved from this transform.

        Args:
            source: ``"none"`` (keep the current weights) or ``"file"``.
            weights_path: Path of the state dict when ``source`` is ``"file"``.
        """
        if source == "none":
            logger.info("No pre-trained weights to load.")
            return
        if source == "file":
            if not weights_path:
                raise ValueError("weights_path must be provided when source is 'file'")
            logger.info("Loading weights from file: %s", weights_path)
            self.load_state_dict(torch.load(weights_path, weights_only=True))
            return
        raise ValueError(f"Unknown source: {source!r} (expected 'none' or 'file')")


__all__ = ["NETWORK_TYPES", "InputNetTransform"]
