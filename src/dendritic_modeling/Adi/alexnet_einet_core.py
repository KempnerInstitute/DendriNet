"""
Combined frozen AlexNet encoder + trainable EI network, used as the 'core' network
when encoder type is 'identity'.

This avoids the pre-encoding step that would require loading all 1.2M ImageNet
images into RAM at once. Instead, the frozen AlexNet runs per-batch during training,
and only the EI weights are updated.

Pipeline:
    raw images (3,224,224)
        -> AlexNetFCEncoder (frozen, num_fc_layers=1 -> 4096-dim)
        -> ConfigurableEINetwork (trainable)
        -> 4096-dim excitatory output

Use in YAML:
    model:
      encoder:
        type: "identity"
      core:
        type: "alexnet_einet_core"
        architecture:
          num_fc_layers: 1    # <-- consumed here; rest of architecture goes to EINet
          excitatory_layer_sizes: [4096]
          ...
"""

import torch
from omegaconf import DictConfig

from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.Adi.alexnet_fc_encoder import AlexNetFCEncoder
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ConfigurableEINetwork,
)


class AlexNetEINetCore(BaseNetwork):
    """
    Frozen AlexNet (features + fc6) followed by a trainable EI network.

    Args:
        config: Full core DictConfig (architecture, connectivity, transfer, etc.).
                Must include config.architecture.num_fc_layers.
        input_dim: The Identity encoder's output_dim (= raw image numel). Ignored
                   internally — the EI network always receives 4096-dim AlexNet output.
    """

    def __init__(self, config: DictConfig, input_dim: int):
        super().__init__()

        num_fc_layers = getattr(config.architecture, "num_fc_layers", 1)

        # Frozen pretrained AlexNet encoder
        self.alexnet_encoder = AlexNetFCEncoder(num_fc_layers=num_fc_layers)
        alexnet_output_dim = self.alexnet_encoder.output_dim  # 4096 for num_fc_layers>=1

        # Trainable EI network — receives AlexNet output, not raw image pixels
        self.ei_network = ConfigurableEINetwork(config, alexnet_output_dim)

        self.output_dim = self.ei_network.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alexnet_encoder(x)  # frozen
        x = self.ei_network(x)       # trainable
        return x

    def get_output_dim(self) -> int:
        return self.output_dim


__all__ = ["AlexNetEINetCore"]
