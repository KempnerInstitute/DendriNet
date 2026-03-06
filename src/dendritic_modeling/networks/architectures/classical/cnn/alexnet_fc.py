"""
AlexNet encoder with configurable FC depth, packaged as a BaseAutoencoder.

Used as the encoder in the K=1 AlexNet/EINet experiment:
    encoder_network = AlexNetFCAutoencoder(num_fc_layers=1)   # frozen, AlexNet up to fc6
    core_network    = ConfigurableEINetwork                    # trainable EI layer
    decoder_network = AlexNetFC8Decoder                        # trainable fc8

AlexNet classifier structure (torchvision):
    classifier[0]: Dropout(0.5)
    classifier[1]: Linear(9216 -> 4096)  = fc6
    classifier[2]: ReLU
    classifier[3]: Dropout(0.5)
    classifier[4]: Linear(4096 -> 4096)  = fc7
    classifier[5]: ReLU
    classifier[6]: Linear(4096 -> 1000)  = fc8  (in AlexNetFC8Decoder, not here)

num_fc_layers controls how many FC layers are included:
    0: features only          -> output_dim = 9216
    1: features + fc6         -> output_dim = 4096  (K=1 condition)
    2: features + fc6 + fc7   -> output_dim = 4096  (baseline condition)
"""

import torch
from torchvision import models
from torchvision.models import AlexNet_Weights

from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.architectures.classical.autoencoder.base import (
    BaseAutoencoder,
    Encoder,
)
from dendritic_modeling.networks.architectures.classical.identity import Identity


class AlexNetFCEncoderLayer(Encoder):
    """
    Frozen pretrained AlexNet encoder with configurable FC depth.

    Inherits from Encoder (satisfies setup_utils.py requirement for encoder_network.encoder).
    Overrides forward to bypass Encoder's sigmoid — raw AlexNet activations are returned.
    Dropout layers from the original classifier are excluded for deterministic behaviour.

    Args:
        num_fc_layers: Number of FC layers (excluding fc8) to include.
            0 -> conv features only, output_dim=9216
            1 -> features + fc6,     output_dim=4096  (K=1)
            2 -> features + fc6+fc7, output_dim=4096  (baseline)
    """

    def __init__(self, num_fc_layers: int = 1):
        # Bypass Encoder.__init__ (it expects a BaseNetwork net argument).
        # Call BaseNetwork.__init__ directly instead.
        BaseNetwork.__init__(self)

        if num_fc_layers not in (0, 1, 2):
            raise ValueError(f"num_fc_layers must be 0, 1, or 2, got {num_fc_layers}")

        alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)

        layers = list(alexnet.features)
        layers.append(torch.nn.AdaptiveAvgPool2d((6, 6)))
        layers.append(torch.nn.Flatten(start_dim=-3))

        if num_fc_layers >= 1:
            layers.append(alexnet.classifier[1])  # Linear(9216 -> 4096) = fc6
            layers.append(alexnet.classifier[2])  # ReLU

        if num_fc_layers >= 2:
            layers.append(alexnet.classifier[4])  # Linear(4096 -> 4096) = fc7
            layers.append(alexnet.classifier[5])  # ReLU

        self.net = torch.nn.Sequential(*layers)
        self.output_dim = 4096 if num_fc_layers >= 1 else 9216

        # Freeze — only the EI layer and fc8 decoder are trained
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:  # (C, H, W) — add batch dim
            x = x[None, ...]

        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])

        output: torch.Tensor = self.net(x)
        return output.reshape(*batch_shape, -1)


class AlexNetFCAutoencoder(BaseAutoencoder):
    """
    Wraps AlexNetFCEncoderLayer as a BaseAutoencoder so that setup_utils.py can access
    encoder_network.encoder (AlexNetFCEncoderLayer) with encoder_network.encoder.output_dim.

    Args:
        num_fc_layers: Passed to AlexNetFCEncoderLayer. Default 1 (K=1 condition).
    """

    def __init__(self, num_fc_layers: int = 1, **kwargs):
        super().__init__()

        self.encoder = AlexNetFCEncoderLayer(num_fc_layers=num_fc_layers)
        self.decoder = Identity(input_dim=self.encoder.output_dim)  # unused, satisfies interface
        self.output_dim = self.encoder.output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


__all__ = ["AlexNetFCAutoencoder", "AlexNetFCEncoderLayer"]
