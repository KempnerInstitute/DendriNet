import torch
import torch.nn as nn

from typing import Tuple, List

from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.architectures.classical.cnn.cnn import (
    CNNDownsample,
    CNNUpsample
)
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP


class Encoder(BaseNetwork):
    def __init__(self, net: BaseNetwork):
        super().__init__()
        self.net = net
        self.output_dim = net.output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class BaseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder: Encoder
        self.decoder: BaseNetwork
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed


class MLPAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        input_dim: int,
        encoder_hidden_dims: List[int],
        latent_dim: int,
        decoder_hidden_dims: List[int],
        activation: str = "relu",
        **kwargs
    ):
        super().__init__()

        encoder_net = MLP(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            output_dim=latent_dim
        )
        self.encoder = Encoder(encoder_net)

        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            output_dim=input_dim
        )

        self.output_dim = latent_dim


class CNNAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        encoder_hidden_dims: List[int],
        encoder_kernel_sizes: List[int],
        encoder_strides: List[int],
        latent_dim: int,
        decoder_hidden_dims: List[int],
        activation: str = "relu",
        **kwargs
    ):
        super().__init__()

        encoder_net = CNNDownsample(
            input_shape=input_shape,
            hidden_dims=encoder_hidden_dims,
            kernel_sizes=encoder_kernel_sizes,
            strides=encoder_strides,
            activation=activation,
            output_dim=latent_dim
        )
        self.encoder = Encoder(encoder_net)
        
        decoder_net = CNNUpsample(
            input_shape=(latent_dim, 1, 1),
            target_shape=input_shape,
            hidden_dims=decoder_hidden_dims,
            activation=activation
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(latent_dim, 1, 1)),
            decoder_net
        )

        self.output_dim = tuple(input_shape)
