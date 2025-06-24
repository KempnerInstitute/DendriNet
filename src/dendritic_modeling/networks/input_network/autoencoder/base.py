import torch
import torch.nn as nn

from typing import Tuple, List

from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.architectures.classical.cnn.cnn import CNN
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP


class BaseAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder: BaseNetwork
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
        activation: str = "relu"
    ):
        super().__init__()

        encoder_net = MLP(
            input_dim=input_dim,
            hidden_dims=encoder_hidden_dims,
            activation=activation,
            output_dim=latent_dim
        )

        self.encoder = nn.Sequential(encoder_net, nn.Sigmoid())

        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            output_dim=input_dim
        )


class CNNAutoencoder(BaseAutoencoder):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        encoder_hidden_dims: List[int],
        encoder_kernel_sizes: List[int],
        encoder_strides: List[int],
        latent_dim: int,
        decoder_hidden_dims: List[int],
        activation: str = "relu"
    ):
        super().__init__()

        encoder_net = CNN(
            input_shape=input_shape,
            hidden_dims=encoder_hidden_dims,
            kernel_sizes=encoder_kernel_sizes,
            strides=encoder_strides,
            activation=activation,
            output_dim=latent_dim
        )

        self.encoder = nn.Sequential(encoder_net, nn.Sigmoid())
        
        input_dim = 1
        for dim in input_shape:
            input_dim *= dim
        
        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            output_dim=input_dim
        )
