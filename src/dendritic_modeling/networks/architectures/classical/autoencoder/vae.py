import torch
import torch.nn as nn
from torch.distributions import Normal

from typing import Tuple, List

from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.architectures.classical.cnn.cnn import CNN
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP
from dendritic_modeling.networks.architectures.classical.autoencoder.base import BaseAutoencoder


class VariationalEncoder(BaseNetwork):
    def __init__(
        self,
        net: BaseNetwork,
        latent_dim: int
    ):
        super().__init__()

        self.net = net

        self.fc_mean = nn.Linear(net.output_dim, latent_dim)
        self.fc_std = nn.Sequential(
            nn.Linear(net.output_dim, latent_dim),
            nn.Softplus()
        )

        self.output_dim = latent_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        mean = self.fc_mean(x)
        std = self.fc_std(x)
        distribution = Normal(mean, std)
        return torch.sigmoid(distribution.sample())


class MLPVariationalAutoencoder(BaseAutoencoder):
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
            activation=activation
        )

        self.encoder = VariationalEncoder(encoder_net, latent_dim)

        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            output_dim=input_dim
        )

        self.output_dim = latent_dim


class CNNVariationalAutoencoder(BaseAutoencoder):
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

        encoder_net = CNN(
            input_shape=input_shape,
            hidden_dims=encoder_hidden_dims,
            kernel_sizes=encoder_kernel_sizes,
            strides=encoder_strides,
            activation=activation
        )

        self.encoder = VariationalEncoder(encoder_net, latent_dim)
        
        input_dim = 1
        for dim in input_shape:
            input_dim *= dim
        
        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            output_dim=input_dim
        )

        self.output_dim = latent_dim
