import torch
import torch.nn as nn
from torch.distributions import Normal

from typing import Tuple, List

from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.architectures.classical.cnn.cnn import (
    CNNDownsample,
    CNNUpsample
)
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

        fc_mean = []
        fc_std = []

        if isinstance(net.output_dim, tuple):
            net_output_dim = 1
            for dim in net.output_dim:
                net_output_dim *= dim
            fc_mean.append(nn.Flatten(start_dim=-3))
            fc_std.append(nn.Flatten(start_dim=-3))
        else:
            net_output_dim = net.output_dim

        fc_mean.append(nn.Linear(net_output_dim, latent_dim))
        fc_std.append(nn.Linear(net_output_dim, latent_dim))
        fc_std.append(nn.Softplus())

        self.fc_mean = nn.Sequential(*fc_mean)
        self.fc_std = nn.Sequential(*fc_std)

        self.output_dim = latent_dim
    
    def latent_distribution(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(x)
        z_mu = self.fc_mean(x)
        z_sigma = self.fc_std(x)
        return z_mu, z_sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z_mu, z_sigma = self.latent_distribution(x)
        z = torch.sigmoid(Normal(z_mu, z_sigma).sample())
        return z


# assumes input space is [0, 1]
class BaseVariationalAutoencoder(BaseAutoencoder):
    def __init__(self):
        super().__init__()
        self.encoder: VariationalEncoder
    

class MLPVariationalAutoencoder(BaseVariationalAutoencoder):
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

        decoder_net = MLP(
            input_dim=latent_dim,
            hidden_dims=decoder_hidden_dims,
            activation=activation,
            output_dim=input_dim
        )
        self.decoder = nn.Sequential(
            decoder_net,
            nn.Sigmoid()
        )

        self.output_dim = latent_dim


class CNNVariationalAutoencoder(BaseVariationalAutoencoder):
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
            activation=activation
        )

        self.encoder = VariationalEncoder(encoder_net, latent_dim)
        
        decoder_net = CNNUpsample(
            input_shape=(latent_dim, 1, 1),
            target_shape=input_shape,
            hidden_dims=decoder_hidden_dims,
            activation=activation
        )
        self.decoder = nn.Sequential(
            nn.Unflatten(dim=-1, unflattened_size=(latent_dim, 1, 1)),
            decoder_net,
            nn.Sigmoid()
        )
        
        self.output_dim = input_shape
