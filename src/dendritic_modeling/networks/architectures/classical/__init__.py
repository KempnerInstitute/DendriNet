from dendritic_modeling.networks.architectures.classical.cnn import (
    AlexNet,
    CNN
)

from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP,
    MatchedParamMLP
)

from dendritic_modeling.networks.architectures.classical.identity import Identity

__all__ = [
    "AlexNet",
    "CNN",
    "MLP",
    "MatchedParamMLP",
    "Identity"
]