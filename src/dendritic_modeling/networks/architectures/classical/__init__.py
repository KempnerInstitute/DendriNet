from dendritic_modeling.networks.architectures.classical.cnn import (
    AlexNet,
    CNNUpsample,
    CNNDownsample
)

from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP,
    MatchedParamMLP
)

from dendritic_modeling.networks.architectures.classical.identity import Identity

__all__ = [
    "AlexNet",
    "CNNUpsample",
    "CNNDownsample",
    "MLP",
    "MatchedParamMLP",
    "Identity"
]
