from dendritic_modeling.networks.architectures.classical.cnn import (
    AlexNet,
    CNNDownsample,
    CNNUpsample,
)
from dendritic_modeling.networks.architectures.classical.identity import Identity
from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP,
    MatchedActiveParamMLP,
    MatchedTotalParamMLP,
    SparseStructuredMLP,
)

__all__ = [
    "MLP",
    "AlexNet",
    "CNNDownsample",
    "CNNUpsample",
    "Identity",
    "MatchedActiveParamMLP",
    "MatchedTotalParamMLP",
    "SparseStructuredMLP",
]
