from dendritic_modeling.networks.architectures.classical.cnn import (
    AlexNet,
    CNNDownsample,
    CNNUpsample,
)
from dendritic_modeling.networks.architectures.classical.identity import Identity
from dendritic_modeling.networks.architectures.classical.mlp import (
    MLP, 
    MatchedTotalParamMLP, 
    MatchedActiveParamMLP,
    SparseStructuredMLP,
)

__all__ = [
    "AlexNet",
    "CNNDownsample",
    "CNNUpsample",
    "Identity",
    "MatchedTotalParamMLP",
    "MatchedActiveParamMLP",
    "MLP",
    "SparseStructuredMLP",
]
