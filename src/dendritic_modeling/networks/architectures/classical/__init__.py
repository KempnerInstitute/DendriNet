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
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)

__all__ = [
    "MLP",
    "AlexNet",
    "CNNDownsample",
    "CNNUpsample",
    "Identity",
    "MatchedActiveParamMLP",
    "MatchedTotalParamMLP",
    "PathwayRouter",
    "SparseStructuredMLP",
]
