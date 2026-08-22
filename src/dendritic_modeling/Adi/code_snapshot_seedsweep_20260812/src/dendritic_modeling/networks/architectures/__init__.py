"""
Network architectures.

This module contains various neural network architectures including
dendritic, excitation-inhibition, and MLP-based models.
"""

from dendritic_modeling.networks.architectures.classical import (
    MLP,
    AlexNet,
    CNNDownsample,
    CNNUpsample,
    Identity,
    MatchedActiveParamMLP,
    MatchedTotalParamMLP,
    PathwayRouter,
    SparseStructuredMLP,
)
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    BlockLinear,
    DeepstLinear,
    DendriNet,
    DendriticBranchLayer,
    ExcitationInhibitionLayer,
    ExcitationInhibitionNetwork,
    GradientScaler,
    IdentityInputTransform,
    IndexInputTransform,
    StochasticTopKLinear,
    TopKLinear,
    TransferLayer,
    VarianceTopKLinear,
    analytical_expectation_dbl_init,
)
from dendritic_modeling.networks.architectures.recurrent import (
    DendriNetState,
    EILayer,
    EILayerConfig,
    EINetwork,
    EINetworkConfig,
    EIState,
    PopulationConfig,
    StatefulDendriNet,
)

__all__ = [
    "MLP",
    "AlexNet",
    "BlockLinear",
    "CNNDownsample",
    "CNNUpsample",
    "DeepstLinear",
    "DendriNet",
    "DendriNetState",
    "DendriticBranchLayer",
    "EILayer",
    "EILayerConfig",
    "EINetwork",
    "EINetworkConfig",
    "EIState",
    "ExcitationInhibitionLayer",
    "ExcitationInhibitionNetwork",
    "GradientScaler",
    "Identity",
    "IdentityInputTransform",
    "IndexInputTransform",
    "MatchedActiveParamMLP",
    "MatchedTotalParamMLP",
    "PathwayRouter",
    "PopulationConfig",
    "SparseStructuredMLP",
    "StatefulDendriNet",
    "StochasticTopKLinear",
    "TopKLinear",
    "TransferLayer",
    "VarianceTopKLinear",
    "analytical_expectation_dbl_init",
]
