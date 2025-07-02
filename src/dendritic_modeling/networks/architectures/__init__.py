"""
Network architectures.

This module contains various neural network architectures including
dendritic, excitation-inhibition, and MLP-based models.
"""

from dendritic_modeling.networks.architectures.classical import (
    AlexNet,
    CNNUpsample,
    CNNDownsample,
    MLP,
    MatchedParamMLP,
    Identity
)

from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionLayer,
    ExcitationInhibitionNetwork,
    IdentityInputTransform,
    IndexInputTransform,
    TransferLayer,
    TopKLinear,
    StochasticTopKLinear,
    VarianceTopKLinear,
    DeepstLinear,
    BlockLinear,
    DendriticBranchLayer,
    DendriNet,
    GradientScaler,
    initialize_dendritic_branch_layer
)

__all__ = [
    "AlexNet",
    "CNNUpsample",
    "CNNDownsample",
    "MLP",
    "MatchedParamMLP",
    "ExcitationInhibitionLayer",
    "ExcitationInhibitionNetwork",
    "IdentityInputTransform",
    "IndexInputTransform",
    "TransferLayer",
    "TopKLinear",
    "StochasticTopKLinear",
    "VarianceTopKLinear",
    "DeepstLinear",
    "BlockLinear",
    "DendriticBranchLayer",
    "DendriNet",
    "GradientScaler",
    "initialize_dendritic_branch_layer",
    "Identity"
]