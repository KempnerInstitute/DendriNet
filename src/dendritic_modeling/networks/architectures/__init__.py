"""
Network architectures.

This module contains various neural network architectures including
dendritic, excitation-inhibition, and MLP-based models.
"""

from dendritic_modeling.networks.architectures.classical import (
    MLP, AlexNet, CNNDownsample, CNNUpsample, Identity, MatchedParamMLP)
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    BlockLinear, DeepstLinear, DendriNet, DendriticBranchLayer,
    ExcitationInhibitionLayer, ExcitationInhibitionNetwork, GradientScaler,
    IdentityInputTransform, IndexInputTransform, StochasticTopKLinear,
    TopKLinear, TransferLayer, VarianceTopKLinear,
    initialize_dendritic_branch_layer)

__all__ = [
    "AlexNet"
    "BlockLinear"
    "CNNDownsample"
    "CNNUpsample"
    "DeepstLinear"
    "DendriNet"
    "DendriticBranchLayer"
    "ExcitationInhibitionLayer"
    "ExcitationInhibitionNetwork"
    "GradientScaler"
    "Identity"
    "IdentityInputTransform"
    "IndexInputTransform"
    "initialize_dendritic_branch_layer"
    "MatchedParamMLP"
    "MLP"
    "StochasticTopKLinear"
    "TopKLinear"
    "TransferLayer"
    "VarianceTopKLinear"
]
