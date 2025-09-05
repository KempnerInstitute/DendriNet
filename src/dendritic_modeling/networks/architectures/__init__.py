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
    MatchedParamMLP,
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
    "analytical_expectation_dbl_init"
    "MatchedParamMLP"
    "MLP"
    "StochasticTopKLinear"
    "TopKLinear"
    "TransferLayer"
    "VarianceTopKLinear"
]
