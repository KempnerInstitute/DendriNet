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

try:
    from dendritic_modeling.networks.architectures.recurrent import (
        DendriticRNNCell,
        DendriticRNNNetwork,
        RecurrentState,
    )
except (ImportError, ModuleNotFoundError):
    pass

__all__ = [
    "MLP",
    "AlexNet",
    "BlockLinear",
    "CNNDownsample",
    "CNNUpsample",
    "DeepstLinear",
    "DendriNet",
    "DendriticBranchLayer",
    "DendriticRNNCell",
    "DendriticRNNNetwork",
    "ExcitationInhibitionLayer",
    "ExcitationInhibitionNetwork",
    "GradientScaler",
    "Identity",
    "IdentityInputTransform",
    "IndexInputTransform",
    "MatchedActiveParamMLP",
    "MatchedTotalParamMLP",
    "RecurrentState",
    "SparseStructuredMLP",
    "StochasticTopKLinear",
    "TopKLinear",
    "TransferLayer",
    "VarianceTopKLinear",
    "analytical_expectation_dbl_init",
]
