"""
Excitation-Inhibition Network architectures.

This module contains network architectures based on excitation-inhibition principles.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import ExcitationInhibitionLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import ExcitationInhibitionNetwork

from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform import (
    IdentityInputTransform,
    IndexInputTransform,
    TransferLayer
)

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    TopKLinear,
    StochasticTopKLinear,
    VarianceTopKLinear,
    DeepstLinear
)

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    BlockLinear,
    DendriticBranchLayer,
    DendriNet,
    GradientScaler,
    initialize_dendritic_branch_layer
)

__all__ = [
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
    "initialize_dendritic_branch_layer"
] 