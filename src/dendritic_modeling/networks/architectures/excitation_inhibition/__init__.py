"""
Excitation-Inhibition Network architectures.

This module contains network architectures based on excitation-inhibition principles.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    BlockLinear,
    DendriNet,
    DendriticBranchLayer,
    GradientScaler,
    analytical_expectation_dbl_init,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_layer import (
    ExcitationInhibitionLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform import (
    IdentityInputTransform,
    IndexInputTransform,
    TransferLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    DeepstLinear,
    StochasticTopKLinear,
    TopKLinear,
    VarianceTopKLinear,
)

__all__ = [
    "BlockLinear",
    "ConfigurableEINetwork",
    "DeepstLinear",
    "DendriNet",
    "DendriticBranchLayer",
    "ExcitationInhibitionLayer",
    "ExcitationInhibitionNetwork",
    "GradientScaler",
    "IdentityInputTransform",
    "IndexInputTransform",
    "StochasticTopKLinear",
    "TopKLinear",
    "TransferLayer",
    "VarianceTopKLinear",
    "analytical_expectation_dbl_init",
]
