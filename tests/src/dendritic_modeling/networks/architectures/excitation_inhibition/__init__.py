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
    DenseToSparseLinear,
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
    StochasticTopKLinear,
    TopKLinear,
    VarianceTopKLinear,
    distance_kernel_probabilities,
    normalize_pathway_name,
    sample_configured_mask,
    sample_distance_kernel_mask,
    sample_fixed_indegree_mask,
    sample_probability_mask,
    validate_connection_mask,
)

__all__ = [
    "BlockLinear",
    "ConfigurableEINetwork",
    "DeepstLinear",
    "DendriNet",
    "DendriticBranchLayer",
    "DenseToSparseLinear",
    "ExcitationInhibitionLayer",
    "ExcitationInhibitionNetwork",
    "GradientScaler",
    "IdentityInputTransform",
    "IndexInputTransform",
    "IndexedDynamicTopKLinear",
    "IndexedSparseLinear",
    "StochasticTopKLinear",
    "TopKLinear",
    "TransferLayer",
    "VarianceTopKLinear",
    "analytical_expectation_dbl_init",
    "distance_kernel_probabilities",
    "normalize_pathway_name",
    "sample_configured_mask",
    "sample_distance_kernel_mask",
    "sample_fixed_indegree_mask",
    "sample_probability_mask",
    "validate_connection_mask",
]
