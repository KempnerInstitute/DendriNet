"""
Linear layer implementations for dendritic networks.

This module provides various linear layer variants including TopK selection,
stochastic selection, variance-based selection, dynamic sparse training,
and block linear layers.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
    DeepstLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed import (
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.stochastic import (
    StochasticTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    distance_kernel_probabilities,
    normalize_pathway_name,
    sample_configured_mask,
    sample_distance_kernel_mask,
    sample_fixed_indegree_mask,
    sample_probability_mask,
    validate_connection_mask,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TOPK_INIT_METHODS,
    TopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import (
    VarianceTopKLinear,
)

__all__ = [
    "TOPK_INIT_METHODS",
    "DeepstLinear",
    "IndexedDynamicTopKLinear",
    "IndexedSparseLinear",
    "StochasticTopKLinear",
    "TopKLinear",
    "VarianceTopKLinear",
    "distance_kernel_probabilities",
    "normalize_pathway_name",
    "sample_configured_mask",
    "sample_distance_kernel_mask",
    "sample_fixed_indegree_mask",
    "sample_probability_mask",
    "validate_connection_mask",
]
