"""
Linear layer implementations for dendritic networks.

This module provides various linear layer variants including TopK selection,
stochastic selection, variance-based selection, dynamic sparse training,
and block linear layers.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
    DeepstLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.stochastic import (
    StochasticTopKLinear,
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
    "StochasticTopKLinear",
    "TopKLinear",
    "VarianceTopKLinear",
]
