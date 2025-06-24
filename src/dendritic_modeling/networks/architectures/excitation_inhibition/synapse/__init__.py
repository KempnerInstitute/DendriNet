"""
Linear layer implementations for dendritic networks.

This module provides various linear layer variants including TopK selection,
stochastic selection, variance-based selection, dynamic sparse training,
and block linear layers.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import TopKLinear, TOPK_INIT_METHODS
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.stochastic import StochasticTopKLinear
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import VarianceTopKLinear
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import DeepstLinear

__all__ = [
    "TopKLinear",
    "StochasticTopKLinear",
    "VarianceTopKLinear",
    "DeepstLinear",
    "TOPK_INIT_METHODS"
] 