"""
Linear layer implementations for dendritic networks.

This module provides various linear layer variants including TopK selection,
stochastic selection, variance-based selection, dynamic sparse training,
and block linear layers.
"""

from dendritic_modeling.networks.layers.linear.topk import TopKLinear, TOPK_INIT_METHODS
from dendritic_modeling.networks.layers.linear.stochastic import StochasticTopKLinear
from dendritic_modeling.networks.layers.linear.variance import VarianceTopKLinear
from dendritic_modeling.networks.layers.linear.deepst import DeepstLinear
from dendritic_modeling.networks.layers.linear.block import BlockLinear

__all__ = [
    "TopKLinear",
    "StochasticTopKLinear",
    "VarianceTopKLinear",
    "DeepstLinear",
    "BlockLinear",
    "TOPK_INIT_METHODS",
] 