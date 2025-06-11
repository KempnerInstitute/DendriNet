"""
Linear layer implementations for dendritic networks.

This module provides various linear layer variants including TopK selection,
stochastic selection, variance-based selection, dynamic sparse training,
and block linear layers.
"""

from .topk import TopKLinear, TOPK_INIT_METHODS
from .stochastic import StochasticTopKLinear
from .variance import VarianceTopKLinear
from .deepst import DeepstLinear
from .block import BlockLinear

__all__ = [
    "TopKLinear",
    "StochasticTopKLinear",
    "VarianceTopKLinear",
    "DeepstLinear",
    "BlockLinear",
    "TOPK_INIT_METHODS",
] 