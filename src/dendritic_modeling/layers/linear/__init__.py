"""
Linear layers for dendritic networks.

This package contains various linear layer implementations specialized for
dendritic computation, including different TopK selection strategies.
"""

# Basic TopK layer
from .topk import TopKLinear, TOPK_INIT_METHODS

# TODO: Import other variants when implemented
# from .stochastic import StochasticTopKLinear  
# from .variance import VarianceTopKLinear
# from .deepst import DeepstLinear
# from .block import BlockLinear

__all__ = [
    "TopKLinear",
    "TOPK_INIT_METHODS",
    # TODO: Add other variants
    # "StochasticTopKLinear",
    # "VarianceTopKLinear", 
    # "DeepstLinear",
    # "BlockLinear",
]
