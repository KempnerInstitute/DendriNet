"""
Layers package for dendritic modeling.

This package contains various neural network layer implementations specialized
for dendritic computation, including TopK variants and specialized linear layers.
"""

from .linear import *

__all__ = [
    # Linear layer variants
    "TopKLinear",
    "TOPK_INIT_METHODS",
]
