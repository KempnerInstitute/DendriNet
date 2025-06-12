"""
Layer implementations for dendritic networks.

This module organizes all layer types including linear, dendritic,
and MLP layers.
"""

# Linear layers
from .linear import (
    TopKLinear,
    StochasticTopKLinear,
    VarianceTopKLinear,
    DeepstLinear,
    BlockLinear,
    TOPK_INIT_METHODS,
)

# Dendritic layers
from .dendritic import DendriticBranchLayer, ExcitationInhibitionLayer

# Aliases for backward compatibility
StochasticSynapseLinear = StochasticTopKLinear
BlockLinearDeepST = DeepstLinear
BlockLinearVariance = VarianceTopKLinear

__all__ = [
    # Linear layers
    "TopKLinear",
    "StochasticTopKLinear",
    "VarianceTopKLinear",
    "DeepstLinear",
    "BlockLinear",
    "TOPK_INIT_METHODS",
    # Dendritic layers
    "DendriticBranchLayer",
    "ExcitationInhibitionLayer",
    # Backward compatibility aliases
    "StochasticSynapseLinear",
    "BlockLinearDeepST",
    "BlockLinearVariance",
] 