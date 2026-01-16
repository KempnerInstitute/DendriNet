"""
MLP architectures.

This module contains various MLP-based neural network architectures.
"""

from dendritic_modeling.networks.architectures.classical.mlp.effective_matched import (
    EffectiveMatchedParamMLP,
)
from dendritic_modeling.networks.architectures.classical.mlp.matched_param import (
    MatchedTotalParamMLP,
    MatchedActiveParamMLP,
    SparseStructuredMLP,
)
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP

__all__ = [
    "MLP", 
    "EffectiveMatchedParamMLP", 
    "MatchedTotalParamMLP", 
    "MatchedActiveParamMLP", 
    "SparseStructuredMLP"
]
