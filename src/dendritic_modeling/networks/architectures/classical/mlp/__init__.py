"""
MLP architectures.

This module contains various MLP-based neural network architectures.
"""
from dendritic_modeling.networks.architectures.classical.mlp.matched_param import (
    MatchedParamMLP,
)
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP

__all__ = [
    "MLP",
    "MatchedParamMLP",
]
