"""
MLP architectures.

This module contains various MLP-based neural network architectures.
"""
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP
from dendritic_modeling.networks.architectures.classical.mlp.matched_param import MatchedParamMLP

__all__ = [
    "MLP",
    "MatchedParamMLP",
]
