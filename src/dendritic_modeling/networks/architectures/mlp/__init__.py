"""
MLP architectures.

This module contains various MLP-based neural network architectures.
"""

from dendritic_modeling.networks.architectures.mlp.matched_param import MatchedParamMLP
from dendritic_modeling.networks.architectures.mlp.simple_mlp import SimpleMLPNet

__all__ = [
    "MatchedParamMLP",
    "SimpleMLPNet",
] 