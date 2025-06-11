"""
MLP architectures.

This module contains various MLP-based neural network architectures.
"""

from .matched_param import MatchedParamMLP
from .simple_mlp import SimpleMLPNet

__all__ = [
    "MatchedParamMLP",
    "SimpleMLPNet",
] 