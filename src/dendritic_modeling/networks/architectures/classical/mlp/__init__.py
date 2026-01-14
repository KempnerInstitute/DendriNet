"""
MLP architectures.

This module contains various MLP-based neural network architectures.
"""

from dendritic_modeling.networks.architectures.classical.mlp.effective_matched import (
    EffectiveMatchedParamMLP,
)
from dendritic_modeling.networks.architectures.classical.mlp.matched_param import (
    MatchedParamMLP,
)
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP
from dendritic_modeling.networks.architectures.classical.mlp.point_mlp import (
    PointMatchedParamMLP,
)

__all__ = ["MLP", "EffectiveMatchedParamMLP", "MatchedParamMLP", "PointMatchedParamMLP"]
