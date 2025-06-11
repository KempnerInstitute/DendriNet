"""
Network architectures.

This module contains various neural network architectures including
dendritic, excitation-inhibition, and MLP-based models.
"""

from .dendritic import DendriNet
from .excitation_inhibition import ExcitationInhibitionNetwork, MLPExcInhNetwork
from .mlp import MatchedParamMLP, SimpleMLPNet

__all__ = [
    # Dendritic architectures
    "DendriNet",
    # Excitation-Inhibition architectures
    "ExcitationInhibitionNetwork",
    "MLPExcInhNetwork",
    # MLP architectures
    "MatchedParamMLP",
    "SimpleMLPNet",
] 