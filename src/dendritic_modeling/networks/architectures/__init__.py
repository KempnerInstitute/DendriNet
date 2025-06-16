"""
Network architectures.

This module contains various neural network architectures including
dendritic, excitation-inhibition, and MLP-based models.
"""

from dendritic_modeling.networks.architectures.dendritic import DendriNet
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork, 
    MLPExcInhNetwork
)
from dendritic_modeling.networks.architectures.mlp import (
    MatchedParamMLP, 
    SimpleMLPNet
)

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