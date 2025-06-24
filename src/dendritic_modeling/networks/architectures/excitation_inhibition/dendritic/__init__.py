"""
Dendritic network architectures.

This module provides complete dendritic network implementations including
DendriNet and DendriNetWithOutputs.
"""

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import BlockLinear
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import DendriNet
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.gradient_scaling import GradientScaler
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import initialize_dendritic_branch_layer

__all__ = [
    "BlockLinear",
    "DendriticBranchLayer",
    "DendriNet",
    "GradientScaler",
    "initialize_dendritic_branch_layer"
]