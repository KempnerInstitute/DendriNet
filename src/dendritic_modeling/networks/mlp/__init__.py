"""
Multi-Layer Perceptron components.

This module contains various MLP-based components for neural networks.
"""

from dendritic_modeling.networks.mlp.feature_extractor import FeatureExtractorMLP
from dendritic_modeling.networks.mlp.simple_layer import SimpleMLPLayer

__all__ = [
    "FeatureExtractorMLP",
    "SimpleMLPLayer",
] 