"""
Baseline models.

This module contains baseline models for comparison purposes.
"""

from dendritic_modeling.models.baselines.mlp import SimpleMLP, SimpleClassifier, MLPWithClassificationHead

__all__ = [
    "SimpleMLP",
    "SimpleClassifier",
    "MLPWithClassificationHead",
] 