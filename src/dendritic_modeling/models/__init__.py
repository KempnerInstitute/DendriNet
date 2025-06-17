"""
Models module.

This module contains various model implementations including
classification, regression, and baseline models.
"""

# Base model
from dendritic_modeling.models.base import BaseModel

# Classification models
from dendritic_modeling.models.classifier import Classifier

# Regression models
from dendritic_modeling.models.regressor import Regressor

# Baseline models
from dendritic_modeling.models.baselines import (
    SimpleMLP, 
    SimpleClassifier, 
    MLPWithClassificationHead
)

__all__ = [
    # Base
    "BaseModel",
    # Classification
    "Classifier",
    # Regression
    "Regressor",
    # Baselines
    "SimpleMLP",
    "SimpleClassifier",
    "MLPWithClassificationHead",
] 