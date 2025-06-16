"""
Models module.

This module contains various model implementations including
classification, regression, and baseline models.
"""

# Base model
from dendritic_modeling.models.base import BaseModel

# Classification models
from dendritic_modeling.models.classification import (
    Classifier, 
    ProbabilisticClassifier
)

# Regression models
from dendritic_modeling.models.regression import (
    Regressor, 
    DeepstRegressor
)

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
    "ProbabilisticClassifier",
    # Regression
    "Regressor",
    "DeepstRegressor",
    # Baselines
    "SimpleMLP",
    "SimpleClassifier",
    "MLPWithClassificationHead",
] 