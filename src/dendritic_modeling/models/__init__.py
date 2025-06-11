"""
Models package for dendritic modeling.

This package contains model wrapper classes:
- classification: Classification models (probabilistic, standard, simple)
- regression: Regression models (standard, DEEPST)
- baselines: Baseline models for comparison
"""

# Import base model class
from .base import BaseModel

# Backward compatibility - import remaining classes from original models.py
from ..models import (
    ProbabilisticClassifier,
    Classifier, 
    Regressor,
    DeepstRegressor,
    SimpleMLP,
    SimpleClassifier,
    MLPWithClassificationHead
) 