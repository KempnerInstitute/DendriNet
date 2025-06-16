"""
Regression models.

This module contains various regression models including
standard regression and deep sparse training regression.
"""

from dendritic_modeling.models.regression.standard import Regressor
from dendritic_modeling.models.regression.deepst import DeepstRegressor

__all__ = [
    "Regressor",
    "DeepstRegressor",
] 