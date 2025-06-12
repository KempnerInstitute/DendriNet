"""
Regression models.

This module contains various regression models including
standard regression and deep sparse training regression.
"""

from .standard import Regressor
from .deepst import DeepstRegressor

__all__ = [
    "Regressor",
    "DeepstRegressor",
] 