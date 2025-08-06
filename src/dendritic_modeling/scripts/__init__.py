"""
Scripts Package

This module contains main training and experiment scripts.
"""

from dendritic_modeling.scripts.training.train_experiments import \
    main as train_experiments_main
from dendritic_modeling.scripts.training.train_experiments import run_analysis

__all__ = [
    "run_analysis",
    "train_experiments_main",
]
