"""
utils.py
========
This module maintains backward compatibility by importing all utility functions
from the new modular structure in the utils/ directory.

The functions have been organized into specialized modules:
- utils.data: Data handling and serialization utilities
- utils.noise: Noise generation functions
- utils.evaluation: Model evaluation metrics
- utils.hooks: PyTorch hook management utilities
- utils.classification: MLP classifiers and training utilities
- utils.analysis: Advanced neural analysis utilities

For new code, consider importing directly from the specific modules for better clarity.
"""

# Import all utilities from the new modular structure
from .utils import *

# This maintains full backward compatibility with existing code that imports from utils.py
# while providing access to the new organized structure via utils.data, utils.noise, etc. 