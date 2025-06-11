"""
plot_utils.py
=============
This module maintains backward compatibility by importing all plotting functions
from the new organized structure in the plotting/ directory.

The functions have been organized into specialized modules:
- plotting.training: Training-related plots (loss curves, parameter sweeps)
- plotting.network: Network visualization (parameters, weights, gradients)
- plotting.analysis: Analysis plots (activations, tuning curves, profiles)
- plotting.utils: Utility functions (CSV export, tree visualization)

For new code, consider importing directly from the specific modules for better clarity.
"""

# Import all plotting functions from the new organized structure
from .plotting import *

# This maintains full backward compatibility with existing code that imports from plot_utils.py
# while providing access to the new organized structure via plotting.training, plotting.network, etc.
