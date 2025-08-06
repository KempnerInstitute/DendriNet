"""
Dendritic Modeling Package
==========================

A comprehensive library for dendritic neural network modeling and analysis.

Package Structure:
- analysis/: Analysis tools for network behavior and information flow
- config/: Configuration management and dataclasses
- datasets/: Synthetic and real dataset implementations
- models/: Model wrappers (classification, regression, baselines)
- networks/: Network architectures, layers, and activations
- plotting/: Visualization and plotting utilities
- scripts/: Main training and experiment scripts
- training/: Training strategies and optimizers
- utils/: General utility functions

For detailed documentation on each module, see the respective module docstrings.
"""

# Core imports
from dendritic_modeling import datasets
from dendritic_modeling import (analysis, config, models, networks, plotting,
                                scripts, training, utils)

# Set up the logging environment
from dendritic_modeling.utils.logging_config import LoggerManager

logger_manager = LoggerManager()
logger = logger_manager.get_logger()

# Version information
__version__ = "0.1.0"

__all__ = [
    "__version__"
    "analysis"
    "config"
    "datasets"
    "logger"
    "models"
    "networks"
    "plotting"
    "scripts"
    "training"
    "utils"
]

# from dendritic_modeling import pid_utils
# from dendritic_modeling.information import (
#     build_sr_matrix,
#     compute_ei_mi,
#     compute_mi_cc,
#     compute_mi_cd,
#     compute_pairwise_sr,
#     compute_synaptic_information,
# )

# Import utility functions
# from dendritic_modeling.utils import (
#     # Utility functions would be imported here if needed
# )
