"""
Refactored plot manager for dendritic models.

This module contains a slimmed-down PlotManager class that delegates to the
organized analysis and plotting modules, providing backward compatibility
while maintaining clean separation of concerns.
"""

import logging
import matplotlib.pyplot as plt
from typing import Optional, Dict

from dendritic_modeling.plotting.analysis import (
    plot_performance_metrics,
    plot_information_analysis,
)
from dendritic_modeling.plotting.network import (
    plot_dendrinet_gradients,
    plot_einet_params,
)

logger = logging.getLogger(__name__)


class PlotManager:
    """Central manager for generating and handling all plots in the codebase."""

    def __init__(self, device: str = "cpu", default_save_path: Optional[str] = None):
        """Initialize with optional device and default save path."""
        self.device = device
        self.default_save_path = default_save_path

    def plot_einet_params(
        self, einet, save_path: Optional[str] = None, filename: str = "einet_params"
    ):
        """Generate EINet parameter plots."""
        save_path = save_path or self.default_save_path
        plot_einet_params(einet, save_path=save_path, filename=filename)

    def plot_dendrinet_gradients(
        self,
        dendrinet,
        save_path: Optional[str] = None,
        filename: str = "dendrinet_grads",
    ):
        """Generate DendriNet gradient plots."""
        save_path = save_path or self.default_save_path
        plot_dendrinet_gradients(dendrinet, save_path=save_path, filename=filename)

    def plot_performance_metrics(
        self,
        results: Dict,
        save_path: Optional[str] = None,
        filename: str = "performance",
    ):
        """Generate performance metric plots."""
        save_path = save_path or self.default_save_path
        return plot_performance_metrics(results, save_path=save_path, filename=filename)

    def plot_information_analysis(
        self,
        info_results: Dict,
        save_path: Optional[str] = None,
        filename: str = "information",
    ):
        """Generate information analysis plots."""
        save_path = save_path or self.default_save_path
        plot_information_analysis(info_results, save_path=save_path, filename=filename)

    # Add methods for other plots as needed, e.g.:
    # def plot_weights(self, weight_data, save_path=None, filename='weights'):
    #     plot_weight_distributions(weight_data, save_path=save_path or self.default_save_path, filename=filename)

    def close_all(self):
        """Close all open matplotlib figures to free memory."""
        plt.close("all")


__all__ = ["PlotManager"]
