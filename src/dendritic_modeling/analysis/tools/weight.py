"""
Synaptic weight analysis module.

This module contains classes for analyzing the statistical distribution of synaptic weights
across the dendritic network, including excitatory weights, inhibitory weights, and branch weights.
"""

from typing import Optional

import torch
import os

from dendritic_modeling.config import WeightAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.utils import save_dict


class WeightAnalyzer:
    """
    Analyzer for computing statistical distributions of synaptic weights.

    Analyzes the weight distributions of excitatory synapses, inhibitory synapses,
    and branch-to-output connections across all dendritic branch layers in the network.
    """

    def __init__(self, params: WeightAnalysisParams):
        """
        Initialize the weight analyzer.

        Args:
            synapse_weights: Whether to analyze excitatory and inhibitory synapse weights
            branch_weights: Whether to analyze branch-to-output connection weights
        """
        self.synapse_weights = getattr(params, "synapse_weights", False)
        self.branch_weights = getattr(params, "branch_weights", False)

    def analyze(
        self, model: BaseModel, save_path: Optional[str] = None, filename: str = "final"
    ):
        """
        Analyze weight distributions in the model.

        Args:
            model: The model to analyze (must have ExcitationInhibitionNetwork)
            save_path: Path to save results (optional)
            filename: Filename for saved results

        Returns:
            Dictionary containing weight statistics if save_path is None, otherwise None
        """
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        # Collect weights from all dendritic branch layers
        exc_weights = []
        inh_weights = []
        branch_weights = []

        for _name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                # Collect excitatory synapse weights
                if self.synapse_weights and module.branch_excitation is not None:
                    exc_w = module.branch_excitation.pruned_weight().flatten()
                    exc_w = exc_w[exc_w > 1e-3].cpu()  # Filter out very small weights
                    exc_weights.append(exc_w)

                # Collect inhibitory synapse weights
                if self.synapse_weights and module.branch_inhibition is not None:
                    inh_w = module.branch_inhibition.pruned_weight().flatten()
                    inh_w = inh_w[inh_w > 1e-3].cpu()  # Filter out very small weights
                    inh_weights.append(inh_w)

                # Collect branch-to-output weights
                if self.branch_weights and hasattr(module, "branches_to_output"):
                    branch_w = module.branches_to_output.log_weight.exp().flatten()
                    branch_w = branch_w.cpu()
                    branch_weights.append(branch_w)

        weights = {}

        # Compute statistics for excitatory weights
        if len(exc_weights) > 0:
            exc_weights = torch.cat(exc_weights, dim=0)
            weights["exc weights"] = {}
            weights["exc weights"]["mean"] = exc_weights.mean().item()
            weights["exc weights"]["std"] = exc_weights.std().item()
            weights["exc weights"]["min"] = exc_weights.min().item()
            weights["exc weights"]["max"] = exc_weights.max().item()
            weights["exc weights"]["median"] = exc_weights.median().item()
            weights["exc weights"]["q1"] = exc_weights.quantile(q=0.25).item()
            weights["exc weights"]["q3"] = exc_weights.quantile(q=0.75).item()

        # Compute statistics for inhibitory weights
        if len(inh_weights) > 0:
            inh_weights = torch.cat(inh_weights, dim=0)
            weights["inh weights"] = {}
            weights["inh weights"]["mean"] = inh_weights.mean().item()
            weights["inh weights"]["std"] = inh_weights.std().item()
            weights["inh weights"]["min"] = inh_weights.min().item()
            weights["inh weights"]["max"] = inh_weights.max().item()
            weights["inh weights"]["median"] = inh_weights.median().item()
            weights["inh weights"]["q1"] = inh_weights.quantile(q=0.25).item()
            weights["inh weights"]["q3"] = inh_weights.quantile(q=0.75).item()

        # Compute statistics for branch weights
        if len(branch_weights) > 0:
            branch_weights = torch.cat(branch_weights, dim=0)
            weights["branch weights"] = {}
            weights["branch weights"]["mean"] = branch_weights.mean().item()
            weights["branch weights"]["std"] = branch_weights.std().item()
            weights["branch weights"]["min"] = branch_weights.min().item()
            weights["branch weights"]["max"] = branch_weights.max().item()
            weights["branch weights"]["median"] = branch_weights.median().item()
            weights["branch weights"]["q1"] = branch_weights.quantile(q=0.25).item()
            weights["branch weights"]["q3"] = branch_weights.quantile(q=0.75).item()

        if save_path is not None:
            save_dict(weights, save_path, filename)

            # Generate weight distribution plots
            try:
                from dendritic_modeling.plotting.analysis import plot_weight_distributions
                import matplotlib.pyplot as plt
                import logging
                import traceback

                plot_save_path = os.path.join(save_path, "plots")
                os.makedirs(plot_save_path, exist_ok=True)

                # Prepare weight data for plotting
                weight_data = {}
                if len(exc_weights) > 0:
                    weight_data["Excitatory"] = exc_weights.detach().numpy()
                if len(inh_weights) > 0:
                    weight_data["Inhibitory"] = inh_weights.detach().numpy()
                if len(branch_weights) > 0:
                    weight_data["Branch"] = branch_weights.detach().numpy()

                if weight_data:
                    try:
                        # Create weight distribution plot
                        logging.info(f"Calling plot_weight_distributions with {len(weight_data)} weight types")
                        result = plot_weight_distributions(
                            weight_data,
                            save_path=plot_save_path,
                            filename=filename
                        )
                        logging.info(f"plot_weight_distributions returned type: {type(result)}")
                        # result is a tuple (fig, ax)
                        if result is not None:
                            if isinstance(result, tuple) and len(result) == 2:
                                fig, ax = result
                                plt.close(fig)
                            else:
                                logging.warning(f"Unexpected return type from plot_weight_distributions: {type(result)}")
                    except Exception as e:
                        logging.error(f"Error in plot_weight_distributions: {e}")
                        import traceback
                        logging.error(traceback.format_exc())

                    # Also create histogram plots
                    fig, axes = plt.subplots(1, len(weight_data), figsize=(5*len(weight_data), 4))
                    if len(weight_data) == 1:
                        axes = [axes]

                    for ax, (name, weights_array) in zip(axes, weight_data.items()):
                        ax.hist(weights_array, bins=50, alpha=0.7, edgecolor='black')
                        ax.set_xlabel('Weight Value')
                        ax.set_ylabel('Count')
                        ax.set_title(f'{name} Weight Distribution')
                        ax.grid(True, alpha=0.3)

                    plt.tight_layout()
                    hist_path = os.path.join(plot_save_path, f"{filename}_weight_histograms.png")
                    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
                    plt.close()

            except Exception as e:
                import logging
                logging.warning(f"Could not generate weight plots: {e}")
        else:
            return weights
