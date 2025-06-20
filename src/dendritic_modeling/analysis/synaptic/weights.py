"""
Synaptic weight analysis module.

This module contains classes for analyzing the statistical distribution of synaptic weights
across the dendritic network, including excitatory weights, inhibitory weights, and branch weights.
"""

from typing import Optional
import torch
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.networks.layers.dendritic import DendriticBranchLayer
from dendritic_modeling.utils import save_dict
from dendritic_modeling.config.analysis import WeightAnalysisParams


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
        self.synapse_weights = params.synapse_weights
        self.branch_weights = params.branch_weights

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
            assert isinstance(model.net, ExcitationInhibitionNetwork)
        except AssertionError:
            return

        # Collect weights from all dendritic branch layers
        exc_weights = []
        inh_weights = []
        branch_weights = []

        for name, module in model.named_modules():
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
        else:
            return weights 