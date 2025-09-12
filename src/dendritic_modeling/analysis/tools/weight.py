"""
Unified Weight Analysis Module.

This module provides comprehensive analysis of synaptic weights across the dendritic network,
including both global statistics and hierarchical per-layer analysis with plotting capabilities.
Combines functionality from the original WeightAnalyzer and SynapticWeightExpectationAnalyzer.
"""

import logging
import time
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config import WeightAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


class WeightAnalyzer(AbstractAnalyzer):
    """
    Unified analyzer for comprehensive synaptic weight analysis.
    
    Provides both global weight statistics and hierarchical per-layer analysis
    with enhanced plotting capabilities including min/max and dendritic strength plots.
    """

    def __init__(self, params: WeightAnalysisParams):
        """Initialize the unified weight analyzer."""
        super().__init__("WeightAnalyzer")
        
        # Original WeightAnalyzer functionality
        self.synapse_weights = getattr(params, "synapse_weights", True)
        self.branch_weights = getattr(params, "branch_weights", True)
        
        # Hierarchical analysis options (from SynapticWeightExpectationAnalyzer)
        self.computation_level = getattr(params, "computation_level", "single_branch")
        self.branch_aggregation = getattr(params, "branch_aggregation", "mean")
        
        # Enhanced analysis modes
        self.per_layer_analysis = getattr(params, "per_layer_analysis", True)
        self.per_einet_analysis = getattr(params, "per_einet_analysis", True)
        self.per_neuron_analysis = getattr(params, "per_neuron_analysis", False)
        
        # Weight types to analyze (granular control)
        self.analyze_excitatory = getattr(params, "analyze_excitatory", True)
        self.analyze_inhibitory = getattr(params, "analyze_inhibitory", True)
        self.analyze_branch_output = getattr(params, "analyze_branch_output", True)
        
        # Statistical measures to compute
        self.compute_mean = getattr(params, "compute_mean", True)
        self.compute_variance = getattr(params, "compute_variance", True)
        self.compute_percentiles = getattr(params, "compute_percentiles", True)
        self.compute_min_max = getattr(params, "compute_min_max", True)
        self.compute_dendritic_strength = getattr(params, "compute_dendritic_strength", True)
        
        # Weight filtering
        self.weight_threshold = getattr(params, "weight_threshold", 1e-6)
        
        self.logger.info(f"Initialized unified WeightAnalyzer with computation_level: {self.computation_level}")

    def analyze(
        self, model: BaseModel, save_path: Optional[str] = None, filename: str = "final"
    ):
        """
        Perform comprehensive weight analysis including global and hierarchical statistics.
        
        Args:
            model: The model to analyze (must have ExcitationInhibitionNetwork)
            save_path: Path to save results (optional)
            filename: Filename for saved results
            
        Returns:
            Dictionary containing all weight analysis results
        """
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            self.logger.error("Model must have ExcitationInhibitionNetwork for weight analysis")
            return None

        self.logger.info("Starting comprehensive weight analysis...")
        start_time = time.time()

        results = {}
        
        # Global weight statistics (original WeightAnalyzer functionality)
        if self.synapse_weights or self.branch_weights:
            global_stats = self._compute_global_statistics(model)
            results["global_statistics"] = global_stats
            
        # Hierarchical per-layer analysis (from SynapticWeightExpectationAnalyzer)
        if self.per_layer_analysis:
            layer_stats = self._compute_layer_statistics(model)
            results["layer_statistics"] = layer_stats
            
        # Enhanced per-layer min/max and dendritic strength analysis
        if self.compute_min_max or self.compute_dendritic_strength:
            enhanced_stats = self._compute_enhanced_layer_statistics(model)
            results["enhanced_layer_statistics"] = enhanced_stats

        elapsed_time = time.time() - start_time
        self.logger.info(f"Weight analysis completed in {elapsed_time:.2f} seconds")

        # Save results and generate plots
        if save_path is not None:
            self.logger.info(f"Saving weight analysis results to {save_path}/{filename}")
            save_dict(results, save_path, filename)
            
            # Generate all plots
            self._generate_all_plots(results, save_path, model)
            
        return results

    def _compute_global_statistics(self, model: BaseModel) -> dict[str, Any]:
        """Compute global weight statistics across the entire network."""
        self.logger.info("Computing global weight statistics...")
        
        # Collect weights from all dendritic branch layers
        exc_weights = []
        inh_weights = []
        branch_weights = []

        for _name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                # Collect excitatory synapse weights
                if self.synapse_weights and self.analyze_excitatory and module.branch_excitation is not None:
                    exc_w = module.branch_excitation.pruned_weight().flatten()
                    exc_w = exc_w[exc_w > self.weight_threshold].cpu()
                    exc_weights.append(exc_w)

                # Collect inhibitory synapse weights
                if self.synapse_weights and self.analyze_inhibitory and module.branch_inhibition is not None:
                    inh_w = module.branch_inhibition.pruned_weight().flatten()
                    inh_w = inh_w[inh_w > self.weight_threshold].cpu()
                    inh_weights.append(inh_w)

                # Collect branch-to-output weights
                if self.branch_weights and self.analyze_branch_output and hasattr(module, "branches_to_output"):
                    # Use the appropriate weight transformation
                    if hasattr(module.branches_to_output, 'weight_transform'):
                        from dendritic_modeling.networks.utils.weight_transforms import apply_weight_transform
                        branch_w = apply_weight_transform(
                            module.branches_to_output.log_weight, 
                            module.branches_to_output.weight_transform
                        ).flatten()
                    else:
                        # Fallback to exp for backward compatibility
                        branch_w = module.branches_to_output.log_weight.exp().flatten()
                    branch_w = branch_w.cpu()
                    branch_weights.append(branch_w)

        global_stats = {}

        # Compute statistics for excitatory weights
        if len(exc_weights) > 0:
            exc_weights = torch.cat(exc_weights, dim=0)
            global_stats["excitatory_weights"] = self._compute_weight_statistics(exc_weights)

        # Compute statistics for inhibitory weights
        if len(inh_weights) > 0:
            inh_weights = torch.cat(inh_weights, dim=0)
            global_stats["inhibitory_weights"] = self._compute_weight_statistics(inh_weights)

        # Compute statistics for branch weights
        if len(branch_weights) > 0:
            branch_weights = torch.cat(branch_weights, dim=0)
            global_stats["branch_weights"] = self._compute_weight_statistics(branch_weights)

        return global_stats

    def _compute_weight_statistics(self, weights: torch.Tensor) -> dict[str, float]:
        """Compute comprehensive statistics for a weight tensor."""
        stats = {}
        
        if self.compute_mean:
            stats["mean"] = weights.mean().item()
            
        if self.compute_variance:
            stats["variance"] = weights.var().item()
            stats["std"] = weights.std().item()
            
        if self.compute_min_max:
            stats["min"] = weights.min().item()
            stats["max"] = weights.max().item()
            
        if self.compute_percentiles:
            stats["median"] = weights.median().item()
            stats["q1"] = weights.quantile(0.25).item()
            stats["q3"] = weights.quantile(0.75).item()
            
        return stats