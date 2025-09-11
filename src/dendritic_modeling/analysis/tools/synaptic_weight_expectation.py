"""
Synaptic Weight Expectation Analysis Module.

This module analyzes the expected (TopK-selected) synaptic weights per branch
within each dendritic layer, computing mean and variance statistics that can
be plotted similar to information analysis with soma→distal ordering.
"""

import logging
import time
from typing import Any, Optional

import numpy as np
from tqdm import tqdm

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config import SynapticWeightExpectationParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


class SynapticWeightExpectationAnalyzer(AbstractAnalyzer):
    """
    Analyzer for computing expected synaptic weight statistics per branch.

    This analyzer computes mean and variance of TopK-selected excitatory and
    inhibitory synaptic weights for each branch within each dendritic layer,
    following the same hierarchical structure as information analysis.
    """

    def __init__(self, params: SynapticWeightExpectationParams):
        """Initialize the analyzer with configuration parameters."""
        super().__init__("SynapticWeightExpectationAnalyzer")

        # Analysis granularity control (similar to information analysis)
        self.computation_level = getattr(params, "computation_level", "single_branch")
        self.branch_aggregation = getattr(params, "branch_aggregation", "mean")

        # Enhanced analysis modes (similar to information analysis)
        self.per_layer_analysis = getattr(params, "per_layer_analysis", False)
        self.per_einet_analysis = getattr(params, "per_einet_analysis", False)
        self.per_neuron_analysis = getattr(params, "per_neuron_analysis", False)

        # Weight types to analyze
        self.analyze_excitatory = getattr(params, "analyze_excitatory", True)
        self.analyze_inhibitory = getattr(params, "analyze_inhibitory", True)
        self.analyze_branch_output = getattr(params, "analyze_branch_output", False)

        # Statistical measures to compute
        self.compute_mean = getattr(params, "compute_mean", True)
        self.compute_variance = getattr(params, "compute_variance", True)
        self.compute_percentiles = getattr(params, "compute_percentiles", False)

        # Weight filtering
        self.weight_threshold = getattr(params, "weight_threshold", 1e-6)

        self.logger.info(
            f"Initialized SynapticWeightExpectationAnalyzer with computation_level: {self.computation_level}"
        )

    def compute_branch_weight_statistics(
        self, module: DendriticBranchLayer, module_name: str
    ) -> dict[str, Any]:
        """
        Compute weight statistics for a single branch layer.

        Args:
            module: DendriticBranchLayer to analyze
            module_name: Name of the module for identification

        Returns:
            Dictionary containing weight statistics per branch
        """
        results = {
            "layer_name": module_name,
            "layer_idx": getattr(module, "layer_idx", -1),
            "branches": {},
        }

        # Get number of branches from the TopK layer structure
        # The branch_excitation/inhibition layers have shape [n_branches, in_features]
        n_branches = 0

        # Check if this is a soma layer without synapses (when somatic_synapses=False)
        is_soma_without_synapses = (
            getattr(module, "layer_idx", -1) == 0
            and (
                module.branch_excitation is None
                or not hasattr(module.branch_excitation, "pre_w")
            )
            and (
                module.branch_inhibition is None
                or not hasattr(module.branch_inhibition, "pre_w")
            )
        )

        if is_soma_without_synapses:
            # For soma without synapses, we still want to show it in plots but with zero values
            # We'll create a single "branch" with zero statistics
            n_branches = 1
            self.logger.debug(
                f"Layer {module_name}: soma without synapses, creating synthetic branch"
            )
        elif module.branch_excitation is not None and hasattr(
            module.branch_excitation, "pre_w"
        ):
            exc_weights = module.branch_excitation.pruned_weight()
            n_branches = exc_weights.shape[0]
            self.logger.debug(
                f"Layer {module_name}: found {n_branches} branches from excitatory weights"
            )
        elif module.branch_inhibition is not None and hasattr(
            module.branch_inhibition, "pre_w"
        ):
            inh_weights = module.branch_inhibition.pruned_weight()
            n_branches = inh_weights.shape[0]
            self.logger.debug(
                f"Layer {module_name}: found {n_branches} branches from inhibitory weights"
            )
        else:
            self.logger.warning(
                f"Layer {module_name}: no valid excitatory or inhibitory connections found"
            )

        self.logger.debug(f"Layer {module_name}: found {n_branches} branches")

        for branch_idx in range(n_branches):
            branch_stats = {}
            self.logger.debug(
                f"Processing branch {branch_idx}/{n_branches} in layer {module_name}"
            )

            if is_soma_without_synapses:
                # For soma without synapses, set all weight statistics to zero
                if self.analyze_excitatory:
                    branch_stats["exc_weight_mean"] = 0.0
                    branch_stats["exc_weight_var"] = 0.0
                    branch_stats["exc_weight_std"] = 0.0
                    branch_stats["exc_n_active_synapses"] = 0
                if self.analyze_inhibitory:
                    branch_stats["inh_weight_mean"] = 0.0
                    branch_stats["inh_weight_var"] = 0.0
                    branch_stats["inh_weight_std"] = 0.0
                    branch_stats["inh_n_active_synapses"] = 0
                if self.analyze_branch_output:
                    branch_stats["branch_output_weight"] = 0.0
            else:
                # Normal analysis for layers with actual synapses
                # Analyze excitatory weights for this branch
                if self.analyze_excitatory and module.branch_excitation is not None:
                    exc_weights = (
                        module.branch_excitation.pruned_weight()
                    )  # Shape: [n_branches, in_features]
                    if branch_idx < exc_weights.shape[0]:
                        branch_exc_weights = exc_weights[
                            branch_idx, :
                        ]  # Weights for this branch
                        # Filter out very small weights
                        active_weights = branch_exc_weights[
                            branch_exc_weights > self.weight_threshold
                        ]

                        if len(active_weights) > 0:
                            if self.compute_mean:
                                branch_stats["exc_weight_mean"] = (
                                    active_weights.mean().item()
                                )
                            if self.compute_variance:
                                branch_stats["exc_weight_var"] = (
                                    active_weights.var().item()
                                )
                                branch_stats["exc_weight_std"] = (
                                    active_weights.std().item()
                                )
                            if self.compute_percentiles:
                                branch_stats["exc_weight_median"] = (
                                    active_weights.median().item()
                                )
                                branch_stats["exc_weight_q25"] = (
                                    active_weights.quantile(0.25).item()
                                )
                                branch_stats["exc_weight_q75"] = (
                                    active_weights.quantile(0.75).item()
                                )

                            branch_stats["exc_n_active_synapses"] = len(active_weights)
                        else:
                            # No active excitatory synapses
                            branch_stats["exc_weight_mean"] = 0.0
                            branch_stats["exc_weight_var"] = 0.0
                            branch_stats["exc_weight_std"] = 0.0
                            branch_stats["exc_n_active_synapses"] = 0

                # Analyze inhibitory weights for this branch
                if self.analyze_inhibitory and module.branch_inhibition is not None:
                    inh_weights = (
                        module.branch_inhibition.pruned_weight()
                    )  # Shape: [n_branches, in_features]
                    if branch_idx < inh_weights.shape[0]:
                        branch_inh_weights = inh_weights[
                            branch_idx, :
                        ]  # Weights for this branch
                        # Filter out very small weights
                        active_weights = branch_inh_weights[
                            branch_inh_weights > self.weight_threshold
                        ]

                        if len(active_weights) > 0:
                            if self.compute_mean:
                                branch_stats["inh_weight_mean"] = (
                                    active_weights.mean().item()
                                )
                            if self.compute_variance:
                                branch_stats["inh_weight_var"] = (
                                    active_weights.var().item()
                                )
                                branch_stats["inh_weight_std"] = (
                                    active_weights.std().item()
                                )
                            if self.compute_percentiles:
                                branch_stats["inh_weight_median"] = (
                                    active_weights.median().item()
                                )
                                branch_stats["inh_weight_q25"] = (
                                    active_weights.quantile(0.25).item()
                                )
                                branch_stats["inh_weight_q75"] = (
                                    active_weights.quantile(0.75).item()
                                )

                            branch_stats["inh_n_active_synapses"] = len(active_weights)
                        else:
                            # No active inhibitory synapses
                            branch_stats["inh_weight_mean"] = 0.0
                            branch_stats["inh_weight_var"] = 0.0
                            branch_stats["inh_weight_std"] = 0.0
                            branch_stats["inh_n_active_synapses"] = 0

                # Analyze branch-to-output weights
                if self.analyze_branch_output and hasattr(module, "branches_to_output"):
                    branch_output_weights = (
                        module.branches_to_output.log_weight.exp()
                    )  # Convert from log space
                    if branch_idx < len(branch_output_weights):
                        weight_val = branch_output_weights[branch_idx].item()
                        branch_stats["branch_output_weight"] = weight_val

            results["branches"][f"branch_{branch_idx}"] = branch_stats
            self.logger.debug(f"Branch {branch_idx} stats: {branch_stats}")

        self.logger.debug(
            f"Layer {module_name} final results: {len(results['branches'])} branches"
        )
        return results

    def _compute_layer_statistics(
        self, branch_results: list[dict[str, Any]]
    ) -> dict[str, float]:
        """
        Compute layer-level statistics from branch results.
        Similar to information analysis layer statistics.
        """
        layer_stats = {}

        # Collect all metrics across branches
        all_metrics = {}
        for branch_result in branch_results:
            for _branch_name, branch_data in branch_result.get("branches", {}).items():
                for metric, value in branch_data.items():
                    if isinstance(value, (int, float)):
                        if metric not in all_metrics:
                            all_metrics[metric] = []
                        all_metrics[metric].append(value)

        # Compute statistics for each metric
        for metric, values in all_metrics.items():
            if values:
                layer_stats[f"{metric}_mean"] = np.mean(values)
                layer_stats[f"{metric}_std"] = np.std(values)
                layer_stats[f"{metric}_min"] = np.min(values)
                layer_stats[f"{metric}_max"] = np.max(values)

        return layer_stats

    def analyze(
        self,
        model: BaseModel,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """
        Perform synaptic weight expectation analysis.

        Args:
            model: The model to analyze
            device: Device to run analysis on
            save_path: Path to save results (optional)
            filename: Filename for saved results

        Returns:
            Dictionary containing weight analysis results
        """
        self.logger.info("Starting synaptic weight expectation analysis...")

        try:
            # Validate model
            if not isinstance(model.core_network, ExcitationInhibitionNetwork):
                self.logger.warning(
                    "Model does not have ExcitationInhibitionNetwork. Skipping analysis."
                )
                return

            model = model.to(device)
            model.eval()

            # Collect weight statistics from all dendritic branch layers
            all_results = []
            layer_results = {}

            # Process each dendritic branch layer
            dendritic_layers = []
            for name, module in model.named_modules():
                if isinstance(module, DendriticBranchLayer):
                    dendritic_layers.append((name, module))
                    self.logger.debug(
                        f"Found dendritic layer: {name}, layer_idx: {getattr(module, 'layer_idx', 'unknown')}"
                    )

            self.logger.info(f"Found {len(dendritic_layers)} dendritic branch layers")

            if self.computation_level == "single_branch":
                # Analyze each branch individually
                for layer_name, module in tqdm(
                    dendritic_layers, desc="Processing layers"
                ):
                    layer_start_time = time.time()

                    # Compute weight statistics for this layer
                    layer_weight_stats = self.compute_branch_weight_statistics(
                        module, layer_name
                    )
                    all_results.append(layer_weight_stats)

                    # Compute layer-level statistics for plotting
                    if layer_weight_stats["branches"]:
                        layer_stats = self._compute_layer_statistics(
                            [layer_weight_stats]
                        )
                        layer_results[layer_name] = layer_stats

                    self.logger.debug(
                        f"Layer {layer_name} completed in {time.time() - layer_start_time:.1f}s"
                    )

                # Prepare results structure
                results = {
                    "computation_level": "single_branch",
                    "layer_results": all_results,
                    "layer_statistics": layer_results,
                    "num_layers": len(all_results),
                }

            elif self.computation_level == "layer_branch":
                # Aggregate branches within each layer
                for layer_name, module in tqdm(
                    dendritic_layers, desc="Processing layers"
                ):
                    layer_weight_stats = self.compute_branch_weight_statistics(
                        module, layer_name
                    )

                    # Aggregate across branches in this layer
                    aggregated_stats = self._compute_layer_statistics(
                        [layer_weight_stats]
                    )
                    layer_results[layer_name] = aggregated_stats
                    all_results.append(
                        {"layer_name": layer_name, "aggregated_stats": aggregated_stats}
                    )

                results = {
                    "computation_level": "layer_branch",
                    "layer_results": all_results,
                    "layer_statistics": layer_results,
                    "aggregation_method": self.branch_aggregation,
                }

            else:  # all_branch
                # Aggregate everything across the entire network
                all_layer_stats = []
                for layer_name, module in dendritic_layers:
                    layer_weight_stats = self.compute_branch_weight_statistics(
                        module, layer_name
                    )
                    all_layer_stats.append(layer_weight_stats)

                # Compute network-wide statistics
                network_stats = self._compute_layer_statistics(all_layer_stats)

                results = {
                    "computation_level": "all_branch",
                    "network_statistics": network_stats,
                    "layer_results": all_layer_stats,
                    "aggregation_method": self.branch_aggregation,
                }

            # Note: No need to add synthetic soma for weight analysis
            # The soma layer already exists as the layer with layer_idx=0
            # When somatic_synapses=False, it will naturally have zero weight values
            somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
            self.logger.info(f"Somatic synapses enabled: {somatic_synapses}")

            # Save results
            if save_path is not None:
                self.logger.info(
                    f"Saving synaptic weight analysis results to {save_path}/{filename}"
                )
                save_dict(results, save_path, filename)

                # Generate plots
                self._generate_weight_plots(results, save_path, somatic_synapses)

                self.logger.info("Synaptic weight analysis completed successfully!")

            return results

        except Exception as e:
            self.logger.error(f"Error in synaptic weight expectation analysis: {e}")
            import traceback

            traceback.print_exc()

    def _generate_weight_plots(
        self, results: dict, save_path: str, somatic_synapses: bool
    ):
        """Generate plots for synaptic weight analysis results."""
        try:
            from dendritic_modeling.plotting.visualizations.synaptic_weight_plots import (
                plot_synaptic_weight_analysis,
            )

            if "layer_statistics" in results:
                plot_synaptic_weight_analysis(
                    results["layer_statistics"],
                    save_path=save_path,
                    somatic_synapses=somatic_synapses,
                )

        except ImportError as e:
            self.logger.warning(
                f"Could not import synaptic weight plotting functions: {e}"
            )
        except Exception as e:
            self.logger.error(f"Error generating synaptic weight plots: {e}")
