"""
Unified Weight Analysis Module.

This module provides comprehensive analysis of synaptic weights across the dendritic network,
including both global statistics and hierarchical per-layer analysis with plotting capabilities.
Provides comprehensive weight analysis with both global and hierarchical statistics.
"""

import logging
import time
from collections.abc import Iterator
from typing import Any, Optional

import numpy as np
import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.einet_core import has_einet_core
from dendritic_modeling.analysis.utils.recurrent_introspection import (
    iter_recurrent_populations,
)
from dendritic_modeling.config import WeightAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.hooks import iter_named_modules_of_type

logger = logging.getLogger(__name__)


def _weight_tensor_statistics(
    weights: torch.Tensor,
    *,
    compute_mean: bool,
    compute_variance: bool,
    compute_min_max: bool,
    compute_percentiles: bool,
) -> dict[str, float]:
    """Compute configured weight statistics with the historical torch reductions."""
    stats = {}

    if compute_mean:
        stats["mean"] = weights.mean().item()

    if compute_variance:
        stats["variance"] = weights.var().item()
        stats["std"] = weights.std().item()

    if compute_min_max:
        stats["min"] = weights.min().item()
        stats["max"] = weights.max().item()

    if compute_percentiles:
        stats["median"] = weights.median().item()
        stats["q1"] = weights.quantile(0.25).item()
        stats["q3"] = weights.quantile(0.75).item()

    return stats


def _branch_synapse_statistics(
    prefix: str,
    active_weights: torch.Tensor,
    *,
    compute_mean: bool,
    compute_variance: bool,
    compute_min_max: bool,
    compute_percentiles: bool,
) -> dict[str, float | int]:
    """Compute active-synapse branch statistics for one synapse type."""
    stats: dict[str, float | int] = {f"{prefix}_n_active_synapses": len(active_weights)}
    if len(active_weights) == 0:
        return stats

    if compute_mean:
        stats[f"{prefix}_weight_mean"] = active_weights.mean().item()
    if compute_variance:
        stats[f"{prefix}_weight_var"] = active_weights.var().item()
    if compute_min_max:
        stats[f"{prefix}_weight_min"] = active_weights.min().item()
        stats[f"{prefix}_weight_max"] = active_weights.max().item()
    if compute_percentiles:
        stats[f"{prefix}_weight_median"] = active_weights.median().item()

    return stats


def _iter_dendritic_populations(
    model: BaseModel,
    *,
    include_inhibitory: bool = True,
):
    """Yield unique population modules that expose dendritic branch layers."""
    core = getattr(model, "core_network", model)
    seen: set[int] = set()

    for record in iter_recurrent_populations(core):
        if not include_inhibitory and record.polarity == "inhibitory":
            continue
        population = record.population
        if not hasattr(population, "branch_layers"):
            continue
        ident = id(population)
        if ident in seen:
            continue
        seen.add(ident)
        yield population

    for layer in getattr(core, "layers", []) or []:
        for attr, polarity in (
            ("excitatory_cells", "excitatory"),
            ("inhibitory_cells", "inhibitory"),
            ("e_population", "excitatory"),
            ("i_population", "inhibitory"),
        ):
            if not include_inhibitory and polarity == "inhibitory":
                continue
            population = getattr(layer, attr, None)
            if not hasattr(population, "branch_layers"):
                continue
            ident = id(population)
            if ident in seen:
                continue
            seen.add(ident)
            yield population


class WeightAnalyzer(AbstractAnalyzer):
    """
    Unified analyzer for comprehensive synaptic weight analysis.

    Provides both global weight statistics and hierarchical per-layer analysis
    with enhanced plotting capabilities including min/max and dendritic strength plots.
    """

    def __init__(self, params: WeightAnalysisParams):
        """Initialize the unified weight analyzer."""
        super().__init__("WeightAnalyzer")

        self.synapse_weights = getattr(params, "synapse_weights", True)
        self.branch_weights = getattr(params, "branch_weights", True)
        self.computation_level = getattr(params, "computation_level", "single_branch")
        self.branch_aggregation = getattr(params, "branch_aggregation", "mean")

        # Enhanced analysis modes
        self.per_layer_analysis = getattr(params, "per_layer_analysis", True)
        self.per_einet_analysis = getattr(params, "per_einet_analysis", True)
        self.per_neuron_analysis = getattr(params, "per_neuron_analysis", False)

        self.analyze_excitatory = getattr(params, "analyze_excitatory", True)
        self.analyze_inhibitory = getattr(params, "analyze_inhibitory", True)
        self.analyze_branch_output = getattr(params, "analyze_branch_output", True)

        self.compute_mean = getattr(params, "compute_mean", True)
        self.compute_variance = getattr(params, "compute_variance", True)
        self.compute_percentiles = getattr(params, "compute_percentiles", True)
        self.compute_min_max = getattr(params, "compute_min_max", True)
        self.compute_dendritic_strength = getattr(
            params, "compute_dendritic_strength", True
        )

        self.weight_threshold = getattr(params, "weight_threshold", 1e-6)

        self.logger.info(
            f"Initialized unified WeightAnalyzer with computation_level: {self.computation_level}"
        )

    def analyze(
        self,
        model: BaseModel,
        save_path: Optional[str] = None,
        filename: str = "final",
        training: bool = False,
    ):
        """
        Perform comprehensive weight analysis including global and hierarchical statistics.

        Args:
            model: The model to analyze (must have ExcitationInhibitionNetwork)
            save_path: Path to save results (optional)
            filename: Filename for saved results
            training: If True, only save data (no plots). If False, generate plots.

        Returns:
            Dictionary containing all weight analysis results
        """
        if not has_einet_core(model):
            return None

        self.logger.info("Starting comprehensive weight analysis...")
        self.logger.info("Weight analysis configuration:")
        self.logger.info(f"  per_layer_analysis: {self.per_layer_analysis}")
        self.logger.info(f"  compute_min_max: {self.compute_min_max}")
        self.logger.info(
            f"  compute_dendritic_strength: {self.compute_dendritic_strength}"
        )
        self.logger.info(f"  save_path: {save_path}")
        start_time = time.time()

        results = {}

        if self.synapse_weights or self.branch_weights:
            global_stats = self._compute_global_statistics(model)
            results["global_statistics"] = global_stats

        if self.per_layer_analysis:
            layer_stats = self._compute_layer_statistics(model)

            # Inject synthetic soma ONLY if somatic_synapses=False and there is no layer_idx=0
            somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
            if not somatic_synapses and layer_stats:
                # Check if synthetic soma is needed
                if "branch_layers.0" not in layer_stats:
                    layer_stats["branch_layers.0"] = {}  # Synthetic soma
                    self.logger.info(
                        "Added synthetic soma layer (branch_layers.0) for somatic_synapses=False"
                    )
                # Dynamically determine expected layers from network structure
                # Get the actual number of dendritic layers from the first EI layer
                expected_max_idx = 0
                for population in _iter_dendritic_populations(model):
                    expected_max_idx = max(
                        expected_max_idx, len(population.branch_layers)
                    )

                # Remove any layers beyond the expected range
                for key in list(layer_stats.keys()):
                    if "branch_layers." in key:
                        idx = int(key.split(".")[-1])
                        if idx > expected_max_idx:
                            del layer_stats[key]  # Remove extra
                            self.logger.info(
                                f"Removed extra layer {key} (idx={idx} > expected_max_idx={expected_max_idx})"
                            )

            results["layer_statistics"] = layer_stats
            self.logger.info(
                f"Generated layer_statistics with {len(layer_stats)} layers: {list(layer_stats.keys())}"
            )

            # Also store raw per-branch statistics for compatibility with the original analyzer
            branch_level_stats = self._compute_branch_level_statistics(model)
            if branch_level_stats:
                results["branch_level_statistics"] = branch_level_stats

        if self.compute_min_max or self.compute_dendritic_strength:
            enhanced_stats = self._compute_enhanced_layer_statistics(model)

            # Inject synthetic Soma for enhanced statistics as well
            somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
            if not somatic_synapses and enhanced_stats:
                if "branch_layers.0" not in enhanced_stats:
                    enhanced_stats = {"branch_layers.0": {}} | enhanced_stats

            results["enhanced_layer_statistics"] = enhanced_stats

        elapsed_time = time.time() - start_time
        self.logger.info(f"Weight analysis completed in {elapsed_time:.2f} seconds")

        # Save results and generate plots
        if save_path is not None:
            self.logger.info(
                f"Saving weight analysis results to {save_path}/{filename}"
            )
            save_dict(results, save_path, filename)

            # Only generate plots during final analysis (not during training)
            if not training:
                self.logger.info("Starting plot generation...")
                self._generate_all_plots(results, save_path, model)
                self.logger.info("Plot generation completed!")
            else:
                self.logger.info("Training mode: plots skipped, only data saved")

        return results

    def _compute_global_statistics(self, model: BaseModel) -> dict[str, Any]:
        """Compute global weight statistics across the entire network."""
        self.logger.info("Computing global weight statistics...")

        # Collect weights from all dendritic branch layers
        exc_weights = []
        inh_weights = []
        branch_weights = []

        for _name, module in iter_named_modules_of_type(model, DendriticBranchLayer):
            # Collect excitatory synapse weights
            if (
                self.synapse_weights
                and self.analyze_excitatory
                and module.branch_excitation is not None
            ):
                exc_w = self._active_weights(
                    module.branch_excitation.pruned_weight().flatten()
                ).cpu()
                exc_weights.append(exc_w)

            # Collect inhibitory synapse weights
            if (
                self.synapse_weights
                and self.analyze_inhibitory
                and module.branch_inhibition is not None
            ):
                inh_w = self._active_weights(
                    module.branch_inhibition.pruned_weight().flatten()
                ).cpu()
                inh_weights.append(inh_w)

            # Collect branch-to-output weights
            if (
                self.branch_weights
                and self.analyze_branch_output
                and hasattr(module, "branches_to_output")
            ):
                branch_w = module.branches_to_output.weight().cpu().flatten()
                branch_weights.append(branch_w)

        global_stats = {}

        # Compute statistics for excitatory weights
        if len(exc_weights) > 0:
            exc_weights = torch.cat(exc_weights, dim=0)
            global_stats["excitatory_weights"] = self._compute_weight_statistics(
                exc_weights
            )

        # Compute statistics for inhibitory weights
        if len(inh_weights) > 0:
            inh_weights = torch.cat(inh_weights, dim=0)
            global_stats["inhibitory_weights"] = self._compute_weight_statistics(
                inh_weights
            )

        # Compute statistics for branch weights
        if len(branch_weights) > 0:
            branch_weights = torch.cat(branch_weights, dim=0)
            global_stats["branch_weights"] = self._compute_weight_statistics(
                branch_weights
            )

        return global_stats

    def _compute_weight_statistics(self, weights: torch.Tensor) -> dict[str, float]:
        """Compute comprehensive statistics for a weight tensor."""
        return _weight_tensor_statistics(
            weights,
            compute_mean=self.compute_mean,
            compute_variance=self.compute_variance,
            compute_min_max=self.compute_min_max,
            compute_percentiles=self.compute_percentiles,
        )

    def _active_weights(self, weights: torch.Tensor) -> torch.Tensor:
        """Return weights above the configured active-synapse threshold."""
        return weights[weights > self.weight_threshold]

    @staticmethod
    def _branch_layer_depth(
        branch_layer: DendriticBranchLayer, fallback_depth: int
    ) -> int:
        return getattr(branch_layer, "layer_idx", fallback_depth)

    def _iter_branch_layers(
        self,
        model: BaseModel,
        *,
        include_inhibitory: bool,
        require_excitatory_dendrinet: bool = False,
        require_layer_idx: bool = False,
    ) -> Iterator[tuple[int, DendriticBranchLayer]]:
        del require_excitatory_dendrinet
        for population in _iter_dendritic_populations(
            model, include_inhibitory=include_inhibitory
        ):
            yield from self._iter_dendrinet_branch_layers(
                population, require_layer_idx=require_layer_idx
            )

    def _iter_dendrinet_branch_layers(
        self, dendrinet, *, require_layer_idx: bool = False
    ) -> Iterator[tuple[int, DendriticBranchLayer]]:
        for depth_idx, branch_layer in enumerate(dendrinet.branch_layers):
            if require_layer_idx and getattr(branch_layer, "layer_idx", None) is None:
                continue
            yield self._branch_layer_depth(branch_layer, depth_idx), branch_layer

    def _compute_layer_statistics(self, model: BaseModel) -> dict[str, Any]:
        """Compute hierarchical per-branch-depth weight statistics across the network.

        This aggregates statistics for each dendritic branch depth index across all
        EI layers, matching the structure expected by the original synaptic weight
        analyzer and plotting utilities (keys like "branch_layers.{idx}").
        """
        self.logger.info("Computing hierarchical layer statistics...")

        # Accumulate branch statistics per dendritic depth across EI layers
        depth_to_branch_list: dict[int, list[dict[str, Any]]] = {}

        for layer_idx_attr, branch_layer in self._iter_branch_layers(
            model, include_inhibitory=True
        ):
            stats_dict = self._compute_branch_weight_statistics(
                branch_layer, module_name=f"branch_layers.{layer_idx_attr}"
            )
            if stats_dict and "branches" in stats_dict:
                if layer_idx_attr not in depth_to_branch_list:
                    depth_to_branch_list[layer_idx_attr] = []
                depth_to_branch_list[layer_idx_attr].extend(stats_dict["branches"])

        # Now aggregate per-depth branch statistics to layer-level metrics
        layer_results: dict[str, Any] = {}
        for depth_idx, branches in depth_to_branch_list.items():
            layer_name = f"branch_layers.{depth_idx}"
            aggregated_stats = self._aggregate_layer_statistics({"branches": branches})
            if aggregated_stats:
                layer_results[layer_name] = aggregated_stats
                self.logger.info(f"Layer {layer_name}: {list(aggregated_stats.keys())}")

        return layer_results

    def _compute_branch_weight_statistics(
        self, module: DendriticBranchLayer, module_name: str
    ) -> dict[str, Any]:
        """Compute weight statistics for a single dendritic branch layer."""
        if not isinstance(module, DendriticBranchLayer):
            return {}

        branch_results = []

        # Determine the maximum number of branches
        max_branches = 0
        if self.analyze_excitatory and module.branch_excitation is not None:
            max_branches = max(
                max_branches, module.branch_excitation.pruned_weight().shape[0]
            )
        if self.analyze_inhibitory and module.branch_inhibition is not None:
            max_branches = max(
                max_branches, module.branch_inhibition.pruned_weight().shape[0]
            )

        # Initialize branch results for all branches
        for branch_idx in range(max_branches):
            branch_stats = {"branch_idx": branch_idx, "layer": module_name}
            branch_results.append(branch_stats)

        # Analyze excitatory weights per branch
        if self.analyze_excitatory and module.branch_excitation is not None:
            exc_weights = (
                module.branch_excitation.pruned_weight()
            )  # [n_branches, in_features]

            for branch_idx in range(exc_weights.shape[0]):
                if branch_idx < len(branch_results):
                    self._record_branch_synapse_statistics(
                        branch_results[branch_idx],
                        prefix="exc",
                        weights=exc_weights[branch_idx, :],
                    )

        # Analyze inhibitory weights per branch
        if self.analyze_inhibitory and module.branch_inhibition is not None:
            inh_weights = module.branch_inhibition.pruned_weight()

            for branch_idx in range(inh_weights.shape[0]):
                if branch_idx < len(branch_results):
                    self._record_branch_synapse_statistics(
                        branch_results[branch_idx],
                        prefix="inh",
                        weights=inh_weights[branch_idx, :],
                    )

        return {"branches": branch_results}

    def _record_branch_synapse_statistics(
        self, branch_stats: dict[str, Any], prefix: str, weights: torch.Tensor
    ) -> None:
        """Add active-synapse weight statistics for one branch and synapse type."""
        active_weights = weights[weights > self.weight_threshold]
        branch_stats.update(
            _branch_synapse_statistics(
                prefix,
                active_weights,
                compute_mean=self.compute_mean,
                compute_variance=self.compute_variance,
                compute_min_max=self.compute_min_max,
                compute_percentiles=self.compute_percentiles,
            )
        )

    def _compute_branch_level_statistics(self, model: BaseModel) -> dict[str, Any]:
        """Compute and return raw per-branch statistics grouped by depth.

        This mirrors the original analyzer's behavior of keeping branch-level data
        before aggregation, to enable downstream tools to recompute or plot with
        different aggregations if desired.
        """
        depth_to_branch_list: dict[int, list[dict[str, Any]]] = {}

        for layer_idx_attr, branch_layer in self._iter_branch_layers(
            model, include_inhibitory=True
        ):
            stats_dict = self._compute_branch_weight_statistics(
                branch_layer, module_name=f"branch_layers.{layer_idx_attr}"
            )
            if not stats_dict or "branches" not in stats_dict:
                continue
            if layer_idx_attr not in depth_to_branch_list:
                depth_to_branch_list[layer_idx_attr] = []
            depth_to_branch_list[layer_idx_attr].extend(stats_dict["branches"])

        # Format similar to layer_statistics but with raw branches list
        formatted: dict[str, Any] = {}
        for depth_idx, branches in depth_to_branch_list.items():
            formatted[f"branch_layers.{depth_idx}"] = {"branches": branches}
        return formatted

    def _aggregate_layer_statistics(
        self, layer_data: dict[str, Any]
    ) -> dict[str, float]:
        """Aggregate branch statistics to layer level."""
        branch_results = layer_data.get("branches", [])
        if not branch_results:
            return {}

        layer_stats = {}

        # Collect all branch statistics
        metrics = [
            "exc_weight_mean",
            "exc_weight_var",
            "exc_weight_min",
            "exc_weight_max",
            "exc_n_active_synapses",
            "inh_weight_mean",
            "inh_weight_var",
            "inh_weight_min",
            "inh_weight_max",
            "inh_n_active_synapses",
        ]

        for metric in metrics:
            values = [
                branch.get(metric) for branch in branch_results if metric in branch
            ]
            if values:
                values = [v for v in values if v is not None]
                if values:
                    layer_stats[f"{metric}_mean"] = np.mean(values)
                    layer_stats[f"{metric}_std"] = np.std(values)
                    if self.compute_min_max:
                        layer_stats[f"{metric}_layer_min"] = np.min(values)
                        layer_stats[f"{metric}_layer_max"] = np.max(values)

        return layer_stats

    def _compute_enhanced_layer_statistics(self, model: BaseModel) -> dict[str, Any]:
        """Compute enhanced per-layer statistics including dendritic strength."""
        self.logger.info("Computing enhanced layer statistics...")

        enhanced_stats: dict[str, Any] = {}

        # Aggregate enhanced statistics per dendritic depth across EI layers
        # Build a dict keyed by actual branch layer_idx to accumulate values
        depth_to_strength: dict[int, float] = {}
        depth_to_weights: dict[int, list[torch.Tensor]] = {}

        for layer_idx_attr, branch_layer in self._iter_branch_layers(
            model,
            include_inhibitory=True,
            require_excitatory_dendrinet=True,
            require_layer_idx=True,
        ):
            self._accumulate_enhanced_branch_statistics(
                branch_layer,
                layer_idx_attr,
                depth_to_strength,
                depth_to_weights,
            )

        # Build enhanced_stats from accumulators
        for layer_idx_attr in sorted(
            set(depth_to_strength.keys()) | set(depth_to_weights.keys())
        ):
            layer_name = f"branch_layers.{layer_idx_attr}"
            enhanced_stats[layer_name] = {}
            if self.compute_dendritic_strength and layer_idx_attr in depth_to_strength:
                enhanced_stats[layer_name]["dendritic_strength"] = depth_to_strength[
                    layer_idx_attr
                ]
            if self.compute_min_max and depth_to_weights.get(layer_idx_attr):
                all_weights = torch.cat(depth_to_weights[layer_idx_attr])
                if all_weights.numel() > 0:
                    enhanced_stats[layer_name][
                        "layer_weight_min"
                    ] = all_weights.min().item()
                    enhanced_stats[layer_name][
                        "layer_weight_max"
                    ] = all_weights.max().item()

        return enhanced_stats

    def _accumulate_enhanced_branch_statistics(
        self,
        branch_layer: DendriticBranchLayer,
        layer_idx_attr: int,
        depth_to_strength: dict[int, float],
        depth_to_weights: dict[int, list[torch.Tensor]],
    ) -> None:
        if layer_idx_attr not in depth_to_strength:
            depth_to_strength[layer_idx_attr] = 0.0
        if layer_idx_attr not in depth_to_weights:
            depth_to_weights[layer_idx_attr] = []

        if self.compute_dendritic_strength:
            if self.analyze_excitatory and branch_layer.branch_excitation is not None:
                active_exc = self._active_weights(
                    branch_layer.branch_excitation.pruned_weight()
                )
                depth_to_strength[layer_idx_attr] += active_exc.sum().item()

            if self.analyze_inhibitory and branch_layer.branch_inhibition is not None:
                active_inh = self._active_weights(
                    branch_layer.branch_inhibition.pruned_weight()
                )
                depth_to_strength[layer_idx_attr] += active_inh.sum().item()

        if self.compute_min_max:
            if self.analyze_excitatory and branch_layer.branch_excitation is not None:
                active_exc = self._active_weights(
                    branch_layer.branch_excitation.pruned_weight().flatten()
                )
                if active_exc.numel() > 0:
                    depth_to_weights[layer_idx_attr].append(active_exc)
            if self.analyze_inhibitory and branch_layer.branch_inhibition is not None:
                active_inh = self._active_weights(
                    branch_layer.branch_inhibition.pruned_weight().flatten()
                )
                if active_inh.numel() > 0:
                    depth_to_weights[layer_idx_attr].append(active_inh)

    def _generate_all_plots(
        self, results: dict[str, Any], save_path: str, model: BaseModel
    ):
        """Generate all weight analysis plots."""
        self.logger.info("Generating weight analysis plots...")
        self.logger.info(f"Results keys: {list(results.keys())}")
        self.logger.info(f"Save path: {save_path}")

        try:
            # Get somatic synapses setting
            somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
            self.logger.info(f"Somatic synapses: {somatic_synapses}")

            # Generate hierarchical plots (original synaptic_weight_expectation plots)
            if "layer_statistics" in results:
                self.logger.info("Generating hierarchical weight plots...")
                self._plot_hierarchical_weights(
                    results["layer_statistics"], save_path, somatic_synapses
                )
                self.logger.info("Hierarchical plots completed")
            else:
                self.logger.warning("No layer_statistics found in results")

            # Generate enhanced plots (min/max and dendritic strength)
            if "enhanced_layer_statistics" in results:
                self.logger.info("Generating enhanced statistics plots...")
                self._plot_enhanced_statistics(
                    results["enhanced_layer_statistics"], save_path, somatic_synapses
                )
                self.logger.info("Enhanced plots completed")
            else:
                self.logger.info(
                    "enhanced_layer_statistics not present; skipping optional enhanced plots"
                )

            # Generate global distribution plots
            if "global_statistics" in results:
                self.logger.info("Generating global distribution plots...")
                self._plot_global_distributions(results["global_statistics"], save_path)
                self.logger.info("Global plots completed")
            else:
                self.logger.warning("No global_statistics found in results")

        except Exception as e:
            self.logger.error(f"Error generating weight plots: {e}")
            import traceback

            traceback.print_exc()

    def _plot_hierarchical_weights(
        self, layer_stats: dict[str, Any], save_path: str, somatic_synapses: bool
    ):
        """Generate all original synaptic weight plots plus the simplified ones."""
        try:
            from dendritic_modeling.plotting.visualizations.synaptic_weight_plots import (
                plot_synaptic_weight_analysis,
            )

            # Generate all the original plots that synaptic_weight_expectation analyzer created
            plot_synaptic_weight_analysis(
                layer_stats,
                save_path=save_path,
                somatic_synapses=somatic_synapses,
            )

        except ImportError as e:
            self.logger.warning(f"Could not import weight plotting functions: {e}")

    def _plot_enhanced_statistics(
        self, enhanced_stats: dict[str, Any], save_path: str, somatic_synapses: bool
    ):
        """Generate enhanced plots for min/max and dendritic strength."""
        try:
            import os

            import matplotlib.pyplot as plt

            from dendritic_modeling.plotting.visualizations.plotting_utils import (
                convert_layer_names,
                get_color_scheme,
                setup_basic_plot,
            )

            # Convert layer names for proper ordering
            converted_results = convert_layer_names(
                enhanced_stats, somatic_synapses=somatic_synapses
            )
            layer_names = list(converted_results.keys())
            colors = get_color_scheme()

            # Create 2-panel plot for enhanced statistics
            _, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Collect data
            depths = []
            min_weights = []
            max_weights = []
            dendritic_strengths = []

            for layer_name in layer_names:
                layer_data = converted_results[layer_name]
                if (
                    "layer_weight_min" in layer_data
                    and "layer_weight_max" in layer_data
                ):
                    depths.append(layer_name)
                    min_weights.append(layer_data["layer_weight_min"])
                    max_weights.append(layer_data["layer_weight_max"])
                    dendritic_strengths.append(
                        layer_data.get("dendritic_strength", 0.0)
                    )

            if depths:
                x_pos = np.arange(len(depths))
                width = 0.35

                # Panel 1: Min/Max weights per layer
                ax1 = axes[0]
                ax1.bar(
                    x_pos - width / 2,
                    min_weights,
                    width,
                    label="Min Weight",
                    color=colors["excitatory"],
                    alpha=0.7,
                )
                ax1.bar(
                    x_pos + width / 2,
                    max_weights,
                    width,
                    label="Max Weight",
                    color=colors["inhibitory"],
                    alpha=0.7,
                )

                setup_basic_plot(
                    ax1,
                    "Min/Max Weights per Layer",
                    "Layer (Soma to Distal)",
                    "Weight Value",
                    grid=True,
                )
                ax1.set_xticks(x_pos)
                ax1.set_xticklabels(depths, rotation=45)
                ax1.legend()

                # Panel 2: Dendritic strength per layer
                ax2 = axes[1]
                ax2.bar(
                    x_pos,
                    dendritic_strengths,
                    width * 2,
                    color=colors["combined"],
                    alpha=0.7,
                )

                setup_basic_plot(
                    ax2,
                    "Total Dendritic Strength per Layer",
                    "Layer (Soma to Distal)",
                    "Total Weight Sum",
                    grid=True,
                )
                ax2.set_xticks(x_pos)
                ax2.set_xticklabels(depths, rotation=45)

            plt.suptitle(
                "Enhanced Weight Analysis: Min/Max and Dendritic Strength", fontsize=16
            )
            plt.tight_layout()

            # Save plot
            filename = os.path.join(save_path, "enhanced_weight_analysis.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.info(f"Saved enhanced weight analysis plot: {filename}")

        except Exception as e:
            self.logger.error(f"Error generating enhanced plots: {e}")

    def _plot_global_distributions(self, global_stats: dict[str, Any], save_path: str):
        """Generate global weight distribution plots."""
        try:
            import os

            import matplotlib.pyplot as plt

            # Create histogram plots for global distributions
            _, axes = plt.subplots(1, 3, figsize=(15, 5))

            weight_types = [
                "excitatory_weights",
                "inhibitory_weights",
                "branch_weights",
            ]
            titles = [
                "Excitatory Weight Distribution",
                "Inhibitory Weight Distribution",
                "Branch Weight Distribution",
            ]

            for i, (weight_type, title) in enumerate(zip(weight_types, titles)):
                if weight_type in global_stats:
                    stats = global_stats[weight_type]
                    ax = axes[i]

                    # Create a simple bar plot of statistics
                    metrics = ["mean", "std", "min", "max", "median"]
                    values = [stats.get(metric, 0) for metric in metrics]

                    ax.bar(metrics, values, alpha=0.7)
                    ax.set_title(title)
                    ax.set_ylabel("Weight Value")
                    ax.tick_params(axis="x", rotation=45)
                else:
                    axes[i].text(
                        0.5,
                        0.5,
                        f"No {weight_type.replace('_', ' ')}",
                        ha="center",
                        va="center",
                        transform=axes[i].transAxes,
                    )
                    axes[i].set_title(titles[i])

            plt.suptitle("Global Weight Statistics", fontsize=16)
            plt.tight_layout()

            # Save plot
            filename = os.path.join(save_path, "global_weight_distributions.png")
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.info(f"Saved global weight distributions plot: {filename}")

        except Exception as e:
            self.logger.error(f"Error generating global distribution plots: {e}")


__all__ = ["WeightAnalyzer"]
