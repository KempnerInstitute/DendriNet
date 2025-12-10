"""
Unified Weight Analysis Module.

This module provides comprehensive analysis of synaptic weights across the dendritic network,
including both global statistics and hierarchical per-layer analysis with plotting capabilities.
Provides comprehensive weight analysis with both global and hierarchical statistics.
"""

import json
import logging
import os
import time
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config import CompartmentSNRAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
    ParametricActivation,
    TopKLinear,
)
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


class CompartmentSNRAnalyzer(AbstractAnalyzer):
    """
    Unified analyzer for comprehensive synaptic weight analysis.

    Provides both global weight statistics and hierarchical per-layer analysis
    with enhanced plotting capabilities including min/max and dendritic strength plots.
    """

    def __init__(self, params: CompartmentSNRAnalysisParams):
        """Initialize the unified compartment statistics analyzer."""
        super().__init__("CompartmentSNRAnalyzer")
        # Enhanced analysis modes
        self.global_analysis = getattr(params, "global_analysis", True)
        self.layer_analysis = getattr(params, "layer_analysis", True)
        self.branch_analysis = getattr(params, "branch_analysis", True)

        self.epsilon = torch.finfo(torch.float32).eps

    def collect_raw_data(self, model: BaseModel, x: torch.Tensor):
        """Collect raw data from the model."""
        self.raw_dict = {}
        handles = []

        exc_dendrinet = model.core_network.layers[-1].excitatory_cells

        for key, module in exc_dendrinet.named_modules():
            if isinstance(module, DendriticBranchLayer):
                self.raw_dict[key] = {}
                for branch_idx in range(module.n_branches):
                    self.raw_dict[key][branch_idx] = {"depth": module.layer_idx}

                if module.branch_excitation is not None:
                    handle = module.branch_excitation.register_forward_hook(
                        lambda module, input, output, layer_key=key: self.forward_hook(
                            module, input, output, layer_key, "exc"
                        )
                    )
                    handles.append(handle)

                if module.branch_inhibition is not None:
                    handle = module.branch_inhibition.register_forward_hook(
                        lambda module, input, output, layer_key=key: self.forward_hook(
                            module, input, output, layer_key, "inh"
                        )
                    )
                    handles.append(handle)

                if hasattr(module, "branches_to_output"):
                    handle = module.branches_to_output.register_forward_hook(
                        lambda module, input, output, layer_key=key: self.forward_hook(
                            module, input, output, layer_key, "upstream"
                        )
                    )
                    handles.append(handle)

                handle = module.reactivation.register_forward_hook(
                    lambda module, input, output, layer_key=key: self.forward_hook(
                        module, input, output, layer_key, "vinf"
                    )
                )
                handles.append(handle)

                handle = module.reactivation.register_forward_hook(
                    lambda module, input, output, layer_key=key: self.forward_hook(
                        module, input, output, layer_key, "vout"
                    )
                )
                handles.append(handle)

        with torch.no_grad():
            _ = model(x)
        self.remove_forward_hooks(handles)

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        for handle in handles:
            handle.remove()

    def forward_hook(
        self,
        module: Union[TopKLinear, ParametricActivation, torch.nn.Identity],
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor,
        key: str,
        activation_type: str,
    ):
        for branch_idx in range(output.shape[-1]):
            if activation_type == "vinf":
                branch_activation = input[0][..., branch_idx].detach().cpu()
            else:  # "exc", "inh", "vout"
                branch_activation = output[..., branch_idx].detach().cpu()

            self.raw_dict[key][branch_idx][activation_type] = branch_activation

    def _compute_statistics(
        self,
        data_tensor: torch.Tensor,
        labels: torch.Tensor,
        prefix: str,
        stats_dict: dict[str, float],
    ) -> dict[str, float]:
        """Compute comprehensive statistics for an activation tensor."""
        total_mean = data_tensor.mean().item()
        total_var = data_tensor.var().item()

        n_samples = data_tensor.shape[0]
        var_E_C = 0

        for label in torch.unique(labels):
            class_data = data_tensor[labels == label]

            class_mean = class_data.mean().item()
            class_prob = class_data.shape[0] / n_samples
            var_E_C += class_prob * (class_mean - total_mean) ** 2

        stats_dict[f"{prefix}_d'^2"] = var_E_C / (total_var + self.epsilon)

        return stats_dict

    def _compute_branch_statistics(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
    ):
        """Compute branch statistics from model and test dataset."""
        model.to(device)
        x, labels = test_dataset[:]
        x: torch.Tensor = x.to(device)
        labels: torch.Tensor = labels.cpu()

        self.collect_raw_data(model, x)

        branch_statistics = {}
        for layer_key, layer_data in self.raw_dict.items():
            branch_statistics[layer_key] = {}
            layer_data: dict[int, dict[str, torch.Tensor]]
            for branch_idx, branch_data in layer_data.items():
                branch_stats = {}

                for prefix, data_tensor in branch_data.items():
                    if prefix == "depth":
                        branch_stats[prefix] = (
                            data_tensor  # data_tensor is the depth index
                        )
                    else:
                        branch_stats = self._compute_statistics(
                            data_tensor, labels, prefix, branch_stats
                        )

                branch_statistics[layer_key][branch_idx] = branch_stats

        return branch_statistics

    def _layer_aggregation(
        self, branch_statistics: dict[str, dict[int, dict[str, float]]]
    ):
        """Compute layer statistics from branch statistics."""
        layer_statistics = {}
        for layer_key, layer_data in branch_statistics.items():
            layer_aggregated: dict[str, list[float]] = {}

            for branch_data in layer_data.values():
                for field, value in branch_data.items():
                    if field not in layer_aggregated.keys() and field != "depth":
                        layer_aggregated[field] = []
                    if field != "depth":
                        layer_aggregated[field].append(value)

            layer_stats = {"depth": branch_data["depth"]}
            for field, values in layer_aggregated.items():
                data_tensor = torch.tensor(values)
                layer_stats[f"{field}_mean"] = data_tensor.mean().item()
                layer_stats[f"{field}_std"] = data_tensor.std().item()

            layer_statistics[layer_key] = layer_stats
        return layer_statistics

    def _global_aggregation(
        self, branch_statistics: dict[str, dict[int, dict[str, float]]]
    ):
        """Compute global statistics from branch statistics."""
        global_aggregated: dict[str, list[float]] = {}
        for layer_data in branch_statistics.values():
            for branch_data in layer_data.values():
                for field, value in branch_data.items():
                    if field not in global_aggregated.keys() and field != "depth":
                        global_aggregated[field] = []
                    if field != "depth":
                        global_aggregated[field].append(value)

        global_statistics = {}
        for field, values in global_aggregated.items():
            data_tensor = torch.tensor(values)
            global_statistics[f"{field}_mean"] = data_tensor.mean().item()
            global_statistics[f"{field}_std"] = data_tensor.std().item()
        return global_statistics

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
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
        try:
            assert isinstance(model.core_network, ExcitationInhibitionNetwork)
        except AssertionError:
            self.logger.error(
                "Model must have ExcitationInhibitionNetwork for weight analysis"
            )
            return None

        self.logger.info("Starting comprehensive excitation-inhibition analysis...")
        start_time = time.time()

        branch_statistics = self._compute_branch_statistics(
            model=model, test_dataset=test_dataset, device=device
        )

        results = {}

        if self.global_analysis:
            global_statistics = self._global_aggregation(branch_statistics)
            results["global_statistics"] = global_statistics

        if self.layer_analysis:
            layer_statistics = self._layer_aggregation(branch_statistics)
            results["layer_statistics"] = layer_statistics

        if self.branch_analysis:
            results["branch_statistics"] = branch_statistics

        elapsed_time = time.time() - start_time
        self.logger.info(f"SNR analysis completed in {elapsed_time:.2f} seconds")

        # Save results and generate plots
        if save_path is not None:
            if training:
                save_path = os.path.join(save_path, "epochs")

            self.logger.info(f"Saving SNR analysis results to {save_path}/{filename}")
            save_dict(results, save_path, f"{filename}.json")

            if not training:
                epochs_data, epoch_numbers = self.load_epochs_data(save_path)
                if self.global_analysis:
                    if epochs_data:
                        plot_global_vs_training(
                            epochs_data["global_statistics"], epoch_numbers, save_path
                        )

                if self.layer_analysis:
                    if epochs_data:
                        plot_layer_vs_training(
                            epochs_data["layer_statistics"], epoch_numbers, save_path
                        )

                    plot_layer_final(results["layer_statistics"], save_path)

                if self.branch_analysis:
                    if epochs_data:
                        pass

                    plot_branch_final(results["branch_statistics"], save_path)

        return results

    def _initialize_agg_structure(self, data: dict) -> dict:
        """
        Recursively initialize an aggregate dictionary structure with empty lists at leaf nodes.

        Args:
            data: The dictionary structure to replicate

        Returns:
            Dictionary with same structure but empty lists at leaf nodes
        """
        agg_structure = {}
        for key, value in data.items():
            if isinstance(value, dict):
                agg_structure[key] = self._initialize_agg_structure(value)
            else:
                # Leaf node - initialize as empty list
                if key == "depth":
                    agg_structure[key] = data[key]
                else:
                    agg_structure[key] = []
        return agg_structure

    def _populate_agg_dict(self, agg_dict: dict, epoch_data: dict):
        """
        Recursively populate aggregate dictionary by appending leaf values from epoch_data.

        Args:
            agg_dict: The aggregate dictionary to populate (modified in place)
            epoch_data: The epoch data dictionary to extract values from
        """
        for key, value in epoch_data.items():
            if isinstance(value, dict):
                if key in agg_dict:
                    self._populate_agg_dict(agg_dict[key], value)
            else:
                # Leaf node - append value to list
                if key in agg_dict and key != "depth":
                    agg_dict[key].append(value)

    def load_epochs_data(self, save_path: str):
        training_path = os.path.join(save_path, "epochs")

        if not os.path.exists(training_path):
            return None, None

        agg_dict_initialized = False
        agg_dict = {}

        # Collect all epoch files and sort by epoch number
        epoch_files = []
        for filename in os.listdir(training_path):
            # Check if file matches the "epoch#" naming convention
            if filename.startswith("epoch") and filename.endswith(".json"):
                # Extract the number part (everything between "epoch" and ".json")
                epoch_number_str = filename[
                    5:-5
                ]  # Remove "epoch" prefix and ".json" suffix
                try:
                    epoch_number = int(epoch_number_str)
                    epoch_files.append((epoch_number, filename))
                except ValueError:
                    # Skip files that don't have numeric epoch numbers
                    continue

        # Sort by epoch number to process in order
        epoch_files.sort(key=lambda x: x[0])

        # Iterate through epoch files in sorted order
        epoch_numbers = []
        for epoch_number, filename in epoch_files:
            epoch_numbers.append(epoch_number)
            # Load the JSON file
            file_path = os.path.join(training_path, filename)
            with open(file_path) as f:
                epoch_data = json.load(f)

            if not agg_dict_initialized:
                # Initialize aggregate dictionary with same structure, empty lists at leaves
                agg_dict = self._initialize_agg_structure(epoch_data)
                agg_dict_initialized = True

            # Populate aggregate dictionary by appending values from this epoch
            self._populate_agg_dict(agg_dict, epoch_data)

        return agg_dict, epoch_numbers


def plot_global_vs_training(global_data: dict, epoch_numbers: list, save_path: str):
    """
    Plot mean +/- 1 standard deviation for all activation types across training epochs.

    Args:
        global_data: Aggregated dictionary with lists of values for each metric
        epoch_numbers: List of epoch numbers corresponding to the data
        save_path: Path to save the plot
    """
    activation_types = ["exc", "inh", "vinf", "vout"]
    colors = {
        "exc": "#e74c3c",
        "inh": "#3498db",
        "vinf": "#9b59b6",
        "vout": "#2c3e50",
    }

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

    for act_type in activation_types:
        mean_key = f"{act_type}_d'^2_mean"
        std_key = f"{act_type}_d'^2_std"

        # Extract mean and std lists from global_data
        if mean_key in global_data and std_key in global_data:
            mean_values = global_data[mean_key]
            std_values = global_data[std_key]

            # Convert to numpy arrays for easier manipulation
            mean_array = np.array(mean_values)
            std_array = np.array(std_values)

            # Plot mean line
            ax.plot(
                epoch_numbers,
                mean_array,
                label=act_type,
                color=colors[act_type],
                linewidth=2,
            )

            # Plot mean +/- std as filled region
            ax.fill_between(
                epoch_numbers,
                mean_array - std_array,
                mean_array + std_array,
                alpha=0.2,
                color=colors[act_type],
            )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(r"d'$^2$", fontsize=12)
    ax.set_title(
        "Global Statistics: Mean ± 1 Standard Deviation Across Training", fontsize=14
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, "global_vs_training.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def plot_layer_vs_training(layer_data: dict, epoch_numbers: list, save_path: str):
    """
    Plot d'^2 for excitation, inhibition, and vout across training epochs,
    with different colored lines for different layers.

    Args:
        layer_data: Aggregated dictionary with layer statistics
                   Structure: {layer_key: {metric_key: [list of values across epochs]}}
        epoch_numbers: List of epoch numbers corresponding to the data
        save_path: Path to save the plot
    """
    # Get all layer keys and sort them for consistent ordering
    layer_keys = sorted(layer_data.keys())

    # Generate colors for each layer using a colormap
    # Use a colormap that provides distinct colors
    cmap = plt.cm.get_cmap("tab10")
    n_layers = len(layer_keys)
    if n_layers == 1:
        layer_colors = {layer_keys[0]: cmap(0)}
    else:
        layer_colors = {
            layer_key: cmap(i / (n_layers - 1))
            for i, layer_key in enumerate(layer_keys)
        }

    # Create figure with 3 subplots (rows)
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 12))

    # Activation types and their corresponding row indices
    activation_types = ["exc", "inh", "vout"]

    for row_idx, act_type in enumerate(activation_types):
        ax = axes[row_idx]

        # Plot each layer
        for layer_key in layer_keys:
            mean_key = f"{act_type}_d'^2_mean"
            std_key = f"{act_type}_d'^2_std"

            if mean_key in layer_data[layer_key] and std_key in layer_data[layer_key]:
                mean_values = layer_data[layer_key][mean_key]
                std_values = layer_data[layer_key][std_key]

                # Convert to numpy arrays
                mean_array = np.array(mean_values)
                std_array = np.array(std_values)

                # Get layer color
                color = layer_colors[layer_key]

                # Plot mean line with layer label
                ax.plot(
                    epoch_numbers,
                    mean_array,
                    label=f"depth {layer_data[layer_key]['depth']}",
                    color=color,
                    linewidth=2,
                )

                # Plot mean +/- std as filled region
                ax.fill_between(
                    epoch_numbers,
                    mean_array - std_array,
                    mean_array + std_array,
                    alpha=0.2,
                    color=color,
                )

        # Set labels and title for each subplot
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(r"d'$^2$", fontsize=12)

        # Set title based on activation type
        act_labels = {"exc": "Excitation", "inh": "Inhibition", "vout": "Vout"}
        ax.set_title(
            f"Layer Statistics: {act_labels[act_type]} d'$^2$ Across Training",
            fontsize=12,
        )
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    fig.savefig(
        os.path.join(save_path, "layer_vs_training.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(fig)


def plot_layer_final(
    layer_statistics: dict, save_path: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Plot layer statistics showing d'^2 for vout, excitation, inhibition, and upstream across layers.

    This function produces a figure analogous to plot_layer_ablation but shows d'^2 values
    instead of accuracy/log-likelihood decrease.

    Args:
        layer_statistics: Dictionary containing layer statistics with structure:
                         {layer_key: {depth: int, exc_d'^2_mean: float,
                                     inh_d'^2_mean: float, vout_d'^2_mean: float,
                                     upstream_d'^2_mean: float, ...}}
        save_path: Path to save plot. If None, returns figure

    Returns:
        Figure if save_path is None, otherwise None
    """
    # Extract activation types we want to plot (in desired order: vout, exc, inh, upstream)
    # vout is always present, others are optional
    activation_types = ["vout", "exc", "inh", "upstream"]
    activation_labels = {
        "exc": "Excitation",
        "inh": "Inhibition",
        "vout": "Vout",
        "upstream": "Upstream",
    }

    # Color scheme - using a different palette
    colors = {
        "vout": "#2c3e50",  # Dark blue-gray
        "exc": "#e74c3c",  # Red
        "inh": "#3498db",  # Blue
        "upstream": "#808080",  # Medium grey
    }

    # Sort layers by depth to ensure correct ordering
    sorted_layers = sorted(
        layer_statistics.items(), key=lambda x: x[1].get("depth", float("inf"))
    )

    # First, get all depths from vout (which is always present)
    plot_depths = []
    for _layer_key, layer_data in sorted_layers:
        depth = layer_data.get("depth")
        vout_mean_key = "vout_d'^2_mean"
        if depth is not None and vout_mean_key in layer_data:
            plot_depths.append(depth)

    if not plot_depths:
        logger.warning("No vout data found in layer statistics")
        return None

    # Create a mapping from depth to layer_data for quick lookup
    depth_to_layer = {}
    for _layer_key, layer_data in sorted_layers:
        depth = layer_data.get("depth")
        if depth is not None:
            depth_to_layer[depth] = layer_data

    # Collect d'^2 values and standard deviations for each activation type
    # Aligned to plot_depths (vout reference) - use None for missing values
    results = {}
    for act_type in activation_types:
        mean_key = f"{act_type}_d'^2_mean"
        std_key = f"{act_type}_d'^2_std"

        y_values = []
        yerr_values = []
        valid_mask = []  # Track which depths have data

        for depth in plot_depths:
            if depth in depth_to_layer:
                layer_data = depth_to_layer[depth]
                if mean_key in layer_data and std_key in layer_data:
                    y_values.append(layer_data[mean_key])
                    yerr_values.append(layer_data[std_key])
                    valid_mask.append(True)
                else:
                    y_values.append(None)
                    yerr_values.append(None)
                    valid_mask.append(False)
            else:
                y_values.append(None)
                yerr_values.append(None)
                valid_mask.append(False)

        # Only add if we have at least some data for this activation type
        if any(valid_mask):
            results[act_type] = {
                "depths": plot_depths,
                "d'^2": y_values,
                "std": yerr_values,
                "valid_mask": valid_mask,
            }

    if not results:
        logger.warning("No layer statistics data found for plotting")
        return None

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Count only activation types that have data
    available_types = [act_type for act_type in activation_types if act_type in results]
    n_activation_types = len(available_types)

    if n_activation_types == 0:
        logger.warning("No activation types with data to plot")
        return None

    width = 0.2

    # Plot bars for each activation type with error bars
    for i, act_type in enumerate(available_types):
        act_data = results[act_type]
        act_depths = np.array(act_data["depths"])
        valid_mask = np.array(act_data["valid_mask"])

        # Filter to only valid data points before converting to numpy
        valid_depths = act_depths[valid_mask]
        valid_y_values = np.array(
            [v for j, v in enumerate(act_data["d'^2"]) if valid_mask[j]]
        )
        valid_yerr_values = np.array(
            [v for j, v in enumerate(act_data["std"]) if valid_mask[j]]
        )

        # Calculate x positions for bars (aligned to plot_depths)
        x = valid_depths + width * (i - (n_activation_types - 1) / 2)

        ax.bar(
            x,
            valid_y_values,
            width=width,
            label=activation_labels[act_type],
            color=colors[act_type],
            yerr=valid_yerr_values,
            capsize=5,
            error_kw={"elinewidth": 1.5, "capthick": 1.5},
        )

    # Create depth labels - if depth=0 label as "Soma", else f"Depth {depth}"
    depth_labels = []
    for depth in plot_depths:
        if depth == 0:
            depth_labels.append("Soma")
        else:
            depth_labels.append(f"Depth {depth}")

    ax.set_xticks(plot_depths)
    ax.set_xticklabels(depth_labels, rotation=45)
    ax.set_xlabel("Branch Depth (Soma → Distal)")
    ax.set_ylabel(r"d'$^2$")
    ax.set_title(r"d'$^2$ vs. Branch Depth")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        plot_path = os.path.join(save_path, "layer_final.png")
        fig.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        return fig


def plot_branch_final(branch_data: dict, save_path: str):
    """
    Plot the final branch statistics.

    Args:
        branch_data: The branch data dictionary to plot
        save_path: Path to save the plot
    """
    agg_data: dict[str, dict[str, list[float]]] = {}

    # First pass: collect all available activation types
    available_activation_types = set()

    for _layer_key, layer_data in branch_data.items():
        for branch_item in layer_data.values():
            # Check which activation types are present
            for act_type in ["exc", "inh", "vinf", "vout", "upstream"]:
                key = f"{act_type}_d'^2"
                if key in branch_item:
                    available_activation_types.add(act_type)

    # vout should always be present, but check anyway
    if "vout" not in available_activation_types:
        logger.warning("No vout data found in branch statistics")
        return

    # Second pass: aggregate data for available activation types
    for _layer_key, layer_data in branch_data.items():
        for branch_item in layer_data.values():
            depth = branch_item["depth"]
            if f"depth {depth}" not in agg_data.keys():
                agg_data[f"depth {depth}"] = {}

            # Initialize lists for available activation types
            for act_type in available_activation_types:
                if act_type not in agg_data[f"depth {depth}"].keys():
                    agg_data[f"depth {depth}"][act_type] = []

            # Append data only for activation types that exist
            for act_type in available_activation_types:
                key = f"{act_type}_d'^2"
                if key in branch_item:
                    agg_data[f"depth {depth}"][act_type].append(branch_item[key])

    # Use available activation types for plotting
    indeps = sorted(available_activation_types)
    deps = sorted(available_activation_types)

    # Build long-form DataFrame from agg_data
    rows = []
    for depth, data in agg_data.items():
        for dep in deps:  # y
            for indep in indeps:  # x
                # Only process if both activation types have data
                if (
                    indep in data
                    and dep in data
                    and len(data[indep]) > 0
                    and len(data[dep]) > 0
                ):
                    x = np.asarray(data[indep])
                    y = np.asarray(data[dep])
                    # Only add if arrays have the same length
                    if len(x) == len(y):
                        rows.append(
                            pd.DataFrame(
                                {
                                    "x": x,
                                    "y": y,
                                    "depth": depth,
                                    "row": dep,
                                    "col": indep,
                                }
                            )
                        )

    if not rows:
        logger.warning("No valid data to plot in branch statistics")
        return

    df = pd.concat(rows, ignore_index=True)

    # Labels - include all available activation types
    labels = {
        "exc": r"$d'^2_{E}$",
        "inh": r"$d'^2_{I}$",
        "vinf": r"$d'^2_{V_{\infty}}$",
        "vout": r"$d'^2_{V_{out}}$",
        "upstream": r"$d'^2_{upstream}$",
    }

    # Scatter faceted by (row=dep, col=indep) with hue=depth
    g = sns.relplot(
        data=df,
        x="x",
        y="y",
        hue="depth",
        row="row",
        col="col",
        kind="scatter",
        row_order=deps,
        col_order=indeps,
        facet_kws={"sharex": "col", "sharey": "row"},
        height=4,
        aspect=1.2,
        s=10,
        alpha=0.5,
    )

    # Add one global regression line per facet and annotate slope/intercept
    def add_regline_and_stats(data: pd.DataFrame, color: Optional[str] = None, **kws):
        ax = plt.gca()
        x = data["x"].to_numpy()
        y = data["y"].to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 2:
            slope, intercept = np.polyfit(x[m], y[m], 1)
            xx = np.linspace(x[m].min(), x[m].max(), 200)
            ax.plot(xx, slope * xx + intercept, color="k", lw=2)
            ax.text(
                0.02,
                0.98,
                f"y={slope:.2g}x+{intercept:.2g}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="k",
            )

    g.map_dataframe(add_regline_and_stats)

    # Axis labels per row/col (only outer axes)
    for i, row_key in enumerate(deps):
        for j, col_key in enumerate(indeps):
            ax = g.axes[i, j]
            ax.grid(True)
            if i == len(deps) - 1:
                ax.set_xlabel(labels.get(col_key, col_key))
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(labels.get(row_key, row_key))
            else:
                ax.set_ylabel("")

    plt.tight_layout()
    g.figure.savefig(
        os.path.join(save_path, "branch_final.png"), dpi=300, bbox_inches="tight"
    )
    plt.close(g.figure)


__all__ = ["CompartmentSNRAnalyzer"]
