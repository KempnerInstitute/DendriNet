"""
Synaptic ablation analysis module.

This module contains classes for performing synaptic ablation studies - removing specific 
synapses while keeping the rest of the network intact to analyze their individual contributions
to overall model performance.
"""

import os
import logging
from typing import List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils import ablate_branch, measure_performance


class SynapticAblationAnalyzer:
    """
    Performs synaptic ablation analysis - removes specific synapses while keeping
    the rest of the network intact. This is the renamed version of "branch_ablation".
    """

    def __init__(
        self,
        ablation_type: str = "whole_layer",  # "whole_layer" or "single_branch"
        layers_to_analyze: List[int] = [],  # Empty means all layers
        mode: str = "direct",  # "direct" or "full"
        batch_size: int = 32,
        device: str = "cuda",
        save_plots: bool = True,
        save_csv: bool = True,
    ):
        self.ablation_type = ablation_type
        self.layers_to_analyze = layers_to_analyze
        self.mode = mode
        self.batch_size = batch_size
        self.device = device
        self.save_plots = save_plots
        self.save_csv = save_csv
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        save_path: Optional[str] = None,
        filename: str = "synaptic_ablation",
    ):
        """Perform synaptic ablation analysis"""
        self.logger.info(
            f"Starting synaptic ablation analysis (type: {self.ablation_type})..."
        )

        # Ensure model is on the correct device
        model = model.to(self.device)
        # Get the underlying network
        if hasattr(model, "net"):
            net = model.net
        else:
            net = model

        # Debug logging
        self.logger.info(f"Model type: {type(model)}")
        self.logger.info(f"Net type: {type(net)}")
        self.logger.info(
            f"Net attributes: {[attr for attr in dir(net) if 'branch' in attr]}"
        )

        # Check layers structure
        if hasattr(net, "layers"):
            self.logger.info(f"Number of EI layers: {len(net.layers)}")
            for i, layer in enumerate(net.layers):
                self.logger.info(
                    f"Layer {i}: has excitatory_cells={hasattr(layer, 'excitatory_cells')}, has inhibitory_cells={hasattr(layer, 'inhibitory_cells')}"
                )
                if (
                    hasattr(layer, "excitatory_cells")
                    and layer.excitatory_cells is not None
                ):
                    self.logger.info(
                        f"  Excitatory cells type: {type(layer.excitatory_cells)}"
                    )
                    self.logger.info(
                        f"  Excitatory has branch_layers: {hasattr(layer.excitatory_cells, 'branch_layers')}"
                    )
                if (
                    hasattr(layer, "inhibitory_cells")
                    and layer.inhibitory_cells is not None
                ):
                    self.logger.info(
                        f"  Inhibitory cells type: {type(layer.inhibitory_cells)}"
                    )
                    self.logger.info(
                        f"  Inhibitory has branch_layers: {hasattr(layer.inhibitory_cells, 'branch_layers')}"
                    )

        # Get branch layers
        try:
            branch_layers = net.branch_layers
            self.logger.info(
                f"Successfully got branch_layers: {len(branch_layers)} layers"
            )
        except Exception as e:
            self.logger.error(f"Error accessing branch_layers: {type(e).__name__}: {e}")
            return

        # Get baseline performance
        baseline_acc = measure_performance(net, test_dataset, device=self.device)
        self.logger.info(f"Baseline accuracy: {baseline_acc:.4f}")

        # Log layer structure for debugging
        self.logger.info(f"Total branch layers: {len(branch_layers)}")
        for i, layer in enumerate(branch_layers):
            has_exc = (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
            )
            has_inh = (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
            )
            self.logger.info(
                f"Layer {i}: has_excitation={has_exc}, has_inhibition={has_inh}"
            )

        # Determine which layers to analyze
        if self.layers_to_analyze:
            layer_indices = self.layers_to_analyze
        else:
            layer_indices = list(range(len(branch_layers)))

        # Store original parameters
        original_params = {}
        for n, p in net.named_parameters():
            original_params[n] = p.detach().clone()

        # Initialize results based on ablation type
        if self.ablation_type == "whole_layer":
            results = self._analyze_whole_layers(
                net,
                branch_layers,
                layer_indices,
                baseline_acc,
                original_params,
                test_dataset,
            )
        else:  # single_branch
            results = self._analyze_single_branches(
                net,
                branch_layers,
                layer_indices,
                baseline_acc,
                original_params,
                test_dataset,
            )

        # Save results
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            self._save_results(results, baseline_acc, save_path, filename)

        return results

    def _analyze_whole_layers(
        self,
        net,
        branch_layers,
        layer_indices,
        baseline_acc,
        original_params,
        test_dataset,
    ):
        """Analyze ablation of whole layers"""
        results = {
            "layer_idx": [],
            "both_ablated_acc": [],
            "exc_ablated_acc": [],
            "inh_ablated_acc": [],
            "both_drop": [],
            "exc_drop": [],
            "inh_drop": [],
        }

        for layer_idx in layer_indices:
            layer = branch_layers[layer_idx]
            self.logger.info(f"Ablating layer {layer_idx}...")

            # Ablate both E and I
            if (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
            ):
                layer.branch_excitation.pre_w.data.zero_()
            if (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
            ):
                layer.branch_inhibition.pre_w.data.zero_()
            both_acc = measure_performance(net, test_dataset, device=self.device)

            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Ablate only excitatory
            if (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
            ):
                layer.branch_excitation.pre_w.data.zero_()
                exc_acc = measure_performance(net, test_dataset, device=self.device)
            else:
                exc_acc = baseline_acc

            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Ablate only inhibitory
            if (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
            ):
                layer.branch_inhibition.pre_w.data.zero_()
                inh_acc = measure_performance(net, test_dataset, device=self.device)
            else:
                inh_acc = baseline_acc

            # Restore parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Store results
            results["layer_idx"].append(layer_idx)
            results["both_ablated_acc"].append(both_acc)
            results["exc_ablated_acc"].append(exc_acc)
            results["inh_ablated_acc"].append(inh_acc)
            results["both_drop"].append(baseline_acc - both_acc)
            results["exc_drop"].append(baseline_acc - exc_acc)
            results["inh_drop"].append(baseline_acc - inh_acc)

            self.logger.info(
                f"Layer {layer_idx}: both_drop={baseline_acc - both_acc:.4f}, "
                f"exc_drop={baseline_acc - exc_acc:.4f}, inh_drop={baseline_acc - inh_acc:.4f}"
            )

        return results

    def _analyze_single_branches(
        self,
        net,
        branch_layers,
        layer_indices,
        baseline_acc,
        original_params,
        test_dataset,
    ):
        """Analyze ablation of individual branches"""
        results = {
            "branch_id": [],
            "layer_idx": [],
            "branch_idx": [],
            "both_ablated_acc": [],
            "exc_ablated_acc": [],
            "inh_ablated_acc": [],
            "both_drop": [],
            "exc_drop": [],
            "inh_drop": [],
        }

        for layer_idx in layer_indices:
            layer = branch_layers[layer_idx]

            # Determine number of branches
            n_branches = 0
            if (
                hasattr(layer, "branch_excitation")
                and layer.branch_excitation is not None
                and hasattr(layer.branch_excitation, "pre_w")
            ):
                n_branches = layer.branch_excitation.pre_w.shape[0]
            elif (
                hasattr(layer, "branch_inhibition")
                and layer.branch_inhibition is not None
                and hasattr(layer.branch_inhibition, "pre_w")
            ):
                n_branches = layer.branch_inhibition.pre_w.shape[0]

            if n_branches == 0:
                continue

            self.logger.info(f"Analyzing {n_branches} branches in layer {layer_idx}...")

            for branch_idx in range(n_branches):
                # Ablate both E and I for this branch
                ablate_branch(net, layer_idx, [branch_idx], mode=self.mode)
                both_acc = measure_performance(net, test_dataset, device=self.device)

                # Restore
                for n, p in net.named_parameters():
                    p.data.copy_(original_params[n])

                # Ablate only excitatory
                if (
                    hasattr(layer, "branch_excitation")
                    and layer.branch_excitation is not None
                ):
                    layer.branch_excitation.pre_w.data[branch_idx].zero_()
                    exc_acc = measure_performance(net, test_dataset, device=self.device)
                else:
                    exc_acc = baseline_acc

                # Restore
                for n, p in net.named_parameters():
                    p.data.copy_(original_params[n])

                # Ablate only inhibitory
                if (
                    hasattr(layer, "branch_inhibition")
                    and layer.branch_inhibition is not None
                ):
                    layer.branch_inhibition.pre_w.data[branch_idx].zero_()
                    inh_acc = measure_performance(net, test_dataset, device=self.device)
                else:
                    inh_acc = baseline_acc

                # Restore
                for n, p in net.named_parameters():
                    p.data.copy_(original_params[n])

                # Store results
                branch_id = f"{layer_idx}-{branch_idx}"
                results["branch_id"].append(branch_id)
                results["layer_idx"].append(layer_idx)
                results["branch_idx"].append(branch_idx)
                results["both_ablated_acc"].append(both_acc)
                results["exc_ablated_acc"].append(exc_acc)
                results["inh_ablated_acc"].append(inh_acc)
                results["both_drop"].append(baseline_acc - both_acc)
                results["exc_drop"].append(baseline_acc - exc_acc)
                results["inh_drop"].append(baseline_acc - inh_acc)

        return results

    def _save_results(self, results, baseline_acc, save_path, filename):
        """Save ablation results"""
        if self.save_csv:
            df = pd.DataFrame(results)
            df["baseline_acc"] = baseline_acc
            csv_path = os.path.join(save_path, f"{filename}_{self.ablation_type}.csv")
            df.to_csv(csv_path, index=False)
            self.logger.info(f"Saved results to {csv_path}")

        if self.save_plots:
            if self.ablation_type == "whole_layer":
                # Reverse layer indices so soma (last layer) becomes 0
                num_layers = len(results["layer_idx"])
                reversed_layer_idx = [
                    num_layers - 1 - idx for idx in results["layer_idx"]
                ]

                # Reverse the results too
                both_drop_reversed = [
                    results["both_drop"][num_layers - 1 - i] for i in range(num_layers)
                ]
                exc_drop_reversed = [
                    results["exc_drop"][num_layers - 1 - i] for i in range(num_layers)
                ]
                inh_drop_reversed = [
                    results["inh_drop"][num_layers - 1 - i] for i in range(num_layers)
                ]

                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                # Left plot: Accuracy drop (bar chart)
                x = np.array(reversed_layer_idx)
                width = 0.25

                ax1.bar(
                    x - width,
                    both_drop_reversed,
                    width,
                    label="Both E+I ablated",
                    color="purple",
                    alpha=0.7,
                )
                ax1.bar(
                    x,
                    exc_drop_reversed,
                    width,
                    label="Excitatory ablated",
                    color="red",
                    alpha=0.7,
                )
                ax1.bar(
                    x + width,
                    inh_drop_reversed,
                    width,
                    label="Inhibitory ablated",
                    color="blue",
                    alpha=0.7,
                )

                # Set x-axis ticks and labels
                ax1.set_xticks(x)
                # Create labels based on position, not value
                x_labels = ["Soma" if i == 0 else f"{i}" for i in range(len(x))]
                ax1.set_xticklabels(x_labels)

                ax1.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax1.set_ylabel("Accuracy Drop", fontsize=14)
                ax1.set_title(
                    f"Synaptic Ablation Analysis - Whole Layers\n(Baseline accuracy: {baseline_acc:.3f})",
                    fontsize=16,
                )
                ax1.legend(fontsize=12)
                ax1.grid(True, axis="y", alpha=0.3)
                ax1.set_ylim([0, max(max(both_drop_reversed), 0.1) * 1.2])

                # Right plot: Remaining accuracy after ablation
                both_acc_reversed = [
                    results["both_ablated_acc"][num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                exc_acc_reversed = [
                    results["exc_ablated_acc"][num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                inh_acc_reversed = [
                    results["inh_ablated_acc"][num_layers - 1 - i]
                    for i in range(num_layers)
                ]

                ax2.plot(
                    x,
                    [baseline_acc] * len(x),
                    "k--",
                    linewidth=2,
                    label=f"Baseline ({baseline_acc:.3f})",
                )
                ax2.plot(
                    x,
                    both_acc_reversed,
                    "o-",
                    label="Both E+I ablated",
                    linewidth=3,
                    markersize=10,
                    color="purple",
                )
                ax2.plot(
                    x,
                    exc_acc_reversed,
                    "s-",
                    label="Excitatory ablated",
                    linewidth=3,
                    markersize=10,
                    color="red",
                )
                ax2.plot(
                    x,
                    inh_acc_reversed,
                    "^-",
                    label="Inhibitory ablated",
                    linewidth=3,
                    markersize=10,
                    color="blue",
                )

                # Set x-axis ticks and labels
                ax2.set_xticks(x)
                # Create labels based on position, not value
                x_labels = ["Soma" if i == 0 else f"{i}" for i in range(len(x))]
                ax2.set_xticklabels(x_labels)

                ax2.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax2.set_ylabel("Classification Accuracy", fontsize=14)
                ax2.set_title("Remaining Accuracy After Layer Ablation", fontsize=16)
                ax2.legend(fontsize=12)
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim([0, baseline_acc * 1.1])

                plt.tight_layout()
                plot_path = os.path.join(save_path, f"{filename}_whole_layer.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

            else:  # single_branch
                # Calculate layer-wise statistics
                df = pd.DataFrame(results)
                layer_stats = (
                    df.groupby("layer_idx")
                    .agg(
                        {
                            "both_drop": ["mean", "std"],
                            "exc_drop": ["mean", "std"],
                            "inh_drop": ["mean", "std"],
                        }
                    )
                    .reset_index()
                )

                # Flatten column names
                layer_stats.columns = [
                    "layer_idx",
                    "both_drop_mean",
                    "both_drop_std",
                    "exc_drop_mean",
                    "exc_drop_std", 
                    "inh_drop_mean",
                    "inh_drop_std",
                ]

                # Create plots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                # Reverse layer order for plotting (soma first)
                layer_stats_sorted = layer_stats.sort_values("layer_idx", ascending=False)
                x = range(len(layer_stats_sorted))

                # Left plot: Mean accuracy drop with error bars
                ax1.errorbar(
                    x,
                    layer_stats_sorted["both_drop_mean"],
                    yerr=layer_stats_sorted["both_drop_std"],
                    label="Both E+I ablated",
                    marker="o",
                    linewidth=2,
                    markersize=8,
                    color="purple",
                )
                ax1.errorbar(
                    x,
                    layer_stats_sorted["exc_drop_mean"],
                    yerr=layer_stats_sorted["exc_drop_std"],
                    label="Excitatory ablated",
                    marker="s",
                    linewidth=2,
                    markersize=8,
                    color="red",
                )
                ax1.errorbar(
                    x,
                    layer_stats_sorted["inh_drop_mean"],
                    yerr=layer_stats_sorted["inh_drop_std"],
                    label="Inhibitory ablated",
                    marker="^",
                    linewidth=2,
                    markersize=8,
                    color="blue",
                )

                # Set x-axis labels (soma first)
                x_labels = []
                for idx in layer_stats_sorted["layer_idx"]:
                    if idx == layer_stats_sorted["layer_idx"].max():
                        x_labels.append("Soma")
                    else:
                        x_labels.append(
                            str(layer_stats_sorted["layer_idx"].max() - idx)
                        )

                ax1.set_xticks(x)
                ax1.set_xticklabels(x_labels)
                ax1.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax1.set_ylabel("Mean Accuracy Drop", fontsize=14)
                ax1.set_title(
                    f"Synaptic Ablation Analysis - Single Branches\n(Baseline accuracy: {baseline_acc:.3f})",
                    fontsize=16,
                )
                ax1.legend(fontsize=12)
                ax1.grid(True, alpha=0.3)

                # Right plot: Distribution of accuracy drops for each layer
                layer_indices = sorted(df["layer_idx"].unique(), reverse=True)
                colors = ["purple", "red", "blue"]
                labels = ["Both E+I", "Excitatory", "Inhibitory"]
                metrics = ["both_drop", "exc_drop", "inh_drop"]

                width = 0.25
                for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
                    means = []
                    for layer_idx in layer_indices:
                        layer_data = df[df["layer_idx"] == layer_idx][metric]
                        means.append(layer_data.mean())

                    x_pos = [j + i * width for j in range(len(layer_indices))]
                    ax2.bar(x_pos, means, width, label=label, color=color, alpha=0.7)

                # Set x-axis labels
                x_labels = []
                for idx in layer_indices:
                    if idx == max(layer_indices):
                        x_labels.append("Soma")
                    else:
                        x_labels.append(str(max(layer_indices) - idx))

                ax2.set_xticks([j + width for j in range(len(layer_indices))])
                ax2.set_xticklabels(x_labels)
                ax2.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax2.set_ylabel("Mean Accuracy Drop", fontsize=14)
                ax2.set_title("Mean Branch Ablation Effects by Layer", fontsize=16)
                ax2.legend(fontsize=12)
                ax2.grid(True, axis="y", alpha=0.3)

                plt.tight_layout()
                plot_path = os.path.join(save_path, f"{filename}_single_branch.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

            self.logger.info(f"Saved plot to {plot_path}") 