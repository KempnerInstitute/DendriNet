"""
Layer contribution analysis module.

This module contains classes for analyzing the contribution of individual layers to overall
model performance by testing each layer in isolation and measuring its individual impact.
"""

import os
import json
import logging
from typing import List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils import measure_performance


class SingleLayerContributionAnalyzer:
    """
    Analyzes the contribution of individual layers by isolating them.
    This is the renamed version of "layer_ablation" - it removes all other layers
    to test what each layer can do on its own.
    """

    def __init__(
        self,
        batch_size: int = 32,
        device: str = "cuda",
        save_plots: bool = True,
        save_csv: bool = True,
    ):
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
        filename: str = "single_layer_contribution",
    ):
        """Analyze each layer's contribution in isolation"""
        self.logger.info("Starting single layer contribution analysis...")

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

        # Get baseline performance with all layers active
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

        # Store original parameters
        original_params = {}
        for n, p in net.named_parameters():
            original_params[n] = p.detach().clone()

        # Initialize results
        results = {
            "layer_idx": [],
            "isolated_baseline_acc": [],
            "isolated_exc_only_acc": [],
            "isolated_inh_only_acc": [],
            "isolated_neither_acc": [],
        }

        # Analyze each layer in isolation
        for layer_idx in range(len(branch_layers)):
            self.logger.info(f"Analyzing layer {layer_idx} in isolation...")

            # First, ablate all OTHER layers (set to very negative values)
            for i in range(len(branch_layers)):
                if i != layer_idx:
                    if (
                        hasattr(branch_layers[i], "branch_excitation")
                        and branch_layers[i].branch_excitation is not None
                    ):
                        branch_layers[i].branch_excitation.pre_w.data.fill_(-1000.0)
                    if (
                        hasattr(branch_layers[i], "branch_inhibition")
                        and branch_layers[i].branch_inhibition is not None
                    ):
                        branch_layers[i].branch_inhibition.pre_w.data.fill_(-1000.0)

            # Measure with only this layer active (both E and I)
            isolated_baseline = measure_performance(
                net, test_dataset, device=self.device
            )

            # Test with only excitatory active
            if (
                hasattr(branch_layers[layer_idx], "branch_inhibition")
                and branch_layers[layer_idx].branch_inhibition is not None
            ):
                branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(-1000.0)
                exc_only_acc = measure_performance(
                    net, test_dataset, device=self.device
                )
                # Restore inhibitory
                for n, p in net.named_parameters():
                    if f"branch_layers.{layer_idx}.branch_inhibition" in n:
                        p.data.copy_(original_params[n])
            else:
                exc_only_acc = isolated_baseline

            # Test with only inhibitory active
            if (
                hasattr(branch_layers[layer_idx], "branch_excitation")
                and branch_layers[layer_idx].branch_excitation is not None
            ):
                branch_layers[layer_idx].branch_excitation.pre_w.data.fill_(-1000.0)
                inh_only_acc = measure_performance(
                    net, test_dataset, device=self.device
                )
                # Restore excitatory
                for n, p in net.named_parameters():
                    if f"branch_layers.{layer_idx}.branch_excitation" in n:
                        p.data.copy_(original_params[n])
            else:
                inh_only_acc = isolated_baseline

            # Test with neither (both ablated)
            if (
                hasattr(branch_layers[layer_idx], "branch_excitation")
                and branch_layers[layer_idx].branch_excitation is not None
            ):
                branch_layers[layer_idx].branch_excitation.pre_w.data.fill_(-1000.0)
            if (
                hasattr(branch_layers[layer_idx], "branch_inhibition")
                and branch_layers[layer_idx].branch_inhibition is not None
            ):
                branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(-1000.0)
            neither_acc = measure_performance(net, test_dataset, device=self.device)

            # Restore all parameters
            for n, p in net.named_parameters():
                p.data.copy_(original_params[n])

            # Store results
            results["layer_idx"].append(layer_idx)
            results["isolated_baseline_acc"].append(isolated_baseline)
            results["isolated_exc_only_acc"].append(exc_only_acc)
            results["isolated_inh_only_acc"].append(inh_only_acc)
            results["isolated_neither_acc"].append(neither_acc)

            self.logger.info(
                f"Layer {layer_idx}: baseline={isolated_baseline:.4f}, "
                f"exc_only={exc_only_acc:.4f}, inh_only={inh_only_acc:.4f}, "
                f"neither={neither_acc:.4f}"
            )

        # Save results
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

            if self.save_csv:
                df = pd.DataFrame(results)
                df["full_network_baseline"] = baseline_acc
                json_path = os.path.join(save_path, f"{filename}.json")
                # Save as JSON instead of CSV
                results_dict = df.to_dict(orient='records')
                with open(json_path, 'w') as f:
                    json.dump(results_dict, f, indent=2)
                self.logger.info(f"Saved results to {json_path}")

            if self.save_plots:
                # Reverse layer indices so soma (last layer) becomes 0
                num_layers = len(results["layer_idx"])
                reversed_layer_idx = [
                    num_layers - 1 - idx for idx in results["layer_idx"]
                ]

                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                # Left plot: Absolute accuracy values
                x = np.array(reversed_layer_idx)

                # Reverse the data to match the x positions
                baseline_acc_reversed = [
                    results["isolated_baseline_acc"][i]
                    for i in range(num_layers - 1, -1, -1)
                ]
                exc_only_acc_reversed = [
                    results["isolated_exc_only_acc"][i]
                    for i in range(num_layers - 1, -1, -1)
                ]
                inh_only_acc_reversed = [
                    results["isolated_inh_only_acc"][i]
                    for i in range(num_layers - 1, -1, -1)
                ]
                neither_acc_reversed = [
                    results["isolated_neither_acc"][i]
                    for i in range(num_layers - 1, -1, -1)
                ]

                ax1.plot(
                    x,
                    baseline_acc_reversed,
                    "o-",
                    label="Both E+I",
                    linewidth=3,
                    markersize=10,
                    color="purple",
                )
                ax1.plot(
                    x,
                    exc_only_acc_reversed,
                    "s-",
                    label="Excitatory only",
                    linewidth=3,
                    markersize=10,
                    color="red",
                )
                ax1.plot(
                    x,
                    inh_only_acc_reversed,
                    "^-",
                    label="Inhibitory only",
                    linewidth=3,
                    markersize=10,
                    color="blue",
                )
                ax1.plot(
                    x,
                    neither_acc_reversed,
                    "v-",
                    label="Neither",
                    linewidth=3,
                    markersize=10,
                    color="gray",
                )
                ax1.axhline(
                    y=baseline_acc,
                    color="k",
                    linestyle="--",
                    linewidth=2,
                    label=f"Full network baseline ({baseline_acc:.3f})",
                )

                # Set x-axis ticks and labels
                ax1.set_xticks(x)
                # Create labels based on position, not value
                x_labels = ["Soma" if i == 0 else f"{i}" for i in range(len(x))]
                ax1.set_xticklabels(x_labels)

                ax1.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax1.set_ylabel("Classification Accuracy", fontsize=14)
                ax1.set_title(
                    "Single Layer Contribution Analysis\n(Each layer tested in isolation)",
                    fontsize=16,
                )
                ax1.legend(fontsize=12)
                ax1.grid(True, alpha=0.3)
                ax1.set_ylim(
                    [0, max(max(results["isolated_baseline_acc"]), baseline_acc) * 1.1]
                )

                # Right plot: Contribution breakdown (bar chart)
                width = 0.25

                # Calculate contributions
                exc_contrib = [
                    b - n
                    for b, n in zip(
                        results["isolated_exc_only_acc"],
                        results["isolated_neither_acc"],
                    )
                ]
                inh_contrib = [
                    b - n
                    for b, n in zip(
                        results["isolated_inh_only_acc"],
                        results["isolated_neither_acc"],
                    )
                ]
                synergy = [
                    full - max(e, i)
                    for full, e, i in zip(
                        results["isolated_baseline_acc"],
                        results["isolated_exc_only_acc"],
                        results["isolated_inh_only_acc"],
                    )
                ]

                # Debug: Print contributions before reversal
                self.logger.info(
                    f"Before reversal - Excitatory contributions: {exc_contrib}"
                )
                self.logger.info(
                    f"Before reversal - Inhibitory contributions: {inh_contrib}"
                )
                self.logger.info(f"Before reversal - Synergy: {synergy}")

                # Reverse the contributions too
                exc_contrib = [exc_contrib[i] for i in range(num_layers - 1, -1, -1)]
                inh_contrib = [inh_contrib[i] for i in range(num_layers - 1, -1, -1)]
                synergy = [synergy[i] for i in range(num_layers - 1, -1, -1)]

                # Debug: Print contributions after reversal
                self.logger.info(
                    f"After reversal - Excitatory contributions: {exc_contrib}"
                )
                self.logger.info(
                    f"After reversal - Inhibitory contributions: {inh_contrib}"
                )
                self.logger.info(f"After reversal - Synergy: {synergy}")
                self.logger.info(f"Plotting x positions: {x.tolist()}")

                ax2.bar(
                    x - width,
                    exc_contrib,
                    width,
                    label="Excitatory contribution",
                    color="red",
                    alpha=0.7,
                )
                ax2.bar(
                    x,
                    inh_contrib,
                    width,
                    label="Inhibitory contribution",
                    color="blue",
                    alpha=0.7,
                )
                ax2.bar(
                    x + width,
                    synergy,
                    width,
                    label="E+I synergy",
                    color="purple",
                    alpha=0.7,
                )

                # Set x-axis
                ax2.set_xticks([i * 3 for i in range(len(reversed_layer_idx))])
                # Create position-based labels
                x_labels = [
                    "Soma" if i == 0 else f"{i}" for i in range(len(reversed_layer_idx))
                ]
                ax2.set_xticklabels(x_labels)
                ax2.set_xlabel("Layer (Soma → Distal)", fontsize=14)

                ax2.set_ylabel("Accuracy Contribution", fontsize=14)
                ax2.set_title(
                    "Layer Contributions Breakdown\n(When tested in isolation)",
                    fontsize=16,
                )
                ax2.legend(fontsize=12)
                ax2.grid(True, axis="y", alpha=0.3)
                ax2.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

                plt.tight_layout()
                plot_path = os.path.join(save_path, f"{filename}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                self.logger.info(f"Saved plot to {plot_path}")

        return results 