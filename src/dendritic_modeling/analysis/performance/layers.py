"""
Layer contribution analysis module.

This module contains classes for analyzing the contribution of individual layers to overall
model performance by testing each layer in isolation and measuring its individual impact.
"""

import os
import json
import logging
from typing import Optional
import torch
import pandas as pd

from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils import measure_performance
from dendritic_modeling.plotting.analysis import LayerAnalysisPlotter
from dendritic_modeling.config.analysis import SingleLayerContributionAnalysisParams


class SingleLayerContributionAnalyzer:
    """
    Analyzes the contribution of individual layers by isolating them.
    This is the renamed version of "layer_ablation" - it removes all other layers
    to test what each layer can do on its own.
    """

    def __init__(self, params: SingleLayerContributionAnalysisParams):
        self.batch_size = params.batch_size
        self.save_plots = params.save_plots
        self.save_csv = params.save_csv
        self.logger = logging.getLogger(__name__)

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cuda",
        save_path: Optional[str] = None,
        filename: str = "single_layer_contribution",
    ):
        """Analyze each layer's contribution in isolation"""
        self.logger.info("Starting single layer contribution analysis...")

        # Ensure model is on the correct device
        model = model.to(device)
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
        baseline_acc = measure_performance(net, test_dataset, device=device)
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
                net, test_dataset, device=device
            )

            # Test with only excitatory active
            if (
                hasattr(branch_layers[layer_idx], "branch_inhibition")
                and branch_layers[layer_idx].branch_inhibition is not None
            ):
                branch_layers[layer_idx].branch_inhibition.pre_w.data.fill_(-1000.0)
                exc_only_acc = measure_performance(
                    net, test_dataset, device=device
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
                    net, test_dataset, device=device
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
            neither_acc = measure_performance(net, test_dataset, device=device)

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
                # Use the new unified plotting system
                plotter = LayerAnalysisPlotter()
                
                # Prepare data in the format expected by plot_layer_contributions
                plot_results = {
                    'accuracies': {
                        i: results['isolated_baseline_acc'][i] 
                        for i in range(len(results['layer_idx']))
                    }
                }
                
                fig, ax = plotter.plot_layer_contributions(
                    plot_results,
                    metrics=['accuracy'],
                    save_path=save_path,
                    filename=filename,
                    reverse_layers=True
                )
                self.logger.info(f"Saved plot to {save_path}/{filename}.png")

        return results 