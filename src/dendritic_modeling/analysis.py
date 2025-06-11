import logging
import math
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from torch.utils.data import DataLoader, TensorDataset

from dendritic_modeling.config import AnalysisConfig
from dendritic_modeling.dendrinet import DendriticBranchLayer
from dendritic_modeling.information import (
    compute_mi_cc,
    compute_mi_cd,
    compute_pairwise_sr,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.utils import (
    add_gaussian_noise,
    add_uniform_noise,
    evaluate_accuracy,
    evaluate_categorical_loglikelihood,
    evaluate_mse,
    save_dict,
)

# Import moved classes from new locations
from dendritic_modeling.analysis.performance.metrics import PerformanceAnalyzer
from dendritic_modeling.analysis.performance.noise import NoisePerturbationAnalyzer
from dendritic_modeling.analysis.performance.ablation import AblationAnalyzer
from dendritic_modeling.analysis.information.mutual_info import EIInformationAnalyzer
from dendritic_modeling.analysis.synaptic.turnover import SynapticTurnoverAnalyzer
from dendritic_modeling.analysis.synaptic.weights import WeightAnalyzer
from dendritic_modeling.analysis.synaptic.activations import SynapticActivationAnalyzer


# PerformanceAnalyzer has been moved to analysis/performance/metrics.py






# NoisePerturbationAnalyzer has been moved to analysis/performance/noise.py


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
        import matplotlib.pyplot as plt
        import pandas as pd

        from dendritic_modeling.utils import ablate_branch, measure_performance

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
        from dendritic_modeling.utils import measure_performance

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
        from dendritic_modeling.utils import ablate_branch, measure_performance

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
        import matplotlib.pyplot as plt
        import pandas as pd

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

                # Reverse layer indices
                num_layers = len(layer_stats)
                reversed_layer_idx = [num_layers - 1 - idx for idx in range(num_layers)]

                # Create figure with two subplots
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

                # Left plot: Mean accuracy drop per layer
                x = np.array(reversed_layer_idx)
                width = 0.25

                # Reverse the statistics
                both_mean = [
                    layer_stats["both_drop"]["mean"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                both_std = [
                    layer_stats["both_drop"]["std"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                exc_mean = [
                    layer_stats["exc_drop"]["mean"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                exc_std = [
                    layer_stats["exc_drop"]["std"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                inh_mean = [
                    layer_stats["inh_drop"]["mean"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]
                inh_std = [
                    layer_stats["inh_drop"]["std"].iloc[num_layers - 1 - i]
                    for i in range(num_layers)
                ]

                ax1.bar(
                    x - width,
                    both_mean,
                    width,
                    yerr=both_std,
                    label="Both E+I",
                    color="purple",
                    alpha=0.7,
                    capsize=5,
                )
                ax1.bar(
                    x,
                    exc_mean,
                    width,
                    yerr=exc_std,
                    label="Excitatory",
                    color="red",
                    alpha=0.7,
                    capsize=5,
                )
                ax1.bar(
                    x + width,
                    inh_mean,
                    width,
                    yerr=inh_std,
                    label="Inhibitory",
                    color="blue",
                    alpha=0.7,
                    capsize=5,
                )

                ax1.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax1.set_ylabel("Mean Accuracy Drop (± std)", fontsize=14)
                ax1.set_title(
                    f"Synaptic Ablation - Single Branches\n(Baseline: {baseline_acc:.3f})",
                    fontsize=16,
                )
                ax1.legend(fontsize=12)
                ax1.grid(True, axis="y", alpha=0.3)

                # Right plot: Distribution of drops across branches
                # Create box plots for each layer
                positions = []
                data_to_plot = []
                labels = []
                colors = []

                for i, layer_idx in enumerate(reversed_layer_idx):
                    original_idx = num_layers - 1 - layer_idx
                    layer_data = df[df["layer_idx"] == original_idx]

                    positions.extend([i * 3 - 0.6, i * 3, i * 3 + 0.6])
                    data_to_plot.extend(
                        [
                            layer_data["both_drop"].values,
                            layer_data["exc_drop"].values,
                            layer_data["inh_drop"].values,
                        ]
                    )
                    if i == 0:  # Only add labels once
                        labels.extend(["Both E+I", "Excitatory", "Inhibitory"])
                    colors.extend(["purple", "red", "blue"])

                bp = ax2.boxplot(
                    data_to_plot, positions=positions, widths=0.5, patch_artist=True
                )

                # Color the boxes
                for patch, color in zip(bp["boxes"], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Set x-axis
                ax2.set_xticks([i * 3 for i in range(len(reversed_layer_idx))])
                # Create position-based labels
                x_labels = [
                    "Soma" if i == 0 else f"{i}" for i in range(len(reversed_layer_idx))
                ]
                ax2.set_xticklabels(x_labels)
                ax2.set_xlabel("Layer (Soma → Distal)", fontsize=14)
                ax2.set_ylabel("Accuracy Drop Distribution", fontsize=14)
                ax2.set_title("Branch-wise Accuracy Drop Distribution", fontsize=16)
                ax2.grid(True, axis="y", alpha=0.3)

                # Add a custom legend
                legend_elements = [
                    Patch(facecolor="purple", alpha=0.7, label="Both E+I"),
                    Patch(facecolor="red", alpha=0.7, label="Excitatory"),
                    Patch(facecolor="blue", alpha=0.7, label="Inhibitory"),
                ]
                ax2.legend(handles=legend_elements, fontsize=12)

                plt.tight_layout()
                plot_path = os.path.join(save_path, f"{filename}_single_branch.png")
                plt.savefig(plot_path, dpi=300, bbox_inches="tight")
                plt.close()

                # Save summary statistics with reversed indices
                if self.save_csv:
                    summary_df = pd.DataFrame(
                        {
                            "layer_idx_reversed": reversed_layer_idx,
                            "layer_idx_original": [
                                num_layers - 1 - idx for idx in reversed_layer_idx
                            ],
                            "both_drop_mean": both_mean,
                            "both_drop_std": both_std,
                            "exc_drop_mean": exc_mean,
                            "exc_drop_std": exc_std,
                            "inh_drop_mean": inh_mean,
                            "inh_drop_std": inh_std,
                        }
                    )
                    summary_path = os.path.join(
                        save_path, f"{filename}_single_branch_summary.csv"
                    )
                    summary_df.to_csv(summary_path, index=False)

            self.logger.info(f"Saved plots to {save_path}")


class AnalysisManager:
    def __init__(
        self,
        model: BaseModel,
        data: Dict[str, torch.utils.data.Dataset],
        analysis_config: AnalysisConfig,
        save_root: Optional[str] = None,
        device: str = "cpu",
        train_config: Optional[Dict] = None,
    ):
        self.model = model
        self.data = data
        self.config = analysis_config
        self.train_config = train_config or {}
        self.save_root = save_root
        self.device = device
        os.makedirs(self.save_root, exist_ok=True)
        # Note: Logging should be configured externally at the application startup.
        self.logger = logging.getLogger(__name__)

        performance_config = self.config.performance_analysis
        self.performance_enabled = performance_config.get("enabled", False)
        self.performance_training = performance_config.get("training", False)
        if self.performance_enabled:
            performance_params: dict = performance_config.get("params", {})
            self.performance_analyzer = PerformanceAnalyzer(
                accuracy=performance_params.get("accuracy", False),
                categorical_loglikelihood=performance_params.get(
                    "categorical_loglikelihood", False
                ),
                mse=performance_params.get("mse", False),
            )

        ei_info_config = self.config.ei_info_analysis
        self.ei_info_enabled = ei_info_config.get("enabled", False)
        self.ei_info_training = ei_info_config.get("training", False)
        if self.ei_info_enabled:
            ei_info_params: dict = ei_info_config.get("params", {})
            self.ei_info_analyzer = EIInformationAnalyzer(
                compute_mi_E_I=ei_info_params.get("compute_mi_E_I", True),
                compute_mi_EI_Vout=ei_info_params.get("compute_mi_EI_Vout", True),
                compute_sr_EI_Vout=ei_info_params.get("compute_sr_EI_Vout", True),
                compute_mi_EI_label=ei_info_params.get("compute_mi_EI_label", True),
                n_neighbors=ei_info_params.get("n_neighbors", 10),
            )

        syn_turnover_config = self.config.synapse_turnover_analysis
        self.syn_turnover_enabled = syn_turnover_config.get("enabled", False)
        self.syn_turnover_training = syn_turnover_config.get("training", False)
        if self.syn_turnover_enabled:
            self.syn_turnover_analyzer = SynapticTurnoverAnalyzer(model=model)

        noise_config = self.config.noise_perturbation_analysis
        self.noise_enabled = noise_config.get("enabled", False)
        self.noise_training = noise_config.get("training", False)
        if self.noise_enabled:
            noise_params: dict = noise_config.get("params", {})
            self.noise_perturbation_analyzer = NoisePerturbationAnalyzer(
                uniform_noise=noise_params.get("uniform_noise", True),
                uniform_magnitudes=noise_params.get("uniform_magnitudes", [0.1]),
                gaussian_noise=noise_params.get("gaussian_noise", True),
                gaussian_sdevs=noise_params.get("gaussian_sdevs", [0.1]),
                n_samples=noise_params.get("n_samples", 10),
                accuracy=noise_params.get("accuracy", True),
                categorical_loglikelihood=noise_params.get(
                    "categorical_loglikelihood", True
                ),
            )

        # Single layer contribution analysis (formerly layer_ablation)
        single_layer_config = self.config.single_layer_contribution
        self.single_layer_enabled = single_layer_config.get("enabled", False)
        self.single_layer_training = single_layer_config.get("training", False)
        if self.single_layer_enabled:
            params = single_layer_config.get("params", {})
            # Get batch_size from train config
            batch_size = self.train_config.get("batch_size", 32)
            self.single_layer_analyzer = SingleLayerContributionAnalyzer(
                batch_size=batch_size,
                device=self.device,
                save_plots=params.get("save_plots", True),
                save_csv=params.get("save_csv", True),
            )

        # Synaptic ablation analysis (formerly branch_ablation)
        synaptic_ablation_config = self.config.synaptic_ablation
        self.synaptic_ablation_enabled = synaptic_ablation_config.get("enabled", False)
        self.synaptic_ablation_training = synaptic_ablation_config.get(
            "training", False
        )
        if self.synaptic_ablation_enabled:
            params = synaptic_ablation_config.get("params", {})
            # Get batch_size from train config
            batch_size = self.train_config.get("batch_size", 32)
            self.synaptic_ablation_analyzer = SynapticAblationAnalyzer(
                ablation_type=params.get("ablation_type", "whole_layer"),
                layers_to_analyze=params.get("layers_to_analyze", []),
                mode=params.get("mode", "direct"),
                batch_size=batch_size,
                device=self.device,
                save_plots=params.get("save_plots", True),
                save_csv=params.get("save_csv", True),
            )

        # Branch information analysis
        branch_info_config = self.config.branch_information
        self.branch_info_enabled = branch_info_config.get("enabled", False)
        self.branch_info_training = branch_info_config.get("training", False)
        if self.branch_info_enabled:
            params = branch_info_config.get("params", {})
            self.branch_info_analyzer = BranchInformationAnalyzer(
                n_neighbors=params.get("n_neighbors", 10),
                use_dvc=params.get("use_dvc", True),
                save_plots=params.get("save_plots", True),
                save_csv=params.get("save_csv", True),
                copula_params=params.get("copula_params", None),
                # Configurable MI metrics to compute
                compute_metrics=params.get("compute_metrics", None),
                # Per-branch vs pooled computation
                compute_per_branch=params.get("compute_per_branch", False),
            )

        ablation_config = self.config.ablation_analysis
        self.ablation_enabled = ablation_config.get("enabled", False)
        self.ablation_training = ablation_config.get("training", False)
        if self.ablation_enabled:
            ablation_params: dict = ablation_config.get("params", {})
            self.ablation_analyzer = AblationAnalyzer(
                ablate_all_synapses=ablation_params.get("ablate_all_synapses", True),
                ablate_excitation=ablation_params.get("ablate_excitation", True),
                ablate_inhibition=ablation_params.get("ablate_inhibition", True),
                ablate_branches=ablation_params.get("ablate_branches", True),
                accuracy=ablation_params.get("accuracy", True),
                categorical_loglikelihood=ablation_params.get(
                    "categorical_loglikelihood", True
                ),
            )

        weight_config = self.config.weight_analysis
        self.weight_enabled = weight_config.get("enabled", False)
        self.weight_training = weight_config.get("training", False)
        if self.weight_enabled:
            weight_params: dict = weight_config.get("params", {})
            self.weight_analyzer = WeightAnalyzer(
                synapse_weights=weight_params.get("synapse_weights", True),
                branch_weights=weight_params.get("branch_weights", True),
            )

        synaptic_activation_config = self.config.synaptic_activation_analysis
        self.synaptic_activation_enabled = synaptic_activation_config.get(
            "enabled", False
        )
        self.synaptic_activation_training = synaptic_activation_config.get(
            "training", False
        )
        if self.synaptic_activation_enabled:
            synaptic_activation_params: dict = synaptic_activation_config.get(
                "params", {}
            )
            self.synaptic_activation_analyzer = SynapticActivationAnalyzer(
                network_summary=synaptic_activation_params.get("network_summary", True),
                layer_summary=synaptic_activation_params.get("layer_summary", True),
                branch_summary=synaptic_activation_params.get("branch_summary", True),
                raw_summary=synaptic_activation_params.get("raw_summary", True),
                logspace=synaptic_activation_params.get("logspace", False),
                downsample=synaptic_activation_params.get("downsample", False),
                samples=synaptic_activation_params.get("samples", 1000),
            )

        branch_activation_config = self.config.branch_activation_analysis
        self.branch_activation_enabled = branch_activation_config.get("enabled", False)
        self.branch_activation_training = branch_activation_config.get(
            "training", False
        )
        if self.branch_activation_enabled:
            branch_activation_params: dict = branch_activation_config.get("params", {})
            self.branch_activation_analyzer = BranchActivationAnalyzer(
                synapse_activations=branch_activation_params.get(
                    "synapse_activations", True
                ),
                branch_activations=branch_activation_params.get(
                    "branch_activations", True
                ),
                network_summary=branch_activation_params.get("network_summary", True),
                layer_summary=branch_activation_params.get("layer_summary", True),
                branch_summary=branch_activation_params.get("branch_summary", True),
                raw_summary=branch_activation_params.get("raw_summary", True),
                logspace=branch_activation_params.get("logspace", False),
                downsample=branch_activation_params.get("downsample", False),
                samples=branch_activation_params.get("samples", 1000),
            )

    def run_analysis(self, filename: str = "final", training: bool = False):
        if self.performance_enabled and (
            (self.performance_training and training) or not training
        ):
            self.performance_analyzer.analyze(
                model=self.model,
                train_ds=self.data["train"],
                valid_ds=self.data["valid"],
                test_ds=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "performance"),
                filename=filename,
            )

        if self.ei_info_enabled and (
            (self.ei_info_training and training) or not training
        ):
            self.ei_info_analyzer.analyze(
                model=self.model,
                dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "ei_info_analysis"),
                filename=filename,
            )

        if self.syn_turnover_enabled and (
            (self.syn_turnover_training and training) or not training
        ):
            self.syn_turnover_analyzer.analyze(
                model=self.model,
                training=training,
                save_path=os.path.join(self.save_root, "synapse_turnover_analysis"),
                filename=filename,
            )

        if self.noise_enabled and ((self.noise_training and training) or not training):
            self.noise_perturbation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "noise_perturbation_analysis"),
                filename=filename,
            )

        if self.single_layer_enabled and (
            (self.single_layer_training and training) or not training
        ):
            self.single_layer_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                save_path=os.path.join(self.save_root, "single_layer_contribution"),
                filename=filename,
            )

        if self.synaptic_ablation_enabled and (
            (self.synaptic_ablation_training and training) or not training
        ):
            self.synaptic_ablation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                save_path=os.path.join(self.save_root, "synaptic_ablation"),
                filename=filename,
            )

        if self.branch_info_enabled and (
            (self.branch_info_training and training) or not training
        ):
            test_loader = DataLoader(
                self.data["test"],
                batch_size=self.train_config.get("batch_size", 32),
                shuffle=False,
                num_workers=0,
            )

            # Extract epoch number from filename if possible
            try:
                epoch = int(filename.split("_")[-1]) if "_" in filename else 0
            except:
                epoch = 0

            self.branch_info_analyzer.analyze(
                model=self.model,
                test_loader=test_loader,
                output_dir=os.path.join(self.save_root, "branch_information"),
                epoch=epoch,
            )

        if self.ablation_enabled and (
            (self.ablation_training and training) or not training
        ):
            self.ablation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "ablation_analysis"),
                filename=filename,
            )

        if self.weight_enabled and (
            (self.weight_training and training) or not training
        ):
            self.weight_analyzer.analyze(
                model=self.model,
                save_path=os.path.join(self.save_root, "weight_analysis"),
                filename=filename,
            )

        if self.synaptic_activation_enabled and (
            (self.synaptic_activation_training and training) or not training
        ):
            self.synaptic_activation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device="cpu",
                save_path=os.path.join(self.save_root, "synaptic_activation_analysis"),
                filename=filename,
            )

        if self.branch_activation_enabled and (
            (self.branch_activation_training and training) or not training
        ):
            self.branch_activation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device="cpu",
                save_path=os.path.join(self.save_root, "branch_activation_analysis"),
                filename=filename,
            )

        # TODO: add other analysis classes here implemented same as above
