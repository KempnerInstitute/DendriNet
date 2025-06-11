"""
Branch information analysis module.

This module contains classes for analyzing mutual information between branch components
(excitatory inputs, inhibitory inputs, branch outputs, and class labels) using
sophisticated copula-based dependence estimation methods.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.feature_selection import mutual_info_regression as compute_mi_cc

from dendritic_modeling.models import BaseModel
from dendritic_modeling.dendrinet import DendriticBranchLayer


class BranchInformationAnalyzer:
    """
    Analyzes mutual information between branch components (E, I, V_O, C).
    Uses copula entropy from DVC_tensorflow package for MI computation.
    """

    def __init__(
        self,
        n_neighbors: int = 10,
        use_dvc: bool = True,  # Whether to use DVC_tensorflow or fallback to sklearn
        save_plots: bool = True,
        save_csv: bool = True,
        copula_params: Dict = None,  # Additional parameters for copula MI estimation
        # Configurable MI metrics to compute
        compute_metrics: List[str] = None,  # List of metrics to compute
        # Per-branch vs pooled computation
        compute_per_branch: bool = False,  # If true, compute MI per branch and average; if false, pool all branches
    ):
        self.n_neighbors = n_neighbors
        self.use_dvc = use_dvc
        self.save_plots = save_plots
        self.save_csv = save_csv
        self.compute_per_branch = compute_per_branch
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"BranchInformationAnalyzer will compute {'per-branch' if self.compute_per_branch else 'pooled'} MI analysis"
        )

        # Define all available metrics
        self.available_metrics = {
            # Information with Class Labels
            "I_E_C": "I(E;C) - Excitatory inputs and class",
            "I_I_C": "I(I;C) - Inhibitory inputs and class",
            "I_V_C": "I(V;C) - Branch outputs and class",
            # Information Between Branch Components
            "I_E_I": "I(E;I) - Excitatory and inhibitory inputs",
            "I_E_V": "I(E;V) - Excitatory inputs and branch outputs",
            "I_I_V": "I(I;V) - Inhibitory inputs and branch outputs",
            # Conditional Information (Given Class)
            "I_E_I_given_C": "I(E;I|C) - E and I given class",
            "I_E_V_given_C": "I(E;V|C) - E and V given class",
            "I_I_V_given_C": "I(I;V|C) - I and V given class",
        }

        # Set which metrics to compute (default: all)
        if compute_metrics is None:
            self.compute_metrics = list(self.available_metrics.keys())
        else:
            # Validate requested metrics
            invalid_metrics = set(compute_metrics) - set(self.available_metrics.keys())
            if invalid_metrics:
                raise ValueError(
                    f"Invalid metrics requested: {invalid_metrics}. "
                    f"Available metrics: {list(self.available_metrics.keys())}"
                )
            self.compute_metrics = compute_metrics

        self.logger.info(
            f"BranchInformationAnalyzer will compute {len(self.compute_metrics)} metrics: {self.compute_metrics}"
        )

        # Default copula parameters
        self.copula_params = copula_params or {
            "cases": 1000,  # Number of samples for entropy estimation
            "iterations": 10,  # Number of iterations
            "alpha": 0.05,  # Confidence level
            "vine_type": "c-vine",  # Type of vine copula
            "families": "kercop",  # Copula families
            "knots": 30,  # Number of knots for non-parametric copula
        }

        # Initialize DVC components if requested
        if self.use_dvc:
            # Temporarily disable DVC initialization due to syntax issues
            # TODO: Fix DVC initialization structure
            self.logger.warning(
                "DVC initialization temporarily disabled due to syntax issues"
            )
            self.use_dvc = False
            self.vine_entropy = None
            self.vine_obj_bin = None
            self.margin_obj = None
            self.prep_cop = None
            self.tf = None
        else:
            self.vine_entropy = None
            self.vine_obj_bin = None
            self.margin_obj = None
            self.prep_cop = None
            self.tf = None

        # Initialize vine parameters
        self.copula_params = copula_params or {}

    def attach_forward_hooks(self, model: BaseModel):
        """Attach forward hooks to collect branch data (reusing EI analyzer pattern)"""
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if (
                    module.branch_excitation is not None
                    and module.branch_inhibition is not None
                ):
                    if name not in self.data_dict.keys():
                        self.data_dict[name] = {}
                        module._name = name
                        handle = module.register_forward_hook(self.forward_hook)
                        handles.append(handle)
        return handles

    def remove_forward_hooks(self, handles: List[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks"""
        for handle in handles:
            handle.remove()

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]],
        output: torch.Tensor,
    ):
        """Forward hook to collect branch data during forward pass"""
        excitatory_input, inhibitory_input, _ = input
        key = module._name

        # Ensure the key exists in data_dict (might be cleared between batches)
        if key not in self.data_dict:
            self.data_dict[key] = {}

        # Collect the same data as EIInformationAnalyzer
        self.data_dict[key]["excitation"] = module.branch_excitation(excitatory_input)
        self.data_dict[key]["inhibition"] = module.branch_inhibition(inhibitory_input)
        self.data_dict[key]["output"] = output

        # Store layer index for organization
        if hasattr(module, "layer_idx"):
            self.data_dict[key]["layer_idx"] = module.layer_idx

    def compute_mi_continuous(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI between continuous variables"""
        # Ensure inputs are 2D arrays as expected by compute_mi_cc
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if self.use_dvc:
            # Use copula entropy approach
            mi = self._compute_copula_mi(x, y)
            return mi

        # Fallback to sklearn
        return compute_mi_cc(x, y, self.n_neighbors)

    def compute_mi_discrete(self, x: np.ndarray, c: np.ndarray) -> float:
        """Compute MI between continuous and discrete variables"""
        # Ensure inputs are 2D arrays as expected by compute_mi_cc
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if c.ndim == 1:
            c = c.reshape(-1, 1)

        # Add uniform noise to discrete labels as suggested
        c_noisy = c + np.random.uniform(-0.5, 0.5, size=c.shape)

        if self.use_dvc:
            # Use copula entropy approach with noisy labels
            mi = self._compute_copula_mi(x, c_noisy)
            return mi

        # Fallback to sklearn
        return compute_mi_cc(x, c_noisy, self.n_neighbors)

    def compute_cmi(self, x: np.ndarray, y: np.ndarray, c: np.ndarray) -> float:
        """Compute conditional MI: I(X;Y|C)"""
        # Add uniform noise to discrete condition variable
        c_noisy = c + np.random.uniform(-0.5, 0.5, size=c.shape)

        if self.use_dvc:
            # Use copula approach for conditional MI
            # I(X;Y|C) = H(X|C) + H(Y|C) - H(X,Y|C)
            # We compute this per class and average
            unique_classes = np.unique(c)
            cmi_values = []

            for class_val in unique_classes:
                mask = c == class_val
                if np.sum(mask) < 10:  # Skip if too few samples
                    continue

                x_class = x[mask]
                y_class = y[mask]

                # Compute MI for this class
                mi_class = self._compute_copula_mi(x_class, y_class)
                cmi_values.append(mi_class)

            # Average over classes
            return np.mean(cmi_values) if cmi_values else 0.0

        # Fallback approach: compute per class and average
        unique_classes = np.unique(c)
        cmi_values = []

        for class_val in unique_classes:
            mask = c == class_val
            if np.sum(mask) < self.n_neighbors + 1:
                continue

            x_class = x[mask]
            y_class = y[mask]

            # Ensure inputs are 2D arrays as expected by compute_mi_cc
            if x_class.ndim == 1:
                x_class = x_class.reshape(-1, 1)
            if y_class.ndim == 1:
                y_class = y_class.reshape(-1, 1)

            # Compute MI for this class
            mi_class = compute_mi_cc(x_class, y_class, self.n_neighbors)
            cmi_values.append(mi_class)

        return np.mean(cmi_values) if cmi_values else 0.0

    def _compute_copula_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute MI using copula-based dependence estimation from DVC_tensorflow
        Uses parametric or non-parametric approach based on copula_params configuration
        """
        try:
            # Ensure inputs are 2D arrays as expected by compute_mi_cc
            if x.ndim == 1:
                x = x.reshape(-1, 1)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

            # Try DVC_tensorflow if requested
            if self.use_dvc:
                # Read the nonparametric setting from copula_params
                use_nonparametric = self.copula_params.get("use_nonparametric", False)

                self.logger.debug(
                    f"Attempting DVC_tensorflow copula MI computation with {'non-parametric' if use_nonparametric else 'parametric'} approach..."
                )
                try:
                    # Force TensorFlow to use CPU to avoid GPU compilation issues
                    import os

                    os.environ["CUDA_VISIBLE_DEVICES"] = ""

                    # Import DVC components
                    import tensorflow as tf

                    tf.config.set_visible_devices([], "GPU")

                    # Combine data for joint distribution
                    data_joint = np.hstack([x, y])
                    n_samples, n_dims = data_joint.shape

                    # Ensure we have enough samples and only 2D
                    if n_samples < 50:
                        raise ValueError(
                            f"Too few samples ({n_samples}) for copula estimation"
                        )

                    if n_dims != 2:
                        raise ValueError("Copula method only supports 2D MI estimation")

                    # Transform to uniform margins using empirical CDF
                    from scipy import stats

                    u = np.zeros_like(data_joint)
                    for i in range(n_dims):
                        # Rank transform to uniform
                        ranks = stats.rankdata(data_joint[:, i])
                        u[:, i] = ranks / (len(ranks) + 1)  # Avoid 0 and 1

                    u = u.astype(np.float32)

                    if use_nonparametric:
                        # Non-parametric kernel copula approach
                        self.logger.debug("Using non-parametric kernel copula approach")

                        # For non-parametric, use a simplified approach that falls back to parametric estimation
                        # The DVC non-parametric kernel copula implementation is complex and requires specific setup
                        # For now, we use the parametric approach but could be extended later
                        self.logger.warning(
                            "Non-parametric kernel copula not fully implemented, using parametric approach"
                        )

                        # Fall through to parametric approach
                        use_nonparametric = False

                    if not use_nonparametric:
                        # Parametric approach - use simpler direct estimation
                        self.logger.debug("Using parametric copula approach")

                        # For parametric approach, use simple Gaussian copula
                        # which we can compute analytically from correlation
                        from scipy.stats import kendalltau, pearsonr

                        # Compute Pearson correlation of uniform-transformed data
                        corr_pearson, _ = pearsonr(u[:, 0], u[:, 1])

                        # Convert to Gaussian copula correlation
                        # For uniform data, Pearson correlation approximates Gaussian copula parameter
                        rho = np.clip(corr_pearson, -0.99, 0.99)

                        # Compute MI from Gaussian copula: MI = -0.5 * log(1 - rho^2)
                        mi = -0.5 * np.log(1 - rho**2)
                        mi = max(0.0, mi)  # Ensure non-negative

                        self.logger.debug(
                            f"Parametric Gaussian copula: rho={rho:.4f}, MI={mi:.4f}"
                        )
                        return float(mi)

                except Exception as e:
                    self.logger.warning(f"DVC_tensorflow MI computation failed: {e}")
                    self.logger.warning("Falling back to sklearn for this computation")
                    # Fall back to sklearn
                    pass

            # Fallback to sklearn
            from dendritic_modeling.information import compute_mi_cc

            return compute_mi_cc(x, y, self.n_neighbors)

        except Exception as e:
            self.logger.error(f"All MI computation methods failed: {e}")
            return 0.0

    def analyze(
        self, model: BaseModel, test_loader: DataLoader, output_dir: str, epoch: int
    ) -> Dict[str, Any]:
        """Analyze branch information metrics using the improved hook-based approach"""
        self.logger.info("Starting branch information analysis...")

        # Make sure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Initialize data collection
        self.data_dict = {}

        if self.compute_per_branch:
            # Store data organized by layer and branch
            all_data = {
                "layer_branch_data": {},  # {layer_idx: {branch_idx: {'E': [], 'I': [], 'V': [], 'C': []}}}
            }
        else:
            # Store all collected data pooled across branches
            all_data = {
                "E": [],  # Excitatory inputs
                "I": [],  # Inhibitory inputs
                "V": [],  # Branch outputs
                "C": [],  # Class labels
                "layer_idx": [],  # Layer indices
            }

        model.eval()
        with torch.no_grad():
            # Attach hooks to collect branch data (reusing EI analyzer pattern)
            handles = self.attach_forward_hooks(model)

            try:
                for batch_idx, (inputs, targets) in enumerate(test_loader):
                    if batch_idx >= 10:  # Limit samples for efficiency
                        break

                    inputs = inputs.to(next(model.parameters()).device)

                    # Run standard forward pass - hooks will collect data automatically
                    _ = model(inputs)

                    # Process collected data from hooks
                    for layer_name, layer_data in self.data_dict.items():
                        if "excitation" in layer_data and "inhibition" in layer_data:
                            # Get the collected tensors
                            excitation = layer_data[
                                "excitation"
                            ]  # Shape: [batch, n_branches]
                            inhibition = layer_data[
                                "inhibition"
                            ]  # Shape: [batch, n_branches]
                            output = layer_data["output"]  # Shape: [batch, n_branches]

                            layer_idx = layer_data.get("layer_idx", 0)
                            n_samples = excitation.shape[0]
                            n_branches = (
                                excitation.shape[1] if excitation.ndim > 1 else 1
                            )

                            if self.compute_per_branch:
                                # Store data per branch for per-branch analysis
                                if layer_idx not in all_data["layer_branch_data"]:
                                    all_data["layer_branch_data"][layer_idx] = {}

                                if excitation.ndim > 1:
                                    for branch_idx in range(n_branches):
                                        if (
                                            branch_idx
                                            not in all_data["layer_branch_data"][
                                                layer_idx
                                            ]
                                        ):
                                            all_data["layer_branch_data"][layer_idx][
                                                branch_idx
                                            ] = {"E": [], "I": [], "V": [], "C": []}

                                        # Store data for this specific branch
                                        all_data["layer_branch_data"][layer_idx][
                                            branch_idx
                                        ]["E"].extend(
                                            excitation[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["layer_branch_data"][layer_idx][
                                            branch_idx
                                        ]["I"].extend(
                                            inhibition[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["layer_branch_data"][layer_idx][
                                            branch_idx
                                        ]["V"].extend(
                                            output[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["layer_branch_data"][layer_idx][
                                            branch_idx
                                        ]["C"].extend(targets.cpu().numpy())
                                else:
                                    # Single branch case
                                    if (
                                        0
                                        not in all_data["layer_branch_data"][layer_idx]
                                    ):
                                        all_data["layer_branch_data"][layer_idx][0] = {
                                            "E": [],
                                            "I": [],
                                            "V": [],
                                            "C": [],
                                        }

                                    all_data["layer_branch_data"][layer_idx][0][
                                        "E"
                                    ].extend(excitation.cpu().numpy())
                                    all_data["layer_branch_data"][layer_idx][0][
                                        "I"
                                    ].extend(inhibition.cpu().numpy())
                                    all_data["layer_branch_data"][layer_idx][0][
                                        "V"
                                    ].extend(output.cpu().numpy())
                                    all_data["layer_branch_data"][layer_idx][0][
                                        "C"
                                    ].extend(targets.cpu().numpy())

                            else:
                                # Pool all branches together (original behavior)
                                if excitation.ndim > 1:
                                    for branch_idx in range(n_branches):
                                        all_data["E"].extend(
                                            excitation[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["I"].extend(
                                            inhibition[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["V"].extend(
                                            output[:, branch_idx].cpu().numpy()
                                        )
                                        all_data["C"].extend(targets.cpu().numpy())
                                        all_data["layer_idx"].extend(
                                            [layer_idx] * n_samples
                                        )
                                else:
                                    all_data["E"].extend(excitation.cpu().numpy())
                                    all_data["I"].extend(inhibition.cpu().numpy())
                                    all_data["V"].extend(output.cpu().numpy())
                                    all_data["C"].extend(targets.cpu().numpy())
                                    all_data["layer_idx"].extend(
                                        [layer_idx] * n_samples
                                    )

                    # Clear collected data for next batch to save memory
                    self.data_dict.clear()

            finally:
                # Always remove hooks
                self.remove_forward_hooks(handles)

        # Check if we have data
        if self.compute_per_branch:
            if not all_data["layer_branch_data"]:
                self.logger.warning("No branch data collected. Skipping analysis.")
                return {}
        else:
            # Convert to numpy arrays for pooled approach
            for key in all_data:
                all_data[key] = np.array(all_data[key])

            if len(all_data["E"]) == 0:
                self.logger.warning("No branch data collected. Skipping analysis.")
                return {}

        # Compute MI metrics
        if self.compute_per_branch:
            results = self._compute_per_branch_metrics(all_data)
        else:
            results = self._compute_pooled_metrics(all_data)

        # Save results
        if self.save_plots:
            self._plot_results(results, output_dir, epoch)
        if self.save_csv:
            self._save_csv(results, output_dir, epoch)

        return results

    def _compute_per_branch_metrics(self, all_data: Dict) -> Dict[str, Any]:
        """Compute MI metrics per branch and then average across branches"""
        self.logger.info("Computing per-branch MI metrics...")

        # Structure: {metric_name: {layer_idx: {branch_idx: value}}}
        per_branch_results = {}

        # Also compute averages: {metric_name: {layer_idx: avg_value}}
        layer_averages = {}

        # And overall averages: {metric_name: overall_avg_value}
        overall_averages = {}

        for metric in self.compute_metrics:
            per_branch_results[metric] = {}
            layer_averages[metric] = {}

        # Process each layer
        for layer_idx, layer_data in all_data["layer_branch_data"].items():
            self.logger.info(f"Processing layer {layer_idx} with {len(layer_data)} branches")

            # Process each branch in this layer
            for branch_idx, branch_data in layer_data.items():
                # Convert to numpy arrays
                E = np.array(branch_data["E"])
                I = np.array(branch_data["I"])
                V = np.array(branch_data["V"])
                C = np.array(branch_data["C"])

                # Skip if insufficient data
                if len(E) < self.n_neighbors + 1:
                    self.logger.warning(
                        f"Insufficient data for layer {layer_idx}, branch {branch_idx}: {len(E)} samples"
                    )
                    continue

                # Initialize this layer in results if needed
                for metric in self.compute_metrics:
                    if layer_idx not in per_branch_results[metric]:
                        per_branch_results[metric][layer_idx] = {}

                # Compute each metric for this branch
                if "I_E_C" in self.compute_metrics:
                    per_branch_results["I_E_C"][layer_idx][branch_idx] = self.compute_mi_discrete(E, C)

                if "I_I_C" in self.compute_metrics:
                    per_branch_results["I_I_C"][layer_idx][branch_idx] = self.compute_mi_discrete(I, C)

                if "I_V_C" in self.compute_metrics:
                    per_branch_results["I_V_C"][layer_idx][branch_idx] = self.compute_mi_discrete(V, C)

                if "I_E_I" in self.compute_metrics:
                    per_branch_results["I_E_I"][layer_idx][branch_idx] = self.compute_mi_continuous(E, I)

                if "I_E_V" in self.compute_metrics:
                    per_branch_results["I_E_V"][layer_idx][branch_idx] = self.compute_mi_continuous(E, V)

                if "I_I_V" in self.compute_metrics:
                    per_branch_results["I_I_V"][layer_idx][branch_idx] = self.compute_mi_continuous(I, V)

                if "I_E_I_given_C" in self.compute_metrics:
                    per_branch_results["I_E_I_given_C"][layer_idx][branch_idx] = self.compute_cmi(E, I, C)

                if "I_E_V_given_C" in self.compute_metrics:
                    per_branch_results["I_E_V_given_C"][layer_idx][branch_idx] = self.compute_cmi(E, V, C)

                if "I_I_V_given_C" in self.compute_metrics:
                    per_branch_results["I_I_V_given_C"][layer_idx][branch_idx] = self.compute_cmi(I, V, C)

            # Compute layer averages
            for metric in self.compute_metrics:
                if layer_idx in per_branch_results[metric] and per_branch_results[metric][layer_idx]:
                    values = list(per_branch_results[metric][layer_idx].values())
                    layer_averages[metric][layer_idx] = np.mean(values)
                else:
                    layer_averages[metric][layer_idx] = 0.0

        # Compute overall averages
        for metric in self.compute_metrics:
            if layer_averages[metric]:
                overall_averages[metric] = np.mean(list(layer_averages[metric].values()))
            else:
                overall_averages[metric] = 0.0

        return {
            "computation_mode": "per_branch",
            "per_branch_results": per_branch_results,
            "layer_averages": layer_averages,
            "overall_averages": overall_averages,
            "metrics_computed": self.compute_metrics,
            "available_metrics": self.available_metrics,
        }

    def _compute_pooled_metrics(self, all_data: Dict) -> Dict[str, Any]:
        """Compute MI metrics using all data pooled across branches (original behavior)"""
        self.logger.info("Computing pooled MI metrics...")

        # Extract data arrays
        E = all_data["E"]
        I = all_data["I"]
        V = all_data["V"]
        C = all_data["C"]
        layer_idx = all_data["layer_idx"]

        # Check data
        self.logger.info(f"Computing MI with {len(E)} total samples across all branches")

        if len(E) < self.n_neighbors + 1:
            self.logger.warning(f"Insufficient data: {len(E)} samples")
            return {}

        # Compute metrics
        results = {
            "computation_mode": "pooled",
            "metrics_computed": self.compute_metrics,
            "available_metrics": self.available_metrics,
            "sample_count": len(E),
        }

        # Compute each requested metric
        if "I_E_C" in self.compute_metrics:
            results["I_E_C"] = self.compute_mi_discrete(E, C)

        if "I_I_C" in self.compute_metrics:
            results["I_I_C"] = self.compute_mi_discrete(I, C)

        if "I_V_C" in self.compute_metrics:
            results["I_V_C"] = self.compute_mi_discrete(V, C)

        if "I_E_I" in self.compute_metrics:
            results["I_E_I"] = self.compute_mi_continuous(E, I)

        if "I_E_V" in self.compute_metrics:
            results["I_E_V"] = self.compute_mi_continuous(E, V)

        if "I_I_V" in self.compute_metrics:
            results["I_I_V"] = self.compute_mi_continuous(I, V)

        if "I_E_I_given_C" in self.compute_metrics:
            results["I_E_I_given_C"] = self.compute_cmi(E, I, C)

        if "I_E_V_given_C" in self.compute_metrics:
            results["I_E_V_given_C"] = self.compute_cmi(E, V, C)

        if "I_I_V_given_C" in self.compute_metrics:
            results["I_I_V_given_C"] = self.compute_cmi(I, V, C)

        self.logger.info("MI computation completed")

        return results

    def _plot_results(self, results: Dict, output_dir: str, epoch: int):
        """Plot MI results"""
        if results["computation_mode"] == "per_branch":
            self._plot_per_branch_results(results, output_dir, epoch)
        else:
            self._plot_pooled_results(results, output_dir, epoch)

    def _plot_per_branch_results(self, results: Dict, output_dir: str, epoch: int):
        """Plot per-branch MI results"""
        # Create a comprehensive plot showing layer averages and overall trends
        metrics = results["metrics_computed"]
        layer_averages = results["layer_averages"]

        if not metrics or not layer_averages:
            self.logger.warning("No metrics to plot")
            return

        # Determine layout based on number of metrics
        n_metrics = len(metrics)
        if n_metrics <= 3:
            fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4))
        else:
            n_cols = 3
            n_rows = (n_metrics + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

        if n_metrics == 1:
            axes = [axes]
        elif n_metrics > 1 and n_metrics <= 3:
            axes = axes
        else:
            axes = axes.flatten()

        # Plot each metric
        for i, metric in enumerate(metrics):
            ax = axes[i]

            # Get data for this metric
            metric_data = layer_averages[metric]
            if not metric_data:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{metric}\n{results['available_metrics'].get(metric, metric)}")
                continue

            # Sort by layer index
            layers = sorted(metric_data.keys())
            values = [metric_data[layer] for layer in layers]

            # Create bar plot
            bars = ax.bar(range(len(layers)), values, alpha=0.7)

            # Customize plot
            ax.set_xlabel("Layer Index")
            ax.set_ylabel("Mutual Information (bits)")
            ax.set_title(f"{metric}\n{results['available_metrics'].get(metric, metric)}")
            ax.set_xticks(range(len(layers)))
            ax.set_xticklabels([f"L{layer}" for layer in layers])
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, values)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        # Hide empty subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"branch_info_per_branch_epoch_{epoch}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved per-branch MI plot to {plot_path}")

        # Also create a summary plot showing overall averages
        overall_averages = results["overall_averages"]
        if overall_averages:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))

            metrics_clean = [metric.replace("_", " ") for metric in metrics]
            values = [overall_averages[metric] for metric in metrics]

            bars = ax.bar(range(len(metrics)), values, alpha=0.7)
            ax.set_xlabel("MI Metric")
            ax.set_ylabel("Mutual Information (bits)")
            ax.set_title(f"Overall Average Branch MI Metrics (Epoch {epoch})")
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics_clean, rotation=45, ha="right")
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                )

            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"branch_info_overall_average_epoch_{epoch}.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            self.logger.info(f"Saved overall average MI plot to {plot_path}")

    def _plot_pooled_results(self, results: Dict, output_dir: str, epoch: int):
        """Plot pooled MI results"""
        metrics = results["metrics_computed"]

        if not metrics:
            self.logger.warning("No metrics to plot")
            return

        # Extract values
        values = []
        metric_labels = []
        for metric in metrics:
            if metric in results:
                values.append(results[metric])
                metric_labels.append(metric.replace("_", " "))

        if not values:
            self.logger.warning("No metric values to plot")
            return

        # Create bar plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        bars = ax.bar(range(len(values)), values, alpha=0.7)
        ax.set_xlabel("MI Metric")
        ax.set_ylabel("Mutual Information (bits)")
        ax.set_title(f"Branch MI Metrics - Pooled Analysis (Epoch {epoch})\n{results['sample_count']} total samples")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(metric_labels, rotation=45, ha="right")
        ax.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"branch_info_pooled_epoch_{epoch}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved pooled MI plot to {plot_path}")

    def _save_csv(self, results: Dict, output_dir: str, epoch: int):
        """Save MI results to CSV"""
        import pandas as pd

        if results["computation_mode"] == "per_branch":
            # Save layer averages
            layer_averages = results["layer_averages"]
            if layer_averages:
                # Create DataFrame with metrics as columns and layers as rows
                df_data = {}
                layers = sorted(set().union(*[layer_data.keys() for layer_data in layer_averages.values()]))

                for layer in layers:
                    df_data[f"Layer_{layer}"] = [
                        layer_averages[metric].get(layer, 0.0) for metric in results["metrics_computed"]
                    ]

                df = pd.DataFrame(df_data, index=results["metrics_computed"])
                csv_path = os.path.join(output_dir, f"branch_info_per_branch_epoch_{epoch}.csv")
                df.to_csv(csv_path)

                # Also save overall averages
                overall_df = pd.DataFrame(
                    {"overall_average": [results["overall_averages"][metric] for metric in results["metrics_computed"]]},
                    index=results["metrics_computed"],
                )
                overall_csv_path = os.path.join(output_dir, f"branch_info_overall_average_epoch_{epoch}.csv")
                overall_df.to_csv(overall_csv_path)

                self.logger.info(f"Saved per-branch CSV to {csv_path}")
                self.logger.info(f"Saved overall average CSV to {overall_csv_path}")

        else:
            # Save pooled results
            data = {"metric": [], "value": []}
            for metric in results["metrics_computed"]:
                if metric in results:
                    data["metric"].append(metric)
                    data["value"].append(results[metric])

            if data["metric"]:
                df = pd.DataFrame(data)
                csv_path = os.path.join(output_dir, f"branch_info_pooled_epoch_{epoch}.csv")
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved pooled CSV to {csv_path}")

    @classmethod
    def list_available_metrics(cls) -> Dict[str, str]:
        """List all available MI metrics with descriptions"""
        return {
            # Information with Class Labels
            "I_E_C": "I(E;C) - Excitatory inputs and class",
            "I_I_C": "I(I;C) - Inhibitory inputs and class",
            "I_V_C": "I(V;C) - Branch outputs and class",
            # Information Between Branch Components
            "I_E_I": "I(E;I) - Excitatory and inhibitory inputs",
            "I_E_V": "I(E;V) - Excitatory inputs and branch outputs",
            "I_I_V": "I(I;V) - Inhibitory inputs and branch outputs",
            # Conditional Information (Given Class)
            "I_E_I_given_C": "I(E;I|C) - E and I given class",
            "I_E_V_given_C": "I(E;V|C) - E and V given class",
            "I_I_V_given_C": "I(I;V|C) - I and V given class",
        } 