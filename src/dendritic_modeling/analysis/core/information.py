"""Integrated information analyzer for dendritic networks.

This module provides a unified interface for computing various information-theoretic
metrics using the multivariate methods from information.py. It integrates seamlessly
with the main configuration system.
"""

import logging
import os
import traceback
from typing import Any, Optional, Union

import numpy as np
import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config import InformationAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.information.estimators import (
    EstimatorConfig,
    InformationEstimatorFactory,
)
from dendritic_modeling.utils.information.pid import (
    check_pid_available,
    pid_synergy_redundancy,
)


class InformationAnalyzer(AbstractAnalyzer):
    """Unified analyzer for computing information-theoretic metrics in dendritic networks.

    This analyzer uses the multivariate-aware methods from information.py to compute
    mutual information between excitatory inputs, inhibitory inputs, branch outputs,
    and class labels. It supports continuous multivariate data natively.
    """

    def __init__(self, params: InformationAnalysisParams):
        """Initialize the analyzer with configuration from main YAML config.

        Parameters
        ----------
        config : InformationAnalysisConfig or dict
            Configuration from the main config system (analysis.information_analysis)
        """
        super().__init__("InformationAnalyzer")

        # Extract configuration parameters
        self.method = getattr(params, "method", "kraskov")
        self.max_samples = getattr(params, "max_samples", None)
        self.n_neighbors = getattr(params, "n_neighbors", 10)
        self.n_bins = getattr(params, "n_bins", 20)
        self.copula_type = getattr(params, "copula_type", "gaussian")

        self.compute_basic_mi = getattr(params, "compute_basic_mi", False)
        self.compute_pairwise_mi = getattr(params, "compute_pairwise_mi", False)
        self.compute_conditional_mi = getattr(params, "compute_conditional_mi", False)
        self.compute_pid = getattr(params, "compute_pid", False)

        self.pid_binarize = getattr(params, "pid_binarize", False)
        self.pid_binarize_method = getattr(params, "pid_binarize_method", "median")
        self.pid_binarize_threshold = getattr(params, "pid_binarize_threshold", 0.5)

        self.include_vinf = getattr(
            params, "include_vinf", False
        )  # Default to False to match 245MR behavior
        self.normalize_mi = getattr(params, "normalize_mi", False)
        self.verbose = getattr(params, "verbose", False)

        # Computation level control - determines the granularity of information analysis
        # "branch": Individual branches (most granular and accurate)
        # "layer": Layer-level aggregation (treats each layer as a unit)
        # "network": Full network aggregation (most efficient)
        self.computation_level = getattr(params, "computation_level", "single_branch")

        # Branch aggregation method - only used when computation_level is "layer_branch" or "all_branch"
        # Options: "mean", "multivariate", "sample"
        self.branch_aggregation = getattr(params, "branch_aggregation", "mean")

        # Per-neuron analysis - return information values per individual neuron
        self.per_neuron_analysis = getattr(params, "per_neuron_analysis", False)

        # Per-EI-network analysis - analyze excitatory and inhibitory networks separately
        self.per_einet_analysis = getattr(params, "per_einet_analysis", False)

        # Per-layer analysis - compute enhanced analysis for each layer separately
        self.per_layer_analysis = getattr(params, "per_layer_analysis", False)

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        # Create estimators for each method
        self.estimators = {}
        # For now, just use the single method specified, but keep the dict structure for compatibility
        estimator_config = EstimatorConfig(
            method=self.method,
            n_neighbors=self.n_neighbors,
            n_bins=self.n_bins,
            copula_type=self.copula_type,
            verbose=self.verbose,
        )
        self.estimators[self.method] = InformationEstimatorFactory.create(estimator_config)

        # Check if PID is available
        self.pid_available = False
        if self.compute_pid:
            try:
                check_pid_available()
                self.pid_available = True
                self.logger.info("PID computation available")
            except ImportError:
                self.logger.warning(
                    "PID package not available. PID metrics will be skipped."
                )

    def _compute_basic_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute mutual information between each variable and C."""
        metrics = {}
        # I(E;C)
        metrics["I(E;C)"] = self.estimator.mutual_information(E, C)
        # I(I;C)
        metrics["I(I;C)"] = self.estimator.mutual_information(I_var, C)
        # I(Vout;C)
        metrics["I(Vout;C)"] = self.estimator.mutual_information(Vout, C)
        # I(Vinf;C) if available
        if Vinf is not None and self.include_vinf:
            metrics["I(Vinf;C)"] = self.estimator.mutual_information(Vinf, C)
        # Joint MI: I(E,I;C)
        EI = np.hstack([E, I_var])
        metrics["I(E,I;C)"] = self.estimator.mutual_information(EI, C)
        # Optionally normalize by entropy
        if self.normalize_mi:
            H_C = self.estimator.entropy(C.reshape(-1, 1))
            # Create a list of keys to avoid modifying dict during iteration
            keys_to_normalize = list(metrics.keys())
            for key in keys_to_normalize:
                metrics[f"{key}_normalized"] = metrics[key] / H_C if H_C > 0 else 0.0
        return metrics

    def _compute_pairwise_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute pairwise mutual information between variables."""
        metrics = {}
        # I(E;I)
        metrics["I(E;I)"] = self.estimator.mutual_information(E, I_var)
        # I(E;Vout)
        metrics["I(E;Vout)"] = self.estimator.mutual_information(E, Vout)
        # I(I;Vout)
        metrics["I(I;Vout)"] = self.estimator.mutual_information(I_var, Vout)
        if Vinf is not None and self.include_vinf:
            # I(E;Vinf)
            metrics["I(E;Vinf)"] = self.estimator.mutual_information(E, Vinf)
            # I(I;Vinf)
            metrics["I(I;Vinf)"] = self.estimator.mutual_information(I_var, Vinf)
            # I(Vout;Vinf)
            metrics["I(Vout;Vinf)"] = self.estimator.mutual_information(Vout, Vinf)
        return metrics

    def _compute_conditional_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute conditional mutual information given C."""
        metrics = {}
        # I(E;I|C)
        metrics["I(E;I|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(E, I_var, C.reshape(-1, 1))
        # I(E;Vout|C)
        metrics["I(E;Vout|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(E, Vout, C.reshape(-1, 1))
        # I(I;Vout|C)
        metrics["I(I;Vout|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(I_var, Vout, C.reshape(-1, 1))
        if Vinf is not None and self.include_vinf:
            # I(E;Vinf|C)
            metrics["I(E;Vinf|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(E, Vinf, C.reshape(-1, 1))
            # I(I;Vinf|C)
            metrics["I(I;Vinf|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(I_var, Vinf, C.reshape(-1, 1))
        return metrics

    def _compute_pid(
        self, E: np.ndarray, I_var: np.ndarray, Vout: np.ndarray, C: np.ndarray
    ) -> dict[str, Any]:
        """Compute Partial Information Decomposition."""
        pid_results = {}

        # Prepare data for PID (may need binarization)
        if self.pid_binarize:
            E_pid = self._binarize(E)
            I_pid = self._binarize(I_var)
            Vout_pid = self._binarize(Vout)
            C_pid = self._binarize(C.reshape(-1, 1))
        else:
            E_pid = E
            I_pid = I_var
            Vout_pid = Vout
            C_pid = C.reshape(-1, 1)

        def _pid_helper(
            source1_array: np.ndarray,
            source1_name: str,
            source2_array: np.ndarray,
            source2_name: str,
            target_array: np.ndarray,
            target_name: str,
        ) -> None:
            """Helper function to compute PID."""
            pid_key = f"PID({source1_name},{source2_name}->{target_name})"
            try:
                synergy, redundancy = pid_synergy_redundancy(
                    [source1_array, source2_array], target_array
                )
                pid_results[pid_key] = {
                    "synergy": synergy,
                    "redundancy": redundancy,
                    f"unique_{source1_name}": self.estimators[
                        self.method
                    ].mutual_information(source1_array, target_array)
                    - redundancy,
                    f"unique_{source2_name}": self.estimators[
                        self.method
                    ].mutual_information(source2_array, target_array)
                    - redundancy,
                }
            except Exception as e:
                self.logger.warning(f"{pid_key} computation failed: {e}")

        # PID: (E, I) -> C
        _pid_helper(E_pid, "E", I_pid, "I", C_pid, "C")
        # PID: (E, Vout) -> C
        _pid_helper(E_pid, "E", Vout_pid, "Vout", C_pid, "C")
        # PID: (I, Vout) -> C
        _pid_helper(I_pid, "I", Vout_pid, "Vout", C_pid, "C")
        # PID: (E, I) -> Vout
        _pid_helper(E_pid, "E", I_pid, "I", Vout_pid, "Vout")
        return pid_results

    def _binarize(self, data: np.ndarray) -> np.ndarray:
        """Binarize data for PID computation."""
        if self.pid_binarize_method == "median":
            threshold = np.median(data, axis=0)
        elif self.pid_binarize_method == "mean":
            threshold = np.mean(data, axis=0)
        elif self.pid_binarize_method == "quantile":
            threshold = np.quantile(data, self.pid_binarize_threshold, axis=0)
        else:
            raise ValueError(f"Unknown binarization method: {self.pid_binarize_method}")

        binarized: np.ndarray = data > threshold
        return binarized.astype(int)

    def _validate_inputs(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
    ):
        """Validate input arrays."""
        n_samples = len(E)

        if len(I_var) != n_samples:
            raise ValueError(
                f"I must have same number of samples as E: {len(I_var)} vs {n_samples}"
            )
        if len(Vout) != n_samples:
            raise ValueError(
                f"Vout must have same number of samples as E: {len(Vout)} vs {n_samples}"
            )
        if len(C) != n_samples:
            raise ValueError(
                f"C must have same number of samples as E: {len(C)} vs {n_samples}"
            )
        if Vinf is not None and len(Vinf) != n_samples:
            raise ValueError(
                f"Vinf must have same number of samples as E: {len(Vinf)} vs {n_samples}"
            )

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D."""
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def get_summary(self, results: dict[str, dict[str, Union[dict, float]]]) -> str:
        """Generate a summary string of the results."""
        summary_lines = [
            "Information Analysis Summary",
            f"Method: {results['method']}",
            f"Samples: {results['n_samples']}",
            "",
        ]

        if "basic_mi" in results:
            summary_lines.append("Basic MI with C:")
            for key, value in results["basic_mi"].items():
                if not key.endswith("_normalized"):
                    summary_lines.append(f"  {key}: {value:.4f}")

        if "pairwise_mi" in results:
            summary_lines.append("\nPairwise MI:")
            for key, value in results["pairwise_mi"].items():
                summary_lines.append(f"  {key}: {value:.4f}")

        if "conditional_mi" in results:
            summary_lines.append("\nConditional MI given C:")
            for key, value in results["conditional_mi"].items():
                summary_lines.append(f"  {key}: {value:.4f}")

        if "pid" in results:
            summary_lines.append("\nPID Results:")
            for pid_key, pid_values in results["pid"].items():
                summary_lines.append(f"  {pid_key}:")
                for metric, value in pid_values.items():
                    summary_lines.append(f"    {metric}: {value:.4f}")

        return "\n".join(summary_lines)

    def compute_general_class_mi(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        dims: Optional[list[int]] = None,
    ) -> dict:
        """Compute MI between activations (optionally specific dims) and class labels using all methods."""
        results = {}
        acts = activations if dims is None else activations[:, dims]
        for method, estimator in self.estimators.items():
            mi = estimator.mutual_information(acts, labels.reshape(-1, 1))
            results[f"{method}_mi_class"] = mi
            if self.normalize:
                h_labels = estimator.entropy(labels.reshape(-1, 1))
                results[f"{method}_normalized_mi_class"] = (
                    mi / h_labels if h_labels > 0 else 0
                )
        return results

    def compute_metrics(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Compute all requested information metrics.

        Parameters
        ----------
        E : np.ndarray
            Excitatory inputs, shape (n_samples,) or (n_samples, n_features)
        I : np.ndarray
            Inhibitory inputs, shape (n_samples,) or (n_samples, n_features)
        Vout : np.ndarray
            Branch outputs, shape (n_samples,) or (n_samples, n_features)
        C : np.ndarray
            Class labels, shape (n_samples,)
        Vinf : np.ndarray, optional
            Soma activation (Vinf), shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        dict[str, Any]
            Dictionary containing all computed metrics
        """
        # Validate inputs
        self._validate_inputs(E, I_var, Vout, C, Vinf)

        # Ensure all arrays are properly shaped
        E = self._ensure_2d(E)
        I_var = self._ensure_2d(I_var)
        Vout = self._ensure_2d(Vout)
        C = C.ravel()  # Labels are always 1D

        if Vinf is not None:
            Vinf = self._ensure_2d(Vinf)

        results = {
            "method": self.method,
            "n_samples": len(E),
        }

        # 1. Basic mutual information with C
        if self.compute_basic_mi:
            self.logger.info("Computing basic MI metrics...")
            results["basic_mi"] = self._compute_basic_mi(E, I_var, Vout, C, Vinf)

        # 2. Pairwise mutual information
        if self.compute_pairwise_mi:
            self.logger.info("Computing pairwise MI metrics...")
            results["pairwise_mi"] = self._compute_pairwise_mi(E, I_var, Vout, Vinf)

        # 3. Conditional mutual information
        if self.compute_conditional_mi:
            self.logger.info("Computing conditional MI metrics...")
            results["conditional_mi"] = self._compute_conditional_mi(
                E, I_var, Vout, C, Vinf
            )

        # 4. PID decomposition
        if self.compute_pid and self.pid_available:
            self.logger.info("Computing PID metrics...")
            results["pid"] = self._compute_pid(E, I_var, Vout, C)

        return results

    def _average_mi_results(
        self, results_list: list[dict[str, dict[str, Union[dict, float]]]]
    ):
        """Average MI results across multiple branches."""
        averaged_results = {
            "method": results_list[0]["method"],
            "n_samples": results_list[0]["n_samples"],
        }

        # Average basic MI
        if "basic_mi" in results_list[0]:
            averaged_basic_mi = {}
            for key in results_list[0]["basic_mi"].keys():
                values = [r["basic_mi"][key] for r in results_list]
                averaged_basic_mi[key] = np.mean(values)
                averaged_basic_mi[f"{key}_std"] = np.std(values)
            averaged_results["basic_mi"] = averaged_basic_mi

        # Average pairwise MI
        if "pairwise_mi" in results_list[0]:
            averaged_pairwise_mi = {}
            for key in results_list[0]["pairwise_mi"].keys():
                values = [r["pairwise_mi"][key] for r in results_list]
                averaged_pairwise_mi[key] = np.mean(values)
                averaged_pairwise_mi[f"{key}_std"] = np.std(values)
            averaged_results["pairwise_mi"] = averaged_pairwise_mi

        # Average conditional MI
        if "conditional_mi" in results_list[0]:
            averaged_conditional_mi = {}
            for key in results_list[0]["conditional_mi"].keys():
                values = [r["conditional_mi"][key] for r in results_list]
                averaged_conditional_mi[key] = np.mean(values)
                averaged_conditional_mi[f"{key}_std"] = np.std(values)
            averaged_results["conditional_mi"] = averaged_conditional_mi

        # Average PID results
        if "pid" in results_list[0]:
            averaged_pid = {}
            for pid_key in results_list[0]["pid"].keys():
                averaged_pid[pid_key] = {}
                for metric in results_list[0]["pid"][pid_key].keys():
                    values = [r["pid"][pid_key][metric] for r in results_list]
                    averaged_pid[pid_key][metric] = np.mean(values)
                    averaged_pid[pid_key][f"{metric}_std"] = np.std(values)
            averaged_results["pid"] = averaged_pid

        return averaged_results

    def _compute_layer_statistics(
        self, results_list: list[dict[str, Any]]
    ) -> dict[str, float]:
        """Compute mean and std of MI results across multiple branches for a given layer."""
        layer_stats = {}

        # Handle nested structure
        if "basic_mi" in results_list[0]:
            # Process basic MI metrics
            if "basic_mi" in results_list[0]:
                for key in results_list[0]["basic_mi"].keys():
                    values = [r["basic_mi"][key] for r in results_list]
                    layer_stats[f"{key}_mean"] = np.mean(values)
                    layer_stats[f"{key}_std"] = np.std(values)

            # Process pairwise MI metrics
            if "pairwise_mi" in results_list[0]:
                for key in results_list[0]["pairwise_mi"].keys():
                    values = [r["pairwise_mi"][key] for r in results_list]
                    layer_stats[f"{key}_mean"] = np.mean(values)
                    layer_stats[f"{key}_std"] = np.std(values)

            # Process conditional MI metrics
            if "conditional_mi" in results_list[0]:
                for key in results_list[0]["conditional_mi"].keys():
                    values = [r["conditional_mi"][key] for r in results_list]
                    layer_stats[f"{key}_mean"] = np.mean(values)
                    layer_stats[f"{key}_std"] = np.std(values)

        return layer_stats

    def attach_forward_hooks(self, model: BaseModel):
        """Attach forward hooks to the model."""
        # Initalize data_dict attribute
        self.data_dict = {}

        handles = []
        for _name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if (
                    module.branch_excitation is not None
                    and module.branch_inhibition is not None
                ):
                    self.data_dict[_name] = {}
                    module._name = _name
                    handle = module.register_forward_hook(self.forward_hook)
                    handles.append(handle)

        return handles

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ):
        """Forward hook to collect branch data."""
        excitatory_input, inhibitory_input, _ = input

        # Get the outputs from TopKLinear layers
        # These represent the weighted sum for each branch (already computed by TopKLinear)
        # Shape: (batch_size, num_branches)
        excitation_per_branch = module.branch_excitation(excitatory_input)
        inhibition_per_branch = module.branch_inhibition(inhibitory_input)

        # Store per-branch data
        self.data_dict[module._name]["excitation"] = excitation_per_branch
        self.data_dict[module._name]["inhibition"] = inhibition_per_branch
        self.data_dict[module._name]["output"] = output

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        for handle in handles:
            handle.remove()

    def ensure_inhibitory_data(
        self, excitation_data: torch.Tensor, inhibition_data: torch.Tensor
    ):
        """Ensure inhibitory data is available, replacing with zeros if missing.

        Args:
            excitation_data: Shape [n_samples, n_branches] - excitatory activations
            inhibition_data: Shape [n_samples, n_branches] - inhibitory activations (can be None)

        Returns:
            Tuple of (excitation_data, inhibition_data) with guaranteed inhibitory data
        """
        # Handle case where inhibitory data might be missing/None
        if inhibition_data is None:
            inhibition_data = torch.zeros_like(excitation_data)
            self.logger.warning(
                "Inhibitory data is missing - using zeros for inhibitory activations"
            )

        return excitation_data, inhibition_data

    def has_inhibitory_data(self, layer_data_dict: dict) -> bool:
        """Check if the network has valid inhibitory data.

        Args:
            layer_data_dict: Dictionary containing layer data from all layers

        Returns:
            True if inhibitory data exists and is not all zeros
        """
        for layer_data in layer_data_dict.values():
            if "inhibition" in layer_data and layer_data["inhibition"] is not None:
                # Check if inhibitory data is not all zeros
                if torch.any(layer_data["inhibition"] != 0):
                    return True
        return False

    def compute_per_neuron_mi(
        self,
        E_data: np.ndarray,
        I_data: np.ndarray,
        Vout_data: np.ndarray,
        C: np.ndarray,
    ):
        """Compute MI for each neuron separately.

        Args:
            E_data: Shape [n_samples, n_neurons] - excitatory activations
            I_data: Shape [n_samples, n_neurons] - inhibitory activations
            Vout_data: Shape [n_samples, n_neurons] - output activations
            C: Shape [n_samples,] - class labels

        Returns:
            Dictionary with per-neuron MI results
        """
        n_samples, n_neurons = E_data.shape
        per_neuron_results = []

        self.logger.info(f"Computing MI for {n_neurons} neurons individually...")

        for neuron_idx in range(n_neurons):
            # Extract data for single neuron
            E_neuron = E_data[:, neuron_idx : neuron_idx + 1]  # Keep 2D shape
            I_neuron = I_data[:, neuron_idx : neuron_idx + 1]
            Vout_neuron = Vout_data[:, neuron_idx : neuron_idx + 1]

            # Compute MI for this neuron
            neuron_results = self.compute_metrics(
                E_neuron, I_neuron, Vout_neuron, C, None
            )
            neuron_results["neuron_index"] = neuron_idx
            per_neuron_results.append(neuron_results)

        # Also compute average across neurons
        averaged_results = self._average_mi_results(per_neuron_results)

        return {
            "per_neuron_results": per_neuron_results,
            "averaged_results": averaged_results,
            "num_neurons": n_neurons,
        }

    def compute_ei_specific_mi(
        self,
        E_data: np.ndarray,
        I_data: np.ndarray,
        Vout_data: np.ndarray,
        C: np.ndarray,
    ):
        """Compute MI separately for excitatory and inhibitory networks.

        Args:
            E_data: Shape [n_samples, n_neurons] - excitatory activations
            I_data: Shape [n_samples, n_neurons] - inhibitory activations
            Vout_data: Shape [n_samples, n_neurons] - output activations
            C: Shape [n_samples,] - class labels

        Returns:
            Dictionary with separate E and I network MI results
        """
        self.logger.info(
            "Computing MI separately for excitatory and inhibitory networks..."
        )

        results = {}

        # Always compute excitatory network MI
        e_results = self.compute_metrics(
            E_data, np.zeros_like(I_data), Vout_data, C, None
        )
        e_results["network_type"] = "excitatory"
        results["excitatory_results"] = e_results

        # Check if inhibitory data is meaningful (not all zeros)
        has_inhibitory = np.any(I_data != 0)

        if has_inhibitory:
            # Compute MI for inhibitory network only
            i_results = self.compute_metrics(
                np.zeros_like(E_data), I_data, Vout_data, C, None
            )
            i_results["network_type"] = "inhibitory"
            results["inhibitory_results"] = i_results

            # Compute MI for combined E+I network (current default)
            combined_results = self.compute_metrics(E_data, I_data, Vout_data, C, None)
            combined_results["network_type"] = "combined"
            results["combined_results"] = combined_results

            self.logger.info(
                "Computed MI for excitatory, inhibitory, and combined networks"
            )
        else:
            # No inhibitory data available
            self.logger.warning(
                "No inhibitory data detected - skipping inhibitory and combined network analysis"
            )
            results["inhibitory_results"] = None
            results["combined_results"] = (
                e_results  # Use excitatory results as combined
            )
            results["inhibitory_data_available"] = False

        results["inhibitory_data_available"] = has_inhibitory
        return results

    def compute_per_layer_enhanced_analysis(self, C: np.ndarray):
        """Compute enhanced analysis (per-neuron and EI-specific) for each layer separately.

        Args:
            C: Class labels for all samples

        Returns:
            Dictionary with per-layer enhanced analysis results
        """
        self.logger.info("Computing per-layer enhanced analysis...")

        per_layer_results = {}

        for layer_name, layer_data in self.data_dict.items():
            if "excitation" not in layer_data:
                continue

            self.logger.info(f"Processing enhanced analysis for layer: {layer_name}")

            # Ensure inhibitory data is available for this layer
            E_filtered, I_filtered = self.ensure_inhibitory_data(
                layer_data["excitation"], layer_data["inhibition"]
            )

            E_layer = E_filtered.detach().cpu().numpy()
            I_layer = I_filtered.detach().cpu().numpy()
            Vout_layer = layer_data["output"].detach().cpu().numpy()

            layer_results = {}

            # Add per-neuron analysis for this layer
            if self.per_neuron_analysis:
                per_neuron_results = self.compute_per_neuron_mi(
                    E_layer, I_layer, Vout_layer, C
                )
                layer_results["per_neuron_analysis"] = per_neuron_results

            # Add EI-network analysis for this layer
            if self.per_einet_analysis:
                ei_specific_results = self.compute_ei_specific_mi(
                    E_layer, I_layer, Vout_layer, C
                )
                layer_results["per_einet_analysis"] = ei_specific_results

            # Add layer metadata
            layer_results["layer_name"] = layer_name
            layer_results["layer_shape"] = {
                "excitation": E_layer.shape,
                "inhibition": I_layer.shape,
                "output": Vout_layer.shape,
            }
            layer_results["has_inhibitory_data"] = np.any(I_layer != 0)

            per_layer_results[layer_name] = layer_results

        return per_layer_results

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """Run integrated information analysis on the model or compute metrics from arrays.

        This method supports both the new API (model-based) and legacy API (array-based).

        New API (using keyword arguments):
            analyze(model=model, test_dataset=test_dataset, device=device, save_path=save_path, filename=filename)

        New API (using positional arguments):
            analyze(model, test_dataset, device, save_path, filename)

        Legacy API (for backward compatibility):
            analyze(E, I_var, Vout, C, Vinf=None)
        """
        # Handle keyword arguments for new API
        if model is not None:
            model_or_E = model
            test_dataset_or_I = test_dataset
            device_or_Vout = device
            save_path_or_C = save_path
            filename_or_Vinf = filename

        # Check if this is the legacy API call (numpy arrays)
        if isinstance(model_or_E, np.ndarray):
            # Legacy API: analyze(E, I_var, Vout, C, Vinf=None)
            E = model_or_E
            I_var = test_dataset_or_I
            Vout = device_or_Vout
            C = save_path_or_C
            Vinf = filename_or_Vinf

            return self.compute_metrics(E, I_var, Vout, C, Vinf)

        # New API: analyze(model, test_dataset, device, save_path, filename)
        model = model_or_E
        test_dataset = test_dataset_or_I
        device = device_or_Vout
        save_path = save_path_or_C
        filename = filename_or_Vinf

        try:
            # Check if model has ExcitationInhibitionNetwork
            if not isinstance(model.core_network, ExcitationInhibitionNetwork):
                self.logger.warning(
                    "Model does not have ExcitationInhibitionNetwork. Skipping information analysis."
                )
                return

            self.logger.info("Running integrated information analysis...")

            # Prepare data
            model = model.to(device)
            model.eval()

            inputs, labels = test_dataset[:]
            inputs: torch.Tensor
            labels: torch.Tensor

            # Optional: Sample a subset of data for faster computation
            if self.max_samples and len(inputs) > self.max_samples:
                self.logger.info(
                    f"Sampling {self.max_samples} samples from {len(inputs)} for faster computation"
                )
                indices = torch.randperm(len(inputs))[: self.max_samples]
                inputs = inputs[indices]
                labels = labels[indices]

            inputs = inputs.to(device)
            labels = labels.to(device)

            # Attach forward hooks
            handles = self.attach_forward_hooks(model)

            # Forward pass to collect activations
            with torch.no_grad():
                _ = model(inputs)

            # Remove hooks
            self.remove_forward_hooks(handles)

            C = labels.detach().cpu().numpy()

            self.data_dict: dict[str, dict[str, torch.Tensor]]

            if self.computation_level == "single_branch":
                # Individual branch analysis: Compute MI for each branch separately and average
                self.logger.info("Computing MI for each branch separately...")

                all_results = []
                branch_names = []
                layer_results = {}  # Store results per layer

                for layer_name, layer_data in self.data_dict.items():
                    if "excitation" not in layer_data:
                        continue

                    # Get data for all branches in this layer
                    E_all_branches_raw = layer_data["excitation"]
                    I_all_branches_raw = layer_data["inhibition"]
                    Vout_all_branches = layer_data["output"].detach().cpu().numpy()

                    # Ensure inhibitory data is available
                    E_all_branches_filtered, I_all_branches_filtered = (
                        self.ensure_inhibitory_data(
                            E_all_branches_raw, I_all_branches_raw
                        )
                    )

                    E_all_branches = E_all_branches_filtered.detach().cpu().numpy()
                    I_all_branches = I_all_branches_filtered.detach().cpu().numpy()

                    # E_all_branches shape: (n_samples, n_branches)
                    n_branches = E_all_branches.shape[1]

                    # Store per-branch results for this layer
                    layer_branch_results = []

                    # Process each branch individually
                    for branch_idx in range(n_branches):
                        # Extract data for this specific branch
                        E_branch = E_all_branches[
                            :, branch_idx : branch_idx + 1
                        ]  # Keep 2D shape
                        I_branch = I_all_branches[:, branch_idx : branch_idx + 1]
                        Vout_branch = Vout_all_branches[:, branch_idx : branch_idx + 1]

                        # Compute MI for this single branch
                        branch_results = self.compute_metrics(
                            E_branch, I_branch, Vout_branch, C, None
                        )
                        all_results.append(branch_results)
                        layer_branch_results.append(branch_results)
                        branch_names.append(f"{layer_name}_branch_{branch_idx}")

                    # Compute layer statistics
                    if layer_branch_results:
                        layer_stats = self._compute_layer_statistics(
                            layer_branch_results
                        )
                        layer_results[layer_name] = layer_stats

                # Average the results across branches
                if all_results:
                    results = self._average_mi_results(all_results)
                    results["branch_names"] = branch_names
                    results["num_branches"] = len(all_results)
                    results["layer_statistics"] = (
                        layer_results  # Add per-layer statistics
                    )
                    results["computation_level"] = "single_branch"
                else:
                    self.logger.warning("No branch results to average")
                    return

            elif self.computation_level == "layer_branch":
                # Layer-level analysis: treat each layer as a unit
                self.logger.info("Computing MI at layer level...")

                all_results = []
                layer_names = []

                for layer_name, layer_data in self.data_dict.items():
                    if "excitation" not in layer_data:
                        continue

                    # Aggregate all branches within this layer using branch_aggregation
                    E_layer_data_raw = layer_data[
                        "excitation"
                    ]  # Shape: [n_samples, n_branches]
                    I_layer_data_raw = layer_data["inhibition"]
                    Vout_layer_data = layer_data["output"]

                    # Ensure inhibitory data is available
                    E_layer_data, I_layer_data = self.ensure_inhibitory_data(
                        E_layer_data_raw, I_layer_data_raw
                    )

                    if self.branch_aggregation == "mean":
                        E_layer = (
                            E_layer_data.mean(dim=1).detach().cpu().numpy()
                        )  # [n_samples,]
                        I_layer = I_layer_data.mean(dim=1).detach().cpu().numpy()
                        Vout_layer = Vout_layer_data.mean(dim=1).detach().cpu().numpy()
                        # Reshape to 2D for MI computation
                        E_layer = E_layer.reshape(-1, 1)
                        I_layer = I_layer.reshape(-1, 1)
                        Vout_layer = Vout_layer.reshape(-1, 1)
                    elif self.branch_aggregation == "multivariate":
                        # Multivariate: Keep all branches as multivariate input (no aggregation)
                        E_layer = (
                            E_layer_data.detach().cpu().numpy()
                        )  # [n_samples, n_branches]
                        I_layer = (
                            I_layer_data.detach().cpu().numpy()
                        )  # [n_samples, n_branches]
                        Vout_layer = (
                            Vout_layer_data.detach().cpu().numpy()
                        )  # [n_samples, n_branches]
                    elif self.branch_aggregation == "sample":
                        # Sample: Treat each branch as a separate observation/sample
                        # Flatten [n_samples, n_branches] -> [n_samples * n_branches, 1]
                        E_layer = (
                            E_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
                        )
                        I_layer = (
                            I_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
                        )
                        Vout_layer = (
                            Vout_layer_data.detach()
                            .cpu()
                            .numpy()
                            .flatten()
                            .reshape(-1, 1)
                        )
                        # Need to replicate C for each branch
                        n_samples, n_branches = E_layer_data.shape
                        C_expanded = np.repeat(
                            C, n_branches, axis=0
                        )  # Expand C to match new sample count
                        C = C_expanded  # Update C for this layer computation
                    else:
                        # Default to mean
                        E_layer = E_layer_data.mean(dim=1).detach().cpu().numpy()
                        I_layer = I_layer_data.mean(dim=1).detach().cpu().numpy()
                        Vout_layer = Vout_layer_data.mean(dim=1).detach().cpu().numpy()
                        # Reshape to 2D for MI computation
                        E_layer = E_layer.reshape(-1, 1)
                        I_layer = I_layer.reshape(-1, 1)
                        Vout_layer = Vout_layer.reshape(-1, 1)

                    # Compute MI for this layer
                    layer_results = self.compute_metrics(
                        E_layer, I_layer, Vout_layer, C, None
                    )
                    all_results.append(layer_results)
                    layer_names.append(layer_name)

                # Average the results across layers
                if all_results:
                    results = self._average_mi_results(all_results)
                    results["layer_names"] = layer_names
                    results["num_layers"] = len(all_results)
                    results["computation_level"] = "layer_branch"
                else:
                    self.logger.warning("No layer results to average")
                    return

            else:  # computation_level == "all_branch"
                # Full network aggregation: properly handle different-sized tensors
                self.logger.info(
                    f"Aggregating branches using method: {self.branch_aggregation} (full network aggregation)"
                )

                # Collect aggregated data from each layer
                layer_excitation = []
                layer_inhibition = []
                layer_output = []

                for layer_data in self.data_dict.values():
                    if "excitation" in layer_data:
                        # Aggregate within each layer first
                        if self.branch_aggregation == "mean":
                            layer_E = layer_data["excitation"].mean(
                                dim=1, keepdim=True
                            )  # [n_samples, 1]
                            layer_I = layer_data["inhibition"].mean(dim=1, keepdim=True)
                            layer_Vout = layer_data["output"].mean(dim=1, keepdim=True)
                        elif self.branch_aggregation == "multivariate":
                            # For network level with multivariate, keep branches separate for each layer
                            layer_E = layer_data[
                                "excitation"
                            ]  # [n_samples, n_branches]
                            layer_I = layer_data[
                                "inhibition"
                            ]  # [n_samples, n_branches]
                            layer_Vout = layer_data["output"]  # [n_samples, n_branches]
                        elif self.branch_aggregation == "sample":
                            # For network level with sample, flatten each layer's branches into samples
                            n_samples, n_branches = layer_data["excitation"].shape
                            layer_E = layer_data["excitation"].view(
                                n_samples * n_branches, 1
                            )
                            layer_I = layer_data["inhibition"].view(
                                n_samples * n_branches, 1
                            )
                            layer_Vout = layer_data["output"].view(
                                n_samples * n_branches, 1
                            )
                        else:
                            # Default to mean
                            layer_E = layer_data["excitation"].mean(dim=1, keepdim=True)
                            layer_I = layer_data["inhibition"].mean(dim=1, keepdim=True)
                            layer_Vout = layer_data["output"].mean(dim=1, keepdim=True)

                        layer_excitation.append(layer_E)
                        layer_inhibition.append(layer_I)
                        layer_output.append(layer_Vout)

                if not layer_excitation:
                    self.logger.warning(
                        "No excitatory activations collected. Skipping information analysis."
                    )
                    return

                # Handle concatenation based on aggregation method
                if self.branch_aggregation == "sample":
                    # For sample aggregation, concatenate along sample dimension (axis=0)
                    E = (
                        torch.cat(layer_excitation, dim=0).detach().cpu().numpy()
                    )  # [n_samples * n_branches * n_layers, 1]
                    I_var = torch.cat(layer_inhibition, dim=0).detach().cpu().numpy()
                    Vout = torch.cat(layer_output, dim=0).detach().cpu().numpy()
                    # Expand C to match the new sample count
                    total_samples = E.shape[0]
                    original_samples = C.shape[0]
                    expansion_factor = total_samples // original_samples
                    C = np.repeat(C, expansion_factor, axis=0)
                elif self.branch_aggregation == "multivariate":
                    # For multivariate, concatenate along feature dimension (axis=1)
                    E = (
                        torch.cat(layer_excitation, dim=1).detach().cpu().numpy()
                    )  # [n_samples, n_branches * n_layers]
                    I_var = torch.cat(layer_inhibition, dim=1).detach().cpu().numpy()
                    Vout = torch.cat(layer_output, dim=1).detach().cpu().numpy()
                else:
                    # For mean/max/sum, concatenate across layers (all tensors are [n_samples, 1])
                    E = (
                        torch.cat(layer_excitation, dim=1).detach().cpu().numpy()
                    )  # [n_samples, n_layers]
                    I_var = torch.cat(layer_inhibition, dim=1).detach().cpu().numpy()
                    Vout = torch.cat(layer_output, dim=1).detach().cpu().numpy()

                self.logger.info(
                    "Aggregated data shapes - "
                    f"E: {E.shape}, I: {I_var.shape}, Vout: {Vout.shape}, C: {C.shape}"
                )

                # Get soma activation (Vinf) if available for aggregate mode
                Vinf = None
                # For now, we'll use the final model output as Vinf
                with torch.no_grad():
                    final_output: torch.Tensor = model(inputs)
                    if hasattr(model, "final_fc"):
                        # Model has final classification layer
                        Vinf = final_output.detach().cpu().numpy()

                # Run information analysis on aggregated data
                results = self.compute_metrics(E, I_var, Vout, C, Vinf)
                results["aggregation_method"] = self.branch_aggregation
                results["computation_level"] = "all_branch"

            # Add enhanced analysis results if requested
            if (
                self.per_neuron_analysis
                or self.per_einet_analysis
                or self.per_layer_analysis
            ):
                self.logger.info("Running enhanced analysis modes...")

                # Add per-layer analysis if requested
                if self.per_layer_analysis:
                    per_layer_results = self.compute_per_layer_enhanced_analysis(C)
                    results["per_layer_analysis"] = per_layer_results

                # Network-level enhanced analysis (existing functionality)
                if self.per_neuron_analysis or self.per_einet_analysis:
                    # Collect all data for enhanced analysis
                    all_E_data = []
                    all_I_data = []
                    all_Vout_data = []

                    for layer_data in self.data_dict.values():
                        if "excitation" in layer_data:
                            # Ensure inhibitory data is available
                            E_filtered, I_filtered = self.ensure_inhibitory_data(
                                layer_data["excitation"], layer_data["inhibition"]
                            )
                            all_E_data.append(E_filtered.detach().cpu().numpy())
                            all_I_data.append(I_filtered.detach().cpu().numpy())
                            all_Vout_data.append(
                                layer_data["output"].detach().cpu().numpy()
                            )

                    # Concatenate data across layers (network-level view)
                    if all_E_data:
                        E_all = np.concatenate(
                            all_E_data, axis=1
                        )  # [n_samples, total_neurons]
                        I_all = np.concatenate(all_I_data, axis=1)
                        Vout_all = np.concatenate(all_Vout_data, axis=1)

                        # Add per-neuron analysis results
                        if self.per_neuron_analysis:
                            per_neuron_results = self.compute_per_neuron_mi(
                                E_all, I_all, Vout_all, C
                            )
                            results["per_neuron_analysis"] = per_neuron_results

                        # Add EI-network analysis results
                        if self.per_einet_analysis:
                            ei_specific_results = self.compute_ei_specific_mi(
                                E_all, I_all, Vout_all, C
                            )
                            results["per_einet_analysis"] = ei_specific_results

            # Add metadata about analysis configuration
            results["per_neuron_analysis_enabled"] = self.per_neuron_analysis
            results["per_einet_analysis_enabled"] = self.per_einet_analysis
            results["per_layer_analysis_enabled"] = self.per_layer_analysis
            results["has_inhibitory_data"] = self.has_inhibitory_data(self.data_dict)

            # Save results
            if save_path is not None:
                save_dict(results, save_path, filename)
                summary = self.get_summary(results)
                summary_path = os.path.join(save_path, f"{filename}_summary.txt")
                with open(summary_path, "w") as f:
                    f.write(summary)
            else:
                return results

        except Exception as e:
            self.logger.error(f"Error in information analysis: {e}")
            traceback.print_exc()

    def compute_class_mi(self, activations, labels):
        return self.estimator.mutual_information(activations, labels)

    def compute_conditional_mi(self, activations, labels, condition):
        return self.estimator.conditional_mutual_information(
            activations, labels, condition
        )
