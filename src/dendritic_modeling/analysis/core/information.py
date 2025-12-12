"""Integrated information analyzer for dendritic networks.

This module provides a unified interface for computing various information-theoretic
metrics using the multivariate methods from information.py. It integrates seamlessly
with the main configuration system.
"""

import logging
import os
import time
from typing import Any, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config import InformationAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.plotting.visualizations.information_plots import (
    plot_fully_aggregated_analysis,
    plot_info_vs_ei_depth_per_dendritic_layer,
    plot_layer_wise_aggregated_analysis,
    plot_network_wise_aggregated_analysis,
    plot_structured_ei_network_analysis,
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
from dendritic_modeling.utils.lda import apply_nonnegative_lda, fit_nonnegative_lda


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

        # Include branch input (Vb) in analysis
        self.include_branch_input = getattr(params, "include_branch_input", True)

        # Compute LDA-based versions of E, I, Vb
        self.compute_lda_weights = getattr(params, "compute_lda_weights", False)

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
        self.estimators[self.method] = InformationEstimatorFactory.create(
            estimator_config
        )

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
        Vb: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute mutual information between each variable and C."""
        self.logger.info(
            f"Computing basic MI: E{E.shape}, I{I_var.shape}, Vout{Vout.shape}, C{C.shape}"
        )
        metrics = {}

        # I(E;C)
        self.logger.info("  Computing I(E;C)...")
        start_time = time.time()
        metrics["I(E;C)"] = self.estimators[self.method].mutual_information(E, C)
        self.logger.info(
            f"  I(E;C) = {metrics['I(E;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(I;C)
        self.logger.info("  Computing I(I;C)...")
        start_time = time.time()
        metrics["I(I;C)"] = self.estimators[self.method].mutual_information(I_var, C)
        self.logger.info(
            f"  I(I;C) = {metrics['I(I;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(Vb;C) if available and enabled
        if Vb is not None and self.include_branch_input:
            self.logger.info("  Computing I(Vb;C)...")
            start_time = time.time()
            metrics["I(Vb;C)"] = self.estimators[self.method].mutual_information(Vb, C)
            self.logger.info(
                f"  I(Vb;C) = {metrics['I(Vb;C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        # I(Vout;C)
        self.logger.info("  Computing I(Vout;C)...")
        start_time = time.time()
        metrics["I(Vout;C)"] = self.estimators[self.method].mutual_information(Vout, C)
        self.logger.info(
            f"  I(Vout;C) = {metrics['I(Vout;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(Vinf;C) if available
        if Vinf is not None and self.include_vinf:
            self.logger.info("  Computing I(Vinf;C)...")
            start_time = time.time()
            metrics["I(Vinf;C)"] = self.estimators[self.method].mutual_information(
                Vinf, C
            )
            self.logger.info(
                f"  I(Vinf;C) = {metrics['I(Vinf;C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        # Joint MI: I(E,I;C)
        self.logger.info("  Computing I(E,I;C)...")
        start_time = time.time()
        EI = np.hstack([E, I_var])
        metrics["I(E,I;C)"] = self.estimators[self.method].mutual_information(EI, C)
        self.logger.info(
            f"  I(E,I;C) = {metrics['I(E,I;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # Optionally normalize by entropy
        if self.normalize_mi:
            self.logger.info("  Computing entropy H(C) for normalization...")
            start_time = time.time()
            H_C = self.estimators[self.method].entropy(C.reshape(-1, 1))
            self.logger.info(
                f"  H(C) = {H_C:.4f} (took {time.time() - start_time:.1f}s)"
            )
            # Create a list of keys to avoid modifying dict during iteration
            keys_to_normalize = list(metrics.keys())
            for key in keys_to_normalize:
                metrics[f"{key}_normalized"] = metrics[key] / H_C if H_C > 0 else 0.0

        self.logger.info("  Basic MI computation completed.")
        return metrics

    def _compute_pairwise_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
        Vb: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute pairwise mutual information between variables."""
        self.logger.info("Computing pairwise MI between neural variables...")
        metrics = {}

        # I(E;I)
        self.logger.info("  Computing I(E;I)...")
        start_time = time.time()
        metrics["I(E;I)"] = self.estimators[self.method].mutual_information(E, I_var)
        self.logger.info(
            f"  I(E;I) = {metrics['I(E;I)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(E;Vout)
        self.logger.info("  Computing I(E;Vout)...")
        start_time = time.time()
        metrics["I(E;Vout)"] = self.estimators[self.method].mutual_information(E, Vout)
        self.logger.info(
            f"  I(E;Vout) = {metrics['I(E;Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(I;Vout)
        self.logger.info("  Computing I(I;Vout)...")
        start_time = time.time()
        metrics["I(I;Vout)"] = self.estimators[self.method].mutual_information(
            I_var, Vout
        )
        self.logger.info(
            f"  I(I;Vout) = {metrics['I(I;Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # Pairwise MI with Vb if available
        if Vb is not None and self.include_branch_input:
            # I(E;Vb)
            self.logger.info("  Computing I(E;Vb)...")
            start_time = time.time()
            metrics["I(E;Vb)"] = self.estimators[self.method].mutual_information(E, Vb)
            self.logger.info(
                f"  I(E;Vb) = {metrics['I(E;Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vb)
            self.logger.info("  Computing I(I;Vb)...")
            start_time = time.time()
            metrics["I(I;Vb)"] = self.estimators[self.method].mutual_information(
                I_var, Vb
            )
            self.logger.info(
                f"  I(I;Vb) = {metrics['I(I;Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(Vb;Vout)
            self.logger.info("  Computing I(Vb;Vout)...")
            start_time = time.time()
            metrics["I(Vb;Vout)"] = self.estimators[self.method].mutual_information(
                Vb, Vout
            )
            self.logger.info(
                f"  I(Vb;Vout) = {metrics['I(Vb;Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        if Vinf is not None and self.include_vinf:
            # I(E;Vinf)
            self.logger.info("  Computing I(E;Vinf)...")
            start_time = time.time()
            metrics["I(E;Vinf)"] = self.estimators[self.method].mutual_information(
                E, Vinf
            )
            self.logger.info(
                f"  I(E;Vinf) = {metrics['I(E;Vinf)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vinf)
            self.logger.info("  Computing I(I;Vinf)...")
            start_time = time.time()
            metrics["I(I;Vinf)"] = self.estimators[self.method].mutual_information(
                I_var, Vinf
            )
            self.logger.info(
                f"  I(I;Vinf) = {metrics['I(I;Vinf)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(Vout;Vinf)
            self.logger.info("  Computing I(Vout;Vinf)...")
            start_time = time.time()
            metrics["I(Vout;Vinf)"] = self.estimators[self.method].mutual_information(
                Vout, Vinf
            )
            self.logger.info(
                f"  I(Vout;Vinf) = {metrics['I(Vout;Vinf)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        self.logger.info("  Pairwise MI computation completed.")
        return metrics

    def _compute_conditional_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
        Vb: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute conditional mutual information given C."""
        self.logger.info(
            "Computing conditional MI metrics (conditioned on class labels)..."
        )
        metrics = {}

        # I(E;I|C)
        self.logger.info("  Computing I(E;I|C)...")
        start_time = time.time()
        metrics["I(E;I|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(E, I_var, C.reshape(-1, 1))
        self.logger.info(
            f"  I(E;I|C) = {metrics['I(E;I|C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(E;Vout|C)
        self.logger.info("  Computing I(E;Vout|C)...")
        start_time = time.time()
        metrics["I(E;Vout|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(E, Vout, C.reshape(-1, 1))
        self.logger.info(
            f"  I(E;Vout|C) = {metrics['I(E;Vout|C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(I;Vout|C)
        self.logger.info("  Computing I(I;Vout|C)...")
        start_time = time.time()
        metrics["I(I;Vout|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(I_var, Vout, C.reshape(-1, 1))
        self.logger.info(
            f"  I(I;Vout|C) = {metrics['I(I;Vout|C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # Conditional MI with Vb if available
        if Vb is not None and self.include_branch_input:
            # I(E;Vb|C)
            self.logger.info("  Computing I(E;Vb|C)...")
            start_time = time.time()
            metrics["I(E;Vb|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(E, Vb, C.reshape(-1, 1))
            self.logger.info(
                f"  I(E;Vb|C) = {metrics['I(E;Vb|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vb|C)
            self.logger.info("  Computing I(I;Vb|C)...")
            start_time = time.time()
            metrics["I(I;Vb|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(I_var, Vb, C.reshape(-1, 1))
            self.logger.info(
                f"  I(I;Vb|C) = {metrics['I(I;Vb|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(Vb;Vout|C)
            self.logger.info("  Computing I(Vb;Vout|C)...")
            start_time = time.time()
            metrics["I(Vb;Vout|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(Vb, Vout, C.reshape(-1, 1))
            self.logger.info(
                f"  I(Vb;Vout|C) = {metrics['I(Vb;Vout|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        if Vinf is not None and self.include_vinf:
            # I(E;Vinf|C)
            self.logger.info("  Computing I(E;Vinf|C)...")
            start_time = time.time()
            metrics["I(E;Vinf|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(E, Vinf, C.reshape(-1, 1))
            self.logger.info(
                f"  I(E;Vinf|C) = {metrics['I(E;Vinf|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vinf|C)
            self.logger.info("  Computing I(I;Vinf|C)...")
            start_time = time.time()
            metrics["I(I;Vinf|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(I_var, Vinf, C.reshape(-1, 1))
            self.logger.info(
                f"  I(I;Vinf|C) = {metrics['I(I;Vinf|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        self.logger.info("  Conditional MI computation completed.")
        return metrics

    def _compute_pid(
        self, E: np.ndarray, I_var: np.ndarray, Vout: np.ndarray, C: np.ndarray
    ) -> dict[str, Any]:
        """Compute Partial Information Decomposition."""
        self.logger.info("Computing Partial Information Decomposition (PID)...")
        pid_results = {}

        # Prepare data for PID (may need binarization)
        self.logger.info(f"  Data preprocessing: binarize={self.pid_binarize}")
        if self.pid_binarize:
            self.logger.info(
                f"  Binarizing data using method: {self.pid_binarize_method}"
            )
            start_time = time.time()
            E_pid = self._binarize(E)
            I_pid = self._binarize(I_var)
            Vout_pid = self._binarize(Vout)
            C_pid = self._binarize(C.reshape(-1, 1))
            self.logger.info(
                f"  Data binarization completed (took {time.time() - start_time:.1f}s)"
            )
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
            self.logger.info(f"  Computing {pid_key}...")
            start_time = time.time()
            try:
                synergy, redundancy = pid_synergy_redundancy(
                    [source1_array, source2_array], target_array
                )

                # Compute unique information
                unique_1 = (
                    self.estimators[self.method].mutual_information(
                        source1_array, target_array
                    )
                    - redundancy
                )
                unique_2 = (
                    self.estimators[self.method].mutual_information(
                        source2_array, target_array
                    )
                    - redundancy
                )

                pid_results[pid_key] = {
                    "synergy": synergy,
                    "redundancy": redundancy,
                    f"unique_{source1_name}": unique_1,
                    f"unique_{source2_name}": unique_2,
                }
                self.logger.info(
                    f"  {pid_key}: synergy={synergy:.4f}, redundancy={redundancy:.4f}, unique_{source1_name}={unique_1:.4f}, unique_{source2_name}={unique_2:.4f} (took {time.time() - start_time:.1f}s)"
                )
            except Exception as e:
                self.logger.warning(
                    f"  {pid_key} computation failed: {e} (took {time.time() - start_time:.1f}s)"
                )

        # PID: (E, I) -> C
        _pid_helper(E_pid, "E", I_pid, "I", C_pid, "C")
        # PID: (E, Vout) -> C
        _pid_helper(E_pid, "E", Vout_pid, "Vout", C_pid, "C")
        # PID: (I, Vout) -> C
        _pid_helper(I_pid, "I", Vout_pid, "Vout", C_pid, "C")
        # PID: (E, I) -> Vout
        _pid_helper(E_pid, "E", I_pid, "I", Vout_pid, "Vout")

        self.logger.info("  PID computation completed.")
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
        Vb: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Compute all requested information metrics.

        Parameters
        ----------
        E : np.ndarray
            Excitatory inputs, shape (n_samples,) or (n_samples, n_features)
        I_var : np.ndarray
            Inhibitory inputs, shape (n_samples,) or (n_samples, n_features)
        Vout : np.ndarray
            Branch outputs, shape (n_samples,) or (n_samples, n_features)
        C : np.ndarray
            Class labels, shape (n_samples,)
        Vinf : np.ndarray, optional
            Soma activation (Vinf), shape (n_samples,) or (n_samples, n_features)
        Vb : np.ndarray, optional
            Branch input (upstream dendritic input), shape (n_samples,) or (n_samples, n_features)

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

        if Vb is not None:
            Vb = self._ensure_2d(Vb)

        results = {
            "method": self.method,
            "n_samples": len(E),
        }

        # 1. Basic mutual information with C
        if self.compute_basic_mi:
            self.logger.info("Computing basic MI metrics...")
            results["basic_mi"] = self._compute_basic_mi(E, I_var, Vout, C, Vinf, Vb)

        # 2. Pairwise mutual information
        if self.compute_pairwise_mi:
            self.logger.info("Computing pairwise MI metrics...")
            results["pairwise_mi"] = self._compute_pairwise_mi(E, I_var, Vout, Vinf, Vb)

        # 3. Conditional mutual information
        if self.compute_conditional_mi:
            self.logger.info("Computing conditional MI metrics...")
            results["conditional_mi"] = self._compute_conditional_mi(
                E, I_var, Vout, C, Vinf, Vb
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

        # Handle nested structure - collect ALL unique keys across all branches
        if results_list and "basic_mi" in results_list[0]:
            # Process each metric category, collecting all unique keys
            for metric_category in ["basic_mi", "pairwise_mi", "conditional_mi"]:
                # Gather all unique keys across all branches
                all_keys = set()
                for r in results_list:
                    if metric_category in r:
                        all_keys.update(r[metric_category].keys())

                # Process each unique key
                for key in all_keys:
                    values = []
                    for r in results_list:
                        if metric_category in r and key in r[metric_category]:
                            values.append(r[metric_category][key])
                    if values:
                        layer_stats[f"{key}_mean"] = np.mean(values)
                        layer_stats[f"{key}_std"] = np.std(values)

        return layer_stats

    def _compute_variance_analysis(self, results: dict) -> dict:
        """
        Compute variance analysis across different hierarchical levels.

        This computes how much information values vary:
        - When per_neuron_analysis=True: variance across branches within neurons
        - When per_layer_analysis=True: variance across neurons within layers
        - When per_einet_analysis=True: variance across E/I networks

        Returns variance statistics that can be plotted alongside means.
        """
        variance_analysis = {}

        # Compute variance based on computation level and analysis modes
        if "layer_statistics" in results:
            layer_stats = results["layer_statistics"]

            # Extract variance information from the existing std values
            # The std values in layer_statistics represent variance across branches
            variance_by_layer = {}

            for layer_name, layer_data in layer_stats.items():
                layer_variances = {}

                # Extract variance from std values (std^2 = variance)
                for metric_key, value in layer_data.items():
                    if metric_key.endswith("_std"):
                        metric_base = metric_key.replace("_std", "")
                        variance_key = f"{metric_base}_variance"
                        layer_variances[variance_key] = value**2  # std^2 = variance

                # Always include layer in variance analysis (even if all variances are zero)
                # This ensures synthetic_soma is included when somatic_synapses=False
                variance_by_layer[layer_name] = layer_variances

            variance_analysis["layer_variances"] = variance_by_layer

        # Add metadata about what type of variance this represents
        variance_analysis["variance_interpretation"] = {
            "layer_variances": "Variance across branches within each layer",
            "computation_level": results.get("computation_level", "unknown"),
            "per_neuron_enabled": results.get("per_neuron_analysis_enabled", False),
            "per_layer_enabled": results.get("per_layer_analysis_enabled", False),
            "per_einet_enabled": results.get("per_einet_analysis_enabled", False),
        }

        return variance_analysis

    def attach_forward_hooks(self, model: BaseModel):
        """Attach forward hooks to the model."""
        # Initialize data_dict attribute and store somatic_synapses config
        self.data_dict = {}
        self.somatic_synapses_enabled = getattr(
            model.core_network, "somatic_synapses", True
        )

        handles = []
        for _name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                # Hook layers that have E/I synapses OR receive branch input
                # This ensures soma layers without synapses are still analyzed
                # for I(Vb;C) - the information in branch input
                has_synapses = (
                    module.branch_excitation is not None
                    and module.branch_inhibition is not None
                )
                has_branch_input = module.input_branches  # Has upstream branches

                if has_synapses or has_branch_input:
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
        excitatory_input, inhibitory_input, branch_input = input

        # Check if this layer has no synapses (e.g., soma with somatic_synapses=False)
        # This is detected by checking if branch_excitation is None
        has_no_synapses = (
            module.branch_excitation is None or module.branch_inhibition is None
        )

        if has_no_synapses:
            # For soma without synapses, create zero tensors for E/I but keep real output
            batch_size = output.shape[0]
            n_branches = output.shape[1] if len(output.shape) > 1 else 1

            # Zero excitation and inhibition since there are no synapses
            excitation_per_branch = torch.zeros(
                batch_size, n_branches, device=output.device
            )
            inhibition_per_branch = torch.zeros(
                batch_size, n_branches, device=output.device
            )
        else:
            # Normal processing for layers with actual synapses
            # Get the outputs from TopKLinear layers
            # These represent the weighted sum for each branch (already computed by TopKLinear)
            # Shape: (batch_size, num_branches)
            excitation_per_branch = (
                module.branch_excitation(excitatory_input)
                if module.branch_excitation is not None
                else torch.zeros_like(output)
            )
            inhibition_per_branch = (
                module.branch_inhibition(inhibitory_input)
                if module.branch_inhibition is not None
                else torch.zeros_like(output)
            )

        # Compute branch_input contribution if available
        if branch_input is not None and module.input_branches:
            branch_current = module.branches_to_output(branch_input)
        else:
            branch_current = torch.zeros_like(output)

        # Store per-branch data
        self.data_dict[module._name]["excitation"] = excitation_per_branch
        self.data_dict[module._name]["inhibition"] = inhibition_per_branch
        self.data_dict[module._name]["branch_input"] = branch_current
        self.data_dict[module._name]["output"] = output
        self.data_dict[module._name]["has_synapses"] = not has_no_synapses

        # Store raw inputs and module for LDA computation if needed
        if hasattr(self, "compute_lda_weights") and self.compute_lda_weights:
            self.data_dict[module._name]["raw_excitatory_input"] = excitatory_input
            self.data_dict[module._name]["raw_inhibitory_input"] = inhibitory_input
            self.data_dict[module._name]["raw_branch_input"] = branch_input
            self.data_dict[module._name]["module"] = module

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        for handle in handles:
            handle.remove()

    def _fit_nonnegative_lda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight_norm_order: Optional[int] = None,
        gamma: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Fit LDA with non-negative weight constraint.

        Delegates to the utility function in dendritic_modeling.utils.lda.
        """
        return fit_nonnegative_lda(X, y, weight_norm_order, gamma)

    def _apply_nonnegative_lda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight_norm_order: Optional[int] = None,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Apply non-negative LDA to get projected values.

        Delegates to the utility function in dendritic_modeling.utils.lda.
        """
        return apply_nonnegative_lda(X, y, weight_norm_order, gamma)

    def compute_lda_aggregated_signals(
        self,
        layer_data: dict,
        C: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute LDA-weighted versions of E, I, Vb, and Vout using TopK-masked features.

        For each branch, fit non-negative constrained LDA only on the input
        dimensions that pass through the TopK mask (i.e., the synapses that are
        actually connected). Non-negative weights match the network's positive
        synaptic weights.

        Vout_lin is computed by applying the same dendritic computation
        (shunting or additive) to E_lin, I_lin, Vb_lin, then passing through
        the network's reactivation function.

        Args:
            layer_data: Dictionary containing raw inputs and module
            C: Class labels, shape [n_samples,]

        Returns:
            Tuple of (E_lin, I_lin, Vb_lin, Vout_lin) each shape [n_samples, n_branches]
        """
        module = layer_data.get("module")
        if module is None:
            return None, None, None, None

        raw_exc = layer_data.get("raw_excitatory_input")
        raw_inh = layer_data.get("raw_inhibitory_input")
        raw_branch = layer_data.get("raw_branch_input")

        n_samples = C.shape[0]
        n_branches = layer_data["excitation"].shape[1]

        E_lin = np.zeros((n_samples, n_branches))
        I_lin = np.zeros((n_samples, n_branches))
        Vb_lin = np.zeros((n_samples, n_branches))
        Vout_lin = np.zeros((n_samples, n_branches))

        # Check if we have enough classes for LDA
        if len(np.unique(C)) <= 1:
            # Fall back to network weights
            E_lin = layer_data["excitation"].detach().cpu().numpy()
            I_lin = layer_data["inhibition"].detach().cpu().numpy()
            Vb_lin = layer_data["branch_input"].detach().cpu().numpy()
            Vout_lin = layer_data["output"].detach().cpu().numpy()
            return E_lin, I_lin, Vb_lin, Vout_lin

        # For E: Fit LDA on TopK-masked excitatory inputs for each branch
        if raw_exc is not None and module.branch_excitation is not None:
            raw_exc_np = raw_exc.detach().cpu().numpy()  # [n_samples, input_dim]

            # Get weight normalization parameters from the module (matching TopKLinear)
            exc_weight_norm_order = getattr(
                module.branch_excitation, "weight_norm_order", None
            )
            exc_gamma = getattr(module.branch_excitation, "gamma", 1.0)

            # Get the TopK mask for excitatory branches [n_branches, input_dim]
            exc_mask = module.branch_excitation.weight_mask().detach().cpu().numpy()

            # Fit separate LDA for each branch using its masked inputs
            for branch_idx in range(n_branches):
                branch_mask = exc_mask[branch_idx, :]  # [input_dim]
                active_indices = np.where(branch_mask > 0)[0]

                if len(active_indices) > 0:
                    # Extract only the active synaptic inputs for this branch
                    branch_inputs = raw_exc_np[:, active_indices]  # [n_samples, K]

                    try:
                        # Fit non-negative LDA on the K active synapses
                        # Use same weight normalization as the network's TopKLinear
                        E_lin[:, branch_idx] = self._apply_nonnegative_lda(
                            branch_inputs,
                            C,
                            weight_norm_order=exc_weight_norm_order,
                            gamma=exc_gamma,
                        )

                    except Exception as e:
                        self.logger.warning(
                            f"LDA failed for E branch {branch_idx}: {e}. Using network weights."
                        )
                        E_lin[:, branch_idx] = (
                            layer_data["excitation"][:, branch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                else:
                    E_lin[:, branch_idx] = (
                        layer_data["excitation"][:, branch_idx].detach().cpu().numpy()
                    )

        # For I: Fit LDA on TopK-masked inhibitory inputs for each branch
        if raw_inh is not None and module.branch_inhibition is not None:
            raw_inh_np = raw_inh.detach().cpu().numpy()  # [n_samples, input_dim]

            # Get weight normalization parameters from the module (matching TopKLinear)
            inh_weight_norm_order = getattr(
                module.branch_inhibition, "weight_norm_order", None
            )
            inh_gamma = getattr(module.branch_inhibition, "gamma", 1.0)

            # Get the TopK mask for inhibitory branches [n_branches, input_dim]
            inh_mask = module.branch_inhibition.weight_mask().detach().cpu().numpy()

            # Fit separate LDA for each branch using its masked inputs
            for branch_idx in range(n_branches):
                branch_mask = inh_mask[branch_idx, :]  # [input_dim]
                active_indices = np.where(branch_mask > 0)[0]

                if len(active_indices) > 0:
                    # Extract only the active synaptic inputs for this branch
                    branch_inputs = raw_inh_np[:, active_indices]  # [n_samples, K]

                    try:
                        # Fit non-negative LDA on the K active synapses
                        # Use same weight normalization as the network's TopKLinear
                        I_lin[:, branch_idx] = self._apply_nonnegative_lda(
                            branch_inputs,
                            C,
                            weight_norm_order=inh_weight_norm_order,
                            gamma=inh_gamma,
                        )

                    except Exception as e:
                        self.logger.warning(
                            f"LDA failed for I branch {branch_idx}: {e}. Using network weights."
                        )
                        I_lin[:, branch_idx] = (
                            layer_data["inhibition"][:, branch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                else:
                    I_lin[:, branch_idx] = (
                        layer_data["inhibition"][:, branch_idx].detach().cpu().numpy()
                    )

        # For Vb: Fit LDA on branch inputs using BlockLinear's block structure
        # branches_to_output is a BlockLinear where each output branch receives input
        # from a contiguous block of upstream branches
        if raw_branch is not None and module.input_branches:
            raw_branch_np = (
                raw_branch.detach().cpu().numpy()
            )  # [n_samples, n_upstream_branches]

            # BlockLinear doesn't have weight_norm_order/gamma like TopKLinear
            # Use defaults (L1 norm, gamma=1.0)
            vb_weight_norm_order = getattr(
                module.branches_to_output, "weight_norm_order", None
            )
            vb_gamma = getattr(module.branches_to_output, "gamma", 1.0)

            # Get block_size from branches_to_output (BlockLinear)
            if hasattr(module.branches_to_output, "block_size"):
                block_size = module.branches_to_output.block_size

                # Fit separate LDA for each output branch using its upstream block
                for branch_idx in range(n_branches):
                    # Each output branch receives from upstream indices [branch_idx * block_size : (branch_idx+1) * block_size]
                    start_idx = branch_idx * block_size
                    end_idx = min((branch_idx + 1) * block_size, raw_branch_np.shape[1])

                    if start_idx < raw_branch_np.shape[1]:
                        # Extract the upstream branches for this output branch
                        branch_inputs = raw_branch_np[:, start_idx:end_idx]

                        if branch_inputs.shape[1] > 0:
                            try:
                                # Fit non-negative LDA on the upstream branches
                                # Use same weight normalization as the network
                                Vb_lin[:, branch_idx] = self._apply_nonnegative_lda(
                                    branch_inputs,
                                    C,
                                    weight_norm_order=vb_weight_norm_order,
                                    gamma=vb_gamma,
                                )

                            except Exception as e:
                                self.logger.warning(
                                    f"LDA failed for Vb branch {branch_idx}: {e}. Using network weights."
                                )
                                Vb_lin[:, branch_idx] = (
                                    layer_data["branch_input"][:, branch_idx]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                        else:
                            Vb_lin[:, branch_idx] = (
                                layer_data["branch_input"][:, branch_idx]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                    else:
                        Vb_lin[:, branch_idx] = (
                            layer_data["branch_input"][:, branch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
            else:
                # Fallback: no block structure, use network weights
                Vb_lin = layer_data["branch_input"].detach().cpu().numpy()

        # Compute Vout_lin using the same dendritic computation as the network
        # With non-negative LDA weights, E_lin, I_lin, Vb_lin are all non-negative
        # so shunting inhibition formula works properly
        use_shunting = getattr(module, "use_shunting", True)
        epsilon = getattr(module, "epsilon", 1e-8)

        if use_shunting:
            # Shunting inhibition: voltage = (E + Vb) / (1 + E + I + Vb)
            # With non-negative inputs, this normalizes properly
            numerator = E_lin + Vb_lin
            denominator = 1.0 + E_lin + I_lin + Vb_lin
            voltage_lin = numerator / (denominator + epsilon)
        else:
            # Additive: voltage = E + Vb - I
            voltage_lin = E_lin + Vb_lin - I_lin

        # Apply reactivation function if available
        if hasattr(module, "reactivation") and module.reactivation is not None:
            import torch

            with torch.no_grad():
                voltage_tensor = torch.tensor(voltage_lin, dtype=torch.float32)
                if next(module.reactivation.parameters(), None) is not None:
                    device = next(module.reactivation.parameters()).device
                    voltage_tensor = voltage_tensor.to(device)
                Vout_lin = module.reactivation(voltage_tensor).cpu().numpy()
        else:
            # No reactivation, use voltage directly
            Vout_lin = voltage_lin

        return E_lin, I_lin, Vb_lin, Vout_lin

    def compute_lda_aggregated_signals_chained(
        self,
        layer_data: dict,
        C: np.ndarray,
        prev_layer_vout_lin: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute LDA-weighted signals with optional chained branch input.

        This is similar to compute_lda_aggregated_signals but allows overriding
        the branch input with Vout_lin from a previous layer, enabling proper
        chained LDA comparison across all network layers.

        Uses non-negative LDA weights to match network's non-negative synaptic weights.

        Args:
            layer_data: Dictionary containing raw inputs and module
            C: Class labels, shape [n_samples,]
            prev_layer_vout_lin: Vout_lin from previous layer, to use as branch input.
                                 If None, uses the raw_branch_input from layer_data.

        Returns:
            Tuple of (E_lin, I_lin, Vb_lin, Vout_lin) each shape [n_samples, n_branches]
        """
        module = layer_data.get("module")
        if module is None:
            return None, None, None, None

        raw_exc = layer_data.get("raw_excitatory_input")
        raw_inh = layer_data.get("raw_inhibitory_input")

        # Use chained branch input if provided, otherwise use raw_branch_input
        if prev_layer_vout_lin is not None:
            # Convert to tensor if needed
            if isinstance(prev_layer_vout_lin, np.ndarray):
                raw_branch = torch.tensor(prev_layer_vout_lin, dtype=torch.float32)
            else:
                raw_branch = prev_layer_vout_lin
        else:
            raw_branch = layer_data.get("raw_branch_input")

        n_samples = C.shape[0]
        n_branches = layer_data["excitation"].shape[1]

        E_lin = np.zeros((n_samples, n_branches))
        I_lin = np.zeros((n_samples, n_branches))
        Vb_lin = np.zeros((n_samples, n_branches))

        # Check if we have enough classes for LDA
        if len(np.unique(C)) <= 1:
            E_lin = layer_data["excitation"].detach().cpu().numpy()
            I_lin = layer_data["inhibition"].detach().cpu().numpy()
            if raw_branch is not None:
                if isinstance(raw_branch, torch.Tensor):
                    Vb_lin = raw_branch.detach().cpu().numpy()
                else:
                    Vb_lin = raw_branch
            Vout_lin = layer_data["output"].detach().cpu().numpy()
            return E_lin, I_lin, Vb_lin, Vout_lin

        # Process each branch using non-negative LDA
        # Get weight normalization parameters from the module (matching TopKLinear)
        exc_weight_norm_order = None
        exc_gamma = 1.0
        if (
            hasattr(module, "branch_excitation")
            and module.branch_excitation is not None
        ):
            exc_weight_norm_order = getattr(
                module.branch_excitation, "weight_norm_order", None
            )
            exc_gamma = getattr(module.branch_excitation, "gamma", 1.0)

        inh_weight_norm_order = None
        inh_gamma = 1.0
        if (
            hasattr(module, "branch_inhibition")
            and module.branch_inhibition is not None
        ):
            inh_weight_norm_order = getattr(
                module.branch_inhibition, "weight_norm_order", None
            )
            inh_gamma = getattr(module.branch_inhibition, "gamma", 1.0)

        vb_weight_norm_order = None
        vb_gamma = 1.0
        if (
            hasattr(module, "branches_to_output")
            and module.branches_to_output is not None
        ):
            vb_weight_norm_order = getattr(
                module.branches_to_output, "weight_norm_order", None
            )
            vb_gamma = getattr(module.branches_to_output, "gamma", 1.0)

        for branch_idx in range(n_branches):
            # E: Non-negative LDA on excitatory inputs
            if raw_exc is not None and hasattr(module, "branch_excitation"):
                try:
                    exc_np = raw_exc.detach().cpu().numpy()
                    # Use same weight normalization as the network's TopKLinear
                    E_lin[:, branch_idx] = self._apply_nonnegative_lda(
                        exc_np,
                        C,
                        weight_norm_order=exc_weight_norm_order,
                        gamma=exc_gamma,
                    )
                except Exception:
                    E_lin[:, branch_idx] = (
                        layer_data["excitation"][:, branch_idx].cpu().numpy()
                    )

            # I: Non-negative LDA on inhibitory inputs
            if raw_inh is not None and hasattr(module, "branch_inhibition"):
                try:
                    inh_np = raw_inh.detach().cpu().numpy()
                    # Use same weight normalization as the network's TopKLinear
                    I_lin[:, branch_idx] = self._apply_nonnegative_lda(
                        inh_np,
                        C,
                        weight_norm_order=inh_weight_norm_order,
                        gamma=inh_gamma,
                    )
                except Exception:
                    I_lin[:, branch_idx] = (
                        layer_data["inhibition"][:, branch_idx].cpu().numpy()
                    )

            # Vb: Non-negative LDA on branch inputs (chained from previous layer)
            if raw_branch is not None and hasattr(module, "branches_to_output"):
                try:
                    if isinstance(raw_branch, torch.Tensor):
                        branch_np = raw_branch.detach().cpu().numpy()
                    else:
                        branch_np = raw_branch
                    # Use same weight normalization as the network's BlockLinear
                    Vb_lin[:, branch_idx] = self._apply_nonnegative_lda(
                        branch_np,
                        C,
                        weight_norm_order=vb_weight_norm_order,
                        gamma=vb_gamma,
                    )
                except Exception:
                    if isinstance(raw_branch, torch.Tensor):
                        Vb_lin[:, branch_idx] = raw_branch[:, branch_idx].cpu().numpy()
                    else:
                        Vb_lin[:, branch_idx] = raw_branch[:, branch_idx]

        # Compute Vout_lin using the same dendritic computation as the network
        # With non-negative LDA weights, E_lin, I_lin, Vb_lin are all non-negative
        use_shunting = getattr(module, "use_shunting", True)
        epsilon = getattr(module, "epsilon", 1e-8)

        if use_shunting:
            numerator = E_lin + Vb_lin
            denominator = 1.0 + E_lin + I_lin + Vb_lin
            voltage_lin = numerator / (denominator + epsilon)
        else:
            voltage_lin = E_lin + Vb_lin - I_lin

        # Apply reactivation
        if hasattr(module, "reactivation") and module.reactivation is not None:
            voltage_tensor = torch.tensor(voltage_lin, dtype=torch.float32)
            device = next(module.parameters()).device
            voltage_tensor = voltage_tensor.to(device)
            with torch.no_grad():
                Vout_lin = module.reactivation(voltage_tensor).cpu().numpy()
        else:
            Vout_lin = voltage_lin

        return E_lin, I_lin, Vb_lin, Vout_lin

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
        _, n_neurons = E_data.shape
        per_neuron_results = []

        self.logger.info(f"Computing MI for {n_neurons} neurons individually...")

        for neuron_idx in range(n_neurons):
            # Extract data for single neuron
            E_neuron = E_data[:, neuron_idx : neuron_idx + 1]  # Keep 2D shape
            I_neuron = I_data[:, neuron_idx : neuron_idx + 1]
            Vout_neuron = Vout_data[:, neuron_idx : neuron_idx + 1]

            # Compute MI for this neuron
            neuron_results = self.compute_metrics(
                E_neuron, I_neuron, Vout_neuron, C, None, None
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
            E_data, np.zeros_like(I_data), Vout_data, C, None, None
        )
        e_results["network_type"] = "excitatory"
        results["excitatory_results"] = e_results

        # Check if inhibitory data is meaningful (not all zeros)
        has_inhibitory = np.any(I_data != 0)

        if has_inhibitory:
            # Compute MI for inhibitory network only
            i_results = self.compute_metrics(
                np.zeros_like(E_data), I_data, Vout_data, C, None, None
            )
            i_results["network_type"] = "inhibitory"
            results["inhibitory_results"] = i_results

            # Compute MI for combined E+I network (current default)
            combined_results = self.compute_metrics(
                E_data, I_data, Vout_data, C, None, None
            )
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

    def _save_per_layer_ei_summaries(
        self, per_layer_data: dict, save_path: str, base_filename: str
    ):
        """Generate and save individual summaries for each layer's E and I networks.

        Args:
            per_layer_data: Dictionary containing per-layer analysis results
            save_path: Directory to save summary files
            base_filename: Base filename for the summaries
        """
        self.logger.info("Generating individual layer E/I network summaries...")

        # Create subfolder for per-layer results
        per_layer_path = os.path.join(save_path, "per_layer_results")
        os.makedirs(per_layer_path, exist_ok=True)

        for layer_name, layer_results in per_layer_data.items():
            if "per_einet_analysis" in layer_results:
                ei_analysis = layer_results["per_einet_analysis"]

                # Clean layer name for filename
                clean_layer_name = layer_name.replace(".", "_").replace("/", "_")

                # Generate excitatory network summary
                if "excitatory_results" in ei_analysis:
                    exc_summary = self.get_summary(ei_analysis["excitatory_results"])
                    exc_summary_path = os.path.join(
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_excitatory_summary.txt",
                    )
                    with open(exc_summary_path, "w") as f:
                        f.write(f"Excitatory Network Summary - {layer_name}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(exc_summary)

                    # Also save the results as JSON for plotting
                    save_dict(
                        ei_analysis["excitatory_results"],
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_excitatory_results",
                    )

                # Generate inhibitory network summary
                if "inhibitory_results" in ei_analysis:
                    inh_summary = self.get_summary(ei_analysis["inhibitory_results"])
                    inh_summary_path = os.path.join(
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_inhibitory_summary.txt",
                    )
                    with open(inh_summary_path, "w") as f:
                        f.write(f"Inhibitory Network Summary - {layer_name}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(inh_summary)

                    # Also save the results as JSON for plotting
                    save_dict(
                        ei_analysis["inhibitory_results"],
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_inhibitory_results",
                    )

                # Generate combined network summary
                if "combined_results" in ei_analysis:
                    combined_summary = self.get_summary(ei_analysis["combined_results"])
                    combined_summary_path = os.path.join(
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_combined_summary.txt",
                    )
                    with open(combined_summary_path, "w") as f:
                        f.write(f"Combined E+I Network Summary - {layer_name}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(combined_summary)

                    # Also save the results as JSON for plotting
                    save_dict(
                        ei_analysis["combined_results"],
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_combined_results",
                    )

        self.logger.info(
            f"Generated individual summaries for {len(per_layer_data)} layers"
        )

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """Run integrated information analysis on the model.

        Parameters
        ----------
        model : BaseModel
            The trained model to analyze
        test_dataset : torch.utils.data.Dataset
            Test dataset for analysis
        device : str, optional
            Device to run analysis on, by default "cpu"
        save_path : Optional[str], optional
            Path to save results, by default None
        filename : str, optional
            Filename for saved results, by default "final"
        """
        self.logger.info("Starting information analysis")
        self.logger.info(
            f"Model: {type(model).__name__}, Dataset size: {len(test_dataset)}, Device: {device}"
        )

        try:
            # Check if model has ExcitationInhibitionNetwork
            if not isinstance(model.core_network, ExcitationInhibitionNetwork):
                self.logger.warning(
                    "Model does not have ExcitationInhibitionNetwork. Skipping information analysis."
                )
                return

            self.logger.info("Model validation passed - starting analysis")
            analysis_start_time = time.time()

            # Prepare data
            self.logger.info(f"Preparing data: moving model to {device}")
            model = model.to(device)
            model.eval()

            self.logger.info("Loading test dataset...")
            inputs, labels = test_dataset[:]
            inputs: torch.Tensor
            labels: torch.Tensor
            self.logger.info(
                f"Loaded dataset: {len(inputs)} samples, input shape: {inputs.shape}"
            )

            # Optional: Sample a subset of data for faster computation
            if self.max_samples and len(inputs) > self.max_samples:
                self.logger.info(
                    f"Sampling {self.max_samples} samples from {len(inputs)} for faster computation"
                )
                indices = torch.randperm(len(inputs))[: self.max_samples]
                inputs = inputs[indices]
                labels = labels[indices]
                self.logger.info(f"After sampling: {len(inputs)} samples")

            self.logger.info(f"Moving data to {device}...")
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Attach forward hooks and run forward pass
            handles = self.attach_forward_hooks(model)
            with torch.no_grad():
                _ = model(inputs)
            self.remove_forward_hooks(handles)
            C = labels.detach().cpu().numpy()

            self.logger.debug(
                f"Data collection completed. Found {len(self.data_dict)} layers."
            )

            self.data_dict: dict[str, dict[str, torch.Tensor]]

            if self.computation_level == "single_branch":
                # Individual branch analysis: Compute MI for each branch separately and average
                all_results = []
                branch_names = []
                layer_results = {}  # Store results per layer

                total_layers = len(
                    [
                        name
                        for name, data in self.data_dict.items()
                        if "excitation" in data
                    ]
                )
                total_branches = sum(
                    data.get("excitation", torch.empty(0, 0)).shape[1]
                    for data in self.data_dict.values()
                    if "excitation" in data
                )
                layer_count = 0

                self.logger.info(
                    f"Single-branch MI analysis: {total_layers} layers, {total_branches} branches, method={self.method}"
                )

                layers_to_process = [
                    (name, data)
                    for name, data in self.data_dict.items()
                    if "excitation" in data
                ]

                for layer_name, layer_data in tqdm(
                    layers_to_process,
                    desc="Processing layers",
                    leave=False,
                    ncols=100,
                ):
                    layer_count += 1

                    # Get data for all branches in this layer
                    E_all_branches_raw = layer_data["excitation"]
                    I_all_branches_raw = layer_data["inhibition"]
                    Vb_all_branches_raw = layer_data.get("branch_input")
                    Vout_all_branches = layer_data["output"].detach().cpu().numpy()

                    # Ensure inhibitory data is available
                    E_all_branches_filtered, I_all_branches_filtered = (
                        self.ensure_inhibitory_data(
                            E_all_branches_raw, I_all_branches_raw
                        )
                    )

                    E_all_branches = E_all_branches_filtered.detach().cpu().numpy()
                    I_all_branches = I_all_branches_filtered.detach().cpu().numpy()

                    # Get Vb data
                    if Vb_all_branches_raw is not None:
                        Vb_all_branches = Vb_all_branches_raw.detach().cpu().numpy()
                    else:
                        Vb_all_branches = np.zeros_like(E_all_branches)

                    # Compute LDA versions if requested
                    if self.compute_lda_weights:
                        E_lin, I_lin, Vb_lin, Vout_lin = (
                            self.compute_lda_aggregated_signals(layer_data, C)
                        )
                    else:
                        E_lin, I_lin, Vb_lin, Vout_lin = None, None, None, None

                    # E_all_branches shape: (n_samples, n_branches)
                    n_branches = E_all_branches.shape[1]

                    # Store per-branch results for this layer
                    layer_branch_results = []
                    layer_start_time = time.time()

                    # Process each branch individually
                    # Extract short layer name for progress bar
                    short_name = (
                        layer_name.split(".")[-1] if "." in layer_name else layer_name
                    )
                    for branch_idx in tqdm(
                        range(n_branches),
                        desc=f"  {short_name}",
                        leave=False,
                        ncols=100,
                        mininterval=0.5,
                    ):
                        branch_start_time = time.time()

                        # Extract data for this specific branch
                        E_branch = E_all_branches[
                            :, branch_idx : branch_idx + 1
                        ]  # Keep 2D shape
                        I_branch = I_all_branches[:, branch_idx : branch_idx + 1]
                        Vout_branch = Vout_all_branches[:, branch_idx : branch_idx + 1]
                        Vb_branch = Vb_all_branches[:, branch_idx : branch_idx + 1]

                        # Compute MI for this single branch with network weights
                        branch_results = self.compute_metrics(
                            E_branch, I_branch, Vout_branch, C, None, Vb_branch
                        )
                        all_results.append(branch_results)
                        layer_branch_results.append(branch_results)
                        branch_names.append(f"{layer_name}_branch_{branch_idx}")

                        # If LDA requested, compute MI for LDA-weighted versions
                        if self.compute_lda_weights and E_lin is not None:
                            E_branch_lin = E_lin[:, branch_idx : branch_idx + 1]
                            I_branch_lin = I_lin[:, branch_idx : branch_idx + 1]
                            Vb_branch_lin = Vb_lin[:, branch_idx : branch_idx + 1]
                            Vout_branch_lin = Vout_lin[:, branch_idx : branch_idx + 1]

                            branch_results_lin = self.compute_metrics(
                                E_branch_lin,
                                I_branch_lin,
                                Vout_branch_lin,  # Use LDA-computed Vout
                                C,
                                None,
                                Vb_branch_lin,
                            )
                            # Rename keys to indicate LDA versions
                            branch_results_lin_renamed = {}
                            for key, value in branch_results_lin.items():
                                if isinstance(value, dict):
                                    renamed_dict = {}
                                    for subkey, subval in value.items():
                                        # Replace E, I, Vb with E_lin, I_lin, Vb_lin in metric names
                                        # Handle both pairwise (;X)) and conditional (;X|C)) patterns
                                        new_subkey = (
                                            subkey.replace("(E;", "(E_lin;")
                                            .replace(";E)", ";E_lin)")
                                            .replace(";E|", ";E_lin|")
                                        )
                                        new_subkey = (
                                            new_subkey.replace("(I;", "(I_lin;")
                                            .replace(";I)", ";I_lin)")
                                            .replace(";I|", ";I_lin|")
                                        )
                                        new_subkey = (
                                            new_subkey.replace("(Vb;", "(Vb_lin;")
                                            .replace(";Vb)", ";Vb_lin)")
                                            .replace(";Vb|", ";Vb_lin|")
                                        )
                                        new_subkey = new_subkey.replace(
                                            "(E,I;", "(E_lin,I_lin;"
                                        )
                                        # Also rename Vout to Vout_lin
                                        # Handle both pairwise (;Vout)) and conditional (;Vout|C)) patterns
                                        new_subkey = (
                                            new_subkey.replace("(Vout;", "(Vout_lin;")
                                            .replace(";Vout)", ";Vout_lin)")
                                            .replace(";Vout|", ";Vout_lin|")
                                        )
                                        renamed_dict[new_subkey] = subval
                                    branch_results_lin_renamed[key] = renamed_dict
                                else:
                                    branch_results_lin_renamed[key] = value

                            # Merge LDA results into branch_results
                            for key, value in branch_results_lin_renamed.items():
                                if isinstance(value, dict) and key in branch_results:
                                    branch_results[key].update(value)

                        self.logger.debug(
                            f"  Branch {branch_idx + 1}/{n_branches} completed (took {time.time() - branch_start_time:.1f}s)"
                        )

                    # Compute layer statistics
                    if layer_branch_results:
                        layer_stats = self._compute_layer_statistics(
                            layer_branch_results
                        )
                        # Add has_synapses flag for plotting code to use
                        if layer_name in self.data_dict:
                            layer_stats["has_synapses"] = self.data_dict[
                                layer_name
                            ].get("has_synapses", True)
                            # Zero out E/I related MI values when layer has no synapses
                            # (e.g., soma with somatic_synapses=False)
                            if not layer_stats["has_synapses"]:
                                for key in list(layer_stats.keys()):
                                    # Zero out any metric involving E or I (but not Vb or Vout alone)
                                    if any(
                                        x in key
                                        for x in [
                                            "I(E;",
                                            "I(I;",
                                            "I(E_lin;",
                                            "I(I_lin;",
                                        ]
                                    ):
                                        layer_stats[key] = 0.0
                        layer_results[layer_name] = layer_stats

                    self.logger.info(
                        f"Layer {layer_name} completed: {n_branches} branches processed in {time.time() - layer_start_time:.1f}s"
                    )

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

                layers_to_process = [
                    (name, data)
                    for name, data in self.data_dict.items()
                    if "excitation" in data
                ]

                for layer_name, layer_data in tqdm(
                    layers_to_process,
                    desc="Processing layers",
                    leave=False,
                    ncols=100,
                ):
                    # Aggregate all branches within this layer using branch_aggregation
                    E_layer_data_raw = layer_data[
                        "excitation"
                    ]  # Shape: [n_samples, n_branches]
                    I_layer_data_raw = layer_data["inhibition"]
                    Vb_layer_data_raw = layer_data.get("branch_input")
                    Vout_layer_data = layer_data["output"]

                    # Ensure inhibitory data is available
                    E_layer_data, I_layer_data = self.ensure_inhibitory_data(
                        E_layer_data_raw, I_layer_data_raw
                    )

                    # Handle Vb data
                    if Vb_layer_data_raw is not None:
                        Vb_layer_data = Vb_layer_data_raw
                    else:
                        Vb_layer_data = torch.zeros_like(E_layer_data)

                    if self.branch_aggregation == "mean":
                        E_layer = (
                            E_layer_data.mean(dim=1).detach().cpu().numpy()
                        )  # [n_samples,]
                        I_layer = I_layer_data.mean(dim=1).detach().cpu().numpy()
                        Vb_layer = Vb_layer_data.mean(dim=1).detach().cpu().numpy()
                        Vout_layer = Vout_layer_data.mean(dim=1).detach().cpu().numpy()
                        # Reshape to 2D for MI computation
                        E_layer = E_layer.reshape(-1, 1)
                        I_layer = I_layer.reshape(-1, 1)
                        Vb_layer = Vb_layer.reshape(-1, 1)
                        Vout_layer = Vout_layer.reshape(-1, 1)
                    elif self.branch_aggregation == "multivariate":
                        # Multivariate: Keep all branches as multivariate input (no aggregation)
                        E_layer = (
                            E_layer_data.detach().cpu().numpy()
                        )  # [n_samples, n_branches]
                        I_layer = (
                            I_layer_data.detach().cpu().numpy()
                        )  # [n_samples, n_branches]
                        Vb_layer = (
                            Vb_layer_data.detach().cpu().numpy()
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
                        Vb_layer = (
                            Vb_layer_data.detach()
                            .cpu()
                            .numpy()
                            .flatten()
                            .reshape(-1, 1)
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
                        Vb_layer = Vb_layer_data.mean(dim=1).detach().cpu().numpy()
                        Vout_layer = Vout_layer_data.mean(dim=1).detach().cpu().numpy()
                        # Reshape to 2D for MI computation
                        E_layer = E_layer.reshape(-1, 1)
                        I_layer = I_layer.reshape(-1, 1)
                        Vb_layer = Vb_layer.reshape(-1, 1)
                        Vout_layer = Vout_layer.reshape(-1, 1)

                    # Compute MI for this layer
                    layer_results = self.compute_metrics(
                        E_layer, I_layer, Vout_layer, C, None, Vb_layer
                    )
                    all_results.append(layer_results)
                    layer_names.append(layer_name)

                # Average the results across layers
                if all_results:
                    results = self._average_mi_results(all_results)
                    results["layer_names"] = layer_names
                    results["num_layers"] = len(all_results)
                    results["computation_level"] = "layer_branch"

                    # Generate layer statistics for plotting (similar to single_branch mode)
                    layer_statistics = {}
                    for _i, (layer_name, layer_result) in enumerate(
                        zip(layer_names, all_results)
                    ):
                        # Convert individual layer result to statistics format expected by plotting
                        layer_stats = self._compute_layer_statistics([layer_result])

                        # Add has_synapses flag for plotting code to use
                        if layer_name in self.data_dict:
                            layer_stats["has_synapses"] = self.data_dict[
                                layer_name
                            ].get("has_synapses", True)
                            # Zero out E/I related MI values when layer has no synapses
                            # (e.g., soma with somatic_synapses=False)
                            if not layer_stats["has_synapses"]:
                                for key in list(layer_stats.keys()):
                                    # Zero out any metric involving E or I (but not Vb or Vout alone)
                                    if any(
                                        x in key
                                        for x in [
                                            "I(E;",
                                            "I(I;",
                                            "I(E_lin;",
                                            "I(I_lin;",
                                        ]
                                    ):
                                        layer_stats[key] = 0.0

                        layer_statistics[layer_name] = layer_stats

                    results["layer_statistics"] = layer_statistics
                    self.logger.info(
                        f"Generated layer statistics for {len(layer_statistics)} layers for plotting"
                    )
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
                layer_branch_input = []
                layer_output = []

                for layer_data in self.data_dict.values():
                    if "excitation" in layer_data:
                        # Aggregate within each layer first
                        if self.branch_aggregation == "mean":
                            layer_E = layer_data["excitation"].mean(
                                dim=1, keepdim=True
                            )  # [n_samples, 1]
                            layer_I = layer_data["inhibition"].mean(dim=1, keepdim=True)
                            layer_Vb_raw = layer_data.get("branch_input")
                            if layer_Vb_raw is not None:
                                layer_Vb = layer_Vb_raw.mean(dim=1, keepdim=True)
                            else:
                                layer_Vb = torch.zeros_like(layer_E)
                            layer_Vout = layer_data["output"].mean(dim=1, keepdim=True)
                        elif self.branch_aggregation == "multivariate":
                            # For network level with multivariate, keep branches separate for each layer
                            layer_E = layer_data[
                                "excitation"
                            ]  # [n_samples, n_branches]
                            layer_I = layer_data[
                                "inhibition"
                            ]  # [n_samples, n_branches]
                            layer_Vb_raw = layer_data.get("branch_input")
                            if layer_Vb_raw is not None:
                                layer_Vb = layer_Vb_raw
                            else:
                                layer_Vb = torch.zeros_like(layer_E)
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
                            layer_Vb_raw = layer_data.get("branch_input")
                            if layer_Vb_raw is not None:
                                layer_Vb = layer_Vb_raw.view(n_samples * n_branches, 1)
                            else:
                                layer_Vb = torch.zeros_like(layer_E)
                            layer_Vout = layer_data["output"].view(
                                n_samples * n_branches, 1
                            )
                        else:
                            # Default to mean
                            layer_E = layer_data["excitation"].mean(dim=1, keepdim=True)
                            layer_I = layer_data["inhibition"].mean(dim=1, keepdim=True)
                            layer_Vb_raw = layer_data.get("branch_input")
                            if layer_Vb_raw is not None:
                                layer_Vb = layer_Vb_raw.mean(dim=1, keepdim=True)
                            else:
                                layer_Vb = torch.zeros_like(layer_E)
                            layer_Vout = layer_data["output"].mean(dim=1, keepdim=True)

                        layer_excitation.append(layer_E)
                        layer_inhibition.append(layer_I)
                        layer_branch_input.append(layer_Vb)
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
                    Vb = torch.cat(layer_branch_input, dim=0).detach().cpu().numpy()
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
                    Vb = torch.cat(layer_branch_input, dim=1).detach().cpu().numpy()
                    Vout = torch.cat(layer_output, dim=1).detach().cpu().numpy()
                else:
                    # For mean/max/sum, concatenate across layers (all tensors are [n_samples, 1])
                    E = (
                        torch.cat(layer_excitation, dim=1).detach().cpu().numpy()
                    )  # [n_samples, n_layers]
                    I_var = torch.cat(layer_inhibition, dim=1).detach().cpu().numpy()
                    Vb = torch.cat(layer_branch_input, dim=1).detach().cpu().numpy()
                    Vout = torch.cat(layer_output, dim=1).detach().cpu().numpy()

                self.logger.info(
                    "Aggregated data shapes - "
                    f"E: {E.shape}, I: {I_var.shape}, Vb: {Vb.shape}, Vout: {Vout.shape}, C: {C.shape}"
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
                results = self.compute_metrics(E, I_var, Vout, C, Vinf, Vb)
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

            # Analysis completion summary
            total_analysis_time = time.time() - analysis_start_time
            self.logger.info("Information analysis completed successfully!")
            self.logger.info("Analysis summary:")
            self.logger.info(f"   - Total time: {total_analysis_time:.1f}s")
            self.logger.info(f"   - Computation level: {self.computation_level}")
            self.logger.info(f"   - Method: {self.method}")
            self.logger.info(f"   - Sample size: {results.get('n_samples', 'unknown')}")
            if "num_branches" in results:
                self.logger.info(f"   - Branches processed: {results['num_branches']}")
            if "num_layers" in results:
                self.logger.info(f"   - Layers processed: {results['num_layers']}")

            # Check if we need to add synthetic soma entry
            # This is only needed when somatic_synapses=False AND the soma layer
            # wasn't captured by forward hooks (shouldn't happen now that we fixed hooks)
            somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
            self.logger.info(f"Somatic synapses enabled: {somatic_synapses}")

            # Find if the actual soma layer is already in layer_statistics
            soma_already_captured = False
            if "layer_statistics" in results:
                for layer_name in results["layer_statistics"].keys():
                    if "branch_layers." in layer_name:
                        try:
                            # Find the highest index layer (soma)
                            branch_idx = int(
                                layer_name.split("branch_layers.")[1].split(".")[0]
                            )
                            # Check if this is a soma layer (no synapses)
                            layer_data = results["layer_statistics"][layer_name]
                            if layer_data.get("has_synapses") is False:
                                soma_already_captured = True
                                self.logger.info(
                                    f"Soma layer already captured: {layer_name}"
                                )
                                break
                        except (ValueError, IndexError):
                            continue

            # Only add synthetic soma if the real soma wasn't captured
            if (
                not somatic_synapses
                and "layer_statistics" in results
                and not soma_already_captured
            ):
                self.logger.info(
                    f"Adding synthetic soma entry to {len(results['layer_statistics'])} existing layers"
                )

                # Find the real soma layer to get its Vout information
                # The real soma is the layer with the highest branch index (corresponds to layer_idx=0)
                soma_vout_mean = 0.0
                soma_vout_std = 0.0

                # Find the layer with the highest branch index in layer_statistics
                max_branch_idx = -1
                real_soma_layer = None

                for layer_name, _layer_data in results["layer_statistics"].items():
                    if "branch_layers." in layer_name:
                        try:
                            branch_idx = int(
                                layer_name.split("branch_layers.")[1].split(".")[0]
                            )
                            if branch_idx > max_branch_idx:
                                max_branch_idx = branch_idx
                                real_soma_layer = layer_name
                        except (ValueError, IndexError):
                            continue

                # Extract I(Vout;C) and I(Vb;C) from the real soma layer
                # The soma still receives branch input from dendrites (Vb),
                # even when somatic_synapses=False (no E/I synapses at soma)
                soma_vb_mean = 0.0
                soma_vb_std = 0.0
                soma_vb_lin_mean = 0.0
                soma_vb_lin_std = 0.0
                soma_vb_vout_mean = 0.0
                soma_vb_vout_std = 0.0
                soma_vb_vout_c_mean = 0.0
                soma_vb_vout_c_std = 0.0
                soma_vb_lin_vout_mean = 0.0
                soma_vb_lin_vout_std = 0.0
                soma_vb_lin_vout_c_mean = 0.0
                soma_vb_lin_vout_c_std = 0.0
                # Vout_lin metrics
                soma_vout_lin_mean = 0.0
                soma_vout_lin_std = 0.0

                if real_soma_layer and real_soma_layer in results["layer_statistics"]:
                    real_soma_stats = results["layer_statistics"][real_soma_layer]
                    soma_vout_mean = real_soma_stats.get("I(Vout;C)_mean", 0.0)
                    soma_vout_std = real_soma_stats.get("I(Vout;C)_std", 0.0)
                    # Get Vb metrics from real soma layer - these represent dendrites feeding into soma
                    soma_vb_mean = real_soma_stats.get("I(Vb;C)_mean", 0.0)
                    soma_vb_std = real_soma_stats.get("I(Vb;C)_std", 0.0)
                    soma_vb_lin_mean = real_soma_stats.get("I(Vb_lin;C)_mean", 0.0)
                    soma_vb_lin_std = real_soma_stats.get("I(Vb_lin;C)_std", 0.0)
                    # Get Vb-related pairwise and conditional MI
                    soma_vb_vout_mean = real_soma_stats.get("I(Vb;Vout)_mean", 0.0)
                    soma_vb_vout_std = real_soma_stats.get("I(Vb;Vout)_std", 0.0)
                    soma_vb_vout_c_mean = real_soma_stats.get("I(Vb;Vout|C)_mean", 0.0)
                    soma_vb_vout_c_std = real_soma_stats.get("I(Vb;Vout|C)_std", 0.0)
                    soma_vb_lin_vout_mean = real_soma_stats.get(
                        "I(Vb_lin;Vout_lin)_mean", 0.0
                    )
                    soma_vb_lin_vout_std = real_soma_stats.get(
                        "I(Vb_lin;Vout_lin)_std", 0.0
                    )
                    soma_vb_lin_vout_c_mean = real_soma_stats.get(
                        "I(Vb_lin;Vout_lin|C)_mean", 0.0
                    )
                    soma_vb_lin_vout_c_std = real_soma_stats.get(
                        "I(Vb_lin;Vout_lin|C)_std", 0.0
                    )
                    # Get Vout_lin metrics from real soma layer
                    soma_vout_lin_mean = real_soma_stats.get("I(Vout_lin;C)_mean", 0.0)
                    soma_vout_lin_std = real_soma_stats.get("I(Vout_lin;C)_std", 0.0)
                    self.logger.info(f"Found real soma layer: {real_soma_layer}")
                    self.logger.info(
                        f"Using real soma I(Vout;C): {soma_vout_mean:.4f} ± {soma_vout_std:.4f}"
                    )
                    self.logger.info(
                        f"Using real soma I(Vb;C): {soma_vb_mean:.4f} ± {soma_vb_std:.4f}"
                    )
                else:
                    self.logger.warning(
                        "Could not find real soma layer for I(Vout;C) computation"
                    )

                # Add synthetic soma entry with appropriate values
                # Note: E/I synapses are zero, but Vb (branch input from dendrites) carries information
                soma_entry = {
                    "I(E;C)_mean": 0.0,  # Zero - no excitatory synapses
                    "I(E;C)_std": 0.0,
                    "I(I;C)_mean": 0.0,  # Zero - no inhibitory synapses
                    "I(I;C)_std": 0.0,
                    # Dendrites still feed into soma - use real soma layer's Vb values
                    "I(Vb;C)_mean": soma_vb_mean,
                    "I(Vb;C)_std": soma_vb_std,
                    "I(Vout;C)_mean": soma_vout_mean,  # Real computed value from soma output
                    "I(Vout;C)_std": soma_vout_std,  # Real computed std from soma output
                    "I(E,I;C)_mean": 0.0,  # Zero - no synapses
                    "I(E,I;C)_std": 0.0,
                    # LDA metrics - E/I are zero (no synapses), but Vb_lin carries information
                    "I(E_lin;C)_mean": 0.0,
                    "I(E_lin;C)_std": 0.0,
                    "I(I_lin;C)_mean": 0.0,
                    "I(I_lin;C)_std": 0.0,
                    "I(Vb_lin;C)_mean": soma_vb_lin_mean,
                    "I(Vb_lin;C)_std": soma_vb_lin_std,
                    "I(Vout_lin;C)_mean": soma_vout_lin_mean,
                    "I(Vout_lin;C)_std": soma_vout_lin_std,
                    "I(E_lin,I_lin;C)_mean": 0.0,
                    "I(E_lin,I_lin;C)_std": 0.0,
                    # Pairwise MI
                    "I(E;I)_mean": 0.0,  # Zero - no synapses
                    "I(E;I)_std": 0.0,
                    "I(E;Vout)_mean": 0.0,  # Zero - no excitatory synapses
                    "I(E;Vout)_std": 0.0,
                    "I(I;Vout)_mean": 0.0,  # Zero - no inhibitory synapses
                    "I(I;Vout)_std": 0.0,
                    "I(E;Vb)_mean": 0.0,
                    "I(E;Vb)_std": 0.0,
                    "I(I;Vb)_mean": 0.0,
                    "I(I;Vb)_std": 0.0,
                    # Vb;Vout is meaningful - dendrites to soma output
                    "I(Vb;Vout)_mean": soma_vb_vout_mean,
                    "I(Vb;Vout)_std": soma_vb_vout_std,
                    # LDA pairwise MI - use Vout_lin for consistency
                    "I(E_lin;I_lin)_mean": 0.0,
                    "I(E_lin;I_lin)_std": 0.0,
                    "I(E_lin;Vout_lin)_mean": 0.0,
                    "I(E_lin;Vout_lin)_std": 0.0,
                    "I(I_lin;Vout_lin)_mean": 0.0,
                    "I(I_lin;Vout_lin)_std": 0.0,
                    "I(E_lin;Vb_lin)_mean": 0.0,
                    "I(E_lin;Vb_lin)_std": 0.0,
                    "I(I_lin;Vb_lin)_mean": 0.0,
                    "I(I_lin;Vb_lin)_std": 0.0,
                    # Vb_lin;Vout_lin is meaningful
                    "I(Vb_lin;Vout_lin)_mean": soma_vb_lin_vout_mean,
                    "I(Vb_lin;Vout_lin)_std": soma_vb_lin_vout_std,
                    # Conditional MI
                    "I(E;I|C)_mean": 0.0,  # Zero - no synapses
                    "I(E;I|C)_std": 0.0,
                    "I(E;Vout|C)_mean": 0.0,  # Zero - no excitatory synapses
                    "I(E;Vout|C)_std": 0.0,
                    "I(I;Vout|C)_mean": 0.0,  # Zero - no inhibitory synapses
                    "I(I;Vout|C)_std": 0.0,
                    "I(E;Vb|C)_mean": 0.0,
                    "I(E;Vb|C)_std": 0.0,
                    "I(I;Vb|C)_mean": 0.0,
                    "I(I;Vb|C)_std": 0.0,
                    # Vb;Vout|C is meaningful
                    "I(Vb;Vout|C)_mean": soma_vb_vout_c_mean,
                    "I(Vb;Vout|C)_std": soma_vb_vout_c_std,
                    # LDA conditional MI - use Vout_lin for consistency
                    "I(E_lin;I_lin|C)_mean": 0.0,
                    "I(E_lin;I_lin|C)_std": 0.0,
                    "I(E_lin;Vout_lin|C)_mean": 0.0,
                    "I(E_lin;Vout_lin|C)_std": 0.0,
                    "I(I_lin;Vout_lin|C)_mean": 0.0,
                    "I(I_lin;Vout_lin|C)_std": 0.0,
                    "I(E_lin;Vb_lin|C)_mean": 0.0,
                    "I(E_lin;Vb_lin|C)_std": 0.0,
                    "I(I_lin;Vb_lin|C)_mean": 0.0,
                    "I(I_lin;Vb_lin|C)_std": 0.0,
                    # Vb_lin;Vout_lin|C is meaningful
                    "I(Vb_lin;Vout_lin|C)_mean": soma_vb_lin_vout_c_mean,
                    "I(Vb_lin;Vout_lin|C)_std": soma_vb_lin_vout_c_std,
                }
                # Add soma entry to layer_statistics
                results["layer_statistics"]["synthetic_soma"] = soma_entry
                self.logger.info(
                    f"Added synthetic soma with real I(Vout;C)={soma_vout_mean:.4f}. Total layers now: {len(results['layer_statistics'])}"
                )
            elif not somatic_synapses:
                self.logger.warning(
                    "somatic_synapses=False but no layer_statistics found in results"
                )

            # Compute variance analysis for plotting (AFTER synthetic soma is added)
            self.logger.info("Computing variance analysis for plotting...")
            variance_analysis = self._compute_variance_analysis(results)
            results["variance_analysis"] = variance_analysis
            self.logger.info(
                f"Variance analysis computed with {len(variance_analysis.get('layer_variances', {}))} layers"
            )

            # Compute LDA vs trained performance comparison if LDA weights are enabled
            if self.compute_lda_weights and save_path is not None:
                try:
                    # Use the inputs/labels already loaded in the analyze method
                    lda_comparison = self.compare_lda_vs_trained_performance(
                        model, inputs, labels, device
                    )
                    results["lda_vs_trained_performance"] = lda_comparison

                    # Plot the comparison if LDA accuracy was computed
                    if lda_comparison.get("lda_accuracy") is not None:
                        from dendritic_modeling.plotting.visualizations.information_plots import (
                            plot_lda_vs_trained_performance,
                        )

                        plot_lda_vs_trained_performance(
                            lda_comparison, save_path, self.logger
                        )
                        self.logger.info(
                            "LDA vs trained performance comparison completed!"
                        )
                    else:
                        self.logger.warning(
                            "LDA accuracy not available, skipping performance plot"
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error computing LDA vs trained performance: {e}"
                    )
                    import traceback

                    traceback.print_exc()

            # Save results
            if save_path is not None:
                self.logger.info(f"Saving results to {save_path}/{filename}")
                save_dict(results, save_path, filename)

                # Generate main summary
                summary = self.get_summary(results)
                summary_path = os.path.join(save_path, f"{filename}_summary.txt")
                with open(summary_path, "w") as f:
                    f.write(summary)

                # Generate individual summaries for each layer's E and I networks
                if "per_layer_analysis" in results:
                    self._save_per_layer_ei_summaries(
                        results["per_layer_analysis"], save_path, filename
                    )

                # Generate plots for information analysis
                self._generate_information_plots(results, save_path, somatic_synapses)

                self.logger.info("Results saved successfully!")

            # Always return results for plotting
            return results

        except Exception as e:
            self.logger.error(f"Error in information analysis: {e}")
            traceback.print_exc()

    def compute_class_mi(self, activations, labels):
        return self.estimators[self.method].mutual_information(activations, labels)

    def compute_conditional_mi(self, activations, labels, condition):
        return self.estimators[self.method].conditional_mutual_information(
            activations, labels, condition
        )

    def compare_lda_vs_trained_performance(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device = None,
    ) -> dict:
        """Compare classification performance using trained weights vs LDA-optimal weights.

        This method uses the data already collected in self.data_dict from the main analysis.
        It computes:
        1. Performance with trained network weights (original model)
        2. Performance with LDA-optimized input aggregation (Vout_lin passed through decoder)

        Args:
            model: Trained model with encoder, core_network, and decoder
            inputs: Input tensor (already processed in main analysis)
            labels: Label tensor (already processed in main analysis)
            device: Device to run computations on

        Returns:
            Dictionary with:
            - trained_accuracy: Accuracy with trained weights
            - lda_accuracy: Accuracy with LDA-optimal aggregation
            - accuracy_difference: LDA - Trained accuracy
        """
        from sklearn.metrics import accuracy_score

        if device is None:
            device = next(model.parameters()).device

        # Ensure model is in eval mode (same state used for all comparisons)
        model.eval()

        self.logger.info("Computing LDA vs trained weights performance comparison...")
        self.logger.info(
            "Note: Using the SAME model for both trained and LDA predictions. "
            "Only the intermediate representations differ."
        )

        # Move labels to CPU/numpy
        if hasattr(labels, "cpu"):
            labels_np = labels.cpu().numpy()
        elif hasattr(labels, "numpy"):
            labels_np = labels.numpy()
        else:
            labels_np = np.array(labels)

        # Get trained model predictions
        with torch.no_grad():
            trained_outputs = model(inputs.to(device))
            trained_preds = trained_outputs.argmax(dim=1).cpu().numpy()

        trained_accuracy = accuracy_score(labels_np, trained_preds)

        # Compute CHAINED LDA through all layers
        # Sort layers by branch index to process in order
        layer_names = list(self.data_dict.keys())

        # Extract and sort layers by branch index
        layers_with_idx = []
        for layer_name in layer_names:
            if "branch_layers" in layer_name:
                try:
                    idx = int(layer_name.split("branch_layers.")[1].split(".")[0])
                    layers_with_idx.append((idx, layer_name))
                except (ValueError, IndexError):
                    continue
        layers_with_idx.sort(key=lambda x: x[0])

        # Chain LDA through all layers
        prev_vout_lin = None  # Output from previous layer (None for first layer)
        final_vout_lin = None

        for _layer_idx, layer_name in layers_with_idx:
            layer_data = self.data_dict[layer_name]
            if "excitation" not in layer_data:
                continue

            # Compute LDA signals with chained branch input
            lda_result = self.compute_lda_aggregated_signals_chained(
                layer_data, labels_np, prev_vout_lin
            )

            if lda_result[0] is not None:
                _E_lin, _I_lin, _Vb_lin, Vout_lin = lda_result
                prev_vout_lin = Vout_lin  # Use as input for next layer
                final_vout_lin = Vout_lin
            else:
                # Fall back to trained network output
                prev_vout_lin = layer_data["output"].detach().cpu().numpy()
                final_vout_lin = prev_vout_lin

        # Use LDA outputs to compute accuracy
        # Note: final_vout_lin is already the LDA-computed output of the soma layer
        # (since the loop above processes ALL layers including soma).
        # We only need to apply the decoder if there is one.
        lda_accuracy = None
        lda_preds = None

        if final_vout_lin is not None:
            with torch.no_grad():
                vout_tensor = torch.tensor(final_vout_lin, dtype=torch.float32).to(
                    device
                )

                if hasattr(model, "decoder") and model.decoder is not None:
                    # Use decoder if available
                    lda_logits = model.decoder(vout_tensor)
                else:
                    # No decoder - final_vout_lin is already the soma layer's output
                    # (LDA was applied to all layers including soma in the loop above)
                    # Use it directly as logits
                    lda_logits = vout_tensor

                lda_preds = lda_logits.argmax(dim=1).cpu().numpy()
                lda_accuracy = accuracy_score(labels_np, lda_preds)
        results = {
            "trained_accuracy": trained_accuracy,
            "lda_accuracy": lda_accuracy,
            "trained_predictions": trained_preds.tolist(),
            "lda_predictions": lda_preds.tolist() if lda_preds is not None else None,
            "accuracy_difference": (
                lda_accuracy - trained_accuracy if lda_accuracy is not None else None
            ),
        }

        if lda_accuracy is not None:
            self.logger.info(
                f"Difference (LDA - Trained): {results['accuracy_difference']:+.4f}"
            )

        return results

    def _generate_information_plots(
        self, results: dict, save_path: str, somatic_synapses: bool
    ):
        """Generate plots for information analysis results."""
        try:
            # Initialize plotter for enhanced visualizations
            # plotter = InformationAnalysisPlotter()  # Unused variable

            # Generate dendritic depth plots based on enabled analysis modes
            if results and "layer_statistics" in results:
                per_layer_enabled = results.get("per_layer_analysis_enabled", False)
                per_einet_enabled = results.get("per_einet_analysis_enabled", False)

                if per_layer_enabled and per_einet_enabled:
                    self.logger.info(
                        "Both per_layer and per_einet analysis enabled - generating detailed comparison plots..."
                    )
                    # Generate structured plots grouped by major layer and network type
                    plot_structured_ei_network_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )
                elif per_layer_enabled and not per_einet_enabled:
                    self.logger.info(
                        "per_layer analysis enabled, per_einet disabled - generating layer-wise plots..."
                    )
                    # Aggregate E/I networks, but keep layers separate
                    plot_layer_wise_aggregated_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )
                elif not per_layer_enabled and per_einet_enabled:
                    self.logger.info(
                        "per_einet analysis enabled, per_layer disabled - generating network-wise plots..."
                    )
                    # Aggregate layers, but keep E/I networks separate
                    plot_network_wise_aggregated_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )
                else:
                    self.logger.info(
                        "Both per_layer and per_einet analysis disabled - generating overall aggregated plots..."
                    )
                    # Aggregate both layers and E/I networks
                    plot_fully_aggregated_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )

            # Generate info vs EI depth per dendritic layer plots if we have layer statistics
            self.logger.info(
                f"Checking for layer_statistics: results exists={results is not None}"
            )
            if results:
                self.logger.info(f"Results keys: {list(results.keys())}")
                self.logger.info(
                    f"layer_statistics in results: {'layer_statistics' in results}"
                )
                if "layer_statistics" in results:
                    self.logger.info(
                        f"layer_statistics length: {len(results['layer_statistics'])}"
                    )

            if (
                results
                and "layer_statistics" in results
                and len(results["layer_statistics"]) > 0
            ):
                self.logger.info(
                    f"Calling plot_info_vs_ei_depth_per_dendritic_layer with {len(results['layer_statistics'])} layer statistics"
                )
                plot_info_vs_ei_depth_per_dendritic_layer(
                    results["layer_statistics"],
                    save_path=save_path,
                    logger=self.logger,
                    somatic_synapses=somatic_synapses,
                )
            else:
                self.logger.warning(
                    "Skipping plot_info_vs_ei_depth_per_dendritic_layer - no layer statistics found"
                )

            # Generate variance analysis plots if variance analysis was computed
            if results and "variance_analysis" in results:
                try:
                    self.logger.info(
                        "Generating information variance analysis plots..."
                    )
                    self.logger.info(
                        f"Variance analysis keys: {list(results['variance_analysis'].keys())}"
                    )
                    from dendritic_modeling.plotting.visualizations.information_plots import (
                        plot_information_variance_analysis,
                    )

                    plot_information_variance_analysis(
                        results["variance_analysis"],
                        save_path=save_path,
                        somatic_synapses=somatic_synapses,
                        logger=self.logger,
                    )
                    self.logger.info("Variance plots completed successfully!")
                except Exception as e:
                    self.logger.error(f"Error generating variance plots: {e}")
                    import traceback

                    traceback.print_exc()
            else:
                self.logger.warning(
                    "No variance_analysis found in results - skipping variance plots"
                )

        except Exception as e:
            self.logger.error(f"Error generating information plots: {e}")
