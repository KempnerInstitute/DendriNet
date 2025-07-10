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

from dendritic_modeling.config import InformationAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.plotting.analysis import plot_information_analysis
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.information.estimators import (
    EstimatorConfig,
    InformationEstimatorFactory,
)
from dendritic_modeling.utils.information.pid import (
    check_pid_available,
    pid_synergy_redundancy,
)


class InformationAnalyzer:
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
        self.logger = logging.getLogger(self.__class__.__name__)

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

        self.include_vinf = getattr(params, "include_vinf", True)  # Default to True for backward compatibility
        self.normalize_mi = getattr(params, "normalize_mi", False)
        self.verbose = getattr(params, "verbose", False)

        self.compute_per_branch = getattr(params, "compute_per_branch", False)
        self.branch_aggregation = getattr(params, "branch_aggregation", "mean")

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        # Create the estimator
        estimator_config = EstimatorConfig(
            method=self.method,
            n_neighbors=self.n_neighbors,
            n_bins=self.n_bins,
            copula_type=self.copula_type,
            verbose=self.verbose,
        )
        self.estimator = InformationEstimatorFactory.create(estimator_config)

        # Check if PID is available
        self.pid_available = False
        if self.compute_pid:
            try:
                check_pid_available()
                self.pid_available = True
                self.logger.info("PID computation available")
            except ImportError:
                self.logger.warning("PID package not available. PID metrics will be skipped.")

    def _compute_basic_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None
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
        I: np.ndarray,
        Vout: np.ndarray,
        Vinf: Optional[np.ndarray] = None
    ) -> dict[str, float]:
        """Compute pairwise mutual information between variables."""
        metrics = {}
        # I(E;I)
        metrics["I(E;I)"] = self.estimator.mutual_information(E, I)
        # I(E;Vout)
        metrics["I(E;Vout)"] = self.estimator.mutual_information(E, Vout)
        # I(I;Vout)
        metrics["I(I;Vout)"] = self.estimator.mutual_information(I, Vout)
        if Vinf is not None and self.include_vinf:
            # I(E;Vinf)
            metrics["I(E;Vinf)"] = self.estimator.mutual_information(E, Vinf)
            # I(I;Vinf)
            metrics["I(I;Vinf)"] = self.estimator.mutual_information(I, Vinf)
            # I(Vout;Vinf)
            metrics["I(Vout;Vinf)"] = self.estimator.mutual_information(Vout, Vinf)
        return metrics

    def _compute_conditional_mi(
        self,
        E: np.ndarray,
        I: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None
    ) -> dict[str, float]:
        """Compute conditional mutual information given C."""
        metrics = {}
        # I(E;I|C)
        metrics["I(E;I|C)"] = self.estimator.conditional_mutual_information(
            E, I, C.reshape(-1, 1)
        )
        # I(E;Vout|C)
        metrics["I(E;Vout|C)"] = self.estimator.conditional_mutual_information(
            E, Vout, C.reshape(-1, 1)
        )
        # I(I;Vout|C)
        metrics["I(I;Vout|C)"] = self.estimator.conditional_mutual_information(
            I, Vout, C.reshape(-1, 1)
        )
        if Vinf is not None and self.include_vinf:
            # I(E;Vinf|C)
            metrics["I(E;Vinf|C)"] = self.estimator.conditional_mutual_information(
                E, Vinf, C.reshape(-1, 1)
            )
            # I(I;Vinf|C)
            metrics["I(I;Vinf|C)"] = self.estimator.conditional_mutual_information(
                I, Vinf, C.reshape(-1, 1)
            )
        return metrics

    def _compute_pid(
        self,
        E: np.ndarray,
        I: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray
    ) -> dict[str, Any]:
        """Compute Partial Information Decomposition."""
        pid_results = {}

        # Prepare data for PID (may need binarization)
        if self.pid_binarize:
            E_pid = self._binarize(E)
            I_pid = self._binarize(I)
            Vout_pid = self._binarize(Vout)
            C_pid = self._binarize(C.reshape(-1, 1))
        else:
            E_pid = E
            I_pid = I
            Vout_pid = Vout
            C_pid = C.reshape(-1, 1)

        def _pid_helper(
            source1_array: np.ndarray,
            source1_name: str,
            source2_array: np.ndarray,
            source2_name: str,
            target_array: np.ndarray,
            target_name: str
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
                    f"unique_{source1_name}": self.estimator.mutual_information(
                        source1_array, target_array
                    ) - redundancy,
                    f"unique_{source2_name}": self.estimator.mutual_information(
                        source2_array, target_array
                    ) - redundancy,
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
        I: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None
    ):
        """Validate input arrays."""
        n_samples = len(E)

        if len(I) != n_samples:
            raise ValueError(f"I must have same number of samples as E: {len(I)} vs {n_samples}")
        if len(Vout) != n_samples:
            raise ValueError(f"Vout must have same number of samples as E: {len(Vout)} vs {n_samples}")
        if len(C) != n_samples:
            raise ValueError(f"C must have same number of samples as E: {len(C)} vs {n_samples}")
        if Vinf is not None and len(Vinf) != n_samples:
            raise ValueError(f"Vinf must have same number of samples as E: {len(Vinf)} vs {n_samples}")

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D."""
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def get_summary(
        self,
        results: dict[str, dict[str, Union[dict, float]]]
    ) -> str:
        """Generate a summary string of the results."""
        summary_lines = [
            "Information Analysis Summary",
            f"Method: {results['method']}",
            f"Samples: {results['n_samples']}",
            ""
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

    def compute_metrics(
        self,
        E: np.ndarray,
        I: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None
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
        self._validate_inputs(E, I, Vout, C, Vinf)

        # Ensure all arrays are properly shaped
        E = self._ensure_2d(E)
        I_var = self._ensure_2d(I)
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
            results["basic_mi"] = self._compute_basic_mi(
                E, I, Vout, C, Vinf
            )

        # 2. Pairwise mutual information
        if self.compute_pairwise_mi:
            self.logger.info("Computing pairwise MI metrics...")
            results["pairwise_mi"] = self._compute_pairwise_mi(
                E, I, Vout, Vinf
            )

        # 3. Conditional mutual information
        if self.compute_conditional_mi:
            self.logger.info("Computing conditional MI metrics...")
            results["conditional_mi"] = self._compute_conditional_mi(
                E, I, Vout, C, Vinf
            )

        # 4. PID decomposition
        if self.compute_pid and self.pid_available:
            self.logger.info("Computing PID metrics...")
            results["pid"] = self._compute_pid(
                E, I, Vout, C
            )

        return results

    def _average_mi_results(
        self,
        results_list: list[dict[str, dict[str, Union[dict, float]]]]
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

    def attach_forward_hooks(self, model: BaseModel):
        """Attach forward hooks to the model."""
        # Initalize data_dict attribute
        self.data_dict = {}

        handles = []
        for name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                if (
                    module.branch_excitation is not None
                    and module.branch_inhibition is not None
                ):
                    self.data_dict[name] = {}
                    module._name = name
                    handle = module.register_forward_hook(self.forward_hook)
                    handles.append(handle)

        return handles

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: torch.Tensor
    ):
        """Forward hook to collect branch data."""
        excitatory_input, inhibitory_input, _ = input
        self.data_dict[module._name]["excitation"] = \
            module.branch_excitation(excitatory_input)
        self.data_dict[module._name]["inhibition"] = \
            module.branch_inhibition(inhibitory_input)
        self.data_dict[module._name]["output"] = output

    def remove_forward_hooks(
        self,
        handles: list[torch.utils.hooks.RemovableHandle]
    ):
        """Remove forward hooks from the model."""
        for handle in handles:
            handle.remove()

    def analyze(
        self,
        model_or_E=None,
        test_dataset_or_I=None,
        device_or_Vout=None,
        save_path_or_C=None,
        filename_or_Vinf=None,
        *,
        model=None,
        test_dataset=None,
        device=None,
        save_path=None,
        filename=None
    ):
        """Run integrated information analysis on the model or compute metrics from arrays.

        This method supports both the new API (model-based) and legacy API (array-based).

        New API (using keyword arguments):
            analyze(model=model, test_dataset=test_dataset, device=device, save_path=save_path, filename=filename)

        New API (using positional arguments):
            analyze(model, test_dataset, device, save_path, filename)

        Legacy API (for backward compatibility):
            analyze(E, I, Vout, C, Vinf=None)
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
            # Legacy API: analyze(E, I, Vout, C, Vinf=None)
            E = model_or_E
            I_var = test_dataset_or_I
            Vout = device_or_Vout
            C = save_path_or_C
            Vinf = filename_or_Vinf

            return self.compute_metrics(E, I, Vout, C, Vinf)

        # New API: analyze(model, test_dataset, device, save_path, filename)
        model = model_or_E
        test_dataset = test_dataset_or_I
        device = device_or_Vout
        save_path = save_path_or_C
        filename = filename_or_Vinf

        try:
            # Check if model has ExcitationInhibitionNetwork
            if not isinstance(model.core_network, ExcitationInhibitionNetwork):
                self.logger.warning("Model does not have ExcitationInhibitionNetwork. Skipping information analysis.")
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
                self.logger.info(f"Sampling {self.max_samples} samples from {len(inputs)} for faster computation")
                indices = torch.randperm(len(inputs))[:self.max_samples]
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

            if self.compute_per_branch:
                # Option 1: Compute MI for each branch separately and take the mean
                self.logger.info("Computing MI for each branch separately...")

                all_results = []
                branch_names = []

                for branch_name, branch_data in self.data_dict.items():
                    if "excitation" not in branch_data:
                        continue

                    # Get single branch data
                    E_branch = branch_data["excitation"].detach().cpu().numpy()
                    I_branch = branch_data["inhibition"].detach().cpu().numpy()
                    Vout_branch = branch_data["output"].detach().cpu().numpy()

                    # Compute MI for this single branch
                    branch_results = self.compute_metrics(
                        E_branch, I_branch, Vout_branch, C, None
                    )
                    all_results.append(branch_results)
                    branch_names.append(branch_name)

                # Average the results across branches
                if all_results:
                    results = self._average_mi_results(all_results)
                    results["branch_names"] = branch_names
                    results["num_branches"] = len(all_results)
                else:
                    self.logger.warning("No branch results to average")
                    return

            else:
                # Option 2: Average branch activations first, then compute MI
                self.logger.info(
                    f"Aggregating branches using method: {self.branch_aggregation}"
                )

                # Collect all branch data
                excitation_list: list[torch.Tensor] = []
                inhibition_list: list[torch.Tensor] = []
                output_list: list[torch.Tensor] = []

                for branch_data in self.data_dict.values():
                    if "excitation" in branch_data:
                        excitation_list.append(branch_data["excitation"])
                    if "inhibition" in branch_data:
                        inhibition_list.append(branch_data["inhibition"])
                    if "output" in branch_data:
                        output_list.append(branch_data["output"])

                if not excitation_list:
                    self.logger.warning("No excitatory activations collected. Skipping information analysis.")
                    return

                # Aggregate activations based on method
                if self.branch_aggregation == "mean":
                    # Average across branches
                    E = torch.stack(excitation_list, dim=0).mean(dim=0).detach().cpu().numpy()
                    I_var = torch.stack(inhibition_list, dim=0).mean(dim=0).detach().cpu().numpy()
                    Vout = torch.stack(output_list, dim=0).mean(dim=0).detach().cpu().numpy()
                elif self.branch_aggregation == "max":
                    # Max pooling across branches
                    E = torch.stack(excitation_list, dim=0).max(dim=0)[0].detach().cpu().numpy()
                    I_var = torch.stack(inhibition_list, dim=0).max(dim=0)[0].detach().cpu().numpy()
                    Vout = torch.stack(output_list, dim=0).max(dim=0)[0].detach().cpu().numpy()
                elif self.branch_aggregation == "sum":
                    # Sum across branches
                    E = torch.stack(excitation_list, dim=0).sum(dim=0).detach().cpu().numpy()
                    I_var = torch.stack(inhibition_list, dim=0).sum(dim=0).detach().cpu().numpy()
                    Vout = torch.stack(output_list, dim=0).sum(dim=0).detach().cpu().numpy()
                elif self.branch_aggregation == "first":
                    # Only use first branch
                    E = excitation_list[0].detach().cpu().numpy()
                    I_var = inhibition_list[0].detach().cpu().numpy()
                    Vout = output_list[0].detach().cpu().numpy()
                else:
                    self.logger.warning(
                        f"Unknown branch aggregation method: "
                        f"{self.branch_aggregation}. "
                        "Using mean."
                    )
                    E = torch.stack(excitation_list, dim=0).mean(dim=0).detach().cpu().numpy()
                    I_var = torch.stack(inhibition_list, dim=0).mean(dim=0).detach().cpu().numpy()
                    Vout = torch.stack(output_list, dim=0).mean(dim=0).detach().cpu().numpy()

                self.logger.info(
                    "Aggregated data shapes - "
                    f"E: {E.shape}, I: {I.shape}, Vout: {Vout.shape}"
                )

                # Get soma activation (Vinf) if available
                Vinf = None
                # For now, we'll use the final model output as Vinf
                with torch.no_grad():
                    final_output: torch.Tensor = model(inputs)
                    if hasattr(model, 'final_fc'):
                        # Model has final classification layer
                        Vinf = final_output.detach().cpu().numpy()

                # Run information analysis on aggregated data
                results = self.compute_metrics(E, I, Vout, C, Vinf)
                results["aggregation_method"] = self.branch_aggregation
                results["num_branches_aggregated"] = len(excitation_list)

            # Save results
            os.makedirs(save_path, exist_ok=True)
            save_dict(results, save_path, filename)

            # Also save a summary report
            summary = self.get_summary(results)
            summary_path = os.path.join(save_path, f"{filename}_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(summary)

            self.logger.info(f"Information analysis completed. Results saved to {save_path}")

            # Generate information plots
            try:
                self.logger.info("Generating information analysis plots...")

                plot_save_path = os.path.join(save_path, "plots")
                os.makedirs(plot_save_path, exist_ok=True)

                # Generate all plots
                figs = plot_information_analysis(results, save_path=plot_save_path, filename_prefix=filename)

                self.logger.info(f"Generated {len(figs)} information analysis plots in {plot_save_path}")

                # Close all figures to free memory
                import matplotlib.pyplot as plt
                for fig in figs:
                    plt.close(fig)

            except Exception as e:
                self.logger.warning(f"Could not generate information plots: {e}")

        except Exception as e:
            self.logger.error(f"Error in information analysis: {e}")
            traceback.print_exc()
