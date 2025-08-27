"""
Correlation analysis for studying how EI networks learn input correlations.

This module provides hierarchical correlation analysis similar to the information analysis,
computing correlations at multiple levels: network-wide, per-layer, per-EI network, and per-branch.
"""

import logging
import os
import traceback
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config import CorrelationAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


class CorrelationAnalyzer(AbstractAnalyzer):
    """Analyze synaptic correlations and tuning properties in EI networks with hierarchical structure."""

    def __init__(self, params: CorrelationAnalysisParams):
        """
        Initialize correlation analyzer with hierarchical analysis capabilities.

        Args:
            params: Configuration parameters for correlation analysis
        """
        super().__init__()
        self.logger = logger
        self.params = params
        
        # Configuration parameters
        self.compute_synaptic_correlations = getattr(params, "compute_synaptic_correlations", True)
        self.compute_branch_correlations = getattr(params, "compute_branch_correlations", True)
        self.compute_conditional_correlations = getattr(params, "compute_conditional_correlations", True)
        self.compute_tuning_curves = getattr(params, "compute_tuning_curves", True)
        
        # Analysis levels
        self.per_layer_analysis = getattr(params, "per_layer_analysis", True)
        self.per_einet_analysis = getattr(params, "per_einet_analysis", True)
        self.per_branch_analysis = getattr(params, "per_branch_analysis", False)
        
        # Computation parameters
        self.max_samples = getattr(params, "max_samples", None)
        self.computation_level = getattr(params, "computation_level", "layer_branch")
        
        self.logger.info(f"Initialized CorrelationAnalyzer with computation_level: {self.computation_level}")

    def attach_forward_hooks(self, model: BaseModel):
        """Attach forward hooks to collect activations (same as information analysis)."""
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
                    
        self.logger.info(f"Attached hooks to {len(handles)} DendriticBranchLayers")
        return handles
        
    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ):
        """Forward hook to collect branch data (same as information analysis)."""
        excitatory_input, inhibitory_input, _ = input
        
        # Get the outputs from TopKLinear layers
        excitation_per_branch = module.branch_excitation(excitatory_input)
        inhibition_per_branch = module.branch_inhibition(inhibitory_input)
        
        # Store per-branch data
        self.data_dict[module._name]["excitation"] = excitation_per_branch
        self.data_dict[module._name]["inhibition"] = inhibition_per_branch
        self.data_dict[module._name]["output"] = output
        # Also store raw inputs if available
        self.data_dict[module._name]["excitatory_input"] = excitatory_input
        self.data_dict[module._name]["inhibitory_input"] = inhibitory_input
        
    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        for handle in handles:
            handle.remove()

    def _compute_correlation_matrix(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute correlation matrix between X and Y (or within X if Y is None).
        
        Args:
            X: First set of variables (n_samples, n_features_X)
            Y: Second set of variables (n_samples, n_features_Y) or None
            
        Returns:
            Correlation matrix of shape (n_features_X, n_features_Y) or (n_features_X, n_features_X)
        """
        if Y is None:
            # Self-correlation (e.g., E-E or I-I)
            return np.corrcoef(X.T)
        else:
            # Cross-correlation (e.g., E-I)
            n_x = X.shape[1]
            n_y = Y.shape[1]
            corr_matrix = np.zeros((n_x, n_y))
            
            for i in range(n_x):
                for j in range(n_y):
                    if np.std(X[:, i]) > 0 and np.std(Y[:, j]) > 0:
                        corr_matrix[i, j] = pearsonr(X[:, i], Y[:, j])[0]
                    else:
                        corr_matrix[i, j] = 0.0
                        
            return corr_matrix
            
    def _compute_conditional_correlations(
        self, 
        X: np.ndarray, 
        Y: Optional[np.ndarray], 
        labels: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute correlations conditioned on class labels.
        
        Args:
            X: First set of variables
            Y: Second set of variables (or None for self-correlation)
            labels: Class labels
            
        Returns:
            Dictionary with total and per-class correlations
        """
        unique_labels = np.unique(labels)
        results = {}
        
        # Total correlation (across all data)
        results["total"] = self._compute_correlation_matrix(X, Y)
        
        # Per-class correlations
        per_class_corrs = []
        for label in unique_labels:
            mask = labels == label
            X_class = X[mask]
            Y_class = Y[mask] if Y is not None else None
            
            results[f"class_{label}"] = self._compute_correlation_matrix(X_class, Y_class)
            per_class_corrs.append(results[f"class_{label}"])
            
        # Noise correlation (average within-class correlation)
        results["noise"] = np.mean(per_class_corrs, axis=0)
        
        # Signal correlation (between-class difference)
        if len(unique_labels) == 2:
            results["signal"] = results[f"class_{unique_labels[1]}"] - results[f"class_{unique_labels[0]}"]
            
        return results
        
    def _compute_synaptic_correlations(
        self, 
        E: np.ndarray, 
        I: np.ndarray, 
        Vout: np.ndarray,
        labels: np.ndarray
    ) -> dict[str, Any]:
        """Compute all synaptic correlation types.
        
        Args:
            E: Excitatory inputs
            I: Inhibitory inputs
            Vout: Branch outputs
            labels: Class labels
            
        Returns:
            Dictionary with correlation results
        """
        results = {}
        
        # E-E correlations
        self.logger.info("Computing E-E correlations...")
        results["EE"] = self._compute_conditional_correlations(E, None, labels)
        
        # I-I correlations
        self.logger.info("Computing I-I correlations...")
        results["II"] = self._compute_conditional_correlations(I, None, labels)
        
        # E-I correlations
        self.logger.info("Computing E-I correlations...")
        results["EI"] = self._compute_conditional_correlations(E, I, labels)
        
        # E-Vout correlations
        self.logger.info("Computing E-Vout correlations...")
        results["E_Vout"] = self._compute_conditional_correlations(E, Vout, labels)
        
        # I-Vout correlations
        self.logger.info("Computing I-Vout correlations...")
        results["I_Vout"] = self._compute_conditional_correlations(I, Vout, labels)
        
        # Vout-Vout correlations
        if Vout.shape[1] > 1:
            self.logger.info("Computing Vout-Vout correlations...")
            results["Vout_Vout"] = self._compute_conditional_correlations(Vout, None, labels)
            
        return results
        
    def _compute_tuning_curves(
        self,
        E: np.ndarray,
        I: np.ndarray,
        Vout: np.ndarray,
        labels: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute tuning curves (class selectivity) for each variable.
        
        Args:
            E: Excitatory inputs
            I: Inhibitory inputs
            Vout: Branch outputs
            labels: Class labels
            
        Returns:
            Dictionary with tuning curves
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) != 2:
            self.logger.warning(f"Tuning curves require binary labels, got {len(unique_labels)} classes")
            return {}
            
        class0_mask = labels == unique_labels[0]
        class1_mask = labels == unique_labels[1]
        
        results = {
            "E_tuning": np.mean(E[class1_mask], axis=0) - np.mean(E[class0_mask], axis=0),
            "I_tuning": np.mean(I[class1_mask], axis=0) - np.mean(I[class0_mask], axis=0),
            "Vout_tuning": np.mean(Vout[class1_mask], axis=0) - np.mean(Vout[class0_mask], axis=0),
        }
        
        return results
        
    def _analyze_single_layer(
        self,
        layer_name: str,
        layer_data: dict[str, torch.Tensor],
        labels: np.ndarray
    ) -> dict[str, Any]:
        """Analyze correlations for a single layer.
        
        Args:
            layer_name: Name of the layer
            layer_data: Dictionary with excitation, inhibition, output tensors
            labels: Class labels
            
        Returns:
            Dictionary with layer-specific correlation results
        """
        self.logger.info(f"Analyzing layer: {layer_name}")
        
        # Extract data
        E = layer_data["excitation"].detach().cpu().numpy()
        I = layer_data["inhibition"].detach().cpu().numpy()
        Vout = layer_data["output"].detach().cpu().numpy()
        
        # Ensure 2D
        if E.ndim == 1:
            E = E.reshape(-1, 1)
        if I.ndim == 1:
            I = I.reshape(-1, 1)
        if Vout.ndim == 1:
            Vout = Vout.reshape(-1, 1)
            
        results = {
            "layer_name": layer_name,
            "data_shapes": {
                "excitation": E.shape,
                "inhibition": I.shape,
                "output": Vout.shape,
            }
        }
        
        # Compute correlations
        if self.compute_synaptic_correlations:
            results["correlations"] = self._compute_synaptic_correlations(E, I, Vout, labels)
            
        # Compute tuning curves
        if self.compute_tuning_curves:
            results["tuning"] = self._compute_tuning_curves(E, I, Vout, labels)
            
        # Compute summary statistics
        results["summary_stats"] = self._compute_summary_statistics(results["correlations"])
        
        return results
        
    def _compute_summary_statistics(self, correlations: dict[str, Any]) -> dict[str, float]:
        """Compute summary statistics from correlation results.
        
        Args:
            correlations: Dictionary with correlation matrices
            
        Returns:
            Dictionary with summary statistics
        """
        stats = {}
        
        for corr_type in ["EE", "II", "EI", "E_Vout", "I_Vout", "Vout_Vout"]:
            if corr_type not in correlations:
                continue
                
            corr_data = correlations[corr_type]
            
            # Get upper triangular indices for self-correlations
            if corr_type in ["EE", "II", "Vout_Vout"]:
                # For self-correlations, exclude diagonal
                for cond in ["total", "noise", "signal"]:
                    if cond in corr_data:
                        matrix = corr_data[cond]
                        upper_tri = matrix[np.triu_indices_from(matrix, k=1)]
                        stats[f"{corr_type}_{cond}_mean"] = np.mean(np.abs(upper_tri))
                        stats[f"{corr_type}_{cond}_std"] = np.std(upper_tri)
            else:
                # For cross-correlations, use all values
                for cond in ["total", "noise", "signal"]:
                    if cond in corr_data:
                        matrix = corr_data[cond]
                        stats[f"{corr_type}_{cond}_mean"] = np.mean(np.abs(matrix))
                        stats[f"{corr_type}_{cond}_std"] = np.std(matrix)
                        
        return stats
        
    def _compute_per_ei_analysis(
        self,
        E: np.ndarray,
        I: np.ndarray,
        Vout: np.ndarray,
        labels: np.ndarray
    ) -> dict[str, Any]:
        """Compute separate analysis for excitatory and inhibitory networks.
        
        Args:
            E: Excitatory inputs
            I: Inhibitory inputs
            Vout: Branch outputs
            labels: Class labels
            
        Returns:
            Dictionary with E-specific and I-specific results
        """
        results = {}
        
        # Excitatory network analysis
        self.logger.info("Computing excitatory network correlations...")
        exc_results = {
            "EE": self._compute_conditional_correlations(E, None, labels),
            "E_Vout": self._compute_conditional_correlations(E, Vout, labels),
        }
        if self.compute_tuning_curves:
            exc_results["tuning"] = {
                "E_tuning": self._compute_tuning_curves(E, I, Vout, labels)["E_tuning"]
            }
        results["excitatory_results"] = {
            "correlations": exc_results,
            "summary_stats": self._compute_summary_statistics(exc_results)
        }
        
        # Inhibitory network analysis
        self.logger.info("Computing inhibitory network correlations...")
        inh_results = {
            "II": self._compute_conditional_correlations(I, None, labels),
            "I_Vout": self._compute_conditional_correlations(I, Vout, labels),
        }
        if self.compute_tuning_curves:
            inh_results["tuning"] = {
                "I_tuning": self._compute_tuning_curves(E, I, Vout, labels)["I_tuning"]
            }
        results["inhibitory_results"] = {
            "correlations": inh_results,
            "summary_stats": self._compute_summary_statistics(inh_results)
        }
        
        # Combined E-I analysis
        self.logger.info("Computing combined E-I network correlations...")
        combined_results = {
            "EI": self._compute_conditional_correlations(E, I, labels),
            "correlations": self._compute_synaptic_correlations(E, I, Vout, labels),
        }
        results["combined_results"] = combined_results
        
        return results

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
    ):
        """Run hierarchical correlation analysis on the model.
        
        Args:
            model: The trained model to analyze
            test_dataset: Test dataset for analysis
            device: Device to run analysis on
            save_path: Path to save results
            filename: Filename for saved results
        """
        self.logger.info("Starting hierarchical correlation analysis")
        self.logger.info(f"Model: {type(model).__name__}, Dataset size: {len(test_dataset)}, Device: {device}")
        
        try:
            # Check if model has ExcitationInhibitionNetwork
            if not isinstance(model.core_network, ExcitationInhibitionNetwork):
                self.logger.warning(
                    "Model does not have ExcitationInhibitionNetwork. Skipping correlation analysis."
                )
                return
                
            self.logger.info("Model validation passed - starting analysis")
            
            # Prepare data
            model = model.to(device)
            model.eval()
            
            self.logger.info("Loading test dataset...")
            inputs, labels = test_dataset[:]
            inputs: torch.Tensor = inputs.to(device)
            labels: torch.Tensor = labels.to(device)
            
            # Optional: Sample a subset of data
            if self.max_samples and len(inputs) > self.max_samples:
                self.logger.info(f"Sampling {self.max_samples} samples from {len(inputs)}")
                indices = torch.randperm(len(inputs))[: self.max_samples]
                inputs = inputs[indices]
                labels = labels[indices]
                
            # Attach forward hooks
            self.logger.info("Attaching forward hooks to collect activations...")
            handles = self.attach_forward_hooks(model)
            
            # Forward pass to collect activations
            self.logger.info("Running forward pass to collect neural activations...")
            with torch.no_grad():
                _ = model(inputs)
                
            # Remove hooks
            self.remove_forward_hooks(handles)
            
            # Convert labels to numpy
            labels_np = labels.cpu().numpy()
            
            # Initialize results
            results = {
                "method": "hierarchical_correlation_analysis",
                "n_samples": len(inputs),
                "n_layers": len(self.data_dict),
                "computation_level": self.computation_level,
                "per_layer_analysis_enabled": self.per_layer_analysis,
                "per_einet_analysis_enabled": self.per_einet_analysis,
            }
            
            # Network-level analysis (aggregate all layers)
            self.logger.info("Computing network-level correlations...")
            all_E = []
            all_I = []
            all_Vout = []
            
            for layer_data in self.data_dict.values():
                E = layer_data["excitation"].detach().cpu().numpy()
                I = layer_data["inhibition"].detach().cpu().numpy()
                Vout = layer_data["output"].detach().cpu().numpy()
                
                # Flatten if needed for network-level analysis
                if E.ndim > 2:
                    E = E.reshape(E.shape[0], -1)
                if I.ndim > 2:
                    I = I.reshape(I.shape[0], -1)
                if Vout.ndim > 2:
                    Vout = Vout.reshape(Vout.shape[0], -1)
                    
                all_E.append(E)
                all_I.append(I)
                all_Vout.append(Vout)
                
            # Concatenate all layers
            network_E = np.hstack(all_E)
            network_I = np.hstack(all_I)
            network_Vout = np.hstack(all_Vout)
            
            # Compute network-level correlations
            results["network_level"] = {
                "correlations": self._compute_synaptic_correlations(
                    network_E, network_I, network_Vout, labels_np
                ),
                "tuning": self._compute_tuning_curves(
                    network_E, network_I, network_Vout, labels_np
                ) if self.compute_tuning_curves else {},
            }
            results["network_level"]["summary_stats"] = self._compute_summary_statistics(
                results["network_level"]["correlations"]
            )
            
            # Layer-level analysis
            if self.per_layer_analysis:
                self.logger.info("Computing per-layer correlations...")
                results["per_layer_analysis"] = {}
                
                for layer_name, layer_data in self.data_dict.items():
                    layer_results = self._analyze_single_layer(layer_name, layer_data, labels_np)
                    results["per_layer_analysis"][layer_name] = layer_results
                    
            # EI-network specific analysis
            if self.per_einet_analysis:
                self.logger.info("Computing per-EI-network correlations...")
                results["per_einet_analysis"] = self._compute_per_ei_analysis(
                    network_E, network_I, network_Vout, labels_np
                )
                
            # Save results if requested
            if save_path is not None:
                os.makedirs(save_path, exist_ok=True)
                save_dict(results, save_path, filename)
                
                # Generate plots
                self._generate_correlation_plots(results, save_path)
                
                # Save summary CSV
                summary = self._create_summary_dataframe(results)
                summary.to_csv(os.path.join(save_path, f"{filename}_summary.csv"), index=False)
                
                self.logger.info(f"Results saved to {save_path}")
                
        except Exception as e:
            self.logger.error(f"Error during correlation analysis: {e}")
            self.logger.error(traceback.format_exc())
            raise

    def analyze_hierarchical(self, *args, **kwargs):
        """Alias for analyze method for backward compatibility."""
        return self.analyze(*args, **kwargs)
            
    def _generate_correlation_plots(self, results: dict[str, Any], save_path: str):
        """Generate plots for correlation analysis results.
        
        Args:
            results: Analysis results
            save_path: Path to save plots
        """
        try:
            from dendritic_modeling.plotting.visualizations.correlation_plots import (
                plot_hierarchical_correlations,
                plot_correlation_matrices,
                plot_tuning_curves,
            )
            
            # Plot hierarchical correlations
            plot_hierarchical_correlations(results, save_path=save_path)
            
            # Plot correlation matrices for network level
            if "network_level" in results:
                plot_correlation_matrices(
                    results["network_level"]["correlations"],
                    save_path=os.path.join(save_path, "network_correlation_matrices.png")
                )
                
            # Plot tuning curves if available
            if "network_level" in results and "tuning" in results["network_level"]:
                plot_tuning_curves(
                    results["network_level"]["tuning"],
                    save_path=os.path.join(save_path, "tuning_curves.png")
                )
                
        except ImportError as e:
            self.logger.warning(f"Could not import plotting functions: {e}")
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
            
    def _create_summary_dataframe(self, results: dict[str, Any]) -> pd.DataFrame:
        """Create a summary dataframe from results.
        
        Args:
            results: Analysis results
            
        Returns:
            Summary dataframe
        """
        summary_data = {
            "method": results["method"],
            "n_samples": results["n_samples"],
            "n_layers": results["n_layers"],
        }
        
        # Add network-level statistics
        if "network_level" in results and "summary_stats" in results["network_level"]:
            for key, value in results["network_level"]["summary_stats"].items():
                summary_data[f"network_{key}"] = value
                
        return pd.DataFrame([summary_data])