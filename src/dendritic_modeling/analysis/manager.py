"""
Analysis manager module.

This module contains the AnalysisManager class which coordinates and orchestrates
all the different analysis tools in the dendritic modeling package.
"""

import logging
import math
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch
from torch.utils.data import DataLoader, TensorDataset

matplotlib.use("Agg")

from dendritic_modeling.config import AnalysisConfig
from dendritic_modeling.networks.layers.dendritic import DendriticBranchLayer
from dendritic_modeling.analysis.information import IntegratedInformationAnalyzer
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

# Import analyzer classes from their new locations
from dendritic_modeling.analysis.performance.metrics import PerformanceAnalyzer
from dendritic_modeling.analysis.performance.noise import NoisePerturbationAnalyzer
from dendritic_modeling.analysis.performance.ablation import AblationAnalyzer
from dendritic_modeling.analysis.performance.layers import SingleLayerContributionAnalyzer
from dendritic_modeling.analysis.performance.synaptic_ablation import SynapticAblationAnalyzer
from dendritic_modeling.analysis.synaptic.turnover import SynapticTurnoverAnalyzer
from dendritic_modeling.analysis.synaptic.weights import WeightAnalyzer
from dendritic_modeling.analysis.activations.synaptic import SynapticActivationAnalyzer
from dendritic_modeling.analysis.activations.branches import BranchActivationAnalyzer


class AnalysisManager:
    """
    Coordinates and orchestrates all analysis tools for dendritic modeling.
    
    This class manages the initialization and execution of various analysis tools
    based on configuration settings, providing a unified interface for running
    comprehensive model analysis.
    """
    
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

        # Initialize Performance Analysis
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

        # Initialize Integrated Information Analysis
        # Uses the InformationAnalysisConfig from the main config
        info_analysis_config = self.config.information_analysis
        self.info_analysis_enabled = hasattr(info_analysis_config, 'method') or isinstance(info_analysis_config, dict)
        if self.info_analysis_enabled:
            self.information_analyzer = IntegratedInformationAnalyzer(info_analysis_config)

        # Initialize Synaptic Turnover Analysis
        syn_turnover_config = self.config.synapse_turnover_analysis
        self.syn_turnover_enabled = syn_turnover_config.get("enabled", False)
        self.syn_turnover_training = syn_turnover_config.get("training", False)
        if self.syn_turnover_enabled:
            self.syn_turnover_analyzer = SynapticTurnoverAnalyzer(model=model)

        # Initialize Noise Perturbation Analysis
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

        # Initialize Single Layer Contribution Analysis (formerly layer_ablation)
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

        # Initialize Synaptic Ablation Analysis (formerly branch_ablation)
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

        # Initialize Ablation Analysis
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

        # Initialize Weight Analysis
        weight_config = self.config.weight_analysis
        self.weight_enabled = weight_config.get("enabled", False)
        self.weight_training = weight_config.get("training", False)
        if self.weight_enabled:
            weight_params: dict = weight_config.get("params", {})
            self.weight_analyzer = WeightAnalyzer(
                synapse_weights=weight_params.get("synapse_weights", True),
                branch_weights=weight_params.get("branch_weights", True),
            )

        # Initialize Synaptic Activation Analysis
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

        # Initialize Branch Activation Analysis
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
        """
        Run all enabled analysis tools.
        
        Args:
            filename: Base filename for saving analysis results
            training: Whether this is being called during training (affects which analyses run)
        """
        # Performance Analysis
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

        # Information Analysis using Integrated Analyzer
        if self.info_analysis_enabled and not training:
            self._run_information_analysis(filename)

        # Synaptic Turnover Analysis
        if self.syn_turnover_enabled and (
            (self.syn_turnover_training and training) or not training
        ):
            self.syn_turnover_analyzer.analyze(
                model=self.model,
                training=training,
                save_path=os.path.join(self.save_root, "synapse_turnover_analysis"),
                filename=filename,
            )

        # Noise Perturbation Analysis
        if self.noise_enabled and ((self.noise_training and training) or not training):
            self.noise_perturbation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "noise_perturbation_analysis"),
                filename=filename,
            )

        # Single Layer Contribution Analysis
        if self.single_layer_enabled and (
            (self.single_layer_training and training) or not training
        ):
            self.single_layer_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                save_path=os.path.join(self.save_root, "single_layer_contribution"),
                filename=filename,
            )

        # Synaptic Ablation Analysis
        if self.synaptic_ablation_enabled and (
            (self.synaptic_ablation_training and training) or not training
        ):
            self.synaptic_ablation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                save_path=os.path.join(self.save_root, "synaptic_ablation"),
                filename=filename,
            )

        # Ablation Analysis
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

        # Weight Analysis
        if self.weight_enabled and (
            (self.weight_training and training) or not training
        ):
            self.weight_analyzer.analyze(
                model=self.model,
                save_path=os.path.join(self.save_root, "weight_analysis"),
                filename=filename,
            )

        # Synaptic Activation Analysis
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

        # Branch Activation Analysis
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

    def _run_information_analysis(self, filename: str):
        """Run integrated information analysis on the model."""
        try:
            # Check if model has ExcitationInhibitionNetwork
            if not isinstance(self.model.net, ExcitationInhibitionNetwork):
                self.logger.warning("Model does not have ExcitationInhibitionNetwork. Skipping information analysis.")
                return

            self.logger.info("Running integrated information analysis...")
            
            # Prepare data
            self.model = self.model.to(self.device)
            self.model.eval()
            
            inputs, labels = self.data["test"][:]
            
            # Optional: Sample a subset of data for faster computation
            max_samples = getattr(self.config.information_analysis, 'max_samples', None)
            if max_samples is None:
                # Try to get from dict config
                if isinstance(self.config.information_analysis, dict):
                    max_samples = self.config.information_analysis.get('max_samples', None)
            
            if max_samples and len(inputs) > max_samples:
                self.logger.info(f"Sampling {max_samples} samples from {len(inputs)} for faster computation")
                indices = torch.randperm(len(inputs))[:max_samples]
                inputs = inputs[indices]
                labels = labels[indices]
            
            inputs = inputs.to(self.device)
            
            # Collect activations from all branches
            data_dict = {}
            
            # Use forward hooks to collect branch data
            handles = []
            for name, module in self.model.named_modules():
                if isinstance(module, DendriticBranchLayer):
                    if (
                        module.branch_excitation is not None
                        and module.branch_inhibition is not None
                    ):
                        data_dict[name] = {}
                        module._name = name
                        
                        def make_hook(module_name):
                            def hook(module, input, output):
                                excitatory_input, inhibitory_input, _ = input
                                data_dict[module_name]["excitation"] = module.branch_excitation(excitatory_input)
                                data_dict[module_name]["inhibition"] = module.branch_inhibition(inhibitory_input)
                                data_dict[module_name]["output"] = output
                            return hook
                        
                        handle = module.register_forward_hook(make_hook(name))
                        handles.append(handle)
            
            # Forward pass to collect activations
            with torch.no_grad():
                _ = self.model(inputs)
            
            # Remove hooks
            for handle in handles:
                handle.remove()
            
            # Get aggregation method from config
            branch_aggregation = "mean"  # default
            compute_per_branch = False
            if hasattr(self.config.information_analysis, 'branch_aggregation'):
                branch_aggregation = self.config.information_analysis.branch_aggregation
            elif isinstance(self.config.information_analysis, dict):
                branch_aggregation = self.config.information_analysis.get('branch_aggregation', 'mean')
            
            # Check if we should compute per-branch MI
            if hasattr(self.config.information_analysis, 'compute_per_branch'):
                compute_per_branch = self.config.information_analysis.compute_per_branch
            elif isinstance(self.config.information_analysis, dict):
                compute_per_branch = self.config.information_analysis.get('compute_per_branch', False)
            
            C = labels.numpy()
            
            if compute_per_branch:
                # Option 1: Compute MI for each branch separately and take the mean
                self.logger.info("Computing MI for each branch separately...")
                
                all_results = []
                branch_names = []
                
                for branch_name, branch_data in data_dict.items():
                    if "excitation" not in branch_data:
                        continue
                        
                    # Get single branch data
                    E_branch = branch_data["excitation"].detach().cpu().numpy()
                    I_branch = branch_data["inhibition"].detach().cpu().numpy()
                    Vout_branch = branch_data["output"].detach().cpu().numpy()
                    
                    # Compute MI for this single branch
                    branch_results = self.information_analyzer.analyze(E_branch, I_branch, Vout_branch, C, None)
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
                self.logger.info(f"Aggregating branches using method: {branch_aggregation}")
                
                # Collect all branch data
                excitation_list = []
                inhibition_list = []
                output_list = []
                
                for branch_data in data_dict.values():
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
                if branch_aggregation == "mean":
                    # Average across branches
                    E = torch.stack(excitation_list, dim=0).mean(dim=0).detach().cpu().numpy()
                    I = torch.stack(inhibition_list, dim=0).mean(dim=0).detach().cpu().numpy()
                    Vout = torch.stack(output_list, dim=0).mean(dim=0).detach().cpu().numpy()
                elif branch_aggregation == "max":
                    # Max pooling across branches
                    E = torch.stack(excitation_list, dim=0).max(dim=0)[0].detach().cpu().numpy()
                    I = torch.stack(inhibition_list, dim=0).max(dim=0)[0].detach().cpu().numpy()
                    Vout = torch.stack(output_list, dim=0).max(dim=0)[0].detach().cpu().numpy()
                elif branch_aggregation == "sum":
                    # Sum across branches
                    E = torch.stack(excitation_list, dim=0).sum(dim=0).detach().cpu().numpy()
                    I = torch.stack(inhibition_list, dim=0).sum(dim=0).detach().cpu().numpy()
                    Vout = torch.stack(output_list, dim=0).sum(dim=0).detach().cpu().numpy()
                elif branch_aggregation == "first":
                    # Only use first branch
                    E = excitation_list[0].detach().cpu().numpy()
                    I = inhibition_list[0].detach().cpu().numpy()
                    Vout = output_list[0].detach().cpu().numpy()
                else:
                    self.logger.warning(f"Unknown branch aggregation method: {branch_aggregation}. Using mean.")
                    E = torch.stack(excitation_list, dim=0).mean(dim=0).detach().cpu().numpy()
                    I = torch.stack(inhibition_list, dim=0).mean(dim=0).detach().cpu().numpy()
                    Vout = torch.stack(output_list, dim=0).mean(dim=0).detach().cpu().numpy()
                
                self.logger.info(f"Aggregated data shapes - E: {E.shape}, I: {I.shape}, Vout: {Vout.shape}")
                
                # Get soma activation (Vinf) if available
                Vinf = None
                # For now, we'll use the final model output as Vinf
                with torch.no_grad():
                    final_output = self.model(inputs)
                    if hasattr(self.model, 'final_fc'):
                        # Model has final classification layer
                        Vinf = final_output.detach().cpu().numpy()
                
                # Run information analysis on aggregated data
                results = self.information_analyzer.analyze(E, I, Vout, C, Vinf)
                results["aggregation_method"] = branch_aggregation
                results["num_branches_aggregated"] = len(excitation_list)
            
            # Save results
            save_path = os.path.join(self.save_root, "information_analysis")
            os.makedirs(save_path, exist_ok=True)
            save_dict(results, save_path, filename)
            
            # Also save a summary report
            summary = self.information_analyzer.get_summary(results)
            summary_path = os.path.join(save_path, f"{filename}_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            self.logger.info(f"Information analysis completed. Results saved to {save_path}")
            
            # Generate information plots
            try:
                from dendritic_modeling.plotting.analysis import plot_information_analysis
                self.logger.info("Generating information analysis plots...")
                
                plot_save_path = os.path.join(save_path, "plots")
                os.makedirs(plot_save_path, exist_ok=True)
                
                # Generate all plots
                figs = plot_information_analysis(results, save_path=plot_save_path, filename=filename)
                
                self.logger.info(f"Generated {len(figs)} information analysis plots in {plot_save_path}")
                
                # Close all figures to free memory
                for fig in figs:
                    plt.close(fig)
                    
            except Exception as e:
                self.logger.warning(f"Could not generate information plots: {e}")
            
        except Exception as e:
            self.logger.error(f"Error in information analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _average_mi_results(self, results_list):
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