"""
Analysis manager module.

This module contains the AnalysisManager class which coordinates and orchestrates
all the different analysis tools in the dendritic modeling package.
"""

import logging
import os
from typing import Dict, Optional

import torch

from dendritic_modeling.models import BaseModel
from dendritic_modeling.config import AnalysisConfig

from dendritic_modeling.analysis.performance import PerformanceAnalyzer
from dendritic_modeling.analysis.noise_perturbation import NoisePerturbationAnalyzer
from dendritic_modeling.analysis.ablation import AblationAnalyzer
from dendritic_modeling.analysis.layer_contribution import SingleLayerContributionAnalyzer
from dendritic_modeling.analysis.synaptic_turnover import SynapticTurnoverAnalyzer
from dendritic_modeling.analysis.weight import WeightAnalyzer
from dendritic_modeling.analysis.synaptic_activation import SynapticActivationAnalyzer
from dendritic_modeling.analysis.branch_activation import BranchActivationAnalyzer
from dendritic_modeling.analysis.information import InformationAnalyzer
from dendritic_modeling.analysis.correlation_analysis import CorrelationAnalyzer


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
        if self.config.performance_analysis.enabled:
            self.performance_analyzer = PerformanceAnalyzer(
                self.config.performance_analysis.params
            )

        # Initialize Integrated Information Analysis
        if self.config.information_analysis.enabled:
            self.information_analyzer = InformationAnalyzer(
                self.config.information_analysis.params
            )

        # Initialize Synaptic Turnover Analysis
        if self.config.synapse_turnover_analysis.enabled:
            self.syn_turnover_analyzer = SynapticTurnoverAnalyzer(model=model)

        # Initialize Noise Perturbation Analysis
        if self.config.noise_perturbation_analysis.enabled:
            self.noise_perturbation_analyzer = NoisePerturbationAnalyzer(
                self.config.noise_perturbation_analysis.params
            )

        # Initialize Single Layer Contribution Analysis (formerly layer_ablation)
        if self.config.single_layer_contribution.enabled:
            self.single_layer_analyzer = SingleLayerContributionAnalyzer(
                self.config.single_layer_contribution.params
            )

        # Initialize Ablation Analysis
        if self.config.ablation_analysis.enabled:
            self.ablation_analyzer = AblationAnalyzer(
                self.config.ablation_analysis.params
            )

        # Initialize Weight Analysis
        if self.config.weight_analysis.enabled:
            self.weight_analyzer = WeightAnalyzer(
                self.config.weight_analysis.params
            )

        # Initialize Synaptic Activation Analysis
        if self.config.synaptic_activation_analysis.enabled:
            self.synaptic_activation_analyzer = SynapticActivationAnalyzer(
                self.config.synaptic_activation_analysis.params
            )

        # Initialize Branch Activation Analysis
        if self.config.branch_activation_analysis.enabled:
            self.branch_activation_analyzer = BranchActivationAnalyzer(
                self.config.branch_activation_analysis.params
            )

        # Initialize Correlation Analysis
        if hasattr(self.config, 'correlation_analysis') and self.config.correlation_analysis.enabled:
            # Note: CorrelationAnalyzer is initialized differently, needs dataset
            self.correlation_analysis_enabled = True
        else:
            self.correlation_analysis_enabled = False

    def run_analysis(self, filename: str = "final", training: bool = False):
        """
        Run all enabled analysis tools.
        
        Args:
            filename: Base filename for saving analysis results
            training: Whether this is being called during training (affects which analyses run)
        """
        # Performance Analysis
        if self.config.performance_analysis.enabled and (
            (self.config.performance_analysis.training and training) 
            or not training
        ):
            self.performance_analyzer.analyze(
                model=self.model,
                train_ds=self.data["train"],
                valid_ds=self.data["valid"],
                test_ds=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "performance"),
                filename=filename
            )

        # Information Analysis
        if self.config.information_analysis.enabled and (
            (self.config.information_analysis.training and training)
            or not training
        ):
            self.information_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "information_analysis"),
                filename=filename
            )

        # Synaptic Turnover Analysis
        if self.config.synapse_turnover_analysis.enabled and (
            (self.config.synapse_turnover_analysis.training and training)
            or not training
        ):
            self.syn_turnover_analyzer.analyze(
                model=self.model,
                training=training,
                save_path=os.path.join(self.save_root, "synapse_turnover_analysis"),
                filename=filename
            )

        # Noise Perturbation Analysis
        if self.config.noise_perturbation_analysis.enabled and (
            (self.config.noise_perturbation_analysis.training and training)
            or not training
        ):
            self.noise_perturbation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "noise_perturbation_analysis"),
                filename=filename
            )

        # Single Layer Contribution Analysis
        if self.config.single_layer_contribution.enabled and (
            (self.config.single_layer_contribution.training and training)
            or not training
        ):
            self.single_layer_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "single_layer_contribution"),
                filename=filename
            )

        # Ablation Analysis
        if self.config.ablation_analysis.enabled and (
            (self.config.ablation_analysis.training and training)
            or not training
        ):
            self.ablation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "ablation_analysis"),
                filename=filename
            )

        # Weight Analysis
        if self.config.weight_analysis.enabled and (
            (self.config.weight_analysis.training and training)
            or not training
        ):
            self.weight_analyzer.analyze(
                model=self.model,
                save_path=os.path.join(self.save_root, "weight_analysis"),
                filename=filename
            )

        # Synaptic Activation Analysis
        if self.config.synaptic_activation_analysis.enabled and (
            (self.config.synaptic_activation_analysis.training and training)
            or not training
        ):
            self.synaptic_activation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device="cpu",
                save_path=os.path.join(self.save_root, "synaptic_activation_analysis"),
                filename=filename
            )

        # Branch Activation Analysis
        if self.config.branch_activation_analysis.enabled and (
            (self.config.branch_activation_analysis.training and training)
            or not training
        ):
            self.branch_activation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device="cpu",
                save_path=os.path.join(self.save_root, "branch_activation_analysis"),
                filename=filename
            )

        # Correlation Analysis
        if self.correlation_analysis_enabled and (
            (hasattr(self.config.correlation_analysis, 'training') and self.config.correlation_analysis.training and training)
            or not training
        ):
            # Initialize with test dataset for correlation structure
            correlation_analyzer = CorrelationAnalyzer(
                model=self.model,
                dataset=self.data["test"],
                device=self.device
            )
            
            # Get information results if available
            information_results = None
            accuracy = None
            
            # Try to load information analysis results if they exist
            info_results_path = os.path.join(self.save_root, "information_analysis", f"{filename}.npz")
            if os.path.exists(info_results_path):
                try:
                    info_data = np.load(info_results_path, allow_pickle=True)
                    information_results = info_data.item() if info_data.files else {}
                    logger.info("Loaded information analysis results for correlation analysis")
                except Exception as e:
                    logger.warning(f"Failed to load information results: {e}")
            
            # Try to get accuracy from recent training
            if hasattr(self.model, 'last_test_accuracy'):
                accuracy = self.model.last_test_accuracy
            elif hasattr(self, 'last_test_accuracy'):
                accuracy = self.last_test_accuracy
            
            correlation_analyzer.analyze(
                save_dir=os.path.join(self.save_root, "correlation_analysis"),
                information_results=information_results,
                accuracy=accuracy
            )
