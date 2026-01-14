"""
Analysis manager module.

This module contains the AnalysisManager class which coordinates and orchestrates
all the different analysis tools in the dendritic modeling package.
"""

import logging
import os
from typing import Optional

import torch

from dendritic_modeling.analysis.core.ablation import AblationAnalyzer
from dendritic_modeling.analysis.core.branch_activation import BranchActivationAnalyzer
from dendritic_modeling.analysis.core.information import InformationAnalyzer
from dendritic_modeling.analysis.metrics.layer_contribution import (
    SingleLayerContributionAnalyzer,
)
from dendritic_modeling.analysis.metrics.noise_perturbation import (
    NoisePerturbationAnalyzer,
)
from dendritic_modeling.analysis.metrics.performance import PerformanceAnalyzer
from dendritic_modeling.analysis.tools.correlation_analysis import CorrelationAnalyzer
from dendritic_modeling.analysis.tools.synaptic_activation import (
    SynapticActivationAnalyzer,
)
from dendritic_modeling.analysis.tools.synaptic_turnover import SynapticTurnoverAnalyzer
from dendritic_modeling.analysis.tools.weight import WeightAnalyzer
from dendritic_modeling.analysis.tools.compartment_statistics import CompartmentStatisticsAnalyzer
from dendritic_modeling.analysis.tools.compartment_snr import CompartmentSNRAnalyzer
from dendritic_modeling.analysis.tools.multiplicative_gain import MultiplicativeGainAnalyzer
from dendritic_modeling.analysis.tools.receptive_fields import ReceptiveFieldAnalyzer
from dendritic_modeling.config import AnalysisConfig
from dendritic_modeling.models import BaseModel


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
        data: dict[str, torch.utils.data.Dataset],
        analysis_config: AnalysisConfig,
        save_root: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model = model
        self.data = data
        self.config = analysis_config
        self.save_root = save_root
        self.device = device
        os.makedirs(self.save_root, exist_ok=True)
        # Logging should be configured externally at the application startup
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
            self.syn_turnover_analyzer = SynapticTurnoverAnalyzer(
                model=model, params=self.config.synapse_turnover_analysis.params
            )

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
            self.weight_analyzer = WeightAnalyzer(self.config.weight_analysis.params)
        
        # Initialize Compartment Statistics Analysis
        if self.config.compartment_statistics_analysis.enabled:
            self.compartment_statistics_analyzer = CompartmentStatisticsAnalyzer(
                self.config.compartment_statistics_analysis.params
            )

        # Initialize Compartment SNR Analysis
        if self.config.compartment_snr_analysis.enabled:
            self.compartment_snr_analyzer = CompartmentSNRAnalyzer(
                self.config.compartment_snr_analysis.params
            )
        
        if self.config.multiplicative_gain_analysis.enabled:
            self.multiplicative_gain_analyzer = MultiplicativeGainAnalyzer(
                self.config.multiplicative_gain_analysis.params
            )

        # Initialize Receptive Field Analysis
        if self.config.receptive_field_analysis.enabled:
            self.receptive_field_analyzer = ReceptiveFieldAnalyzer(
                self.config.receptive_field_analysis.params
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
        if (
            hasattr(self.config, "correlation_analysis")
            and self.config.correlation_analysis.enabled
        ):
            self.correlation_analysis_enabled = True
            self.correlation_analyzer = CorrelationAnalyzer(
                params=self.config.correlation_analysis.params
            )
        else:
            self.correlation_analysis_enabled = False

    def set_pruning_results(self, pruning_results: dict):
        """Set pruning results for inclusion in performance plots."""
        self.pruning_results = pruning_results

    def run_analysis(self, filename: str = "final", training: bool = False):
        """
        Run all enabled analysis tools.

        Args:
            filename: Base filename for saving analysis results
            training: Whether this is being called during training (affects which analyses run)
        """
        # Performance Analysis
        if self.config.performance_analysis.enabled and (
            (self.config.performance_analysis.training and training) or not training
        ):
            # Determine save path based on whether this is training or final analysis
            if training:
                # Save epoch results in a subfolder
                save_path = os.path.join(self.save_root, "performance", "epochs")
                os.makedirs(save_path, exist_ok=True)
            else:
                # Save final results in main performance folder
                save_path = os.path.join(self.save_root, "performance")

            self.performance_analyzer.analyze(
                model=self.model,
                train_ds=self.data["train"],
                valid_ds=self.data["valid"],
                test_ds=self.data["test"],
                device=self.device,
                save_path=save_path,
                filename=filename,
                training=training,
            )

        # Information Analysis
        if self.config.information_analysis.enabled and (
            (self.config.information_analysis.training and training) or not training
        ):
            self.logger.info("Starting information analysis...")
            self.information_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "information_analysis"),
                filename=filename,
            )
            self.logger.info("Information analysis completed successfully")

        # Synaptic Turnover Analysis
        if self.config.synapse_turnover_analysis.enabled and (
            (self.config.synapse_turnover_analysis.training and training)
            or not training
        ):
            self.syn_turnover_analyzer.analyze(
                model=self.model,
                training=training,
                save_path=os.path.join(self.save_root, "synapse_turnover_analysis"),
                filename=filename,
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
                filename=filename,
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
                filename=filename,
            )

        # Ablation Analysis
        if self.config.ablation_analysis.enabled and (
            (self.config.ablation_analysis.training and training) or not training
        ):
            self.ablation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "ablation_analysis"),
                filename=filename,
                training=training,
            )

        # Weight Analysis
        if self.config.weight_analysis.enabled and (
            (self.config.weight_analysis.training and training) or not training
        ):
            self.weight_analyzer.analyze(
                model=self.model,
                save_path=os.path.join(self.save_root, "weight_analysis"),
                filename=filename,
                training=training,
            )
        
        # Compartment Statistics Analysis
        if self.config.compartment_statistics_analysis.enabled and (
            (self.config.compartment_statistics_analysis.training and training) or not training
        ):
            self.compartment_statistics_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "compartment_statistics_analysis"),
                filename=filename,
                training=training
            )
        
        # Compartment SNR Analysis
        if self.config.compartment_snr_analysis.enabled and (
            (self.config.compartment_snr_analysis.training and training) or not training
        ):
            self.compartment_snr_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "compartment_snr_analysis"),
                filename=filename,
                training=training
            )

        # Multiplicative Gain Analysis
        if self.config.multiplicative_gain_analysis.enabled and (
            (self.config.multiplicative_gain_analysis.training and training) or not training
        ):
            self.multiplicative_gain_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "multiplicative_gain_analysis"),
                filename=filename,
                training=training
            )

        # Receptive Field Analysis
        if self.config.receptive_field_analysis.enabled and (
            (self.config.receptive_field_analysis.training and training) or not training
        ):
            self.receptive_field_analyzer.analyze(
                model=self.model,
                save_path=os.path.join(self.save_root, "receptive_field_analysis"),
                filename=filename,
            )

        # Synaptic Activation Analysis
        if self.config.synaptic_activation_analysis.enabled and (
            (self.config.synaptic_activation_analysis.training and training)
            or not training
        ):
            self.synaptic_activation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "synaptic_activation_analysis"),
                filename=filename,
            )

        # Branch Activation Analysis
        if self.config.branch_activation_analysis.enabled and (
            (self.config.branch_activation_analysis.training and training)
            or not training
        ):
            self.branch_activation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "branch_activation_analysis"),
                filename=filename,
            )

        # Correlation Analysis
        if self.correlation_analysis_enabled and (
            (
                hasattr(self.config.correlation_analysis, "training")
                and self.config.correlation_analysis.training
                and training
            )
            or not training
        ):
            self.logger.info("Starting correlation analysis...")
            self.correlation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "correlation_analysis"),
                filename=filename,
            )
            self.logger.info("Correlation analysis completed successfully")
