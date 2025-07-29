"""
Analysis manager module.

This module contains the AnalysisManager class which coordinates and orchestrates
all the different analysis tools in the dendritic modeling package.
"""

import logging
import os

import numpy as np

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
from dendritic_modeling.plotting.visualizations.information_analysis import plot_information_analysis
from dendritic_modeling.plotting.visualizations.general_analysis_plots import (
    plot_performance_metrics,
    plot_synapse_turnover,
    plot_weight_distributions,
)


class AnalysisManager:
    """Manager for running multiple analyses on models."""

    def __init__(self, model, data, analysis_config, save_root, device='cpu', train_config=None):
        """Initialize with model, data, and configs."""
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
            # Get snapshot interval from config (default to every 5 epochs for efficiency)
            snapshot_interval = getattr(
                self.config.synapse_turnover_analysis, 'snapshot_interval', 5
            )
            self.syn_turnover_analyzer = SynapticTurnoverAnalyzer(
                model=model, 
                snapshot_interval=snapshot_interval
            )
            self.logger.info(
                f"Synaptic turnover analyzer initialized with snapshot interval: {snapshot_interval} epochs"
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
            # Note: CorrelationAnalyzer is initialized differently, needs dataset
            self.correlation_analysis_enabled = True
        else:
            self.correlation_analysis_enabled = False

    def run_analysis(self, filename: str = "final", training: bool = False):
        """Run configured analyses and generate plots."""
        # Run analyses and get results
        if self.config.performance_analysis.enabled and (
            (self.config.performance_analysis.training and training) or not training
        ):
            # Determine save path based on whether this is training or final analysis
            if training:
                # Save epoch results in a subfolder
                epoch_save_path = os.path.join(self.save_root, "performance", "epochs")
            else:
                # Save final results in main performance folder
                epoch_save_path = os.path.join(self.save_root, "performance")
            
            perf_results = self.performance_analyzer.analyze(
                model=self.model,
                train_ds=self.data["train"],
                valid_ds=self.data["valid"],
                test_ds=self.data["test"],
                device=self.device,
                save_path=epoch_save_path,
                filename=filename,
            )
            
            # Only generate plots when training is complete
            if not training:
                # Generate bar plots only for final results (when training=false)
                if not self.config.performance_analysis.training:
                    plot_performance_metrics(
                        perf_results, save_path=os.path.join(self.save_root, "performance")
                    )
                
                # Generate epoch evolution plots if training analysis was enabled
                if self.config.performance_analysis.training:
                    from dendritic_modeling.plotting.visualizations.performance_plots import PerformancePlotter
                    plotter = PerformancePlotter()
                    plotter.plot_performance_evolution(
                        performance_dir=os.path.join(self.save_root, "performance", "epochs"),
                        save_path=os.path.join(self.save_root, "performance"),
                    )

        # Information Analysis
        if self.config.information_analysis.enabled and (
            (self.config.information_analysis.training and training) or not training
        ):
            info_results = self.information_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "information_analysis"),
                filename=filename,
            )
            if not training:
                plot_information_analysis(
                    info_results,
                    save_path=os.path.join(self.save_root, "information_analysis"),
                )
                
                # Also plot per-layer information if available
                if "layer_statistics" in info_results:
                    from dendritic_modeling.plotting.visualizations.information_analysis import InformationAnalysisPlotter
                    plotter = InformationAnalysisPlotter()
                    plotter.plot_per_layer_information(
                        info_results["layer_statistics"],
                        save_path=os.path.join(self.save_root, "information_analysis"),
                        filename_prefix="per_layer"
                    )

        # Synaptic Turnover Analysis
        if self.config.synapse_turnover_analysis.enabled and (
            (self.config.synapse_turnover_analysis.training and training)
            or not training
        ):
            if not training:
                # Final analysis mode: compute turnover from collected snapshots
                # Check if we have collected snapshots during training
                if len(self.syn_turnover_analyzer.data_dict_list) < 2:
                    self.logger.warning(
                        "Not enough snapshots collected during training for turnover analysis. "
                        f"Found {len(self.syn_turnover_analyzer.data_dict_list)} snapshots."
                    )
                    # If no snapshots during training, collect a final snapshot
                    if len(self.syn_turnover_analyzer.data_dict_list) == 0:
                        self.syn_turnover_analyzer.analyze(
                            model=self.model,
                            training=True,  # Collect snapshot
                            save_path=None,
                            filename=filename,
                        )
                
                # Now analyze the turnover between all collected snapshots
                turnover_results = self.syn_turnover_analyzer.analyze(
                    model=self.model,
                    training=False,  # Analyze mode
                    save_path=None,  # Don't save to file yet, get results
                    filename=filename,
                )
                
                # Save the results
                if turnover_results is not None:
                    self.syn_turnover_analyzer.analyze(
                        model=self.model,
                        training=False,  # Analyze mode
                        save_path=os.path.join(self.save_root, "synapse_turnover_analysis"),
                        filename=filename,
                    )
                    
                    # Generate plots
                    plot_synapse_turnover(
                        turnover_results,
                        save_path=os.path.join(self.save_root, "synapse_turnover_analysis"),
                        filename="synapse_turnover"
                    )
                else:
                    self.logger.warning("No turnover results to plot")
            else:
                # Training mode: just collect snapshot
                self.syn_turnover_analyzer.analyze(
                    model=self.model,
                    training=training,
                    save_path=None,  # Don't save individual snapshots
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
            )

        # Weight Analysis
        if self.config.weight_analysis.enabled and (
            (self.config.weight_analysis.training and training) or not training
        ):
            weight_results = self.weight_analyzer.analyze(
                model=self.model,
                save_path=os.path.join(self.save_root, "weight_analysis"),
                filename=filename,
            )
            if not training:
                plot_weight_distributions(
                    weight_results,
                    save_path=os.path.join(self.save_root, "weight_analysis"),
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
                device="cpu",
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
            # Initialize with test dataset for correlation structure
            correlation_analyzer = CorrelationAnalyzer(
                model=self.model, dataset=self.data["test"], device=self.device
            )

            # Get information results if available
            information_results = None
            accuracy = None

            # Try to load information analysis results if they exist
            info_results_path = os.path.join(
                self.save_root, "information_analysis", f"{filename}.npz"
            )
            if os.path.exists(info_results_path):
                try:
                    info_data = np.load(info_results_path, allow_pickle=True)
                    information_results = info_data.item() if info_data.files else {}
                    self.logger.info(
                        "Loaded information analysis results for correlation analysis"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to load information results: {e}")

            # Try to get accuracy from recent training
            if hasattr(self.model, "last_test_accuracy"):
                accuracy = self.model.last_test_accuracy
            elif hasattr(self, "last_test_accuracy"):
                accuracy = self.last_test_accuracy

            correlation_analyzer.analyze(
                save_dir=os.path.join(self.save_root, "correlation_analysis"),
                information_results=information_results,
                accuracy=accuracy,
            )
