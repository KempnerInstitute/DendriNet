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
from dendritic_modeling.analysis.metrics.adversarial_robustness import (
    AdversarialRobustnessAnalyzer,
)
from dendritic_modeling.analysis.metrics.layer_contribution import (
    SingleLayerContributionAnalyzer,
)
from dendritic_modeling.analysis.metrics.noise_perturbation import (
    NoisePerturbationAnalyzer,
)
from dendritic_modeling.analysis.metrics.performance import PerformanceAnalyzer
from dendritic_modeling.analysis.tools.compartment_snr import CompartmentSNRAnalyzer
from dendritic_modeling.analysis.tools.compartment_statistics import (
    CompartmentStatisticsAnalyzer,
)
from dendritic_modeling.analysis.tools.correlation_analysis import CorrelationAnalyzer
from dendritic_modeling.analysis.tools.dendritic_timetraces import (
    DendriticTimetracesAnalyzer,
)
from dendritic_modeling.analysis.tools.inhibitory_specialization import (
    InhibitorySpecializationAnalyzer,
)
from dendritic_modeling.analysis.tools.jacobian_spectrum import JacobianSpectrumAnalyzer
from dendritic_modeling.analysis.tools.multiplicative_gain import (
    MultiplicativeGainAnalyzer,
)
from dendritic_modeling.analysis.tools.reactivation_dynamics import (
    ReactivationDynamicsAnalyzer,
)
from dendritic_modeling.analysis.tools.receptive_fields import ReceptiveFieldAnalyzer
from dendritic_modeling.analysis.tools.routing_analysis import RoutingAnalyzer
from dendritic_modeling.analysis.tools.synaptic_activation import (
    SynapticActivationAnalyzer,
)
from dendritic_modeling.analysis.tools.synaptic_turnover import SynapticTurnoverAnalyzer
from dendritic_modeling.analysis.tools.weight import WeightAnalyzer
from dendritic_modeling.config import AnalysisConfig
from dendritic_modeling.models import BaseModel, Regressor


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

        self._sanitize_performance_metrics_for_task()

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

        # Initialize Reactivation Dynamics Analysis
        if (
            getattr(self.config, "reactivation_dynamics_analysis", None)
            and self.config.reactivation_dynamics_analysis.enabled
        ):
            self.reactivation_dynamics_analyzer = ReactivationDynamicsAnalyzer(
                self.config.reactivation_dynamics_analysis.params
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

        # Initialize Routing Analysis (recurrent models only)
        if (
            hasattr(self.config, "routing_analysis")
            and self.config.routing_analysis.enabled
        ):
            self.routing_analyzer = RoutingAnalyzer()

        # Initialize Inhibitory Specialization Analysis (recurrent E-I models only)
        inh_spec_cfg = getattr(self.config, "inhibitory_specialization_analysis", None)
        if inh_spec_cfg is not None and inh_spec_cfg.enabled:
            self.inhibitory_specialization_analyzer = InhibitorySpecializationAnalyzer(
                inh_spec_cfg.params
            )

        # Initialize Jacobian Spectrum Analysis (recurrent dendritic models only)
        jac_cfg = getattr(self.config, "jacobian_spectrum_analysis", None)
        if jac_cfg is not None and jac_cfg.enabled:
            self.jacobian_spectrum_analyzer = JacobianSpectrumAnalyzer(jac_cfg.params)

        # Initialize Dendritic Timetraces Analysis (recurrent dendritic models only)
        ddt_cfg = getattr(self.config, "dendritic_timetraces_analysis", None)
        if ddt_cfg is not None and ddt_cfg.enabled:
            self.dendritic_timetraces_analyzer = DendriticTimetracesAnalyzer(
                ddt_cfg.params
            )

        # Initialize Adversarial Robustness Analysis
        if (
            hasattr(self.config, "adversarial_robustness_analysis")
            and self.config.adversarial_robustness_analysis.enabled
        ):
            self.adversarial_robustness_analyzer = AdversarialRobustnessAnalyzer(
                self.config.adversarial_robustness_analysis.params
            )

    def _sanitize_performance_metrics_for_task(self) -> None:
        """Disable incompatible default metrics for regression models."""
        perf_cfg = getattr(self.config, "performance_analysis", None)
        params = getattr(perf_cfg, "params", None) if perf_cfg is not None else None
        if params is None:
            return

        is_regression = (
            isinstance(self.model, Regressor)
            or getattr(self.model, "task", None) == "regression"
        )
        if is_regression:
            changed = []
            for name in ("accuracy", "auc", "categorical_loglikelihood"):
                if getattr(params, name, False):
                    setattr(params, name, False)
                    changed.append(name)
            if changed:
                for name in ("mse", "cosine_similarity"):
                    if not getattr(params, name, False):
                        setattr(params, name, True)
                self.logger.info(
                    "Regression model detected; disabled incompatible performance "
                    "metrics %s and enabled mse/cosine_similarity",
                    changed,
                )

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
        runtime = (
            getattr(self.config.runtime, "training", None)
            if training
            else getattr(self.config.runtime, "final", None)
        )

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

            perf_cfg = self.config.performance_analysis
            splits = (
                getattr(perf_cfg, "training_splits", None)
                if training
                else getattr(perf_cfg, "final_splits", None)
            )

            self.performance_analyzer.analyze(
                model=self.model,
                train_ds=self.data["train"],
                valid_ds=self.data["valid"],
                test_ds=self.data["test"],
                device=self.device,
                save_path=save_path,
                filename=filename,
                training=training,
                runtime=runtime,
                splits=splits,
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
                runtime=runtime,
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
                runtime=runtime,
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
                runtime=runtime,
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
                runtime=runtime,
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

        # Reactivation Dynamics Analysis
        rd_cfg = getattr(self.config, "reactivation_dynamics_analysis", None)
        if (
            rd_cfg is not None
            and rd_cfg.enabled
            and ((rd_cfg.training and training) or not training)
        ):
            self.reactivation_dynamics_analyzer.analyze(
                model=self.model,
                train_ds=self.data.get("train"),
                valid_ds=self.data.get("valid"),
                test_ds=self.data.get("test"),
                device=self.device,
                save_path=os.path.join(self.save_root, "reactivation_dynamics"),
                filename=filename,
                training=training,
                runtime=runtime,
            )

        # Compartment Statistics Analysis
        if self.config.compartment_statistics_analysis.enabled and (
            (self.config.compartment_statistics_analysis.training and training)
            or not training
        ):
            self.compartment_statistics_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(
                    self.save_root, "compartment_statistics_analysis"
                ),
                filename=filename,
                training=training,
                runtime=runtime,
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
                training=training,
                runtime=runtime,
            )

        # Multiplicative Gain Analysis
        if self.config.multiplicative_gain_analysis.enabled and (
            (self.config.multiplicative_gain_analysis.training and training)
            or not training
        ):
            self.multiplicative_gain_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "multiplicative_gain_analysis"),
                filename=filename,
                training=training,
                runtime=runtime,
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
                runtime=runtime,
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
                runtime=runtime,
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
                runtime=runtime,
            )
            self.logger.info("Correlation analysis completed successfully")

        # Routing Analysis (recurrent models)
        if (
            hasattr(self.config, "routing_analysis")
            and self.config.routing_analysis.enabled
            and hasattr(self, "routing_analyzer")
            and (
                (
                    hasattr(self.config.routing_analysis, "training")
                    and self.config.routing_analysis.training
                    and training
                )
                or not training
            )
        ):
            self.logger.info("Starting routing analysis...")
            self.routing_analyzer.analyze(
                model=self.model,
                data=self.data["test"],
                device=self.device,
                save_path=os.path.join(self.save_root, "routing_analysis"),
                filename=filename,
            )
            self.logger.info("Routing analysis completed successfully")

        # Inhibitory Specialization Analysis (recurrent E-I models)
        inh_spec_cfg = getattr(self.config, "inhibitory_specialization_analysis", None)
        if (
            inh_spec_cfg is not None
            and inh_spec_cfg.enabled
            and hasattr(self, "inhibitory_specialization_analyzer")
            and (
                (getattr(inh_spec_cfg, "training", False) and training) or not training
            )
        ):
            self.logger.info("Starting inhibitory specialization analysis...")
            self.inhibitory_specialization_analyzer.analyze(
                model=self.model,
                device=self.device,
                save_path=os.path.join(self.save_root, "inhibitory_specialization"),
                filename=filename,
            )
            self.logger.info(
                "Inhibitory specialization analysis completed successfully"
            )

        # Jacobian Spectrum Analysis (recurrent dendritic models)
        jac_cfg = getattr(self.config, "jacobian_spectrum_analysis", None)
        if (
            jac_cfg is not None
            and jac_cfg.enabled
            and hasattr(self, "jacobian_spectrum_analyzer")
            and ((getattr(jac_cfg, "training", False) and training) or not training)
        ):
            self.logger.info("Starting jacobian spectrum analysis...")
            self.jacobian_spectrum_analyzer.analyze(
                model=self.model,
                device=self.device,
                save_path=os.path.join(self.save_root, "jacobian_spectrum"),
                filename=filename,
            )
            self.logger.info("Jacobian spectrum analysis completed successfully")

        # Dendritic Timetraces Analysis (recurrent dendritic models)
        ddt_cfg = getattr(self.config, "dendritic_timetraces_analysis", None)
        if (
            ddt_cfg is not None
            and ddt_cfg.enabled
            and hasattr(self, "dendritic_timetraces_analyzer")
            and ((getattr(ddt_cfg, "training", False) and training) or not training)
        ):
            self.logger.info("Starting dendritic timetraces analysis...")
            self.dendritic_timetraces_analyzer.analyze(
                model=self.model,
                device=self.device,
                save_path=os.path.join(self.save_root, "dendritic_timetraces"),
                filename=filename,
            )
            self.logger.info("Dendritic timetraces analysis completed successfully")

        # Adversarial Robustness Analysis
        if (
            hasattr(self.config, "adversarial_robustness_analysis")
            and self.config.adversarial_robustness_analysis.enabled
            and hasattr(self, "adversarial_robustness_analyzer")
            and (
                (
                    hasattr(self.config.adversarial_robustness_analysis, "training")
                    and self.config.adversarial_robustness_analysis.training
                    and training
                )
                or not training
            )
        ):
            self.logger.info("Starting adversarial robustness analysis...")
            self.adversarial_robustness_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=os.path.join(
                    self.save_root, "adversarial_robustness_analysis"
                ),
                filename=filename,
                runtime=runtime,
            )
            self.logger.info("Adversarial robustness analysis completed successfully")
