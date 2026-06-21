"""
Analysis manager module.

This module contains the AnalysisManager class which coordinates and orchestrates
all the different analysis tools in the dendritic modeling package.
"""

import logging
import os
from dataclasses import dataclass
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
from dendritic_modeling.analysis.registry import get_analyzer_spec
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
from dendritic_modeling.analysis.tools.local_rule_components import (
    LocalRuleComponentAnalyzer,
)
from dendritic_modeling.analysis.tools.multiplicative_gain import (
    MultiplicativeGainAnalyzer,
)
from dendritic_modeling.analysis.tools.reactivation_dynamics import (
    ReactivationDynamicsAnalyzer,
)
from dendritic_modeling.analysis.tools.receptive_fields import ReceptiveFieldAnalyzer
from dendritic_modeling.analysis.tools.routing_analysis import RoutingAnalyzer
from dendritic_modeling.analysis.tools.spike_trains import SpikeTrainAnalyzer
from dendritic_modeling.analysis.tools.synaptic_activation import (
    SynapticActivationAnalyzer,
)
from dendritic_modeling.analysis.tools.synaptic_turnover import SynapticTurnoverAnalyzer
from dendritic_modeling.analysis.tools.weight import WeightAnalyzer
from dendritic_modeling.config import AnalysisConfig
from dendritic_modeling.models import BaseModel, Regressor


@dataclass(frozen=True)
class _AnalyzerInitializer:
    """Construction metadata used by AnalysisManager."""

    registry_name: str
    analyzer_cls: type
    pass_model: bool = False
    no_params: bool = False

    @property
    def config_field(self) -> str:
        spec = get_analyzer_spec(self.registry_name)
        if spec is None:
            raise KeyError(f"Analyzer {self.registry_name!r} is not registered")
        return spec.config_field

    @property
    def attribute_name(self) -> str:
        spec = get_analyzer_spec(self.registry_name)
        if spec is None:
            raise KeyError(f"Analyzer {self.registry_name!r} is not registered")
        return spec.attribute_name


@dataclass(frozen=True)
class _TestDatasetAnalysisRunner:
    """Execution metadata for analyzers sharing the test-dataset signature."""

    config_field: str
    analyzer_attr: str
    save_subdir: str
    pass_training: bool = False
    require_analyzer: bool = False


_ANALYZER_INITIALIZERS = (
    _AnalyzerInitializer("performance", PerformanceAnalyzer),
    _AnalyzerInitializer("information", InformationAnalyzer),
    _AnalyzerInitializer("synapse_turnover", SynapticTurnoverAnalyzer, pass_model=True),
    _AnalyzerInitializer("noise_perturbation", NoisePerturbationAnalyzer),
    _AnalyzerInitializer("single_layer_contribution", SingleLayerContributionAnalyzer),
    _AnalyzerInitializer("ablation", AblationAnalyzer),
    _AnalyzerInitializer("weight", WeightAnalyzer),
    _AnalyzerInitializer("compartment_statistics", CompartmentStatisticsAnalyzer),
    _AnalyzerInitializer("compartment_snr", CompartmentSNRAnalyzer),
    _AnalyzerInitializer("multiplicative_gain", MultiplicativeGainAnalyzer),
    _AnalyzerInitializer("receptive_field", ReceptiveFieldAnalyzer),
    _AnalyzerInitializer("reactivation_dynamics", ReactivationDynamicsAnalyzer),
    _AnalyzerInitializer("synaptic_activation", SynapticActivationAnalyzer),
    _AnalyzerInitializer("branch_activation", BranchActivationAnalyzer),
    _AnalyzerInitializer("local_rule_components", LocalRuleComponentAnalyzer),
    _AnalyzerInitializer("correlation", CorrelationAnalyzer),
    _AnalyzerInitializer("routing", RoutingAnalyzer, no_params=True),
    _AnalyzerInitializer("inhibitory_specialization", InhibitorySpecializationAnalyzer),
    _AnalyzerInitializer("jacobian_spectrum", JacobianSpectrumAnalyzer),
    _AnalyzerInitializer("dendritic_timetraces", DendriticTimetracesAnalyzer),
    _AnalyzerInitializer("spike_train", SpikeTrainAnalyzer),
    _AnalyzerInitializer("adversarial_robustness", AdversarialRobustnessAnalyzer),
)


_PRE_WEIGHT_TEST_DATASET_ANALYSES = (
    _TestDatasetAnalysisRunner(
        "noise_perturbation_analysis",
        "noise_perturbation_analyzer",
        "noise_perturbation_analysis",
    ),
    _TestDatasetAnalysisRunner(
        "single_layer_contribution",
        "single_layer_analyzer",
        "single_layer_contribution",
    ),
    _TestDatasetAnalysisRunner(
        "ablation_analysis",
        "ablation_analyzer",
        "ablation_analysis",
        pass_training=True,
    ),
)


_POST_REACTIVATION_TEST_DATASET_ANALYSES = (
    _TestDatasetAnalysisRunner(
        "compartment_statistics_analysis",
        "compartment_statistics_analyzer",
        "compartment_statistics_analysis",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner(
        "compartment_snr_analysis",
        "compartment_snr_analyzer",
        "compartment_snr_analysis",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner(
        "multiplicative_gain_analysis",
        "multiplicative_gain_analyzer",
        "multiplicative_gain_analysis",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner(
        "receptive_field_analysis",
        "receptive_field_analyzer",
        "receptive_field_analysis",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner(
        "synaptic_activation_analysis",
        "synaptic_activation_analyzer",
        "synaptic_activation_analysis",
    ),
    _TestDatasetAnalysisRunner(
        "branch_activation_analysis",
        "branch_activation_analyzer",
        "branch_activation_analysis",
    ),
)


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
        self._initialize_configured_analyzers()

    def _initialize_configured_analyzers(self) -> None:
        """Instantiate enabled analyzers from registry-backed metadata."""
        for initializer in _ANALYZER_INITIALIZERS:
            cfg = getattr(self.config, initializer.config_field, None)
            if cfg is None or not getattr(cfg, "enabled", False):
                continue
            params = getattr(cfg, "params", None)
            if initializer.no_params:
                analyzer = initializer.analyzer_cls()
            elif initializer.pass_model:
                analyzer = initializer.analyzer_cls(model=self.model, params=params)
            else:
                analyzer = initializer.analyzer_cls(params)
            setattr(self, initializer.attribute_name, analyzer)
        self.correlation_analysis_enabled = hasattr(self, "correlation_analyzer")

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

    @staticmethod
    def _should_run_config(cfg: object | None, training: bool) -> bool:
        """Return whether an analysis config is enabled for this phase."""
        if cfg is None or not getattr(cfg, "enabled", False):
            return False
        return (getattr(cfg, "training", False) and training) or not training

    def _save_path(self, *parts: str, create: bool = False) -> str:
        """Build an analysis output path and optionally create it."""
        path = os.path.join(self.save_root, *parts)
        if create:
            os.makedirs(path, exist_ok=True)
        return path

    def _run_test_dataset_analysis(
        self,
        *,
        cfg: object | None,
        analyzer_attr: str,
        save_subdir: str,
        filename: str,
        training: bool,
        runtime: object | None,
        pass_training: bool = False,
        require_analyzer: bool = False,
    ) -> None:
        """Run an analyzer with the common test-dataset call signature."""
        if not self._should_run_config(cfg, training):
            return
        if require_analyzer and not hasattr(self, analyzer_attr):
            return

        kwargs = {
            "model": self.model,
            "test_dataset": self.data["test"],
            "device": self.device,
            "save_path": self._save_path(save_subdir),
            "filename": filename,
            "runtime": runtime,
        }
        if pass_training:
            kwargs["training"] = training

        getattr(self, analyzer_attr).analyze(**kwargs)

    def _run_test_dataset_analysis_runner(
        self,
        runner: _TestDatasetAnalysisRunner,
        *,
        filename: str,
        training: bool,
        runtime: object | None,
    ) -> None:
        """Run a table-declared test-dataset analyzer."""
        self._run_test_dataset_analysis(
            cfg=getattr(self.config, runner.config_field, None),
            analyzer_attr=runner.analyzer_attr,
            save_subdir=runner.save_subdir,
            filename=filename,
            training=training,
            runtime=runtime,
            pass_training=runner.pass_training,
            require_analyzer=runner.require_analyzer,
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
        if self._should_run_config(self.config.performance_analysis, training):
            # Determine save path based on whether this is training or final analysis
            if training:
                # Save epoch results in a subfolder
                save_path = self._save_path("performance", "epochs", create=True)
            else:
                # Save final results in main performance folder
                save_path = self._save_path("performance")

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
        if self._should_run_config(self.config.information_analysis, training):
            self.logger.info("Starting information analysis...")
            self.information_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=self._save_path("information_analysis"),
                filename=filename,
                runtime=runtime,
            )
            self.logger.info("Information analysis completed successfully")

        # Synaptic Turnover Analysis
        if self._should_run_config(self.config.synapse_turnover_analysis, training):
            self.syn_turnover_analyzer.analyze(
                model=self.model,
                training=training,
                save_path=self._save_path("synapse_turnover_analysis"),
                filename=filename,
            )

        for runner in _PRE_WEIGHT_TEST_DATASET_ANALYSES:
            self._run_test_dataset_analysis_runner(
                runner,
                filename=filename,
                training=training,
                runtime=runtime,
            )

        # Weight Analysis
        if self._should_run_config(self.config.weight_analysis, training):
            self.weight_analyzer.analyze(
                model=self.model,
                save_path=self._save_path("weight_analysis"),
                filename=filename,
                training=training,
            )

        # Reactivation Dynamics Analysis
        rd_cfg = getattr(self.config, "reactivation_dynamics_analysis", None)
        if self._should_run_config(rd_cfg, training):
            self.reactivation_dynamics_analyzer.analyze(
                model=self.model,
                train_ds=self.data.get("train"),
                valid_ds=self.data.get("valid"),
                test_ds=self.data.get("test"),
                device=self.device,
                save_path=self._save_path("reactivation_dynamics"),
                filename=filename,
                training=training,
                runtime=runtime,
            )

        for runner in _POST_REACTIVATION_TEST_DATASET_ANALYSES:
            self._run_test_dataset_analysis_runner(
                runner,
                filename=filename,
                training=training,
                runtime=runtime,
            )

        # Local Rule Component Analysis
        lrc_cfg = getattr(self.config, "local_rule_component_analysis", None)
        if (
            lrc_cfg is not None
            and self._should_run_config(lrc_cfg, training)
            and hasattr(self, "local_rule_component_analyzer")
        ):
            self.local_rule_component_analyzer.analyze(
                model=self.model,
                train_ds=self.data.get("train"),
                valid_ds=self.data.get("valid"),
                test_dataset=self.data.get("test"),
                device=self.device,
                save_path=self._save_path("local_rule_components"),
                filename=filename,
                training=training,
                runtime=runtime,
            )

        # Correlation Analysis
        if self.correlation_analysis_enabled and self._should_run_config(
            self.config.correlation_analysis, training
        ):
            self.logger.info("Starting correlation analysis...")
            self.correlation_analyzer.analyze(
                model=self.model,
                test_dataset=self.data["test"],
                device=self.device,
                save_path=self._save_path("correlation_analysis"),
                filename=filename,
                runtime=runtime,
            )
            self.logger.info("Correlation analysis completed successfully")

        # Routing Analysis (recurrent models)
        if self._should_run_config(
            getattr(self.config, "routing_analysis", None), training
        ) and hasattr(self, "routing_analyzer"):
            self.logger.info("Starting routing analysis...")
            self.routing_analyzer.analyze(
                model=self.model,
                data=self.data["test"],
                device=self.device,
                save_path=self._save_path("routing_analysis"),
                filename=filename,
            )
            self.logger.info("Routing analysis completed successfully")

        # Inhibitory Specialization Analysis (recurrent E-I models)
        inh_spec_cfg = getattr(self.config, "inhibitory_specialization_analysis", None)
        if self._should_run_config(inh_spec_cfg, training) and hasattr(
            self, "inhibitory_specialization_analyzer"
        ):
            self.logger.info("Starting inhibitory specialization analysis...")
            self.inhibitory_specialization_analyzer.analyze(
                model=self.model,
                device=self.device,
                save_path=self._save_path("inhibitory_specialization"),
                filename=filename,
            )
            self.logger.info(
                "Inhibitory specialization analysis completed successfully"
            )

        # Jacobian Spectrum Analysis (recurrent dendritic models)
        jac_cfg = getattr(self.config, "jacobian_spectrum_analysis", None)
        if self._should_run_config(jac_cfg, training) and hasattr(
            self, "jacobian_spectrum_analyzer"
        ):
            self.logger.info("Starting jacobian spectrum analysis...")
            self.jacobian_spectrum_analyzer.analyze(
                model=self.model,
                device=self.device,
                save_path=self._save_path("jacobian_spectrum"),
                filename=filename,
            )
            self.logger.info("Jacobian spectrum analysis completed successfully")

        # Dendritic Timetraces Analysis (recurrent dendritic models)
        ddt_cfg = getattr(self.config, "dendritic_timetraces_analysis", None)
        if self._should_run_config(ddt_cfg, training) and hasattr(
            self, "dendritic_timetraces_analyzer"
        ):
            self.logger.info("Starting dendritic timetraces analysis...")
            self.dendritic_timetraces_analyzer.analyze(
                model=self.model,
                device=self.device,
                save_path=self._save_path("dendritic_timetraces"),
                filename=filename,
            )
            self.logger.info("Dendritic timetraces analysis completed successfully")

        # Spike Train Analysis (spiking recurrent models)
        spike_cfg = getattr(self.config, "spike_train_analysis", None)
        if self._should_run_config(spike_cfg, training) and hasattr(
            self, "spike_train_analyzer"
        ):
            self.logger.info("Starting spike train analysis...")
            self.spike_train_analyzer.analyze(
                model=self.model,
                data=self.data["test"],
                device=self.device,
                save_path=self._save_path("spike_train_analysis"),
                filename=filename,
                runtime=runtime,
            )
            self.logger.info("Spike train analysis completed successfully")

        # Adversarial Robustness Analysis
        if self._should_run_config(
            getattr(self.config, "adversarial_robustness_analysis", None),
            training,
        ) and hasattr(self, "adversarial_robustness_analyzer"):
            self.logger.info("Starting adversarial robustness analysis...")
            self._run_test_dataset_analysis(
                cfg=getattr(self.config, "adversarial_robustness_analysis", None),
                analyzer_attr="adversarial_robustness_analyzer",
                save_subdir="adversarial_robustness_analysis",
                filename=filename,
                training=training,
                runtime=runtime,
                require_analyzer=True,
            )
            self.logger.info("Adversarial robustness analysis completed successfully")
