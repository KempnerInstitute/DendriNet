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
from dendritic_modeling.analysis.registry import AnalyzerSpec, get_analyzer_spec
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


def _registered_analyzer_spec(registry_name: str) -> AnalyzerSpec:
    """Return registered analyzer metadata or fail loudly on registry drift."""
    spec = get_analyzer_spec(registry_name)
    if spec is None:
        raise KeyError(f"Analyzer {registry_name!r} is not registered")
    return spec


@dataclass(frozen=True)
class _AnalyzerInitializer:
    """Construction metadata used by AnalysisManager."""

    registry_name: str
    analyzer_cls: type
    pass_model: bool = False
    no_params: bool = False

    @property
    def config_field(self) -> str:
        return _registered_analyzer_spec(self.registry_name).config_field

    @property
    def attribute_name(self) -> str:
        return _registered_analyzer_spec(self.registry_name).attribute_name


@dataclass(frozen=True)
class _TestDatasetAnalysisRunner:
    """Execution metadata for analyzers sharing the test-dataset signature."""

    registry_name: str
    pass_training: bool = False
    require_analyzer: bool = False

    @property
    def config_field(self) -> str:
        return _registered_analyzer_spec(self.registry_name).config_field

    @property
    def analyzer_attr(self) -> str:
        return _registered_analyzer_spec(self.registry_name).attribute_name

    @property
    def save_subdir(self) -> str:
        return _registered_analyzer_spec(self.registry_name).save_subdir


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
    _TestDatasetAnalysisRunner("noise_perturbation"),
    _TestDatasetAnalysisRunner("single_layer_contribution"),
    _TestDatasetAnalysisRunner(
        "ablation",
        pass_training=True,
    ),
)


_POST_REACTIVATION_TEST_DATASET_ANALYSES = (
    _TestDatasetAnalysisRunner(
        "compartment_statistics",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner(
        "compartment_snr",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner(
        "multiplicative_gain",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner(
        "receptive_field",
        pass_training=True,
    ),
    _TestDatasetAnalysisRunner("synaptic_activation"),
    _TestDatasetAnalysisRunner("branch_activation"),
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

    def _analysis_config(self, registry_name: str) -> object | None:
        """Return the config object for a registered analyzer."""
        spec = _registered_analyzer_spec(registry_name)
        return getattr(self.config, spec.config_field, None)

    def _analysis_analyzer(self, registry_name: str) -> object:
        """Return the analyzer instance for a registered analyzer."""
        spec = _registered_analyzer_spec(registry_name)
        return getattr(self, spec.attribute_name)

    def _has_analysis_analyzer(self, registry_name: str) -> bool:
        """Return whether the registered analyzer has been initialized."""
        spec = _registered_analyzer_spec(registry_name)
        return hasattr(self, spec.attribute_name)

    def _analysis_save_path(
        self, registry_name: str, *parts: str, create: bool = False
    ) -> str:
        """Build an output path from registry-declared analyzer metadata."""
        spec = _registered_analyzer_spec(registry_name)
        return self._save_path(spec.save_subdir, *parts, create=create)

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
            cfg=self._analysis_config(runner.registry_name),
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

    def _analysis_runtime(self, training: bool) -> object | None:
        """Return the runtime profile for the current analysis phase."""
        return (
            getattr(self.config.runtime, "training", None)
            if training
            else getattr(self.config.runtime, "final", None)
        )

    def _run_performance_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run performance analysis with split-specific options."""
        perf_cfg = self._analysis_config("performance")
        if not self._should_run_config(perf_cfg, training):
            return

        save_path = (
            self._analysis_save_path("performance", "epochs", create=True)
            if training
            else self._analysis_save_path("performance")
        )
        splits = (
            getattr(perf_cfg, "training_splits", None)
            if training
            else getattr(perf_cfg, "final_splits", None)
        )

        self._analysis_analyzer("performance").analyze(
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

    def _run_information_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run information analysis."""
        if not self._should_run_config(self._analysis_config("information"), training):
            return

        self.logger.info("Starting information analysis...")
        self._analysis_analyzer("information").analyze(
            model=self.model,
            test_dataset=self.data["test"],
            device=self.device,
            save_path=self._analysis_save_path("information"),
            filename=filename,
            runtime=runtime,
        )
        self.logger.info("Information analysis completed successfully")

    def _run_synaptic_turnover_analysis(self, *, filename: str, training: bool) -> None:
        """Run synaptic turnover analysis."""
        if not self._should_run_config(
            self._analysis_config("synapse_turnover"), training
        ):
            return

        self._analysis_analyzer("synapse_turnover").analyze(
            model=self.model,
            training=training,
            save_path=self._analysis_save_path("synapse_turnover"),
            filename=filename,
        )

    def _run_weight_analysis(self, *, filename: str, training: bool) -> None:
        """Run weight analysis."""
        if not self._should_run_config(self._analysis_config("weight"), training):
            return

        self._analysis_analyzer("weight").analyze(
            model=self.model,
            save_path=self._analysis_save_path("weight"),
            filename=filename,
            training=training,
        )

    def _run_reactivation_dynamics_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run reactivation dynamics analysis."""
        if not self._should_run_config(
            self._analysis_config("reactivation_dynamics"), training
        ):
            return

        self._analysis_analyzer("reactivation_dynamics").analyze(
            model=self.model,
            train_ds=self.data.get("train"),
            valid_ds=self.data.get("valid"),
            test_ds=self.data.get("test"),
            device=self.device,
            save_path=self._analysis_save_path("reactivation_dynamics"),
            filename=filename,
            training=training,
            runtime=runtime,
        )

    def _run_local_rule_component_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run local-rule component analysis."""
        if not (
            self._should_run_config(
                self._analysis_config("local_rule_components"), training
            )
            and self._has_analysis_analyzer("local_rule_components")
        ):
            return

        self._analysis_analyzer("local_rule_components").analyze(
            model=self.model,
            train_ds=self.data.get("train"),
            valid_ds=self.data.get("valid"),
            test_dataset=self.data.get("test"),
            device=self.device,
            save_path=self._analysis_save_path("local_rule_components"),
            filename=filename,
            training=training,
            runtime=runtime,
        )

    def _run_correlation_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run correlation analysis."""
        if not (
            self.correlation_analysis_enabled
            and self._should_run_config(self._analysis_config("correlation"), training)
        ):
            return

        self.logger.info("Starting correlation analysis...")
        self._analysis_analyzer("correlation").analyze(
            model=self.model,
            test_dataset=self.data["test"],
            device=self.device,
            save_path=self._analysis_save_path("correlation"),
            filename=filename,
            runtime=runtime,
        )
        self.logger.info("Correlation analysis completed successfully")

    def _run_routing_analysis(self, *, filename: str, training: bool) -> None:
        """Run recurrent routing analysis."""
        if not (
            self._should_run_config(self._analysis_config("routing"), training)
            and self._has_analysis_analyzer("routing")
        ):
            return

        self.logger.info("Starting routing analysis...")
        self._analysis_analyzer("routing").analyze(
            model=self.model,
            data=self.data["test"],
            device=self.device,
            save_path=self._analysis_save_path("routing"),
            filename=filename,
        )
        self.logger.info("Routing analysis completed successfully")

    def _run_inhibitory_specialization_analysis(
        self, *, filename: str, training: bool
    ) -> None:
        """Run inhibitory specialization analysis."""
        if not (
            self._should_run_config(
                self._analysis_config("inhibitory_specialization"), training
            )
            and self._has_analysis_analyzer("inhibitory_specialization")
        ):
            return

        self.logger.info("Starting inhibitory specialization analysis...")
        self._analysis_analyzer("inhibitory_specialization").analyze(
            model=self.model,
            device=self.device,
            save_path=self._analysis_save_path("inhibitory_specialization"),
            filename=filename,
        )
        self.logger.info("Inhibitory specialization analysis completed successfully")

    def _run_jacobian_spectrum_analysis(self, *, filename: str, training: bool) -> None:
        """Run jacobian spectrum analysis."""
        if not (
            self._should_run_config(
                self._analysis_config("jacobian_spectrum"), training
            )
            and self._has_analysis_analyzer("jacobian_spectrum")
        ):
            return

        self.logger.info("Starting jacobian spectrum analysis...")
        self._analysis_analyzer("jacobian_spectrum").analyze(
            model=self.model,
            device=self.device,
            save_path=self._analysis_save_path("jacobian_spectrum"),
            filename=filename,
        )
        self.logger.info("Jacobian spectrum analysis completed successfully")

    def _run_dendritic_timetraces_analysis(
        self, *, filename: str, training: bool
    ) -> None:
        """Run dendritic timetraces analysis."""
        if not (
            self._should_run_config(
                self._analysis_config("dendritic_timetraces"), training
            )
            and self._has_analysis_analyzer("dendritic_timetraces")
        ):
            return

        self.logger.info("Starting dendritic timetraces analysis...")
        self._analysis_analyzer("dendritic_timetraces").analyze(
            model=self.model,
            device=self.device,
            save_path=self._analysis_save_path("dendritic_timetraces"),
            filename=filename,
        )
        self.logger.info("Dendritic timetraces analysis completed successfully")

    def _run_spike_train_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run spike-train analysis."""
        if not (
            self._should_run_config(self._analysis_config("spike_train"), training)
            and self._has_analysis_analyzer("spike_train")
        ):
            return

        self.logger.info("Starting spike train analysis...")
        self._analysis_analyzer("spike_train").analyze(
            model=self.model,
            data=self.data["test"],
            device=self.device,
            save_path=self._analysis_save_path("spike_train"),
            filename=filename,
            runtime=runtime,
        )
        self.logger.info("Spike train analysis completed successfully")

    def _run_adversarial_robustness_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run adversarial robustness analysis."""
        if not (
            self._should_run_config(
                self._analysis_config("adversarial_robustness"), training
            )
            and self._has_analysis_analyzer("adversarial_robustness")
        ):
            return

        self.logger.info("Starting adversarial robustness analysis...")
        spec = _registered_analyzer_spec("adversarial_robustness")
        self._run_test_dataset_analysis(
            cfg=self._analysis_config("adversarial_robustness"),
            analyzer_attr=spec.attribute_name,
            save_subdir=spec.save_subdir,
            filename=filename,
            training=training,
            runtime=runtime,
            require_analyzer=True,
        )
        self.logger.info("Adversarial robustness analysis completed successfully")

    def run_analysis(self, filename: str = "final", training: bool = False):
        """
        Run all enabled analysis tools.

        Args:
            filename: Base filename for saving analysis results
            training: Whether this is being called during training (affects which analyses run)
        """
        runtime = self._analysis_runtime(training)

        self._run_performance_analysis(
            filename=filename, training=training, runtime=runtime
        )
        self._run_information_analysis(
            filename=filename, training=training, runtime=runtime
        )
        self._run_synaptic_turnover_analysis(filename=filename, training=training)

        for runner in _PRE_WEIGHT_TEST_DATASET_ANALYSES:
            self._run_test_dataset_analysis_runner(
                runner,
                filename=filename,
                training=training,
                runtime=runtime,
            )

        self._run_weight_analysis(filename=filename, training=training)
        self._run_reactivation_dynamics_analysis(
            filename=filename, training=training, runtime=runtime
        )

        for runner in _POST_REACTIVATION_TEST_DATASET_ANALYSES:
            self._run_test_dataset_analysis_runner(
                runner,
                filename=filename,
                training=training,
                runtime=runtime,
            )

        self._run_local_rule_component_analysis(
            filename=filename, training=training, runtime=runtime
        )
        self._run_correlation_analysis(
            filename=filename, training=training, runtime=runtime
        )
        self._run_routing_analysis(filename=filename, training=training)
        self._run_inhibitory_specialization_analysis(
            filename=filename, training=training
        )
        self._run_jacobian_spectrum_analysis(filename=filename, training=training)
        self._run_dendritic_timetraces_analysis(filename=filename, training=training)
        self._run_spike_train_analysis(
            filename=filename, training=training, runtime=runtime
        )
        self._run_adversarial_robustness_analysis(
            filename=filename, training=training, runtime=runtime
        )
