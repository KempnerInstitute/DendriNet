"""
Analysis manager module.

This module contains the AnalysisManager class which coordinates and orchestrates
all the different analysis tools in the dendritic modeling package.
"""

import logging
import os
from typing import Optional

import torch

from dendritic_modeling.analysis.managers.analysis_plan import (
    _ANALYZER_INITIALIZERS,
    _DENDRITIC_TIMETRACES_ANALYSIS,
    _INHIBITORY_SPECIALIZATION_ANALYSIS,
    _JACOBIAN_SPECTRUM_ANALYSIS,
    _MODEL_ONLY_ANALYSES,
    _ORDERED_ANALYSIS_SEQUENCE,
    _POST_REACTIVATION_TEST_DATASET_ANALYSES,
    _PRE_WEIGHT_TEST_DATASET_ANALYSES,
    _ModelOnlyAnalysisRunner,
    _registered_analyzer_spec,
    _TestDatasetAnalysisRunner,
)
from dendritic_modeling.config import AnalysisConfig
from dendritic_modeling.models import BaseModel, Regressor

__all__ = [
    "_DENDRITIC_TIMETRACES_ANALYSIS",
    "_INHIBITORY_SPECIALIZATION_ANALYSIS",
    "_JACOBIAN_SPECTRUM_ANALYSIS",
    "_MODEL_ONLY_ANALYSES",
    "_POST_REACTIVATION_TEST_DATASET_ANALYSES",
    "_PRE_WEIGHT_TEST_DATASET_ANALYSES",
    "AnalysisManager",
]


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
            setattr(
                self,
                initializer.attribute_name,
                initializer.build(model=self.model, params=params),
            )
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

    def _should_run_registered_analysis(
        self,
        registry_name: str,
        *,
        training: bool,
        require_analyzer: bool = False,
    ) -> bool:
        """Return whether a registered analyzer should run in this phase."""
        if not self._should_run_config(self._analysis_config(registry_name), training):
            return False
        if require_analyzer and not self._has_analysis_analyzer(registry_name):
            return False
        return True

    def _run_test_dataset_analysis(
        self,
        *,
        registry_name: str,
        filename: str,
        training: bool,
        runtime: object | None,
        pass_training: bool = False,
        require_analyzer: bool = False,
    ) -> None:
        """Run an analyzer with the common test-dataset call signature."""
        if not self._should_run_registered_analysis(
            registry_name,
            training=training,
            require_analyzer=require_analyzer,
        ):
            return

        kwargs = {
            "model": self.model,
            "test_dataset": self.data["test"],
            "device": self.device,
            "save_path": self._analysis_save_path(registry_name),
            "filename": filename,
            "runtime": runtime,
        }
        if pass_training:
            kwargs["training"] = training

        self._analysis_analyzer(registry_name).analyze(**kwargs)

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
            registry_name=runner.registry_name,
            filename=filename,
            training=training,
            runtime=runtime,
            pass_training=runner.pass_training,
            require_analyzer=runner.require_analyzer,
        )

    def _run_test_dataset_analysis_runners(
        self,
        runners: tuple[_TestDatasetAnalysisRunner, ...],
        *,
        filename: str,
        training: bool,
        runtime: object | None,
    ) -> None:
        """Run a table of test-dataset analyzers in declaration order."""
        for runner in runners:
            self._run_test_dataset_analysis_runner(
                runner,
                filename=filename,
                training=training,
                runtime=runtime,
            )

    def _run_logged_test_dataset_analysis(
        self,
        *,
        registry_name: str,
        filename: str,
        training: bool,
        runtime: object | None,
        start_message: str,
        complete_message: str,
        require_analyzer: bool = False,
    ) -> None:
        """Run a common test-dataset analyzer with start/complete logging."""
        if not self._should_run_registered_analysis(
            registry_name,
            training=training,
            require_analyzer=require_analyzer,
        ):
            return

        self.logger.info(start_message)
        self._run_test_dataset_analysis(
            registry_name=registry_name,
            filename=filename,
            training=training,
            runtime=runtime,
            require_analyzer=require_analyzer,
        )
        self.logger.info(complete_message)

    def _run_logged_data_analysis(
        self,
        *,
        registry_name: str,
        filename: str,
        training: bool,
        start_message: str,
        complete_message: str,
        runtime: object | None = None,
        pass_runtime: bool = False,
        require_analyzer: bool = True,
    ) -> None:
        """Run a logged analyzer with the common data= test-set signature."""
        if not self._should_run_registered_analysis(
            registry_name,
            training=training,
            require_analyzer=require_analyzer,
        ):
            return

        kwargs = {
            "model": self.model,
            "data": self.data["test"],
            "device": self.device,
            "save_path": self._analysis_save_path(registry_name),
            "filename": filename,
        }
        if pass_runtime:
            kwargs["runtime"] = runtime

        self.logger.info(start_message)
        self._analysis_analyzer(registry_name).analyze(**kwargs)
        self.logger.info(complete_message)

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
        self._run_logged_test_dataset_analysis(
            registry_name="information",
            filename=filename,
            training=training,
            runtime=runtime,
            start_message="Starting information analysis...",
            complete_message="Information analysis completed successfully",
        )

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
        if not self._should_run_registered_analysis(
            "local_rule_components",
            training=training,
            require_analyzer=True,
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
        self._run_logged_test_dataset_analysis(
            registry_name="correlation",
            filename=filename,
            training=training,
            runtime=runtime,
            start_message="Starting correlation analysis...",
            complete_message="Correlation analysis completed successfully",
            require_analyzer=True,
        )

    def _run_routing_analysis(self, *, filename: str, training: bool) -> None:
        """Run recurrent routing analysis."""
        self._run_logged_data_analysis(
            registry_name="routing",
            filename=filename,
            training=training,
            start_message="Starting routing analysis...",
            complete_message="Routing analysis completed successfully",
        )

    def _run_model_only_analysis(
        self,
        *,
        registry_name: str,
        filename: str,
        training: bool,
        start_message: str,
        complete_message: str,
    ) -> None:
        """Run an analyzer with the common model/device/save-path signature."""
        if not self._should_run_registered_analysis(
            registry_name,
            training=training,
            require_analyzer=True,
        ):
            return

        self.logger.info(start_message)
        self._analysis_analyzer(registry_name).analyze(
            model=self.model,
            device=self.device,
            save_path=self._analysis_save_path(registry_name),
            filename=filename,
        )
        self.logger.info(complete_message)

    def _run_model_only_analysis_runner(
        self,
        runner: _ModelOnlyAnalysisRunner,
        *,
        filename: str,
        training: bool,
    ) -> None:
        """Run a table-declared model-only analyzer."""
        self._run_model_only_analysis(
            registry_name=runner.registry_name,
            filename=filename,
            training=training,
            start_message=runner.start_message,
            complete_message=runner.complete_message,
        )

    def _run_model_only_analysis_runners(
        self,
        runners: tuple[_ModelOnlyAnalysisRunner, ...],
        *,
        filename: str,
        training: bool,
    ) -> None:
        """Run a table of model-only analyzers in declaration order."""
        for runner in runners:
            self._run_model_only_analysis_runner(
                runner,
                filename=filename,
                training=training,
            )

    def _run_inhibitory_specialization_analysis(
        self, *, filename: str, training: bool
    ) -> None:
        """Run inhibitory specialization analysis."""
        self._run_model_only_analysis_runner(
            _INHIBITORY_SPECIALIZATION_ANALYSIS,
            filename=filename,
            training=training,
        )

    def _run_jacobian_spectrum_analysis(self, *, filename: str, training: bool) -> None:
        """Run jacobian spectrum analysis."""
        self._run_model_only_analysis_runner(
            _JACOBIAN_SPECTRUM_ANALYSIS,
            filename=filename,
            training=training,
        )

    def _run_dendritic_timetraces_analysis(
        self, *, filename: str, training: bool
    ) -> None:
        """Run dendritic timetraces analysis."""
        self._run_model_only_analysis_runner(
            _DENDRITIC_TIMETRACES_ANALYSIS,
            filename=filename,
            training=training,
        )

    def _run_spike_train_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run spike-train analysis."""
        self._run_logged_data_analysis(
            registry_name="spike_train",
            filename=filename,
            training=training,
            start_message="Starting spike train analysis...",
            complete_message="Spike train analysis completed successfully",
            runtime=runtime,
            pass_runtime=True,
        )

    def _run_adversarial_robustness_analysis(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run adversarial robustness analysis."""
        self._run_logged_test_dataset_analysis(
            registry_name="adversarial_robustness",
            filename=filename,
            training=training,
            runtime=runtime,
            start_message="Starting adversarial robustness analysis...",
            complete_message="Adversarial robustness analysis completed successfully",
            require_analyzer=True,
        )

    def _run_ordered_analysis_sequence(
        self, *, filename: str, training: bool, runtime: object | None
    ) -> None:
        """Run analyzer wrappers in the historically defined order."""
        for step in _ORDERED_ANALYSIS_SEQUENCE:
            step.run(
                self,
                filename=filename,
                training=training,
                runtime=runtime,
            )

    def run_analysis(self, filename: str = "final", training: bool = False):
        """
        Run all enabled analysis tools.

        Args:
            filename: Base filename for saving analysis results
            training: Whether this is being called during training (affects which analyses run)
        """
        runtime = self._analysis_runtime(training)
        self._run_ordered_analysis_sequence(
            filename=filename,
            training=training,
            runtime=runtime,
        )
