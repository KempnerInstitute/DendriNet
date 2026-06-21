"""Integrated information analyzer for dendritic networks.

This module provides a unified interface for computing various information-theoretic
metrics using the multivariate methods from information.py. It works directly
with the main configuration system.
"""

import logging
import time
import traceback
from typing import Any, Optional

import torch

from dendritic_modeling.analysis.core import information_config as _information_config
from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.core.information_all_branch_mixin import (
    InformationAllBranchMixin,
)
from dendritic_modeling.analysis.core.information_branch_mixin import (
    InformationBranchAnalysisMixin,
)
from dendritic_modeling.analysis.core.information_data_mixin import (
    InformationDataMixin,
    _InformationAnalysisData as _InformationAnalysisData,
)
from dendritic_modeling.analysis.core.information_dispatch_mixin import (
    InformationDispatchMixin,
)
from dendritic_modeling.analysis.core.information_enhanced_mixin import (
    InformationEnhancedAnalysisMixin,
)
from dendritic_modeling.analysis.core.information_finalization_mixin import (
    InformationFinalizationMixin,
)
from dendritic_modeling.analysis.core.information_hooks_mixin import (
    InformationHooksMixin,
)
from dendritic_modeling.analysis.core.information_lda_mixin import (
    InformationLdaComparisonMixin,
)
from dendritic_modeling.analysis.core.information_lifecycle_mixin import (
    InformationLifecycleMixin,
)
from dendritic_modeling.analysis.core.information_metrics_mixin import (
    InformationMetricsMixin,
)
from dendritic_modeling.analysis.core.information_plotting_mixin import (
    InformationPlottingMixin,
)
from dendritic_modeling.analysis.core.information_signal_mixin import (
    InformationSignalAggregationMixin,
)
from dendritic_modeling.analysis.core.information_single_branch_mixin import (
    InformationSingleBranchMixin,
)
from dendritic_modeling.analysis.core.information_synthetic_soma_mixin import (
    InformationSyntheticSomaMixin,
)
from dendritic_modeling.config import InformationAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils.information.estimators import (
    EstimatorConfig,
    InformationEstimatorFactory,
)
from dendritic_modeling.utils.information.pid import (
    check_pid_available,
)

# Backward-compatible aliases for config helpers that moved out of this module.
_InformationComponentSelection = _information_config._InformationComponentSelection
_InformationScopeSelection = _information_config._InformationScopeSelection
_InformationSignalVariantSelection = (
    _information_config._InformationSignalVariantSelection
)
_InformationLabelShuffleNullConfig = (
    _information_config._InformationLabelShuffleNullConfig
)
_InformationPidConfig = _information_config._InformationPidConfig
_InformationGaussianFisherConfig = _information_config._InformationGaussianFisherConfig
_InformationAblationAlignedConfig = (
    _information_config._InformationAblationAlignedConfig
)
_InformationRuntimeConfig = _information_config._InformationRuntimeConfig
_InformationGranularityConfig = _information_config._InformationGranularityConfig
_InformationEstimatorCoreConfig = _information_config._InformationEstimatorCoreConfig
_get_config_value = _information_config._get_config_value
_get_config_path = _information_config._get_config_path
_normalize_config_token = _information_config._normalize_config_token
_parse_information_component_selection = (
    _information_config._parse_information_component_selection
)
_parse_information_scope_selection = (
    _information_config._parse_information_scope_selection
)
_build_information_signal_variant_selection = (
    _information_config._build_information_signal_variant_selection
)
_build_information_label_shuffle_null_config = (
    _information_config._build_information_label_shuffle_null_config
)
_build_information_pid_config = _information_config._build_information_pid_config
_build_information_gaussian_fisher_config = (
    _information_config._build_information_gaussian_fisher_config
)
_build_information_ablation_aligned_config = (
    _information_config._build_information_ablation_aligned_config
)
_build_information_runtime_config = (
    _information_config._build_information_runtime_config
)
_build_information_granularity_config = (
    _information_config._build_information_granularity_config
)
_build_information_estimator_core_config = (
    _information_config._build_information_estimator_core_config
)
_build_information_estimator_method_params = (
    _information_config._build_information_estimator_method_params
)


def _build_information_estimators(
    *,
    params: Any,
    estimator_cfg: Any,
    method: str,
    n_neighbors: int,
    n_bins: int,
    copula_type: str,
    verbose: bool,
) -> dict[str, Any]:
    """Build the estimator registry used by the analyzer."""
    method_params = _build_information_estimator_method_params(
        params=params,
        estimator_cfg=estimator_cfg,
        method=method,
    )
    estimator_config = EstimatorConfig(
        method=method,
        n_neighbors=n_neighbors,
        n_bins=n_bins,
        copula_type=copula_type,
        verbose=verbose,
        method_params=method_params,
    )
    return {method: InformationEstimatorFactory.create(estimator_config)}


class InformationAnalyzer(
    InformationMetricsMixin,
    InformationHooksMixin,
    InformationDataMixin,
    InformationDispatchMixin,
    InformationSyntheticSomaMixin,
    InformationSignalAggregationMixin,
    InformationLifecycleMixin,
    InformationLdaComparisonMixin,
    InformationPlottingMixin,
    InformationFinalizationMixin,
    InformationEnhancedAnalysisMixin,
    InformationBranchAnalysisMixin,
    InformationAllBranchMixin,
    InformationSingleBranchMixin,
    AbstractAnalyzer,
):
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
            Configuration from the main config system (analysis.information)
        """
        super().__init__("InformationAnalyzer")
        get_config = _get_config_value

        # New (preferred) grouped configs (all optional for backward compatibility)
        selection_cfg = get_config(params, "selection", None)
        compute_cfg = get_config(params, "compute", None)
        estimator_cfg = get_config(params, "estimator", None)
        baselines_cfg = get_config(params, "baselines", None)
        pid_cfg = get_config(params, "pid", None)
        gaussian_fisher_cfg = get_config(compute_cfg, "gaussian_fisher", None)

        # ------------------------------------------------------------------
        # Extract configuration parameters (support both new nested and legacy flat keys).
        # ------------------------------------------------------------------
        estimator_core_config = _build_information_estimator_core_config(
            params=params,
            estimator_cfg=estimator_cfg,
            compute_cfg=compute_cfg,
        )
        self.method = estimator_core_config.method
        self.max_samples = estimator_core_config.max_samples
        self.n_neighbors = estimator_core_config.n_neighbors
        self.n_bins = estimator_core_config.n_bins
        self.copula_type = estimator_core_config.copula_type

        label_shuffle_null = _build_information_label_shuffle_null_config(
            params=params,
            baselines_cfg=baselines_cfg,
        )
        self.mi_null_shuffles = label_shuffle_null.mi_null_shuffles
        self.mi_null_seed = label_shuffle_null.mi_null_seed
        self.mi_null_components = label_shuffle_null.mi_null_components

        # ------------------------------------------------------------------
        # Legacy boolean selection (still supported).
        # ------------------------------------------------------------------
        self.compute_basic_mi = bool(getattr(params, "compute_basic_mi", False))
        self.compute_pairwise_mi = bool(getattr(params, "compute_pairwise_mi", False))
        self.compute_conditional_mi = bool(
            getattr(params, "compute_conditional_mi", False)
        )
        self.compute_pid = bool(getattr(params, "compute_pid", False))
        # Enabled via selection.components ("gaussian_fisher").
        self.compute_gaussian_fisher = False

        pid_config = _build_information_pid_config(params=params, pid_cfg=pid_cfg)
        self.pid_binarize = pid_config.pid_binarize
        self.pid_binarize_method = pid_config.pid_binarize_method
        self.pid_binarize_threshold = pid_config.pid_binarize_threshold

        ablation_aligned_config = _build_information_ablation_aligned_config(
            params=params,
            compute_cfg=compute_cfg,
        )
        self.compute_ablation_aligned_mi = (
            ablation_aligned_config.compute_ablation_aligned_mi
        )
        self.compute_upstream_unique_cmi = (
            ablation_aligned_config.compute_upstream_unique_cmi
        )
        self.compute_layer_total_proxies = (
            ablation_aligned_config.compute_layer_total_proxies
        )
        self.layer_total_topk = ablation_aligned_config.layer_total_topk

        # ------------------------------------------------------------------
        # Preferred list-based selection of components.
        #
        # If provided, overrides the individual compute_* booleans above.
        # ------------------------------------------------------------------
        components_raw = list(
            get_config(
                selection_cfg, "components", get_config(params, "components", [])
            )
            or []
        )
        component_selection = _parse_information_component_selection(components_raw)
        if component_selection is not None:
            self.compute_basic_mi = component_selection.compute_basic_mi
            self.compute_pairwise_mi = component_selection.compute_pairwise_mi
            self.compute_conditional_mi = component_selection.compute_conditional_mi
            self.compute_pid = component_selection.compute_pid
            self.compute_gaussian_fisher = component_selection.compute_gaussian_fisher
            self.compute_ablation_aligned_mi = (
                component_selection.compute_ablation_aligned_mi
            )
            self.compute_layer_total_proxies = (
                component_selection.compute_layer_total_proxies
            )
            self.compute_upstream_unique_cmi = (
                component_selection.compute_upstream_unique_cmi
            )

        # ------------------------------------------------------------------
        # PID is currently disabled.
        #
        # The existing PID implementation depends on an optional external backend
        # and assumes discrete variables; our current wiring/binarization is not
        # robust for general use (e.g., multi-class C). To avoid silently
        # producing misleading numbers, we ignore PID requests for now.
        # ------------------------------------------------------------------
        if self.compute_pid:
            self.logger.warning(
                "PID requested via selection.components, but PID is currently disabled "
                "and will be skipped."
            )
            self.compute_pid = False

        gaussian_fisher_config = _build_information_gaussian_fisher_config(
            gaussian_fisher_cfg=gaussian_fisher_cfg
        )
        self.gaussian_fisher_aggregation = (
            gaussian_fisher_config.gaussian_fisher_aggregation
        )
        self.gaussian_fisher_mi_transform = (
            gaussian_fisher_config.gaussian_fisher_mi_transform
        )
        self.gaussian_fisher_eps = gaussian_fisher_config.gaussian_fisher_eps

        runtime_config = _build_information_runtime_config(
            params=params,
            compute_cfg=compute_cfg,
        )
        self.include_vinf = runtime_config.include_vinf
        self.normalize_mi = runtime_config.normalize_mi
        if runtime_config.invalid_output_units is not None:
            self.logger.warning(
                "Unknown output_units=%r; falling back to 'bits'.",
                runtime_config.invalid_output_units,
            )
        self.output_units = runtime_config.output_units
        self.verbose = runtime_config.verbose

        granularity_config = _build_information_granularity_config(
            params=params,
            compute_cfg=compute_cfg,
        )
        self.computation_level = granularity_config.computation_level
        self.branch_aggregation = granularity_config.branch_aggregation
        self.per_neuron_analysis = granularity_config.per_neuron_analysis
        self.per_einet_analysis = granularity_config.per_einet_analysis
        self.per_layer_analysis = granularity_config.per_layer_analysis

        # Preferred list-based selection of scopes (overrides per_* flags if provided).
        #
        # `views` is a backward-compatible alias for `scopes`.
        scopes_raw = list(
            get_config(selection_cfg, "scopes", get_config(params, "scopes", [])) or []
        )
        if not scopes_raw:
            scopes_raw = list(
                get_config(selection_cfg, "views", get_config(params, "views", []))
                or []
            )
        scope_selection = _parse_information_scope_selection(scopes_raw)
        if scope_selection is not None:
            self.per_neuron_analysis = scope_selection.per_neuron_analysis
            self.per_einet_analysis = scope_selection.per_einet_analysis
            self.per_layer_analysis = scope_selection.per_layer_analysis

        signal_variant_selection = _build_information_signal_variant_selection(
            params=params,
            selection_cfg=selection_cfg,
            compute_cfg=compute_cfg,
            baselines_cfg=baselines_cfg,
        )
        self.compute_lda_weights = signal_variant_selection.compute_lda_weights
        self.lda_n_shuffles = signal_variant_selection.lda_n_shuffles
        self.random_weights_seed = signal_variant_selection.random_weights_seed

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        # Create estimators for each method
        self.estimators = _build_information_estimators(
            params=params,
            estimator_cfg=estimator_cfg,
            method=self.method,
            n_neighbors=self.n_neighbors,
            n_bins=self.n_bins,
            copula_type=self.copula_type,
            verbose=self.verbose,
        )

        # Check if PID is available
        self.pid_available = False
        if self.compute_pid:
            try:
                check_pid_available()
                self.pid_available = True
                self.logger.info("PID computation available")
            except ImportError:
                self.logger.warning(
                    "PID package not available. PID metrics will be skipped."
                )

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
        runtime: Optional[EvaluationRuntimeConfig] = None,
    ):
        """Run integrated information analysis on the model.

        Parameters
        ----------
        model : BaseModel
            The trained model to analyze
        test_dataset : torch.utils.data.Dataset
            Test dataset for analysis
        device : str, optional
            Device to run analysis on, by default "cpu"
        save_path : Optional[str], optional
            Path to save results, by default None
        filename : str, optional
            Filename for saved results, by default "final"
        """
        self.logger.info("Starting information analysis")
        self.logger.info(
            f"Model: {type(model).__name__}, Dataset size: {len(test_dataset)}, Device: {device}"
        )

        lifecycle_state = self._capture_model_lifecycle_state(model)
        try:
            # Check if model has ExcitationInhibitionNetwork
            if not self._model_supports_information_analysis(model):
                return

            self.logger.info("Model validation passed - starting analysis")
            analysis_start_time = time.time()

            # Prepare data
            model = self._prepare_model_for_analysis(model, device)
            lifecycle_state = self._mark_model_moved(lifecycle_state)

            return self._run_analysis_pipeline(
                model=model,
                test_dataset=test_dataset,
                device=device,
                save_path=save_path,
                filename=filename,
                runtime=runtime,
                analysis_start_time=analysis_start_time,
            )

        except Exception as e:
            self.logger.error(f"Error in information analysis: {e}")
            traceback.print_exc()
        finally:
            self._restore_model_after_analysis(
                model=model,
                lifecycle_state=lifecycle_state,
            )

    def _run_analysis_pipeline(
        self,
        *,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str,
        save_path: Optional[str],
        filename: str,
        runtime: Optional[EvaluationRuntimeConfig],
        analysis_start_time: float,
    ) -> dict[str, Any] | None:
        """Run data collection, metric dispatch, enhanced outputs, and finalization."""
        analysis_data = self._collect_analysis_data(
            model=model,
            test_dataset=test_dataset,
            device=device,
            runtime=runtime,
        )
        inputs = analysis_data.inputs
        labels = analysis_data.labels
        C = analysis_data.C
        entropy_C = analysis_data.entropy_C

        results = self._compute_results_for_level(
            model=model,
            inputs=inputs,
            C=C,
            entropy_C=entropy_C,
        )
        if results is None:
            return None

        self._add_enhanced_analysis_results(results, C)
        return self._complete_analysis_results(
            results=results,
            model=model,
            inputs=inputs,
            labels=labels,
            analysis_start_time=analysis_start_time,
            device=device,
            save_path=save_path,
            filename=filename,
        )
