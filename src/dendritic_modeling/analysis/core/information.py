"""Integrated information analyzer for dendritic networks.

This module provides a unified interface for computing various information-theoretic
metrics using the multivariate methods from information.py. It works directly
with the main configuration system.
"""

import logging
import os
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

from dendritic_modeling.analysis.core import information_config as _information_config
from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.core.information_hooks_mixin import (
    InformationHooksMixin,
)
from dendritic_modeling.analysis.core.information_metrics_mixin import (
    InformationMetricsMixin,
)
from dendritic_modeling.analysis.core.information_signal_mixin import (
    InformationSignalAggregationMixin,
)
from dendritic_modeling.analysis.utils.einet_core import has_einet_core
from dendritic_modeling.analysis.utils.runtime import (
    get_model_device,
    materialize_dataset,
)
from dendritic_modeling.config import InformationAnalysisParams
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.plotting.visualizations.information_plots import (
    plot_fully_aggregated_analysis,
    plot_info_vs_ei_depth_per_dendritic_layer,
    plot_layer_wise_aggregated_analysis,
    plot_network_wise_aggregated_analysis,
    plot_structured_ei_network_analysis,
)
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.information.estimators import (
    EstimatorConfig,
    InformationEstimatorFactory,
)
from dendritic_modeling.utils.information.pid import (
    check_pid_available,
)


@dataclass(frozen=True)
class _InformationAnalysisData:
    """Materialized inputs and labels used by information analysis."""

    inputs: torch.Tensor
    labels: torch.Tensor
    C: np.ndarray
    entropy_C: Optional[float]


@dataclass(frozen=True)
class _SingleBranchLayerData:
    """Prepared branch-wise arrays and availability flags for one layer."""

    excitation: np.ndarray
    inhibition: np.ndarray
    output: np.ndarray
    branch_input: np.ndarray | None
    has_exc_synapses: bool
    has_inh_synapses: bool
    has_branch_input: bool


@dataclass(frozen=True)
class _SingleBranchLdaSignals:
    """Prepared LDA aggregate signals for one layer."""

    excitation: np.ndarray | None
    inhibition: np.ndarray | None
    branch_input: np.ndarray | None
    output: np.ndarray | None
    shuffled: tuple[tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray], ...]


@dataclass(frozen=True)
class _AllBranchMetricData:
    """Prepared full-network arrays for all-branch information metrics."""

    excitation: np.ndarray
    inhibition: np.ndarray
    output: np.ndarray
    labels: np.ndarray
    branch_input: np.ndarray | None


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


class InformationAnalyzer(
    InformationMetricsMixin,
    InformationHooksMixin,
    InformationSignalAggregationMixin,
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
        self.estimators = {}
        # For now, just use the single method specified, but keep the dict structure for compatibility
        method_params = _build_information_estimator_method_params(
            params=params,
            estimator_cfg=estimator_cfg,
            method=self.method,
        )
        estimator_config = EstimatorConfig(
            method=self.method,
            n_neighbors=self.n_neighbors,
            n_bins=self.n_bins,
            copula_type=self.copula_type,
            verbose=self.verbose,
            method_params=method_params,
        )
        self.estimators[self.method] = InformationEstimatorFactory.create(
            estimator_config
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

    def _collect_analysis_data(
        self,
        *,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str,
        runtime: Optional[EvaluationRuntimeConfig],
    ) -> _InformationAnalysisData:
        """Materialize data and collect branch activations through forward hooks."""
        self.logger.info("Loading test dataset...")
        items = materialize_dataset(
            test_dataset,
            runtime,
            explicit_max_samples=self.max_samples,
            device=device,
        )
        inputs: torch.Tensor = items[0].to(device)
        labels: torch.Tensor = items[1].to(device)
        self.logger.info(
            f"Loaded dataset: {len(inputs)} samples, input shape: {inputs.shape}"
        )

        handles = self.attach_forward_hooks(model)
        try:
            with torch.no_grad():
                _ = model(inputs)
        finally:
            self.remove_forward_hooks(handles)

        C = labels.detach().cpu().numpy()
        self.logger.debug(
            f"Data collection completed. Found {len(self.data_dict)} layers."
        )

        entropy_C: Optional[float]
        if self._is_discrete_class_labels(C):
            entropy_C = self._discrete_entropy_nats(C)
        else:
            entropy_C = None

        return _InformationAnalysisData(
            inputs=inputs,
            labels=labels,
            C=C,
            entropy_C=entropy_C,
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

        original_device = get_model_device(model)
        was_training = getattr(model, "training", False)
        moved = False
        try:
            # Check if model has ExcitationInhibitionNetwork
            if not self._model_supports_information_analysis(model):
                return

            self.logger.info("Model validation passed - starting analysis")
            analysis_start_time = time.time()

            # Prepare data
            model = self._prepare_model_for_analysis(model, device)
            moved = True

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
                return

            self._add_enhanced_analysis_results(results, C)

            # Add metadata about analysis configuration
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

        except Exception as e:
            self.logger.error(f"Error in information analysis: {e}")
            traceback.print_exc()
        finally:
            if moved:
                model.to(original_device)
                if was_training:
                    model.train()

    def _model_supports_information_analysis(self, model: BaseModel) -> bool:
        """Return whether a model exposes the EI core needed for information analysis."""
        if has_einet_core(model):
            return True

        self.logger.warning(
            "Model does not expose an ExcitationInhibitionNetwork. Skipping information analysis."
        )
        return False

    def _prepare_model_for_analysis(self, model: BaseModel, device: str) -> BaseModel:
        """Move a model to the analysis device and switch it to evaluation mode."""
        self.logger.info(f"Preparing data: moving model to {device}")
        model = model.to(device)
        model.eval()
        return model

    def _compute_results_for_level(
        self,
        *,
        model: BaseModel,
        inputs: torch.Tensor,
        C: np.ndarray,
        entropy_C: Optional[float],
    ) -> dict[str, Any] | None:
        """Dispatch to the configured information-analysis granularity."""
        if self.computation_level == "single_branch":
            return self._compute_single_branch_results(C, entropy_C)

        if self.computation_level == "layer_branch":
            return self._compute_layer_branch_results(C)

        return self._compute_all_branch_results(
            model=model,
            inputs=inputs,
            C=C,
        )

    def _merge_renamed_metric_dicts_in_place(
        self,
        *,
        target: dict[str, Any],
        source: dict[str, Any],
        token_map: dict[str, str],
    ) -> None:
        """Merge nested metric dictionaries after renaming variable tokens."""
        for key, value in source.items():
            if isinstance(value, dict) and key in target:
                renamed_dict = {}
                for subkey, subval in value.items():
                    new_subkey = self._rename_metric_key_tokens(subkey, token_map)
                    renamed_dict[new_subkey] = subval
                target[key].update(renamed_dict)

    def _merge_average_renamed_numeric_metrics_in_place(
        self,
        *,
        target: dict[str, Any],
        sources: list[dict[str, Any]],
        token_map: dict[str, str],
    ) -> None:
        """Average numeric nested metric values, rename them, and merge into target."""
        metrics_acc: dict[str, dict[str, list[float]]] = {}
        for source in sources:
            for key, value in source.items():
                if isinstance(value, dict):
                    if key not in metrics_acc:
                        metrics_acc[key] = {}
                    for subkey, subval in value.items():
                        if isinstance(subval, (int, float)):
                            if subkey not in metrics_acc[key]:
                                metrics_acc[key][subkey] = []
                            metrics_acc[key][subkey].append(subval)

        for key, subdict in metrics_acc.items():
            if key in target and isinstance(target[key], dict):
                for subkey, values_list in subdict.items():
                    if values_list:
                        avg_value = np.mean(values_list)
                        new_subkey = self._rename_metric_key_tokens(subkey, token_map)
                        target[key][new_subkey] = avg_value

    def _apply_layer_synapse_availability_in_place(
        self,
        layer_stats: dict[str, Any],
        layer_name: str,
    ) -> None:
        """Annotate layer statistics and remove synapse metrics when none exist."""
        if layer_name not in self.data_dict:
            return

        layer_stats["has_synapses"] = self.data_dict[layer_name].get(
            "has_synapses", True
        )
        if layer_stats["has_synapses"]:
            return

        for key in list(layer_stats.keys()):
            if any(
                metric_token in key
                for metric_token in [
                    "I(E;",
                    "I(I;",
                    "I(E_lin;",
                    "I(I_lin;",
                ]
            ):
                layer_stats.pop(key, None)

    def _aggregate_layer_branch_arrays_for_metrics(
        self,
        *,
        E_layer_data: torch.Tensor,
        I_layer_data: torch.Tensor,
        Vout_layer_data: torch.Tensor,
        Vb_layer_data: torch.Tensor | None,
        C: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Prepare one layer's branch arrays for layer-level information metrics."""
        C_for_layer = C

        if self.branch_aggregation == "multivariate":
            E_layer = E_layer_data.detach().cpu().numpy()
            I_layer = I_layer_data.detach().cpu().numpy()
            Vb_layer = (
                Vb_layer_data.detach().cpu().numpy()
                if Vb_layer_data is not None
                else None
            )
            Vout_layer = Vout_layer_data.detach().cpu().numpy()
        elif self.branch_aggregation == "sample":
            E_layer = E_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
            I_layer = I_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
            Vb_layer = (
                Vb_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
                if Vb_layer_data is not None
                else None
            )
            Vout_layer = Vout_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
            _n_samples, n_branches = E_layer_data.shape
            C_for_layer = np.repeat(C, n_branches, axis=0)
        else:
            E_layer = E_layer_data.mean(dim=1).detach().cpu().numpy()
            I_layer = I_layer_data.mean(dim=1).detach().cpu().numpy()
            if Vb_layer_data is not None:
                Vb_layer = Vb_layer_data.mean(dim=1).detach().cpu().numpy()
                Vb_layer = Vb_layer.reshape(-1, 1)
            else:
                Vb_layer = None
            Vout_layer = Vout_layer_data.mean(dim=1).detach().cpu().numpy()
            E_layer = E_layer.reshape(-1, 1)
            I_layer = I_layer.reshape(-1, 1)
            Vout_layer = Vout_layer.reshape(-1, 1)

        return E_layer, I_layer, Vout_layer, C_for_layer, Vb_layer

    def _aggregate_all_branch_layer_tensors(
        self,
        layer_data: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, bool]:
        """Prepare one layer's branch tensors for full-network aggregation."""
        missing_vb = False

        if self.branch_aggregation == "mean":
            layer_E = layer_data["excitation"].mean(dim=1, keepdim=True)
            layer_I = layer_data["inhibition"].mean(dim=1, keepdim=True)
            layer_Vb_raw = layer_data.get("branch_input")
            if layer_Vb_raw is not None:
                layer_Vb = layer_Vb_raw.mean(dim=1, keepdim=True)
            else:
                layer_Vb = None
                missing_vb = True
            layer_Vout = layer_data["output"].mean(dim=1, keepdim=True)
        elif self.branch_aggregation == "multivariate":
            layer_E = layer_data["excitation"]
            layer_I = layer_data["inhibition"]
            layer_Vb_raw = layer_data.get("branch_input")
            if layer_Vb_raw is not None:
                layer_Vb = layer_Vb_raw
            else:
                layer_Vb = None
                missing_vb = True
            layer_Vout = layer_data["output"]
        elif self.branch_aggregation == "sample":
            n_samples, n_branches = layer_data["excitation"].shape
            layer_E = layer_data["excitation"].view(n_samples * n_branches, 1)
            layer_I = layer_data["inhibition"].view(n_samples * n_branches, 1)
            layer_Vb_raw = layer_data.get("branch_input")
            if layer_Vb_raw is not None:
                layer_Vb = layer_Vb_raw.view(n_samples * n_branches, 1)
            else:
                layer_Vb = None
                missing_vb = True
            layer_Vout = layer_data["output"].view(n_samples * n_branches, 1)
        else:
            layer_E = layer_data["excitation"].mean(dim=1, keepdim=True)
            layer_I = layer_data["inhibition"].mean(dim=1, keepdim=True)
            layer_Vb_raw = layer_data.get("branch_input")
            if layer_Vb_raw is not None:
                layer_Vb = layer_Vb_raw.mean(dim=1, keepdim=True)
            else:
                layer_Vb = None
                missing_vb = True
            layer_Vout = layer_data["output"].mean(dim=1, keepdim=True)

        return layer_E, layer_I, layer_Vout, layer_Vb, missing_vb

    def _concatenate_all_branch_tensors_for_metrics(
        self,
        *,
        layer_excitation: list[torch.Tensor],
        layer_inhibition: list[torch.Tensor],
        layer_output: list[torch.Tensor],
        layer_branch_input: list[torch.Tensor],
        missing_vb_in_any_layer: bool,
        C: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """Concatenate per-layer tensors for full-network information metrics."""
        C_for_metrics = C
        if self.branch_aggregation == "sample":
            E = torch.cat(layer_excitation, dim=0).detach().cpu().numpy()
            I_var = torch.cat(layer_inhibition, dim=0).detach().cpu().numpy()
            Vb = (
                torch.cat(layer_branch_input, dim=0).detach().cpu().numpy()
                if (layer_branch_input and not missing_vb_in_any_layer)
                else None
            )
            Vout = torch.cat(layer_output, dim=0).detach().cpu().numpy()
            total_samples = E.shape[0]
            original_samples = C.shape[0]
            expansion_factor = total_samples // original_samples
            C_for_metrics = np.repeat(C, expansion_factor, axis=0)
        else:
            E = torch.cat(layer_excitation, dim=1).detach().cpu().numpy()
            I_var = torch.cat(layer_inhibition, dim=1).detach().cpu().numpy()
            Vb = (
                torch.cat(layer_branch_input, dim=1).detach().cpu().numpy()
                if layer_branch_input
                else None
            )
            Vout = torch.cat(layer_output, dim=1).detach().cpu().numpy()

        return E, I_var, Vout, C_for_metrics, Vb

    def _compute_final_output_signal(
        self,
        *,
        model: BaseModel,
        inputs: torch.Tensor,
    ) -> np.ndarray | None:
        """Run the model and return final output only when the model exposes final_fc."""
        with torch.no_grad():
            final_output: torch.Tensor = model(inputs)
            if hasattr(model, "final_fc"):
                return final_output.detach().cpu().numpy()
        return None

    def _try_add_label_shuffle_null_baselines_in_place(
        self,
        results: dict[str, Any],
        *,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vb: np.ndarray | None,
        Vinf: np.ndarray | None,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
        seed_offset: int,
        failure_context: str,
    ) -> None:
        """Add label-shuffle null baselines when enabled, logging failures."""
        if self.mi_null_shuffles <= 0:
            return

        try:
            self._add_label_shuffle_null_baselines_in_place(
                results,
                E=E,
                I_var=I_var,
                Vout=Vout,
                C=C,
                Vb=Vb,
                Vinf=Vinf,
                has_exc_synapses=has_exc_synapses,
                has_inh_synapses=has_inh_synapses,
                has_branch_input=has_branch_input,
                seed_offset=seed_offset,
            )
        except Exception as e:
            self.logger.warning(
                f"Null (label-shuffle) baseline failed for {failure_context}: {e}"
            )

    def _compute_branch_metrics_with_null_baselines(
        self,
        *,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vb: np.ndarray | None,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
        seed_offset: int,
        failure_context: str,
    ) -> dict[str, Any]:
        """Compute branch metrics and attach optional label-shuffle baselines."""
        branch_results = self.compute_metrics(E, I_var, Vout, C, None, Vb)
        self._try_add_label_shuffle_null_baselines_in_place(
            branch_results,
            E=E,
            I_var=I_var,
            Vout=Vout,
            C=C,
            Vb=Vb,
            Vinf=None,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input and (Vb is not None),
            seed_offset=seed_offset,
            failure_context=failure_context,
        )
        return branch_results

    def _prepare_single_branch_layer_data(
        self,
        layer_data: dict[str, Any],
    ) -> _SingleBranchLayerData:
        """Convert one hook-captured layer into branch-wise arrays and flags."""
        E_all_branches_raw = layer_data["excitation"]
        I_all_branches_raw = layer_data["inhibition"]
        Vb_all_branches_raw = layer_data.get("branch_input")

        has_exc_synapses = bool(layer_data.get("has_exc_synapses", True))
        has_inh_synapses = bool(layer_data.get("has_inh_synapses", True))
        has_branch_input = bool(
            layer_data.get("has_branch_input", Vb_all_branches_raw is not None)
        )

        E_all_branches_filtered, I_all_branches_filtered = self.ensure_inhibitory_data(
            E_all_branches_raw, I_all_branches_raw
        )

        branch_input = (
            Vb_all_branches_raw.detach().cpu().numpy()
            if Vb_all_branches_raw is not None
            else None
        )

        return _SingleBranchLayerData(
            excitation=E_all_branches_filtered.detach().cpu().numpy(),
            inhibition=I_all_branches_filtered.detach().cpu().numpy(),
            output=layer_data["output"].detach().cpu().numpy(),
            branch_input=branch_input,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input,
        )

    def _prepare_single_branch_lda_signals(
        self,
        layer_data: dict[str, Any],
        C: np.ndarray,
    ) -> _SingleBranchLdaSignals:
        """Compute deterministic and shuffled LDA aggregate signals for one layer."""
        if not self.compute_lda_weights:
            return _SingleBranchLdaSignals(
                excitation=None,
                inhibition=None,
                branch_input=None,
                output=None,
                shuffled=(),
            )

        E_lin, I_lin, Vb_lin, Vout_lin = self.compute_lda_aggregated_signals(
            layer_data, C
        )

        shuffled_signals = []
        if self.lda_n_shuffles > 0:
            for shuffle_idx in range(self.lda_n_shuffles):
                E_sh, I_sh, Vb_sh, Vout_sh = self.compute_shuffled_aggregated_signals(
                    layer_data, C, shuffle_idx
                )
                shuffled_signals.append((E_sh, I_sh, Vb_sh, Vout_sh))

        return _SingleBranchLdaSignals(
            excitation=E_lin,
            inhibition=I_lin,
            branch_input=Vb_lin,
            output=Vout_lin,
            shuffled=tuple(shuffled_signals),
        )

    def _compute_single_branch_result(
        self,
        *,
        prepared_layer: _SingleBranchLayerData,
        lda_signals: _SingleBranchLdaSignals,
        C: np.ndarray,
        branch_idx: int,
        layer_count: int,
    ) -> dict[str, Any]:
        """Compute and filter all metrics for one branch in a prepared layer."""
        E_branch = prepared_layer.excitation[:, branch_idx : branch_idx + 1]
        I_branch = prepared_layer.inhibition[:, branch_idx : branch_idx + 1]
        Vout_branch = prepared_layer.output[:, branch_idx : branch_idx + 1]
        Vb_branch = (
            prepared_layer.branch_input[:, branch_idx : branch_idx + 1]
            if prepared_layer.branch_input is not None
            else None
        )

        branch_results = self._compute_branch_metrics_with_null_baselines(
            E=E_branch,
            I_var=I_branch,
            Vout=Vout_branch,
            C=C,
            Vb=Vb_branch,
            has_exc_synapses=prepared_layer.has_exc_synapses,
            has_inh_synapses=prepared_layer.has_inh_synapses,
            has_branch_input=prepared_layer.has_branch_input,
            seed_offset=int(layer_count * 10_000 + branch_idx),
            failure_context=f"branch {branch_idx}",
        )

        self._merge_lda_branch_metrics_in_place(
            target=branch_results,
            E_lin=lda_signals.excitation,
            I_lin=lda_signals.inhibition,
            Vb_lin=lda_signals.branch_input,
            Vout_lin=lda_signals.output,
            C=C,
            branch_idx=branch_idx,
            layer_count=layer_count,
            has_exc_synapses=prepared_layer.has_exc_synapses,
            has_inh_synapses=prepared_layer.has_inh_synapses,
            has_branch_input=prepared_layer.has_branch_input,
        )

        self._merge_shuffled_branch_metrics_in_place(
            target=branch_results,
            shuffled_signals=lda_signals.shuffled,
            C=C,
            branch_idx=branch_idx,
            has_branch_input=prepared_layer.has_branch_input,
        )

        self._filter_result_metrics_in_place(
            branch_results,
            has_exc_synapses=prepared_layer.has_exc_synapses,
            has_inh_synapses=prepared_layer.has_inh_synapses,
            has_branch_input=prepared_layer.has_branch_input,
        )
        return branch_results

    def _finalize_single_branch_results(
        self,
        *,
        all_results: list[dict[str, Any]],
        branch_names: list[str],
        layer_results: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Average branch metrics and attach single-branch analysis metadata."""
        if not all_results:
            self.logger.warning("No branch results to average")
            return None

        results = self._average_mi_results(all_results)
        results["branch_names"] = branch_names
        results["num_branches"] = len(all_results)
        results["layer_statistics"] = layer_results
        results["computation_level"] = "single_branch"
        return results

    def _compute_single_branch_results(
        self,
        C: np.ndarray,
        entropy_C: Optional[float],
    ) -> dict[str, Any] | None:
        """Compute information metrics for individual branches and average them."""
        all_results = []
        branch_names = []
        layer_results = {}

        total_layers = len(
            [name for name, data in self.data_dict.items() if "excitation" in data]
        )
        total_branches = sum(
            data.get("excitation", torch.empty(0, 0)).shape[1]
            for data in self.data_dict.values()
            if "excitation" in data
        )
        layer_count = 0

        self.logger.info(
            f"Single-branch MI analysis: {total_layers} layers, {total_branches} branches, method={self.method}"
        )

        layers_to_process = [
            (name, data)
            for name, data in self.data_dict.items()
            if "excitation" in data
        ]

        for layer_name, layer_data in tqdm(
            layers_to_process,
            desc="Processing layers",
            leave=False,
            ncols=100,
        ):
            layer_count += 1

            prepared_layer = self._prepare_single_branch_layer_data(layer_data)
            lda_signals = self._prepare_single_branch_lda_signals(layer_data, C)

            n_branches = prepared_layer.excitation.shape[1]
            layer_branch_results = []
            layer_start_time = time.time()

            short_name = layer_name.split(".")[-1] if "." in layer_name else layer_name
            for branch_idx in tqdm(
                range(n_branches),
                desc=f"  {short_name}",
                leave=False,
                ncols=100,
                mininterval=0.5,
            ):
                branch_start_time = time.time()

                branch_results = self._compute_single_branch_result(
                    prepared_layer=prepared_layer,
                    lda_signals=lda_signals,
                    C=C,
                    branch_idx=branch_idx,
                    layer_count=layer_count,
                )
                all_results.append(branch_results)
                layer_branch_results.append(branch_results)
                branch_names.append(f"{layer_name}_branch_{branch_idx}")

                self.logger.debug(
                    f"  Branch {branch_idx + 1}/{n_branches} completed (took {time.time() - branch_start_time:.1f}s)"
                )

            if layer_branch_results:
                layer_stats = self._compute_layer_statistics(
                    layer_branch_results,
                    entropy_C=(entropy_C if self.compute_layer_total_proxies else None),
                    topk=self.layer_total_topk,
                )
                self._apply_layer_synapse_availability_in_place(layer_stats, layer_name)
                layer_results[layer_name] = layer_stats

            self.logger.info(
                f"Layer {layer_name} completed: {n_branches} branches processed in {time.time() - layer_start_time:.1f}s"
            )

        return self._finalize_single_branch_results(
            all_results=all_results,
            branch_names=branch_names,
            layer_results=layer_results,
        )

    def _merge_lda_branch_metrics_in_place(
        self,
        *,
        target: dict[str, Any],
        E_lin: np.ndarray | None,
        I_lin: np.ndarray | None,
        Vb_lin: np.ndarray | None,
        Vout_lin: np.ndarray | None,
        C: np.ndarray,
        branch_idx: int,
        layer_count: int,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
    ) -> None:
        """Compute one branch's LDA-aggregated metrics and merge renamed keys."""
        if not self.compute_lda_weights or E_lin is None:
            return

        E_branch_lin = E_lin[:, branch_idx : branch_idx + 1]
        I_branch_lin = I_lin[:, branch_idx : branch_idx + 1]
        Vb_branch_lin = (
            Vb_lin[:, branch_idx : branch_idx + 1]
            if has_branch_input and Vb_lin is not None
            else None
        )
        Vout_branch_lin = Vout_lin[:, branch_idx : branch_idx + 1]

        branch_results_lin = self._compute_branch_metrics_with_null_baselines(
            E=E_branch_lin,
            I_var=I_branch_lin,
            Vout=Vout_branch_lin,
            C=C,
            Vb=Vb_branch_lin,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input,
            seed_offset=int(1_000_000 + layer_count * 10_000 + branch_idx),
            failure_context=f"LDA branch {branch_idx}",
        )
        self._merge_renamed_metric_dicts_in_place(
            target=target,
            source=branch_results_lin,
            token_map={
                "E": "E_lin",
                "I": "I_lin",
                "Vb": "Vb_lin",
                "Vout": "Vout_lin",
                "Vinf": "Vinf_lin",
            },
        )

    def _merge_shuffled_branch_metrics_in_place(
        self,
        *,
        target: dict[str, Any],
        shuffled_signals: Sequence[
            tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]
        ],
        C: np.ndarray,
        branch_idx: int,
        has_branch_input: bool,
    ) -> None:
        """Compute one branch's shuffled LDA baselines and merge averaged keys."""
        if not shuffled_signals:
            return

        shuffled_results = []
        for E_sh, I_sh, Vb_sh, Vout_sh in shuffled_signals:
            E_branch_sh = E_sh[:, branch_idx : branch_idx + 1]
            I_branch_sh = I_sh[:, branch_idx : branch_idx + 1]
            Vb_branch_sh = (
                Vb_sh[:, branch_idx : branch_idx + 1]
                if has_branch_input and Vb_sh is not None
                else None
            )
            Vout_branch_sh = Vout_sh[:, branch_idx : branch_idx + 1]

            branch_results_sh = self.compute_metrics(
                E_branch_sh,
                I_branch_sh,
                Vout_branch_sh,
                C,
                None,
                Vb_branch_sh,
            )
            shuffled_results.append(branch_results_sh)

        self._merge_average_renamed_numeric_metrics_in_place(
            target=target,
            sources=shuffled_results,
            token_map={
                "E": "E_sh",
                "I": "I_sh",
                "Vb": "Vb_sh",
                "Vout": "Vout_sh",
                "Vinf": "Vinf_sh",
            },
        )

    def _compute_layer_branch_result(
        self,
        *,
        layer_name: str,
        layer_data: dict[str, Any],
        C: np.ndarray,
        seed_offset: int,
    ) -> dict[str, Any]:
        """Compute layer-level metrics after aggregating one layer's branches."""
        E_layer_data_raw = layer_data["excitation"]
        I_layer_data_raw = layer_data["inhibition"]
        Vb_layer_data_raw = layer_data.get("branch_input")
        Vout_layer_data = layer_data["output"]

        has_exc_synapses = bool(layer_data.get("has_exc_synapses", True))
        has_inh_synapses = bool(layer_data.get("has_inh_synapses", True))
        has_branch_input = Vb_layer_data_raw is not None

        E_layer_data, I_layer_data = self.ensure_inhibitory_data(
            E_layer_data_raw, I_layer_data_raw
        )

        # Keep Vb as None when it does not exist, e.g. in the outermost distal layer.
        Vb_layer_data = Vb_layer_data_raw if has_branch_input else None
        E_layer, I_layer, Vout_layer, C_for_layer, Vb_layer = (
            self._aggregate_layer_branch_arrays_for_metrics(
                E_layer_data=E_layer_data,
                I_layer_data=I_layer_data,
                Vout_layer_data=Vout_layer_data,
                Vb_layer_data=Vb_layer_data,
                C=C,
            )
        )

        layer_results = self.compute_metrics(
            E_layer, I_layer, Vout_layer, C_for_layer, None, Vb_layer
        )
        self._filter_result_metrics_in_place(
            layer_results,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input,
        )
        self._try_add_label_shuffle_null_baselines_in_place(
            layer_results,
            E=E_layer,
            I_var=I_layer,
            Vout=Vout_layer,
            C=C_for_layer,
            Vb=Vb_layer,
            Vinf=None,
            has_exc_synapses=has_exc_synapses,
            has_inh_synapses=has_inh_synapses,
            has_branch_input=has_branch_input and (Vb_layer is not None),
            seed_offset=seed_offset,
            failure_context=f"layer {layer_name}",
        )
        return layer_results

    def _compute_layer_branch_results(self, C: np.ndarray) -> dict[str, Any] | None:
        """Compute information metrics after aggregating branches within each layer."""
        self.logger.info("Computing MI at layer level...")

        all_results = []
        layer_names = []

        layers_to_process = [
            (name, data)
            for name, data in self.data_dict.items()
            if "excitation" in data
        ]

        for layer_name, layer_data in tqdm(
            layers_to_process,
            desc="Processing layers",
            leave=False,
            ncols=100,
        ):
            layer_results = self._compute_layer_branch_result(
                layer_name=layer_name,
                layer_data=layer_data,
                C=C,
                seed_offset=len(all_results),
            )
            all_results.append(layer_results)
            layer_names.append(layer_name)

        return self._finalize_layer_branch_results(
            all_results=all_results,
            layer_names=layer_names,
        )

    def _finalize_layer_branch_results(
        self,
        *,
        all_results: list[dict[str, Any]],
        layer_names: list[str],
    ) -> dict[str, Any] | None:
        """Average layer metrics and attach layer-branch analysis metadata."""
        if not all_results:
            self.logger.warning("No layer results to average")
            return None

        results = self._average_mi_results(all_results)
        results["layer_names"] = layer_names
        results["num_layers"] = len(all_results)
        results["computation_level"] = "layer_branch"

        layer_statistics = {}
        for layer_name, layer_result in zip(layer_names, all_results):
            layer_stats = self._compute_layer_statistics([layer_result])
            self._apply_layer_synapse_availability_in_place(layer_stats, layer_name)
            layer_statistics[layer_name] = layer_stats

        results["layer_statistics"] = layer_statistics
        self.logger.info(
            f"Generated layer statistics for {len(layer_statistics)} layers for plotting"
        )
        return results

    def _collect_all_branch_metric_data(
        self,
        C: np.ndarray,
    ) -> _AllBranchMetricData | None:
        """Collect and concatenate full-network branch tensors for metric computation."""
        layer_excitation = []
        layer_inhibition = []
        layer_branch_input = []
        missing_vb_in_any_layer = False
        layer_output = []

        for layer_data in self.data_dict.values():
            if "excitation" not in layer_data:
                continue

            layer_E, layer_I, layer_Vout, layer_Vb, missing_vb = (
                self._aggregate_all_branch_layer_tensors(layer_data)
            )
            missing_vb_in_any_layer = missing_vb_in_any_layer or missing_vb

            layer_excitation.append(layer_E)
            layer_inhibition.append(layer_I)
            if layer_Vb is not None:
                layer_branch_input.append(layer_Vb)
            layer_output.append(layer_Vout)

        if not layer_excitation:
            self.logger.warning(
                "No excitatory activations collected. Skipping information analysis."
            )
            return None

        E, I_var, Vout, C_for_metrics, Vb = (
            self._concatenate_all_branch_tensors_for_metrics(
                layer_excitation=layer_excitation,
                layer_inhibition=layer_inhibition,
                layer_output=layer_output,
                layer_branch_input=layer_branch_input,
                missing_vb_in_any_layer=missing_vb_in_any_layer,
                C=C,
            )
        )

        self.logger.info(
            "Aggregated data shapes - "
            f"E: {E.shape}, I: {I_var.shape}, Vb: {None if Vb is None else Vb.shape}, Vout: {Vout.shape}, C: {C_for_metrics.shape}"
        )

        return _AllBranchMetricData(
            excitation=E,
            inhibition=I_var,
            output=Vout,
            labels=C_for_metrics,
            branch_input=Vb,
        )

    def _finalize_all_branch_results(
        self,
        *,
        metric_data: _AllBranchMetricData,
        Vinf: np.ndarray | None,
    ) -> dict[str, Any]:
        """Compute all-branch metrics and attach aggregation metadata."""
        results = self.compute_metrics(
            metric_data.excitation,
            metric_data.inhibition,
            metric_data.output,
            metric_data.labels,
            Vinf,
            metric_data.branch_input,
        )
        self._try_add_label_shuffle_null_baselines_in_place(
            results,
            E=metric_data.excitation,
            I_var=metric_data.inhibition,
            Vout=metric_data.output,
            C=metric_data.labels,
            Vb=metric_data.branch_input,
            Vinf=Vinf,
            has_exc_synapses=True,
            has_inh_synapses=True,
            has_branch_input=metric_data.branch_input is not None,
            seed_offset=999_999,
            failure_context="all-branch analysis",
        )
        results["aggregation_method"] = self.branch_aggregation
        results["computation_level"] = "all_branch"
        return results

    def _compute_all_branch_results(
        self,
        *,
        model: BaseModel,
        inputs: torch.Tensor,
        C: np.ndarray,
    ) -> dict[str, Any] | None:
        """Compute full-network information metrics across all collected branches."""
        self.logger.info(
            f"Aggregating branches using method: {self.branch_aggregation} (full network aggregation)"
        )

        metric_data = self._collect_all_branch_metric_data(C)
        if metric_data is None:
            return None

        Vinf = self._compute_final_output_signal(model=model, inputs=inputs)

        return self._finalize_all_branch_results(
            metric_data=metric_data,
            Vinf=Vinf,
        )

    def _collect_enhanced_analysis_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Collect concatenated E/I/Vout arrays for enhanced analysis modes."""
        all_E_data = []
        all_I_data = []
        all_Vout_data = []

        for layer_data in self.data_dict.values():
            if "excitation" in layer_data:
                E_filtered, I_filtered = self.ensure_inhibitory_data(
                    layer_data["excitation"], layer_data["inhibition"]
                )
                all_E_data.append(E_filtered.detach().cpu().numpy())
                all_I_data.append(I_filtered.detach().cpu().numpy())
                all_Vout_data.append(layer_data["output"].detach().cpu().numpy())

        if not all_E_data:
            return None

        E_all = np.concatenate(all_E_data, axis=1)
        I_all = np.concatenate(all_I_data, axis=1)
        Vout_all = np.concatenate(all_Vout_data, axis=1)
        return E_all, I_all, Vout_all

    def _add_enhanced_analysis_results(
        self, results: dict[str, Any], C: np.ndarray
    ) -> None:
        """Add optional per-layer, per-neuron, and EI-specific analysis results."""
        if not (
            self.per_neuron_analysis
            or self.per_einet_analysis
            or self.per_layer_analysis
        ):
            return

        self.logger.info("Running enhanced analysis modes...")

        if self.per_layer_analysis:
            per_layer_results = self.compute_per_layer_enhanced_analysis(C)
            results["per_layer_analysis"] = per_layer_results

        if not (self.per_neuron_analysis or self.per_einet_analysis):
            return

        enhanced_arrays = self._collect_enhanced_analysis_arrays()
        if enhanced_arrays is None:
            return

        E_all, I_all, Vout_all = enhanced_arrays

        if self.per_neuron_analysis:
            per_neuron_results = self.compute_per_neuron_mi(E_all, I_all, Vout_all, C)
            results["per_neuron_analysis"] = per_neuron_results

        if self.per_einet_analysis:
            ei_specific_results = self.compute_ei_specific_mi(E_all, I_all, Vout_all, C)
            results["per_einet_analysis"] = ei_specific_results

    def _complete_analysis_results(
        self,
        *,
        results: dict[str, Any],
        model: BaseModel,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        analysis_start_time: float,
        device: str,
        save_path: Optional[str],
        filename: str,
    ) -> dict[str, Any]:
        """Add metadata, finalize units/plots, save outputs, and return results."""
        self._add_analysis_metadata(results)

        total_analysis_time = time.time() - analysis_start_time
        self._log_analysis_summary(results, total_analysis_time)

        somatic_synapses = self._add_synthetic_soma_entry_if_needed(results, model)
        self._finalize_information_units_and_variance(results)
        self._maybe_add_lda_vs_trained_performance(
            results=results,
            model=model,
            inputs=inputs,
            labels=labels,
            device=device,
            save_path=save_path,
        )
        self._save_analysis_outputs(
            results=results,
            save_path=save_path,
            filename=filename,
            somatic_synapses=somatic_synapses,
        )
        return results

    def _add_analysis_metadata(self, results: dict[str, Any]) -> None:
        """Attach analyzer configuration metadata to result dictionaries."""
        results["per_neuron_analysis_enabled"] = self.per_neuron_analysis
        results["per_einet_analysis_enabled"] = self.per_einet_analysis
        results["per_layer_analysis_enabled"] = self.per_layer_analysis
        results["has_inhibitory_data"] = self.has_inhibitory_data(self.data_dict)

    def _log_analysis_summary(
        self, results: dict[str, Any], total_analysis_time: float
    ) -> None:
        """Log a concise summary of completed information analysis."""
        self.logger.info("Information analysis completed successfully!")
        self.logger.info("Analysis summary:")
        self.logger.info(f"   - Total time: {total_analysis_time:.1f}s")
        self.logger.info(f"   - Computation level: {self.computation_level}")
        self.logger.info(f"   - Method: {self.method}")
        self.logger.info(f"   - Sample size: {results.get('n_samples', 'unknown')}")
        if "num_branches" in results:
            self.logger.info(f"   - Branches processed: {results['num_branches']}")
        if "num_layers" in results:
            self.logger.info(f"   - Layers processed: {results['num_layers']}")

    def _finalize_information_units_and_variance(self, results: dict[str, Any]) -> None:
        results["information_units"] = getattr(self, "output_units", "nats")
        if results["information_units"] == "bits":
            self._convert_information_units_in_place(results)

        self.logger.info("Computing variance analysis for plotting...")
        variance_analysis = self._compute_variance_analysis(results)
        results["variance_analysis"] = variance_analysis
        self.logger.info(
            f"Variance analysis computed with {len(variance_analysis.get('layer_variances', {}))} layers"
        )

    def _maybe_add_lda_vs_trained_performance(
        self,
        *,
        results: dict[str, Any],
        model: BaseModel,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        device: str,
        save_path: Optional[str],
    ) -> None:
        if not (self.compute_lda_weights and save_path is not None):
            return

        try:
            lda_comparison = self.compare_lda_vs_trained_performance(
                model, inputs, labels, device
            )
            results["lda_vs_trained_performance"] = lda_comparison

            if lda_comparison.get("lda_accuracy") is not None:
                from dendritic_modeling.plotting.visualizations.information_plots import (
                    plot_lda_vs_trained_performance,
                )

                plot_lda_vs_trained_performance(lda_comparison, save_path, self.logger)
                self.logger.info("LDA vs trained performance comparison completed!")
            else:
                self.logger.warning(
                    "LDA accuracy not available, skipping performance plot"
                )
        except Exception as e:
            self.logger.error(f"Error computing LDA vs trained performance: {e}")
            traceback.print_exc()

    def _save_analysis_outputs(
        self,
        *,
        results: dict[str, Any],
        save_path: Optional[str],
        filename: str,
        somatic_synapses: bool,
    ) -> None:
        if save_path is None:
            return

        self.logger.info(f"Saving results to {save_path}/{filename}")
        save_dict(results, save_path, filename)

        self._write_analysis_summary_file(
            results=results, save_path=save_path, filename=filename
        )
        self._run_post_save_visualizations(
            results=results,
            save_path=save_path,
            filename=filename,
            somatic_synapses=somatic_synapses,
        )
        self.logger.info("Results saved successfully!")

    def _write_analysis_summary_file(
        self,
        *,
        results: dict[str, Any],
        save_path: str,
        filename: str,
    ) -> None:
        """Write the text summary next to serialized analysis results."""
        summary = self.get_summary(results)
        summary_path = os.path.join(save_path, f"{filename}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary)

    def _run_post_save_visualizations(
        self,
        *,
        results: dict[str, Any],
        save_path: str,
        filename: str,
        somatic_synapses: bool,
    ) -> None:
        """Generate optional post-save summaries and plots."""
        try:
            if "per_layer_analysis" in results:
                self._save_per_layer_ei_summaries(
                    results["per_layer_analysis"], save_path, filename
                )
            self._generate_information_plots(results, save_path, somatic_synapses)
        except Exception as plot_err:
            self.logger.warning(f"Post-save visualization failed: {plot_err}")

    def _add_synthetic_soma_entry_if_needed(
        self, results: dict[str, Any], model: BaseModel
    ) -> bool:
        # This is only needed when somatic_synapses=False and the soma layer
        # was not captured by the forward hooks.
        somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
        self.logger.info(f"Somatic synapses enabled: {somatic_synapses}")

        soma_already_captured = self._soma_layer_already_captured(results)
        if self._should_add_synthetic_soma_entry(
            somatic_synapses=somatic_synapses,
            results=results,
            soma_already_captured=soma_already_captured,
        ):
            self.logger.info(
                f"Adding synthetic soma entry to {len(results['layer_statistics'])} existing layers"
            )
            layer_statistics = results["layer_statistics"]
            real_soma_layer = self._find_real_soma_layer(layer_statistics)
            soma_stats = self._extract_soma_statistics(
                real_soma_layer, layer_statistics
            )
            soma_entry = self._build_synthetic_soma_entry(soma_stats)
            results["layer_statistics"]["synthetic_soma"] = soma_entry
            self.logger.info(
                "Added synthetic soma with real "
                f"I(Vout;C)={soma_stats['soma_vout_mean']:.4f}. "
                f"Total layers now: {len(results['layer_statistics'])}"
            )
        elif not somatic_synapses and "layer_statistics" in results:
            self.logger.info("Soma layer already captured; skipping synthetic soma.")
        elif not somatic_synapses and "layer_statistics" not in results:
            self.logger.warning(
                "somatic_synapses=False but no layer_statistics found in results"
            )
        return somatic_synapses

    @staticmethod
    def _should_add_synthetic_soma_entry(
        *,
        somatic_synapses: bool,
        results: dict[str, Any],
        soma_already_captured: bool,
    ) -> bool:
        """Return whether a synthetic soma layer should be added to results."""
        return (
            not somatic_synapses
            and "layer_statistics" in results
            and not soma_already_captured
        )

    def _soma_layer_already_captured(self, results: dict[str, Any]) -> bool:
        if "layer_statistics" not in results:
            return False

        for layer_name in results["layer_statistics"].keys():
            if self._parse_branch_layer_index(layer_name) is None:
                continue
            layer_data = results["layer_statistics"][layer_name]
            if layer_data.get("has_synapses") is False:
                self.logger.info(f"Soma layer already captured: {layer_name}")
                return True
        return False

    @staticmethod
    def _parse_branch_layer_index(layer_name: str) -> Optional[int]:
        """Extract the branch-layer index from a module path, if present."""
        if "branch_layers." not in layer_name:
            return None
        try:
            return int(layer_name.split("branch_layers.")[1].split(".")[0])
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _find_real_soma_layer(layer_statistics: dict[str, Any]) -> Optional[str]:
        max_branch_idx = -1
        real_soma_layer = None

        for layer_name in layer_statistics.keys():
            branch_idx = InformationAnalyzer._parse_branch_layer_index(layer_name)
            if branch_idx is None:
                continue
            if branch_idx > max_branch_idx:
                max_branch_idx = branch_idx
                real_soma_layer = layer_name

        return real_soma_layer

    def _extract_soma_statistics(
        self, real_soma_layer: Optional[str], layer_statistics: dict[str, Any]
    ) -> dict[str, float]:
        soma_stats = self._empty_soma_statistics()

        if real_soma_layer and real_soma_layer in layer_statistics:
            real_soma_stats = layer_statistics[real_soma_layer]
            soma_stats.update(self._copy_soma_statistics_from_layer(real_soma_stats))
            self._log_soma_statistics(real_soma_layer, soma_stats)
        else:
            self.logger.warning(
                "Could not find real soma layer for I(Vout;C) computation"
            )

        return soma_stats

    @classmethod
    def _empty_soma_statistics(cls) -> dict[str, float]:
        """Return zero-valued soma statistics for every supported metric."""
        return dict.fromkeys(cls._soma_statistics_metric_map(), 0.0)

    @classmethod
    def _copy_soma_statistics_from_layer(
        cls, real_soma_stats: dict[str, Any]
    ) -> dict[str, float]:
        """Copy soma metrics from a captured real soma layer into internal keys."""
        return {
            soma_key: real_soma_stats.get(metric_key, 0.0)
            for soma_key, metric_key in cls._soma_statistics_metric_map().items()
        }

    def _log_soma_statistics(
        self, real_soma_layer: str, soma_stats: dict[str, float]
    ) -> None:
        """Log the real soma values reused for the synthetic soma entry."""
        self.logger.info(f"Found real soma layer: {real_soma_layer}")
        self.logger.info(
            "Using real soma "
            f"I(Vout;C): {soma_stats['soma_vout_mean']:.4f} "
            f"± {soma_stats['soma_vout_std']:.4f}"
        )
        self.logger.info(
            "Using real soma "
            f"I(Vb;C): {soma_stats['soma_vb_mean']:.4f} "
            f"± {soma_stats['soma_vb_std']:.4f}"
        )

    @staticmethod
    def _soma_statistics_metric_map() -> dict[str, str]:
        """Map internal soma-stat keys to source layer-stat metric keys."""
        return {
            "soma_vout_mean": "I(Vout;C)_mean",
            "soma_vout_std": "I(Vout;C)_std",
            "soma_vb_mean": "I(Vb;C)_mean",
            "soma_vb_std": "I(Vb;C)_std",
            "soma_vb_lin_mean": "I(Vb_lin;C)_mean",
            "soma_vb_lin_std": "I(Vb_lin;C)_std",
            "soma_vb_vout_mean": "I(Vb;Vout)_mean",
            "soma_vb_vout_std": "I(Vb;Vout)_std",
            "soma_vb_vout_c_mean": "I(Vb;Vout|C)_mean",
            "soma_vb_vout_c_std": "I(Vb;Vout|C)_std",
            "soma_vb_lin_vout_mean": "I(Vb_lin;Vout_lin)_mean",
            "soma_vb_lin_vout_std": "I(Vb_lin;Vout_lin)_std",
            "soma_vb_lin_vout_c_mean": "I(Vb_lin;Vout_lin|C)_mean",
            "soma_vb_lin_vout_c_std": "I(Vb_lin;Vout_lin|C)_std",
            "soma_vout_lin_mean": "I(Vout_lin;C)_mean",
            "soma_vout_lin_std": "I(Vout_lin;C)_std",
        }

    def _build_synthetic_soma_entry(
        self, soma_stats: dict[str, float]
    ) -> dict[str, Any]:
        soma_entry = {
            metric_key: (0.0 if soma_stat_key is None else soma_stats[soma_stat_key])
            for metric_key, soma_stat_key in self._synthetic_soma_metric_sources()
        }
        return self._filter_synthetic_soma_entry_metrics(soma_entry)

    @staticmethod
    def _synthetic_soma_metric_sources() -> tuple[tuple[str, Optional[str]], ...]:
        """Map synthetic soma output metrics to copied soma-stat keys."""
        return (
            ("I(E;C)_mean", None),
            ("I(E;C)_std", None),
            ("I(I;C)_mean", None),
            ("I(I;C)_std", None),
            ("I(Vb;C)_mean", "soma_vb_mean"),
            ("I(Vb;C)_std", "soma_vb_std"),
            ("I(Vout;C)_mean", "soma_vout_mean"),
            ("I(Vout;C)_std", "soma_vout_std"),
            ("I(E,I;C)_mean", None),
            ("I(E,I;C)_std", None),
            ("I(E_lin;C)_mean", None),
            ("I(E_lin;C)_std", None),
            ("I(I_lin;C)_mean", None),
            ("I(I_lin;C)_std", None),
            ("I(Vb_lin;C)_mean", "soma_vb_lin_mean"),
            ("I(Vb_lin;C)_std", "soma_vb_lin_std"),
            ("I(Vout_lin;C)_mean", "soma_vout_lin_mean"),
            ("I(Vout_lin;C)_std", "soma_vout_lin_std"),
            ("I(E_lin,I_lin;C)_mean", None),
            ("I(E_lin,I_lin;C)_std", None),
            ("I(E;I)_mean", None),
            ("I(E;I)_std", None),
            ("I(E;Vout)_mean", None),
            ("I(E;Vout)_std", None),
            ("I(I;Vout)_mean", None),
            ("I(I;Vout)_std", None),
            ("I(E;Vb)_mean", None),
            ("I(E;Vb)_std", None),
            ("I(I;Vb)_mean", None),
            ("I(I;Vb)_std", None),
            ("I(Vb;Vout)_mean", "soma_vb_vout_mean"),
            ("I(Vb;Vout)_std", "soma_vb_vout_std"),
            ("I(E_lin;I_lin)_mean", None),
            ("I(E_lin;I_lin)_std", None),
            ("I(E_lin;Vout_lin)_mean", None),
            ("I(E_lin;Vout_lin)_std", None),
            ("I(I_lin;Vout_lin)_mean", None),
            ("I(I_lin;Vout_lin)_std", None),
            ("I(E_lin;Vb_lin)_mean", None),
            ("I(E_lin;Vb_lin)_std", None),
            ("I(I_lin;Vb_lin)_mean", None),
            ("I(I_lin;Vb_lin)_std", None),
            ("I(Vb_lin;Vout_lin)_mean", "soma_vb_lin_vout_mean"),
            ("I(Vb_lin;Vout_lin)_std", "soma_vb_lin_vout_std"),
            ("I(E;I|C)_mean", None),
            ("I(E;I|C)_std", None),
            ("I(E;Vout|C)_mean", None),
            ("I(E;Vout|C)_std", None),
            ("I(I;Vout|C)_mean", None),
            ("I(I;Vout|C)_std", None),
            ("I(E;Vb|C)_mean", None),
            ("I(E;Vb|C)_std", None),
            ("I(I;Vb|C)_mean", None),
            ("I(I;Vb|C)_std", None),
            ("I(Vb;Vout|C)_mean", "soma_vb_vout_c_mean"),
            ("I(Vb;Vout|C)_std", "soma_vb_vout_c_std"),
            ("I(E_lin;I_lin|C)_mean", None),
            ("I(E_lin;I_lin|C)_std", None),
            ("I(E_lin;Vout_lin|C)_mean", None),
            ("I(E_lin;Vout_lin|C)_std", None),
            ("I(I_lin;Vout_lin|C)_mean", None),
            ("I(I_lin;Vout_lin|C)_std", None),
            ("I(E_lin;Vb_lin|C)_mean", None),
            ("I(E_lin;Vb_lin|C)_std", None),
            ("I(I_lin;Vb_lin|C)_mean", None),
            ("I(I_lin;Vb_lin|C)_std", None),
            ("I(Vb_lin;Vout_lin|C)_mean", "soma_vb_lin_vout_c_mean"),
            ("I(Vb_lin;Vout_lin|C)_std", "soma_vb_lin_vout_c_std"),
        )

    def _filter_synthetic_soma_entry_metrics(
        self, soma_entry: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply soma availability rules to a synthetic soma result entry."""
        filtered_entry = {
            key: value
            for key, value in soma_entry.items()
            if self._keep_metric_key_given_availability(
                key,
                has_exc_synapses=False,
                has_inh_synapses=False,
                has_branch_input=True,
            )
        }
        filtered_entry["has_synapses"] = False
        return filtered_entry

    def compute_class_mi(self, activations, labels):
        return self.estimators[self.method].mutual_information(activations, labels)

    def compute_conditional_mi(self, activations, labels, condition):
        return self.estimators[self.method].conditional_mutual_information(
            activations, labels, condition
        )

    def compare_lda_vs_trained_performance(
        self,
        model: torch.nn.Module,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device = None,
    ) -> dict:
        """Compare classification performance using trained weights vs LDA-optimal weights.

        This method uses the data already collected in self.data_dict from the main analysis.
        It computes:
        1. Performance with trained network weights (original model)
        2. Performance with LDA-optimized input aggregation (Vout_lin passed through decoder)

        Args:
            model: Trained model with encoder, core_network, and decoder
            inputs: Input tensor (already processed in main analysis)
            labels: Label tensor (already processed in main analysis)
            device: Device to run computations on

        Returns:
            Dictionary with:
            - trained_accuracy: Accuracy with trained weights
            - lda_accuracy: Accuracy with LDA-optimal aggregation
            - accuracy_difference: LDA - Trained accuracy
        """
        from sklearn.metrics import accuracy_score

        if device is None:
            device = get_model_device(model)

        # Ensure model is in eval mode (same state used for all comparisons)
        model.eval()

        self.logger.info("Computing LDA vs trained weights performance comparison...")
        self.logger.info(
            "Note: Using the SAME model for both trained and LDA predictions. "
            "Only the intermediate representations differ."
        )

        # Move labels to CPU/numpy
        if hasattr(labels, "cpu"):
            labels_np = labels.cpu().numpy()
        elif hasattr(labels, "numpy"):
            labels_np = labels.numpy()
        else:
            labels_np = np.array(labels)

        # Get trained model predictions
        with torch.no_grad():
            trained_outputs = model(inputs.to(device))
            trained_preds = trained_outputs.argmax(dim=1).cpu().numpy()

        trained_accuracy = accuracy_score(labels_np, trained_preds)

        # Compute CHAINED LDA through all layers
        # Sort layers by branch index to process in order
        layer_names = list(self.data_dict.keys())

        # Extract and sort layers by branch index
        layers_with_idx = []
        for layer_name in layer_names:
            idx = self._parse_branch_layer_index(layer_name)
            if idx is not None:
                layers_with_idx.append((idx, layer_name))
        layers_with_idx.sort(key=lambda x: x[0])

        # Chain LDA through all layers
        prev_vout_lin = None  # Output from previous layer (None for first layer)
        final_vout_lin = None

        for _layer_idx, layer_name in layers_with_idx:
            layer_data = self.data_dict[layer_name]
            if "excitation" not in layer_data:
                continue

            # Compute LDA signals with chained branch input
            lda_result = self.compute_lda_aggregated_signals_chained(
                layer_data, labels_np, prev_vout_lin
            )

            if lda_result[0] is not None:
                _E_lin, _I_lin, _Vb_lin, Vout_lin = lda_result
                prev_vout_lin = Vout_lin  # Use as input for next layer
                final_vout_lin = Vout_lin
            else:
                # Fall back to trained network output
                prev_vout_lin = layer_data["output"].detach().cpu().numpy()
                final_vout_lin = prev_vout_lin

        # Use LDA outputs to compute accuracy
        # Note: final_vout_lin is already the LDA-computed output of the soma layer
        # (since the loop above processes ALL layers including soma).
        # We only need to apply the decoder if there is one.
        lda_accuracy = None
        lda_preds = None

        if final_vout_lin is not None:
            with torch.no_grad():
                vout_tensor = torch.tensor(final_vout_lin, dtype=torch.float32).to(
                    device
                )

                if hasattr(model, "decoder") and model.decoder is not None:
                    # Use decoder if available
                    lda_logits = model.decoder(vout_tensor)
                else:
                    # No decoder - final_vout_lin is already the soma layer's output
                    # (LDA was applied to all layers including soma in the loop above)
                    # Use it directly as logits
                    lda_logits = vout_tensor

                lda_preds = lda_logits.argmax(dim=1).cpu().numpy()
                lda_accuracy = accuracy_score(labels_np, lda_preds)
        results = {
            "trained_accuracy": trained_accuracy,
            "lda_accuracy": lda_accuracy,
            "trained_predictions": trained_preds.tolist(),
            "lda_predictions": lda_preds.tolist() if lda_preds is not None else None,
            "accuracy_difference": (
                lda_accuracy - trained_accuracy if lda_accuracy is not None else None
            ),
        }

        if lda_accuracy is not None:
            self.logger.info(
                f"Difference (LDA - Trained): {results['accuracy_difference']:+.4f}"
            )

        return results

    def _generate_information_plots(
        self, results: dict, save_path: str, somatic_synapses: bool
    ):
        """Generate plots for information analysis results."""
        try:
            # Initialize plotter for enhanced visualizations
            # plotter = InformationAnalysisPlotter()  # Unused variable

            # Generate dendritic depth plots based on enabled analysis modes
            if results and "layer_statistics" in results:
                per_layer_enabled = results.get("per_layer_analysis_enabled", False)
                per_einet_enabled = results.get("per_einet_analysis_enabled", False)

                if per_layer_enabled and per_einet_enabled:
                    self.logger.info(
                        "Both per_layer and per_einet analysis enabled - generating detailed comparison plots..."
                    )
                    # Generate structured plots grouped by major layer and network type
                    plot_structured_ei_network_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )
                elif per_layer_enabled and not per_einet_enabled:
                    self.logger.info(
                        "per_layer analysis enabled, per_einet disabled - generating layer-wise plots..."
                    )
                    # Aggregate E/I networks, but keep layers separate
                    plot_layer_wise_aggregated_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )
                elif not per_layer_enabled and per_einet_enabled:
                    self.logger.info(
                        "per_einet analysis enabled, per_layer disabled - generating network-wise plots..."
                    )
                    # Aggregate layers, but keep E/I networks separate
                    plot_network_wise_aggregated_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )
                else:
                    self.logger.info(
                        "Both per_layer and per_einet analysis disabled - generating overall aggregated plots..."
                    )
                    # Aggregate both layers and E/I networks
                    plot_fully_aggregated_analysis(
                        results["layer_statistics"],
                        save_path=save_path,
                        logger=self.logger,
                        somatic_synapses=somatic_synapses,
                    )

            # Generate info vs EI depth per dendritic layer plots if we have layer statistics
            self.logger.debug(
                f"Checking for layer_statistics: results exists={results is not None}"
            )
            if results:
                self.logger.debug(f"Results keys: {list(results.keys())}")
                self.logger.debug(
                    f"layer_statistics in results: {'layer_statistics' in results}"
                )
                if "layer_statistics" in results:
                    self.logger.debug(
                        f"layer_statistics length: {len(results['layer_statistics'])}"
                    )

            if (
                results
                and "layer_statistics" in results
                and len(results["layer_statistics"]) > 0
            ):
                self.logger.debug(
                    f"Calling plot_info_vs_ei_depth_per_dendritic_layer with {len(results['layer_statistics'])} layer statistics"
                )
                plot_info_vs_ei_depth_per_dendritic_layer(
                    results["layer_statistics"],
                    save_path=save_path,
                    logger=self.logger,
                    somatic_synapses=somatic_synapses,
                )
            else:
                self.logger.warning(
                    "Skipping plot_info_vs_ei_depth_per_dendritic_layer - no layer statistics found"
                )

            # Generate variance analysis plots if variance analysis was computed
            if results and "variance_analysis" in results:
                try:
                    self.logger.info(
                        "Generating information variance analysis plots..."
                    )
                    self.logger.debug(
                        f"Variance analysis keys: {list(results['variance_analysis'].keys())}"
                    )
                    from dendritic_modeling.plotting.visualizations.information_plots import (
                        plot_information_variance_analysis,
                    )

                    plot_information_variance_analysis(
                        results["variance_analysis"],
                        save_path=save_path,
                        somatic_synapses=somatic_synapses,
                        logger=self.logger,
                    )
                    self.logger.info("Variance plots completed successfully!")
                except Exception as e:
                    self.logger.error(f"Error generating variance plots: {e}")
                    traceback.print_exc()
            else:
                self.logger.warning(
                    "No variance_analysis found in results - skipping variance plots"
                )

        except Exception as e:
            self.logger.error(f"Error generating information plots: {e}")
