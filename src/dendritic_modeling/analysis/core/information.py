"""Integrated information analyzer for dendritic networks.

This module provides a unified interface for computing various information-theoretic
metrics using the multivariate methods from information.py. It works directly
with the main configuration system.
"""

import logging
import os
import time
import traceback
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from tqdm import tqdm

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

        def _get(obj: Any, key: str, default: Any = None) -> Any:
            """Get `key` from obj via attribute/dict access (DictConfig-safe)."""
            if obj is None:
                return default
            try:
                val = getattr(obj, key)
            except Exception:
                try:
                    val = obj.get(key, default)  # type: ignore[attr-defined]
                except Exception:
                    return default
            return default if val is None else val

        def _get_path(obj: Any, path: tuple[str, ...], default: Any = None) -> Any:
            cur = obj
            for k in path:
                cur = _get(cur, k, None)
                if cur is None:
                    return default
            return cur

        # New (preferred) grouped configs (all optional for backward compatibility)
        selection_cfg = _get(params, "selection", None)
        compute_cfg = _get(params, "compute", None)
        estimator_cfg = _get(params, "estimator", None)
        baselines_cfg = _get(params, "baselines", None)
        pid_cfg = _get(params, "pid", None)
        gaussian_fisher_cfg = _get(compute_cfg, "gaussian_fisher", None)

        # ------------------------------------------------------------------
        # Extract configuration parameters (support both new nested and legacy flat keys).
        # ------------------------------------------------------------------
        self.method = str(
            _get(estimator_cfg, "method", _get(params, "method", "kraskov"))
        )
        self.max_samples = _get(
            compute_cfg, "max_samples", _get(params, "max_samples", None)
        )

        self.n_neighbors = int(
            _get_path(
                estimator_cfg,
                ("kraskov", "n_neighbors"),
                _get(params, "n_neighbors", 10),
            )
        )
        self.n_bins = int(
            _get_path(estimator_cfg, ("binned", "n_bins"), _get(params, "n_bins", 20))
        )
        self.copula_type = str(
            _get_path(
                estimator_cfg,
                ("copula", "type"),
                _get(params, "copula_type", "gaussian"),
            )
        )

        # Label-shuffle null baseline (finite-sample bias diagnostic)
        self.mi_null_shuffles = int(
            _get_path(
                baselines_cfg,
                ("label_shuffle_null", "n_shuffles"),
                _get(params, "mi_null_shuffles", 0),
            )
            or 0
        )
        self.mi_null_seed = int(
            _get_path(
                baselines_cfg,
                ("label_shuffle_null", "seed"),
                _get(params, "mi_null_seed", 0),
            )
            or 0
        )

        def _norm(x: str) -> str:
            return str(x).strip().lower().replace("-", "_").replace(" ", "_")

        # Which metric groups should receive label-shuffle (*_null) baselines.
        # Default: only basic MI terms involving C.
        null_components_raw = list(
            _get_path(
                baselines_cfg,
                ("label_shuffle_null", "components"),
                _get(params, "mi_null_components", ["basic"]),
            )
            or ["basic"]
        )
        self.mi_null_components = {
            _norm(x) for x in null_components_raw if str(x).strip()
        } or {"basic"}

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

        self.pid_binarize = bool(
            _get(pid_cfg, "binarize", _get(params, "pid_binarize", False))
        )
        self.pid_binarize_method = str(
            _get(
                pid_cfg,
                "binarize_method",
                _get(params, "pid_binarize_method", "median"),
            )
        )
        self.pid_binarize_threshold = float(
            _get(
                pid_cfg,
                "binarize_threshold",
                _get(params, "pid_binarize_threshold", 0.5),
            )
        )

        # Ablation-aligned extras (conditional MI terms and single-branch -> layer proxies)
        self.compute_ablation_aligned_mi = bool(
            getattr(params, "compute_ablation_aligned_mi", True)
        )
        self.compute_upstream_unique_cmi = bool(
            getattr(params, "compute_upstream_unique_cmi", False)
        )
        self.compute_layer_total_proxies = bool(
            getattr(params, "compute_layer_total_proxies", True)
        )
        self.layer_total_topk = int(
            _get(compute_cfg, "layer_total_topk", _get(params, "layer_total_topk", 10))
        )

        # ------------------------------------------------------------------
        # Preferred list-based selection of components.
        #
        # If provided, overrides the individual compute_* booleans above.
        # ------------------------------------------------------------------
        components_raw = list(
            _get(selection_cfg, "components", _get(params, "components", [])) or []
        )
        if components_raw:
            comps = {_norm(x) for x in components_raw if str(x).strip()}
            self.compute_basic_mi = bool({"basic", "basic_mi"}.intersection(comps))
            self.compute_pairwise_mi = bool(
                {"pairwise", "pairwise_mi"}.intersection(comps)
            )
            self.compute_conditional_mi = bool(
                {"conditional", "conditional_mi"}.intersection(comps)
            )
            self.compute_pid = "pid" in comps
            self.compute_gaussian_fisher = bool(
                {"gaussian_fisher", "gaussianfisher", "fisher"}.intersection(comps)
            )

            self.compute_ablation_aligned_mi = bool(
                {"ablation_aligned", "ablation"}.intersection(comps)
            )
            self.compute_layer_total_proxies = bool(
                {"layer_proxies", "layer_total_proxies", "layer_total"}.intersection(
                    comps
                )
            )
            self.compute_upstream_unique_cmi = bool(
                {"upstream_unique_cmi", "upstream_unique"}.intersection(comps)
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

        # Gaussian/Fisher proxy configuration (enabled via selection.components).
        self.gaussian_fisher_aggregation = (
            str(_get(gaussian_fisher_cfg, "aggregation", "sum")).strip().lower()
        )
        self.gaussian_fisher_mi_transform = (
            str(_get(gaussian_fisher_cfg, "mi_transform", "half_log1p")).strip().lower()
        )
        self.gaussian_fisher_eps = float(
            _get(
                gaussian_fisher_cfg,
                "ridge",
                _get(gaussian_fisher_cfg, "eps", 1e-8),
            )
        )

        self.include_vinf = bool(
            _get(compute_cfg, "include_vinf", _get(params, "include_vinf", False))
        )  # Default to False to match 245MR behavior
        self.normalize_mi = bool(
            _get(compute_cfg, "normalize_mi", _get(params, "normalize_mi", False))
        )
        # Estimators operate in nats; for reporting/plots we optionally convert to bits.
        self.output_units = _norm(
            _get(compute_cfg, "output_units", _get(params, "output_units", "bits"))
        )
        if self.output_units not in {"bits", "bit", "nats", "nat"}:
            self.logger.warning(
                "Unknown output_units=%r; falling back to 'bits'.", self.output_units
            )
            self.output_units = "bits"
        # Canonicalize.
        self.output_units = "bits" if self.output_units in {"bits", "bit"} else "nats"
        self.verbose = bool(
            _get(compute_cfg, "verbose", _get(params, "verbose", False))
        )

        # Computation level control - determines the granularity of information analysis
        # "branch": Individual branches (most granular and accurate)
        # "layer": Layer-level aggregation (treats each layer as a unit)
        # "network": Full network aggregation (most efficient)
        self.computation_level = str(
            _get(
                compute_cfg,
                "computation_level",
                _get(params, "computation_level", "single_branch"),
            )
        )

        # Branch aggregation method - only used when computation_level is "layer_branch" or "all_branch"
        # Options: "mean", "multivariate", "sample"
        self.branch_aggregation = str(
            _get(
                compute_cfg,
                "branch_aggregation",
                _get(params, "branch_aggregation", "mean"),
            )
        )

        # Per-neuron analysis - return information values per individual neuron
        self.per_neuron_analysis = bool(
            _get(
                compute_cfg,
                "per_neuron_analysis",
                _get(params, "per_neuron_analysis", False),
            )
        )

        # Per-EI-network analysis - analyze excitatory and inhibitory networks separately
        self.per_einet_analysis = bool(
            _get(
                compute_cfg,
                "per_einet_analysis",
                _get(params, "per_einet_analysis", False),
            )
        )

        # Per-layer analysis - compute enhanced analysis for each layer separately
        self.per_layer_analysis = bool(
            _get(
                compute_cfg,
                "per_layer_analysis",
                _get(params, "per_layer_analysis", False),
            )
        )

        # Preferred list-based selection of scopes (overrides per_* flags if provided).
        #
        # `views` is a backward-compatible alias for `scopes`.
        scopes_raw = list(
            _get(selection_cfg, "scopes", _get(params, "scopes", [])) or []
        )
        if not scopes_raw:
            scopes_raw = list(
                _get(selection_cfg, "views", _get(params, "views", [])) or []
            )
        if scopes_raw:
            scopes = {_norm(x) for x in scopes_raw if str(x).strip()}
            self.per_neuron_analysis = bool(
                {"neuron", "per_neuron"}.intersection(scopes)
            )
            self.per_einet_analysis = bool({"einet", "per_einet"}.intersection(scopes))
            self.per_layer_analysis = bool({"layer", "per_layer"}.intersection(scopes))

        # Compute LDA-based versions of E, I, Vb
        self.compute_lda_weights = bool(
            _get(
                compute_cfg,
                "compute_lda_weights",
                _get(params, "compute_lda_weights", False),
            )
        )

        # Number of random-weight baselines for LDA aggregation (*_sh) to generate.
        #
        # Preferred key: baselines.random_weights.n_shuffles
        # Backward-compatible alias (legacy): baselines.signal_shuffle.n_shuffles
        self.lda_n_shuffles = int(
            _get_path(
                baselines_cfg,
                ("random_weights", "n_shuffles"),
                _get_path(
                    baselines_cfg,
                    ("signal_shuffle", "n_shuffles"),
                    _get(params, "lda_n_shuffles", 0),
                ),
            )
            or 0
        )
        self.random_weights_seed = int(
            _get_path(
                baselines_cfg,
                ("random_weights", "seed"),
                _get_path(
                    baselines_cfg,
                    ("signal_shuffle", "seed"),
                    _get(params, "random_weights_seed", 0),
                ),
            )
            or 0
        )

        # Preferred list-based selection of signal variants.
        variants_raw = list(
            _get(selection_cfg, "signal_variants", _get(params, "signal_variants", []))
            or []
        )
        if variants_raw:
            variants = {_norm(x) for x in variants_raw if str(x).strip()}
            want_lda = "lda" in variants
            want_random = bool({"random", "shuffle", "shuffled"}.intersection(variants))

            # "random" is a baseline for the LDA aggregation; enable LDA when requested.
            self.compute_lda_weights = bool(want_lda or want_random)
            self.lda_n_shuffles = self.lda_n_shuffles if want_random else 0

        if self.verbose:
            self.logger.setLevel(logging.DEBUG)

        # Create estimators for each method
        self.estimators = {}
        # For now, just use the single method specified, but keep the dict structure for compatibility
        method_params: dict[str, Any] = {}
        decoder_cfg = _get(estimator_cfg, "decoder", None)
        decoder_cont_cfg = _get(decoder_cfg, "continuous", None)
        binned_cfg = _get(estimator_cfg, "binned", None)
        if _norm(self.method) == "decoder":
            method_params = {
                "cv_folds": int(
                    _get(decoder_cfg, "cv_folds", _get(params, "decoder_cv_folds", 5))
                ),
                "C": float(_get(decoder_cfg, "C", _get(params, "decoder_C", 1.0))),
                "standardize": bool(
                    _get(
                        decoder_cfg,
                        "standardize",
                        _get(params, "decoder_standardize", True),
                    )
                ),
                "seed": int(_get(decoder_cfg, "seed", _get(params, "decoder_seed", 0))),
                "continuous_strategy": str(
                    _get(
                        decoder_cont_cfg,
                        "strategy",
                        _get(params, "decoder_continuous_strategy", "none"),
                    )
                ),
                "continuous_max_dim": int(
                    _get(
                        decoder_cont_cfg,
                        "max_total_dim",
                        _get(params, "decoder_continuous_max_total_dim", 2),
                    )
                ),
                "gaussian_ridge": float(
                    _get(
                        decoder_cont_cfg,
                        "gaussian_ridge",
                        _get(params, "decoder_gaussian_ridge", 1e-6),
                    )
                ),
                # Used if decoder_continuous_strategy="binned"
                "binning_strategy": str(
                    _get(
                        binned_cfg,
                        "binning_strategy",
                        _get(params, "binning_strategy", "quantile"),
                    )
                ),
                "bias_correction": str(
                    _get(
                        binned_cfg,
                        "bias_correction",
                        _get(params, "binning_bias_correction", "none"),
                    )
                ),
            }
        elif _norm(self.method) == "binned":
            method_params = {
                "binning_strategy": str(
                    _get(
                        binned_cfg,
                        "binning_strategy",
                        _get(params, "binning_strategy", "quantile"),
                    )
                ),
                "bias_correction": str(
                    _get(
                        binned_cfg,
                        "bias_correction",
                        _get(params, "binning_bias_correction", "none"),
                    )
                ),
            }
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
            if not has_einet_core(model):
                self.logger.warning(
                    "Model does not expose an ExcitationInhibitionNetwork. Skipping information analysis."
                )
                return

            self.logger.info("Model validation passed - starting analysis")
            analysis_start_time = time.time()

            # Prepare data
            self.logger.info(f"Preparing data: moving model to {device}")
            model = model.to(device)
            model.eval()
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

            if self.computation_level == "single_branch":
                results = self._compute_single_branch_results(C, entropy_C)
                if results is None:
                    return

            elif self.computation_level == "layer_branch":
                results = self._compute_layer_branch_results(C)
                if results is None:
                    return

            else:  # computation_level == "all_branch"
                results = self._compute_all_branch_results(
                    model=model,
                    inputs=inputs,
                    C=C,
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

            E_all_branches_raw = layer_data["excitation"]
            I_all_branches_raw = layer_data["inhibition"]
            Vb_all_branches_raw = layer_data.get("branch_input")
            Vout_all_branches = layer_data["output"].detach().cpu().numpy()

            # Variable availability flags for filtering/averaging:
            # - synapse-driven variables only exist if the corresponding synapses exist
            # - dendritic branch input (Vb) only exists if branch_input was present
            has_exc_synapses = bool(layer_data.get("has_exc_synapses", True))
            has_inh_synapses = bool(layer_data.get("has_inh_synapses", True))
            has_branch_input = bool(
                layer_data.get("has_branch_input", Vb_all_branches_raw is not None)
            )

            E_all_branches_filtered, I_all_branches_filtered = (
                self.ensure_inhibitory_data(E_all_branches_raw, I_all_branches_raw)
            )

            E_all_branches = E_all_branches_filtered.detach().cpu().numpy()
            I_all_branches = I_all_branches_filtered.detach().cpu().numpy()

            if Vb_all_branches_raw is not None:
                Vb_all_branches = Vb_all_branches_raw.detach().cpu().numpy()
            else:
                Vb_all_branches = None

            if self.compute_lda_weights:
                E_lin, I_lin, Vb_lin, Vout_lin = self.compute_lda_aggregated_signals(
                    layer_data, C
                )
            else:
                E_lin, I_lin, Vb_lin, Vout_lin = None, None, None, None

            shuffled_signals_list = []
            if self.compute_lda_weights and self.lda_n_shuffles > 0:
                for shuffle_idx in range(self.lda_n_shuffles):
                    E_sh, I_sh, Vb_sh, Vout_sh = (
                        self.compute_shuffled_aggregated_signals(
                            layer_data, C, shuffle_idx
                        )
                    )
                    shuffled_signals_list.append((E_sh, I_sh, Vb_sh, Vout_sh))

            n_branches = E_all_branches.shape[1]
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

                E_branch = E_all_branches[:, branch_idx : branch_idx + 1]
                I_branch = I_all_branches[:, branch_idx : branch_idx + 1]
                Vout_branch = Vout_all_branches[:, branch_idx : branch_idx + 1]
                Vb_branch = (
                    Vb_all_branches[:, branch_idx : branch_idx + 1]
                    if Vb_all_branches is not None
                    else None
                )

                branch_results = self.compute_metrics(
                    E_branch, I_branch, Vout_branch, C, None, Vb_branch
                )
                self._try_add_label_shuffle_null_baselines_in_place(
                    branch_results,
                    E=E_branch,
                    I_var=I_branch,
                    Vout=Vout_branch,
                    C=C,
                    Vb=Vb_branch,
                    Vinf=None,
                    has_exc_synapses=has_exc_synapses,
                    has_inh_synapses=has_inh_synapses,
                    has_branch_input=has_branch_input and (Vb_branch is not None),
                    seed_offset=int(layer_count * 10_000 + branch_idx),
                    failure_context=f"branch {branch_idx}",
                )
                all_results.append(branch_results)
                layer_branch_results.append(branch_results)
                branch_names.append(f"{layer_name}_branch_{branch_idx}")

                if self.compute_lda_weights and E_lin is not None:
                    E_branch_lin = E_lin[:, branch_idx : branch_idx + 1]
                    I_branch_lin = I_lin[:, branch_idx : branch_idx + 1]
                    Vb_branch_lin = (
                        Vb_lin[:, branch_idx : branch_idx + 1]
                        if has_branch_input and Vb_lin is not None
                        else None
                    )
                    Vout_branch_lin = Vout_lin[:, branch_idx : branch_idx + 1]

                    branch_results_lin = self.compute_metrics(
                        E_branch_lin,
                        I_branch_lin,
                        Vout_branch_lin,
                        C,
                        None,
                        Vb_branch_lin,
                    )
                    self._try_add_label_shuffle_null_baselines_in_place(
                        branch_results_lin,
                        E=E_branch_lin,
                        I_var=I_branch_lin,
                        Vout=Vout_branch_lin,
                        C=C,
                        Vb=Vb_branch_lin,
                        Vinf=None,
                        has_exc_synapses=has_exc_synapses,
                        has_inh_synapses=has_inh_synapses,
                        has_branch_input=has_branch_input
                        and (Vb_branch_lin is not None),
                        seed_offset=int(1_000_000 + layer_count * 10_000 + branch_idx),
                        failure_context=f"LDA branch {branch_idx}",
                    )
                    self._merge_renamed_metric_dicts_in_place(
                        target=branch_results,
                        source=branch_results_lin,
                        token_map={
                            "E": "E_lin",
                            "I": "I_lin",
                            "Vb": "Vb_lin",
                            "Vout": "Vout_lin",
                            "Vinf": "Vinf_lin",
                        },
                    )

                if shuffled_signals_list:
                    shuffled_results = []
                    for _shuffle_idx, (E_sh, I_sh, Vb_sh, Vout_sh) in enumerate(
                        shuffled_signals_list
                    ):
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
                        target=branch_results,
                        sources=shuffled_results,
                        token_map={
                            "E": "E_sh",
                            "I": "I_sh",
                            "Vb": "Vb_sh",
                            "Vout": "Vout_sh",
                            "Vinf": "Vinf_sh",
                        },
                    )

                self._filter_result_metrics_in_place(
                    branch_results,
                    has_exc_synapses=has_exc_synapses,
                    has_inh_synapses=has_inh_synapses,
                    has_branch_input=has_branch_input,
                )

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

        if not all_results:
            self.logger.warning("No branch results to average")
            return None

        results = self._average_mi_results(all_results)
        results["branch_names"] = branch_names
        results["num_branches"] = len(all_results)
        results["layer_statistics"] = layer_results
        results["computation_level"] = "single_branch"
        return results

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

            # Keep Vb as None when it doesn't exist (e.g., outermost distal layer)
            # or when config disables branch input analysis.
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
                seed_offset=len(all_results),
                failure_context=f"layer {layer_name}",
            )
            all_results.append(layer_results)
            layer_names.append(layer_name)

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

        layer_excitation = []
        layer_inhibition = []
        layer_branch_input = []
        missing_vb_in_any_layer = False
        layer_output = []

        for layer_data in self.data_dict.values():
            if "excitation" in layer_data:
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

        Vinf = self._compute_final_output_signal(model=model, inputs=inputs)

        results = self.compute_metrics(E, I_var, Vout, C_for_metrics, Vinf, Vb)
        self._try_add_label_shuffle_null_baselines_in_place(
            results,
            E=E,
            I_var=I_var,
            Vout=Vout,
            C=C_for_metrics,
            Vb=Vb,
            Vinf=Vinf,
            has_exc_synapses=True,
            has_inh_synapses=True,
            has_branch_input=Vb is not None,
            seed_offset=999_999,
            failure_context="all-branch analysis",
        )
        results["aggregation_method"] = self.branch_aggregation
        results["computation_level"] = "all_branch"
        return results

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

        summary = self.get_summary(results)
        summary_path = os.path.join(save_path, f"{filename}_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary)

        try:
            if "per_layer_analysis" in results:
                self._save_per_layer_ei_summaries(
                    results["per_layer_analysis"], save_path, filename
                )
            self._generate_information_plots(results, save_path, somatic_synapses)
        except Exception as plot_err:
            self.logger.warning(f"Post-save visualization failed: {plot_err}")

        self.logger.info("Results saved successfully!")

    def _add_synthetic_soma_entry_if_needed(
        self, results: dict[str, Any], model: BaseModel
    ) -> bool:
        # This is only needed when somatic_synapses=False and the soma layer
        # was not captured by the forward hooks.
        somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
        self.logger.info(f"Somatic synapses enabled: {somatic_synapses}")

        soma_already_captured = self._soma_layer_already_captured(results)
        if (
            not somatic_synapses
            and "layer_statistics" in results
            and not soma_already_captured
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

    def _soma_layer_already_captured(self, results: dict[str, Any]) -> bool:
        if "layer_statistics" not in results:
            return False

        for layer_name in results["layer_statistics"].keys():
            if "branch_layers." not in layer_name:
                continue
            try:
                int(layer_name.split("branch_layers.")[1].split(".")[0])
                layer_data = results["layer_statistics"][layer_name]
                if layer_data.get("has_synapses") is False:
                    self.logger.info(f"Soma layer already captured: {layer_name}")
                    return True
            except (ValueError, IndexError):
                continue
        return False

    @staticmethod
    def _find_real_soma_layer(layer_statistics: dict[str, Any]) -> Optional[str]:
        max_branch_idx = -1
        real_soma_layer = None

        for layer_name in layer_statistics.keys():
            if "branch_layers." not in layer_name:
                continue
            try:
                branch_idx = int(layer_name.split("branch_layers.")[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            if branch_idx > max_branch_idx:
                max_branch_idx = branch_idx
                real_soma_layer = layer_name

        return real_soma_layer

    def _extract_soma_statistics(
        self, real_soma_layer: Optional[str], layer_statistics: dict[str, Any]
    ) -> dict[str, float]:
        soma_stats = {
            "soma_vout_mean": 0.0,
            "soma_vout_std": 0.0,
            "soma_vb_mean": 0.0,
            "soma_vb_std": 0.0,
            "soma_vb_lin_mean": 0.0,
            "soma_vb_lin_std": 0.0,
            "soma_vb_vout_mean": 0.0,
            "soma_vb_vout_std": 0.0,
            "soma_vb_vout_c_mean": 0.0,
            "soma_vb_vout_c_std": 0.0,
            "soma_vb_lin_vout_mean": 0.0,
            "soma_vb_lin_vout_std": 0.0,
            "soma_vb_lin_vout_c_mean": 0.0,
            "soma_vb_lin_vout_c_std": 0.0,
            "soma_vout_lin_mean": 0.0,
            "soma_vout_lin_std": 0.0,
        }

        if real_soma_layer and real_soma_layer in layer_statistics:
            real_soma_stats = layer_statistics[real_soma_layer]
            soma_stats.update(
                {
                    "soma_vout_mean": real_soma_stats.get("I(Vout;C)_mean", 0.0),
                    "soma_vout_std": real_soma_stats.get("I(Vout;C)_std", 0.0),
                    "soma_vb_mean": real_soma_stats.get("I(Vb;C)_mean", 0.0),
                    "soma_vb_std": real_soma_stats.get("I(Vb;C)_std", 0.0),
                    "soma_vb_lin_mean": real_soma_stats.get("I(Vb_lin;C)_mean", 0.0),
                    "soma_vb_lin_std": real_soma_stats.get("I(Vb_lin;C)_std", 0.0),
                    "soma_vb_vout_mean": real_soma_stats.get("I(Vb;Vout)_mean", 0.0),
                    "soma_vb_vout_std": real_soma_stats.get("I(Vb;Vout)_std", 0.0),
                    "soma_vb_vout_c_mean": real_soma_stats.get(
                        "I(Vb;Vout|C)_mean", 0.0
                    ),
                    "soma_vb_vout_c_std": real_soma_stats.get("I(Vb;Vout|C)_std", 0.0),
                    "soma_vb_lin_vout_mean": real_soma_stats.get(
                        "I(Vb_lin;Vout_lin)_mean", 0.0
                    ),
                    "soma_vb_lin_vout_std": real_soma_stats.get(
                        "I(Vb_lin;Vout_lin)_std", 0.0
                    ),
                    "soma_vb_lin_vout_c_mean": real_soma_stats.get(
                        "I(Vb_lin;Vout_lin|C)_mean", 0.0
                    ),
                    "soma_vb_lin_vout_c_std": real_soma_stats.get(
                        "I(Vb_lin;Vout_lin|C)_std", 0.0
                    ),
                    "soma_vout_lin_mean": real_soma_stats.get(
                        "I(Vout_lin;C)_mean", 0.0
                    ),
                    "soma_vout_lin_std": real_soma_stats.get("I(Vout_lin;C)_std", 0.0),
                }
            )
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
        else:
            self.logger.warning(
                "Could not find real soma layer for I(Vout;C) computation"
            )

        return soma_stats

    def _build_synthetic_soma_entry(
        self, soma_stats: dict[str, float]
    ) -> dict[str, Any]:
        soma_entry = {
            "I(E;C)_mean": 0.0,
            "I(E;C)_std": 0.0,
            "I(I;C)_mean": 0.0,
            "I(I;C)_std": 0.0,
            "I(Vb;C)_mean": soma_stats["soma_vb_mean"],
            "I(Vb;C)_std": soma_stats["soma_vb_std"],
            "I(Vout;C)_mean": soma_stats["soma_vout_mean"],
            "I(Vout;C)_std": soma_stats["soma_vout_std"],
            "I(E,I;C)_mean": 0.0,
            "I(E,I;C)_std": 0.0,
            "I(E_lin;C)_mean": 0.0,
            "I(E_lin;C)_std": 0.0,
            "I(I_lin;C)_mean": 0.0,
            "I(I_lin;C)_std": 0.0,
            "I(Vb_lin;C)_mean": soma_stats["soma_vb_lin_mean"],
            "I(Vb_lin;C)_std": soma_stats["soma_vb_lin_std"],
            "I(Vout_lin;C)_mean": soma_stats["soma_vout_lin_mean"],
            "I(Vout_lin;C)_std": soma_stats["soma_vout_lin_std"],
            "I(E_lin,I_lin;C)_mean": 0.0,
            "I(E_lin,I_lin;C)_std": 0.0,
            "I(E;I)_mean": 0.0,
            "I(E;I)_std": 0.0,
            "I(E;Vout)_mean": 0.0,
            "I(E;Vout)_std": 0.0,
            "I(I;Vout)_mean": 0.0,
            "I(I;Vout)_std": 0.0,
            "I(E;Vb)_mean": 0.0,
            "I(E;Vb)_std": 0.0,
            "I(I;Vb)_mean": 0.0,
            "I(I;Vb)_std": 0.0,
            "I(Vb;Vout)_mean": soma_stats["soma_vb_vout_mean"],
            "I(Vb;Vout)_std": soma_stats["soma_vb_vout_std"],
            "I(E_lin;I_lin)_mean": 0.0,
            "I(E_lin;I_lin)_std": 0.0,
            "I(E_lin;Vout_lin)_mean": 0.0,
            "I(E_lin;Vout_lin)_std": 0.0,
            "I(I_lin;Vout_lin)_mean": 0.0,
            "I(I_lin;Vout_lin)_std": 0.0,
            "I(E_lin;Vb_lin)_mean": 0.0,
            "I(E_lin;Vb_lin)_std": 0.0,
            "I(I_lin;Vb_lin)_mean": 0.0,
            "I(I_lin;Vb_lin)_std": 0.0,
            "I(Vb_lin;Vout_lin)_mean": soma_stats["soma_vb_lin_vout_mean"],
            "I(Vb_lin;Vout_lin)_std": soma_stats["soma_vb_lin_vout_std"],
            "I(E;I|C)_mean": 0.0,
            "I(E;I|C)_std": 0.0,
            "I(E;Vout|C)_mean": 0.0,
            "I(E;Vout|C)_std": 0.0,
            "I(I;Vout|C)_mean": 0.0,
            "I(I;Vout|C)_std": 0.0,
            "I(E;Vb|C)_mean": 0.0,
            "I(E;Vb|C)_std": 0.0,
            "I(I;Vb|C)_mean": 0.0,
            "I(I;Vb|C)_std": 0.0,
            "I(Vb;Vout|C)_mean": soma_stats["soma_vb_vout_c_mean"],
            "I(Vb;Vout|C)_std": soma_stats["soma_vb_vout_c_std"],
            "I(E_lin;I_lin|C)_mean": 0.0,
            "I(E_lin;I_lin|C)_std": 0.0,
            "I(E_lin;Vout_lin|C)_mean": 0.0,
            "I(E_lin;Vout_lin|C)_std": 0.0,
            "I(I_lin;Vout_lin|C)_mean": 0.0,
            "I(I_lin;Vout_lin|C)_std": 0.0,
            "I(E_lin;Vb_lin|C)_mean": 0.0,
            "I(E_lin;Vb_lin|C)_std": 0.0,
            "I(I_lin;Vb_lin|C)_mean": 0.0,
            "I(I_lin;Vb_lin|C)_std": 0.0,
            "I(Vb_lin;Vout_lin|C)_mean": soma_stats["soma_vb_lin_vout_c_mean"],
            "I(Vb_lin;Vout_lin|C)_std": soma_stats["soma_vb_lin_vout_c_std"],
        }
        soma_entry = {
            k: v
            for k, v in soma_entry.items()
            if self._keep_metric_key_given_availability(
                k,
                has_exc_synapses=False,
                has_inh_synapses=False,
                has_branch_input=True,
            )
        }
        soma_entry["has_synapses"] = False
        return soma_entry

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
            if "branch_layers" in layer_name:
                try:
                    idx = int(layer_name.split("branch_layers.")[1].split(".")[0])
                    layers_with_idx.append((idx, layer_name))
                except (ValueError, IndexError):
                    continue
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
