"""Integrated information analyzer for dendritic networks.

This module provides a unified interface for computing various information-theoretic
metrics using the multivariate methods from information.py. It works directly
with the main configuration system.
"""

import logging
import os
import re
import time
import traceback
from typing import Any, Optional, Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.config import InformationAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
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
    pid_synergy_redundancy,
)
from dendritic_modeling.utils.lda import (
    apply_nonnegative_lda,
    apply_random_weights,
    fit_nonnegative_lda,
)


class InformationAnalyzer(AbstractAnalyzer):
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
            Configuration from the main config system (analysis.information_analysis)
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

    def _compute_basic_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
        Vb: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute mutual information between each variable and C."""
        self.logger.debug(
            f"Computing basic MI: E{E.shape}, I{I_var.shape}, Vout{Vout.shape}, C{C.shape}"
        )
        metrics = {}

        # I(E;C)
        self.logger.debug("  Computing I(E;C)...")
        start_time = time.time()
        metrics["I(E;C)"] = self.estimators[self.method].mutual_information(E, C)
        self.logger.debug(
            f"  I(E;C) = {metrics['I(E;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(I;C)
        self.logger.debug("  Computing I(I;C)...")
        start_time = time.time()
        metrics["I(I;C)"] = self.estimators[self.method].mutual_information(I_var, C)
        self.logger.debug(
            f"  I(I;C) = {metrics['I(I;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(Vb;C) if available
        if Vb is not None:
            self.logger.debug("  Computing I(Vb;C)...")
            start_time = time.time()
            metrics["I(Vb;C)"] = self.estimators[self.method].mutual_information(Vb, C)
            self.logger.debug(
                f"  I(Vb;C) = {metrics['I(Vb;C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        # I(Vout;C)
        self.logger.debug("  Computing I(Vout;C)...")
        start_time = time.time()
        metrics["I(Vout;C)"] = self.estimators[self.method].mutual_information(Vout, C)
        self.logger.debug(
            f"  I(Vout;C) = {metrics['I(Vout;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(Vinf;C) if available
        if Vinf is not None and self.include_vinf:
            self.logger.debug("  Computing I(Vinf;C)...")
            start_time = time.time()
            metrics["I(Vinf;C)"] = self.estimators[self.method].mutual_information(
                Vinf, C
            )
            self.logger.debug(
                f"  I(Vinf;C) = {metrics['I(Vinf;C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        # Joint MI: I(E,I;C)
        self.logger.debug("  Computing I(E,I;C)...")
        start_time = time.time()
        EI = np.hstack([E, I_var])
        metrics["I(E,I;C)"] = self.estimators[self.method].mutual_information(EI, C)
        self.logger.debug(
            f"  I(E,I;C) = {metrics['I(E,I;C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # Optionally normalize by entropy
        if self.normalize_mi:
            self.logger.debug("  Computing entropy H(C) for normalization...")
            start_time = time.time()
            H_C = self.estimators[self.method].entropy(C.reshape(-1, 1))
            self.logger.debug(
                f"  H(C) = {H_C:.4f} (took {time.time() - start_time:.1f}s)"
            )
            # Create a list of keys to avoid modifying dict during iteration
            keys_to_normalize = list(metrics.keys())
            for key in keys_to_normalize:
                metrics[f"{key}_normalized"] = metrics[key] / H_C if H_C > 0 else 0.0

        self.logger.debug("  Basic MI computation completed.")
        return metrics

    def _compute_pairwise_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
        Vb: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute pairwise mutual information between variables."""
        self.logger.debug("Computing pairwise MI between neural variables...")
        metrics = {}

        # I(E;I)
        self.logger.debug("  Computing I(E;I)...")
        start_time = time.time()
        metrics["I(E;I)"] = self.estimators[self.method].mutual_information(E, I_var)
        self.logger.debug(
            f"  I(E;I) = {metrics['I(E;I)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(E;Vout)
        self.logger.debug("  Computing I(E;Vout)...")
        start_time = time.time()
        metrics["I(E;Vout)"] = self.estimators[self.method].mutual_information(E, Vout)
        self.logger.debug(
            f"  I(E;Vout) = {metrics['I(E;Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(I;Vout)
        self.logger.debug("  Computing I(I;Vout)...")
        start_time = time.time()
        metrics["I(I;Vout)"] = self.estimators[self.method].mutual_information(
            I_var, Vout
        )
        self.logger.debug(
            f"  I(I;Vout) = {metrics['I(I;Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # Pairwise MI with Vb if available
        if Vb is not None:
            # I(E;Vb)
            self.logger.debug("  Computing I(E;Vb)...")
            start_time = time.time()
            metrics["I(E;Vb)"] = self.estimators[self.method].mutual_information(E, Vb)
            self.logger.debug(
                f"  I(E;Vb) = {metrics['I(E;Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vb)
            self.logger.debug("  Computing I(I;Vb)...")
            start_time = time.time()
            metrics["I(I;Vb)"] = self.estimators[self.method].mutual_information(
                I_var, Vb
            )
            self.logger.debug(
                f"  I(I;Vb) = {metrics['I(I;Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(Vb;Vout)
            self.logger.debug("  Computing I(Vb;Vout)...")
            start_time = time.time()
            metrics["I(Vb;Vout)"] = self.estimators[self.method].mutual_information(
                Vb, Vout
            )
            self.logger.debug(
                f"  I(Vb;Vout) = {metrics['I(Vb;Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        if Vinf is not None and self.include_vinf:
            # I(E;Vinf)
            self.logger.debug("  Computing I(E;Vinf)...")
            start_time = time.time()
            metrics["I(E;Vinf)"] = self.estimators[self.method].mutual_information(
                E, Vinf
            )
            self.logger.debug(
                f"  I(E;Vinf) = {metrics['I(E;Vinf)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vinf)
            self.logger.debug("  Computing I(I;Vinf)...")
            start_time = time.time()
            metrics["I(I;Vinf)"] = self.estimators[self.method].mutual_information(
                I_var, Vinf
            )
            self.logger.debug(
                f"  I(I;Vinf) = {metrics['I(I;Vinf)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(Vout;Vinf)
            self.logger.debug("  Computing I(Vout;Vinf)...")
            start_time = time.time()
            metrics["I(Vout;Vinf)"] = self.estimators[self.method].mutual_information(
                Vout, Vinf
            )
            self.logger.debug(
                f"  I(Vout;Vinf) = {metrics['I(Vout;Vinf)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        self.logger.debug("  Pairwise MI computation completed.")
        return metrics

    def _compute_conditional_mi(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
        Vb: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute conditional mutual information given C."""
        self.logger.debug(
            "Computing conditional MI metrics (conditioned on class labels)..."
        )
        metrics = {}

        # I(E;I|C)
        self.logger.debug("  Computing I(E;I|C)...")
        start_time = time.time()
        metrics["I(E;I|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(E, I_var, C.reshape(-1, 1))
        self.logger.debug(
            f"  I(E;I|C) = {metrics['I(E;I|C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(E;Vout|C)
        self.logger.debug("  Computing I(E;Vout|C)...")
        start_time = time.time()
        metrics["I(E;Vout|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(E, Vout, C.reshape(-1, 1))
        self.logger.debug(
            f"  I(E;Vout|C) = {metrics['I(E;Vout|C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # I(I;Vout|C)
        self.logger.debug("  Computing I(I;Vout|C)...")
        start_time = time.time()
        metrics["I(I;Vout|C)"] = self.estimators[
            self.method
        ].conditional_mutual_information(I_var, Vout, C.reshape(-1, 1))
        self.logger.debug(
            f"  I(I;Vout|C) = {metrics['I(I;Vout|C)']:.4f} (took {time.time() - start_time:.1f}s)"
        )

        # Conditional MI with Vb if available
        if Vb is not None:
            # I(E;Vb|C)
            self.logger.debug("  Computing I(E;Vb|C)...")
            start_time = time.time()
            metrics["I(E;Vb|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(E, Vb, C.reshape(-1, 1))
            self.logger.debug(
                f"  I(E;Vb|C) = {metrics['I(E;Vb|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vb|C)
            self.logger.debug("  Computing I(I;Vb|C)...")
            start_time = time.time()
            metrics["I(I;Vb|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(I_var, Vb, C.reshape(-1, 1))
            self.logger.debug(
                f"  I(I;Vb|C) = {metrics['I(I;Vb|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(Vb;Vout|C)
            self.logger.debug("  Computing I(Vb;Vout|C)...")
            start_time = time.time()
            metrics["I(Vb;Vout|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(Vb, Vout, C.reshape(-1, 1))
            self.logger.debug(
                f"  I(Vb;Vout|C) = {metrics['I(Vb;Vout|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        # ------------------------------------------------------------------
        # Ablation-aligned conditional MI proxies
        #
        # These measure *unique* class information that each component provides
        # beyond other components, which is often closer to what layer ablations
        # probe than raw I(E;C) / I(I;C).
        # ------------------------------------------------------------------
        estimator = self.estimators[self.method]
        C_col = C.reshape(-1, 1)

        if self.compute_ablation_aligned_mi and Vb is not None:
            # How much class info is present in the layer output beyond upstream input
            self.logger.debug("  Computing I(Vout;C|Vb)...")
            start_time = time.time()
            metrics["I(Vout;C|Vb)"] = estimator.conditional_mutual_information(
                Vout, C_col, Vb
            )
            self.logger.debug(
                f"  I(Vout;C|Vb) = {metrics['I(Vout;C|Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Complement: how much upstream class information is *not* reflected in the layer output
            # (helps interpret upstream vs synapse contributions).
            self.logger.debug("  Computing I(Vb;C|Vout)...")
            start_time = time.time()
            metrics["I(Vb;C|Vout)"] = estimator.conditional_mutual_information(
                Vb, C_col, Vout
            )
            self.logger.debug(
                f"  I(Vb;C|Vout) = {metrics['I(Vb;C|Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # How much class information in Vout remains after conditioning on synapse inputs
            # (proxy for upstream contribution to class information in Vout).
            self.logger.debug("  Computing I(Vout;C|E,I)...")
            start_time = time.time()
            EI = np.hstack([E, I_var])
            metrics["I(Vout;C|E,I)"] = estimator.conditional_mutual_information(
                Vout, C_col, EI
            )
            self.logger.debug(
                f"  I(Vout;C|E,I) = {metrics['I(Vout;C|E,I)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Optional (can be expensive for high-dimensional synapse sets):
            # unique upstream class information beyond the full synapse set.
            if self.compute_upstream_unique_cmi:
                self.logger.debug("  Computing I(Vb;C|E,I)...")
                start_time = time.time()
                try:
                    metrics["I(Vb;C|E,I)"] = estimator.conditional_mutual_information(
                        Vb, C_col, EI
                    )
                except Exception as e:
                    self.logger.warning(f"  Failed to compute I(Vb;C|E,I): {e}")
                    metrics["I(Vb;C|E,I)"] = 0.0
                self.logger.debug(
                    f"  I(Vb;C|E,I) = {metrics['I(Vb;C|E,I)']:.4f} (took {time.time() - start_time:.1f}s)"
                )

            # ------------------------------------------------------------------
            # Information-processing conditionals (E/I/Vb -> Vout)
            #
            # These quantify *unique* statistical influence on the dendritic output,
            # independent of class labels. They complement ablation results because
            # they isolate each input's contribution to Vout beyond the others.
            # ------------------------------------------------------------------
            self.logger.debug("  Computing I(E;Vout|I,Vb)...")
            start_time = time.time()
            metrics["I(E;Vout|I,Vb)"] = estimator.conditional_mutual_information(
                E, Vout, np.hstack([I_var, Vb])
            )
            self.logger.debug(
                f"  I(E;Vout|I,Vb) = {metrics['I(E;Vout|I,Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(I;Vout|E,Vb)...")
            start_time = time.time()
            metrics["I(I;Vout|E,Vb)"] = estimator.conditional_mutual_information(
                I_var, Vout, np.hstack([E, Vb])
            )
            self.logger.debug(
                f"  I(I;Vout|E,Vb) = {metrics['I(I;Vout|E,Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(Vb;Vout|E,I)...")
            start_time = time.time()
            metrics["I(Vb;Vout|E,I)"] = estimator.conditional_mutual_information(
                Vb, Vout, EI
            )
            self.logger.debug(
                f"  I(Vb;Vout|E,I) = {metrics['I(Vb;Vout|E,I)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Unique class information in E / I beyond upstream input
            self.logger.debug("  Computing I(E;C|Vb)...")
            start_time = time.time()
            metrics["I(E;C|Vb)"] = estimator.conditional_mutual_information(
                E, C_col, Vb
            )
            self.logger.debug(
                f"  I(E;C|Vb) = {metrics['I(E;C|Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(I;C|Vb)...")
            start_time = time.time()
            metrics["I(I;C|Vb)"] = estimator.conditional_mutual_information(
                I_var, C_col, Vb
            )
            self.logger.debug(
                f"  I(I;C|Vb) = {metrics['I(I;C|Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Unique class information in (E,I) beyond upstream input (proxy for "all synapses")
            self.logger.debug("  Computing I(E,I;C|Vb)...")
            start_time = time.time()
            EI = np.hstack([E, I_var])
            metrics["I(E,I;C|Vb)"] = estimator.conditional_mutual_information(
                EI, C_col, Vb
            )
            self.logger.debug(
                f"  I(E,I;C|Vb) = {metrics['I(E,I;C|Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Fine-grained: unique info in E (or I) beyond upstream + the other synapse type
            self.logger.debug("  Computing I(E;C|I,Vb)...")
            start_time = time.time()
            metrics["I(E;C|I,Vb)"] = estimator.conditional_mutual_information(
                E, C_col, np.hstack([I_var, Vb])
            )
            self.logger.debug(
                f"  I(E;C|I,Vb) = {metrics['I(E;C|I,Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(I;C|E,Vb)...")
            start_time = time.time()
            metrics["I(I;C|E,Vb)"] = estimator.conditional_mutual_information(
                I_var, C_col, np.hstack([E, Vb])
            )
            self.logger.debug(
                f"  I(I;C|E,Vb) = {metrics['I(I;C|E,Vb)']:.4f} (took {time.time() - start_time:.1f}s)"
            )
        elif self.compute_ablation_aligned_mi:
            # No upstream branch input (outermost layer): still compute unique info between E and I
            self.logger.debug("  Computing I(E;C|I)...")
            start_time = time.time()
            metrics["I(E;C|I)"] = estimator.conditional_mutual_information(
                E, C_col, I_var
            )
            self.logger.debug(
                f"  I(E;C|I) = {metrics['I(E;C|I)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(I;C|E)...")
            start_time = time.time()
            metrics["I(I;C|E)"] = estimator.conditional_mutual_information(
                I_var, C_col, E
            )
            self.logger.debug(
                f"  I(I;C|E) = {metrics['I(I;C|E)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Information-processing conditionals without upstream input.
            self.logger.debug("  Computing I(E;Vout|I)...")
            start_time = time.time()
            metrics["I(E;Vout|I)"] = estimator.conditional_mutual_information(
                E, Vout, I_var
            )
            self.logger.debug(
                f"  I(E;Vout|I) = {metrics['I(E;Vout|I)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(I;Vout|E)...")
            start_time = time.time()
            metrics["I(I;Vout|E)"] = estimator.conditional_mutual_information(
                I_var, Vout, E
            )
            self.logger.debug(
                f"  I(I;Vout|E) = {metrics['I(I;Vout|E)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            self.logger.debug("  Computing I(Vout;C|E,I)...")
            start_time = time.time()
            EI = np.hstack([E, I_var])
            metrics["I(Vout;C|E,I)"] = estimator.conditional_mutual_information(
                Vout, C_col, EI
            )
            self.logger.debug(
                f"  I(Vout;C|E,I) = {metrics['I(Vout;C|E,I)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

        if Vinf is not None and self.include_vinf:
            # I(E;Vinf|C)
            self.logger.debug("  Computing I(E;Vinf|C)...")
            start_time = time.time()
            metrics["I(E;Vinf|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(E, Vinf, C.reshape(-1, 1))
            self.logger.debug(
                f"  I(E;Vinf|C) = {metrics['I(E;Vinf|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # I(I;Vinf|C)
            self.logger.debug("  Computing I(I;Vinf|C)...")
            start_time = time.time()
            metrics["I(I;Vinf|C)"] = self.estimators[
                self.method
            ].conditional_mutual_information(I_var, Vinf, C.reshape(-1, 1))
            self.logger.debug(
                f"  I(I;Vinf|C) = {metrics['I(I;Vinf|C)']:.4f} (took {time.time() - start_time:.1f}s)"
            )

            # Somatic processing conditionals (low-dimensional and useful for "Vout -> Vinf -> C" chain).
            if self.compute_ablation_aligned_mi:
                self.logger.debug("  Computing I(Vinf;C|Vout)...")
                start_time = time.time()
                metrics["I(Vinf;C|Vout)"] = estimator.conditional_mutual_information(
                    Vinf, C_col, Vout
                )
                self.logger.debug(
                    f"  I(Vinf;C|Vout) = {metrics['I(Vinf;C|Vout)']:.4f} (took {time.time() - start_time:.1f}s)"
                )

                self.logger.debug("  Computing I(Vout;C|Vinf)...")
                start_time = time.time()
                metrics["I(Vout;C|Vinf)"] = estimator.conditional_mutual_information(
                    Vout, C_col, Vinf
                )
                self.logger.debug(
                    f"  I(Vout;C|Vinf) = {metrics['I(Vout;C|Vinf)']:.4f} (took {time.time() - start_time:.1f}s)"
                )

        self.logger.debug("  Conditional MI computation completed.")
        return metrics

    def _gaussian_fisher_proxy(
        self, x: np.ndarray, labels: np.ndarray
    ) -> tuple[float, float]:
        """Return (F, I_proxy) for I(X;C) under a Gaussian/Fisher proxy."""
        eps = float(getattr(self, "gaussian_fisher_eps", 1e-8))
        agg = str(getattr(self, "gaussian_fisher_aggregation", "sum")).strip().lower()
        mi_transform = (
            str(getattr(self, "gaussian_fisher_mi_transform", "half_log1p"))
            .strip()
            .lower()
        )

        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        else:
            x = x.reshape(x.shape[0], -1)

        labels = np.asarray(labels).reshape(-1)
        uniq = np.unique(labels)
        if x.shape[0] <= 2 or uniq.size <= 1:
            return 0.0, 0.0

        if uniq.size == 2:
            c0, c1 = uniq[0], uniq[1]
            m0 = labels == c0
            m1 = labels == c1
            if np.sum(m0) <= 1 or np.sum(m1) <= 1:
                return 0.0, 0.0
            mu0 = np.mean(x[m0], axis=0)
            mu1 = np.mean(x[m1], axis=0)
            var0 = np.var(x[m0], axis=0)
            var1 = np.var(x[m1], axis=0)
            pooled_var = 0.5 * (var0 + var1)
            fisher = (mu1 - mu0) ** 2 / (pooled_var + eps)
        else:
            means = []
            variances = []
            for c in uniq:
                mc = labels == c
                if np.sum(mc) <= 1:
                    continue
                means.append(np.mean(x[mc], axis=0))
                variances.append(np.var(x[mc], axis=0))
            if len(means) <= 1:
                return 0.0, 0.0
            means = np.stack(means, axis=0)
            variances = np.stack(variances, axis=0)
            between_var = np.var(means, axis=0)
            within_var = np.mean(variances, axis=0)
            fisher = between_var / (within_var + eps)

        fisher = np.maximum(0.0, fisher)

        if agg == "mean":
            F = float(np.mean(fisher))
        elif agg == "max":
            F = float(np.max(fisher))
        else:
            F = float(np.sum(fisher))

        if mi_transform == "none":
            I_proxy = F
        elif mi_transform == "log1p":
            I_proxy = float(np.log1p(F))
        else:
            I_proxy = float(0.5 * np.log1p(F))

        return F, I_proxy

    def _fisher_mi_from_f(self, f: float) -> float:
        """Map a Fisher/discriminability-like scalar to an MI-like proxy."""
        mi_transform = (
            str(getattr(self, "gaussian_fisher_mi_transform", "half_log1p"))
            .strip()
            .lower()
        )

        f = float(f)
        if not np.isfinite(f) or f <= 0.0:
            return 0.0

        if mi_transform == "none":
            return f
        if mi_transform == "log1p":
            return float(np.log1p(f))
        return float(0.5 * np.log1p(f))

    def _fisher_dprime2_binary(self, x: np.ndarray, labels: np.ndarray) -> float:
        """Binary-class Fisher discriminability: d'^2 = Δμᵀ Σ^{-1} Δμ (ridge-regularized)."""
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        else:
            x = x.reshape(x.shape[0], -1)

        labels = np.asarray(labels).reshape(-1)
        uniq = np.unique(labels)
        if uniq.size != 2 or x.shape[0] <= 2:
            return 0.0

        c0, c1 = uniq[0], uniq[1]
        m0 = labels == c0
        m1 = labels == c1
        if np.sum(m0) <= 1 or np.sum(m1) <= 1:
            return 0.0

        x0 = x[m0]
        x1 = x[m1]
        mu0 = np.mean(x0, axis=0)
        mu1 = np.mean(x1, axis=0)
        dmu = mu1 - mu0

        cov0 = np.atleast_2d(np.cov(x0, rowvar=False))
        cov1 = np.atleast_2d(np.cov(x1, rowvar=False))
        pooled = 0.5 * (cov0 + cov1)

        eps = float(getattr(self, "gaussian_fisher_eps", 1e-8))
        pooled = pooled + eps * np.eye(pooled.shape[0])

        try:
            sol = np.linalg.solve(pooled, dmu)
        except np.linalg.LinAlgError:
            sol = np.linalg.lstsq(pooled, dmu, rcond=None)[0]

        d2 = float(dmu @ sol)
        if not np.isfinite(d2):
            return 0.0
        return float(max(0.0, d2))

    def _fisher_dprime2_pairwise_avg(self, x: np.ndarray, labels: np.ndarray) -> float:
        """Multi-class surrogate: average binary d'^2 over all class pairs."""
        x = np.asarray(x)
        if x.ndim == 1:
            x = x[:, None]
        else:
            x = x.reshape(x.shape[0], -1)

        labels = np.asarray(labels).reshape(-1)
        uniq = np.unique(labels)
        if uniq.size <= 1 or x.shape[0] <= 2:
            return 0.0
        if uniq.size == 2:
            return self._fisher_dprime2_binary(x, labels)

        d2s: list[float] = []
        for i in range(uniq.size):
            for j in range(i + 1, uniq.size):
                a, b = uniq[i], uniq[j]
                m = (labels == a) | (labels == b)
                if np.sum(m) <= 2:
                    continue
                d2 = self._fisher_dprime2_binary(x[m], labels[m])
                if np.isfinite(d2):
                    d2s.append(float(d2))

        if not d2s:
            return 0.0
        return float(np.mean(d2s))

    def _compute_gaussian_fisher(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
        Vb: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute Gaussian/Fisher proxy metrics for available variables."""
        metrics: dict[str, float] = {}

        for name, X in [
            ("E", E),
            ("I", I_var),
            ("Vb", Vb),
            ("Vout", Vout),
            ("Vinf", Vinf if (Vinf is not None and self.include_vinf) else None),
        ]:
            if X is None:
                continue
            F, I_proxy = self._gaussian_fisher_proxy(X, C)
            # Use an I(.;.) prefix so LDA/shuffled variants get token-renamed.
            metrics[f"I({name};C)_fisher_F"] = float(F)
            metrics[f"I({name};C)_fisher"] = float(I_proxy)

        # Information-flow-style increment from upstream branches (Vb) beyond local synapses (E,I).
        # This mirrors the Schur-complement increment in the theory draft (Sec. 'Multi-branch Trees
        # and Information Flow') but uses a Fisher/geometry surrogate (d'^2) computed from moments.
        if Vb is not None:
            try:
                EI = np.hstack([E, I_var])
                EIVb = np.hstack([E, I_var, Vb])
                d2_full = self._fisher_dprime2_pairwise_avg(EIVb, C)
                d2_base = self._fisher_dprime2_pairwise_avg(EI, C)
                inc = float(max(0.0, d2_full - d2_base))
                metrics["I(Vb;C|E,I)_fisher_F"] = inc
                metrics["I(Vb;C|E,I)_fisher"] = float(self._fisher_mi_from_f(inc))
            except Exception as e:
                self.logger.warning(
                    f"Failed to compute Fisher flow increment I(Vb;C|E,I)_fisher: {e}"
                )
                metrics["I(Vb;C|E,I)_fisher_F"] = 0.0
                metrics["I(Vb;C|E,I)_fisher"] = 0.0

        return metrics

    def _compute_pid(
        self, E: np.ndarray, I_var: np.ndarray, Vout: np.ndarray, C: np.ndarray
    ) -> dict[str, Any]:
        """Compute Partial Information Decomposition."""
        self.logger.info("Computing Partial Information Decomposition (PID)...")
        pid_results = {}

        # Prepare data for PID (may need binarization)
        self.logger.info(f"  Data preprocessing: binarize={self.pid_binarize}")
        if self.pid_binarize:
            self.logger.info(
                f"  Binarizing data using method: {self.pid_binarize_method}"
            )
            start_time = time.time()
            E_pid = self._binarize(E)
            I_pid = self._binarize(I_var)
            Vout_pid = self._binarize(Vout)
            C_pid = self._binarize(C.reshape(-1, 1))
            self.logger.info(
                f"  Data binarization completed (took {time.time() - start_time:.1f}s)"
            )
        else:
            E_pid = E
            I_pid = I_var
            Vout_pid = Vout
            C_pid = C.reshape(-1, 1)

        def _pid_helper(
            source1_array: np.ndarray,
            source1_name: str,
            source2_array: np.ndarray,
            source2_name: str,
            target_array: np.ndarray,
            target_name: str,
        ) -> None:
            """Helper function to compute PID."""
            pid_key = f"PID({source1_name},{source2_name}->{target_name})"
            self.logger.info(f"  Computing {pid_key}...")
            start_time = time.time()
            try:
                synergy, redundancy = pid_synergy_redundancy(
                    [source1_array, source2_array], target_array
                )

                # Compute unique information
                unique_1 = (
                    self.estimators[self.method].mutual_information(
                        source1_array, target_array
                    )
                    - redundancy
                )
                unique_2 = (
                    self.estimators[self.method].mutual_information(
                        source2_array, target_array
                    )
                    - redundancy
                )

                pid_results[pid_key] = {
                    "synergy": synergy,
                    "redundancy": redundancy,
                    f"unique_{source1_name}": unique_1,
                    f"unique_{source2_name}": unique_2,
                }
                self.logger.info(
                    f"  {pid_key}: synergy={synergy:.4f}, redundancy={redundancy:.4f}, unique_{source1_name}={unique_1:.4f}, unique_{source2_name}={unique_2:.4f} (took {time.time() - start_time:.1f}s)"
                )
            except Exception as e:
                self.logger.warning(
                    f"  {pid_key} computation failed: {e} (took {time.time() - start_time:.1f}s)"
                )

        # PID: (E, I) -> C
        _pid_helper(E_pid, "E", I_pid, "I", C_pid, "C")
        # PID: (E, Vout) -> C
        _pid_helper(E_pid, "E", Vout_pid, "Vout", C_pid, "C")
        # PID: (I, Vout) -> C
        _pid_helper(I_pid, "I", Vout_pid, "Vout", C_pid, "C")
        # PID: (E, I) -> Vout
        _pid_helper(E_pid, "E", I_pid, "I", Vout_pid, "Vout")

        self.logger.info("  PID computation completed.")
        return pid_results

    def _binarize(self, data: np.ndarray) -> np.ndarray:
        """Binarize data for PID computation."""
        if self.pid_binarize_method == "median":
            threshold = np.median(data, axis=0)
        elif self.pid_binarize_method == "mean":
            threshold = np.mean(data, axis=0)
        elif self.pid_binarize_method == "quantile":
            threshold = np.quantile(data, self.pid_binarize_threshold, axis=0)
        else:
            raise ValueError(f"Unknown binarization method: {self.pid_binarize_method}")

        binarized: np.ndarray = data > threshold
        return binarized.astype(int)

    def _validate_inputs(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
    ):
        """Validate input arrays."""
        n_samples = len(E)

        if len(I_var) != n_samples:
            raise ValueError(
                f"I must have same number of samples as E: {len(I_var)} vs {n_samples}"
            )
        if len(Vout) != n_samples:
            raise ValueError(
                f"Vout must have same number of samples as E: {len(Vout)} vs {n_samples}"
            )
        if len(C) != n_samples:
            raise ValueError(
                f"C must have same number of samples as E: {len(C)} vs {n_samples}"
            )
        if Vinf is not None and len(Vinf) != n_samples:
            raise ValueError(
                f"Vinf must have same number of samples as E: {len(Vinf)} vs {n_samples}"
            )

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D."""
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def get_summary(self, results: dict[str, dict[str, Union[dict, float]]]) -> str:
        """Generate a summary string of the results."""
        summary_lines = [
            "Information Analysis Summary",
            f"Method: {results['method']}",
            f"Samples: {results['n_samples']}",
            "",
        ]

        if "basic_mi" in results:
            summary_lines.append("Basic MI with C:")
            for key, value in results["basic_mi"].items():
                if not key.endswith("_normalized"):
                    summary_lines.append(f"  {key}: {value:.4f}")

        if "pairwise_mi" in results:
            summary_lines.append("\nPairwise MI:")
            for key, value in results["pairwise_mi"].items():
                summary_lines.append(f"  {key}: {value:.4f}")

        if "conditional_mi" in results:
            summary_lines.append("\nConditional MI given C:")
            for key, value in results["conditional_mi"].items():
                summary_lines.append(f"  {key}: {value:.4f}")

        if "gaussian_fisher" in results:
            summary_lines.append("\nGaussian/Fisher proxy:")
            for key, value in results["gaussian_fisher"].items():
                summary_lines.append(f"  {key}: {value:.4f}")

        if "pid" in results:
            summary_lines.append("\nPID Results:")
            for pid_key, pid_values in results["pid"].items():
                summary_lines.append(f"  {pid_key}:")
                for metric, value in pid_values.items():
                    summary_lines.append(f"    {metric}: {value:.4f}")

        return "\n".join(summary_lines)

    def compute_general_class_mi(
        self,
        activations: np.ndarray,
        labels: np.ndarray,
        dims: Optional[list[int]] = None,
    ) -> dict:
        """Compute MI between activations (optionally specific dims) and class labels using all methods."""
        results = {}
        acts = activations if dims is None else activations[:, dims]
        for method, estimator in self.estimators.items():
            mi = estimator.mutual_information(acts, labels.reshape(-1, 1))
            results[f"{method}_mi_class"] = mi
            if self.normalize_mi:
                h_labels = estimator.entropy(labels.reshape(-1, 1))
                results[f"{method}_normalized_mi_class"] = (
                    mi / h_labels if h_labels > 0 else 0
                )
        return results

    def compute_metrics(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
        Vb: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Compute all requested information metrics.

        Parameters
        ----------
        E : np.ndarray
            Excitatory inputs, shape (n_samples,) or (n_samples, n_features)
        I_var : np.ndarray
            Inhibitory inputs, shape (n_samples,) or (n_samples, n_features)
        Vout : np.ndarray
            Branch outputs, shape (n_samples,) or (n_samples, n_features)
        C : np.ndarray
            Class labels, shape (n_samples,)
        Vinf : np.ndarray, optional
            Soma activation (Vinf), shape (n_samples,) or (n_samples, n_features)
        Vb : np.ndarray, optional
            Branch input (upstream dendritic input), shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        dict[str, Any]
            Dictionary containing all computed metrics
        """
        # Validate inputs
        self._validate_inputs(E, I_var, Vout, C, Vinf)

        # Ensure all arrays are properly shaped
        E = self._ensure_2d(E)
        I_var = self._ensure_2d(I_var)
        Vout = self._ensure_2d(Vout)
        C = C.ravel()  # Labels are always 1D

        if Vinf is not None:
            Vinf = self._ensure_2d(Vinf)

        if Vb is not None:
            Vb = self._ensure_2d(Vb)

        results = {
            "method": self.method,
            "n_samples": len(E),
        }

        # 1. Basic mutual information with C
        if self.compute_basic_mi:
            self.logger.info("Computing basic MI metrics...")
            results["basic_mi"] = self._compute_basic_mi(E, I_var, Vout, C, Vinf, Vb)

        # 2. Pairwise mutual information
        if self.compute_pairwise_mi:
            self.logger.info("Computing pairwise MI metrics...")
            results["pairwise_mi"] = self._compute_pairwise_mi(E, I_var, Vout, Vinf, Vb)

        # 3. Conditional mutual information
        if self.compute_conditional_mi:
            self.logger.info("Computing conditional MI metrics...")
            results["conditional_mi"] = self._compute_conditional_mi(
                E, I_var, Vout, C, Vinf, Vb
            )

        # 4. Gaussian/Fisher MI proxy
        if self.compute_gaussian_fisher:
            self.logger.info("Computing Gaussian/Fisher proxy metrics...")
            results["gaussian_fisher"] = self._compute_gaussian_fisher(
                E, I_var, Vout, C, Vinf, Vb
            )

        # 4. PID decomposition
        if self.compute_pid and self.pid_available:
            self.logger.info("Computing PID metrics...")
            results["pid"] = self._compute_pid(E, I_var, Vout, C)

        return results

    def _average_mi_results(
        self, results_list: list[dict[str, dict[str, Union[dict, float]]]]
    ):
        """Average MI results across multiple branches."""
        if not results_list:
            return {"method": self.method, "n_samples": 0}

        averaged_results: dict[str, Any] = {
            "method": results_list[0].get("method", self.method),
            "n_samples": results_list[0].get("n_samples", 0),
        }

        for metric_category in (
            "basic_mi",
            "pairwise_mi",
            "conditional_mi",
            "gaussian_fisher",
        ):
            # Collect all keys across results (some branches/layers may omit metrics
            # when the underlying variable does not exist, e.g. no synapses or no Vb).
            all_keys: set[str] = set()
            for r in results_list:
                if metric_category in r and isinstance(r[metric_category], dict):
                    all_keys.update(r[metric_category].keys())

            if not all_keys:
                continue

            averaged_category: dict[str, float] = {}
            for key in all_keys:
                values: list[float] = []
                for r in results_list:
                    if (
                        metric_category in r
                        and isinstance(r[metric_category], dict)
                        and key in r[metric_category]
                    ):
                        val = r[metric_category][key]
                        if isinstance(val, (int, float, np.number)) and np.isfinite(
                            val
                        ):
                            values.append(float(val))
                if values:
                    averaged_category[key] = float(np.mean(values))
                    averaged_category[f"{key}_std"] = float(np.std(values))

            if averaged_category:
                averaged_results[metric_category] = averaged_category

        # Average PID results (nested dicts)
        if any("pid" in r for r in results_list):
            pid_keys: set[str] = set()
            for r in results_list:
                if "pid" in r and isinstance(r["pid"], dict):
                    pid_keys.update(r["pid"].keys())

            averaged_pid: dict[str, dict[str, float]] = {}
            for pid_key in pid_keys:
                # Collect metric keys under each pid_key
                metric_keys: set[str] = set()
                for r in results_list:
                    if (
                        "pid" in r
                        and isinstance(r["pid"], dict)
                        and pid_key in r["pid"]
                        and isinstance(r["pid"][pid_key], dict)
                    ):
                        metric_keys.update(r["pid"][pid_key].keys())

                if not metric_keys:
                    continue

                averaged_pid[pid_key] = {}
                for metric in metric_keys:
                    values: list[float] = []
                    for r in results_list:
                        if (
                            "pid" in r
                            and isinstance(r["pid"], dict)
                            and pid_key in r["pid"]
                            and isinstance(r["pid"][pid_key], dict)
                            and metric in r["pid"][pid_key]
                        ):
                            val = r["pid"][pid_key][metric]
                            if isinstance(val, (int, float, np.number)) and np.isfinite(
                                val
                            ):
                                values.append(float(val))
                    if values:
                        averaged_pid[pid_key][metric] = float(np.mean(values))
                        averaged_pid[pid_key][f"{metric}_std"] = float(np.std(values))

            if averaged_pid:
                averaged_results["pid"] = averaged_pid

        return averaged_results

    @staticmethod
    def _is_discrete_class_labels(labels: np.ndarray) -> bool:
        """Heuristic: decide whether labels are discrete class IDs."""
        labels = np.asarray(labels).ravel()
        if labels.size == 0:
            return False
        unique = np.unique(labels)
        # Typical classification: small number of unique values
        if unique.size <= 50:
            return True
        # If the labels are integer-like and the number of unique labels is still smallish
        if unique.size <= 200 and np.allclose(unique, np.round(unique), atol=1e-6):
            return True
        return False

    @staticmethod
    def _discrete_entropy_nats(labels: np.ndarray) -> float:
        """Discrete entropy H(C) in natural units (nats)."""
        labels = np.asarray(labels).ravel()
        if labels.size == 0:
            return 0.0
        _, counts = np.unique(labels, return_counts=True)
        p = counts.astype(float) / float(counts.sum())
        # Natural log: matches the Kraskov MI units (nats)
        return float(-np.sum(p * np.log(p + 1e-12)))

    @staticmethod
    def _metric_targets_class(metric_key: str) -> bool:
        """Return True if the metric is of the form I(X;C) or I(X;C|...)."""
        if (
            not isinstance(metric_key, str)
            or "I(" not in metric_key
            or ")" not in metric_key
        ):
            return False
        lpar = metric_key.find("I(")
        lpar = metric_key.find("(", lpar)
        rpar = metric_key.find(")", lpar)
        if lpar == -1 or rpar == -1 or rpar <= lpar + 1:
            return False
        inside = metric_key[lpar + 1 : rpar]
        main = inside.split("|", 1)[0]
        parts = [p.strip() for p in main.split(";")]
        if len(parts) < 2:
            return False
        return parts[1] == "C"

    def _compute_layer_statistics(
        self,
        results_list: list[dict[str, Any]],
        *,
        entropy_C: Optional[float] = None,
        topk: int = 10,
    ) -> dict[str, float]:
        """Compute per-layer summary stats from per-branch MI results.

        Notes
        -----
        Raw per-branch MI does *not* uniquely determine the true layer-level
        multivariate MI (redundancy/synergy matter). To help relate single-branch
        results to layer ablations, we provide additional *heuristic* branch
        aggregations for class-targeted metrics:

        - `<metric>_sum`: sum across branches (scales with number of branches)
        - `<metric>_sum_clipped`: min(sum, H(C)) for class-targeted metrics
        - `<metric>_union`: saturating "independent evidence" union:
              H(C) * (1 - Π_b (1 - I_b/H(C)))
        - `<metric>_topk_sum`: sum of top-K branches (K configurable)
        - `<metric>_max`: max branch value
        """
        layer_stats: dict[str, float] = {}

        if not results_list:
            return layer_stats

        # Handle nested structure - collect ALL unique keys across all branches
        for metric_category in ["basic_mi", "pairwise_mi", "conditional_mi"]:
            all_keys: set[str] = set()
            for r in results_list:
                if metric_category in r and isinstance(r[metric_category], dict):
                    all_keys.update(r[metric_category].keys())

            for key in all_keys:
                values: list[float] = []
                for r in results_list:
                    if (
                        metric_category in r
                        and isinstance(r[metric_category], dict)
                        and key in r[metric_category]
                    ):
                        val = r[metric_category][key]
                        if isinstance(val, (int, float, np.number)) and np.isfinite(
                            val
                        ):
                            values.append(float(val))

                if not values:
                    continue

                arr = np.asarray(values, dtype=float)
                layer_stats[f"{key}_mean"] = float(np.mean(arr))

                # Default std: across branches (between-branch variability).
                std_val = float(np.std(arr))

                # For permutation-null baselines, prefer the within-shuffle spread.
                # The baseline is stored as:
                #   <metric>_null      : mean across permutations (per-branch)
                #   <metric>_null_std  : std across permutations (per-branch)
                # For plotting, it is more informative for <metric>_null error bars
                # to reflect <metric>_null_std rather than branch-to-branch variability
                # of the null mean.
                if isinstance(key, str) and key.endswith("_null"):
                    null_std_key = f"{key}_std"
                    null_stds: list[float] = []
                    for r in results_list:
                        if (
                            metric_category in r
                            and isinstance(r[metric_category], dict)
                            and null_std_key in r[metric_category]
                        ):
                            v = r[metric_category][null_std_key]
                            if isinstance(v, (int, float, np.number)) and np.isfinite(
                                v
                            ):
                                null_stds.append(float(v))
                    if null_stds:
                        std_val = float(
                            np.sqrt(np.mean(np.asarray(null_stds, dtype=float) ** 2))
                        )

                layer_stats[f"{key}_std"] = std_val
                layer_stats[f"{key}_n"] = float(arr.size)

                # Only compute "layer total" heuristics for class-targeted metrics.
                if (
                    entropy_C is None
                    or entropy_C <= 0
                    or not self._metric_targets_class(key)
                ):
                    continue

                n = float(arr.size)
                base_std = float(layer_stats.get(f"{key}_std", float(np.std(arr))))

                s = float(np.sum(arr))
                layer_stats[f"{key}_sum_mean"] = s
                # Approximate spread of the summed proxy using branch-level variability.
                # If branch values were independent draws, std(sum) ≈ sqrt(n) * std(branch).
                layer_stats[f"{key}_sum_std"] = float(np.sqrt(n) * base_std)

                # Clip at H(C) (a valid upper bound for I(X;C) and I(X;C|Z))
                layer_stats[f"{key}_sum_clipped_mean"] = float(min(s, entropy_C))
                layer_stats[f"{key}_sum_clipped_std"] = (
                    layer_stats[f"{key}_sum_std"] if s < entropy_C else 0.0
                )

                # Saturating union-of-independent-evidence heuristic (bounded by H(C))
                frac = np.clip(arr / entropy_C, 0.0, 1.0)
                one_minus = 1.0 - frac
                prod_one_minus = float(np.prod(one_minus))
                union_frac = 1.0 - prod_one_minus
                layer_stats[f"{key}_union_mean"] = float(entropy_C * union_frac)

                # Delta-method style spread estimate for the union proxy using branch variability.
                # union = H(C) * (1 - Π(1 - I_b/H(C)))  =>  d union / d I_i = Π_{j!=i}(1 - frac_j)
                # Approximate Var(union) ≈ std(branch)^2 * Σ_i (d union / d I_i)^2
                if base_std > 0 and arr.size > 0:
                    g = np.zeros_like(one_minus)
                    mask = (frac > 0.0) & (frac < 1.0) & (one_minus > 0.0)
                    if np.any(mask):
                        g[mask] = prod_one_minus / one_minus[mask]
                        union_std = float(base_std * np.sqrt(np.sum(g**2)))
                        layer_stats[f"{key}_union_std"] = float(
                            min(max(union_std, 0.0), entropy_C)
                        )
                    else:
                        layer_stats[f"{key}_union_std"] = 0.0
                else:
                    layer_stats[f"{key}_union_std"] = 0.0

                # Concentration diagnostics
                layer_stats[f"{key}_max_mean"] = float(np.max(arr))
                # Provide a simple scale for max/topK proxies as well (helps plotting).
                layer_stats[f"{key}_max_std"] = base_std

                k = int(max(1, topk))
                k = min(k, arr.size)
                topk_sum = float(np.sum(np.sort(arr)[-k:]))
                layer_stats[f"{key}_top{topk}_sum_mean"] = topk_sum
                layer_stats[f"{key}_top{topk}_sum_std"] = float(np.sqrt(k) * base_std)

        return layer_stats

    @staticmethod
    def _extract_metric_tokens(metric_key: str) -> list[str]:
        """Extract variable tokens from metric keys like 'I(E;Vb|C)'.

        Returns an empty list for non-metric keys.
        """
        if not isinstance(metric_key, str):
            return []
        if "I(" not in metric_key or ")" not in metric_key:
            return []

        lpar = metric_key.find("I(")
        lpar = metric_key.find("(", lpar)
        rpar = metric_key.find(")", lpar)
        if lpar == -1 or rpar == -1 or rpar <= lpar + 1:
            return []

        inside = metric_key[lpar + 1 : rpar]
        inside = inside.replace("|", ";").replace(",", ";")
        tokens = [t.strip() for t in inside.split(";") if t.strip()]
        return tokens

    @staticmethod
    def _rename_metric_key_tokens(metric_key: str, token_map: dict[str, str]) -> str:
        """Rename variable tokens inside MI keys.

        This is used to generate LDA/shuffled metric names (e.g. E -> E_lin) while
        preserving separators, and it works for tokens in both the main and
        conditioning parts (after '|').
        """
        if not isinstance(metric_key, str):
            return metric_key
        if "I(" not in metric_key:
            return metric_key

        renamed = metric_key
        for src, dst in token_map.items():
            # Replace full-token occurrences inside the I( ... ) block.
            # Tokens are delimited by one of: (, ;, ,, |, )
            renamed = re.sub(
                rf"(?<=[(;,|]){re.escape(src)}(?=[,;|)])",
                dst,
                renamed,
            )
        return renamed

    def _keep_metric_key_given_availability(
        self,
        metric_key: str,
        *,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
    ) -> bool:
        """Return True if the metric is meaningful given available variables.

        We drop any metric that references:
        - E / E_* when there are no excitatory synapses
        - I / I_* when there are no inhibitory synapses
        - Vb / Vb_* when branch input does not exist (e.g., outermost distal layer)
        """
        tokens = self._extract_metric_tokens(metric_key)
        if not tokens:
            return True

        if (not has_exc_synapses) and any(t.startswith("E") for t in tokens):
            return False
        if (not has_inh_synapses) and any(t.startswith("I") for t in tokens):
            return False
        if (not has_branch_input) and any(t.startswith("Vb") for t in tokens):
            return False

        return True

    def _filter_result_metrics_in_place(
        self,
        result: dict[str, Any],
        *,
        has_exc_synapses: bool,
        has_inh_synapses: bool,
        has_branch_input: bool,
    ) -> None:
        """Remove metrics that depend on non-existent variables (in-place)."""
        for metric_category in (
            "basic_mi",
            "pairwise_mi",
            "conditional_mi",
            "gaussian_fisher",
        ):
            if metric_category not in result or not isinstance(
                result[metric_category], dict
            ):
                continue
            result[metric_category] = {
                k: v
                for k, v in result[metric_category].items()
                if self._keep_metric_key_given_availability(
                    k,
                    has_exc_synapses=has_exc_synapses,
                    has_inh_synapses=has_inh_synapses,
                    has_branch_input=has_branch_input,
                )
            }

    @staticmethod
    def _should_convert_information_key_to_bits(key: Any) -> bool:
        """Heuristic: return True if `key` is an MI/CMI-like metric name in nats.

        We convert keys that contain an MI-style token (e.g. "I(E;C)") and are
        *not* clearly dimensionless or in non-MI units.
        """
        if not isinstance(key, str):
            return False
        if "I(" not in key:
            return False
        # Dimensionless ratios; do not convert.
        if "_normalized" in key:
            return False
        # Fisher discriminability proxy (d'^2-like); not an information unit.
        if "_fisher_F" in key:
            return False
        # Counts used for proxy bookkeeping.
        if key.endswith("_n"):
            return False
        return True

    def _convert_information_units_in_place(self, obj: Any) -> None:
        """Convert MI/CMI-like quantities in-place from nats -> bits.

        Estimators operate in nats internally. Many plots and summaries are in bits,
        so we convert any MI/CMI-like metrics (including *_std and *_null variants)
        before saving/plotting when `compute.output_units="bits"`.
        """
        if getattr(self, "output_units", "bits") != "bits":
            return

        nat_to_bit = 1.0 / float(np.log(2.0))
        nat_to_bit_sq = nat_to_bit * nat_to_bit

        def _convert_any(x: Any) -> None:
            if isinstance(x, dict):
                for k in list(x.keys()):
                    v = x.get(k)
                    if isinstance(v, (dict, list)):
                        _convert_any(v)
                    elif isinstance(v, (int, float, np.floating)):
                        if not self._should_convert_information_key_to_bits(k):
                            continue
                        scale = (
                            nat_to_bit_sq
                            if isinstance(k, str) and k.endswith(("_var", "_variance"))
                            else nat_to_bit
                        )
                        x[k] = float(v) * scale
            elif isinstance(x, list):
                for item in x:
                    _convert_any(item)

        _convert_any(obj)

    def _add_label_shuffle_null_baselines_in_place(
        self,
        result: dict[str, Any],
        *,
        E: Optional[np.ndarray],
        I_var: Optional[np.ndarray],
        Vout: Optional[np.ndarray],
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
        Vb: Optional[np.ndarray] = None,
        has_exc_synapses: bool = True,
        has_inh_synapses: bool = True,
        has_branch_input: bool = True,
        seed_offset: int = 0,
    ) -> None:
        r"""Add permutation (*_null) baselines for selected MI/CMI metrics (in-place).

        This is a finite-sample bias / null diagnostic: it permutes *one of the
        variables* across samples and recomputes MI/CMI using the *same* estimator
        selected by ``estimator.method``.

        Notes
        -----
        - For ``I(X;Y)`` we use ``estimator.mutual_information_null(X, Y)`` which
          permutes ``Y`` across samples.
        - For ``I(X;Y|Z)`` we permute ``Y`` and recompute
          ``estimator.conditional_mutual_information(X, Y_perm, Z)``.
          If the conditioning variable is exactly ``Z=C`` (discrete labels),
          we permute ``Y`` \emph{within each class} to preserve the marginal
          distribution of ``Y|C`` while destroying within-class dependence.
        """
        if int(getattr(self, "mi_null_shuffles", 0) or 0) <= 0:
            return

        null_components = getattr(self, "mi_null_components", {"basic"})
        want_all = "all" in null_components
        want_basic = want_all or ("basic" in null_components)
        want_pairwise = want_all or ("pairwise" in null_components)
        want_conditional = want_all or bool(
            {"conditional", "ablation_aligned", "upstream_unique_cmi"}.intersection(
                null_components
            )
        )
        categories: list[str] = []
        if want_basic:
            categories.append("basic_mi")
        if want_pairwise:
            categories.append("pairwise_mi")
        if want_conditional:
            categories.append("conditional_mi")
        if not categories:
            return

        estimator = self.estimators[self.method]
        n_shuf = int(self.mi_null_shuffles)
        seed_base = int(getattr(self, "mi_null_seed", 0) or 0) + int(seed_offset)
        C_vec = np.asarray(C).ravel()
        C_col = C_vec.reshape(-1, 1)

        def _parse_mi_key(mi_key: Any) -> Any:
            if not isinstance(mi_key, str):
                return None
            key = mi_key.strip().replace(" ", "")
            if not (key.startswith("I(") and key.endswith(")")):
                return None
            inside = key[2:-1]
            if ";" not in inside:
                return None
            left, right = inside.split(";", 1)
            if "|" in right:
                y, z = right.split("|", 1)
                z_vars = [t for t in z.split(",") if t]
            else:
                y = right
                z_vars = []
            x_vars = [t for t in left.split(",") if t]
            return x_vars, y, z_vars

        def _var_data(token: str) -> Optional[np.ndarray]:
            base = token.split("_", 1)[0]
            if base == "C":
                return C_col
            if base == "E":
                return E if has_exc_synapses else None
            if base == "I":
                return I_var if has_inh_synapses else None
            if base == "Vb":
                return Vb if has_branch_input else None
            if base == "Vout":
                return Vout
            if base == "Vinf":
                return Vinf
            return None

        perms: Optional[list[np.ndarray]] = None
        seed_off = 0
        for category in categories:
            if category not in result or not isinstance(result[category], dict):
                continue
            metrics_dict: dict[str, Any] = result[category]
            for metric_key in list(metrics_dict.keys()):
                if not isinstance(metric_key, str):
                    continue
                if metric_key.endswith(("_null", "_std", "_normalized")):
                    continue
                if f"{metric_key}_null" in metrics_dict:
                    continue

                parsed = _parse_mi_key(metric_key)
                if parsed is None:
                    continue
                x_vars, y_var, z_vars = parsed

                x_parts = [_var_data(t) for t in x_vars]
                if any(p is None for p in x_parts):
                    continue
                x = np.hstack([p for p in x_parts if p is not None])

                y_arr = _var_data(y_var)
                if y_arr is None:
                    continue

                # Unconditional: I(X;Y)
                if not z_vars:
                    null_mean, null_std = estimator.mutual_information_null(
                        x,
                        y_arr,
                        n_shuffles=n_shuf,
                        seed=seed_base + seed_off,
                    )
                    metrics_dict[f"{metric_key}_null"] = float(null_mean)
                    metrics_dict[f"{metric_key}_null_std"] = float(null_std)
                    seed_off += 1
                    continue

                # Conditional: I(X;Y|Z)
                z_parts = [_var_data(t) for t in z_vars]
                if any(p is None for p in z_parts):
                    continue
                z = np.hstack([p for p in z_parts if p is not None])

                stratify_by_c = len(z_vars) == 1 and z_vars[0].split("_", 1)[0] == "C"
                if perms is None:
                    rng = np.random.default_rng(seed_base + 10_000)
                    perms = [rng.permutation(C_vec.shape[0]) for _ in range(n_shuf)]

                vals: list[float] = []
                for perm in perms:
                    if stratify_by_c:
                        # Permute Y within each class label to preserve Y|C.
                        y_perm = np.asarray(y_arr).copy()
                        for c_val in np.unique(C_vec):
                            idx = np.flatnonzero(C_vec == c_val)
                            if idx.size <= 1:
                                continue
                            idx_perm = rng.permutation(idx)
                            y_perm[idx] = y_arr[idx_perm]
                    else:
                        y_perm = y_arr[perm]
                    vals.append(
                        float(estimator.conditional_mutual_information(x, y_perm, z))
                    )
                metrics_dict[f"{metric_key}_null"] = float(np.mean(vals))
                metrics_dict[f"{metric_key}_null_std"] = float(np.std(vals))

    def _compute_variance_analysis(self, results: dict) -> dict:
        """
        Compute variance analysis across different hierarchical levels.

        This computes how much information values vary:
        - When per_neuron_analysis=True: variance across branches within neurons
        - When per_layer_analysis=True: variance across neurons within layers
        - When per_einet_analysis=True: variance across E/I networks

        Returns variance statistics that can be plotted alongside means.
        """
        variance_analysis = {}

        # Compute variance based on computation level and analysis modes
        if "layer_statistics" in results:
            layer_stats = results["layer_statistics"]

            # Extract variance information from the existing std values
            # The std values in layer_statistics represent variance across branches
            variance_by_layer = {}

            for layer_name, layer_data in layer_stats.items():
                layer_variances = {}

                # Extract variance from std values (std^2 = variance)
                for metric_key, value in layer_data.items():
                    if metric_key.endswith("_std"):
                        metric_base = metric_key.replace("_std", "")
                        variance_key = f"{metric_base}_variance"
                        layer_variances[variance_key] = value**2  # std^2 = variance

                # Always include layer in variance analysis (even if all variances are zero)
                # This ensures synthetic_soma is included when somatic_synapses=False
                variance_by_layer[layer_name] = layer_variances

            variance_analysis["layer_variances"] = variance_by_layer

        # Add metadata about what type of variance this represents
        variance_analysis["variance_interpretation"] = {
            "layer_variances": "Variance across branches within each layer",
            "computation_level": results.get("computation_level", "unknown"),
            "per_neuron_enabled": results.get("per_neuron_analysis_enabled", False),
            "per_layer_enabled": results.get("per_layer_analysis_enabled", False),
            "per_einet_enabled": results.get("per_einet_analysis_enabled", False),
        }

        return variance_analysis

    def attach_forward_hooks(self, model: BaseModel):
        """Attach forward hooks to the model."""
        # Initialize data_dict attribute and store somatic_synapses config
        self.data_dict = {}
        self.somatic_synapses_enabled = getattr(
            model.core_network, "somatic_synapses", True
        )

        handles = []
        for _name, module in model.named_modules():
            if isinstance(module, DendriticBranchLayer):
                # Hook layers that have E/I synapses OR receive branch input
                # This ensures soma layers without synapses are still analyzed
                # for I(Vb;C) - the information in branch input
                has_synapses = (
                    module.branch_excitation is not None
                    or module.branch_inhibition is not None
                )
                has_branch_input = module.input_branches  # Has upstream branches

                if has_synapses or has_branch_input:
                    self.data_dict[_name] = {}
                    module._name = _name
                    handle = module.register_forward_hook(self.forward_hook)
                    handles.append(handle)

        return handles

    def forward_hook(
        self,
        module: DendriticBranchLayer,
        input: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        output: torch.Tensor,
    ):
        """Forward hook to collect branch data."""
        excitatory_input, inhibitory_input, branch_input = input

        # A layer can legitimately have only E synapses or only I synapses.
        # "No synapses" means BOTH synapse modules are absent (e.g. soma when
        # somatic_synapses=False).
        has_no_synapses = (
            module.branch_excitation is None and module.branch_inhibition is None
        )

        if has_no_synapses:
            # For soma without synapses, create zero tensors for E/I but keep real output
            batch_size = output.shape[0]
            n_branches = output.shape[1] if len(output.shape) > 1 else 1

            # Zero excitation and inhibition since there are no synapses
            excitation_per_branch = torch.zeros(
                batch_size, n_branches, device=output.device
            )
            inhibition_per_branch = torch.zeros(
                batch_size, n_branches, device=output.device
            )
        else:
            # Normal processing for layers with actual synapses
            # Get the outputs from TopKLinear layers
            # These represent the weighted sum for each branch (already computed by TopKLinear)
            # Shape: (batch_size, num_branches)
            excitation_per_branch = (
                module.branch_excitation(excitatory_input)
                if module.branch_excitation is not None and excitatory_input is not None
                else torch.zeros_like(output)
            )
            inhibition_per_branch = (
                module.branch_inhibition(inhibitory_input)
                if module.branch_inhibition is not None and inhibitory_input is not None
                else torch.zeros_like(output)
            )

        # Compute branch_input contribution if available
        if branch_input is not None and module.input_branches:
            branch_current = module.branches_to_output(branch_input)
        else:
            branch_current = None

        # Store per-branch data
        self.data_dict[module._name]["excitation"] = excitation_per_branch
        self.data_dict[module._name]["inhibition"] = inhibition_per_branch
        self.data_dict[module._name]["branch_input"] = branch_current
        self.data_dict[module._name]["output"] = output
        self.data_dict[module._name]["has_synapses"] = not has_no_synapses
        self.data_dict[module._name]["has_exc_synapses"] = (
            module.branch_excitation is not None
        )
        self.data_dict[module._name]["has_inh_synapses"] = (
            module.branch_inhibition is not None and inhibitory_input is not None
        )
        self.data_dict[module._name]["has_branch_input"] = (
            branch_input is not None and module.input_branches
        )

        # Store raw inputs and module for LDA computation if needed
        if hasattr(self, "compute_lda_weights") and self.compute_lda_weights:
            self.data_dict[module._name]["raw_excitatory_input"] = excitatory_input
            self.data_dict[module._name]["raw_inhibitory_input"] = inhibitory_input
            self.data_dict[module._name]["raw_branch_input"] = branch_input
            self.data_dict[module._name]["module"] = module

    def remove_forward_hooks(self, handles: list[torch.utils.hooks.RemovableHandle]):
        """Remove forward hooks from the model."""
        for handle in handles:
            handle.remove()

    def _fit_nonnegative_lda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight_norm_order: Optional[int] = None,
        gamma: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Fit LDA with non-negative weight constraint.

        Delegates to the utility function in dendritic_modeling.utils.lda.
        """
        return fit_nonnegative_lda(X, y, weight_norm_order, gamma)

    def _apply_nonnegative_lda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight_norm_order: Optional[int] = None,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Apply non-negative LDA to get projected values.

        Delegates to the utility function in dendritic_modeling.utils.lda.
        """
        return apply_nonnegative_lda(X, y, weight_norm_order, gamma)

    def compute_lda_aggregated_signals(
        self,
        layer_data: dict,
        C: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute LDA-weighted versions of E, I, Vb, and Vout using TopK-masked features.

        For each branch, fit non-negative constrained LDA only on the input
        dimensions that pass through the TopK mask (i.e., the synapses that are
        actually connected). Non-negative weights match the network's positive
        synaptic weights.

        Vout_lin is computed by applying the same dendritic computation
        (shunting or additive) to E_lin, I_lin, Vb_lin, then passing through
        the network's reactivation function.

        Args:
            layer_data: Dictionary containing raw inputs and module
            C: Class labels, shape [n_samples,]

        Returns:
            Tuple of (E_lin, I_lin, Vb_lin, Vout_lin) each shape [n_samples, n_branches]
        """
        module = layer_data.get("module")
        if module is None:
            return None, None, None, None

        raw_exc = layer_data.get("raw_excitatory_input")
        raw_inh = layer_data.get("raw_inhibitory_input")
        raw_branch = layer_data.get("raw_branch_input")

        n_samples = C.shape[0]
        n_branches = layer_data["excitation"].shape[1]

        E_lin = np.zeros((n_samples, n_branches))
        I_lin = np.zeros((n_samples, n_branches))
        Vb_lin = np.zeros((n_samples, n_branches))
        Vout_lin = np.zeros((n_samples, n_branches))

        # Check if we have enough classes for LDA
        if len(np.unique(C)) <= 1:
            # Fall back to network weights
            E_lin = layer_data["excitation"].detach().cpu().numpy()
            I_lin = layer_data["inhibition"].detach().cpu().numpy()
            vb_fallback = layer_data.get("branch_input")
            if vb_fallback is not None:
                Vb_lin = vb_fallback.detach().cpu().numpy()
            else:
                Vb_lin = np.zeros_like(E_lin)
            Vout_lin = layer_data["output"].detach().cpu().numpy()
            return E_lin, I_lin, Vb_lin, Vout_lin

        # For E: Fit LDA on TopK-masked excitatory inputs for each branch
        if raw_exc is not None and module.branch_excitation is not None:
            raw_exc_np = raw_exc.detach().cpu().numpy()  # [n_samples, input_dim]

            # Get weight normalization parameters from the module (matching TopKLinear)
            exc_weight_norm_order = getattr(
                module.branch_excitation, "weight_norm_order", None
            )
            exc_gamma = getattr(module.branch_excitation, "gamma", 1.0)

            # Get the TopK mask for excitatory branches [n_branches, input_dim]
            exc_mask = module.branch_excitation.weight_mask().detach().cpu().numpy()

            # Fit separate LDA for each branch using its masked inputs
            for branch_idx in range(n_branches):
                branch_mask = exc_mask[branch_idx, :]  # [input_dim]
                active_indices = np.where(branch_mask > 0)[0]

                if len(active_indices) > 0:
                    # Extract only the active synaptic inputs for this branch
                    branch_inputs = raw_exc_np[:, active_indices]  # [n_samples, K]

                    try:
                        # Fit non-negative LDA on the K active synapses
                        # Use same weight normalization as the network's TopKLinear
                        E_lin[:, branch_idx] = self._apply_nonnegative_lda(
                            branch_inputs,
                            C,
                            weight_norm_order=exc_weight_norm_order,
                            gamma=exc_gamma,
                        )

                    except Exception as e:
                        self.logger.warning(
                            f"LDA failed for E branch {branch_idx}: {e}. Using network weights."
                        )
                        E_lin[:, branch_idx] = (
                            layer_data["excitation"][:, branch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                else:
                    E_lin[:, branch_idx] = (
                        layer_data["excitation"][:, branch_idx].detach().cpu().numpy()
                    )

        # For I: Fit LDA on TopK-masked inhibitory inputs for each branch
        if raw_inh is not None and module.branch_inhibition is not None:
            raw_inh_np = raw_inh.detach().cpu().numpy()  # [n_samples, input_dim]

            # Get weight normalization parameters from the module (matching TopKLinear)
            inh_weight_norm_order = getattr(
                module.branch_inhibition, "weight_norm_order", None
            )
            inh_gamma = getattr(module.branch_inhibition, "gamma", 1.0)

            # Get the TopK mask for inhibitory branches [n_branches, input_dim]
            inh_mask = module.branch_inhibition.weight_mask().detach().cpu().numpy()

            # Fit separate LDA for each branch using its masked inputs
            for branch_idx in range(n_branches):
                branch_mask = inh_mask[branch_idx, :]  # [input_dim]
                active_indices = np.where(branch_mask > 0)[0]

                if len(active_indices) > 0:
                    # Extract only the active synaptic inputs for this branch
                    branch_inputs = raw_inh_np[:, active_indices]  # [n_samples, K]

                    try:
                        # Fit non-negative LDA on the K active synapses
                        # Use same weight normalization as the network's TopKLinear
                        I_lin[:, branch_idx] = self._apply_nonnegative_lda(
                            branch_inputs,
                            C,
                            weight_norm_order=inh_weight_norm_order,
                            gamma=inh_gamma,
                        )

                    except Exception as e:
                        self.logger.warning(
                            f"LDA failed for I branch {branch_idx}: {e}. Using network weights."
                        )
                        I_lin[:, branch_idx] = (
                            layer_data["inhibition"][:, branch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                else:
                    I_lin[:, branch_idx] = (
                        layer_data["inhibition"][:, branch_idx].detach().cpu().numpy()
                    )

        # For Vb: Fit LDA on branch inputs using BlockLinear's block structure
        # branches_to_output is a BlockLinear where each output branch receives input
        # from a contiguous block of upstream branches
        if raw_branch is not None and module.input_branches:
            raw_branch_np = (
                raw_branch.detach().cpu().numpy()
            )  # [n_samples, n_upstream_branches]

            # BlockLinear doesn't have weight_norm_order/gamma like TopKLinear
            # Use defaults (L1 norm, gamma=1.0)
            vb_weight_norm_order = getattr(
                module.branches_to_output, "weight_norm_order", None
            )
            vb_gamma = getattr(module.branches_to_output, "gamma", 1.0)

            # Get block_size from branches_to_output (BlockLinear)
            if hasattr(module.branches_to_output, "block_size"):
                block_size = module.branches_to_output.block_size

                # Fit separate LDA for each output branch using its upstream block
                for branch_idx in range(n_branches):
                    # Each output branch receives from upstream indices [branch_idx * block_size : (branch_idx+1) * block_size]
                    start_idx = branch_idx * block_size
                    end_idx = min((branch_idx + 1) * block_size, raw_branch_np.shape[1])

                    if start_idx < raw_branch_np.shape[1]:
                        # Extract the upstream branches for this output branch
                        branch_inputs = raw_branch_np[:, start_idx:end_idx]

                        if branch_inputs.shape[1] > 0:
                            try:
                                # Fit non-negative LDA on the upstream branches
                                # Use same weight normalization as the network
                                Vb_lin[:, branch_idx] = self._apply_nonnegative_lda(
                                    branch_inputs,
                                    C,
                                    weight_norm_order=vb_weight_norm_order,
                                    gamma=vb_gamma,
                                )

                            except Exception as e:
                                self.logger.warning(
                                    f"LDA failed for Vb branch {branch_idx}: {e}. Using network weights."
                                )
                                Vb_lin[:, branch_idx] = (
                                    layer_data["branch_input"][:, branch_idx]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                        else:
                            Vb_lin[:, branch_idx] = (
                                layer_data["branch_input"][:, branch_idx]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                    else:
                        Vb_lin[:, branch_idx] = (
                            layer_data["branch_input"][:, branch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
            else:
                # Fallback: no block structure, use network weights
                Vb_lin = layer_data["branch_input"].detach().cpu().numpy()

        # Compute Vout_lin using the same dendritic computation as the network
        # With non-negative LDA weights, E_lin, I_lin, Vb_lin are all non-negative
        # so shunting inhibition formula works properly
        use_shunting = getattr(module, "use_shunting", True)
        epsilon = getattr(module, "epsilon", 1e-8)

        if use_shunting:
            # Shunting inhibition: voltage = (E + Vb) / (1 + E + I + Vb)
            # With non-negative inputs, this normalizes properly
            numerator = E_lin + Vb_lin
            denominator = 1.0 + E_lin + I_lin + Vb_lin
            voltage_lin = numerator / (denominator + epsilon)
        else:
            # Additive: voltage = E + Vb - I
            voltage_lin = E_lin + Vb_lin - I_lin

        # Apply reactivation function if available
        if hasattr(module, "reactivation") and module.reactivation is not None:
            import torch

            with torch.no_grad():
                voltage_tensor = torch.tensor(voltage_lin, dtype=torch.float32)
                if next(module.reactivation.parameters(), None) is not None:
                    device = next(module.reactivation.parameters()).device
                    voltage_tensor = voltage_tensor.to(device)
                Vout_lin = module.reactivation(voltage_tensor).cpu().numpy()
        else:
            # No reactivation, use voltage directly
            Vout_lin = voltage_lin

        return E_lin, I_lin, Vb_lin, Vout_lin

    def compute_shuffled_aggregated_signals(
        self,
        layer_data: dict,
        C: np.ndarray,
        shuffle_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute random-weighted versions of E, I, Vb for baseline comparison.

        Uses random non-negative weights with same normalization as LDA weights.
        This provides a random baseline to compare against the LDA-derived signals,
        showing how much improvement LDA provides over chance.

        Args:
            layer_data: Dictionary containing raw inputs and module
            C: Class labels, shape [n_samples,] (unused but kept for API consistency)
            shuffle_idx: Random seed offset for reproducibility across shuffles

        Returns:
            Tuple of (E_sh, I_sh, Vb_sh, Vout_sh) each shape [n_samples, n_branches]
        """
        module = layer_data.get("module")
        if module is None:
            return None, None, None, None

        raw_exc = layer_data.get("raw_excitatory_input")
        raw_inh = layer_data.get("raw_inhibitory_input")
        raw_branch = layer_data.get("raw_branch_input")

        n_samples = C.shape[0]
        n_branches = layer_data["excitation"].shape[1]

        E_sh = np.zeros((n_samples, n_branches))
        I_sh = np.zeros((n_samples, n_branches))
        Vb_sh = np.zeros((n_samples, n_branches))
        Vout_sh = np.zeros((n_samples, n_branches))

        # Create RNG with reproducible seed
        rng = np.random.default_rng(int(self.random_weights_seed) + int(shuffle_idx))

        # For E: Apply random weights on TopK-masked excitatory inputs
        if raw_exc is not None and module.branch_excitation is not None:
            raw_exc_np = raw_exc.detach().cpu().numpy()
            exc_weight_norm_order = getattr(
                module.branch_excitation, "weight_norm_order", None
            )
            exc_gamma = getattr(module.branch_excitation, "gamma", 1.0)
            exc_mask = module.branch_excitation.weight_mask().detach().cpu().numpy()

            for branch_idx in range(n_branches):
                branch_mask = exc_mask[branch_idx, :]
                active_indices = np.where(branch_mask > 0)[0]
                if len(active_indices) > 0:
                    branch_inputs = raw_exc_np[:, active_indices]
                    E_sh[:, branch_idx] = apply_random_weights(
                        branch_inputs,
                        weight_norm_order=exc_weight_norm_order,
                        gamma=exc_gamma,
                        rng=rng,
                    )

        # For I: Apply random weights on TopK-masked inhibitory inputs
        if raw_inh is not None and module.branch_inhibition is not None:
            raw_inh_np = raw_inh.detach().cpu().numpy()
            inh_weight_norm_order = getattr(
                module.branch_inhibition, "weight_norm_order", None
            )
            inh_gamma = getattr(module.branch_inhibition, "gamma", 1.0)
            inh_mask = module.branch_inhibition.weight_mask().detach().cpu().numpy()

            for branch_idx in range(n_branches):
                branch_mask = inh_mask[branch_idx, :]
                active_indices = np.where(branch_mask > 0)[0]
                if len(active_indices) > 0:
                    branch_inputs = raw_inh_np[:, active_indices]
                    I_sh[:, branch_idx] = apply_random_weights(
                        branch_inputs,
                        weight_norm_order=inh_weight_norm_order,
                        gamma=inh_gamma,
                        rng=rng,
                    )

        # For Vb: Apply random weights on branch inputs using BlockLinear structure
        if raw_branch is not None and module.input_branches:
            raw_branch_np = raw_branch.detach().cpu().numpy()
            vb_weight_norm_order = getattr(
                module.branches_to_output, "weight_norm_order", None
            )
            vb_gamma = getattr(module.branches_to_output, "gamma", 1.0)

            if hasattr(module.branches_to_output, "block_size"):
                block_size = module.branches_to_output.block_size
                for branch_idx in range(n_branches):
                    start_idx = branch_idx * block_size
                    end_idx = min((branch_idx + 1) * block_size, raw_branch_np.shape[1])
                    if start_idx < raw_branch_np.shape[1]:
                        branch_inputs = raw_branch_np[:, start_idx:end_idx]
                        if branch_inputs.shape[1] > 0:
                            Vb_sh[:, branch_idx] = apply_random_weights(
                                branch_inputs,
                                weight_norm_order=vb_weight_norm_order,
                                gamma=vb_gamma,
                                rng=rng,
                            )

        # Compute Vout_sh using the same dendritic computation as the network
        use_shunting = getattr(module, "use_shunting", True)
        epsilon = getattr(module, "epsilon", 1e-8)

        if use_shunting:
            numerator = E_sh + Vb_sh
            denominator = 1.0 + E_sh + I_sh + Vb_sh
            voltage_sh = numerator / (denominator + epsilon)
        else:
            voltage_sh = E_sh + Vb_sh - I_sh

        # Apply reactivation function if available
        if hasattr(module, "reactivation") and module.reactivation is not None:
            import torch

            with torch.no_grad():
                voltage_tensor = torch.tensor(voltage_sh, dtype=torch.float32)
                if next(module.reactivation.parameters(), None) is not None:
                    device = next(module.reactivation.parameters()).device
                    voltage_tensor = voltage_tensor.to(device)
                Vout_sh = module.reactivation(voltage_tensor).cpu().numpy()
        else:
            Vout_sh = voltage_sh

        return E_sh, I_sh, Vb_sh, Vout_sh

    def compute_lda_aggregated_signals_chained(
        self,
        layer_data: dict,
        C: np.ndarray,
        prev_layer_vout_lin: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute LDA-weighted signals with optional chained branch input.

        This is similar to compute_lda_aggregated_signals but allows overriding
        the branch input with Vout_lin from a previous layer, enabling proper
        chained LDA comparison across all network layers.

        Uses non-negative LDA weights to match network's non-negative synaptic weights.

        Args:
            layer_data: Dictionary containing raw inputs and module
            C: Class labels, shape [n_samples,]
            prev_layer_vout_lin: Vout_lin from previous layer, to use as branch input.
                                 If None, uses the raw_branch_input from layer_data.

        Returns:
            Tuple of (E_lin, I_lin, Vb_lin, Vout_lin) each shape [n_samples, n_branches]
        """
        # The "chained" path is identical to the standard LDA aggregation, except
        # that the upstream branch input can be replaced with the previous layer's
        # LDA-computed output (Vout_lin). To avoid duplicated logic (and to ensure
        # we still respect TopK masks / BlockLinear block structure), we delegate
        # to `compute_lda_aggregated_signals` after injecting the overridden
        # `raw_branch_input` when provided.

        if prev_layer_vout_lin is None:
            return self.compute_lda_aggregated_signals(layer_data, C)

        try:
            # Convert override to a torch Tensor so downstream code can `.detach()`.
            if isinstance(prev_layer_vout_lin, torch.Tensor):
                raw_branch_override = prev_layer_vout_lin
            else:
                raw_branch_override = torch.as_tensor(
                    prev_layer_vout_lin, dtype=torch.float32
                )

            # Move override to the same device as existing layer tensors if possible.
            device = None
            for key in (
                "raw_excitatory_input",
                "raw_inhibitory_input",
                "excitation",
                "inhibition",
                "output",
            ):
                t = layer_data.get(key)
                if isinstance(t, torch.Tensor):
                    device = t.device
                    break
            if device is not None:
                raw_branch_override = raw_branch_override.to(device)

            tmp = dict(layer_data)
            tmp["raw_branch_input"] = raw_branch_override
            return self.compute_lda_aggregated_signals(tmp, C)
        except Exception as e:
            self.logger.warning(
                "Chained LDA branch-input override failed; falling back to raw branch input. Error: %s",
                e,
            )
            return self.compute_lda_aggregated_signals(layer_data, C)

    def ensure_inhibitory_data(
        self, excitation_data: torch.Tensor, inhibition_data: torch.Tensor
    ):
        """Ensure inhibitory data is available, replacing with zeros if missing.

        Args:
            excitation_data: Shape [n_samples, n_branches] - excitatory activations
            inhibition_data: Shape [n_samples, n_branches] - inhibitory activations (can be None)

        Returns:
            Tuple of (excitation_data, inhibition_data) with guaranteed inhibitory data
        """
        # Handle case where inhibitory data might be missing/None
        if inhibition_data is None:
            inhibition_data = torch.zeros_like(excitation_data)
            self.logger.warning(
                "Inhibitory data is missing - using zeros for inhibitory activations"
            )

        return excitation_data, inhibition_data

    def has_inhibitory_data(self, layer_data_dict: dict) -> bool:
        """Check if the network has valid inhibitory data.

        Args:
            layer_data_dict: Dictionary containing layer data from all layers

        Returns:
            True if inhibitory data exists and is not all zeros
        """
        for layer_data in layer_data_dict.values():
            if "inhibition" in layer_data and layer_data["inhibition"] is not None:
                # Check if inhibitory data is not all zeros
                if torch.any(layer_data["inhibition"] != 0):
                    return True
        return False

    def compute_per_neuron_mi(
        self,
        E_data: np.ndarray,
        I_data: np.ndarray,
        Vout_data: np.ndarray,
        C: np.ndarray,
    ):
        """Compute MI for each neuron separately.

        Args:
            E_data: Shape [n_samples, n_neurons] - excitatory activations
            I_data: Shape [n_samples, n_neurons] - inhibitory activations
            Vout_data: Shape [n_samples, n_neurons] - output activations
            C: Shape [n_samples,] - class labels

        Returns:
            Dictionary with per-neuron MI results
        """
        _, n_neurons = E_data.shape
        per_neuron_results = []

        self.logger.info(f"Computing MI for {n_neurons} neurons individually...")

        for neuron_idx in range(n_neurons):
            # Extract data for single neuron
            E_neuron = E_data[:, neuron_idx : neuron_idx + 1]  # Keep 2D shape
            I_neuron = I_data[:, neuron_idx : neuron_idx + 1]
            Vout_neuron = Vout_data[:, neuron_idx : neuron_idx + 1]

            # Compute MI for this neuron
            neuron_results = self.compute_metrics(
                E_neuron, I_neuron, Vout_neuron, C, None, None
            )
            neuron_results["neuron_index"] = neuron_idx
            per_neuron_results.append(neuron_results)

        # Also compute average across neurons
        averaged_results = self._average_mi_results(per_neuron_results)

        return {
            "per_neuron_results": per_neuron_results,
            "averaged_results": averaged_results,
            "num_neurons": n_neurons,
        }

    def compute_ei_specific_mi(
        self,
        E_data: np.ndarray,
        I_data: np.ndarray,
        Vout_data: np.ndarray,
        C: np.ndarray,
    ):
        """Compute MI separately for excitatory and inhibitory networks.

        Args:
            E_data: Shape [n_samples, n_neurons] - excitatory activations
            I_data: Shape [n_samples, n_neurons] - inhibitory activations
            Vout_data: Shape [n_samples, n_neurons] - output activations
            C: Shape [n_samples,] - class labels

        Returns:
            Dictionary with separate E and I network MI results
        """
        self.logger.info(
            "Computing MI separately for excitatory and inhibitory networks..."
        )

        results = {}

        # Always compute excitatory network MI
        e_results = self.compute_metrics(
            E_data, np.zeros_like(I_data), Vout_data, C, None, None
        )
        e_results["network_type"] = "excitatory"
        results["excitatory_results"] = e_results

        # Check if inhibitory data is meaningful (not all zeros)
        has_inhibitory = np.any(I_data != 0)

        if has_inhibitory:
            # Compute MI for inhibitory network only
            i_results = self.compute_metrics(
                np.zeros_like(E_data), I_data, Vout_data, C, None, None
            )
            i_results["network_type"] = "inhibitory"
            results["inhibitory_results"] = i_results

            # Compute MI for combined E+I network (current default)
            combined_results = self.compute_metrics(
                E_data, I_data, Vout_data, C, None, None
            )
            combined_results["network_type"] = "combined"
            results["combined_results"] = combined_results

            self.logger.info(
                "Computed MI for excitatory, inhibitory, and combined networks"
            )
        else:
            # No inhibitory data available
            self.logger.warning(
                "No inhibitory data detected - skipping inhibitory and combined network analysis"
            )
            results["inhibitory_results"] = None
            results["combined_results"] = (
                e_results  # Use excitatory results as combined
            )
            results["inhibitory_data_available"] = False

        results["inhibitory_data_available"] = has_inhibitory
        return results

    def compute_per_layer_enhanced_analysis(self, C: np.ndarray):
        """Compute enhanced analysis (per-neuron and EI-specific) for each layer separately.

        Args:
            C: Class labels for all samples

        Returns:
            Dictionary with per-layer enhanced analysis results
        """
        self.logger.info("Computing per-layer enhanced analysis...")

        per_layer_results = {}

        for layer_name, layer_data in self.data_dict.items():
            if "excitation" not in layer_data:
                continue

            self.logger.info(f"Processing enhanced analysis for layer: {layer_name}")

            # Ensure inhibitory data is available for this layer
            E_filtered, I_filtered = self.ensure_inhibitory_data(
                layer_data["excitation"], layer_data["inhibition"]
            )

            E_layer = E_filtered.detach().cpu().numpy()
            I_layer = I_filtered.detach().cpu().numpy()
            Vout_layer = layer_data["output"].detach().cpu().numpy()

            layer_results = {}

            # Add per-neuron analysis for this layer
            if self.per_neuron_analysis:
                per_neuron_results = self.compute_per_neuron_mi(
                    E_layer, I_layer, Vout_layer, C
                )
                layer_results["per_neuron_analysis"] = per_neuron_results

            # Add EI-network analysis for this layer
            if self.per_einet_analysis:
                ei_specific_results = self.compute_ei_specific_mi(
                    E_layer, I_layer, Vout_layer, C
                )
                layer_results["per_einet_analysis"] = ei_specific_results

            # Add layer metadata
            layer_results["layer_name"] = layer_name
            layer_results["layer_shape"] = {
                "excitation": E_layer.shape,
                "inhibition": I_layer.shape,
                "output": Vout_layer.shape,
            }
            layer_results["has_inhibitory_data"] = np.any(I_layer != 0)

            per_layer_results[layer_name] = layer_results

        return per_layer_results

    def _save_per_layer_ei_summaries(
        self, per_layer_data: dict, save_path: str, base_filename: str
    ):
        """Generate and save individual summaries for each layer's E and I networks.

        Args:
            per_layer_data: Dictionary containing per-layer analysis results
            save_path: Directory to save summary files
            base_filename: Base filename for the summaries
        """
        self.logger.info("Generating individual layer E/I network summaries...")

        # Create subfolder for per-layer results
        per_layer_path = os.path.join(save_path, "per_layer_results")
        os.makedirs(per_layer_path, exist_ok=True)

        for layer_name, layer_results in per_layer_data.items():
            if "per_einet_analysis" in layer_results:
                ei_analysis = layer_results["per_einet_analysis"]

                # Clean layer name for filename
                clean_layer_name = layer_name.replace(".", "_").replace("/", "_")

                # Generate excitatory network summary
                if "excitatory_results" in ei_analysis:
                    exc_summary = self.get_summary(ei_analysis["excitatory_results"])
                    exc_summary_path = os.path.join(
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_excitatory_summary.txt",
                    )
                    with open(exc_summary_path, "w") as f:
                        f.write(f"Excitatory Network Summary - {layer_name}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(exc_summary)

                    # Also save the results as JSON for plotting
                    save_dict(
                        ei_analysis["excitatory_results"],
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_excitatory_results",
                    )

                # Generate inhibitory network summary
                if ei_analysis.get("inhibitory_results") is not None:
                    inh_summary = self.get_summary(ei_analysis["inhibitory_results"])
                    inh_summary_path = os.path.join(
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_inhibitory_summary.txt",
                    )
                    with open(inh_summary_path, "w") as f:
                        f.write(f"Inhibitory Network Summary - {layer_name}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(inh_summary)

                    # Also save the results as JSON for plotting
                    save_dict(
                        ei_analysis["inhibitory_results"],
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_inhibitory_results",
                    )

                # Generate combined network summary
                if ei_analysis.get("combined_results") is not None:
                    combined_summary = self.get_summary(ei_analysis["combined_results"])
                    combined_summary_path = os.path.join(
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_combined_summary.txt",
                    )
                    with open(combined_summary_path, "w") as f:
                        f.write(f"Combined E+I Network Summary - {layer_name}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(combined_summary)

                    # Also save the results as JSON for plotting
                    save_dict(
                        ei_analysis["combined_results"],
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_combined_results",
                    )

        self.logger.info(
            f"Generated individual summaries for {len(per_layer_data)} layers"
        )

    def analyze(
        self,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str = "cpu",
        save_path: Optional[str] = None,
        filename: str = "final",
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

        try:
            # Check if model has ExcitationInhibitionNetwork
            if not isinstance(model.core_network, ExcitationInhibitionNetwork):
                self.logger.warning(
                    "Model does not have ExcitationInhibitionNetwork. Skipping information analysis."
                )
                return

            self.logger.info("Model validation passed - starting analysis")
            analysis_start_time = time.time()

            # Prepare data
            self.logger.info(f"Preparing data: moving model to {device}")
            model = model.to(device)
            model.eval()

            self.logger.info("Loading test dataset...")
            try:
                items = test_dataset[:]
                inputs, labels = items[0], items[1]
            except Exception:
                self.logger.warning(
                    "Dataset does not support full slicing; materializing via DataLoader."
                )
                xs = []
                ys = []
                max_samples = self.max_samples if self.max_samples else None
                total = 0

                for batch in DataLoader(test_dataset, batch_size=1024, shuffle=False):
                    x_batch, y_batch = batch[0], batch[1]
                    if max_samples is not None and total >= max_samples:
                        break

                    if (
                        max_samples is not None
                        and total + x_batch.shape[0] > max_samples
                    ):
                        keep = max_samples - total
                        x_batch = x_batch[:keep]
                        y_batch = y_batch[:keep]

                    xs.append(x_batch)
                    ys.append(y_batch)
                    total += x_batch.shape[0]

                inputs = torch.cat(xs, dim=0)
                labels = torch.cat(ys, dim=0)
            inputs: torch.Tensor
            labels: torch.Tensor
            self.logger.info(
                f"Loaded dataset: {len(inputs)} samples, input shape: {inputs.shape}"
            )

            # Optional: Sample a subset of data for faster computation
            if self.max_samples and len(inputs) > self.max_samples:
                self.logger.info(
                    f"Sampling {self.max_samples} samples from {len(inputs)} for faster computation"
                )
                indices = torch.randperm(len(inputs))[: self.max_samples]
                inputs = inputs[indices]
                labels = labels[indices]
                self.logger.info(f"After sampling: {len(inputs)} samples")

            self.logger.info(f"Moving data to {device}...")
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Attach forward hooks and run forward pass
            handles = self.attach_forward_hooks(model)
            with torch.no_grad():
                _ = model(inputs)
            self.remove_forward_hooks(handles)
            C = labels.detach().cpu().numpy()

            self.logger.debug(
                f"Data collection completed. Found {len(self.data_dict)} layers."
            )

            self.data_dict: dict[str, dict[str, torch.Tensor]]

            # Discrete class entropy (nats). Used for "layer total" heuristics in
            # single-branch mode (e.g. sum/union aggregations of per-branch MI).
            entropy_C: Optional[float]
            if self._is_discrete_class_labels(C):
                entropy_C = self._discrete_entropy_nats(C)
            else:
                entropy_C = None

            if self.computation_level == "single_branch":
                # Individual branch analysis: Compute MI for each branch separately and average
                all_results = []
                branch_names = []
                layer_results = {}  # Store results per layer

                total_layers = len(
                    [
                        name
                        for name, data in self.data_dict.items()
                        if "excitation" in data
                    ]
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

                    # Get data for all branches in this layer
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
                        layer_data.get(
                            "has_branch_input", Vb_all_branches_raw is not None
                        )
                    )

                    # Ensure inhibitory data is available
                    E_all_branches_filtered, I_all_branches_filtered = (
                        self.ensure_inhibitory_data(
                            E_all_branches_raw, I_all_branches_raw
                        )
                    )

                    E_all_branches = E_all_branches_filtered.detach().cpu().numpy()
                    I_all_branches = I_all_branches_filtered.detach().cpu().numpy()

                    # Get Vb data
                    if Vb_all_branches_raw is not None:
                        Vb_all_branches = Vb_all_branches_raw.detach().cpu().numpy()
                    else:
                        Vb_all_branches = None

                    # Compute LDA versions if requested
                    if self.compute_lda_weights:
                        E_lin, I_lin, Vb_lin, Vout_lin = (
                            self.compute_lda_aggregated_signals(layer_data, C)
                        )
                    else:
                        E_lin, I_lin, Vb_lin, Vout_lin = None, None, None, None

                    # Compute random-weight versions for baseline comparison
                    shuffled_signals_list = []
                    if self.compute_lda_weights and self.lda_n_shuffles > 0:
                        for shuffle_idx in range(self.lda_n_shuffles):
                            E_sh, I_sh, Vb_sh, Vout_sh = (
                                self.compute_shuffled_aggregated_signals(
                                    layer_data, C, shuffle_idx
                                )
                            )
                            shuffled_signals_list.append((E_sh, I_sh, Vb_sh, Vout_sh))

                    # E_all_branches shape: (n_samples, n_branches)
                    n_branches = E_all_branches.shape[1]

                    # Store per-branch results for this layer
                    layer_branch_results = []
                    layer_start_time = time.time()

                    # Process each branch individually
                    # Extract short layer name for progress bar
                    short_name = (
                        layer_name.split(".")[-1] if "." in layer_name else layer_name
                    )
                    for branch_idx in tqdm(
                        range(n_branches),
                        desc=f"  {short_name}",
                        leave=False,
                        ncols=100,
                        mininterval=0.5,
                    ):
                        branch_start_time = time.time()

                        # Extract data for this specific branch
                        E_branch = E_all_branches[
                            :, branch_idx : branch_idx + 1
                        ]  # Keep 2D shape
                        I_branch = I_all_branches[:, branch_idx : branch_idx + 1]
                        Vout_branch = Vout_all_branches[:, branch_idx : branch_idx + 1]
                        Vb_branch = (
                            Vb_all_branches[:, branch_idx : branch_idx + 1]
                            if Vb_all_branches is not None
                            else None
                        )

                        # Compute MI for this single branch with network weights
                        branch_results = self.compute_metrics(
                            E_branch, I_branch, Vout_branch, C, None, Vb_branch
                        )
                        # Optional: estimate finite-sample bias via a label-shuffle null baseline
                        # (produces *_null metrics) using the same estimator.method.
                        if self.mi_null_shuffles > 0:
                            try:
                                self._add_label_shuffle_null_baselines_in_place(
                                    branch_results,
                                    E=E_branch,
                                    I_var=I_branch,
                                    Vout=Vout_branch,
                                    C=C,
                                    Vb=Vb_branch,
                                    Vinf=None,
                                    has_exc_synapses=has_exc_synapses,
                                    has_inh_synapses=has_inh_synapses,
                                    has_branch_input=has_branch_input
                                    and (Vb_branch is not None),
                                    seed_offset=int(layer_count * 10_000 + branch_idx),
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Null (label-shuffle) baseline failed for branch {branch_idx}: {e}"
                                )
                        all_results.append(branch_results)
                        layer_branch_results.append(branch_results)
                        branch_names.append(f"{layer_name}_branch_{branch_idx}")

                        # If LDA requested, compute MI for LDA-weighted versions
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
                                Vout_branch_lin,  # Use LDA-computed Vout
                                C,
                                None,
                                Vb_branch_lin,
                            )
                            if self.mi_null_shuffles > 0:
                                try:
                                    self._add_label_shuffle_null_baselines_in_place(
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
                                        seed_offset=int(
                                            1_000_000
                                            + layer_count * 10_000
                                            + branch_idx
                                        ),
                                    )
                                except Exception as e:
                                    self.logger.warning(
                                        f"Null (label-shuffle) baseline failed for LDA branch {branch_idx}: {e}"
                                    )
                            # Rename keys to indicate LDA versions
                            branch_results_lin_renamed = {}
                            for key, value in branch_results_lin.items():
                                if isinstance(value, dict):
                                    renamed_dict = {}
                                    for subkey, subval in value.items():
                                        new_subkey = self._rename_metric_key_tokens(
                                            subkey,
                                            {
                                                "E": "E_lin",
                                                "I": "I_lin",
                                                "Vb": "Vb_lin",
                                                "Vout": "Vout_lin",
                                                "Vinf": "Vinf_lin",
                                            },
                                        )
                                        renamed_dict[new_subkey] = subval
                                    branch_results_lin_renamed[key] = renamed_dict
                                else:
                                    branch_results_lin_renamed[key] = value

                            # Merge LDA results into branch_results
                            for key, value in branch_results_lin_renamed.items():
                                if isinstance(value, dict) and key in branch_results:
                                    branch_results[key].update(value)

                        # If shuffled baseline requested, compute MI for random weights
                        if shuffled_signals_list:
                            # Accumulate results from all shuffles
                            shuffled_metrics_acc = {}
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

                                # Accumulate values for averaging
                                for key, value in branch_results_sh.items():
                                    if isinstance(value, dict):
                                        if key not in shuffled_metrics_acc:
                                            shuffled_metrics_acc[key] = {}
                                        for subkey, subval in value.items():
                                            if isinstance(subval, (int, float)):
                                                if (
                                                    subkey
                                                    not in shuffled_metrics_acc[key]
                                                ):
                                                    shuffled_metrics_acc[key][
                                                        subkey
                                                    ] = []
                                                shuffled_metrics_acc[key][
                                                    subkey
                                                ].append(subval)

                            # Average the accumulated shuffled metrics and rename
                            for key, subdict in shuffled_metrics_acc.items():
                                if key in branch_results and isinstance(
                                    branch_results[key], dict
                                ):
                                    for subkey, values_list in subdict.items():
                                        if values_list:
                                            avg_value = np.mean(values_list)
                                            new_subkey = self._rename_metric_key_tokens(
                                                subkey,
                                                {
                                                    "E": "E_sh",
                                                    "I": "I_sh",
                                                    "Vb": "Vb_sh",
                                                    "Vout": "Vout_sh",
                                                    "Vinf": "Vinf_sh",
                                                },
                                            )
                                            branch_results[key][new_subkey] = avg_value

                        # Drop metrics that depend on non-existent variables so they don't
                        # contribute zeros to any averaging/statistics.
                        self._filter_result_metrics_in_place(
                            branch_results,
                            has_exc_synapses=has_exc_synapses,
                            has_inh_synapses=has_inh_synapses,
                            has_branch_input=has_branch_input,
                        )

                        self.logger.debug(
                            f"  Branch {branch_idx + 1}/{n_branches} completed (took {time.time() - branch_start_time:.1f}s)"
                        )

                    # Compute layer statistics
                    if layer_branch_results:
                        layer_stats = self._compute_layer_statistics(
                            layer_branch_results,
                            entropy_C=(
                                entropy_C if self.compute_layer_total_proxies else None
                            ),
                            topk=self.layer_total_topk,
                        )
                        # Add has_synapses flag for plotting code to use
                        if layer_name in self.data_dict:
                            layer_stats["has_synapses"] = self.data_dict[
                                layer_name
                            ].get("has_synapses", True)
                            # Zero out E/I related MI values when layer has no synapses
                            # (e.g., soma with somatic_synapses=False)
                            if not layer_stats["has_synapses"]:
                                for key in list(layer_stats.keys()):
                                    # Remove any metric involving E or I (they don't exist here),
                                    # so downstream plots/averages don't get diluted by zeros.
                                    if any(
                                        x in key
                                        for x in [
                                            "I(E;",
                                            "I(I;",
                                            "I(E_lin;",
                                            "I(I_lin;",
                                        ]
                                    ):
                                        layer_stats.pop(key, None)
                        layer_results[layer_name] = layer_stats

                    self.logger.info(
                        f"Layer {layer_name} completed: {n_branches} branches processed in {time.time() - layer_start_time:.1f}s"
                    )

                # Average the results across branches
                if all_results:
                    results = self._average_mi_results(all_results)
                    results["branch_names"] = branch_names
                    results["num_branches"] = len(all_results)
                    results["layer_statistics"] = (
                        layer_results  # Add per-layer statistics
                    )
                    results["computation_level"] = "single_branch"
                else:
                    self.logger.warning("No branch results to average")
                    return

            elif self.computation_level == "layer_branch":
                # Layer-level analysis: treat each layer as a unit
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
                    # Aggregate all branches within this layer using branch_aggregation
                    E_layer_data_raw = layer_data[
                        "excitation"
                    ]  # Shape: [n_samples, n_branches]
                    I_layer_data_raw = layer_data["inhibition"]
                    Vb_layer_data_raw = layer_data.get("branch_input")
                    Vout_layer_data = layer_data["output"]

                    has_exc_synapses = bool(layer_data.get("has_exc_synapses", True))
                    has_inh_synapses = bool(layer_data.get("has_inh_synapses", True))
                    has_branch_input = Vb_layer_data_raw is not None

                    # Ensure inhibitory data is available
                    E_layer_data, I_layer_data = self.ensure_inhibitory_data(
                        E_layer_data_raw, I_layer_data_raw
                    )

                    # Keep Vb as None when it doesn't exist (e.g., outermost distal layer)
                    # or when config disables branch input analysis.
                    Vb_layer_data = Vb_layer_data_raw if has_branch_input else None
                    C_for_layer = C

                    if self.branch_aggregation == "mean":
                        E_layer = (
                            E_layer_data.mean(dim=1).detach().cpu().numpy()
                        )  # [n_samples,]
                        I_layer = I_layer_data.mean(dim=1).detach().cpu().numpy()
                        if Vb_layer_data is not None:
                            Vb_layer = Vb_layer_data.mean(dim=1).detach().cpu().numpy()
                            Vb_layer = Vb_layer.reshape(-1, 1)
                        else:
                            Vb_layer = None
                        Vout_layer = Vout_layer_data.mean(dim=1).detach().cpu().numpy()
                        # Reshape to 2D for MI computation
                        E_layer = E_layer.reshape(-1, 1)
                        I_layer = I_layer.reshape(-1, 1)
                        Vout_layer = Vout_layer.reshape(-1, 1)
                    elif self.branch_aggregation == "multivariate":
                        # Multivariate: Keep all branches as multivariate input (no aggregation)
                        E_layer = (
                            E_layer_data.detach().cpu().numpy()
                        )  # [n_samples, n_branches]
                        I_layer = (
                            I_layer_data.detach().cpu().numpy()
                        )  # [n_samples, n_branches]
                        Vb_layer = (
                            Vb_layer_data.detach().cpu().numpy()
                            if Vb_layer_data is not None
                            else None
                        )
                        Vout_layer = (
                            Vout_layer_data.detach().cpu().numpy()
                        )  # [n_samples, n_branches]
                    elif self.branch_aggregation == "sample":
                        # Sample: Treat each branch as a separate observation/sample
                        # Flatten [n_samples, n_branches] -> [n_samples * n_branches, 1]
                        E_layer = (
                            E_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
                        )
                        I_layer = (
                            I_layer_data.detach().cpu().numpy().flatten().reshape(-1, 1)
                        )
                        Vb_layer = (
                            Vb_layer_data.detach()
                            .cpu()
                            .numpy()
                            .flatten()
                            .reshape(-1, 1)
                            if Vb_layer_data is not None
                            else None
                        )
                        Vout_layer = (
                            Vout_layer_data.detach()
                            .cpu()
                            .numpy()
                            .flatten()
                            .reshape(-1, 1)
                        )
                        # Need to replicate C for each branch
                        n_samples, n_branches = E_layer_data.shape
                        C_expanded = np.repeat(
                            C, n_branches, axis=0
                        )  # Expand C to match new sample count
                        C_for_layer = C_expanded
                    else:
                        # Default to mean
                        E_layer = E_layer_data.mean(dim=1).detach().cpu().numpy()
                        I_layer = I_layer_data.mean(dim=1).detach().cpu().numpy()
                        if Vb_layer_data is not None:
                            Vb_layer = Vb_layer_data.mean(dim=1).detach().cpu().numpy()
                            Vb_layer = Vb_layer.reshape(-1, 1)
                        else:
                            Vb_layer = None
                        Vout_layer = Vout_layer_data.mean(dim=1).detach().cpu().numpy()
                        # Reshape to 2D for MI computation
                        E_layer = E_layer.reshape(-1, 1)
                        I_layer = I_layer.reshape(-1, 1)
                        Vout_layer = Vout_layer.reshape(-1, 1)

                    # Compute MI for this layer
                    layer_results = self.compute_metrics(
                        E_layer, I_layer, Vout_layer, C_for_layer, None, Vb_layer
                    )
                    self._filter_result_metrics_in_place(
                        layer_results,
                        has_exc_synapses=has_exc_synapses,
                        has_inh_synapses=has_inh_synapses,
                        has_branch_input=has_branch_input,
                    )
                    if self.mi_null_shuffles > 0:
                        try:
                            self._add_label_shuffle_null_baselines_in_place(
                                layer_results,
                                E=E_layer,
                                I_var=I_layer,
                                Vout=Vout_layer,
                                C=C_for_layer,
                                Vb=Vb_layer,
                                Vinf=None,
                                has_exc_synapses=has_exc_synapses,
                                has_inh_synapses=has_inh_synapses,
                                has_branch_input=has_branch_input
                                and (Vb_layer is not None),
                                seed_offset=len(all_results),
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Null (label-shuffle) baseline failed for layer {layer_name}: {e}"
                            )
                    all_results.append(layer_results)
                    layer_names.append(layer_name)

                # Average the results across layers
                if all_results:
                    results = self._average_mi_results(all_results)
                    results["layer_names"] = layer_names
                    results["num_layers"] = len(all_results)
                    results["computation_level"] = "layer_branch"

                    # Generate layer statistics for plotting (similar to single_branch mode)
                    layer_statistics = {}
                    for _i, (layer_name, layer_result) in enumerate(
                        zip(layer_names, all_results)
                    ):
                        # Convert individual layer result to statistics format expected by plotting
                        layer_stats = self._compute_layer_statistics([layer_result])

                        # Add has_synapses flag for plotting code to use
                        if layer_name in self.data_dict:
                            layer_stats["has_synapses"] = self.data_dict[
                                layer_name
                            ].get("has_synapses", True)
                            # Zero out E/I related MI values when layer has no synapses
                            # (e.g., soma with somatic_synapses=False)
                            if not layer_stats["has_synapses"]:
                                for key in list(layer_stats.keys()):
                                    # Remove any metric involving E or I (they don't exist here),
                                    # so downstream plots/averages don't get diluted by zeros.
                                    if any(
                                        x in key
                                        for x in [
                                            "I(E;",
                                            "I(I;",
                                            "I(E_lin;",
                                            "I(I_lin;",
                                        ]
                                    ):
                                        layer_stats.pop(key, None)

                        layer_statistics[layer_name] = layer_stats

                    results["layer_statistics"] = layer_statistics
                    self.logger.info(
                        f"Generated layer statistics for {len(layer_statistics)} layers for plotting"
                    )
                else:
                    self.logger.warning("No layer results to average")
                    return

            else:  # computation_level == "all_branch"
                # Full network aggregation: properly handle different-sized tensors
                self.logger.info(
                    f"Aggregating branches using method: {self.branch_aggregation} (full network aggregation)"
                )

                # Collect aggregated data from each layer
                layer_excitation = []
                layer_inhibition = []
                layer_branch_input = []
                missing_vb_in_any_layer = False
                layer_output = []

                for layer_data in self.data_dict.values():
                    if "excitation" in layer_data:
                        # Aggregate within each layer first
                        if self.branch_aggregation == "mean":
                            layer_E = layer_data["excitation"].mean(
                                dim=1, keepdim=True
                            )  # [n_samples, 1]
                            layer_I = layer_data["inhibition"].mean(dim=1, keepdim=True)
                            layer_Vb_raw = layer_data.get("branch_input")
                            if layer_Vb_raw is not None:
                                layer_Vb = layer_Vb_raw.mean(dim=1, keepdim=True)
                            else:
                                layer_Vb = None
                                missing_vb_in_any_layer = True
                            layer_Vout = layer_data["output"].mean(dim=1, keepdim=True)
                        elif self.branch_aggregation == "multivariate":
                            # For network level with multivariate, keep branches separate for each layer
                            layer_E = layer_data[
                                "excitation"
                            ]  # [n_samples, n_branches]
                            layer_I = layer_data[
                                "inhibition"
                            ]  # [n_samples, n_branches]
                            layer_Vb_raw = layer_data.get("branch_input")
                            if layer_Vb_raw is not None:
                                layer_Vb = layer_Vb_raw
                            else:
                                layer_Vb = None
                                missing_vb_in_any_layer = True
                            layer_Vout = layer_data["output"]  # [n_samples, n_branches]
                        elif self.branch_aggregation == "sample":
                            # For network level with sample, flatten each layer's branches into samples
                            n_samples, n_branches = layer_data["excitation"].shape
                            layer_E = layer_data["excitation"].view(
                                n_samples * n_branches, 1
                            )
                            layer_I = layer_data["inhibition"].view(
                                n_samples * n_branches, 1
                            )
                            layer_Vb_raw = layer_data.get("branch_input")
                            if layer_Vb_raw is not None:
                                layer_Vb = layer_Vb_raw.view(n_samples * n_branches, 1)
                            else:
                                layer_Vb = None
                                missing_vb_in_any_layer = True
                            layer_Vout = layer_data["output"].view(
                                n_samples * n_branches, 1
                            )
                        else:
                            # Default to mean
                            layer_E = layer_data["excitation"].mean(dim=1, keepdim=True)
                            layer_I = layer_data["inhibition"].mean(dim=1, keepdim=True)
                            layer_Vb_raw = layer_data.get("branch_input")
                            if layer_Vb_raw is not None:
                                layer_Vb = layer_Vb_raw.mean(dim=1, keepdim=True)
                            else:
                                layer_Vb = None
                                missing_vb_in_any_layer = True
                            layer_Vout = layer_data["output"].mean(dim=1, keepdim=True)

                        layer_excitation.append(layer_E)
                        layer_inhibition.append(layer_I)
                        if layer_Vb is not None:
                            layer_branch_input.append(layer_Vb)
                        layer_output.append(layer_Vout)

                if not layer_excitation:
                    self.logger.warning(
                        "No excitatory activations collected. Skipping information analysis."
                    )
                    return

                # Handle concatenation based on aggregation method
                C_for_metrics = C
                if self.branch_aggregation == "sample":
                    # For sample aggregation, concatenate along sample dimension (axis=0)
                    E = (
                        torch.cat(layer_excitation, dim=0).detach().cpu().numpy()
                    )  # [n_samples * n_branches * n_layers, 1]
                    I_var = torch.cat(layer_inhibition, dim=0).detach().cpu().numpy()
                    # Vb only exists if it exists for ALL included layers in sample-aggregation
                    Vb = (
                        torch.cat(layer_branch_input, dim=0).detach().cpu().numpy()
                        if (layer_branch_input and not missing_vb_in_any_layer)
                        else None
                    )
                    Vout = torch.cat(layer_output, dim=0).detach().cpu().numpy()
                    # Expand C to match the new sample count
                    total_samples = E.shape[0]
                    original_samples = C.shape[0]
                    expansion_factor = total_samples // original_samples
                    C_for_metrics = np.repeat(C, expansion_factor, axis=0)
                elif self.branch_aggregation == "multivariate":
                    # For multivariate, concatenate along feature dimension (axis=1)
                    E = (
                        torch.cat(layer_excitation, dim=1).detach().cpu().numpy()
                    )  # [n_samples, n_branches * n_layers]
                    I_var = torch.cat(layer_inhibition, dim=1).detach().cpu().numpy()
                    Vb = (
                        torch.cat(layer_branch_input, dim=1).detach().cpu().numpy()
                        if layer_branch_input
                        else None
                    )
                    Vout = torch.cat(layer_output, dim=1).detach().cpu().numpy()
                else:
                    # For mean/max/sum, concatenate across layers (all tensors are [n_samples, 1])
                    E = (
                        torch.cat(layer_excitation, dim=1).detach().cpu().numpy()
                    )  # [n_samples, n_layers]
                    I_var = torch.cat(layer_inhibition, dim=1).detach().cpu().numpy()
                    Vb = (
                        torch.cat(layer_branch_input, dim=1).detach().cpu().numpy()
                        if layer_branch_input
                        else None
                    )
                    Vout = torch.cat(layer_output, dim=1).detach().cpu().numpy()

                self.logger.info(
                    "Aggregated data shapes - "
                    f"E: {E.shape}, I: {I_var.shape}, Vb: {None if Vb is None else Vb.shape}, Vout: {Vout.shape}, C: {C_for_metrics.shape}"
                )

                # Get soma activation (Vinf) if available for aggregate mode
                Vinf = None
                # For now, we'll use the final model output as Vinf
                with torch.no_grad():
                    final_output: torch.Tensor = model(inputs)
                    if hasattr(model, "final_fc"):
                        # Model has final classification layer
                        Vinf = final_output.detach().cpu().numpy()

                # Run information analysis on aggregated data
                results = self.compute_metrics(E, I_var, Vout, C_for_metrics, Vinf, Vb)
                if self.mi_null_shuffles > 0:
                    try:
                        self._add_label_shuffle_null_baselines_in_place(
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
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Null (label-shuffle) baseline failed for all-branch analysis: {e}"
                        )
                results["aggregation_method"] = self.branch_aggregation
                results["computation_level"] = "all_branch"

            # Add enhanced analysis results if requested
            if (
                self.per_neuron_analysis
                or self.per_einet_analysis
                or self.per_layer_analysis
            ):
                self.logger.info("Running enhanced analysis modes...")

                # Add per-layer analysis if requested
                if self.per_layer_analysis:
                    per_layer_results = self.compute_per_layer_enhanced_analysis(C)
                    results["per_layer_analysis"] = per_layer_results

                # Network-level enhanced analysis (existing functionality)
                if self.per_neuron_analysis or self.per_einet_analysis:
                    # Collect all data for enhanced analysis
                    all_E_data = []
                    all_I_data = []
                    all_Vout_data = []

                    for layer_data in self.data_dict.values():
                        if "excitation" in layer_data:
                            # Ensure inhibitory data is available
                            E_filtered, I_filtered = self.ensure_inhibitory_data(
                                layer_data["excitation"], layer_data["inhibition"]
                            )
                            all_E_data.append(E_filtered.detach().cpu().numpy())
                            all_I_data.append(I_filtered.detach().cpu().numpy())
                            all_Vout_data.append(
                                layer_data["output"].detach().cpu().numpy()
                            )

                    # Concatenate data across layers (network-level view)
                    if all_E_data:
                        E_all = np.concatenate(
                            all_E_data, axis=1
                        )  # [n_samples, total_neurons]
                        I_all = np.concatenate(all_I_data, axis=1)
                        Vout_all = np.concatenate(all_Vout_data, axis=1)

                        # Add per-neuron analysis results
                        if self.per_neuron_analysis:
                            per_neuron_results = self.compute_per_neuron_mi(
                                E_all, I_all, Vout_all, C
                            )
                            results["per_neuron_analysis"] = per_neuron_results

                        # Add EI-network analysis results
                        if self.per_einet_analysis:
                            ei_specific_results = self.compute_ei_specific_mi(
                                E_all, I_all, Vout_all, C
                            )
                            results["per_einet_analysis"] = ei_specific_results

            # Add metadata about analysis configuration
            results["per_neuron_analysis_enabled"] = self.per_neuron_analysis
            results["per_einet_analysis_enabled"] = self.per_einet_analysis
            results["per_layer_analysis_enabled"] = self.per_layer_analysis
            results["has_inhibitory_data"] = self.has_inhibitory_data(self.data_dict)

            # Analysis completion summary
            total_analysis_time = time.time() - analysis_start_time
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

            # Check if we need to add synthetic soma entry
            # This is only needed when somatic_synapses=False AND the soma layer
            # wasn't captured by forward hooks (shouldn't happen now that we fixed hooks)
            somatic_synapses = getattr(model.core_network, "somatic_synapses", True)
            self.logger.info(f"Somatic synapses enabled: {somatic_synapses}")

            # Find if the actual soma layer is already in layer_statistics
            soma_already_captured = False
            if "layer_statistics" in results:
                for layer_name in results["layer_statistics"].keys():
                    if "branch_layers." in layer_name:
                        try:
                            # Find the highest index layer (soma)
                            branch_idx = int(
                                layer_name.split("branch_layers.")[1].split(".")[0]
                            )
                            # Check if this is a soma layer (no synapses)
                            layer_data = results["layer_statistics"][layer_name]
                            if layer_data.get("has_synapses") is False:
                                soma_already_captured = True
                                self.logger.info(
                                    f"Soma layer already captured: {layer_name}"
                                )
                                break
                        except (ValueError, IndexError):
                            continue

            # Only add synthetic soma if the real soma wasn't captured
            if (
                not somatic_synapses
                and "layer_statistics" in results
                and not soma_already_captured
            ):
                self.logger.info(
                    f"Adding synthetic soma entry to {len(results['layer_statistics'])} existing layers"
                )

                # Find the real soma layer to get its Vout information
                # The real soma is the layer with the highest branch index (corresponds to layer_idx=0)
                soma_vout_mean = 0.0
                soma_vout_std = 0.0

                # Find the layer with the highest branch index in layer_statistics
                max_branch_idx = -1
                real_soma_layer = None

                for layer_name, _layer_data in results["layer_statistics"].items():
                    if "branch_layers." in layer_name:
                        try:
                            branch_idx = int(
                                layer_name.split("branch_layers.")[1].split(".")[0]
                            )
                            if branch_idx > max_branch_idx:
                                max_branch_idx = branch_idx
                                real_soma_layer = layer_name
                        except (ValueError, IndexError):
                            continue

                # Extract I(Vout;C) and I(Vb;C) from the real soma layer
                # The soma still receives branch input from dendrites (Vb),
                # even when somatic_synapses=False (no E/I synapses at soma)
                soma_vb_mean = 0.0
                soma_vb_std = 0.0
                soma_vb_lin_mean = 0.0
                soma_vb_lin_std = 0.0
                soma_vb_vout_mean = 0.0
                soma_vb_vout_std = 0.0
                soma_vb_vout_c_mean = 0.0
                soma_vb_vout_c_std = 0.0
                soma_vb_lin_vout_mean = 0.0
                soma_vb_lin_vout_std = 0.0
                soma_vb_lin_vout_c_mean = 0.0
                soma_vb_lin_vout_c_std = 0.0
                # Vout_lin metrics
                soma_vout_lin_mean = 0.0
                soma_vout_lin_std = 0.0

                if real_soma_layer and real_soma_layer in results["layer_statistics"]:
                    real_soma_stats = results["layer_statistics"][real_soma_layer]
                    soma_vout_mean = real_soma_stats.get("I(Vout;C)_mean", 0.0)
                    soma_vout_std = real_soma_stats.get("I(Vout;C)_std", 0.0)
                    # Get Vb metrics from real soma layer - these represent dendrites feeding into soma
                    soma_vb_mean = real_soma_stats.get("I(Vb;C)_mean", 0.0)
                    soma_vb_std = real_soma_stats.get("I(Vb;C)_std", 0.0)
                    soma_vb_lin_mean = real_soma_stats.get("I(Vb_lin;C)_mean", 0.0)
                    soma_vb_lin_std = real_soma_stats.get("I(Vb_lin;C)_std", 0.0)
                    # Get Vb-related pairwise and conditional MI
                    soma_vb_vout_mean = real_soma_stats.get("I(Vb;Vout)_mean", 0.0)
                    soma_vb_vout_std = real_soma_stats.get("I(Vb;Vout)_std", 0.0)
                    soma_vb_vout_c_mean = real_soma_stats.get("I(Vb;Vout|C)_mean", 0.0)
                    soma_vb_vout_c_std = real_soma_stats.get("I(Vb;Vout|C)_std", 0.0)
                    soma_vb_lin_vout_mean = real_soma_stats.get(
                        "I(Vb_lin;Vout_lin)_mean", 0.0
                    )
                    soma_vb_lin_vout_std = real_soma_stats.get(
                        "I(Vb_lin;Vout_lin)_std", 0.0
                    )
                    soma_vb_lin_vout_c_mean = real_soma_stats.get(
                        "I(Vb_lin;Vout_lin|C)_mean", 0.0
                    )
                    soma_vb_lin_vout_c_std = real_soma_stats.get(
                        "I(Vb_lin;Vout_lin|C)_std", 0.0
                    )
                    # Get Vout_lin metrics from real soma layer
                    soma_vout_lin_mean = real_soma_stats.get("I(Vout_lin;C)_mean", 0.0)
                    soma_vout_lin_std = real_soma_stats.get("I(Vout_lin;C)_std", 0.0)
                    self.logger.info(f"Found real soma layer: {real_soma_layer}")
                    self.logger.info(
                        f"Using real soma I(Vout;C): {soma_vout_mean:.4f} ± {soma_vout_std:.4f}"
                    )
                    self.logger.info(
                        f"Using real soma I(Vb;C): {soma_vb_mean:.4f} ± {soma_vb_std:.4f}"
                    )
                else:
                    self.logger.warning(
                        "Could not find real soma layer for I(Vout;C) computation"
                    )

                # Add synthetic soma entry with appropriate values
                # Note: E/I synapses are zero, but Vb (branch input from dendrites) carries information
                soma_entry = {
                    "I(E;C)_mean": 0.0,  # Zero - no excitatory synapses
                    "I(E;C)_std": 0.0,
                    "I(I;C)_mean": 0.0,  # Zero - no inhibitory synapses
                    "I(I;C)_std": 0.0,
                    # Dendrites still feed into soma - use real soma layer's Vb values
                    "I(Vb;C)_mean": soma_vb_mean,
                    "I(Vb;C)_std": soma_vb_std,
                    "I(Vout;C)_mean": soma_vout_mean,  # Real computed value from soma output
                    "I(Vout;C)_std": soma_vout_std,  # Real computed std from soma output
                    "I(E,I;C)_mean": 0.0,  # Zero - no synapses
                    "I(E,I;C)_std": 0.0,
                    # LDA metrics - E/I are zero (no synapses), but Vb_lin carries information
                    "I(E_lin;C)_mean": 0.0,
                    "I(E_lin;C)_std": 0.0,
                    "I(I_lin;C)_mean": 0.0,
                    "I(I_lin;C)_std": 0.0,
                    "I(Vb_lin;C)_mean": soma_vb_lin_mean,
                    "I(Vb_lin;C)_std": soma_vb_lin_std,
                    "I(Vout_lin;C)_mean": soma_vout_lin_mean,
                    "I(Vout_lin;C)_std": soma_vout_lin_std,
                    "I(E_lin,I_lin;C)_mean": 0.0,
                    "I(E_lin,I_lin;C)_std": 0.0,
                    # Pairwise MI
                    "I(E;I)_mean": 0.0,  # Zero - no synapses
                    "I(E;I)_std": 0.0,
                    "I(E;Vout)_mean": 0.0,  # Zero - no excitatory synapses
                    "I(E;Vout)_std": 0.0,
                    "I(I;Vout)_mean": 0.0,  # Zero - no inhibitory synapses
                    "I(I;Vout)_std": 0.0,
                    "I(E;Vb)_mean": 0.0,
                    "I(E;Vb)_std": 0.0,
                    "I(I;Vb)_mean": 0.0,
                    "I(I;Vb)_std": 0.0,
                    # Vb;Vout is meaningful - dendrites to soma output
                    "I(Vb;Vout)_mean": soma_vb_vout_mean,
                    "I(Vb;Vout)_std": soma_vb_vout_std,
                    # LDA pairwise MI - use Vout_lin for consistency
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
                    # Vb_lin;Vout_lin is meaningful
                    "I(Vb_lin;Vout_lin)_mean": soma_vb_lin_vout_mean,
                    "I(Vb_lin;Vout_lin)_std": soma_vb_lin_vout_std,
                    # Conditional MI
                    "I(E;I|C)_mean": 0.0,  # Zero - no synapses
                    "I(E;I|C)_std": 0.0,
                    "I(E;Vout|C)_mean": 0.0,  # Zero - no excitatory synapses
                    "I(E;Vout|C)_std": 0.0,
                    "I(I;Vout|C)_mean": 0.0,  # Zero - no inhibitory synapses
                    "I(I;Vout|C)_std": 0.0,
                    "I(E;Vb|C)_mean": 0.0,
                    "I(E;Vb|C)_std": 0.0,
                    "I(I;Vb|C)_mean": 0.0,
                    "I(I;Vb|C)_std": 0.0,
                    # Vb;Vout|C is meaningful
                    "I(Vb;Vout|C)_mean": soma_vb_vout_c_mean,
                    "I(Vb;Vout|C)_std": soma_vb_vout_c_std,
                    # LDA conditional MI - use Vout_lin for consistency
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
                    # Vb_lin;Vout_lin|C is meaningful
                    "I(Vb_lin;Vout_lin|C)_mean": soma_vb_lin_vout_c_mean,
                    "I(Vb_lin;Vout_lin|C)_std": soma_vb_lin_vout_c_std,
                }
                # Remove any metrics that depend on non-existent synapses (E/I) so we don't
                # implicitly treat them as zero-valued measurements.
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
                # Add soma entry to layer_statistics
                results["layer_statistics"]["synthetic_soma"] = soma_entry
                self.logger.info(
                    f"Added synthetic soma with real I(Vout;C)={soma_vout_mean:.4f}. Total layers now: {len(results['layer_statistics'])}"
                )
            elif not somatic_synapses and "layer_statistics" in results:
                # The real soma layer (has_synapses=False) was captured by hooks.
                self.logger.info(
                    "Soma layer already captured; skipping synthetic soma."
                )
            elif not somatic_synapses and "layer_statistics" not in results:
                self.logger.warning(
                    "somatic_synapses=False but no layer_statistics found in results"
                )

            # ------------------------------------------------------------------
            # Unit handling: estimators produce nats; optionally convert to bits
            # for saved results and plots.
            # ------------------------------------------------------------------
            results["information_units"] = getattr(self, "output_units", "nats")
            if results["information_units"] == "bits":
                self._convert_information_units_in_place(results)

            # Compute variance analysis for plotting (AFTER synthetic soma is added)
            self.logger.info("Computing variance analysis for plotting...")
            variance_analysis = self._compute_variance_analysis(results)
            results["variance_analysis"] = variance_analysis
            self.logger.info(
                f"Variance analysis computed with {len(variance_analysis.get('layer_variances', {}))} layers"
            )

            # Compute LDA vs trained performance comparison if LDA weights are enabled
            if self.compute_lda_weights and save_path is not None:
                try:
                    # Use the inputs/labels already loaded in the analyze method
                    lda_comparison = self.compare_lda_vs_trained_performance(
                        model, inputs, labels, device
                    )
                    results["lda_vs_trained_performance"] = lda_comparison

                    # Plot the comparison if LDA accuracy was computed
                    if lda_comparison.get("lda_accuracy") is not None:
                        from dendritic_modeling.plotting.visualizations.information_plots import (
                            plot_lda_vs_trained_performance,
                        )

                        plot_lda_vs_trained_performance(
                            lda_comparison, save_path, self.logger
                        )
                        self.logger.info(
                            "LDA vs trained performance comparison completed!"
                        )
                    else:
                        self.logger.warning(
                            "LDA accuracy not available, skipping performance plot"
                        )
                except Exception as e:
                    self.logger.error(
                        f"Error computing LDA vs trained performance: {e}"
                    )
                    traceback.print_exc()

            # Save results
            if save_path is not None:
                self.logger.info(f"Saving results to {save_path}/{filename}")
                save_dict(results, save_path, filename)

                # Generate main summary
                summary = self.get_summary(results)
                summary_path = os.path.join(save_path, f"{filename}_summary.txt")
                with open(summary_path, "w") as f:
                    f.write(summary)

                # Generate individual summaries for each layer's E and I networks
                try:
                    if "per_layer_analysis" in results:
                        self._save_per_layer_ei_summaries(
                            results["per_layer_analysis"], save_path, filename
                        )
                    self._generate_information_plots(
                        results, save_path, somatic_synapses
                    )
                except Exception as plot_err:
                    self.logger.warning(f"Post-save visualization failed: {plot_err}")

                self.logger.info("Results saved successfully!")

            # Always return results for plotting
            return results

        except Exception as e:
            self.logger.error(f"Error in information analysis: {e}")
            traceback.print_exc()

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
            device = next(model.parameters()).device

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
