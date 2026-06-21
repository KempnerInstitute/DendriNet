"""Configuration parsing helpers for information analysis."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class _InformationComponentSelection:
    """Boolean analysis-component switches parsed from selection.components."""

    compute_basic_mi: bool
    compute_pairwise_mi: bool
    compute_conditional_mi: bool
    compute_pid: bool
    compute_gaussian_fisher: bool
    compute_ablation_aligned_mi: bool
    compute_layer_total_proxies: bool
    compute_upstream_unique_cmi: bool


@dataclass(frozen=True)
class _InformationScopeSelection:
    """Boolean analysis-scope switches parsed from selection.scopes/views."""

    per_neuron_analysis: bool
    per_einet_analysis: bool
    per_layer_analysis: bool


@dataclass(frozen=True)
class _InformationSignalVariantSelection:
    """LDA and random-weight signal settings parsed from grouped/legacy config."""

    compute_lda_weights: bool
    lda_n_shuffles: int
    random_weights_seed: int


@dataclass(frozen=True)
class _InformationLabelShuffleNullConfig:
    """Label-shuffle null baseline settings parsed from grouped/legacy config."""

    mi_null_shuffles: int
    mi_null_seed: int
    mi_null_components: set[str]


@dataclass(frozen=True)
class _InformationPidConfig:
    """PID preprocessing settings parsed from grouped/legacy config."""

    pid_binarize: bool
    pid_binarize_method: str
    pid_binarize_threshold: float


@dataclass(frozen=True)
class _InformationGaussianFisherConfig:
    """Gaussian/Fisher proxy settings parsed from grouped config."""

    gaussian_fisher_aggregation: str
    gaussian_fisher_mi_transform: str
    gaussian_fisher_eps: float


@dataclass(frozen=True)
class _InformationAblationAlignedConfig:
    """Ablation-aligned metric defaults parsed from legacy/grouped config."""

    compute_ablation_aligned_mi: bool
    compute_upstream_unique_cmi: bool
    compute_layer_total_proxies: bool
    layer_total_topk: int


@dataclass(frozen=True)
class _InformationRuntimeConfig:
    """Runtime/reporting settings parsed from grouped/legacy config."""

    include_vinf: bool
    normalize_mi: bool
    output_units: str
    verbose: bool
    invalid_output_units: str | None


@dataclass(frozen=True)
class _InformationGranularityConfig:
    """Granularity and enhanced-view settings parsed from grouped/legacy config."""

    computation_level: str
    branch_aggregation: str
    per_neuron_analysis: bool
    per_einet_analysis: bool
    per_layer_analysis: bool


@dataclass(frozen=True)
class _InformationEstimatorCoreConfig:
    """Estimator core settings parsed from grouped/legacy config."""

    method: str
    max_samples: Any
    n_neighbors: int
    n_bins: int
    copula_type: str


def _get_config_value(obj: Any, key: str, default: Any = None) -> Any:
    """Get a config value via attribute or mapping access."""
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


def _get_config_path(
    obj: Any,
    path: tuple[str, ...],
    default: Any = None,
) -> Any:
    """Get a nested config value with the same None-as-missing semantics."""
    cur = obj
    for key in path:
        cur = _get_config_value(cur, key, None)
        if cur is None:
            return default
    return cur


def _normalize_config_token(value: Any) -> str:
    """Normalize config selector tokens for legacy and grouped configs."""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _parse_information_component_selection(
    components_raw: Sequence[Any],
) -> _InformationComponentSelection | None:
    """Parse selection.components into analysis component switches."""
    if not components_raw:
        return None

    components = {
        _normalize_config_token(component)
        for component in components_raw
        if str(component).strip()
    }
    return _InformationComponentSelection(
        compute_basic_mi=bool({"basic", "basic_mi"}.intersection(components)),
        compute_pairwise_mi=bool({"pairwise", "pairwise_mi"}.intersection(components)),
        compute_conditional_mi=bool(
            {"conditional", "conditional_mi"}.intersection(components)
        ),
        compute_pid="pid" in components,
        compute_gaussian_fisher=bool(
            {"gaussian_fisher", "gaussianfisher", "fisher"}.intersection(components)
        ),
        compute_ablation_aligned_mi=bool(
            {"ablation_aligned", "ablation"}.intersection(components)
        ),
        compute_layer_total_proxies=bool(
            {"layer_proxies", "layer_total_proxies", "layer_total"}.intersection(
                components
            )
        ),
        compute_upstream_unique_cmi=bool(
            {"upstream_unique_cmi", "upstream_unique"}.intersection(components)
        ),
    )


def _parse_information_scope_selection(
    scopes_raw: Sequence[Any],
) -> _InformationScopeSelection | None:
    """Parse selection.scopes/views into enhanced analysis switches."""
    if not scopes_raw:
        return None

    scopes = {
        _normalize_config_token(scope) for scope in scopes_raw if str(scope).strip()
    }
    return _InformationScopeSelection(
        per_neuron_analysis=bool({"neuron", "per_neuron"}.intersection(scopes)),
        per_einet_analysis=bool({"einet", "per_einet"}.intersection(scopes)),
        per_layer_analysis=bool({"layer", "per_layer"}.intersection(scopes)),
    )


def _build_information_signal_variant_selection(
    *,
    params: Any,
    selection_cfg: Any,
    compute_cfg: Any,
    baselines_cfg: Any,
) -> _InformationSignalVariantSelection:
    """Build LDA/random-weight signal settings from grouped or legacy config."""
    compute_lda_weights = bool(
        _get_config_value(
            compute_cfg,
            "compute_lda_weights",
            _get_config_value(params, "compute_lda_weights", False),
        )
    )
    lda_n_shuffles = int(
        _get_config_path(
            baselines_cfg,
            ("random_weights", "n_shuffles"),
            _get_config_path(
                baselines_cfg,
                ("signal_shuffle", "n_shuffles"),
                _get_config_value(params, "lda_n_shuffles", 0),
            ),
        )
        or 0
    )
    random_weights_seed = int(
        _get_config_path(
            baselines_cfg,
            ("random_weights", "seed"),
            _get_config_path(
                baselines_cfg,
                ("signal_shuffle", "seed"),
                _get_config_value(params, "random_weights_seed", 0),
            ),
        )
        or 0
    )

    variants_raw = list(
        _get_config_value(
            selection_cfg,
            "signal_variants",
            _get_config_value(params, "signal_variants", []),
        )
        or []
    )
    if variants_raw:
        variants = {
            _normalize_config_token(variant)
            for variant in variants_raw
            if str(variant).strip()
        }
        want_lda = "lda" in variants
        want_random = bool({"random", "shuffle", "shuffled"}.intersection(variants))
        compute_lda_weights = bool(want_lda or want_random)
        lda_n_shuffles = lda_n_shuffles if want_random else 0

    return _InformationSignalVariantSelection(
        compute_lda_weights=compute_lda_weights,
        lda_n_shuffles=lda_n_shuffles,
        random_weights_seed=random_weights_seed,
    )


def _build_information_label_shuffle_null_config(
    *,
    params: Any,
    baselines_cfg: Any,
) -> _InformationLabelShuffleNullConfig:
    """Build label-shuffle null baseline settings from grouped or legacy config."""
    mi_null_shuffles = int(
        _get_config_path(
            baselines_cfg,
            ("label_shuffle_null", "n_shuffles"),
            _get_config_value(params, "mi_null_shuffles", 0),
        )
        or 0
    )
    mi_null_seed = int(
        _get_config_path(
            baselines_cfg,
            ("label_shuffle_null", "seed"),
            _get_config_value(params, "mi_null_seed", 0),
        )
        or 0
    )
    null_components_raw = list(
        _get_config_path(
            baselines_cfg,
            ("label_shuffle_null", "components"),
            _get_config_value(params, "mi_null_components", ["basic"]),
        )
        or ["basic"]
    )
    mi_null_components = {
        _normalize_config_token(component)
        for component in null_components_raw
        if str(component).strip()
    } or {"basic"}

    return _InformationLabelShuffleNullConfig(
        mi_null_shuffles=mi_null_shuffles,
        mi_null_seed=mi_null_seed,
        mi_null_components=mi_null_components,
    )


def _build_information_pid_config(
    *, params: Any, pid_cfg: Any
) -> _InformationPidConfig:
    """Build PID preprocessing settings from grouped or legacy config."""
    return _InformationPidConfig(
        pid_binarize=bool(
            _get_config_value(
                pid_cfg,
                "binarize",
                _get_config_value(params, "pid_binarize", False),
            )
        ),
        pid_binarize_method=str(
            _get_config_value(
                pid_cfg,
                "binarize_method",
                _get_config_value(params, "pid_binarize_method", "median"),
            )
        ),
        pid_binarize_threshold=float(
            _get_config_value(
                pid_cfg,
                "binarize_threshold",
                _get_config_value(params, "pid_binarize_threshold", 0.5),
            )
        ),
    )


def _build_information_gaussian_fisher_config(
    *, gaussian_fisher_cfg: Any
) -> _InformationGaussianFisherConfig:
    """Build Gaussian/Fisher proxy settings from grouped config."""
    return _InformationGaussianFisherConfig(
        gaussian_fisher_aggregation=str(
            _get_config_value(gaussian_fisher_cfg, "aggregation", "sum")
        )
        .strip()
        .lower(),
        gaussian_fisher_mi_transform=str(
            _get_config_value(gaussian_fisher_cfg, "mi_transform", "half_log1p")
        )
        .strip()
        .lower(),
        gaussian_fisher_eps=float(
            _get_config_value(
                gaussian_fisher_cfg,
                "ridge",
                _get_config_value(gaussian_fisher_cfg, "eps", 1e-8),
            )
        ),
    )


def _build_information_ablation_aligned_config(
    *,
    params: Any,
    compute_cfg: Any,
) -> _InformationAblationAlignedConfig:
    """Build ablation-aligned metric defaults from legacy/grouped config."""
    return _InformationAblationAlignedConfig(
        compute_ablation_aligned_mi=bool(
            getattr(params, "compute_ablation_aligned_mi", True)
        ),
        compute_upstream_unique_cmi=bool(
            getattr(params, "compute_upstream_unique_cmi", False)
        ),
        compute_layer_total_proxies=bool(
            getattr(params, "compute_layer_total_proxies", True)
        ),
        layer_total_topk=int(
            _get_config_value(
                compute_cfg,
                "layer_total_topk",
                _get_config_value(params, "layer_total_topk", 10),
            )
        ),
    )


def _build_information_runtime_config(
    *,
    params: Any,
    compute_cfg: Any,
) -> _InformationRuntimeConfig:
    """Build runtime/reporting settings from grouped or legacy config."""
    output_units = _normalize_config_token(
        _get_config_value(
            compute_cfg,
            "output_units",
            _get_config_value(params, "output_units", "bits"),
        )
    )
    invalid_output_units = None
    if output_units not in {"bits", "bit", "nats", "nat"}:
        invalid_output_units = output_units
        output_units = "bits"
    output_units = "bits" if output_units in {"bits", "bit"} else "nats"

    return _InformationRuntimeConfig(
        include_vinf=bool(
            _get_config_value(
                compute_cfg,
                "include_vinf",
                _get_config_value(params, "include_vinf", False),
            )
        ),
        normalize_mi=bool(
            _get_config_value(
                compute_cfg,
                "normalize_mi",
                _get_config_value(params, "normalize_mi", False),
            )
        ),
        output_units=output_units,
        verbose=bool(
            _get_config_value(
                compute_cfg,
                "verbose",
                _get_config_value(params, "verbose", False),
            )
        ),
        invalid_output_units=invalid_output_units,
    )


def _build_information_granularity_config(
    *,
    params: Any,
    compute_cfg: Any,
) -> _InformationGranularityConfig:
    """Build granularity and enhanced-view settings from grouped/legacy config."""
    return _InformationGranularityConfig(
        computation_level=str(
            _get_config_value(
                compute_cfg,
                "computation_level",
                _get_config_value(params, "computation_level", "single_branch"),
            )
        ),
        branch_aggregation=str(
            _get_config_value(
                compute_cfg,
                "branch_aggregation",
                _get_config_value(params, "branch_aggregation", "mean"),
            )
        ),
        per_neuron_analysis=bool(
            _get_config_value(
                compute_cfg,
                "per_neuron_analysis",
                _get_config_value(params, "per_neuron_analysis", False),
            )
        ),
        per_einet_analysis=bool(
            _get_config_value(
                compute_cfg,
                "per_einet_analysis",
                _get_config_value(params, "per_einet_analysis", False),
            )
        ),
        per_layer_analysis=bool(
            _get_config_value(
                compute_cfg,
                "per_layer_analysis",
                _get_config_value(params, "per_layer_analysis", False),
            )
        ),
    )


def _build_information_estimator_core_config(
    *,
    params: Any,
    estimator_cfg: Any,
    compute_cfg: Any,
) -> _InformationEstimatorCoreConfig:
    """Build estimator identity/sample/bin settings from grouped or legacy config."""
    return _InformationEstimatorCoreConfig(
        method=str(
            _get_config_value(
                estimator_cfg,
                "method",
                _get_config_value(params, "method", "kraskov"),
            )
        ),
        max_samples=_get_config_value(
            compute_cfg,
            "max_samples",
            _get_config_value(params, "max_samples", None),
        ),
        n_neighbors=int(
            _get_config_path(
                estimator_cfg,
                ("kraskov", "n_neighbors"),
                _get_config_value(params, "n_neighbors", 10),
            )
        ),
        n_bins=int(
            _get_config_path(
                estimator_cfg,
                ("binned", "n_bins"),
                _get_config_value(params, "n_bins", 20),
            )
        ),
        copula_type=str(
            _get_config_path(
                estimator_cfg,
                ("copula", "type"),
                _get_config_value(params, "copula_type", "gaussian"),
            )
        ),
    )


def _build_information_estimator_method_params(
    *,
    params: Any,
    estimator_cfg: Any,
    method: str,
) -> dict[str, Any]:
    """Build estimator-specific method parameters from grouped or legacy config."""
    decoder_cfg = _get_config_value(estimator_cfg, "decoder", None)
    decoder_cont_cfg = _get_config_value(decoder_cfg, "continuous", None)
    binned_cfg = _get_config_value(estimator_cfg, "binned", None)
    normalized_method = _normalize_config_token(method)

    if normalized_method == "decoder":
        return {
            "cv_folds": int(
                _get_config_value(
                    decoder_cfg,
                    "cv_folds",
                    _get_config_value(params, "decoder_cv_folds", 5),
                )
            ),
            "C": float(
                _get_config_value(
                    decoder_cfg,
                    "C",
                    _get_config_value(params, "decoder_C", 1.0),
                )
            ),
            "standardize": bool(
                _get_config_value(
                    decoder_cfg,
                    "standardize",
                    _get_config_value(params, "decoder_standardize", True),
                )
            ),
            "seed": int(
                _get_config_value(
                    decoder_cfg,
                    "seed",
                    _get_config_value(params, "decoder_seed", 0),
                )
            ),
            "continuous_strategy": str(
                _get_config_value(
                    decoder_cont_cfg,
                    "strategy",
                    _get_config_value(params, "decoder_continuous_strategy", "none"),
                )
            ),
            "continuous_max_dim": int(
                _get_config_value(
                    decoder_cont_cfg,
                    "max_total_dim",
                    _get_config_value(params, "decoder_continuous_max_total_dim", 2),
                )
            ),
            "gaussian_ridge": float(
                _get_config_value(
                    decoder_cont_cfg,
                    "gaussian_ridge",
                    _get_config_value(params, "decoder_gaussian_ridge", 1e-6),
                )
            ),
            "binning_strategy": str(
                _get_config_value(
                    binned_cfg,
                    "binning_strategy",
                    _get_config_value(params, "binning_strategy", "quantile"),
                )
            ),
            "bias_correction": str(
                _get_config_value(
                    binned_cfg,
                    "bias_correction",
                    _get_config_value(params, "binning_bias_correction", "none"),
                )
            ),
        }

    if normalized_method == "binned":
        return {
            "binning_strategy": str(
                _get_config_value(
                    binned_cfg,
                    "binning_strategy",
                    _get_config_value(params, "binning_strategy", "quantile"),
                )
            ),
            "bias_correction": str(
                _get_config_value(
                    binned_cfg,
                    "bias_correction",
                    _get_config_value(params, "binning_bias_correction", "none"),
                )
            ),
        }

    return {}
