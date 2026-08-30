from dataclasses import dataclass, field
from typing import Optional

from dendritic_modeling.analysis.registry import get_registered_analyzers
from dendritic_modeling.config.base import BaseConfig


@dataclass
class EvaluationRuntimeConfig(BaseConfig):
    """Controls how analysis iterates over a dataset.

    mode:
        "auto"        - materialize if estimated size < *materialize_threshold_mb*,
                         otherwise stream via DataLoader.
        "materialize" - always load the full dataset into memory (old behaviour).
        "stream"      - always iterate batch-by-batch through a DataLoader.

    max_batches:
        If set, cap evaluation to this many batches per split (useful for
        cheap approximate metrics during training).

    max_samples:
        If set, cap the number of examples exposed to the analysis.
        This is useful for expensive hook-based analyses.

    materialize_threshold_mb:
        Dataset size threshold (MB) for the ``"auto"`` mode switch.
        Default 1024 MB keeps CIFAR/MNIST fully materialized.
    """

    mode: str = "auto"
    batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = 2
    in_order: Optional[bool] = None
    max_batches: Optional[int] = None
    max_samples: Optional[int] = None
    materialize_threshold_mb: int = 1024


@dataclass
class AnalysisRuntimeConfig(BaseConfig):
    """Shared runtime profiles for analysis during training and final evaluation."""

    training: EvaluationRuntimeConfig = field(default_factory=EvaluationRuntimeConfig)
    final: EvaluationRuntimeConfig = field(default_factory=EvaluationRuntimeConfig)


@dataclass
class PerformanceAnalysisParams(BaseConfig):
    """Parameters for performance evaluation metrics.

    Each boolean toggles whether that metric is computed/reported:
    - accuracy
    - auc
    - categorical_loglikelihood
    - mse
    - cosine_similarity
    """

    accuracy: bool = True
    auc: bool = True
    categorical_loglikelihood: bool = True
    mse: bool = False
    cosine_similarity: bool = False


@dataclass
class PerformanceAnalysisConfig(BaseConfig):
    """Enable/disable performance analysis.

    - enabled: turns the analysis module on/off.
    - training: if true, run during training.
    - training_splits / final_splits: which dataset splits are evaluated
      during training-time vs. final evaluation.
    """

    enabled: bool = False
    training: bool = False
    params: PerformanceAnalysisParams = field(default_factory=PerformanceAnalysisParams)
    training_splits: list[str] = field(
        default_factory=lambda: ["train", "valid", "test"]
    )
    final_splits: list[str] = field(default_factory=lambda: ["train", "valid", "test"])


@dataclass
class InformationSelectionParams(BaseConfig):
    """Selectors for what to compute and how to break it down.

    - components: metric families to compute
      - "basic": I(E;C), I(I;C), I(Vb;C), I(Vout;C), I(E,I;C), ...
      - "pairwise": I(E;I), I(E;Vout), I(Vb;Vout), ...
      - "conditional": conditional MI terms like I(E;Vout|C)
      - "gaussian_fisher": Gaussian/Fisher MI proxy metrics (fast; theory-inspired)
      - "pid": partial information decomposition (currently disabled; reserved)
      - "ablation_aligned": extra CMI terms designed to mirror ablations,
        such as I(Vout;C|E), I(Vout;C|I), or I(Vout;C|E,I)
      - "layer_proxies": layer-total proxies derived from single-branch metrics
      - "upstream_unique_cmi": adds I(Vb;C|E,I) (can be expensive)
    - scopes: extra summaries/breakdowns to run
      - "layer": per-layer breakdown
      - "einet": separate excitatory vs inhibitory pathway summaries
      - "neuron": per-neuron breakdown (expensive)
    - signal_variants: which signal variants to compute/plot
      - "network": DendriNet signals (always included)
      - "lda": *_lin variants using LDA projections
      - "random": *_sh random-weight baseline for LDA (requires baselines.random_weights.n_shuffles > 0)
    """

    components: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)
    signal_variants: list[str] = field(default_factory=list)
    # Backward-compatible alias (deprecated): prefer `scopes`.
    views: list[str] = field(default_factory=list)


@dataclass
class InformationGaussianFisherParams(BaseConfig):
    """Gaussian/Fisher proxy parameters (enabled via selection.components)."""

    aggregation: str = "sum"  # "sum"|"mean"|"max"
    mi_transform: str = "half_log1p"  # "half_log1p"|"log1p"|"none"
    eps: float = 1e-8
    # Optional alias for `eps` (if provided, prefer ridge over eps).
    ridge: Optional[float] = None


@dataclass
class InformationComputeParams(BaseConfig):
    """Runtime/aggregation controls (method-agnostic)."""

    max_samples: Optional[int] = None
    include_vinf: bool = True  # Include soma activation (Vinf)
    normalize_mi: bool = False  # Normalize by H(C)
    # Output units for MI/CMI-like quantities reported by the analyzer/plots.
    # Estimators operate in nats internally; the analyzer can convert to bits for reporting.
    output_units: str = "bits"  # "bits"|"nats"
    verbose: bool = False

    computation_level: str = (
        "single_branch"  # "single_branch", "layer_branch", "all_branch"
    )
    branch_aggregation: str = "mean"  # "mean", "multivariate", "sample"

    # Layer-total proxies derived from single-branch metrics
    layer_total_topk: int = 10  # K used for *_topK_sum metrics (int >= 1)

    # Method-specific compute configs (enabled via selection.components)
    gaussian_fisher: InformationGaussianFisherParams = field(
        default_factory=InformationGaussianFisherParams
    )

    # Legacy (deprecated) scope toggles (use selection.scopes)
    per_neuron_analysis: bool = False
    per_einet_analysis: bool = False
    per_layer_analysis: bool = False

    # Legacy (deprecated) signal-variant toggles (use selection.signal_variants / baselines)
    compute_lda_weights: bool = False


@dataclass
class InformationPIDParams(BaseConfig):
    """PID-specific options (currently disabled; reserved for future use).

    Notes
    -----
    PID is currently disabled:
    - Synergy/redundancy are computed via an optional external PID backend.
      If that backend is not installed, PID metrics are skipped.
    - The reported "unique_*" terms are computed as:
      ``I(source;target) - redundancy`` using the *same* MI estimator selected by
      ``estimator.method``.

    Because typical PID backends assume **discrete** variables, we default to
    binarization. Treat PID numbers as exploratory diagnostics, not a primary
    metric.
    """

    binarize: bool = True
    binarize_method: str = "median"  # "median", "mean", "quantile"
    binarize_threshold: float = 0.5


@dataclass
class InformationEstimatorKraskovParams(BaseConfig):
    """Kraskov/KSG (kNN) estimator hyperparameters."""

    n_neighbors: int = 10


@dataclass
class InformationEstimatorBinnedParams(BaseConfig):
    """Histogram/binning plug-in estimator hyperparameters."""

    n_bins: int = 20
    binning_strategy: str = "quantile"  # "quantile", "uniform"
    bias_correction: str = "none"  # "none", "miller_madow"


@dataclass
class InformationEstimatorDecoderContinuousParams(BaseConfig):
    """Continuous fallback options for the decoder estimator."""

    strategy: str = "none"  # "none", "binned", "gaussian"
    max_total_dim: int = 2  # max total dim (x+y [+z]) for continuous MI/CMI
    gaussian_ridge: float = 1e-6


@dataclass
class InformationEstimatorDecoderParams(BaseConfig):
    """Decoder-based estimator hyperparameters (discrete target C)."""

    cv_folds: int = 5
    C: float = 1.0
    standardize: bool = True
    seed: int = 0
    continuous: InformationEstimatorDecoderContinuousParams = field(
        default_factory=InformationEstimatorDecoderContinuousParams
    )


@dataclass
class InformationEstimatorCopulaParams(BaseConfig):
    """Copula-based estimator hyperparameters (experimental).

    - type="gaussian": Gaussian copula (closed-form in the univariate case).
    - type="kernel": nonparametric KDE on the copula density (sample-hungry).
    - type="dvc": placeholder for a TF/DVC implementation; currently falls back
      to the simple copula path in this repository.

    Note: for multivariate inputs, the current implementation falls back to a
    kNN/KSG estimator on rank-transformed (uniform-marginal) data.
    """

    type: str = "gaussian"  # "gaussian", "kernel", "dvc"


@dataclass
class InformationEstimatorSklearnParams(BaseConfig):
    """Sklearn MI estimator hyperparameters (histogram/KDE depending on dimensionality)."""

    n_bins: int = 20
    bandwidth: Optional[float] = None


@dataclass
class InformationEstimatorParams(BaseConfig):
    """Estimator selection and method-specific hyperparameters.

    - method: "kraskov", "decoder", "binned", "sklearn", "copula", "auto"
    """

    method: str = "kraskov"
    kraskov: InformationEstimatorKraskovParams = field(
        default_factory=InformationEstimatorKraskovParams
    )
    binned: InformationEstimatorBinnedParams = field(
        default_factory=InformationEstimatorBinnedParams
    )
    decoder: InformationEstimatorDecoderParams = field(
        default_factory=InformationEstimatorDecoderParams
    )
    copula: InformationEstimatorCopulaParams = field(
        default_factory=InformationEstimatorCopulaParams
    )
    sklearn: InformationEstimatorSklearnParams = field(
        default_factory=InformationEstimatorSklearnParams
    )


@dataclass
class InformationBaselinesRandomWeightsParams(BaseConfig):
    """Random-weight baseline for LDA aggregation (produces *_sh metrics).

    This baseline generates random non-negative weights (with the same
    normalization/constraints as LDA weights) and recomputes the aggregated
    signals before running MI. It is **not** a permutation/null baseline.
    """

    n_shuffles: int = 0  # int >= 0; 0 disables
    seed: int = 0


@dataclass
class InformationBaselinesLabelShuffleNullParams(BaseConfig):
    """Permutation null baseline for MI/CMI terms (produces *_null metrics).

    This is a finite-sample bias / null diagnostic: it permutes the label vector
    (or, more generally, permutes one variable across samples) and recomputes MI/CMI
    using the same estimator selected by ``estimator.method``.

    You can control which metric groups receive a *_null counterpart via
    ``components``.

    - components=["basic"]: add *_null for basic MI terms (typically I(*;C))
    - components=["pairwise"]: add *_null for pairwise MI terms like I(E;I), I(E;Vout)
    - components=["conditional"]: add *_null for conditional MI terms like I(E;Vout|C)
    - components=["all"]: apply to all available groups
    """

    n_shuffles: int = 0  # int >= 0; 0 disables
    seed: int = 0
    components: list[str] = field(
        default_factory=lambda: ["basic"]
    )  # "basic", "pairwise", "conditional", "ablation_aligned", "layer_proxies", "upstream_unique_cmi", "all"


@dataclass
class InformationBaselinesParams(BaseConfig):
    """Baseline configuration for information analysis.

    All baselines use the same estimator selected by ``estimator.method``. They
    differ only in how surrogate data are generated:
    - random_weights: random-weight aggregation baseline (*_sh)
    - label_shuffle_null: label-permutation null (*_null)
    """

    random_weights: InformationBaselinesRandomWeightsParams = field(
        default_factory=InformationBaselinesRandomWeightsParams
    )
    label_shuffle_null: InformationBaselinesLabelShuffleNullParams = field(
        default_factory=InformationBaselinesLabelShuffleNullParams
    )


@dataclass
class InformationAnalysisParams(BaseConfig):
    """Parameters for information-theoretic analysis.

    Core string options (new grouped structure):
    - estimator.method: "kraskov", "decoder", "binned", "sklearn", "copula", "auto"
    - estimator.copula.type (method="copula"): "gaussian", "kernel", "dvc"
      (experimental; "dvc" is a placeholder and currently falls back to the simple copula path)
    - compute.computation_level: "single_branch", "layer_branch", "all_branch"
    - compute.branch_aggregation (layer_branch/all_branch): "mean", "multivariate", "sample"
    - pid.binarize_method: "median", "mean", "quantile" (PID is currently disabled)

    Baselines are computed with the same ``estimator.method``; baseline settings
    only control how surrogate data are constructed and how many repeats to run.

    Preferred (list-based) selectors:
    - selection.components: which families of metrics to compute
      - "basic": I(E;C), I(I;C), I(Vout;C)
      - "pairwise": I(E;I), I(E;Vout), I(I;Vout)
      - "conditional": conditional MI terms like I(E;C|I,Vb)
      - "pid": partial information decomposition (if available)
      - "ablation_aligned": extra conditional terms designed to mirror ablations,
        such as I(Vout;C|E), I(Vout;C|I), or I(Vout;C|E,I)
      - "layer_proxies": layer-total proxies derived from single-branch metrics
      - "upstream_unique_cmi": adds I(Vb;C|E,I) (can be expensive)
    - selection.scopes: which additional breakdowns to run
      - "layer": per-layer breakdown
      - "einet": separate excitatory vs inhibitory pathway summaries
      - "neuron": per-neuron breakdown (expensive)
    - selection.signal_variants: which signal variants to compute/plot
      - "network": DendriNet signals
      - "lda": *_lin variants using LDA projections
      - "random": *_sh random-weight baseline (requires baselines.random_weights.n_shuffles > 0)

    If any list-based selector is non-empty, it overrides the corresponding legacy
    boolean flags.

    Decoder-specific options (method="decoder"):
    - decoder_continuous_strategy: "none" (skip), "binned" (discretize+count), "gaussian" (Gaussian MI)
    - decoder_continuous_max_total_dim: maximum total dimension for continuous MI/CMI under decoder
    - decoder_gaussian_ridge: small diagonal ridge for covariance stability (gaussian strategy)

    Binned-specific options (method="binned"):
    - binning_strategy: "quantile" or "uniform"
    - binning_bias_correction: "none" or "miller_madow"
    """

    # ------------------------------------------------------------------
    # New (preferred) grouped structure
    # ------------------------------------------------------------------
    selection: InformationSelectionParams = field(
        default_factory=InformationSelectionParams
    )
    compute: InformationComputeParams = field(default_factory=InformationComputeParams)
    estimator: InformationEstimatorParams = field(
        default_factory=InformationEstimatorParams
    )
    baselines: InformationBaselinesParams = field(
        default_factory=InformationBaselinesParams
    )
    pid: InformationPIDParams = field(default_factory=InformationPIDParams)

    # ------------------------------------------------------------------
    # Legacy flat keys (DEPRECATED; still supported for backward compatibility).
    #
    # MIGRATION GUIDE -- use the grouped sub-dataclasses above instead:
    #   method, n_neighbors, n_bins, copula_type, decoder_*, binning_*
    #       -> estimator.method, estimator.kraskov.n_neighbors, etc.
    #   max_samples, include_vinf, normalize_mi, verbose, computation_level,
    #   branch_aggregation, per_neuron_analysis, per_einet_analysis,
    #   per_layer_analysis, compute_lda_weights
    #       -> compute.<field>
    #   components, scopes, views, signal_variants
    #       -> selection.<field>
    #   pid_binarize, pid_binarize_method, pid_binarize_threshold
    #       -> pid.<field>
    #   mi_null_shuffles, mi_null_seed, lda_n_shuffles
    #       -> baselines.<field>
    # ------------------------------------------------------------------

    # Estimation method: "kraskov", "decoder", "binned", "sklearn", "copula", "auto"
    method: str = "kraskov"

    # Optional: limit the number of samples to analyze
    max_samples: Optional[int] = None

    # Method-specific parameters
    n_neighbors: int = 10  # For Kraskov
    n_bins: int = 20  # For binned estimator + sklearn histogram
    copula_type: str = "gaussian"  # For copula: "gaussian", "kernel", "dvc"

    # Decoder (method="decoder") hyperparameters
    decoder_cv_folds: int = 5
    decoder_C: float = 1.0
    decoder_standardize: bool = True
    decoder_seed: int = 0
    decoder_continuous_strategy: str = "none"  # "none", "binned", "gaussian"
    decoder_continuous_max_total_dim: int = (
        2  # only used if decoder_continuous_strategy != "none"
    )
    decoder_gaussian_ridge: float = 1e-6  # covariance ridge for gaussian strategy

    # Binned (method="binned") hyperparameters
    binning_strategy: str = "quantile"  # "quantile" or "uniform"
    binning_bias_correction: str = "none"  # "none" or "miller_madow"

    # Optional: estimate finite-sample (estimator) bias via a shuffled-null baseline.
    #
    # If > 0, the analyzer will compute MI again after permuting the target across
    # samples (e.g. shuffling class labels) and report the null mean/std as a
    # diagnostic. This is separate from `lda_n_shuffles`, which creates random
    # *signal variants* (random-weight baselines).
    mi_null_shuffles: int = 0  # int >= 0; 0 disables
    mi_null_seed: int = 0  # RNG seed for the null permutations

    # Which metrics to compute
    compute_basic_mi: bool = True  # I(E;C), I(I;C), I(Vout;C)
    compute_pairwise_mi: bool = True  # I(E;I), I(E;Vout), I(I;Vout)
    compute_conditional_mi: bool = True  # I(E;I|C), I(E;Vout|C), etc.

    # New (preferred): list-based selection of analysis components.
    #
    # If non-empty, this overrides the individual compute_* boolean flags above.
    # Supported entries (synonyms are accepted by the analyzer):
    # - "basic" / "basic_mi"
    # - "pairwise" / "pairwise_mi"
    # - "conditional" / "conditional_mi"
    # - "pid"
    # - "ablation_aligned"
    # - "layer_proxies" / "layer_total_proxies"
    # - "upstream_unique_cmi"
    components: list[str] = field(default_factory=list)

    # Ablation-aligned information components (cheap extra CMI terms)
    # These are designed to better track "remove E/I/Vb/all" style ablations.
    compute_ablation_aligned_mi: bool = True

    # Optional extra upstream conditionals (can be expensive for high-dimensional synapse sets)
    # - Adds I(Vb;C|E,I) when Vb exists.
    compute_upstream_unique_cmi: bool = False

    # Layer-total proxies derived from single-branch MI (no high-dim multivariate MI)
    # Only used in computation_level="single_branch".
    # Produces additional per-layer metrics like:
    #   I(E;C|I,Vb)_sum, I(E;C|I,Vb)_union, I(E;C|I,Vb)_topK_sum, ...
    compute_layer_total_proxies: bool = True
    layer_total_topk: int = 10  # K used for *_topK_sum metrics

    # New (preferred): list-based selection of analysis "scopes" (breakdowns).
    #
    # If non-empty, this overrides the per_* boolean flags below.
    # Supported entries:
    # - "neuron" / "per_neuron"
    # - "einet" / "per_einet"
    # - "layer" / "per_layer"
    scopes: list[str] = field(default_factory=list)

    # Backward-compatible alias (deprecated): prefer `scopes`.
    views: list[str] = field(default_factory=list)

    # PID settings
    pid_binarize: bool = True
    pid_binarize_method: str = "median"  # "median", "mean", "quantile"
    pid_binarize_threshold: float = 0.5

    # Additional options
    include_vinf: bool = True  # Include soma activation
    normalize_mi: bool = False  # Normalize by entropy
    verbose: bool = False

    # Computation Level Control - determines the granularity of information analysis
    computation_level: str = (
        "single_branch"  # Options: "single_branch", "layer_branch", "all_branch"
    )
    # "single_branch": Individual branches (most granular and accurate)
    # "layer_branch": Layer-level aggregation (treats each layer as a unit)
    # "all_branch": Full network aggregation (most efficient)

    # Branch aggregation method - only used when computation_level is "layer_branch" or "all_branch"
    branch_aggregation: str = "mean"  # Options: "mean", "multivariate", "sample"

    # Per-neuron analysis - return information values per individual neuron
    per_neuron_analysis: bool = False  # If True, compute MI for each neuron separately

    # Per-EI-network analysis - analyze excitatory and inhibitory networks separately
    per_einet_analysis: bool = False  # If True, compute separate E and I network MI

    # Per-layer analysis - compute enhanced analysis for each layer separately
    per_layer_analysis: bool = False  # If True, compute per-layer enhanced analysis

    # Compute LDA-based versions of E, I, Vb
    compute_lda_weights: bool = False  # If True, compute E_lin, I_lin, Vb_lin using LDA

    # New (preferred): choose which "signal variants" to compute/plot.
    #
    # If non-empty, this overrides compute_lda_weights / lda_n_shuffles behavior:
    # - Always include "network" (DendriNet) metrics.
    # - Add "lda" to compute *_lin variants.
    # - Add "random" (or "shuffle") to compute *_sh variants (requires lda_n_shuffles > 0).
    signal_variants: list[str] = field(default_factory=list)

    # Number of random weight shuffles for LDA baseline comparison
    # If > 0, computes MI with random weights (same normalization as LDA) as a baseline
    lda_n_shuffles: int = 0


@dataclass
class InformationAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: InformationAnalysisParams = field(default_factory=InformationAnalysisParams)


@dataclass
class SynapseTurnoverAnalysisParams(BaseConfig):
    """Parameters for synapse turnover analysis.

    - snapshot_interval: how often to snapshot connectivity (in epochs).
    - synaptic_analysis: whether to compute connectivity summaries.
    - save_synaptic_report: whether to write a detailed report to disk.
    """

    snapshot_interval: int = 1  # Snapshot frequency in epochs
    synaptic_analysis: bool = True  # Enable synaptic connectivity analysis
    save_synaptic_report: bool = True  # Save synaptic analysis report


@dataclass
class SynapseTurnoverAnalysisConfig(BaseConfig):
    """Enable/disable synapse turnover analysis."""

    enabled: bool = False
    training: bool = False
    params: SynapseTurnoverAnalysisParams = field(
        default_factory=SynapseTurnoverAnalysisParams
    )


@dataclass
class NoisePerturbationAnalysisParams(BaseConfig):
    """Parameters for noise perturbation analysis.

    Noise settings:
    - uniform_noise: enable uniform noise sweeps.
    - uniform_magnitudes: list of magnitudes for uniform noise.
    - gaussian_noise: enable Gaussian noise sweeps.
    - gaussian_sdevs: list of standard deviations for Gaussian noise.
    - n_samples: number of noisy replicates per magnitude/sdev.
    - clamp_range: optional ``[min, max]`` output clamp. Set to ``None`` for
      unnormalized neural activity where noise should not be clipped to an
      image-like range.

    Metric toggles (same semantics as PerformanceAnalysisParams):
    - accuracy, auc, categorical_loglikelihood, mse, cosine_similarity
    """

    uniform_noise: bool = True
    uniform_magnitudes: list[float] = field(default_factory=lambda: [0.1, 0.2])
    gaussian_noise: bool = True
    gaussian_sdevs: list[float] = field(default_factory=lambda: [0.1, 0.2])
    n_samples: int = 10
    clamp_range: Optional[list[float]] = field(default_factory=lambda: [0.0, 1.0])
    accuracy: bool = True
    auc: bool = True
    categorical_loglikelihood: bool = True
    mse: bool = True
    cosine_similarity: bool = True
    seed: int = 0


@dataclass
class NoisePerturbationAnalysisConfig(BaseConfig):
    """Enable/disable noise perturbation analysis."""

    enabled: bool = False
    training: bool = False
    params: NoisePerturbationAnalysisParams = field(
        default_factory=NoisePerturbationAnalysisParams
    )


@dataclass
class AdversarialRobustnessAnalysisParams(BaseConfig):
    """Parameters for adversarial robustness (FGSM) evaluation."""

    epsilons: list[float] = field(
        default_factory=lambda: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
    )
    batch_size: int = 256


@dataclass
class AdversarialRobustnessAnalysisConfig(BaseConfig):
    """Enable/disable adversarial robustness analysis."""

    enabled: bool = False
    training: bool = False
    params: AdversarialRobustnessAnalysisParams = field(
        default_factory=AdversarialRobustnessAnalysisParams
    )


@dataclass
class AblationSelectionParams(BaseConfig):
    """Preferred (grouped) selectors for ablation analysis."""

    levels: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)


@dataclass
class AblationMethodsParams(BaseConfig):
    """Preferred (grouped) method settings for ablation analysis."""

    ablation_methods: list[str] = field(default_factory=lambda: ["lesion"])
    shuffle_seed: int = 0
    clamp_statistic: str = "mean"  # "mean" (supported), "median" (future)


@dataclass
class AblationAnalysisParams(BaseConfig):
    """Configuration for ablation / intervention analysis.

    Notes
    -----
    We support multiple ablation *methods*:
    - "lesion": structural ablation by zeroing the relevant connection weights
      (current behavior; can change baseline/gain and induce saturation/vanishing).
    - "shuffle": signal intervention that permutes the targeted signal across samples
      (destroys label-alignment while preserving marginal distribution).
    - "mean_clamp": signal intervention that replaces the targeted signal with its
      batch mean (preserves operating point; removes trial-to-trial variability).

    Core string options:
    - clamp_statistic: "mean" (supported), "median" (reserved)

    Preferred (list-based) selectors:
    - levels: what granularity to ablate
      - "layer": ablate whole dendritic layers
      - "compartment": ablate specific compartments (lesion-only)
    - targets: what signal/connection class to ablate
      - "all_synapses": excitatory + inhibitory together
      - "excitation": excitatory synapses only
      - "inhibition": inhibitory synapses only
      - "upstream": upstream branch input / Vb pathway
    - metrics: what to report
      - "accuracy", "auc", "categorical_loglikelihood", "mse", "cosine_similarity"
      - "pred_label_mi_bits": I(C; C_hat) in bits for predicted labels C_hat

    All ablation configuration is specified through the nested selection and methods
    fields below.
    """

    selection: AblationSelectionParams = field(default_factory=AblationSelectionParams)
    methods: AblationMethodsParams = field(default_factory=AblationMethodsParams)


@dataclass
class AblationAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: AblationAnalysisParams = field(default_factory=AblationAnalysisParams)


@dataclass
class WeightAnalysisParams(BaseConfig):
    """Parameters for weight analysis.

    Core string options:
    - computation_level: "single_branch", "layer_branch", "all_branch"
    - branch_aggregation (layer_branch/all_branch): "mean", "multivariate", "sample"

    Scope toggles:
    - per_layer_analysis, per_einet_analysis, per_neuron_analysis

    Content toggles:
    - synapse_weights, branch_weights
    - analyze_excitatory, analyze_inhibitory, analyze_branch_output

    Summary statistic toggles:
    - compute_mean, compute_variance, compute_percentiles, compute_min_max,
      compute_dendritic_strength
    """

    synapse_weights: bool = True  # Analyze excitatory and inhibitory synapse weights
    branch_weights: bool = True  # Analyze branch-to-output connection weights

    computation_level: str = (
        "single_branch"  # Options: "single_branch", "layer_branch", "all_branch"
    )
    branch_aggregation: str = "mean"  # Options: "mean", "multivariate", "sample"

    # Enhanced analysis modes
    per_layer_analysis: bool = True  # Analyze each layer separately (soma to distal)
    per_einet_analysis: bool = (
        True  # Separate excitatory and inhibitory network analysis
    )
    per_neuron_analysis: bool = False  # Analyze individual neurons (expensive)

    analyze_excitatory: bool = True  # Analyze excitatory synaptic weights
    analyze_inhibitory: bool = True  # Analyze inhibitory synaptic weights
    analyze_branch_output: bool = True  # Analyze branch-to-output weights

    compute_mean: bool = True  # Compute mean weights per branch
    compute_variance: bool = True  # Compute variance of weights per branch
    compute_percentiles: bool = True  # Compute percentiles (median, quartiles)
    compute_min_max: bool = True  # Compute min/max per layer
    compute_dendritic_strength: bool = (
        True  # Compute total dendritic strength per layer
    )

    weight_threshold: float = 1e-6  # Minimum weight to consider as "active"


@dataclass
class WeightAnalysisConfig(BaseConfig):
    """Enable/disable weight analysis."""

    enabled: bool = False
    training: bool = False
    params: WeightAnalysisParams = field(default_factory=WeightAnalysisParams)


@dataclass
class CompartmentStatisticsAnalysisParams(BaseConfig):
    """Parameters for compartment statistics analysis.

    Toggles:
    - synapse_weights / branch_weights: include weight statistics.
    - inputs / activations: include signal statistics.
    - global_analysis / layer_analysis / branch_analysis: which aggregation levels.
    - compute_mean / compute_variance / compute_percentiles / compute_min_max /
      compute_entropy: which summary statistics.
    - per_class_analysis: split statistics by class label.

    Sampling:
    - n_samples: optional limit on number of samples processed.
    """

    synapse_weights: bool = True  # Analyze excitatory and inhibitory synapse weights
    branch_weights: bool = True  # Analyze branch-to-output connection weights

    inputs: bool = True  # Analyze inputs
    activations: bool = True  # Analyze activations

    global_analysis: bool = True  # Analyze global statistics
    layer_analysis: bool = True  # Analyze layer statistics
    branch_analysis: bool = True  # Analyze branch statistics

    compute_mean: bool = True  # Compute mean weights per branch
    compute_variance: bool = True  # Compute variance of weights per branch
    compute_percentiles: bool = True  # Compute percentiles (median, quartiles)
    compute_min_max: bool = True  # Compute min/max per layer
    compute_entropy: bool = True  # Compute entropy per layer

    per_class_analysis: bool = False  # Analyze statistics separately for each class
    n_samples: Optional[int] = None  # Number of samples to analyze
    seed: int = 0


@dataclass
class CompartmentStatisticsAnalysisConfig(BaseConfig):
    """Enable/disable compartment statistics analysis."""

    enabled: bool = False
    training: bool = False
    params: CompartmentStatisticsAnalysisParams = field(
        default_factory=CompartmentStatisticsAnalysisParams
    )


@dataclass
class CompartmentSNRAnalysisParams(BaseConfig):
    """Parameters for compartment SNR analysis.

    Toggles which aggregation levels to compute:
    - input_analysis
    - global_analysis
    - layer_analysis
    - branch_analysis
    """

    input_analysis: bool = True  # Analyze input-level statistics
    input_capture_mode: str = "branch"  # "branch" or "network"
    global_analysis: bool = True  # Analyze global statistics
    layer_analysis: bool = True  # Analyze layer statistics
    branch_analysis: bool = True  # Analyze branch statistics

    global_aggregation: bool = True  # Aggregate global statistics
    layer_aggregation: bool = True  # Aggregate layer statistics
    branch_aggregation: bool = True  # Aggregate branch statistics

    epsilon: float = 1e-6


@dataclass
class CompartmentSNRAnalysisConfig(BaseConfig):
    """Enable/disable compartment SNR analysis."""

    enabled: bool = False
    training: bool = False
    params: CompartmentSNRAnalysisParams = field(
        default_factory=CompartmentSNRAnalysisParams
    )


@dataclass
class MultiplicativeGainAnalysisParams(BaseConfig):
    uniform_gain: bool = True
    gain_sampling: str = "log_uniform"
    gain_min: float = 0.0
    gain_max: float = 10.0
    n_steps: int = 10
    logspace: bool = False
    input_preprocess: str = (
        "none"  # "none", "relu", "softplus", "shift_min", "global_shift_min"
    )
    noise_model: str = "none"  # "none", "poisson"
    poisson_eps: float = 1e-6
    accuracy: bool = False
    auc: bool = False
    categorical_loglikelihood: bool = False
    mse: bool = False
    cosine_similarity: bool = False


@dataclass
class MultiplicativeGainAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: MultiplicativeGainAnalysisParams = field(
        default_factory=MultiplicativeGainAnalysisParams
    )


@dataclass
class CompositionalOODAnalysisParams(BaseConfig):
    """Metrics to score on one frozen compositional-OOD draw.

    Names match AblationAnalyzer: ``accuracy``, ``auc``,
    ``categorical_loglikelihood``, ``mse``, ``cosine_similarity``,
    ``pred_label_mi_bits``.
    """

    metrics: list[str] = field(default_factory=list)


@dataclass
class CompositionalOODAnalysisConfig(BaseConfig):
    """Enable/disable compositional OOD performance analysis."""

    enabled: bool = False
    training: bool = False
    params: CompositionalOODAnalysisParams = field(
        default_factory=CompositionalOODAnalysisParams
    )


@dataclass
class SynapticActivationAnalysisParams(BaseConfig):
    """Parameters for synaptic activation analysis.

    Toggles:
    - network_summary / layer_summary / branch_summary / raw_summary: what to report.
    - logspace: whether to plot/report in log-space.
    - downsample: whether to subsample activations for speed/memory.

    - samples: number of examples to keep when downsampling/raw summaries are enabled.
    """

    network_summary: bool = True
    layer_summary: bool = True
    branch_summary: bool = True
    raw_summary: bool = True
    logspace: bool = False
    downsample: bool = True
    samples: int = 5


@dataclass
class SynapticActivationAnalysisConfig(BaseConfig):
    """Enable/disable synaptic activation analysis."""

    enabled: bool = False
    training: bool = False
    params: SynapticActivationAnalysisParams = field(
        default_factory=SynapticActivationAnalysisParams
    )


@dataclass
class BranchActivationAnalysisParams(BaseConfig):
    """Parameters for branch activation analysis.

    Toggles:
    - synapse_activations / branch_activations: what signals to log.
    - network_summary / layer_summary / branch_summary / raw_summary: what to report.
    - logspace: whether to plot/report in log-space.
    - downsample: whether to subsample for speed/memory.

    - samples: number of examples to keep when downsampling/raw summaries are enabled.
    """

    synapse_activations: bool = True
    branch_activations: bool = True
    network_summary: bool = True
    layer_summary: bool = True
    branch_summary: bool = True
    raw_summary: bool = True
    logspace: bool = False
    downsample: bool = True
    samples: int = 5


@dataclass
class BranchActivationAnalysisConfig(BaseConfig):
    """Enable/disable branch activation analysis."""

    enabled: bool = False
    training: bool = False
    params: BranchActivationAnalysisParams = field(
        default_factory=BranchActivationAnalysisParams
    )


@dataclass
class LocalRuleComponentAnalysisParams(BaseConfig):
    """Parameters for LocalCA component trajectory analysis.

    The analyzer records the fast factors in the local rule by branch depth and
    synapse type: branch voltage, input resistance, E/I driving force, active
    presynaptic drive, conductance-form eligibility, rule-correct eligibility,
    optional scalar/per-soma broadcast error, and the resulting local update
    factor before optimizer-specific parameterization.

    - max_samples: deterministic cap on examples used per snapshot.
    - include_dendritic: also record branch-to-parent conductance terms.
    - include_broadcast: compute an approximate LocalCA broadcast from labels.
    - broadcast_mode: "scalar" or "per_soma" for the recorded error field.
    - error_mode: "auto", "ce", "mse", or "none".
    """

    max_samples: Optional[int] = 1024
    include_dendritic: bool = True
    include_broadcast: bool = True
    broadcast_mode: str = "scalar"
    error_mode: str = "auto"
    e_rev_exc: float = 1.0
    e_rev_inh: float = 0.0
    save_json: bool = True
    save_csv: bool = True
    append_training_csv: bool = True


@dataclass
class LocalRuleComponentAnalysisConfig(BaseConfig):
    """Enable/disable LocalCA local-rule component analysis."""

    enabled: bool = False
    training: bool = False
    params: LocalRuleComponentAnalysisParams = field(
        default_factory=LocalRuleComponentAnalysisParams
    )


@dataclass
class SingleLayerContributionAnalysisParams(BaseConfig):
    """Parameters for single-layer contribution analysis.

    Contribution modes:
    - excitation_contribution: effect of excitation alone
    - inhibition_contribution: effect of inhibition alone
    - both_contribution: combined E+I effect

    Metric toggles (same semantics as PerformanceAnalysisParams):
    - accuracy, auc, categorical_loglikelihood, mse, cosine_similarity
    """

    excitation_contribution: bool = True
    inhibition_contribution: bool = True
    both_contribution: bool = True
    accuracy: bool = True
    auc: bool = True
    categorical_loglikelihood: bool = True
    mse: bool = True
    cosine_similarity: bool = True


@dataclass
class SingleLayerContributionAnalysisConfig(BaseConfig):
    """Enable/disable single-layer contribution analysis."""

    enabled: bool = False
    training: bool = False
    params: SingleLayerContributionAnalysisParams = field(
        default_factory=SingleLayerContributionAnalysisParams
    )


@dataclass
class ReceptiveFieldAnalysisParams(BaseConfig):
    """Parameters for receptive field visualization.

    - rf_shape: optional (H, W) receptive field shape if known.
    - exc_rf_shape / inh_rf_shape: optional synapse-specific receptive-field
      shapes; fall back to rf_shape when omitted.
    - compute_activation_tuning: also summarize class tuning of branch signals.
    - compute_component_rfs: run compact E/I mean and conditioned component RF
      maps, matching the empirical paper's soma-level RF summary.
    - compute_rf_tuning: run the aggregate RF tuning,
      discriminability, entropy, and compartment activation analysis.
    """

    n_rows: int = 2
    n_cols: int = 2
    rf_shape: Optional[tuple[int, int]] = None
    exc_rf_shape: Optional[tuple[int, int]] = None
    inh_rf_shape: Optional[tuple[int, int]] = None
    fig_save_format: str = "png"
    compute_activation_tuning: bool = False
    compute_component_rfs: bool = False
    plot_component_rfs: bool = True
    compute_rf_tuning: bool = False
    plot_rf_tuning: bool = True
    rf_tuning_layer_index: int = 0
    n_class_examples: int = 5
    save_raw_data: bool = True
    load_raw_data: bool = False
    raw_filename: Optional[str] = None
    epsilon: float = 1e-8


@dataclass
class ReceptiveFieldAnalysisConfig(BaseConfig):
    """Enable/disable receptive field visualization."""

    enabled: bool = False
    training: bool = False
    params: ReceptiveFieldAnalysisParams = field(
        default_factory=ReceptiveFieldAnalysisParams
    )


@dataclass
class ReactivationDynamicsAnalysisParams(BaseConfig):
    """Parameters for the reactivation-dynamics analyzer.

    The analyzer snapshots each ``ParametricTanh`` layer per epoch,
    capturing the learnable ``(log_m, b)`` parameters together with
    statistics of the pre-reactivation voltage ``V`` and the post-gate
    output ``r``. ``max_samples`` caps the batch size drawn from the
    training dataset to keep the analyzer lightweight.
    """

    max_samples: int = 1024


@dataclass
class ReactivationDynamicsAnalysisConfig(BaseConfig):
    """Enable/disable reactivation-dynamics analysis.

    When ``enabled=True`` and ``training=True``, the analyzer fires each
    epoch (one JSON file per epoch under
    ``<save_root>/reactivation_dynamics/epochs/``). This provides a direct
    record of gate parameters, voltage statistics, and output occupancy
    throughout training.
    """

    enabled: bool = False
    training: bool = True
    params: ReactivationDynamicsAnalysisParams = field(
        default_factory=ReactivationDynamicsAnalysisParams
    )


@dataclass
class CorrelationSelectionParams(BaseConfig):
    """Preferred (grouped) selectors for correlation analysis."""

    components: list[str] = field(default_factory=list)
    pairs: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)


@dataclass
class CorrelationComputeParams(BaseConfig):
    """Runtime/aggregation controls for correlation analysis."""

    computation_level: str = (
        "layer_branch"  # "synaptic", "single_branch", "layer_branch", "all_branch"
    )
    max_samples: Optional[int] = 5000
    batch_process: bool = True
    # Breakdown/visualization level (overridden by selection.scopes if provided)
    per_layer_analysis: bool = True
    per_einet_analysis: bool = True
    per_neuron_analysis: bool = False


@dataclass
class CorrelationSamplingParams(BaseConfig):
    """Sampling controls (used only when computation_level="synaptic")."""

    n_synapse_pairs: int = 10000
    sampling_strategy: str = "stratified"  # "random", "stratified", "importance"
    stratified_categories: list[str] = field(
        default_factory=lambda: ["within_branch", "within_layer", "across_layer"]
    )
    max_pairs_per_category: int = 1000


@dataclass
class CorrelationAnalysisParams(BaseConfig):
    """Parameters for correlation analysis.

    Core string options:
    - computation_level:
      - "synaptic": individual synapse correlations (uses sampling)
      - "single_branch": branch-by-branch correlations
      - "layer_branch": all branches in a layer together
      - "all_branch": full network aggregation
    - sampling_strategy (synaptic only): "random", "stratified", "importance"

    Preferred (list-based) selectors:
    - components: which correlation components to compute
      - "total": correlations across all samples
      - "noise": within-class (stimulus-independent) correlations
      - "signal": between-class difference (binary-only currently)
      - "tuning": tuning/selectivity summary across classes
    - pairs: which variable pairs to analyze
      - "EE", "II", "EI"
      - "E_output", "I_output", "output_output"
    - scopes: which breakdowns to include in outputs
      - "layer", "einet", "neuron"

    All ablation configuration is specified through the nested selection and methods
    fields below.
    """

    # ------------------------------------------------------------------
    # New (preferred) grouped structure
    # ------------------------------------------------------------------
    selection: CorrelationSelectionParams = field(
        default_factory=CorrelationSelectionParams
    )
    compute: CorrelationComputeParams = field(default_factory=CorrelationComputeParams)
    sampling: CorrelationSamplingParams = field(
        default_factory=CorrelationSamplingParams
    )

    # ------------------------------------------------------------------
    # Legacy flat keys (deprecated; still supported for backward compatibility)
    # ------------------------------------------------------------------

    # Computation level (aligned with information analysis)
    # "synaptic": Individual synapse correlations (most detailed, uses sampling)
    # "single_branch": Branch-by-branch analysis (detailed)
    # "layer_branch": All branches in a layer together (balanced)
    # "all_branch": Full network aggregation (most efficient)
    computation_level: str = "layer_branch"

    # New (preferred): list-based selection of analysis options.
    # If non-empty, these override the boolean flags below.
    components: list[str] = field(default_factory=list)
    pairs: list[str] = field(default_factory=list)
    scopes: list[str] = field(default_factory=list)

    # Correlation types to compute
    compute_noise_correlations: bool = (
        True  # Within-class (stimulus-independent) correlations
    )
    compute_signal_correlations: bool = (
        True  # Between-class (stimulus-driven) correlations
    )
    compute_total_correlations: bool = True  # Across all data
    compute_tuning_curves: bool = (
        True  # Class selectivity (mean difference between classes)
    )

    # Which correlation pairs to analyze
    analyze_EE: bool = True  # Excitatory-Excitatory correlations
    analyze_II: bool = True  # Inhibitory-Inhibitory correlations
    analyze_EI: bool = True  # Excitatory-Inhibitory cross-correlations
    analyze_E_output: bool = True  # Excitatory-Output (Vout) correlations
    analyze_I_output: bool = True  # Inhibitory-Output (Vout) correlations
    analyze_output_output: bool = True  # Output-Output (Vout-Vout) correlations

    # Hierarchical analysis options (similar to information analysis)
    per_layer_analysis: bool = True  # Analyze each layer separately
    per_einet_analysis: bool = True  # Separate E and I network analysis
    per_neuron_analysis: bool = False  # Analyze individual neurons (expensive)

    # Sampling parameters (only used when computation_level="synaptic")
    n_synapse_pairs: int = 10000  # Number of synapse pairs to sample
    sampling_strategy: str = "stratified"  # "random", "stratified", "importance"
    stratified_categories: list[str] = field(
        default_factory=lambda: ["within_branch", "within_layer", "across_layer"]
    )
    max_pairs_per_category: int = (
        1000  # Maximum pairs per category in stratified sampling
    )

    # Performance optimization
    max_samples: Optional[int] = 5000  # Limit data samples for faster computation
    batch_process: bool = True  # Process correlations in batches for memory efficiency


@dataclass
class CorrelationAnalysisConfig(BaseConfig):
    """Enable/disable correlation analysis."""

    enabled: bool = False
    training: bool = False
    params: CorrelationAnalysisParams = field(default_factory=CorrelationAnalysisParams)


@dataclass
class RoutingAnalysisConfig(BaseConfig):
    """Configuration for branch routing (alpha) analysis in recurrent models.

    Only applicable to recurrent EINetwork cores with store_routing=True.
    """

    enabled: bool = False
    training: bool = False


@dataclass
class InhibitorySpecializationAnalysisParams(BaseConfig):
    """Parameters for inhibitory-specialization (gate-index) analysis.

    - gate_threshold: absolute mean-G cutoff separating gate/context-specialists
      from diffuse neurons (per-I view) and soma/distal-dominant from balanced
      pairs (per-pair view).
    - std_threshold: std-G cutoff above which a neuron is classified
      mixed-conditional rather than a consistent specialist.
    - save_figures: whether to write a PDF alongside the JSON results.
    """

    gate_threshold: float = 0.1
    std_threshold: float = 0.15
    save_figures: bool = True


@dataclass
class InhibitorySpecializationAnalysisConfig(BaseConfig):
    """Enable/disable inhibitory-specialization analysis.

    Applicable only to recurrent E-I dendritic cores that carry at least one
    layer with a local inhibitory pool wired through ``branch_rec_inhibition``.
    Silently skipped for feedforward-only or baseline RNN cores.
    """

    enabled: bool = False
    training: bool = False
    params: InhibitorySpecializationAnalysisParams = field(
        default_factory=InhibitorySpecializationAnalysisParams
    )


@dataclass
class JacobianSpectrumAnalysisParams(BaseConfig):
    """Parameters for per-level Jacobian / recurrent-spectrum analysis.

    - top_k: number of leading singular values to keep per branch layer.
    - save_figures: whether to write a PDF alongside the JSON results.
    """

    top_k: int = 16
    save_figures: bool = True


@dataclass
class JacobianSpectrumAnalysisConfig(BaseConfig):
    """Enable/disable per-level Jacobian / recurrent-spectrum analysis.

    Reads the effective ``branch_recurrent`` / ``branch_rec_inhibition``
    weights of the excitatory population at each branch level and reports
    their top singular values. Skipped on baseline RNN cores and on layers
    without recurrent compartments.
    """

    enabled: bool = False
    training: bool = False
    params: JacobianSpectrumAnalysisParams = field(
        default_factory=JacobianSpectrumAnalysisParams
    )


@dataclass
class DendriticTimetracesAnalysisParams(BaseConfig):
    """Parameters for dendritic-time-trace simulation.

    The analyzer simulates ``V_l(t)`` and ``hat V_l(t)`` for the trained
    τ and reactivation parameters on a synthetic two-pulse stimulus.

    - n_timesteps: trial length.
    - stim1_window / stim2_window: ``(start, end)`` index tuples in [0, T).
      Set either to ``None`` to disable the corresponding pulse.
    - stim1_amp / stim2_amp: amplitudes of the two pulses.
    - response_window: optional shaded region marking the response period.
    - noise_std: white noise added to the input; 0 disables.
    - seed: RNG seed for the noise tensor (deterministic).
    - save_figures: whether to write a PDF per E-I layer.
    """

    n_timesteps: int = 90
    stim1_window: Optional[tuple[int, int]] = (5, 25)
    stim2_window: Optional[tuple[int, int]] = (55, 75)
    response_window: Optional[tuple[int, int]] = (75, 90)
    stim1_amp: float = 0.8
    stim2_amp: float = 0.6
    noise_std: float = 0.05
    seed: int = 0
    save_figures: bool = True


@dataclass
class DendriticTimetracesAnalysisConfig(BaseConfig):
    """Enable/disable dendritic-time-trace simulation.

    Simulates the learned per-level cascade dynamics (τ, m, b) on a
    synthetic two-pulse trial. Skipped on baseline RNN cores or on
    dendritic populations without ``log_tau`` exposed.
    """

    enabled: bool = False
    training: bool = False
    params: DendriticTimetracesAnalysisParams = field(
        default_factory=DendriticTimetracesAnalysisParams
    )


@dataclass
class SpikeTrainAnalysisParams(BaseConfig):
    """Parameters for spike-train analysis of LIF recurrent populations.

    - dt: seconds or arbitrary time units per recurrent step; rates are
      reported as spikes per ``dt``.
    - threshold: spike binarization threshold for recorded LIF outputs.
    - max_samples: optional cap on dataset examples for analysis.
    - include_raster: include compact event lists for the first
      ``max_raster_samples`` examples.
    - include_per_timestep_rate: include a population firing-rate trace.
    """

    dt: float = 1.0
    threshold: float = 0.5
    max_samples: Optional[int] = 512
    include_raster: bool = True
    max_raster_samples: int = 8
    include_per_timestep_rate: bool = True


@dataclass
class SpikeTrainAnalysisConfig(BaseConfig):
    """Enable/disable spike-train analysis for opt-in LIF recurrent cores."""

    enabled: bool = False
    training: bool = False
    params: SpikeTrainAnalysisParams = field(default_factory=SpikeTrainAnalysisParams)


@dataclass
class AnalysisConfig(BaseConfig):
    """Top-level analysis configuration container.

    Each field corresponds to an analysis module under `analysis.*` in YAML.
    """

    runtime: AnalysisRuntimeConfig = field(default_factory=AnalysisRuntimeConfig)

    performance_analysis: PerformanceAnalysisConfig = field(
        default_factory=PerformanceAnalysisConfig
    )
    synapse_turnover_analysis: SynapseTurnoverAnalysisConfig = field(
        default_factory=SynapseTurnoverAnalysisConfig
    )
    noise_perturbation_analysis: NoisePerturbationAnalysisConfig = field(
        default_factory=NoisePerturbationAnalysisConfig
    )
    single_layer_contribution: SingleLayerContributionAnalysisConfig = field(
        default_factory=SingleLayerContributionAnalysisConfig
    )
    ablation_analysis: AblationAnalysisConfig = field(
        default_factory=AblationAnalysisConfig
    )
    weight_analysis: WeightAnalysisConfig = field(default_factory=WeightAnalysisConfig)

    compartment_statistics_analysis: CompartmentStatisticsAnalysisConfig = field(
        default_factory=CompartmentStatisticsAnalysisConfig
    )

    compartment_snr_analysis: CompartmentSNRAnalysisConfig = field(
        default_factory=CompartmentSNRAnalysisConfig
    )
    multiplicative_gain_analysis: MultiplicativeGainAnalysisConfig = field(
        default_factory=MultiplicativeGainAnalysisConfig
    )
    compositional_ood_analysis: CompositionalOODAnalysisConfig = field(
        default_factory=CompositionalOODAnalysisConfig
    )
    synaptic_activation_analysis: SynapticActivationAnalysisConfig = field(
        default_factory=SynapticActivationAnalysisConfig
    )
    branch_activation_analysis: BranchActivationAnalysisConfig = field(
        default_factory=BranchActivationAnalysisConfig
    )
    local_rule_component_analysis: LocalRuleComponentAnalysisConfig = field(
        default_factory=LocalRuleComponentAnalysisConfig
    )
    information_analysis: InformationAnalysisConfig = field(
        default_factory=InformationAnalysisConfig
    )
    receptive_field_analysis: ReceptiveFieldAnalysisConfig = field(
        default_factory=ReceptiveFieldAnalysisConfig
    )
    reactivation_dynamics_analysis: ReactivationDynamicsAnalysisConfig = field(
        default_factory=ReactivationDynamicsAnalysisConfig
    )
    correlation_analysis: CorrelationAnalysisConfig = field(
        default_factory=CorrelationAnalysisConfig
    )
    routing_analysis: RoutingAnalysisConfig = field(
        default_factory=RoutingAnalysisConfig
    )
    inhibitory_specialization_analysis: InhibitorySpecializationAnalysisConfig = field(
        default_factory=InhibitorySpecializationAnalysisConfig
    )
    jacobian_spectrum_analysis: JacobianSpectrumAnalysisConfig = field(
        default_factory=JacobianSpectrumAnalysisConfig
    )
    dendritic_timetraces_analysis: DendriticTimetracesAnalysisConfig = field(
        default_factory=DendriticTimetracesAnalysisConfig
    )
    spike_train_analysis: SpikeTrainAnalysisConfig = field(
        default_factory=SpikeTrainAnalysisConfig
    )
    adversarial_robustness_analysis: AdversarialRobustnessAnalysisConfig = field(
        default_factory=AdversarialRobustnessAnalysisConfig
    )

    def any_enabled(self) -> bool:
        """Return whether the final training analysis gate should run."""
        return any(
            bool(getattr(getattr(self, spec.config_field, None), "enabled", False))
            for spec in get_registered_analyzers().values()
        )
