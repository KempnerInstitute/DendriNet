from dataclasses import dataclass, field
from typing import Optional

from dendritic_modeling.config.base import BaseConfig


@dataclass
class PerformanceAnalysisParams(BaseConfig):
    accuracy: bool = True
    auc: bool = True
    categorical_loglikelihood: bool = True
    mse: bool = False
    cosine_similarity: bool = False


@dataclass
class PerformanceAnalysisConfig(BaseConfig):
    enabled: bool = True
    training: bool = False
    params: PerformanceAnalysisParams = field(default_factory=PerformanceAnalysisParams)


@dataclass
class InformationAnalysisParams(BaseConfig):
    """Configuration for integrated information analysis."""

    # Estimation method: "kraskov", "sklearn", "copula", "auto"
    method: str = "kraskov"

    # Optional: limit the number of samples to analyze
    max_samples: Optional[int] = None

    # Method-specific parameters
    n_neighbors: int = 10  # For Kraskov
    n_bins: int = 20  # For sklearn histogram
    copula_type: str = "gaussian"  # For copula: "gaussian", "kernel", "dvc"

    # Which metrics to compute
    compute_basic_mi: bool = True  # I(E;C), I(I;C), I(Vout;C)
    compute_pairwise_mi: bool = True  # I(E;I), I(E;Vout), I(I;Vout)
    compute_conditional_mi: bool = True  # I(E;I|C), I(E;Vout|C), etc.
    compute_pid: bool = False  # Partial Information Decomposition

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

    # New (preferred): list-based selection of analysis "views".
    #
    # If non-empty, this overrides the per_* boolean flags below.
    # Supported entries:
    # - "neuron" / "per_neuron"
    # - "einet" / "per_einet"
    # - "layer" / "per_layer"
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
    enabled: bool = True
    training: bool = False
    params: InformationAnalysisParams = field(default_factory=InformationAnalysisParams)


@dataclass
class SynapseTurnoverAnalysisParams(BaseConfig):
    """Parameters for synapse turnover analysis."""

    snapshot_interval: int = 1  # Snapshot frequency in epochs
    synaptic_analysis: bool = True  # Enable synaptic connectivity analysis
    save_synaptic_report: bool = True  # Save synaptic analysis report


@dataclass
class SynapseTurnoverAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: SynapseTurnoverAnalysisParams = field(
        default_factory=SynapseTurnoverAnalysisParams
    )


@dataclass
class NoisePerturbationAnalysisParams(BaseConfig):
    uniform_noise: bool = True
    uniform_magnitudes: list[float] = field(default_factory=lambda: [0.1, 0.2])
    gaussian_noise: bool = True
    gaussian_sdevs: list[float] = field(default_factory=lambda: [0.1, 0.2])
    n_samples: int = 10
    accuracy: bool = True
    auc: bool = True
    categorical_loglikelihood: bool = True
    mse: bool = True
    cosine_similarity: bool = True


@dataclass
class NoisePerturbationAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: NoisePerturbationAnalysisParams = field(
        default_factory=NoisePerturbationAnalysisParams
    )


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
    """

    # Which ablation/intervention methods to run. Example:
    # ["lesion", "shuffle", "mean_clamp"]
    ablation_methods: list[str] = field(default_factory=lambda: ["lesion"])

    # Settings for signal interventions
    shuffle_seed: int = 0
    clamp_statistic: str = "mean"  # "mean" (supported), "median" (future)

    # New (preferred): list-based selection of what to run.
    #
    # If non-empty, these override the individual boolean flags below.
    # - levels: ["layer", "compartment"]
    # - targets: ["excitation", "inhibition", "upstream", "all_synapses"]
    # - metrics: ["accuracy", "auc", "categorical_loglikelihood", "mse", "cosine_similarity", "pred_label_mi_bits"]
    levels: list[str] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)

    layer_ablation: bool = True
    compartment_ablation: bool = True
    ablate_all_synapses: bool = True
    ablate_excitation: bool = True
    ablate_inhibition: bool = True
    ablate_upstream: bool = True

    # Optional: compute an information-theoretic score on predictions.
    # This is I(C; Ĉ) where Ĉ is the argmax predicted class (in bits).
    # It is NOT I(Vinf;C), but it's a useful "how much label information is in predictions"
    # measure that can separate baseline/gain effects from label-aligned signal effects.
    compute_pred_label_mi: bool = False

    accuracy: bool = True
    auc: bool = True
    categorical_loglikelihood: bool = True
    mse: bool = True
    cosine_similarity: bool = True


@dataclass
class AblationAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: AblationAnalysisParams = field(default_factory=AblationAnalysisParams)


@dataclass
class WeightAnalysisParams(BaseConfig):
    """Enhanced configuration for unified weight analysis."""

    synapse_weights: bool = True  # Analyze excitatory and inhibitory synapse weights
    branch_weights: bool = True  # Analyze branch-to-output connection weights

    computation_level: str = (
        "single_branch"  # Options: "single_branch", "layer_branch", "all_branch"
    )
    branch_aggregation: str = "mean"  # Options: "mean", "multivariate", "sample"

    # Enhanced analysis modes
    per_layer_analysis: bool = True  # Analyze each layer separately (soma→distal)
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
    enabled: bool = False
    training: bool = False
    params: WeightAnalysisParams = field(default_factory=WeightAnalysisParams)


@dataclass
class CompartmentStatisticsAnalysisParams(BaseConfig):
    """Enhanced configuration for compartment statistics analysis."""

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


@dataclass
class CompartmentStatisticsAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: CompartmentStatisticsAnalysisParams = field(
        default_factory=CompartmentStatisticsAnalysisParams
    )


@dataclass
class CompartmentSNRAnalysisParams(BaseConfig):
    """Enhanced configuration for compartment SNR analysis."""

    global_analysis: bool = True  # Analyze global statistics
    layer_analysis: bool = True  # Analyze layer statistics
    branch_analysis: bool = True  # Analyze branch statistics


@dataclass
class CompartmentSNRAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: CompartmentSNRAnalysisParams = field(
        default_factory=CompartmentSNRAnalysisParams
    )


@dataclass
class SynapticActivationAnalysisParams(BaseConfig):
    network_summary: bool = True
    layer_summary: bool = True
    branch_summary: bool = True
    raw_summary: bool = True
    logspace: bool = False
    downsample: bool = True
    samples: int = 5


@dataclass
class SynapticActivationAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: SynapticActivationAnalysisParams = field(
        default_factory=SynapticActivationAnalysisParams
    )


@dataclass
class BranchActivationAnalysisParams(BaseConfig):
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
    enabled: bool = False
    training: bool = False
    params: BranchActivationAnalysisParams = field(
        default_factory=BranchActivationAnalysisParams
    )


@dataclass
class SingleLayerContributionAnalysisParams(BaseConfig):
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
    enabled: bool = False
    training: bool = False
    params: SingleLayerContributionAnalysisParams = field(
        default_factory=SingleLayerContributionAnalysisParams
    )


@dataclass
class ReceptiveFieldAnalysisParams(BaseConfig):
    n_rows: int = 2
    n_cols: int = 2
    exc_rf_shape: Optional[tuple[int, int]] = None
    inh_rf_shape: Optional[tuple[int, int]] = None


@dataclass
class ReceptiveFieldAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: ReceptiveFieldAnalysisParams = field(
        default_factory=ReceptiveFieldAnalysisParams
    )


@dataclass
class CorrelationAnalysisParams(BaseConfig):
    # Computation level (aligned with information analysis)
    # "synaptic": Individual synapse correlations (most detailed, uses sampling)
    # "single_branch": Branch-by-branch analysis (detailed)
    # "layer_branch": All branches in a layer together (balanced)
    # "all_branch": Full network aggregation (most efficient)
    computation_level: str = "layer_branch"

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
    analyze_EE: bool = True  # Excitatory-Excitatory correlations  # noqa: N815
    analyze_II: bool = True  # Inhibitory-Inhibitory correlations  # noqa: N815
    analyze_EI: bool = True  # Excitatory-Inhibitory cross-correlations  # noqa: N815
    analyze_E_output: bool = True  # Excitatory-Output (Vout) correlations  # noqa: N815
    analyze_I_output: bool = True  # Inhibitory-Output (Vout) correlations  # noqa: N815
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
    enabled: bool = False
    training: bool = False
    params: CorrelationAnalysisParams = field(default_factory=CorrelationAnalysisParams)


@dataclass
class AnalysisConfig(BaseConfig):
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
    synaptic_activation_analysis: SynapticActivationAnalysisConfig = field(
        default_factory=SynapticActivationAnalysisConfig
    )
    branch_activation_analysis: BranchActivationAnalysisConfig = field(
        default_factory=BranchActivationAnalysisConfig
    )
    information_analysis: InformationAnalysisConfig = field(
        default_factory=InformationAnalysisConfig
    )
    receptive_field_analysis: ReceptiveFieldAnalysisConfig = field(
        default_factory=ReceptiveFieldAnalysisConfig
    )
    correlation_analysis: CorrelationAnalysisConfig = field(
        default_factory=CorrelationAnalysisConfig
    )
