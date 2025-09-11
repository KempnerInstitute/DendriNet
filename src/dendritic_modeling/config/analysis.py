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


@dataclass
class InformationAnalysisConfig(BaseConfig):
    enabled: bool = True
    training: bool = False
    params: InformationAnalysisParams = field(default_factory=InformationAnalysisParams)


@dataclass
class SynapseTurnoverAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False


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
    layer_ablation: bool = True
    compartment_ablation: bool = True
    ablate_all_synapses: bool = True
    ablate_excitation: bool = True
    ablate_inhibition: bool = True
    ablate_upstream: bool = True
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
    synapse_weights: bool = True
    branch_weights: bool = True


@dataclass
class WeightAnalysisConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: WeightAnalysisParams = field(default_factory=WeightAnalysisParams)


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
class SynapticWeightExpectationParams(BaseConfig):
    """Configuration for synaptic weight expectation analysis."""

    # Computation Level Control (similar to information analysis)
    computation_level: str = (
        "single_branch"  # Options: "single_branch", "layer_branch", "all_branch"
    )
    branch_aggregation: str = "mean"  # Options: "mean", "multivariate", "sample"

    # Enhanced analysis modes (similar to information analysis)
    per_layer_analysis: bool = False  # Analyze each layer separately
    per_einet_analysis: bool = (
        False  # Separate excitatory and inhibitory network analysis
    )
    per_neuron_analysis: bool = False  # Analyze individual neurons (expensive)

    # Weight types to analyze
    analyze_excitatory: bool = True  # Analyze excitatory synaptic weights
    analyze_inhibitory: bool = True  # Analyze inhibitory synaptic weights
    analyze_branch_output: bool = False  # Analyze branch-to-output weights

    # Statistical measures to compute
    compute_mean: bool = True  # Compute mean weights per branch
    compute_variance: bool = True  # Compute variance/std of weights per branch
    compute_percentiles: bool = False  # Compute percentiles (median, quartiles)

    # Weight filtering
    weight_threshold: float = 1e-6  # Minimum weight to consider as "active"


@dataclass
class SynapticWeightExpectationConfig(BaseConfig):
    enabled: bool = False
    training: bool = False
    params: SynapticWeightExpectationParams = field(
        default_factory=SynapticWeightExpectationParams
    )


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
    synaptic_activation_analysis: SynapticActivationAnalysisConfig = field(
        default_factory=SynapticActivationAnalysisConfig
    )
    branch_activation_analysis: BranchActivationAnalysisConfig = field(
        default_factory=BranchActivationAnalysisConfig
    )
    information_analysis: InformationAnalysisConfig = field(
        default_factory=InformationAnalysisConfig
    )
    correlation_analysis: CorrelationAnalysisConfig = field(
        default_factory=CorrelationAnalysisConfig
    )
    synaptic_weight_expectation: SynapticWeightExpectationConfig = field(
        default_factory=SynapticWeightExpectationConfig
    )
