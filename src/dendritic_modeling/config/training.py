from dataclasses import dataclass, field
from typing import Any, Optional

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.regularization import (
    PruningConfig,
    RegularizationConfig,
    SelectiveConfig as _SelectiveConfig,
)

# Backward-compatible re-export: downstream code imports SelectiveConfig
# from config.training in several places.
SelectiveConfig = _SelectiveConfig


@dataclass
class ParamGroupsConfig(BaseConfig):
    """Parameter groups with different learning rates."""

    lr: float = 0.01
    split_params: bool = True
    topk_lr: float = 0.001
    blocklinear_lr: float = 0.0001
    reactivation_lr: float = 0.0001
    decoder_lr: float = 0.001


@dataclass
class CommonTrainerConfig(BaseConfig):
    """Common training parameters shared across strategies."""

    epochs: int = 1000
    batch_size: int = 512
    shuffle: bool = True
    grad_clip_value: float = 2.0
    loss_function: str = "ce"
    early_stopping: bool = True
    patience: int = 500
    load_best_state_dict: bool = True
    save_path: Optional[str] = None
    plot_losses: bool = True
    suppress_prints: bool = False
    print_every: int = 10
    use_amp: bool = False
    enable_nan_checking: bool = False
    enable_adaptive_clipping: bool = False
    weight_decay_rate: float = 0.01
    weight_boosting: bool = False

    param_groups: ParamGroupsConfig = field(default_factory=ParamGroupsConfig)


@dataclass
class StandardStrategyConfig(BaseConfig):
    """Standard backpropagation strategy configuration."""

    # Uses common config only


@dataclass
class MultiStageStrategyConfig(BaseConfig):
    """Multi-stage training strategy configuration."""

    analyze_between_stages: bool = True
    use_best_from_each_stage: bool = True
    continue_on_failure: bool = False

    stages: list[dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "learning_strategy": "standard",
                "epochs": 100,
                "batch_size": 512,
                "save_checkpoint": False,
                "param_groups": {
                    "lr": 0.01,
                    "split_params": True,
                    "topk_lr": 0.001,
                    "blocklinear_lr": 0.0001,
                    "reactivation_lr": 0.0001,
                    "decoder_lr": 0.001,
                },
                "trainer_config": {
                    "loss_function": "ce",
                    "early_stopping": True,
                    "patience": 500,
                },
            },
            {
                "learning_strategy": "local_ca",
                "epochs": 100,
                "batch_size": 512,
                "reset_optimizer": False,
                "save_checkpoint": False,
                "param_groups": {
                    "lr": 0.0001,
                    "split_params": True,
                    "topk_lr": 0.0001,
                    "blocklinear_lr": 0.00001,
                    "reactivation_lr": 0.00001,
                    "decoder_lr": 0.001,
                },
                "trainer_config": {"loss_function": "ce", "early_stopping": False},
            },
        ]
    )


@dataclass
class LocalCAStrategyConfig(BaseConfig):
    """Local credit assignment strategy configuration."""

    rule_variant: str = "4f"
    error_mode: str = "auto"
    error_broadcast_mode: str = "scalar"
    ema_alpha: float = 0.1
    use_conductance_scaling: bool = True
    use_driving_force: bool = True
    dynamics_mode: str = "auto"  # "auto", "conductance", "additive"
    theta: float = 0.0
    e_rev_exc: float = 1.0
    e_rev_inh: float = 0.0
    clip_grad_value: float = 5.0
    normalize_by_batch: bool = False

    phi_mode: str = "conditional"
    phi_ridge_lambda: float = 0.001
    rho_estimator: str = "ema"
    phi_estimator: str = "conditional_ema"
    augment_k: int = 8
    augment_noise_sigma: float = 0.01
    rls_forgetting: float = 0.99
    update_inactive_weights: bool = False

    hsic_enabled: bool = True
    hsic_weight: float = 0.01
    hsic_self_weight: float = 0.3
    hsic_target_weight: float = 0.3
    hsic_target_source: str = "labels"
    hsic_kernel: str = "rbf"
    hsic_sigma: float = 1.0
    hsic_degree: int = 2
    hsic_coef0: float = 1.0
    hsic_grad_clip_value: float = 0.1
    hsic_warmup_epochs: int = 5
    hsic_apply_last_layer_only: bool = False

    rho_mode: str = "dot"
    encoder_update_mode: str = "none"
    decoder_update_mode: str = "backprop"
    update_reactivation: bool = True
    freeze_encoder_epochs: int = 0
    freeze_reactivation_epochs: int = 20
    freeze_decoder_epochs: int = 0
    training_schedule: list = field(default_factory=list)
    layer_wise_rho_scale: float = 0.0
    layer_wise_phi_scale: float = 0.0


@dataclass
class VoltageStabilizationStrategyConfig(BaseConfig):
    """Voltage stabilization strategy configuration."""

    pretrain_epochs: int = 100
    stabilize_mode: str = "vout"
    freeze_reactivation: bool = True
    target_voltage: float = 0.5
    loss_metric: str = "mse"


@dataclass
class HomeostaticControlStrategyConfig(BaseConfig):
    """Homeostatic control strategy configuration."""

    pretrain_epochs: int = 100
    stabilize_mode: str = "vinf"
    freeze_reactivation: bool = False
    target_voltage: float = 0.5
    loss_metric: str = "mse"


@dataclass
class FreezeLayersStrategyConfig(BaseConfig):
    """Freeze layers strategy configuration."""

    pretrain_epochs: int = 10
    reverse_training: bool = False
    epochs_per_layer: int = 50
    final_tune_epochs: int = 10


@dataclass
class FreezeBranchesStrategyConfig(BaseConfig):
    """Freeze branches strategy configuration."""

    pretrain_epochs: int = 10
    reverse_training: bool = False
    epochs_per_branch: int = 50
    final_tune_epochs: int = 10


@dataclass
class FreezeBranchKLStrategyConfig(BaseConfig):
    """Freeze branch KL strategy configuration."""

    pretrain_epochs: int = 10
    reverse_training: bool = False
    epochs_per_branch: int = 50
    final_tune_epochs: int = 10
    kl_weight: float = 0.1


@dataclass
class TrainingStrategiesConfig(BaseConfig):
    """Training strategies configuration."""

    standard: StandardStrategyConfig = field(default_factory=StandardStrategyConfig)
    multi_stage: MultiStageStrategyConfig = field(
        default_factory=MultiStageStrategyConfig
    )
    local_ca: LocalCAStrategyConfig = field(default_factory=LocalCAStrategyConfig)
    voltage_stabilization: VoltageStabilizationStrategyConfig = field(
        default_factory=VoltageStabilizationStrategyConfig
    )
    homeostatic_control: HomeostaticControlStrategyConfig = field(
        default_factory=HomeostaticControlStrategyConfig
    )
    freeze_layers: FreezeLayersStrategyConfig = field(
        default_factory=FreezeLayersStrategyConfig
    )
    freeze_branches: FreezeBranchesStrategyConfig = field(
        default_factory=FreezeBranchesStrategyConfig
    )
    freeze_branch_kl: FreezeBranchKLStrategyConfig = field(
        default_factory=FreezeBranchKLStrategyConfig
    )


@dataclass
class EncoderTrainingConfig(BaseConfig):
    """Encoder training configuration."""

    strategy: str = "standard"

    common: CommonTrainerConfig = field(
        default_factory=lambda: CommonTrainerConfig(
            loss_function="log_mse",
            epochs=500,
            early_stopping=True,
            patience=100,
            batch_size=512,
            shuffle=True,
            grad_clip_value=5.0,
            load_best_state_dict=True,
            plot_losses=False,
            suppress_prints=False,
            print_every=10,
            use_amp=False,
            enable_nan_checking=False,
            enable_adaptive_clipping=False,
        )
    )

    strategies: TrainingStrategiesConfig = field(
        default_factory=TrainingStrategiesConfig
    )


@dataclass
class OptimizerConfig(BaseConfig):
    """Configuration for optimizer selection and parameters."""

    name: str = "adam"  # Options: "adam", "sgd", "rmsprop", "adamw"
    lr: float = 0.001
    weight_decay: float = 0.0  # Built-in L2 regularization (global)
    momentum: float = 0.9  # For SGD
    betas: tuple = (0.9, 0.999)  # For Adam/AdamW
    eps: float = 1e-8  # For Adam/AdamW


@dataclass
class MainTrainingConfig(BaseConfig):
    """Main network training configuration."""

    strategy: str = "standard"

    common: CommonTrainerConfig = field(default_factory=CommonTrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    strategies: TrainingStrategiesConfig = field(
        default_factory=TrainingStrategiesConfig
    )
    learning_strategy_config: Optional["LocalRuleConfig"] = field(
        default_factory=lambda: None
    )


@dataclass
class ThreeFactorConfig(BaseConfig):
    """Parameters for 3-factor (3F) base learning rule.

    The 3F rule uses: presynaptic * postsynaptic * error
    """

    # Compartmental dynamics
    # "auto": conductance-aware updates for shunting layers, additive-consistent
    # updates for additive layers.
    # "conductance": force conductance-aware updates (legacy behavior).
    # "additive": force additive-consistent updates.
    dynamics_mode: str = "auto"  # "auto", "conductance", "additive"
    use_conductance_scaling: bool = True  # Scale by R_tot = 1/g_tot
    use_driving_force: bool = True  # Use (E_rev - V_n) driving force
    theta: float = 0.0  # Leak reversal potential
    e_rev_exc: float = 1.0  # Excitatory reversal potential
    e_rev_inh: float = 0.0  # Inhibitory reversal (shunting)

    # Additive gain control mode (only used when dynamics_mode is "additive").
    # "none"            : post_factor = e_n  (current default, no local modulation)
    # "input_dependent" : post_factor = e_n * pseudo_R * pseudo_drive
    #                     pseudo_R = 1/(1 + |activations|), pseudo_drive = (v_mean - v_n + 1)
    # "running_stats"   : post_factor = e_n * (1 / running_std_n)
    # "learned_gain"    : post_factor = e_n * sigmoid(learnable_param_n)
    additive_gain_mode: str = "none"
    additive_stats_ema_alpha: float = 0.01  # EMA rate for running_stats mode


@dataclass
class FourFactorConfig(BaseConfig):
    """Parameters for 4-factor (4F) morphology correlation.

    Adds morphology factor rho to 3F rule.
    """

    # Morphology correlation mode
    rho_mode: str = "pearson"  # "pearson", "dot", "none"
    # Estimator for single-sample scenarios
    rho_estimator: str = "ema"  # "ema", "augment", "ema+augment"
    # EMA smoothing rate for rho statistics
    ema_alpha: float = 0.05
    # Layer-wise scaling: rho / sqrt(layer_depth)
    layer_wise_rho_scale: float = 1.0
    # Augmentation parameters (for rho_estimator="augment")
    augment_k: int = 8  # Number of augmented samples
    augment_noise_sigma: float = 0.01  # Noise level


@dataclass
class FiveFactorConfig(BaseConfig):
    """Parameters for 5-factor (5F) conditional information proxy.

    Adds information factor φ to 4F rule.
    """

    # Information proxy mode
    phi_mode: str = "conditional"  # "conditional", "variance"
    # Estimator for single-sample scenarios
    phi_estimator: str = "conditional_ema"  # "conditional_ema", "variance_ema"
    # Ridge regularization for conditional regression
    phi_ridge_lambda: float = 1e-3
    # Layer-wise scaling: φ / sqrt(layer_depth)
    layer_wise_phi_scale: float = 1.0
    # RLS forgetting factor (if using RLS-based estimators)
    rls_forgetting: float = 0.99
    # Stability clamp for the phi confidence factor
    phi_clamp_min: float = 0.25
    phi_clamp_max: float = 4.0


@dataclass
class MorphologyAwareConfig(BaseConfig):
    """Parameters for morphology-aware extensions.

    These extensions use dendritic tree topology explicitly.
    """

    # Path-integrated propagation: approximate ∏ R_k g_k attenuation
    use_path_propagation: bool = False
    path_factor_mode: str = "per_branch"  # "per_branch", "scalar_mean"

    # Branch-specific depth modulation: rho_j = rho_base / (depth_j + alpha)
    morphology_modulator_mode: str = "none"  # "none", "depth", "centrality"
    morphology_depth_offset: float = 1.0  # alpha parameter
    morphology_centrality_metric: str = "betweenness"  # For centrality mode

    # Dendritic normalization: Δg_j ← Δg_j / (Σ_k g_k + ε)
    use_dendritic_normalization: bool = False

    # Apical vs basal branch differentiation
    use_branch_type_rules: bool = False
    apical_branch_scale: float = 1.0  # Scale for apical (feedback) branches
    basal_branch_scale: float = 1.0  # Scale for basal (feedforward) branches
    use_branch_length_modulation: bool = False

    # Router/pathway-derived branch roles (preferred over heuristic branch types)
    use_branch_role_rules: bool = False
    branch_role_source: str = "router_pathways"  # "router_pathways"
    specialized_branch_scale: float = 1.0
    mixed_branch_scale: float = 1.0
    branch_role_power: float = 1.0
    branch_role_alignment_weight: float = 0.0


@dataclass
class HSICConfig(BaseConfig):
    """Parameters for HSIC auxiliary loss."""

    enabled: bool = False
    weight: float = 0.0
    self_weight: float = 1.0  # Self-decorrelation
    target_weight: float = 1.0  # Target-correlation
    target_source: str = "labels"  # "labels" or "logits"
    # Kernel configuration
    kernel: str = "linear"  # "linear", "rbf", "polynomial"
    sigma: float = 1.0  # RBF bandwidth
    degree: int = 2  # Polynomial degree
    coef0: float = 1.0  # Polynomial coefficient
    # Stability
    grad_clip_value: float = 0.0
    warmup_epochs: int = 0
    apply_last_layer_only: bool = False


@dataclass
class InhibitoryHomeostasisConfig(BaseConfig):
    """Parameters for local inhibitory homeostasis updates.

    These auxiliary updates are applied only to inhibitory synapses and are
    designed to steer branches toward stable shunting regimes using local state.
    """

    enabled: bool = False
    mode: str = "r_tot"  # "r_tot", "voltage"
    weight: float = 0.0
    target_r_tot: float = 0.35
    target_voltage: float = 0.15


@dataclass
class LocalRuleConfig(BaseConfig):
    """Configuration for local learning rules.

    Organizes parameters by learning rule method for clarity.
    Each rule (3F/4F/5F) has its own nested configuration.
    """

    # ========== CORE CONFIGURATION ==========
    rule_variant: str = "3f"  # "3f", "4f", "5f"

    # ========== ERROR SIGNAL ==========
    error_mode: str = "auto"  # "auto", "mse", "ce", "bce"
    error_broadcast_mode: str = "scalar"  # "scalar", "per_soma", "local_mismatch", "low_rank", "path_transport", "pathway_vector"
    error_noise_sigma: float = (
        0.0  # Additive Gaussian noise on broadcast error (0 = off)
    )
    broadcast_rank: int = 4  # Rank for low-rank vectorized broadcast
    broadcast_init_scale: float = 1.0  # Scale for low-rank random broadcast
    pathway_broadcast_residual: float = 0.25  # Blend with direct local error
    pathway_activity_gate_strength: float = 1.0  # Modulate by local pathway activity

    # ========== BROADCAST BANDWIDTH CONTROL ==========
    # Reduce broadcast precision to test low-bandwidth biological plausibility
    broadcast_bandwidth: str = "full"  # "full", "sign_only", "quantized", "sparse_topk"
    broadcast_bits: int = (
        8  # Quantization bits (2, 4, 8); only used when bandwidth="quantized"
    )
    broadcast_topk_fraction: float = (
        0.3  # Fraction of neurons to keep; only for "sparse_topk"
    )

    # ========== RULE-SPECIFIC PARAMETERS ==========
    # 3-factor rule parameters
    three_factor: ThreeFactorConfig = field(default_factory=ThreeFactorConfig)

    # 4-factor rule parameters (includes 3F)
    four_factor: FourFactorConfig = field(default_factory=FourFactorConfig)

    # 5-factor rule parameters (includes 4F)
    five_factor: FiveFactorConfig = field(default_factory=FiveFactorConfig)

    # Morphology-aware extensions
    morphology_aware: MorphologyAwareConfig = field(
        default_factory=MorphologyAwareConfig
    )

    # HSIC auxiliary loss
    hsic: HSICConfig = field(default_factory=HSICConfig)

    # Local inhibitory homeostasis
    inhibitory_homeostasis: InhibitoryHomeostasisConfig = field(
        default_factory=InhibitoryHomeostasisConfig
    )

    # ========== PARAMETER UPDATE CONTROL ==========
    update_inactive_weights: bool = False
    update_reactivation: bool = True
    encoder_update_mode: str = "none"  # "backprop", "none"
    decoder_update_mode: str = "backprop"  # "backprop", "local", "none"

    # ========== TRAINING SCHEDULE ==========
    freeze_encoder_epochs: int = 0
    freeze_reactivation_epochs: int = 0
    freeze_decoder_epochs: int = 0
    training_schedule: Optional[list[dict[str, Any]]] = None

    # ========== OPTIMIZATION ==========
    clip_grad_value: float = 5.0
    normalize_by_batch: bool = True


@dataclass
class ReportingConfig(BaseConfig):
    """Configuration for reporting metrics."""

    report_non_pruned: bool = True  # Report number of non-pruned synapses
    save_pruning_stats: bool = True  # Save pruning statistics to file
    detailed_branch_report: bool = True  # Detailed per-branch reporting
    redo_analysis_after_pruning: bool = (
        True  # Redo all analyses after pruning for comparison
    )


@dataclass
class TrainingConfig(BaseConfig):
    """Complete training configuration."""

    encoder: EncoderTrainingConfig = field(default_factory=EncoderTrainingConfig)
    main: MainTrainingConfig = field(default_factory=MainTrainingConfig)
