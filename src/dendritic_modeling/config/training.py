from dataclasses import dataclass, field
from typing import Any, Optional

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.regularization import PruningConfig, RegularizationConfig


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
    decoder_update_mode: str = "backprop"
    update_reactivation: bool = True
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
class MainTrainingConfig(BaseConfig):
    """Main network training configuration."""

    strategy: str = "standard"

    common: CommonTrainerConfig = field(default_factory=CommonTrainerConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    strategies: TrainingStrategiesConfig = field(
        default_factory=TrainingStrategiesConfig
    )


@dataclass
class TrainingConfig(BaseConfig):
    """Complete training configuration."""

    encoder: EncoderTrainingConfig = field(default_factory=EncoderTrainingConfig)
    main: MainTrainingConfig = field(default_factory=MainTrainingConfig)


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
class SelectiveConfig(BaseConfig):
    """Configuration for selective application of regularization/pruning."""

    layers: list[int] = field(
        default_factory=list
    )  # List of layer indices (empty = all)
    branches: list[str] = field(
        default_factory=list
    )  # List of branch names (empty = all)
    weight_types: list[str] = field(default_factory=lambda: ["all"])  # Types of weights


@dataclass
class RegularizationConfig(BaseConfig):
    """Configuration for all regularization types."""

    # L1/L2 regularization
    l1_weight: float = 0.0  # Strength of L1 regularization (0 to disable)
    l2_weight: float = 0.0  # Strength of L2 regularization (0 to disable)
    selective: SelectiveConfig = field(default_factory=SelectiveConfig)

    # Parameter group specific regularization
    split_params: bool = (
        False  # Split parameters into groups for differential regularization
    )
    param_group_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "topk": {"l1_weight": 0.0, "l2_weight": 0.0},
            "blocklinear": {"l1_weight": 0.0, "l2_weight": 0.0},
            "reactivation": {"l1_weight": 0.0, "l2_weight": 0.0},
            "decoder": {"l1_weight": 0.0, "l2_weight": 0.0},
        }
    )

    # E/I Weight Ratio Regularization
    enforce_ei_weight_ratio: bool = False  # Enable E/I weight ratio regularization
    target_ei_weight_ratio: float = 0.5  # Target excitatory/inhibitory weight ratio
    ei_ratio_loss_weight: float = (
        0.5  # Regularization strength (fraction of total loss)
    )
    ei_ratio_scope: str = "per_branch"  # Options: "per_branch", "per_layer", "global"
    ei_ratio_metric: str = "mean"  # Options: "mean", "median", "rms"


@dataclass
class PruningConfig(BaseConfig):
    """Configuration for weight pruning."""

    enabled: bool = False
    threshold: float = 2  # Prune weights with abs(value) < threshold
    selective: SelectiveConfig = field(
        default_factory=lambda: SelectiveConfig(
            weight_types=["excitatory", "inhibitory"]
        )
    )
    re_evaluate_after: bool = False  # Whether to re-evaluate after pruning
    dynamic_pruning: bool = False  # Whether to prune during training
    pruning_frequency: int = 10  # Epochs between dynamic pruning

    report_non_pruned: bool = True  # Report number of non-pruned synapses
    save_pruning_stats: bool = False  # Save pruning statistics to file
    detailed_branch_report: bool = True  # Detailed per-branch reporting
    redo_analysis_after_pruning: bool = (
        False  # Redo all analyses after pruning for comparison
    )


@dataclass
class ThreeFactorConfig(BaseConfig):
    """Parameters for 3-factor (3F) base learning rule.

    The 3F rule uses: presynaptic * postsynaptic * error
    """

    # Compartmental dynamics
    use_conductance_scaling: bool = True  # Scale by R_tot = 1/g_tot
    use_driving_force: bool = True  # Use (E_rev - V_n) driving force
    theta: float = 0.0  # Leak reversal potential
    e_rev_exc: float = 1.0  # Excitatory reversal potential
    e_rev_inh: float = 0.0  # Inhibitory reversal (shunting)


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


@dataclass
class MorphologyAwareConfig(BaseConfig):
    """Parameters for morphology-aware extensions.

    These extensions explicitly leverage dendritic tree topology.
    """

    # Path-integrated propagation: approximate ∏ R_k g_k attenuation
    use_path_propagation: bool = False

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
class LocalRuleConfig(BaseConfig):
    """Configuration for local learning rules.

    Organizes parameters by learning rule method for clarity.
    Each rule (3F/4F/5F) has its own nested configuration.
    """

    # ========== CORE CONFIGURATION ==========
    rule_variant: str = "3f"  # "3f", "4f", "5f"

    # ========== ERROR SIGNAL ==========
    error_mode: str = "auto"  # "auto", "mse", "ce", "bce"
    error_broadcast_mode: str = "scalar"  # "scalar", "per_soma", "local_mismatch"

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

    # ========== PARAMETER UPDATE CONTROL ==========
    update_inactive_weights: bool = False
    update_reactivation: bool = True
    decoder_update_mode: str = "backprop"  # "backprop", "local", "none"

    # ========== TRAINING SCHEDULE ==========
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
