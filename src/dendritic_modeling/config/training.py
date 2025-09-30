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
class LocalRuleConfig(BaseConfig):
    """Configuration for local learning rules.

    Attributes
    ----------
    rule_variant: str
        '3f' for three-factor, '4f' to include morphology correlation, '5f' to
        include both morphology and information proxy.
    error_mode: str
        'auto', 'mse', 'ce', or 'bce' for deriving the soma error signal.
    error_broadcast_mode: str
        How to broadcast the soma error to compartments: 'scalar' (default, per-sample
        scalar broadcast to all somas), 'per_soma' (map output-dim error to per-soma),
        or 'local_mismatch' (modulate scalar error by local parent-vs-soma mismatch).
    ema_alpha: float
        EMA smoothing for morphology/information statistics.
    use_conductance_scaling: bool
        If True, scale updates by R_tot = 1 / (g_tot) per compartment.
    use_driving_force: bool
        If True, include (E_rev - V_n) multiplier for synapses.
    theta: float
        Leak reversal potential (post term uses V_n - theta if driving_force=False).
    e_rev_exc: float
        Excitatory reversal potential used for driving force.
    e_rev_inh: float
        Inhibitory reversal potential proxy (typically 0 for shunting).
    clip_grad_value: float
        Clip absolute gradient values before optimizer.step().
    normalize_by_batch: bool
        If True, average updates over batch.
    """

    rule_variant: str = "3f"
    error_mode: str = "auto"
    error_broadcast_mode: str = "scalar"
    ema_alpha: float = 0.05
    use_conductance_scaling: bool = True
    use_driving_force: bool = True
    theta: float = 0.0
    e_rev_exc: float = 1.0
    e_rev_inh: float = 0.0
    clip_grad_value: float = 5.0
    normalize_by_batch: bool = True
    # If True, update inactive TopK weights too (mask only applied in forward)
    update_inactive_weights: bool = False
    # phi (information proxy) configuration
    phi_mode: str = "conditional"  # "conditional" uses parent voltages; "variance" uses layer variance
    phi_ridge_lambda: float = 1e-3  # Regularization for conditional regression
    # Single-sample estimators
    rho_estimator: str = "ema"  # "ema", "augment", "ema+augment"
    phi_estimator: str = "conditional_ema"  # "conditional_ema", "variance_ema"
    augment_k: int = 8
    augment_noise_sigma: float = 0.01
    rls_forgetting: float = 0.99
    # Rho (morphology factor) mode: "pearson" uses correlation; "dot" uses dot-product proxy; "none" disables
    rho_mode: str = "pearson"
    # HSIC/pHSIC auxiliary loss configuration
    hsic_enabled: bool = False
    hsic_weight: float = 0.0
    hsic_self_weight: float = 1.0
    hsic_target_weight: float = 1.0
    hsic_target_source: str = "labels"  # "labels" or "logits"
    hsic_kernel: str = "linear"  # "linear", "rbf", "polynomial"
    hsic_sigma: float = 1.0  # RBF kernel bandwidth (gamma = 1/(2*sigma^2))
    hsic_degree: int = 2  # Polynomial kernel degree
    hsic_coef0: float = 1.0  # Polynomial kernel coefficient
    # HSIC stability options
    hsic_grad_clip_value: float = 0.0  # Clip HSIC gradients elementwise if > 0
    hsic_warmup_epochs: int = 0  # Linearly ramp HSIC weights over first N epochs
    hsic_apply_last_layer_only: bool = False  # Apply HSIC only on last dendritic layer
    # Decoder update strategy: "backprop" or "none"
    decoder_update_mode: str = "backprop"
    # Gate reactivation parameter updates
    update_reactivation: bool = True
    # Optional epoch-based freezing and phased training
    freeze_reactivation_epochs: int = (
        0  # Freeze reactivation params for initial N epochs
    )
    freeze_decoder_epochs: int = 0  # Freeze decoder for initial N epochs
    # Optional per-epoch schedule overriding the above. Each entry may include:
    # { start_epoch: int, end_epoch: int, update_reactivation: bool, decoder_update_mode: str }
    training_schedule: Optional[list[dict[str, Any]]] = None
    # Layer-wise scaling for morphology factors
    layer_wise_rho_scale: float = 1.0  # Scale rho by 1/sqrt(layer_depth) if > 0
    layer_wise_phi_scale: float = 1.0  # Scale phi by 1/sqrt(layer_depth) if > 0


@dataclass
class ReportingConfig(BaseConfig):
    """Configuration for reporting metrics."""

    report_non_pruned: bool = True  # Report number of non-pruned synapses
    save_pruning_stats: bool = True  # Save pruning statistics to file
    detailed_branch_report: bool = True  # Detailed per-branch reporting
    redo_analysis_after_pruning: bool = (
        True  # Redo all analyses after pruning for comparison
    )
