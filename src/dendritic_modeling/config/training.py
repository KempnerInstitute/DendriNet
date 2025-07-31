from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

from dendritic_modeling.config.base import BaseConfig


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
    layers: List[int] = field(default_factory=list)  # List of layer indices (empty = all)
    branches: List[str] = field(default_factory=list)  # List of branch names (empty = all)
    weight_types: List[str] = field(default_factory=lambda: ["all"])  # Types of weights


@dataclass
class RegularizationConfig(BaseConfig):
    """Configuration for L1/L2 regularization."""
    l1_weight: float = 0.0  # Strength of L1 regularization (0 to disable)
    l2_weight: float = 0.0  # Strength of L2 regularization (0 to disable)
    selective: SelectiveConfig = field(default_factory=SelectiveConfig)


@dataclass
class PruningConfig(BaseConfig):
    """Configuration for weight pruning."""
    enabled: bool = False
    threshold: float = 0.001  # Prune weights with abs(value) < threshold
    selective: SelectiveConfig = field(default_factory=SelectiveConfig)
    re_evaluate_after: bool = True  # Whether to re-evaluate after pruning
    dynamic_pruning: bool = False  # Whether to prune during training
    pruning_frequency: int = 10  # Epochs between dynamic pruning


@dataclass
class ReportingConfig(BaseConfig):
    """Configuration for reporting metrics."""
    report_non_pruned: bool = True  # Report number of non-pruned synapses
    save_pruning_stats: bool = True  # Save pruning statistics to file
    detailed_branch_report: bool = True  # Detailed per-branch reporting


@dataclass
class ParamGroupsConfig(BaseConfig):
    lr: float = 0.001
    split_params: bool = True
    topk_lr: float = 0.001
    blocklinear_lr: float = 0.0001
    reactivation_lr: float = 0.0001
    decoder_lr: float = 0.0001


@dataclass
class TrainerConfig(BaseConfig):
    loss_function: str = "mse"
    epochs: int = 50
    early_stopping: bool = False
    patience: int = 10
    batch_size: int = 128
    shuffle: bool = True
    grad_clip_value: float = 5.0
    load_best_state_dict: bool = True
    plot_losses: bool = False
    save_path: Optional[str] = None
    suppress_prints: bool = False
    print_every: int = 10
    # Mixed precision training
    use_amp: bool = False
    # for learning strategies other than standard:
    pretrain_epochs: int = 50
    target_voltage: float = 0.5
    kl_weight: float = 0.1
    reverse_training: bool = False
    epochs_per_layer: int = 50
    epochs_per_branch: int = 50
    final_tune_epochs: int = 10


@dataclass
class EncoderTrainConfig(BaseConfig):
    load_save_root: Optional[str] = None
    lr: float = 0.001
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


@dataclass
class MainTrainConfig(BaseConfig):
    learning_strategy: str = "standard"
    weight_decay_rate: float = 0.01  # Deprecated, use optimizer.weight_decay instead
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    param_groups: ParamGroupsConfig = field(default_factory=ParamGroupsConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)


@dataclass
class TrainConfig(BaseConfig):
    seed: int = 0
    train_valid_split: float = 0.8
    encoder: EncoderTrainConfig = field(default_factory=EncoderTrainConfig)
    main: MainTrainConfig = field(default_factory=MainTrainConfig)
    run_name: str = 'experiment'
    # Performance features
    enable_amp: bool = False
    enable_profiling: bool = False
    profiling_output_dir: str = 'outputs/profiling/'
    enable_hooks: bool = True
    use_fsdp: bool = False
    fsdp_config: Optional[Dict[str, Any]] = None
    checkpointing: Dict[str, Any] = field(default_factory=lambda: {'enabled': True, 'save_every_n_epochs': 10, 'checkpoint_dir': 'outputs/checkpoints/'})
