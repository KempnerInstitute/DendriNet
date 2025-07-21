from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from dendritic_modeling.config.base import BaseConfig


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
    weight_decay_rate: float = 0.01
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
    performance_mode: bool = False
    use_fsdp: bool = False
    fsdp_config: Optional[Dict[str, Any]] = None
    checkpointing: Dict[str, Any] = field(default_factory=lambda: {'enabled': True, 'save_every_n_epochs': 10, 'checkpoint_dir': 'outputs/checkpoints/'})
