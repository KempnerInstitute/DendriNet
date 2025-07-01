from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from dendritic_modeling.config.base import BaseConfig


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
    # for learning strategies other than standard:
    pretrain_epochs: int = 50
    target_voltage: float = 0.5
    kl_weight: float = 0.1
    reverse_training: bool = False
    epochs_per_layer: int = 50
    epochs_per_branch: int = 50
    final_tune_epochs: int = 10


@dataclass
class TrainConfig(BaseConfig):
    learning_strategy: str = "mle"
    seed: int = 0
    train_valid_split: float = 0.8
    optim_dict: Dict[str, Any] = field(
        default_factory=dict
    )  # Optimizer / param group settings
    weight_decay_rate: float = 0.1
    encoder_net_trainer: TrainerConfig = field(default_factory=TrainerConfig)
    main_trainer: TrainerConfig = field(default_factory=TrainerConfig)
 