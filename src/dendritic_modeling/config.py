# config.py
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field, asdict

from omegaconf import OmegaConf as om

#
# 1) Base configuration class
#
class BaseConfig:
    @classmethod
    def load(cls, path: str):
        conf = om.load(path)
        return cls(**conf)

    def save(self, path: str):
        om.save(config=self, f=path)

    def asdict(self):
        return asdict(self)

#
# 2) EINetParameters & FeedForwardParameters
#
@dataclass
class EINetParameters(BaseConfig):
    """ Parameters for EINet network """
    input_dim: int = 784
    excitatory_layer_sizes: list = field(default_factory=lambda: [10])
    inhibitory_layer_sizes: list = field(default_factory=lambda: [20])
    excitatory_branch_factors: list = field(default_factory=lambda: [24])
    inhibitory_branch_factors: list = field(default_factory=list)
    ee_synapses_per_branch_per_layer: list = field(default_factory=lambda: [24])
    ei_synapses_per_branch_per_layer: list = field(default_factory=lambda: [100])
    ie_synapses_per_branch_per_layer: list = field(default_factory=lambda: [1])
    ii_synapses_per_branch_per_layer: list = field(default_factory=list)
    reactivate: bool = True
    somatic_synapses: bool = True
    topk_init_method: str = "xavier_normal"
    use_shunting: bool = True

@dataclass
class FeedForwardParameters(BaseConfig):
    """ Parameters for feedforward network """
    input_dim: int = 784
    hidden_layer_sizes:  List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.2
    output_dim: int = 10

#
# 3) Task configuration
#
@dataclass
class BaseTaskParameters(BaseConfig):
    """ Base class for all task parameters """
    pass

@dataclass
class LineParameters(BaseTaskParameters):
    """ Parameters for line task """
    train_size: int = 1000
    valid_size: int = 100
    test_size: int = 100
    image_size: int = 10
    seed: int = 42

@dataclass
class TaskConfig(BaseConfig):
    """ Configuration for task hyperparameters """
    dataset: str = "line"
    parameters: Union[LineParameters] = field(default_factory=LineParameters)

#
# 4) Model config 
#
@dataclass
class NetworkConfig(BaseConfig):
    """ Configuration for network hyperparameters """
    type: str = "EINet"
    parameters: Union[EINetParameters, FeedForwardParameters] = field(default_factory=EINetParameters)

@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model hyperparameters."""
    task: str = "classification"
    probabilistic: bool = False
    network: NetworkConfig = field(default_factory=NetworkConfig)

#
# 5) Train & Wandb & Settings
#
@dataclass
class TrainConfig(BaseConfig):
    """Configuration for training hyperparameters."""
    run_name: str = "dm_test"
    seed: int = 42
    train_valid_split: float = 0.8
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 32
    grad_clip_value: float = 0.5
    weight_decay: float = 0.0

@dataclass
class WandbConfig(BaseConfig):
    """Configuration for Weights & Biases (W&B) integration."""
    entity: str = "your_wandb_entity"
    project: str = "dendritic_modeling_pr"
    group: str = "debug"
    tags: list = field(default_factory=list)

@dataclass
class SettingsConfig(BaseConfig):
    """Configuration for all other settings."""
    visualizations: bool = False

#
# 6) The main Config
#
@dataclass
class Config:
    """Primary configuration that combines everything."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    settings: SettingsConfig = field(default_factory=SettingsConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    def asdict(self, exclude=None):
        exclude = exclude or []
        config_dict = {
            "model": self.model.asdict() if hasattr(self.model, 'asdict') else self.model,
            "train": self.train.asdict() if hasattr(self.train, 'asdict') else self.train,
            "wandb": self.wandb.asdict() if hasattr(self.wandb, 'asdict') else self.wandb,
            "settings": self.settings.asdict() if hasattr(self.settings, 'asdict') else self.settings,
            "task": self.task.asdict() if hasattr(self.task, 'asdict') else self.task,
        }
        for key in exclude:
            config_dict.pop(key, None)
        return config_dict

#
# 7) load_config function
#
def load_config(path: str = None) -> Config:
    """
    Load from YAML at `path`. We assume the YAML has top-level keys:
      model, train, wandb, settings, task
    If missing, we default them.
    """
    if path:
        raw_conf = om.load(path)
        return Config(
            model=ModelConfig(**raw_conf.get("model", {})),
            train=TrainConfig(**raw_conf.get("train", {})),
            wandb=WandbConfig(**raw_conf.get("wandb", {})),
            settings=SettingsConfig(**raw_conf.get("settings", {})),
            task=TaskConfig(**raw_conf.get("task", {})),
        )
    else:
        return Config()