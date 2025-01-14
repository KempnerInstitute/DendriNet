# conf.py
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field, asdict

from omegaconf import OmegaConf as om

# Base configuration class
class BaseConfig:
    @classmethod
    def load(cls, path: str):
        conf = om.load(path)
        return cls(**conf)

    def save(self, path: str):
        om.save(config=self, f=path)

    def asdict(self):
        return asdict(self)
    
# Base class for all network parameters
@dataclass
class BaseNetworkParameters(BaseConfig):
    """ Base class for all network parameters """
    pass

# Parameters for EINet network
@dataclass
class EINetParameters(BaseNetworkParameters):
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

# Parameters for feedforward network
@dataclass
class FeedForwardParameters(BaseNetworkParameters):
    """ Parameters for feedforward network """
    input_dim: int = 784
    hidden_layer_sizes:  List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.2
    output_dim: int = 10


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


# Task configuration
@dataclass
class TaskConfig(BaseConfig):
    """ Configuration for task hyperparameters """
    dataset: str = "line"
    parameters: Union[LineParameters] = field(default_factory=LineParameters)



# Network configuration
@dataclass
class NetworkConfig(BaseConfig):
    """ Configuration for network hyperparameters """
    type: str = "EINet"
    parameters: Union[EINetParameters, FeedForwardParameters] = field(default_factory=EINetParameters)


@dataclass
class WandbConfig(BaseConfig):
    """Configuration for Weights & Biases (W&B) integration."""
    entity: str = "your_wandb_entity"
    project: str = "dendritic_modeling_pr"
    group: str = "debug"
    tags: list = field(default_factory=list)

@dataclass
class ModelConfig(BaseConfig):
    """Configuration for model hyperparameters."""
    task: str = "classification"
    probabilistic: bool = False
    network: NetworkConfig = field(default_factory=NetworkConfig)


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


# Settings configuration (all other parameters)
@dataclass
class SettingsConfig(BaseConfig):
    """Configuration for all other settings."""
    visualizations: bool = False

@dataclass
class Config:
    """Primary configuration that combines model and training configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    settings: SettingsConfig = field(default_factory=SettingsConfig)
    task: TaskConfig = field(default_factory=TaskConfig)

    def asdict(self, exclude=None):
        """Convert the Config class to a dictionary, with options to exclude keys."""
        exclude = exclude or []
        config_dict = {
            "model": self.model.asdict() if hasattr(self.model, 
                                                    'asdict') else self.model,
            "train": self.train.asdict() if hasattr(self.train, 
                                                    'asdict') else self.train,
            "wandb": self.wandb,
            "settings": self.settings,
            "task": self.task.asdict() if hasattr(self.task, 
                                                  'asdict') else self.task
        }
        
        # Remove excluded keys
        for key in exclude:
            config_dict.pop(key, None)
        
        return config_dict




def load_config(path: str = None) -> Config:
    """Load configuration from a file or use defaults."""
    if path:
        raw_conf = om.load(path)
        return Config(
            model=ModelConfig(**raw_conf.get("model", {})),
            train=TrainConfig(**raw_conf.get("train", {})),
            wandb=WandbConfig(**raw_conf.get("wandb", {})),
            settings=SettingsConfig(**raw_conf.get("settings", {}))
        )
    return Config()


