# config.py
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field, asdict

from omegaconf import OmegaConf as om

########################################################
# BaseConfig
########################################################

class BaseConfig:
    @classmethod
    def load(cls, path: str):
        conf = om.load(path)
        return cls(**conf)

    def save(self, path: str):
        om.save(config=self, f=path)

    def asdict(self):
        return asdict(self)

########################################################
# Network parameter classes
########################################################

@dataclass
class EINetParameters(BaseConfig):
    """Parameters for the EINet-based dendritic model."""
    input_dim: int = 784
    excitatory_layer_sizes: List[int] = field(default_factory=lambda: [10])
    inhibitory_layer_sizes: List[int] = field(default_factory=lambda: [20])
    excitatory_branch_factors: List[int] = field(default_factory=lambda: [1,2])
    inhibitory_branch_factors: List[int] = field(default_factory=list)

    ee_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [24])
    ei_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [100])
    ie_synapses_per_branch_per_layer: List[int] = field(default_factory=lambda: [1])
    ii_synapses_per_branch_per_layer: List[int] = field(default_factory=list)

    reactivate: bool = True
    somatic_synapses: bool = True
    topk_init_method: str = "xavier_normal"
    use_shunting: bool = True

@dataclass
class FeedForwardParameters(BaseConfig):
    """Parameters for a simple feedforward network."""
    input_dim: int = 784
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.2
    output_dim: int = 10

########################################################
# Task config
########################################################

@dataclass
class BaseTaskParameters(BaseConfig):
    pass

@dataclass
class GenericTaskParameters(BaseTaskParameters):
    train_size: int = 8000
    valid_size: int = 1000
    test_size: int = 1000
    image_size: int = 10
    seed: int = 0

@dataclass
class TaskConfig(BaseConfig):
    """ Specifies which dataset and parameters to use. """
    dataset: str = "line"
    parameters: GenericTaskParameters = field(default_factory=GenericTaskParameters)

########################################################
# Model config
########################################################

@dataclass
class NetworkConfig(BaseConfig):
    """Holds type plus EInet / Feedforward parameters."""
    type: str = "einet"  # or 'feedforward'
    parameters: Union[EINetParameters, FeedForwardParameters] = field(default_factory=EINetParameters)

@dataclass
class ModelConfig(BaseConfig):
    task: str = "classification"
    probabilistic: bool = False
    network: NetworkConfig = field(default_factory=NetworkConfig)

    def asdict(self):
        """
        Convert to dict, but also flatten network config.
        """
        return {
            "task": self.task,
            "probabilistic": self.probabilistic,
            "network": {
                "type": self.network.type,
                "parameters": asdict(self.network.parameters),
            }
        }

########################################################
# Train config
########################################################

@dataclass
class TrainConfig(BaseConfig):
    run_name: str = "dm_test"
    seed: int = 42
    train_valid_split: float = 1.0
    lr: float = 1e-3
    epochs: int = 20
    batch_size: int = 64
    grad_clip_value: float = 0.5
    weight_decay: float = 0.0

    def asdict(self):
        return {
            "run_name": self.run_name,
            "seed": self.seed,
            "train_valid_split": self.train_valid_split,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_clip_value": self.grad_clip_value,
            "weight_decay": self.weight_decay,
        }

########################################################
# Wandb config
########################################################

@dataclass
class WandbConfig(BaseConfig):
    entity: str = ""
    project: str = "dendritic_modeling_pr"
    group: str = "debug"
    tags: list = field(default_factory=list)

########################################################
# Settings config
########################################################

@dataclass
class SettingsConfig(BaseConfig):
    visualizations: bool = False

########################################################
# Primary aggregator
########################################################

@dataclass
class Config(BaseConfig):
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    settings: SettingsConfig = field(default_factory=SettingsConfig)

    def asdict(self, exclude=None):
        exclude = exclude or []
        config_dict = {
            "model": self.model.asdict(),
            "train": self.train.asdict(),
            "task": asdict(self.task),
            "wandb": asdict(self.wandb),
            "settings": asdict(self.settings),
        }
        for key in exclude:
            config_dict.pop(key, None)
        return config_dict


########################################################
# load_config
########################################################

def load_config(path: str = None) -> Config:
    if path:
        raw_conf = om.load(path)

        # Build model config
        raw_model = raw_conf.get("model", {})
        model_cfg = ModelConfig(
            task=raw_model.get("task", "classification"),
            probabilistic=raw_model.get("probabilistic", False),
            network=NetworkConfig(
                type=raw_model.get("network", {}).get("type", "einet"),
                parameters=EINetParameters(**raw_model["network"]["parameters"])
                  if raw_model["network"]["type"].lower() == "einet" 
                  else FeedForwardParameters(**raw_model["network"]["parameters"])
            )
        )

        # Build train config
        raw_train = raw_conf.get("train", {})
        train_cfg = TrainConfig(**raw_train)

        # Build task config
        raw_task = raw_conf.get("task", {})
        # We might store parameters in raw_task['parameters']
        param_obj = GenericTaskParameters(**raw_task.get("parameters", {}))
        task_cfg = TaskConfig(
            dataset=raw_task.get("dataset", "line"),
            parameters=param_obj
        )

        # Build wandb config
        raw_wandb = raw_conf.get("wandb", {})
        wandb_cfg = WandbConfig(**raw_wandb)

        # Build settings config
        raw_settings = raw_conf.get("settings", {})
        settings_cfg = SettingsConfig(**raw_settings)

        return Config(
            model=model_cfg,
            train=train_cfg,
            task=task_cfg,
            wandb=wandb_cfg,
            settings=settings_cfg
        )
    else:
        # Return defaults
        return Config()