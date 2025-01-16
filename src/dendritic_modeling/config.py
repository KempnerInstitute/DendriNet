# config.py
from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field, asdict

from omegaconf import OmegaConf as om

###################################################
# Base classes
###################################################

class BaseConfig:
    @classmethod
    def load(cls, path: str):
        conf = om.load(path)
        return cls(**conf)

    def save(self, path: str):
        om.save(config=self, f=path)

    def asdict(self):
        return asdict(self)

@dataclass
class GenericTaskParameters(BaseConfig):
    train_size: int = 8000
    valid_size: int = 1000
    test_size: int = 1000
    image_size: int = 10
    n_orient_classes: int = 8
    n_bits: int = 8
    seed: int = 0

@dataclass
class TaskConfig(BaseConfig):
    """
    Holds the dataset name and sub-parameters for your experiments.
    dataset can be:
       "line", "center_surround", "orientation_bars", "multi_xor", "mnist",
       "info_shunting", "noise_resilience", "feature_selectivity",
       "context_gating", "learning_dynamics", "hierarchical_processing"
    """
    dataset: str = "line"
    parameters: GenericTaskParameters = field(default_factory=GenericTaskParameters)

@dataclass
class EINetParameters(BaseConfig):
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
    input_dim: int = 784
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [128,64])
    activation: str = "relu"
    dropout: float = 0.2
    output_dim: int = 10

@dataclass
class NetworkConfig(BaseConfig):
    type: str = "einet"  # or "feedforward"
    parameters: Union[EINetParameters, FeedForwardParameters] = field(default_factory=EINetParameters)

@dataclass
class ModelConfig(BaseConfig):
    task: str = "classification"
    probabilistic: bool = False
    network: NetworkConfig = field(default_factory=NetworkConfig)

    def asdict(self):
        return {
            "task": self.task,
            "probabilistic": self.probabilistic,
            "network": {
                "type": self.network.type,
                "parameters": asdict(self.network.parameters),
            }
        }

@dataclass
class TrainConfig(BaseConfig):
    train_valid_split: float = 1.0
    lr: float = 0.001
    epochs: int = 20
    batch_size: int = 64
    grad_clip_value: float = 0.5
    weight_decay: float = 0.0

    def asdict(self):
        return {
            "train_valid_split": self.train_valid_split,
            "lr": self.lr,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "grad_clip_value": self.grad_clip_value,
            "weight_decay": self.weight_decay,
        }

@dataclass
class WandbConfig(BaseConfig):
    entity: str = ""
    project: str = "dendritic_modeling"
    group: str = "debug"
    tags: list = field(default_factory=list)

@dataclass
class SettingsConfig(BaseConfig):
    visualizations: bool = False

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

def load_config(path: str = None) -> Config:
    if path:
        raw_conf = om.load(path)
        # Build model
        raw_model = raw_conf.get("model", {})
        network_type = raw_model.get("network", {}).get("type", "einet").lower()
        net_params = raw_model.get("network", {}).get("parameters", {})

        if network_type == "einet":
            netp = EINetParameters(**net_params)
        else:
            netp = FeedForwardParameters(**net_params)

        model_cfg = ModelConfig(
            task=raw_model.get("task", "classification"),
            probabilistic=raw_model.get("probabilistic", False),
            network=NetworkConfig(
                type=network_type,
                parameters=netp
            )
        )

        raw_train = raw_conf.get("train", {})
        train_cfg = TrainConfig(**raw_train)

        raw_task = raw_conf.get("task", {})
        generic_params = raw_task.get("parameters", {})
        task_params = GenericTaskParameters(**generic_params)
        task_cfg = TaskConfig(
            dataset=raw_task.get("dataset", "line"),
            parameters=task_params
        )

        raw_wandb = raw_conf.get("wandb", {})
        wandb_cfg = WandbConfig(**raw_wandb)

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
        return Config()