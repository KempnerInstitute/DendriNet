from pathlib import Path
from typing import List, Union
from dataclasses import dataclass, field, asdict

from omegaconf import OmegaConf as om


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
class EINetParameters(BaseConfig):
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
    init_gain: float = 1.0
    init_inhib_offset: float = 0.0
    reactivation_initial_m: float = 1.5
    reactivation_initial_b: float = 1.0
    gradient_strategy: str = "none"
    gradient_scale_factor: float = 1.0
    weight_decay_rate: float = 0.0
    local_learning: str = "none"
    local_loss_weight: float = 0.0


@dataclass
class FeedForwardParameters(BaseConfig):
    input_dim: int = 784
    hidden_layer_sizes: List[int] = field(default_factory=lambda: [128, 64])
    activation: str = "relu"
    dropout: float = 0.2
    output_dim: int = 10


@dataclass
class BaseTaskParameters(BaseConfig):
    pass


@dataclass
class LineParameters(BaseTaskParameters):
    train_size: int = 1000
    valid_size: int = 100
    test_size: int = 100
    image_size: int = 10
    seed: int = 42


@dataclass
class TaskConfig(BaseConfig):
    dataset: str = "line"
    parameters: Union[LineParameters] = field(default_factory=LineParameters)


@dataclass
class NetworkConfig(BaseConfig):
    type: str = "EINet"
    parameters: Union[EINetParameters, FeedForwardParameters] = field(default_factory=EINetParameters)


@dataclass
class ModelConfig(BaseConfig):
    task: str = "classification"
    probabilistic: bool = False
    network: NetworkConfig = field(default_factory=NetworkConfig)


@dataclass
class TrainConfig(BaseConfig):
    run_name: str = "dm_test"
    seed: int = 42
    train_valid_split: float = 0.8
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 32
    grad_clip_value: float = 0.5
    epochs_per_layer: int = 50
    two_step_training: bool = False
    pretrain_epochs: int = 5
    maintrain_epochs: int = 45


@dataclass
class WandbConfig(BaseConfig):
    entity: str = "your_wandb_entity"
    project: str = "dendritic_modeling_pr"
    group: str = "debug"
    tags: list = field(default_factory=list)


@dataclass
class AnalysisConfig(BaseConfig):
    enable_ablation: bool = False
    enable_mutual_info: bool = False


@dataclass
class VisualizationConfig(BaseConfig):
    """
    A new dataclass for visualization settings.
    """
    enabled: bool = False
    save_path: str = "results/visuals"
    plots: List[str] = field(default_factory=list)


@dataclass
class Config:
    """
    Main config. Removed the old SettingsConfig reference. 
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)

    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def asdict(self, exclude=None):
        exclude = exclude or []
        config_dict = {
            "model": self.model.asdict() if hasattr(self.model, 'asdict') else self.model,
            "train": self.train.asdict() if hasattr(self.train, 'asdict') else self.train,
            "wandb": self.wandb.asdict() if hasattr(self.wandb, 'asdict') else self.wandb,
            "task": self.task.asdict() if hasattr(self.task, 'asdict') else self.task,
            "analysis": self.analysis.asdict() if hasattr(self.analysis, 'asdict') else self.analysis,
            # new addition:
            "visualization": self.visualization.asdict() if hasattr(self.visualization, 'asdict') else self.visualization,
        }
        for key in exclude:
            config_dict.pop(key, None)
        return config_dict


def load_config(path: str = None) -> Config:
    """
    Loads from a YAML path using OmegaConf, constructs and returns a Config object.
    """
    if path:
        raw_conf = om.load(path)
        return Config(
            model=ModelConfig(**raw_conf.get("model", {})),
            train=TrainConfig(**raw_conf.get("train", {})),
            wandb=WandbConfig(**raw_conf.get("wandb", {})),
            task=TaskConfig(**raw_conf.get("task", {})),
            analysis=AnalysisConfig(**raw_conf.get("analysis", {})),

            visualization=VisualizationConfig(**raw_conf.get("visualization", {})),
        )
    else:
        return Config()