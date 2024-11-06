# conf.py
from omegaconf import OmegaConf as om
from dataclasses import dataclass, field, asdict
from pathlib import Path

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


@dataclass
class TrainConfig(BaseConfig):
    """Configuration for training hyperparameters."""
    run_name: str = "dm_test"
    seed: int = 42
    train_valid_split: float = 0.8
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 32


@dataclass
class Config:
    """Primary configuration that combines model and training configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def asdict(self, exclude=None):
        """Convert the Config class to a dictionary, with options to exclude keys."""
        exclude = exclude or []
        config_dict = {
            "model": self.model.asdict() if hasattr(self.model, 
                                                    'asdict') else self.model,
            "train": self.train.asdict() if hasattr(self.train, 
                                                    'asdict') else self.train,
            "wandb": self.wandb,
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
        )
    return Config()
