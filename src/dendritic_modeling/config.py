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
    train_valid_split: float = 0.8
    lr: float = 1e-3
    epochs: int = 200
    batch_size: int = 32


@dataclass
class Config:
    """Primary configuration that combines model and training configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str = None) -> Config:
    """Load configuration from a file or use defaults."""
    if path:
        raw_conf = om.load(path)
        return Config(
            model=ModelConfig(**raw_conf.get("model", {})),
            train=TrainConfig(**raw_conf.get("train", {})),
        )
    return Config()
