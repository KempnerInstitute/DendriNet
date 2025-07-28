"""
config.py
---------
Configuration file for dendritic_modeling.
This module defines the configuration dataclasses for the model, training, task, etc.
It loads and saves configurations using OmegaConf.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict

from omegaconf import OmegaConf

from functools import lru_cache

from dendritic_modeling.config.analysis import AnalysisConfig
from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.model import ModelConfig
from dendritic_modeling.config.training import TrainConfig


@dataclass
class WandbConfig(BaseConfig):
    entity: Optional[str] = None
    project: str = "dendritic-modeling"
    group: str = "debug"
    tags: List[str] = field(default_factory=lambda: ["debug"])
    use_wandb: bool = False
    # Enhanced logging options
    enabled: bool = True
    log_freq: int = 100
    log_gradients: bool = False
    log_parameters: bool = False
    log_activations: bool = False


@dataclass
class TaskConfig(BaseConfig):
    dataset: str = "mnist"
    data_path: str = ""
    parameters: dict = field(default_factory=dict)


@dataclass
class OutputsConfig(BaseConfig):
    run_name: str = "exp_run"
    dir: str = "outputs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)


@lru_cache(maxsize=10)  # Cache up to 10 configs
def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a file or use defaults."""
    if path:
        raw_conf = OmegaConf.load(path)
        return Config(
            model=ModelConfig(**raw_conf.get("model", {})),
            train=TrainConfig(**raw_conf.get("train", {})),
            wandb=WandbConfig(**raw_conf.get("wandb", {})),
            analysis=AnalysisConfig(**raw_conf.get("analysis", {})),
            task=TaskConfig(**raw_conf.get("task", {})),
            outputs=OutputsConfig(**raw_conf.get("outputs", {})),
        )
    return Config()
