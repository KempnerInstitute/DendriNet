"""
config.py
---------
Configuration file for dendritic_modeling.
This module defines the configuration dataclasses for the model, training, task, etc.
It loads and saves configurations using OmegaConf.
"""
from dataclasses import dataclass, field
from typing import Dict, List

from omegaconf import OmegaConf as om

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.analysis import AnalysisConfig
from dendritic_modeling.config.model import ModelConfig
from dendritic_modeling.config.training import TrainConfig

@dataclass
class VisualizationConfig(BaseConfig):
    enabled: bool = False
    visualize_epochs: List[int] = field(default_factory=lambda: [])
    plots: List[str] = field(default_factory=lambda: ["branch_info", "ablation"])
    parameters: Dict = field(default_factory=dict)


@dataclass
class WandbConfig(BaseConfig):
    entity: str = ""
    project: str = ""
    group: str = ""
    tags: List[str] = field(default_factory=list)
    use_wandb: bool = False


@dataclass
class TaskConfig(BaseConfig):
    dataset: str = "mnist"
    data_path: str = ""
    parameters: Dict = field(default_factory=dict)


@dataclass
class OutputsConfig(BaseConfig):
    dir: str = "outputs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)


def load_config(path: str = None) -> Config:
    """Load configuration from a file or use defaults."""
    if path:
        raw_conf = om.load(path)
        return Config(
            model=ModelConfig(**raw_conf.get("model", {})),
            train=TrainConfig(**raw_conf.get("train", {})),
            wandb=WandbConfig(**raw_conf.get("wandb", {})),
            visualization=VisualizationConfig(**raw_conf.get("visualization", {})),
            analysis=AnalysisConfig(**raw_conf.get("analysis", {})),
            task=TaskConfig(**raw_conf.get("task", {})),
            outputs=OutputsConfig(**raw_conf.get("outputs", {})),
        )
    return Config()
