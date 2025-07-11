"""
config.py
---------
Configuration file for dendritic_modeling.
This module defines the configuration dataclasses for the model, training, task, etc.
It loads and saves configurations using OmegaConf.
"""

from dataclasses import dataclass, field
from typing import Optional

from omegaconf import OmegaConf

from dendritic_modeling.config.analysis import AnalysisConfig
from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.model import ModelConfig
from dendritic_modeling.config.training import TrainConfig


@dataclass
class VisualizationConfig(BaseConfig):
    enabled: bool = False
    visualize_epochs: list[int] = field(default_factory=list)
    plots: list[str] = field(default_factory=lambda: ["branch_info", "ablation"])
    parameters: dict = field(default_factory=dict)


@dataclass
class WandbConfig(BaseConfig):
    entity: str = ""
    project: str = ""
    group: str = ""
    tags: list[str] = field(default_factory=list)
    use_wandb: bool = False


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
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    outputs: OutputsConfig = field(default_factory=OutputsConfig)


def load_config(path: Optional[str] = None) -> Config:
    """Load configuration from a file or use defaults."""
    if path:
        raw_conf = OmegaConf.load(path)
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
