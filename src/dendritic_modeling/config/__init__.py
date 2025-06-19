"""
Configuration module for dendritic modeling.

This module contains configuration dataclasses and utilities.
"""

from dendritic_modeling.config.config import (
    Config,
    ModelConfig,
    TrainConfig,
    TrainerConfig,
    TaskConfig,
    AnalysisConfig,
    InformationAnalysisConfig,
    VisualizationConfig,
    WandbConfig,
    OutputsConfig,
    EINetParameters,
    FeedForwardParameters,
    NetworkConfig,
    BaseModelConfig,
    BaseConfig,
    load_config
)

__all__ = [
    "Config",
    "ModelConfig", 
    "TrainConfig",
    "TrainerConfig",
    "TaskConfig",
    "AnalysisConfig",
    "InformationAnalysisConfig",
    "VisualizationConfig",
    "WandbConfig",
    "OutputsConfig",
    "EINetParameters",
    "FeedForwardParameters",
    "NetworkConfig",
    "BaseModelConfig",
    "BaseConfig",
    "load_config"
] 