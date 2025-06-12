"""
Configuration module for dendritic modeling.

This module contains configuration dataclasses and utilities.
"""

from .config import (
    Config,
    ModelConfig,
    TrainConfig,
    TaskConfig,
    AnalysisConfig,
    VisualizationConfig,
    WandbConfig,
    OutputsConfig,
    EINetParameters,
    FeedForwardParameters,
    ModelNetworkConfig,
    load_config,
    BaseConfig,
)

__all__ = [
    "Config",
    "ModelConfig", 
    "TrainConfig",
    "TaskConfig",
    "AnalysisConfig",
    "VisualizationConfig",
    "WandbConfig",
    "OutputsConfig",
    "EINetParameters",
    "FeedForwardParameters",
    "ModelNetworkConfig",
    "load_config",
    "BaseConfig",
] 