"""
Configuration module for dendritic modeling.

This module contains configuration dataclasses and utilities.
"""

from dendritic_modeling.config.base import BaseConfig

from dendritic_modeling.config.model import (
    TransferFunctionParams,
    EINetParameters,
    CoreNetworkConfig,
    EncoderNetworkParameters,
    EncoderNetworkConfig,
    DecoderNetworkConfig,
    ModelConfig
)

from dendritic_modeling.config.training import (
    TrainerConfig,
    TrainConfig
)

from dendritic_modeling.config.analysis import (
    PerformanceAnalysisParams,
    PerformanceAnalysisConfig,
    InformationAnalysisParams,
    InformationAnalysisConfig,
    SynapseTurnoverAnalysisConfig,
    NoisePerturbationAnalysisParams,
    NoisePerturbationAnalysisConfig,
    AblationAnalysisParams,
    AblationAnalysisConfig,
    WeightAnalysisParams,
    WeightAnalysisConfig,
    SynapticActivationAnalysisParams,
    SynapticActivationAnalysisConfig,
    BranchActivationAnalysisParams,
    BranchActivationAnalysisConfig,
    SingleLayerContributionAnalysisParams,
    SingleLayerContributionAnalysisConfig,
    AnalysisConfig
)

from dendritic_modeling.config.config import (
    VisualizationConfig,
    WandbConfig,
    TaskConfig,
    OutputsConfig,
    Config,
    load_config
)

__all__ = [
    "BaseConfig",
    "EINetParameters",
    "CoreNetworkConfig",
    "EncoderNetworkParameters",
    "EncoderNetworkConfig",
    "DecoderNetworkConfig",
    "TransferFunctionParams",
    "ModelConfig",
    "TrainerConfig",
    "TrainConfig",
    "VisualizationConfig",
    "WandbConfig",
    "TaskConfig",
    "OutputsConfig",
    "PerformanceAnalysisParams",
    "PerformanceAnalysisConfig",
    "InformationAnalysisParams",
    "InformationAnalysisConfig",
    "SynapseTurnoverAnalysisConfig",
    "NoisePerturbationAnalysisParams",
    "NoisePerturbationAnalysisConfig",
    "AblationAnalysisParams",
    "AblationAnalysisConfig",
    "WeightAnalysisParams",
    "WeightAnalysisConfig",
    "SynapticActivationAnalysisParams",
    "SynapticActivationAnalysisConfig",
    "BranchActivationAnalysisParams",
    "BranchActivationAnalysisConfig",
    "SingleLayerContributionAnalysisParams",
    "SingleLayerContributionAnalysisConfig",
    "AnalysisConfig",
    "Config",
    "load_config"
]