"""
Configuration module for dendritic modeling.

This module contains configuration dataclasses and utilities.
"""

from dendritic_modeling.config.analysis import (
    AblationAnalysisConfig,
    AblationAnalysisParams,
    AnalysisConfig,
    BranchActivationAnalysisConfig,
    BranchActivationAnalysisParams,
    CorrelationAnalysisConfig,
    CorrelationAnalysisParams,
    InformationAnalysisConfig,
    InformationAnalysisParams,
    NoisePerturbationAnalysisConfig,
    NoisePerturbationAnalysisParams,
    PerformanceAnalysisConfig,
    PerformanceAnalysisParams,
    SingleLayerContributionAnalysisConfig,
    SingleLayerContributionAnalysisParams,
    SynapseTurnoverAnalysisConfig,
    SynapticActivationAnalysisConfig,
    SynapticActivationAnalysisParams,
    WeightAnalysisConfig,
    WeightAnalysisParams,
)
from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.config import (
    Config,
    OutputsConfig,
    TaskConfig,
    WandbConfig,
    load_config,
)
from dendritic_modeling.config.model import (
    CoreNetworkConfig,
    DecoderNetworkConfig,
    EINetParameters,
    EncoderNetworkConfig,
    EncoderNetworkParameters,
    ModelConfig,
    TransferFunctionParams,
)
from dendritic_modeling.config.training import (
    EncoderTrainConfig,
    MainTrainConfig,
    ParamGroupsConfig,
    TrainConfig,
    TrainerConfig,
)

__all__ = [
    "AblationAnalysisConfig",
    "AblationAnalysisParams",
    "AnalysisConfig",
    "BaseConfig",
    "BranchActivationAnalysisConfig",
    "BranchActivationAnalysisParams",
    "Config",
    "CoreNetworkConfig",
    "CorrelationAnalysisConfig",
    "CorrelationAnalysisParams",
    "DecoderNetworkConfig",
    "EINetParameters",
    "EncoderNetworkConfig",
    "EncoderNetworkParameters",
    "EncoderTrainConfig",
    "InformationAnalysisConfig",
    "InformationAnalysisParams",
    "load_config",
    "MainTrainConfig",
    "ModelConfig",
    "NoisePerturbationAnalysisConfig",
    "NoisePerturbationAnalysisParams",
    "OutputsConfig",
    "ParamGroupsConfig",
    "PerformanceAnalysisConfig",
    "PerformanceAnalysisParams",
    "SingleLayerContributionAnalysisConfig",
    "SingleLayerContributionAnalysisParams",
    "SynapseTurnoverAnalysisConfig",
    "SynapticActivationAnalysisConfig",
    "SynapticActivationAnalysisParams",
    "TaskConfig",
    "TrainConfig",
    "TrainerConfig",
    "TransferFunctionParams",
    "WandbConfig",
    "WeightAnalysisConfig",
    "WeightAnalysisParams",
]
