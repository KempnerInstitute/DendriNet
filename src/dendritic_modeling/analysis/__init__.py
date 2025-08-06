"""
Analysis Module
===============

This module provides comprehensive analysis tools for dendritic networks, including:

Core Analysis:
- Ablation studies
- Branch activation analysis
- Information-theoretic analysis

Metrics:
- Performance metrics
- Layer contribution analysis
- Noise perturbation analysis
- Visual neuron metrics

Tools:
- Correlation analysis
- Weight analysis
- Synaptic analysis

Classifiers:
- MLP classifiers for activation decoding
"""

# Core analyzers
# Classifiers
from .core import AblationAnalyzer, BranchActivationAnalyzer, InformationAnalyzer

# Manager
from .managers.analysis_manager import AnalysisManager

# Metrics
from .metrics import (
    NoisePerturbationAnalyzer,
    PerformanceAnalyzer,
    SingleLayerContributionAnalyzer,
    direction_selectivity_index,
    length_selectivity_index,
    orientation_selectivity_index,
    size_tuning_index,
)

# Tools
from .tools import (
    CorrelationAnalyzer,
    SynapticActivationAnalyzer,
    SynapticTurnoverAnalyzer,
    WeightAnalyzer,
)

__all__ = [
    # Core
    "AblationAnalyzer"
    # Manager
    "AnalysisManager"
    "BranchActivationAnalyzer"
    # Tools
    "CorrelationAnalyzer"
    "direction_selectivity_index"
    "InformationAnalyzer"
    "length_selectivity_index"
    # Metrics
    "NoisePerturbationAnalyzer"
    "orientation_selectivity_index"
    "PerformanceAnalyzer"
    "SingleLayerContributionAnalyzer"
    "size_tuning_index"
    "SynapticActivationAnalyzer"
    "SynapticTurnoverAnalyzer"
    "WeightAnalyzer"
]
