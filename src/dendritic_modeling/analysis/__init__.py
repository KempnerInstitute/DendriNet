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
from .classifiers import (
    MLPClassifier,
    analyze_layerwise_ablation,
    evaluate_classifier,
    get_ablated_activations,
    get_layer_activations,
    train_and_evaluate_classifier,
    train_classifier,
)
from .core import (
    AblationAnalyzer,
    BranchActivationAnalyzer,
    InformationAnalyzer,
)

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
    "AblationAnalyzer",
    # Manager
    "AnalysisManager",
    "BranchActivationAnalyzer",
    # Tools
    "CorrelationAnalyzer",
    "InformationAnalyzer",
    # Classifiers
    "MLPClassifier",
    # Metrics
    "NoisePerturbationAnalyzer",
    "PerformanceAnalyzer",
    "SingleLayerContributionAnalyzer",
    "SynapticActivationAnalyzer",
    "SynapticTurnoverAnalyzer",
    "WeightAnalyzer",
    "analyze_layerwise_ablation",
    "direction_selectivity_index",
    "evaluate_classifier",
    "get_ablated_activations",
    "get_layer_activations",
    "length_selectivity_index",
    "orientation_selectivity_index",
    "size_tuning_index",
    "train_and_evaluate_classifier",
    "train_classifier",
]
