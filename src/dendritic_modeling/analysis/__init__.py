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
from .core import (
    AblationAnalyzer,
    BranchActivationAnalyzer,
    InformationAnalyzer
)

# Metrics
from .metrics import (
    PerformanceAnalyzer,
    SingleLayerContributionAnalyzer,
    NoisePerturbationAnalyzer,
    orientation_selectivity_index,
    direction_selectivity_index,
    size_tuning_index,
    length_selectivity_index
)

# Tools
from .tools import (
    CorrelationAnalyzer,
    WeightAnalyzer,
    SynapticActivationAnalyzer,
    SynapticTurnoverAnalyzer
)

# Classifiers
from .classifiers import (
    MLPClassifier,
    train_classifier,
    evaluate_classifier,
    get_layer_activations,
    get_ablated_activations,
    analyze_layerwise_ablation,
    train_and_evaluate_classifier
)

# Manager
from .managers.analysis_manager import AnalysisManager

__all__ = [
    # Core
    "AblationAnalyzer",
    "BranchActivationAnalyzer",
    "InformationAnalyzer",
    # Metrics
    "PerformanceAnalyzer",
    "SingleLayerContributionAnalyzer",
    "NoisePerturbationAnalyzer",
    "orientation_selectivity_index",
    "direction_selectivity_index",
    "size_tuning_index",
    "length_selectivity_index",
    # Tools
    "CorrelationAnalyzer",
    "WeightAnalyzer",
    "SynapticActivationAnalyzer",
    "SynapticTurnoverAnalyzer",
    # Classifiers
    "MLPClassifier",
    "train_classifier",
    "evaluate_classifier",
    "get_layer_activations",
    "get_ablated_activations",
    "analyze_layerwise_ablation",
    "train_and_evaluate_classifier",
    # Manager
    "AnalysisManager"
] 