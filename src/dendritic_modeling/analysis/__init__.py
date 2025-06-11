"""
Analysis package for dendritic modeling.

This package provides comprehensive analysis tools for evaluating model performance,
synaptic properties, ablation studies, and information flow in dendritic networks.
"""

# Performance analysis
from .performance.metrics import PerformanceAnalyzer
from .performance.noise import NoisePerturbationAnalyzer
from .performance.ablation import AblationAnalyzer

# Information analysis
from .information.mutual_info import EIInformationAnalyzer

# Synaptic analysis
from .synaptic.turnover import SynapticTurnoverAnalyzer
from .synaptic.weights import WeightAnalyzer
from .synaptic.activations import SynapticActivationAnalyzer
from .synaptic.branch_activations import BranchActivationAnalyzer

# Main analysis classes from the analysis module
from .analysis import (
    BranchInformationAnalyzer,
    SingleLayerContributionAnalyzer, 
    SynapticAblationAnalyzer,
    AnalysisManager,
)

__all__ = [
    # Performance analysis
    "PerformanceAnalyzer",
    "NoisePerturbationAnalyzer", 
    "AblationAnalyzer",
    
    # Information analysis
    "EIInformationAnalyzer",
    
    # Synaptic analysis
    "SynapticTurnoverAnalyzer",
    "WeightAnalyzer",
    "SynapticActivationAnalyzer",
    "BranchActivationAnalyzer",
    
    # Main analysis classes
    "BranchInformationAnalyzer",
    "SingleLayerContributionAnalyzer",
    "SynapticAblationAnalyzer", 
    "AnalysisManager",
] 