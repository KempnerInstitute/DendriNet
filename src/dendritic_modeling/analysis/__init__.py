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
from .information.branch_info import BranchInformationAnalyzer

# Activation analysis
from .activations.synaptic import SynapticActivationAnalyzer
from .activations.branches import BranchActivationAnalyzer

# Synaptic analysis
from .synaptic.turnover import SynapticTurnoverAnalyzer
from .synaptic.weights import WeightAnalyzer

# Main analysis classes from the analysis module
from .analysis import (
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
    "BranchInformationAnalyzer",
    
    # Activation analysis
    "SynapticActivationAnalyzer",
    "BranchActivationAnalyzer",
    
    # Synaptic analysis
    "SynapticTurnoverAnalyzer",
    "WeightAnalyzer",
    
    # Main analysis classes
    "SingleLayerContributionAnalyzer",
    "SynapticAblationAnalyzer", 
    "AnalysisManager",
] 