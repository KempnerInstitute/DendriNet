"""
Analysis package for dendritic modeling.

This package provides comprehensive analysis tools for evaluating model performance,
synaptic properties, ablation studies, and information flow in dendritic networks.
"""

# Performance analysis
from .performance.metrics import PerformanceAnalyzer
from .performance.noise import NoisePerturbationAnalyzer
from .performance.ablation import AblationAnalyzer
from .performance.layers import SingleLayerContributionAnalyzer
from .performance.synaptic_ablation import SynapticAblationAnalyzer

# Information analysis
from .information import EIInformationAnalyzer, BranchInformationAnalyzer

# Activation analysis
from .activations.synaptic import SynapticActivationAnalyzer
from .activations.branches import BranchActivationAnalyzer

# Synaptic analysis
from .synaptic.turnover import SynapticTurnoverAnalyzer
from .synaptic.weights import WeightAnalyzer

# Analysis manager
from .manager import AnalysisManager

# Import new organized analysis classes
from .branches.interactive import InteractiveBranchAnalyzer
from .dendritic.comprehensive import ComprehensiveDendriticAnalyzer
from .managers.plot_manager import PlotManager

__all__ = [
    # Performance analysis
    "PerformanceAnalyzer",
    "NoisePerturbationAnalyzer", 
    "AblationAnalyzer",
    "SingleLayerContributionAnalyzer",
    "SynapticAblationAnalyzer",
    
    # Information analysis
    "EIInformationAnalyzer",
    "BranchInformationAnalyzer",
    
    # Activation analysis
    "SynapticActivationAnalyzer",
    "BranchActivationAnalyzer",
    
    # Synaptic analysis
    "SynapticTurnoverAnalyzer",
    "WeightAnalyzer",
    
    # Analysis manager
    "AnalysisManager",

    # New organized analysis classes
    "InteractiveBranchAnalyzer",
    "ComprehensiveDendriticAnalyzer",
    "PlotManager",
] 