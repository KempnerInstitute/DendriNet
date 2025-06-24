"""
Analysis package for dendritic modeling.

This package provides comprehensive analysis tools for evaluating model performance,
synaptic properties, ablation studies, and information flow in dendritic networks.
"""

# Performance analysis
from dendritic_modeling.analysis.performance import PerformanceAnalyzer
from dendritic_modeling.analysis.noise_perturbation import NoisePerturbationAnalyzer
from dendritic_modeling.analysis.ablation import AblationAnalyzer
from dendritic_modeling.analysis.layer_contribution import SingleLayerContributionAnalyzer

# Information analysis
from dendritic_modeling.analysis.information import InformationAnalyzer

# Activation analysis
from dendritic_modeling.analysis.synaptic_activation import SynapticActivationAnalyzer
from dendritic_modeling.analysis.branch_activation import BranchActivationAnalyzer

# Synaptic analysis
from dendritic_modeling.analysis.synaptic_turnover import SynapticTurnoverAnalyzer
from dendritic_modeling.analysis.weight import WeightAnalyzer

# Analysis manager
from dendritic_modeling.analysis.manager import AnalysisManager

# Import new organized analysis classes
from dendritic_modeling.analysis.branch_interactive import InteractiveBranchAnalyzer
from dendritic_modeling.analysis.dendrite_comprehensive import ComprehensiveDendriticAnalyzer
from dendritic_modeling.analysis.managers import PlotManager

__all__ = [
    # Performance analysis
    "PerformanceAnalyzer",
    "NoisePerturbationAnalyzer", 
    "AblationAnalyzer",
    "SingleLayerContributionAnalyzer",
    
    # Information analysis
    "InformationAnalyzer",
    
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
    "PlotManager"
] 