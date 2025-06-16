"""
Analysis package for dendritic modeling.

This package provides comprehensive analysis tools for evaluating model performance,
synaptic properties, ablation studies, and information flow in dendritic networks.
"""

# Performance analysis
from dendritic_modeling.analysis.performance.metrics import PerformanceAnalyzer
from dendritic_modeling.analysis.performance.noise import NoisePerturbationAnalyzer
from dendritic_modeling.analysis.performance.ablation import AblationAnalyzer
from dendritic_modeling.analysis.performance.layers import SingleLayerContributionAnalyzer
from dendritic_modeling.analysis.performance.synaptic_ablation import SynapticAblationAnalyzer

# Information analysis
from dendritic_modeling.analysis.information.mutual_info import EIInformationAnalyzer
from dendritic_modeling.analysis.information.branch_info import BranchInformationAnalyzer

# Activation analysis
from dendritic_modeling.analysis.activations.synaptic import SynapticActivationAnalyzer
from dendritic_modeling.analysis.activations.branches import BranchActivationAnalyzer

# Synaptic analysis
from dendritic_modeling.analysis.synaptic.turnover import SynapticTurnoverAnalyzer
from dendritic_modeling.analysis.synaptic.weights import WeightAnalyzer

# Analysis manager
from dendritic_modeling.analysis.manager import AnalysisManager

# Import new organized analysis classes
from dendritic_modeling.analysis.branches.interactive import InteractiveBranchAnalyzer
from dendritic_modeling.analysis.dendritic.comprehensive import ComprehensiveDendriticAnalyzer
from dendritic_modeling.analysis.managers.plot_manager import PlotManager

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