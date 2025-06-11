"""
Analysis package for dendritic modeling.

This package contains various analysis modules:
- performance: Performance metrics and evaluation
- information: Information theory and mutual information analysis
- synaptic: Synaptic analysis including activation, turnover, and ablation
- manager: Analysis coordination and management
"""

# Import from new structure
from .performance import PerformanceAnalyzer

# Backward compatibility - import remaining classes from original analysis.py
from ..analysis import (
    EIInformationAnalyzer,
    SynapticTurnoverAnalyzer,
    NoisePerturbationAnalyzer,
    BranchInformationAnalyzer,
    SingleLayerContributionAnalyzer,
    AblationAnalyzer,
    WeightAnalyzer,
    SynapticActivationAnalyzer,
    BranchActivationAnalyzer,
    SynapticAblationAnalyzer,
    AnalysisManager,
)

__all__ = [
    'PerformanceAnalyzer',
    'EIInformationAnalyzer',
    'SynapticTurnoverAnalyzer', 
    'NoisePerturbationAnalyzer',
    'BranchInformationAnalyzer',
    'SingleLayerContributionAnalyzer',
    'AblationAnalyzer',
    'WeightAnalyzer',
    'SynapticActivationAnalyzer',
    'BranchActivationAnalyzer',
    'SynapticAblationAnalyzer',
    'AnalysisManager',
] 