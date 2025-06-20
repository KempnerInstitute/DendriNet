"""
Information analysis subpackage.

This subpackage provides unified information-theoretic analysis for dendritic networks,
including mutual information, conditional mutual information, and partial information decomposition.
Uses multivariate methods from information.py as the core computation engine.
"""

# Core computation functions (multivariate-aware)
from dendritic_modeling.analysis.information.information import (
    compute_mi_cc,
    compute_mi_cd,
    compute_EI_MI,
    compute_pairwise_sr,
    compute_pairwise_sr_with_mi,
    build_sr_matrix,
    compute_synaptic_information,
)

# PID utilities
from dendritic_modeling.analysis.information.pid_utils import (
    check_pid_available,
    pid_synergy_redundancy,
)

# Main integrated analyzer
from dendritic_modeling.analysis.information.integrated_analyzer import InformationAnalyzer

__all__ = [
    # Core computation functions
    "compute_mi_cc",
    "compute_mi_cd",
    "compute_EI_MI",
    "compute_pairwise_sr",
    "compute_pairwise_sr_with_mi",
    "build_sr_matrix",
    "compute_synaptic_information",
    # PID utilities
    "check_pid_available",
    "pid_synergy_redundancy",
    # Main analyzer
    "InformationAnalyzer",
] 