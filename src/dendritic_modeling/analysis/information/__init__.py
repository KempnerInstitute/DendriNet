"""
Information analysis subpackage.

This subpackage contains analyzers focused on information theory and mutual information
analysis for understanding information flow in dendritic networks.
"""

from .mutual_info import EIInformationAnalyzer
from .branch_info import BranchInformationAnalyzer

# Import from information.py
from .information import (
    compute_mi_cc,
    compute_mi_cd,
    compute_EI_MI,
    compute_pairwise_sr,
    compute_pairwise_sr_with_mi,
    build_sr_matrix,
    compute_synaptic_information,
)

# Import from pid_utils.py
from .pid_utils import (
    check_pid_available,
    pid_synergy_redundancy,
)

__all__ = [
    # Analyzer classes
    "EIInformationAnalyzer",
    "BranchInformationAnalyzer",
    # Information theory functions
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
] 