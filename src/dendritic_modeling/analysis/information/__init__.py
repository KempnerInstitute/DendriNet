"""
Information analysis subpackage.

This subpackage contains analyzers focused on information theory and mutual information
analysis for understanding information flow in dendritic networks.
"""

from .mutual_info import EIInformationAnalyzer
from .branch_info import BranchInformationAnalyzer

__all__ = [
    "EIInformationAnalyzer",
    "BranchInformationAnalyzer",
] 