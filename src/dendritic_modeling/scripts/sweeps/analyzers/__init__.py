"""Sweep analyzers for different experiment types."""

from pathlib import Path
from typing import Optional

from .base_analyzer import BaseSweepAnalyzer
from .branch_analyzer import BranchSweepAnalyzer
from .ei_analyzer import EISweepAnalyzer
from .stoch_analyzer import StochasticDatasetSweepAnalyzer
from .general_analyzer import GeneralSweepAnalyzer
from .local_learning_analyzer import LocalLearningSweepAnalyzer

# Analyzer registry
ANALYZER_REGISTRY = {
    "ei": EISweepAnalyzer,
    "stoch": StochasticDatasetSweepAnalyzer,
    "local_learning": LocalLearningSweepAnalyzer,
    "branch": BranchSweepAnalyzer,
    "general": GeneralSweepAnalyzer,
}


def get_analyzer(sweep_type: str) -> BaseSweepAnalyzer:
    """
    Get analyzer instance for a sweep type.

    Args:
        sweep_type: Type of sweep

    Returns:
        Analyzer instance

    Raises:
        ValueError: If sweep type not recognized
    """
    analyzer_class = ANALYZER_REGISTRY.get(sweep_type)
    if not analyzer_class:
        raise ValueError(
            f"Unknown sweep type: {sweep_type}. "
            f"Available types: {list(ANALYZER_REGISTRY.keys())}"
        )
    return analyzer_class()


def auto_detect_sweep_type(results_dir: Path) -> Optional[str]:
    """
    Auto-detect sweep type from results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Detected sweep type or None
    """
    # Try each analyzer's detection method
    for sweep_type, analyzer_class in ANALYZER_REGISTRY.items():
        analyzer = analyzer_class()
        if analyzer.can_handle(results_dir):
            return sweep_type

    return None


__all__ = [
    "ANALYZER_REGISTRY",
    "BaseSweepAnalyzer",
    "BranchSweepAnalyzer",
    "EISweepAnalyzer",
    "StochasticDatasetSweepAnalyzer",
    "GeneralSweepAnalyzer",
    "LocalLearningSweepAnalyzer",
    "auto_detect_sweep_type",
    "get_analyzer",
]
