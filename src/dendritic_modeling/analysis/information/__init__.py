"""
Information analysis subpackage.

This subpackage contains analyzers focused on information theory and mutual information
analysis for understanding information flow in dendritic networks.

The package now provides an integrated framework with multiple estimation methods:
- Kraskov (k-nearest neighbors)
- Scikit-learn (various methods)
- Copula-based estimation
- PID (Partial Information Decomposition)
"""

# Legacy imports for backward compatibility
from .legacy import EIInformationAnalyzer, BranchInformationAnalyzer

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

# New integrated framework imports
from .estimators import (
    BaseInformationEstimator,
    KraskovEstimator,
    SklearnEstimator,
    CopulaEstimator,
    EstimatorConfig,
    EstimationMethod,
    InformationEstimatorFactory,
)

from .integrated_analyzer import (
    IntegratedInformationAnalyzer,
    IntegratedAnalyzerConfig,
)

from .config_utils import (
    create_analyzer_config_from_main_config,
    InformationAnalysisPreset,
    EXAMPLE_YAML_CONFIGS,
)

__all__ = [
    # Legacy analyzer classes
    "EIInformationAnalyzer",
    "BranchInformationAnalyzer",
    # Legacy information theory functions
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
    # New framework - Estimators
    "BaseInformationEstimator",
    "KraskovEstimator",
    "SklearnEstimator",
    "CopulaEstimator",
    "EstimatorConfig",
    "EstimationMethod",
    "InformationEstimatorFactory",
    # New framework - Integrated analyzer
    "IntegratedInformationAnalyzer",
    "IntegratedAnalyzerConfig",
    # New framework - Configuration
    "create_analyzer_config_from_main_config",
    "InformationAnalysisPreset",
    "EXAMPLE_YAML_CONFIGS",
] 