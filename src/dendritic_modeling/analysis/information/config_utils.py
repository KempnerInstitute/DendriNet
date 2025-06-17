"""Configuration utilities for information analysis.

This module provides utilities to convert between the main config system
and the information analysis framework configurations.
"""

from typing import Dict, Union
from dataclasses import asdict

from dendritic_modeling.config import InformationAnalysisConfig
from .estimators import EstimatorConfig, EstimationMethod
from .integrated_analyzer import IntegratedAnalyzerConfig


def create_analyzer_config_from_main_config(
    config: Union[InformationAnalysisConfig, Dict]
) -> IntegratedAnalyzerConfig:
    """Convert main config to analyzer config.
    
    Parameters
    ----------
    config : InformationAnalysisConfig or dict
        Configuration from the main config system
        
    Returns
    -------
    IntegratedAnalyzerConfig
        Configuration for the integrated analyzer
    """
    if isinstance(config, dict):
        config_dict = config
    else:
        config_dict = asdict(config)
    
    # Create estimator config
    estimator_config = EstimatorConfig(
        method=config_dict.get("method", "kraskov"),
        n_neighbors=config_dict.get("n_neighbors", 10),
        n_bins=config_dict.get("n_bins", 20),
        copula_type=config_dict.get("copula_type", "gaussian"),
        verbose=config_dict.get("verbose", False),
    )
    
    # Create analyzer config
    return IntegratedAnalyzerConfig(
        estimator_config=estimator_config,
        compute_basic_mi=config_dict.get("compute_basic_mi", True),
        compute_pairwise_mi=config_dict.get("compute_pairwise_mi", True),
        compute_conditional_mi=config_dict.get("compute_conditional_mi", True),
        compute_pid=config_dict.get("compute_pid", False),
        pid_binarize=config_dict.get("pid_binarize", True),
        pid_binarize_method=config_dict.get("pid_binarize_method", "median"),
        pid_binarize_threshold=config_dict.get("pid_binarize_threshold", 0.5),
        include_vinf=config_dict.get("include_vinf", True),
        normalize_mi=config_dict.get("normalize_mi", False),
        verbose=config_dict.get("verbose", False),
    )


class InformationAnalysisPreset:
    """Preset configurations for common analysis scenarios."""
    
    @staticmethod
    def basic_mi_analysis(method: str = "kraskov") -> Dict:
        """Basic MI analysis without PID."""
        return {
            "method": method,
            "compute_basic_mi": True,
            "compute_pairwise_mi": True,
            "compute_conditional_mi": False,
            "compute_pid": False,
        }
    
    @staticmethod
    def full_analysis(method: str = "kraskov") -> Dict:
        """Complete analysis including all metrics and PID."""
        return {
            "method": method,
            "compute_basic_mi": True,
            "compute_pairwise_mi": True,
            "compute_conditional_mi": True,
            "compute_pid": True,
        }
    
    @staticmethod
    def conditional_analysis(method: str = "kraskov") -> Dict:
        """Focus on conditional MI analysis."""
        return {
            "method": method,
            "compute_basic_mi": False,
            "compute_pairwise_mi": False,
            "compute_conditional_mi": True,
            "compute_pid": False,
        }
    
    @staticmethod
    def pid_analysis(method: str = "kraskov") -> Dict:
        """Focus on PID decomposition."""
        return {
            "method": method,
            "compute_basic_mi": True,  # Needed for PID context
            "compute_pairwise_mi": False,
            "compute_conditional_mi": False,
            "compute_pid": True,
        }


# Example configurations for YAML files
EXAMPLE_YAML_CONFIGS = {
    "kraskov_full": """
# Full analysis with Kraskov method
information_analysis:
  method: kraskov
  n_neighbors: 10
  compute_basic_mi: true
  compute_pairwise_mi: true
  compute_conditional_mi: true
  compute_pid: true
  include_vinf: true
  normalize_mi: false
""",
    
    "sklearn_fast": """
# Fast analysis with sklearn
information_analysis:
  method: sklearn
  n_bins: 20
  compute_basic_mi: true
  compute_pairwise_mi: true
  compute_conditional_mi: false
  compute_pid: false
  normalize_mi: true
""",
    
    "copula_conditional": """
# Conditional MI focus with copula method
information_analysis:
  method: copula
  copula_type: gaussian
  compute_basic_mi: false
  compute_pairwise_mi: false
  compute_conditional_mi: true
  compute_pid: false
""",
    
    "pid_only": """
# PID decomposition only
information_analysis:
  method: kraskov
  n_neighbors: 5
  compute_basic_mi: true  # Required for PID
  compute_pairwise_mi: false
  compute_conditional_mi: false
  compute_pid: true
  pid_binarize: true
  pid_binarize_method: median
""",
} 