"""Configuration utilities for information analysis.

This module provides convenient configuration builders and presets
for different information analysis scenarios.
"""

from typing import Dict, Optional, Union
from dataclasses import dataclass, field, asdict

from .estimators import EstimationMethod, EstimatorConfig
from .integrated_analyzer import IntegratedAnalyzerConfig


@dataclass
class InformationAnalysisPreset:
    """Preset configurations for common analysis scenarios."""
    
    @staticmethod
    def basic_mi_analysis(method: Union[str, EstimationMethod] = "kraskov") -> IntegratedAnalyzerConfig:
        """Basic MI analysis without PID."""
        return IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method=method),
            compute_basic_mi=True,
            compute_pairwise_mi=True,
            compute_conditional_mi=False,
            compute_pid=False,
        )
    
    @staticmethod
    def full_analysis(method: Union[str, EstimationMethod] = "kraskov") -> IntegratedAnalyzerConfig:
        """Complete analysis including all metrics and PID."""
        return IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method=method),
            compute_basic_mi=True,
            compute_pairwise_mi=True,
            compute_conditional_mi=True,
            compute_pid=True,
        )
    
    @staticmethod
    def conditional_analysis(method: Union[str, EstimationMethod] = "kraskov") -> IntegratedAnalyzerConfig:
        """Focus on conditional MI analysis."""
        return IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method=method),
            compute_basic_mi=False,
            compute_pairwise_mi=False,
            compute_conditional_mi=True,
            compute_pid=False,
        )
    
    @staticmethod
    def pid_analysis(method: Union[str, EstimationMethod] = "kraskov") -> IntegratedAnalyzerConfig:
        """Focus on PID decomposition."""
        return IntegratedAnalyzerConfig(
            estimator_config=EstimatorConfig(method=method),
            compute_basic_mi=True,  # Needed for PID context
            compute_pairwise_mi=False,
            compute_conditional_mi=False,
            compute_pid=True,
        )


@dataclass
class ConfigBuilder:
    """Builder for creating custom configurations."""
    
    # Core settings
    method: Union[str, EstimationMethod] = "auto"
    
    # Metric selection
    compute_basic_mi: bool = True
    compute_pairwise_mi: bool = True
    compute_conditional_mi: bool = True
    compute_pid: bool = True
    
    # Method-specific parameters
    n_neighbors: int = 10  # Kraskov
    n_bins: int = 20  # Sklearn histogram
    copula_type: str = "gaussian"  # Copula
    
    # PID settings
    pid_binarize: bool = True
    pid_binarize_method: str = "median"
    
    # Options
    include_vinf: bool = True
    normalize_mi: bool = False
    verbose: bool = False
    
    def with_method(self, method: Union[str, EstimationMethod]) -> "ConfigBuilder":
        """Set the estimation method."""
        self.method = method
        return self
    
    def with_metrics(
        self,
        basic: bool = True,
        pairwise: bool = True,
        conditional: bool = True,
        pid: bool = True
    ) -> "ConfigBuilder":
        """Configure which metrics to compute."""
        self.compute_basic_mi = basic
        self.compute_pairwise_mi = pairwise
        self.compute_conditional_mi = conditional
        self.compute_pid = pid
        return self
    
    def with_kraskov_params(self, n_neighbors: int = 10) -> "ConfigBuilder":
        """Configure Kraskov-specific parameters."""
        self.n_neighbors = n_neighbors
        return self
    
    def with_sklearn_params(self, n_bins: int = 20) -> "ConfigBuilder":
        """Configure sklearn-specific parameters."""
        self.n_bins = n_bins
        return self
    
    def with_copula_params(self, copula_type: str = "gaussian") -> "ConfigBuilder":
        """Configure copula-specific parameters."""
        self.copula_type = copula_type
        return self
    
    def with_pid_params(
        self,
        binarize: bool = True,
        binarize_method: str = "median"
    ) -> "ConfigBuilder":
        """Configure PID-specific parameters."""
        self.pid_binarize = binarize
        self.pid_binarize_method = binarize_method
        return self
    
    def with_options(
        self,
        include_vinf: bool = True,
        normalize_mi: bool = False,
        verbose: bool = False
    ) -> "ConfigBuilder":
        """Configure additional options."""
        self.include_vinf = include_vinf
        self.normalize_mi = normalize_mi
        self.verbose = verbose
        return self
    
    def build(self) -> IntegratedAnalyzerConfig:
        """Build the final configuration."""
        estimator_config = EstimatorConfig(
            method=self.method,
            n_neighbors=self.n_neighbors,
            n_bins=self.n_bins,
            copula_type=self.copula_type,
            verbose=self.verbose,
        )
        
        return IntegratedAnalyzerConfig(
            estimator_config=estimator_config,
            compute_basic_mi=self.compute_basic_mi,
            compute_pairwise_mi=self.compute_pairwise_mi,
            compute_conditional_mi=self.compute_conditional_mi,
            compute_pid=self.compute_pid,
            pid_binarize=self.pid_binarize,
            pid_binarize_method=self.pid_binarize_method,
            include_vinf=self.include_vinf,
            normalize_mi=self.normalize_mi,
            verbose=self.verbose,
        )


def load_config_from_dict(config_dict: Dict) -> IntegratedAnalyzerConfig:
    """Load configuration from a dictionary (e.g., from YAML/JSON file)."""
    # Handle nested estimator config
    if "estimator_config" in config_dict and isinstance(config_dict["estimator_config"], dict):
        config_dict["estimator_config"] = EstimatorConfig(**config_dict["estimator_config"])
    
    return IntegratedAnalyzerConfig(**config_dict)


def save_config_to_dict(config: IntegratedAnalyzerConfig) -> Dict:
    """Convert configuration to dictionary for saving."""
    config_dict = asdict(config)
    
    # Convert EstimationMethod enum to string
    if "estimator_config" in config_dict and "method" in config_dict["estimator_config"]:
        method = config_dict["estimator_config"]["method"]
        if hasattr(method, "value"):
            config_dict["estimator_config"]["method"] = method.value
    
    return config_dict


# Example configurations for different use cases
EXAMPLE_CONFIGS = {
    "kraskov_full": ConfigBuilder()
        .with_method("kraskov")
        .with_metrics(basic=True, pairwise=True, conditional=True, pid=True)
        .with_kraskov_params(n_neighbors=10)
        .build(),
    
    "sklearn_basic": ConfigBuilder()
        .with_method("sklearn")
        .with_metrics(basic=True, pairwise=True, conditional=False, pid=False)
        .with_sklearn_params(n_bins=20)
        .build(),
    
    "copula_conditional": ConfigBuilder()
        .with_method("copula")
        .with_metrics(basic=False, pairwise=False, conditional=True, pid=False)
        .with_copula_params(copula_type="gaussian")
        .build(),
    
    "fast_analysis": ConfigBuilder()
        .with_method("sklearn")
        .with_metrics(basic=True, pairwise=False, conditional=False, pid=False)
        .with_sklearn_params(n_bins=10)
        .build(),
} 