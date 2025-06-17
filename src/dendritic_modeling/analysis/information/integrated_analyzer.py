"""Integrated information analyzer for dendritic networks.

This module provides a unified interface for computing various information-theoretic
metrics using the multivariate methods from information.py. It integrates seamlessly
with the main configuration system.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import logging

from .estimators import (
    BaseInformationEstimator,
    EstimatorConfig,
    InformationEstimatorFactory
)
from .pid_utils import check_pid_available, pid_synergy_redundancy


class IntegratedInformationAnalyzer:
    """Unified analyzer for computing information-theoretic metrics in dendritic networks.
    
    This analyzer uses the multivariate-aware methods from information.py to compute
    mutual information between excitatory inputs, inhibitory inputs, branch outputs,
    and class labels. It supports continuous multivariate data natively.
    """
    
    def __init__(self, config: Union[Dict, Any]):
        """Initialize the analyzer with configuration from main YAML config.
        
        Parameters
        ----------
        config : InformationAnalysisConfig or dict
            Configuration from the main config system (analysis.information_analysis)
        """
        # Handle both dict and dataclass configs
        if hasattr(config, '__dict__'):
            # It's a dataclass, convert to dict
            config_dict = {k: v for k, v in config.__dict__.items()}
        else:
            config_dict = config
            
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Extract configuration parameters
        self.method = config_dict.get("method", "kraskov")
        self.n_neighbors = config_dict.get("n_neighbors", 10)
        self.n_bins = config_dict.get("n_bins", 20)
        self.copula_type = config_dict.get("copula_type", "gaussian")
        
        self.compute_basic_mi = config_dict.get("compute_basic_mi", True)
        self.compute_pairwise_mi = config_dict.get("compute_pairwise_mi", True)
        self.compute_conditional_mi = config_dict.get("compute_conditional_mi", True)
        self.compute_pid = config_dict.get("compute_pid", False)
        
        self.pid_binarize = config_dict.get("pid_binarize", True)
        self.pid_binarize_method = config_dict.get("pid_binarize_method", "median")
        self.pid_binarize_threshold = config_dict.get("pid_binarize_threshold", 0.5)
        
        self.include_vinf = config_dict.get("include_vinf", True)
        self.normalize_mi = config_dict.get("normalize_mi", False)
        self.verbose = config_dict.get("verbose", False)
        
        if self.verbose:
            self.logger.setLevel(logging.DEBUG)
        
        # Create the estimator
        estimator_config = EstimatorConfig(
            method=self.method,
            n_neighbors=self.n_neighbors,
            n_bins=self.n_bins,
            copula_type=self.copula_type,
            verbose=self.verbose,
        )
        self.estimator = InformationEstimatorFactory.create(estimator_config)
        
        # Check if PID is available
        self.pid_available = False
        if self.compute_pid:
            try:
                check_pid_available()
                self.pid_available = True
                self.logger.info("PID computation available")
            except ImportError:
                self.logger.warning("PID package not available. PID metrics will be skipped.")
    
    def analyze(
        self,
        E: np.ndarray,
        I: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute all requested information metrics.
        
        Parameters
        ----------
        E : np.ndarray
            Excitatory inputs, shape (n_samples,) or (n_samples, n_features)
        I : np.ndarray
            Inhibitory inputs, shape (n_samples,) or (n_samples, n_features)
        Vout : np.ndarray
            Branch outputs, shape (n_samples,) or (n_samples, n_features)
        C : np.ndarray
            Class labels, shape (n_samples,)
        Vinf : np.ndarray, optional
            Soma activation (Vinf), shape (n_samples,) or (n_samples, n_features)
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing all computed metrics
        """
        # Validate inputs
        self._validate_inputs(E, I, Vout, C, Vinf)
        
        # Ensure all arrays are properly shaped
        E = self._ensure_2d(E)
        I = self._ensure_2d(I)
        Vout = self._ensure_2d(Vout)
        C = C.ravel()  # Labels are always 1D
        
        if Vinf is not None:
            Vinf = self._ensure_2d(Vinf)
        
        results = {
            "method": self.method,
            "n_samples": len(E),
        }
        
        # 1. Basic mutual information with C
        if self.compute_basic_mi:
            self.logger.info("Computing basic MI metrics...")
            results["basic_mi"] = self._compute_basic_mi(E, I, Vout, C, Vinf)
        
        # 2. Pairwise mutual information
        if self.compute_pairwise_mi:
            self.logger.info("Computing pairwise MI metrics...")
            results["pairwise_mi"] = self._compute_pairwise_mi(E, I, Vout, Vinf)
        
        # 3. Conditional mutual information
        if self.compute_conditional_mi:
            self.logger.info("Computing conditional MI metrics...")
            results["conditional_mi"] = self._compute_conditional_mi(E, I, Vout, C, Vinf)
        
        # 4. PID decomposition
        if self.compute_pid and self.pid_available:
            self.logger.info("Computing PID metrics...")
            results["pid"] = self._compute_pid(E, I, Vout, C)
        
        return results
    
    def _compute_basic_mi(
        self,
        E: np.ndarray,
        I: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute mutual information between each variable and C."""
        metrics = {}
        
        # I(E;C)
        metrics["I(E;C)"] = self.estimator.mutual_information(E, C)
        
        # I(I;C)
        metrics["I(I;C)"] = self.estimator.mutual_information(I, C)
        
        # I(Vout;C)
        metrics["I(Vout;C)"] = self.estimator.mutual_information(Vout, C)
        
        # I(Vinf;C) if available
        if Vinf is not None and self.include_vinf:
            metrics["I(Vinf;C)"] = self.estimator.mutual_information(Vinf, C)
        
        # Joint MI: I(E,I;C)
        EI = np.hstack([E, I])
        metrics["I(E,I;C)"] = self.estimator.mutual_information(EI, C)
        
        # Optionally normalize by entropy
        if self.normalize_mi:
            H_C = self.estimator.entropy(C.reshape(-1, 1))
            # Create a list of keys to avoid modifying dict during iteration
            keys_to_normalize = list(metrics.keys())
            for key in keys_to_normalize:
                metrics[f"{key}_normalized"] = metrics[key] / H_C if H_C > 0 else 0.0
        
        return metrics
    
    def _compute_pairwise_mi(
        self,
        E: np.ndarray,
        I: np.ndarray,
        Vout: np.ndarray,
        Vinf: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute pairwise mutual information between variables."""
        metrics = {}
        
        # I(E;I)
        metrics["I(E;I)"] = self.estimator.mutual_information(E, I)
        
        # I(E;Vout)
        metrics["I(E;Vout)"] = self.estimator.mutual_information(E, Vout)
        
        # I(I;Vout)
        metrics["I(I;Vout)"] = self.estimator.mutual_information(I, Vout)
        
        if Vinf is not None and self.include_vinf:
            # I(E;Vinf)
            metrics["I(E;Vinf)"] = self.estimator.mutual_information(E, Vinf)
            
            # I(I;Vinf)
            metrics["I(I;Vinf)"] = self.estimator.mutual_information(I, Vinf)
            
            # I(Vout;Vinf)
            metrics["I(Vout;Vinf)"] = self.estimator.mutual_information(Vout, Vinf)
        
        return metrics
    
    def _compute_conditional_mi(
        self,
        E: np.ndarray,
        I: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Compute conditional mutual information given C."""
        metrics = {}
        
        # I(E;I|C)
        metrics["I(E;I|C)"] = self.estimator.conditional_mutual_information(E, I, C.reshape(-1, 1))
        
        # I(E;Vout|C)
        metrics["I(E;Vout|C)"] = self.estimator.conditional_mutual_information(E, Vout, C.reshape(-1, 1))
        
        # I(I;Vout|C)
        metrics["I(I;Vout|C)"] = self.estimator.conditional_mutual_information(I, Vout, C.reshape(-1, 1))
        
        if Vinf is not None and self.include_vinf:
            # I(E;Vinf|C)
            metrics["I(E;Vinf|C)"] = self.estimator.conditional_mutual_information(E, Vinf, C.reshape(-1, 1))
            
            # I(I;Vinf|C)
            metrics["I(I;Vinf|C)"] = self.estimator.conditional_mutual_information(I, Vinf, C.reshape(-1, 1))
        
        return metrics
    
    def _compute_pid(
        self,
        E: np.ndarray,
        I: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray
    ) -> Dict[str, Any]:
        """Compute Partial Information Decomposition."""
        pid_results = {}
        
        # Prepare data for PID (may need binarization)
        if self.pid_binarize:
            E_pid = self._binarize(E)
            I_pid = self._binarize(I)
            Vout_pid = self._binarize(Vout)
            C_pid = self._binarize(C.reshape(-1, 1))
        else:
            E_pid = E
            I_pid = I
            Vout_pid = Vout
            C_pid = C.reshape(-1, 1)
        
        # PID: (E, I) -> C
        try:
            synergy, redundancy = pid_synergy_redundancy([E_pid, I_pid], C_pid)
            pid_results["PID(E,I->C)"] = {
                "synergy": synergy,
                "redundancy": redundancy,
                "unique_E": self.estimator.mutual_information(E_pid, C_pid) - redundancy,
                "unique_I": self.estimator.mutual_information(I_pid, C_pid) - redundancy,
            }
        except Exception as e:
            self.logger.warning(f"PID(E,I->C) computation failed: {e}")
        
        # PID: (E, Vout) -> C
        try:
            synergy, redundancy = pid_synergy_redundancy([E_pid, Vout_pid], C_pid)
            pid_results["PID(E,Vout->C)"] = {
                "synergy": synergy,
                "redundancy": redundancy,
                "unique_E": self.estimator.mutual_information(E_pid, C_pid) - redundancy,
                "unique_Vout": self.estimator.mutual_information(Vout_pid, C_pid) - redundancy,
            }
        except Exception as e:
            self.logger.warning(f"PID(E,Vout->C) computation failed: {e}")
        
        # PID: (I, Vout) -> C
        try:
            synergy, redundancy = pid_synergy_redundancy([I_pid, Vout_pid], C_pid)
            pid_results["PID(I,Vout->C)"] = {
                "synergy": synergy,
                "redundancy": redundancy,
                "unique_I": self.estimator.mutual_information(I_pid, C_pid) - redundancy,
                "unique_Vout": self.estimator.mutual_information(Vout_pid, C_pid) - redundancy,
            }
        except Exception as e:
            self.logger.warning(f"PID(I,Vout->C) computation failed: {e}")
        
        # PID: (E, I) -> Vout
        try:
            synergy, redundancy = pid_synergy_redundancy([E_pid, I_pid], Vout_pid)
            pid_results["PID(E,I->Vout)"] = {
                "synergy": synergy,
                "redundancy": redundancy,
                "unique_E": self.estimator.mutual_information(E_pid, Vout_pid) - redundancy,
                "unique_I": self.estimator.mutual_information(I_pid, Vout_pid) - redundancy,
            }
        except Exception as e:
            self.logger.warning(f"PID(E,I->Vout) computation failed: {e}")
        
        return pid_results
    
    def _binarize(self, data: np.ndarray) -> np.ndarray:
        """Binarize data for PID computation."""
        if self.pid_binarize_method == "median":
            threshold = np.median(data, axis=0)
        elif self.pid_binarize_method == "mean":
            threshold = np.mean(data, axis=0)
        elif self.pid_binarize_method == "quantile":
            threshold = np.quantile(data, self.pid_binarize_threshold, axis=0)
        else:
            raise ValueError(f"Unknown binarization method: {self.pid_binarize_method}")
        
        return (data > threshold).astype(int)
    
    def _validate_inputs(
        self,
        E: np.ndarray,
        I: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: Optional[np.ndarray] = None
    ):
        """Validate input arrays."""
        n_samples = len(E)
        
        if len(I) != n_samples:
            raise ValueError(f"I must have same number of samples as E: {len(I)} vs {n_samples}")
        
        if len(Vout) != n_samples:
            raise ValueError(f"Vout must have same number of samples as E: {len(Vout)} vs {n_samples}")
        
        if len(C) != n_samples:
            raise ValueError(f"C must have same number of samples as E: {len(C)} vs {n_samples}")
        
        if Vinf is not None and len(Vinf) != n_samples:
            raise ValueError(f"Vinf must have same number of samples as E: {len(Vinf)} vs {n_samples}")
    
    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D."""
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr
    
    def get_summary(self, results: Dict[str, Any]) -> str:
        """Generate a summary string of the results."""
        summary_lines = [
            f"Information Analysis Summary",
            f"Method: {results['method']}",
            f"Samples: {results['n_samples']}",
            ""
        ]
        
        if "basic_mi" in results:
            summary_lines.append("Basic MI with C:")
            for key, value in results["basic_mi"].items():
                if not key.endswith("_normalized"):
                    summary_lines.append(f"  {key}: {value:.4f}")
        
        if "pairwise_mi" in results:
            summary_lines.append("\nPairwise MI:")
            for key, value in results["pairwise_mi"].items():
                summary_lines.append(f"  {key}: {value:.4f}")
        
        if "conditional_mi" in results:
            summary_lines.append("\nConditional MI given C:")
            for key, value in results["conditional_mi"].items():
                summary_lines.append(f"  {key}: {value:.4f}")
        
        if "pid" in results:
            summary_lines.append("\nPID Results:")
            for pid_key, pid_values in results["pid"].items():
                summary_lines.append(f"  {pid_key}:")
                for metric, value in pid_values.items():
                    summary_lines.append(f"    {metric}: {value:.4f}")
        
        return "\n".join(summary_lines) 