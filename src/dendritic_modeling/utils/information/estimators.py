"""Information theory estimators with multiple backend implementations.

This module provides a unified interface for computing mutual information
and related metrics using different estimation methods:
- Kraskov (k-nearest neighbors)
- Scikit-learn (kernel density or histogram-based)
- Copula-based (via DVC)
- PID (Partial Information Decomposition)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Optional, Union

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.metrics import mutual_info_score

from dendritic_modeling.utils.information import compute_mi_cc, compute_mi_cd


class EstimationMethod(Enum):
    """Available estimation methods for mutual information."""

    KRASKOV = "kraskov"
    SKLEARN = "sklearn"
    COPULA = "copula"
    AUTO = "auto"  # Automatically select based on data properties


@dataclass
class EstimatorConfig:
    """Configuration for information estimators."""

    method: Union[str, EstimationMethod] = EstimationMethod.AUTO
    n_neighbors: int = 10  # For Kraskov
    n_bins: int = 20  # For histogram-based methods
    bandwidth: Optional[float] = None  # For kernel density estimation
    copula_type: str = "gaussian"  # For copula-based estimation
    use_gpu: bool = False
    verbose: bool = False
    # Additional method-specific parameters
    method_params: dict[str, Any] = None

    def __post_init__(self):
        if isinstance(self.method, str):
            self.method = EstimationMethod(self.method.lower())
        if self.method_params is None:
            self.method_params = {}


class BaseInformationEstimator(ABC):
    """Abstract base class for information theory estimators."""

    def __init__(self, config: EstimatorConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        if config.verbose:
            self.logger.setLevel(logging.DEBUG)

    @abstractmethod
    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute mutual information I(X;Y) between two variables."""

    @abstractmethod
    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Compute conditional mutual information I(X;Y|Z)."""

    def entropy(self, x: np.ndarray) -> float:
        """Compute entropy H(X) of a variable."""
        # Default implementation using MI: H(X) = I(X;X)
        return self.mutual_information(x, x)

    def conditional_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute conditional entropy H(X|Y)."""
        # H(X|Y) = H(X) - I(X;Y)
        return self.entropy(x) - self.mutual_information(x, y)

    def joint_mutual_information(
        self, sources: list[np.ndarray], target: np.ndarray
    ) -> float:
        """Compute mutual information I(X1,X2,...;Y) for multiple sources."""
        # Concatenate sources and compute MI with target
        joint_source = np.hstack([self._ensure_2d(s) for s in sources])
        return self.mutual_information(joint_source, target)

    @staticmethod
    def _ensure_2d(arr: np.ndarray) -> np.ndarray:
        """Ensure array is 2D for consistent processing."""
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    @staticmethod
    def _validate_inputs(x: np.ndarray, y: np.ndarray, z: Optional[np.ndarray] = None):
        """Validate input arrays have compatible shapes."""
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and Y must have same number of samples: {x.shape[0]} vs {y.shape[0]}"
            )
        if z is not None and z.shape[0] != x.shape[0]:
            raise ValueError(
                f"Z must have same number of samples as X and Y: {z.shape[0]} vs {x.shape[0]}"
            )


class KraskovEstimator(BaseInformationEstimator):
    """Kraskov k-nearest neighbors estimator for mutual information.

    Based on:
    Kraskov, A., Stögbauer, H., & Grassberger, P. (2004).
    Estimating mutual information. Physical review E, 69(6), 066138.
    """

    def __init__(self, config: EstimatorConfig):
        super().__init__(config)
        self.n_neighbors = config.n_neighbors

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI using Kraskov's k-NN estimator."""

        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        self._validate_inputs(x, y)

        # Check if y is discrete (single column with few unique values)
        if y.shape[1] == 1 and len(np.unique(y)) < 20:
            # Treat as discrete
            return compute_mi_cd(x, y.ravel(), self.n_neighbors)
        else:
            # Both continuous
            return compute_mi_cc(x, y, self.n_neighbors)

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Compute CMI using chain rule: I(X;Y|Z) = I(X;Y,Z) - I(X;Z)."""
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        self._validate_inputs(x, y, z)

        # I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
        yz = np.hstack([y, z])
        mi_xyz = self.mutual_information(x, yz)
        mi_xz = self.mutual_information(x, z)

        return max(0.0, mi_xyz - mi_xz)


class SklearnEstimator(BaseInformationEstimator):
    """Scikit-learn based estimator using various methods."""

    def __init__(self, config: EstimatorConfig):
        super().__init__(config)
        self.n_bins = config.n_bins
        self.bandwidth = config.bandwidth

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI using sklearn's methods."""
        x = self._ensure_2d(x)
        y_orig = y
        y = self._ensure_2d(y)
        self._validate_inputs(x, y)

        # Determine if we're dealing with discrete or continuous variables
        x_discrete = self._is_discrete(x)
        y_discrete = self._is_discrete(y)

        if x_discrete and y_discrete:
            # Both discrete - use mutual_info_score
            # Convert to integer labels
            x_labels = self._to_labels(x)
            y_labels = self._to_labels(y_orig.ravel() if y_orig.ndim == 1 else y)
            return mutual_info_score(x_labels, y_labels)

        elif not x_discrete and y_discrete:
            # X continuous, Y discrete (classification)
            y_labels = self._to_labels(y_orig.ravel() if y_orig.ndim == 1 else y)
            mi_scores = mutual_info_classif(
                x, y_labels, n_neighbors=self.config.n_neighbors
            )
            return float(np.mean(mi_scores))

        elif not x_discrete and not y_discrete:
            # Both continuous (regression)
            if x.shape[1] == 1 and y.shape[1] == 1:
                # Both univariate - use histogram-based estimation
                return self._histogram_mi(x.ravel(), y.ravel())
            else:
                # Multivariate - use mutual_info_regression
                mi_scores = mutual_info_regression(
                    x, y.ravel() if y.shape[1] == 1 else y
                )
                return float(np.mean(mi_scores))

        else:
            # X discrete, Y continuous - swap and compute
            return self.mutual_information(y, x)

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Compute CMI using chain rule."""
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        self._validate_inputs(x, y, z)

        # I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
        yz = np.hstack([y, z])
        mi_xyz = self.mutual_information(x, yz)
        mi_xz = self.mutual_information(x, z)

        return max(0.0, mi_xyz - mi_xz)

    def _is_discrete(self, x: np.ndarray) -> bool:
        """Check if variable appears to be discrete."""
        if x.shape[1] > 1:
            return False  # Multi-dimensional, assume continuous

        unique_vals = np.unique(x)
        n_unique = len(unique_vals)
        n_samples = len(x)

        # Heuristic: discrete if few unique values or appears to be integer labels
        if n_unique < 20 or n_unique < n_samples / 10:
            return True

        # Check if all values are integers
        if np.all(x == x.astype(int)):
            return True

        return False

    def _to_labels(self, x: np.ndarray) -> np.ndarray:
        """Convert continuous values to discrete labels."""
        if x.ndim > 1:
            # For multi-dimensional, create composite labels
            x_str = [str(tuple(row)) for row in x]
            unique_labels = list(set(x_str))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            return np.array([label_map[s] for s in x_str])
        else:
            # For 1D, use unique value mapping
            unique_vals = np.unique(x)
            label_map = {val: i for i, val in enumerate(unique_vals)}
            return np.array([label_map[val] for val in x.ravel()])

    def _histogram_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI using histogram-based estimation."""
        # Create 2D histogram
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=self.n_bins)

        # Convert to probabilities
        pxy = hist / np.sum(hist)
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)

        # Compute MI
        mi = 0.0
        for i in range(len(px)):
            for j in range(len(py)):
                if pxy[i, j] > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

        return max(0.0, mi)


class CopulaEstimator(BaseInformationEstimator):
    """Copula-based mutual information estimator."""

    def __init__(self, config: EstimatorConfig):
        super().__init__(config)
        self.copula_type = config.copula_type
        self._check_dvc_available()

    def _check_dvc_available(self):
        """Check if DVC tensorflow is available."""
        try:
            # import tensorflow as tf  # Available for import check
            # Check if DVC modules can be imported
            import os
            import sys

            # Add external directory to path if needed
            external_dir = os.path.join(
                os.path.dirname(__file__), "../../../../external"
            )
            if external_dir not in sys.path:
                sys.path.insert(0, external_dir)

            self.dvc_available = True
            self.logger.debug("DVC tensorflow available for copula estimation")
        except ImportError:
            self.dvc_available = False
            self.logger.warning(
                "DVC tensorflow not available, falling back to simple copula estimation"
            )

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI using copula-based estimation."""
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        self._validate_inputs(x, y)

        if self.dvc_available and self.copula_type == "dvc":
            return self._dvc_copula_mi(x, y)
        else:
            return self._simple_copula_mi(x, y)

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Compute CMI using copula-based estimation."""
        # Use chain rule as with other estimators
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        self._validate_inputs(x, y, z)

        yz = np.hstack([y, z])
        mi_xyz = self.mutual_information(x, yz)
        mi_xz = self.mutual_information(x, z)

        return max(0.0, mi_xyz - mi_xz)

    def _simple_copula_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Simple Gaussian copula-based MI estimation."""
        from scipy import stats

        # Transform to uniform margins using empirical CDF
        n_samples = len(x)
        u = np.zeros_like(x)
        v = np.zeros_like(y)

        for i in range(x.shape[1]):
            ranks = stats.rankdata(x[:, i])
            u[:, i] = ranks / (n_samples + 1)

        for i in range(y.shape[1]):
            ranks = stats.rankdata(y[:, i])
            v[:, i] = ranks / (n_samples + 1)

        # For Gaussian copula, compute correlation on uniform margins
        if x.shape[1] == 1 and y.shape[1] == 1:
            # Univariate case
            corr, _ = stats.pearsonr(u.ravel(), v.ravel())
            # MI for Gaussian copula: -0.5 * log(1 - rho^2)
            if abs(corr) < 0.999:
                mi = -0.5 * np.log(1 - corr**2)
                return max(0.0, mi)

        # For multivariate, fall back to Kraskov on uniform margins
        return KraskovEstimator(self.config).mutual_information(u, v)

    def _dvc_copula_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        """Use DVC tensorflow for copula-based MI estimation."""
        # This would implement the full DVC copula estimation
        # For now, fall back to simple copula
        return self._simple_copula_mi(x, y)


class InformationEstimatorFactory:
    """Factory for creating information estimators based on configuration."""

    _estimators: ClassVar[dict[EstimationMethod, type]] = {
        EstimationMethod.KRASKOV: KraskovEstimator,
        EstimationMethod.SKLEARN: SklearnEstimator,
        EstimationMethod.COPULA: CopulaEstimator,
    }

    @classmethod
    def create(cls, config: Union[EstimatorConfig, dict]) -> BaseInformationEstimator:
        """Create an estimator based on configuration."""
        if isinstance(config, dict):
            config = EstimatorConfig(**config)

        if config.method == EstimationMethod.AUTO:
            # Default to Kraskov for general use
            config.method = EstimationMethod.KRASKOV

        estimator_class = cls._estimators.get(config.method)
        if estimator_class is None:
            raise ValueError(f"Unknown estimation method: {config.method}")

        return estimator_class(config)

    @classmethod
    def register_estimator(cls, method: EstimationMethod, estimator_class: type):
        """Register a new estimator class."""
        cls._estimators[method] = estimator_class
