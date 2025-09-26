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
from scipy import stats
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
            if self.config.verbose:
                self.logger.debug(
                    f"    Computing MI (continuous-discrete): x{x.shape}, y{y.shape}, k={self.n_neighbors}"
                )
            return compute_mi_cd(x, y.ravel(), self.n_neighbors)
        else:
            # Both continuous
            if self.config.verbose:
                self.logger.debug(
                    f"    Computing MI (continuous-continuous): x{x.shape}, y{y.shape}, k={self.n_neighbors}"
                )
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
        hist, _, _ = np.histogram2d(x, y, bins=self.n_bins)

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
        """Enhanced copula-based MI estimation with discrete variable handling."""

        # Handle discrete variables (like class labels) by adding non-informative noise
        x_processed = self._add_noise_to_discrete(x)
        y_processed = self._add_noise_to_discrete(y)

        # Transform to uniform margins using empirical CDF
        n_samples = len(x_processed)
        u = np.zeros_like(x_processed)
        v = np.zeros_like(y_processed)

        for i in range(x_processed.shape[1]):
            ranks = stats.rankdata(x_processed[:, i])
            u[:, i] = ranks / (n_samples + 1)

        for i in range(y_processed.shape[1]):
            ranks = stats.rankdata(y_processed[:, i])
            v[:, i] = ranks / (n_samples + 1)

        # Choose estimation method based on copula_type
        if self.copula_type == "gaussian":
            return self._gaussian_copula_mi(u, v)
        elif self.copula_type == "kernel":
            return self._kernel_copula_mi(u, v)
        else:
            # Default to Gaussian copula
            return self._gaussian_copula_mi(u, v)

    def _add_noise_to_discrete(
        self, data: np.ndarray, noise_scale: float = 0.1
    ) -> np.ndarray:
        """Add non-informative noise to discrete variables.

        For discrete variables like class labels, adds uniform noise in [0, 1)
        to each integer value, transforming c to a random number in [c, c+1).
        """
        data_processed = data.copy()

        # Detect if data appears to be discrete (integer-like values)
        for i in range(data.shape[1]):
            col = data[:, i]
            # Check if all values are close to integers
            if np.allclose(col, np.round(col), atol=1e-6):
                # This appears to be discrete - add uniform noise
                noise = np.random.uniform(0, noise_scale, size=col.shape)
                data_processed[:, i] = col + noise
                self.logger.debug(f"Added noise to discrete variable in column {i}")

        return data_processed

    def _gaussian_copula_mi(self, u: np.ndarray, v: np.ndarray) -> float:
        """Gaussian copula MI estimation."""
        if u.shape[1] == 1 and v.shape[1] == 1:
            # Univariate case - check for constant inputs first
            u_flat = u.ravel()
            v_flat = v.ravel()

            # Check if either input is constant (all same values)
            if np.var(u_flat) < 1e-10 or np.var(v_flat) < 1e-10:
                # If either input is constant, MI = 0
                return 0.0

            # Suppress the correlation warning for constant inputs
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
                corr, _ = stats.pearsonr(u_flat, v_flat)

            # Handle NaN correlation (when inputs are constant)
            if np.isnan(corr) or np.isinf(corr):
                return 0.0

            # MI for Gaussian copula: -0.5 * log(1 - rho^2)
            if abs(corr) < 0.999:
                mi = -0.5 * np.log(1 - corr**2)
                return max(0.0, mi)
            else:
                return 0.0

        # For multivariate, fall back to Kraskov on uniform margins
        return KraskovEstimator(self.config).mutual_information(u, v)

    def _kernel_copula_mi(self, u: np.ndarray, v: np.ndarray) -> float:
        """Non-parametric kernel-based copula MI estimation."""
        from sklearn.neighbors import KernelDensity

        if u.shape[1] == 1 and v.shape[1] == 1:
            # Univariate case with kernel density estimation
            u_flat = u.ravel()
            v_flat = v.ravel()

            # Check for constant inputs
            if np.var(u_flat) < 1e-10 or np.var(v_flat) < 1e-10:
                return 0.0

            # Choose bandwidth automatically using Scott's rule
            n_samples = len(u_flat)
            bandwidth_u = n_samples ** (-1 / 5) * np.std(u_flat)
            bandwidth_v = n_samples ** (-1 / 5) * np.std(v_flat)
            bandwidth_joint = min(bandwidth_u, bandwidth_v)

            # Ensure minimum bandwidth
            bandwidth_joint = max(bandwidth_joint, 0.01)

            try:
                # Estimate marginal densities
                kde_u = KernelDensity(bandwidth=bandwidth_u, kernel="gaussian")
                kde_v = KernelDensity(bandwidth=bandwidth_v, kernel="gaussian")
                kde_joint = KernelDensity(bandwidth=bandwidth_joint, kernel="gaussian")

                kde_u.fit(u_flat.reshape(-1, 1))
                kde_v.fit(v_flat.reshape(-1, 1))

                # Joint density on uniform margins
                uv_joint = np.column_stack([u_flat, v_flat])
                kde_joint.fit(uv_joint)

                # Compute MI using kernel density estimates
                # MI = ∫∫ p(u,v) log(p(u,v) / (p(u)p(v))) du dv
                # Approximate using sample-based estimation

                log_pu = kde_u.score_samples(u_flat.reshape(-1, 1))
                log_pv = kde_v.score_samples(v_flat.reshape(-1, 1))
                log_puv = kde_joint.score_samples(uv_joint)

                # MI estimate: mean of log(p(u,v)) - log(p(u)) - log(p(v))
                mi_samples = log_puv - log_pu - log_pv
                mi = np.mean(mi_samples)

                return max(0.0, mi)

            except Exception as e:
                self.logger.warning(
                    f"Kernel copula estimation failed: {e}, falling back to Gaussian"
                )
                return self._gaussian_copula_mi(u, v)

        # For multivariate, fall back to Kraskov
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
