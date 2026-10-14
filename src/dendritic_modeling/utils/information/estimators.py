"""Information theory estimators with multiple backend implementations.

This module provides a unified interface for computing mutual information
and related metrics using different estimation methods:
- Kraskov (k-nearest neighbors)
- Decoder (cross-validated classifier log-loss)
- Binned (histogram plug-in)
- Scikit-learn (k-nearest-neighbour MI for discrete targets)
- Copula (Gaussian copula, or a kernel copula density for scalars)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Optional, Union

import numpy as np
from scipy import special, stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, mutual_info_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dendritic_modeling.config.choices import normalize_choice
from dendritic_modeling.utils.information.nearest_neighbors import (
    compute_cmi_ccc,
    compute_mi_cc,
    compute_mi_cd,
)

COPULA_TYPES = ("gaussian", "kernel")


class EstimationMethod(Enum):
    """Available estimation methods for mutual information."""

    KRASKOV = "kraskov"
    DECODER = "decoder"  # Cross-validated decoder-based I(X;C) for discrete targets
    BINNED = "binned"  # Histogram/binning plug-in estimator (best for low-dim)
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
        self.copula_type = normalize_choice(
            self.copula_type, COPULA_TYPES, field="copula_type"
        )


class BaseInformationEstimator(ABC):
    """Abstract base class for information theory estimators."""

    def __init__(self, config: EstimatorConfig):
        self.config = config
        # Inside the package's logger tree, so the records reach the run log.
        self.logger = logging.getLogger(f"{__name__}.{type(self).__name__}")
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

    def _chain_rule_cmi(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        """Compute CMI with the chain rule using this estimator's MI backend."""
        yz = np.hstack([y, z])
        mi_xyz = self.mutual_information(x, yz)
        mi_xz = self.mutual_information(x, z)
        return max(0.0, mi_xyz - mi_xz)

    def mutual_information_null(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        n_shuffles: int = 100,
        seed: int = 0,
        return_values: bool = False,
    ):
        """Permutation (shuffled-null) baseline for MI.

        This is a practical way to estimate finite-sample bias and a null
        distribution: it permutes `y` across samples and recomputes I(X;Y).

        Returns
        -------
        (mean, std) or (mean, std, values) in nats, where values has length n_shuffles.
        """
        n = int(n_shuffles)
        if n <= 0:
            return (
                (0.0, 0.0, np.array([], dtype=float)) if return_values else (0.0, 0.0)
            )

        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        self._validate_inputs(x, y)

        rng = np.random.default_rng(int(seed))
        vals = np.empty(n, dtype=float)
        for i in range(n):
            perm = rng.permutation(y.shape[0])
            vals[i] = float(self.mutual_information(x, y[perm]))

        mean = float(np.mean(vals))
        std = float(np.std(vals))
        return (mean, std, vals) if return_values else (mean, std)

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

    @staticmethod
    def _is_discrete_1d(values: np.ndarray) -> bool:
        """Heuristic: decide whether a 1D array should be treated as discrete.

        We treat a variable as discrete if:
        - It has a small number of unique values (e.g. class labels), OR
        - It is integer-like with a moderate number of unique values.
        """
        values = np.asarray(values).ravel()
        if values.size == 0:
            return False
        unique = np.unique(values)
        if unique.size <= 1:
            return True
        # Small-unique AND clearly not continuous-like (unique << n_samples)
        if unique.size <= 20 and unique.size <= max(5, values.size // 5):
            return True
        # Integer-like labels (classification), allow more uniques.
        if unique.size <= 200 and np.allclose(unique, np.round(unique), atol=1e-6):
            return True
        return False

    @staticmethod
    def _to_int_labels(values: np.ndarray) -> np.ndarray:
        """Map arbitrary 1D values to contiguous integer labels [0..K-1]."""
        values = np.asarray(values).ravel()
        unique = np.unique(values)
        # np.unique returns sorted uniques; searchsorted is fast and stable.
        return np.searchsorted(unique, values).astype(int)

    def _mutual_information_with_k(self, x: np.ndarray, y: np.ndarray, k: int) -> float:
        """Mutual information with an explicit k (neighbors) parameter."""
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        self._validate_inputs(x, y)

        x_is_discrete = x.shape[1] == 1 and self._is_discrete_1d(x[:, 0])
        y_is_discrete = y.shape[1] == 1 and self._is_discrete_1d(y[:, 0])

        # Handle discrete-discrete exactly (in nats).
        if x_is_discrete and y_is_discrete:
            x_lab = self._to_int_labels(x.ravel())
            y_lab = self._to_int_labels(y.ravel())
            return float(mutual_info_score(x_lab, y_lab))

        # Handle mixed continuous/discrete using Ross (2014).
        if x_is_discrete and not y_is_discrete:
            # compute_mi_cd expects (continuous, discrete)
            return compute_mi_cd(y, self._to_int_labels(x.ravel()), k)
        if not x_is_discrete and y_is_discrete:
            return compute_mi_cd(x, self._to_int_labels(y.ravel()), k)

        # Continuous-continuous KSG.
        return compute_mi_cc(x, y, k)

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute MI using Kraskov's k-NN estimator."""
        if self.config.verbose:
            self.logger.debug(
                f"    Computing MI: x{np.asarray(x).shape}, y{np.asarray(y).shape}, k={self.n_neighbors}"
            )
        return self._mutual_information_with_k(x, y, self.n_neighbors)

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Compute CMI I(X;Y|Z) with discrete-aware handling.

        Cases handled explicitly:
        - Z discrete (e.g. class labels): I(X;Y|Z) = Σ_z p(z) I(X;Y | Z=z)
        - Y discrete, Z continuous: I(X;Y|Z) = I([X,Z];Y) - I(Z;Y)
        - Otherwise (all continuous): default chain-rule approximation
        """
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        self._validate_inputs(x, y, z)

        k = int(self.n_neighbors)
        if k < 1:
            return 0.0

        y_is_discrete = y.shape[1] == 1 and self._is_discrete_1d(y[:, 0])
        z_is_discrete = z.shape[1] == 1 and self._is_discrete_1d(z[:, 0])

        # If Z is discrete, compute a weighted average of within-class MI.
        if z_is_discrete:
            z_labels = z.ravel()
            n_total = int(z_labels.size)
            if n_total <= 1:
                return 0.0

            cmi = 0.0
            for val in np.unique(z_labels):
                mask = z_labels == val
                n = int(np.sum(mask))
                # ``kneighbors(X=None)`` excludes each fitted sample itself, so
                # a subgroup only needs one other sample and k <= n - 1.
                if n <= 1:
                    continue
                k_sub = min(k, n - 1)
                if k_sub < 1:
                    continue
                try:
                    mi_sub = self._mutual_information_with_k(x[mask], y[mask], k_sub)
                except ValueError as exc:
                    # Be tolerant to rare degenerate subgroups where sklearn rejects
                    # neighbor queries (e.g., effectively tiny fit sets).
                    if "n_neighbors" in str(exc) and "n_samples_fit" in str(exc):
                        if self.config.verbose:
                            self.logger.debug(
                                "Skipping tiny/degenerate CMI subgroup for z=%s (n=%d): %s",
                                str(val),
                                n,
                                str(exc),
                            )
                        continue
                    raise
                cmi += (n / n_total) * mi_sub
            return float(max(0.0, cmi))

        # If Y is discrete and Z is continuous, use chain rule with the mixed MI estimator.
        if y_is_discrete:
            y_labels = self._to_int_labels(y.ravel())
            xz = np.hstack([x, z])
            mi_xz_y = compute_mi_cd(xz, y_labels, k)
            mi_z_y = compute_mi_cd(z, y_labels, k)
            return float(max(0.0, mi_xz_y - mi_z_y))

        # Default (continuous) chain rule: I(X;Y|Z) = I(X;[Y,Z]) - I(X;Z)
        return float(compute_cmi_ccc(x, y, z, k))


class BinnedEstimator(BaseInformationEstimator):
    """Binning / histogram plug-in estimator (discrete MI on discretized data).

    This is most appropriate for low-dimensional settings. For high-dimensional
    vectors, binning quickly becomes sparse and strongly biased; treat it as a
    diagnostic/baseline rather than a primary estimator.
    """

    def __init__(self, config: EstimatorConfig):
        super().__init__(config)
        self.n_bins = int(config.n_bins)
        mp = dict(config.method_params or {})
        self.binning_strategy = str(mp.get("binning_strategy", "quantile")).lower()
        self.bias_correction = str(mp.get("bias_correction", "none")).lower()

    def _bin_continuous(self, x: np.ndarray) -> np.ndarray:
        x = self._ensure_2d(np.asarray(x))
        n, d = x.shape
        if self.n_bins < 2:
            return np.zeros((n, d), dtype=int)

        binned = np.zeros((n, d), dtype=int)
        for j in range(d):
            col = x[:, j]
            if np.allclose(col, col[0], atol=1e-12):
                binned[:, j] = 0
                continue

            if self.binning_strategy == "uniform":
                edges = np.linspace(col.min(), col.max(), self.n_bins + 1)[1:-1]
            else:
                # Default: quantile binning
                qs = np.linspace(0.0, 1.0, self.n_bins + 1)[1:-1]
                edges = np.quantile(col, qs)

            edges = np.unique(edges)
            if edges.size == 0:
                binned[:, j] = 0
            else:
                binned[:, j] = np.digitize(col, edges, right=False)

        return binned

    def _to_states(self, x: np.ndarray) -> np.ndarray:
        x = self._ensure_2d(np.asarray(x))
        if x.shape[1] == 1 and KraskovEstimator._is_discrete_1d(x[:, 0]):
            return KraskovEstimator._to_int_labels(x[:, 0])

        binned = self._bin_continuous(x)
        _, inv = np.unique(binned, axis=0, return_inverse=True)
        return inv.astype(int)

    @staticmethod
    def _miller_madow_mi_correction(
        n_samples: int, x_states: np.ndarray, y_states: np.ndarray
    ) -> float:
        n = float(max(1, int(n_samples)))
        kx = int(np.unique(x_states).size)
        ky = int(np.unique(y_states).size)
        _, inv_xy = np.unique(
            np.column_stack([x_states, y_states]), axis=0, return_inverse=True
        )
        kxy = int(np.unique(inv_xy).size)
        return float((kx + ky - kxy - 1) / (2.0 * n))

    @staticmethod
    def _miller_madow_cmi_correction(
        n_samples: int,
        x_states: np.ndarray,
        y_states: np.ndarray,
        z_states: np.ndarray,
    ) -> float:
        """Return the entropy-expansion correction for ``I(X;Y|Z)``."""
        n = float(max(1, int(n_samples)))

        def state_count(*columns: np.ndarray) -> int:
            states = np.column_stack(columns)
            return int(np.unique(states, axis=0).shape[0])

        kxz = state_count(x_states, z_states)
        kyz = state_count(y_states, z_states)
        kz = int(np.unique(z_states).size)
        kxyz = state_count(x_states, y_states, z_states)
        return float((kxz + kyz - kz - kxyz) / (2.0 * n))

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        self._validate_inputs(x, y)

        xs = self._to_states(x)
        ys = self._to_states(y)
        mi = float(mutual_info_score(xs, ys))

        if self.bias_correction in {"miller_madow", "mm"}:
            mi += self._miller_madow_mi_correction(xs.size, xs, ys)

        return float(max(0.0, mi))

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        self._validate_inputs(x, y, z)

        xs = self._to_states(x)
        ys = self._to_states(y)
        zs = self._to_states(z)

        yz = np.column_stack([ys, zs])
        _, yz_states = np.unique(yz, axis=0, return_inverse=True)

        mi_x_yz = float(mutual_info_score(xs, yz_states))
        mi_x_z = float(mutual_info_score(xs, zs))
        cmi = mi_x_yz - mi_x_z
        if self.bias_correction in {"miller_madow", "mm"}:
            cmi += self._miller_madow_cmi_correction(xs.size, xs, ys, zs)
        return float(max(0.0, cmi))


class DecoderEstimator(BaseInformationEstimator):
    """Decoder-based MI estimator for discrete targets.

    For I(X;C) where C is discrete (class labels), estimate:
      I(X;C) = H(C) - H(C|X)
    where H(C|X) is approximated by cross-validated negative log-likelihood of
    a probabilistic classifier (default: multinomial logistic regression).

    Continuous-continuous MI/CMI is only supported when explicitly enabled via
    `continuous_strategy` ("binned" or "gaussian") and the total dimension is
    <= `continuous_max_dim`; otherwise this returns NaN (with a warning).
    """

    def __init__(self, config: EstimatorConfig):
        super().__init__(config)
        self.n_bins = int(config.n_bins)
        mp = dict(config.method_params or {})
        self.cv_folds = int(mp.get("cv_folds", 5))
        self.C = float(mp.get("C", 1.0))
        self.standardize = bool(mp.get("standardize", True))
        self.seed = int(mp.get("seed", 0))
        self.continuous_strategy = str(mp.get("continuous_strategy", "none")).lower()
        self.continuous_max_dim = int(mp.get("continuous_max_dim", 2))
        self.gaussian_ridge = float(mp.get("gaussian_ridge", 1e-6))
        self.binning_strategy = str(mp.get("binning_strategy", "quantile")).lower()
        self.bias_correction = str(mp.get("bias_correction", "none")).lower()

    def _unsupported(self, msg: str) -> float:
        self.logger.warning(msg)
        return float("nan")

    def _binned_helper(self) -> BinnedEstimator:
        cfg = EstimatorConfig(
            method=EstimationMethod.BINNED,
            n_bins=self.n_bins,
            method_params={
                "binning_strategy": self.binning_strategy,
                "bias_correction": self.bias_correction,
            },
        )
        return BinnedEstimator(cfg)

    def _cov(self, x: np.ndarray) -> np.ndarray:
        x = self._ensure_2d(np.asarray(x))
        cov = np.cov(x, rowvar=False)
        cov = np.atleast_2d(cov)
        cov = cov + float(self.gaussian_ridge) * np.eye(cov.shape[0])
        return cov

    def _gaussian_mi(self, x: np.ndarray, y: np.ndarray) -> float:
        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        xy = np.hstack([x, y])

        sx, ldx = np.linalg.slogdet(self._cov(x))
        sy, ldy = np.linalg.slogdet(self._cov(y))
        sxy, ldxy = np.linalg.slogdet(self._cov(xy))
        if sx <= 0 or sy <= 0 or sxy <= 0:
            return self._unsupported(
                "decoder(gaussian): non-positive definite covariance"
            )
        mi = 0.5 * (ldx + ldy - ldxy)
        return float(max(0.0, mi))

    def _gaussian_cmi(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        z = self._ensure_2d(np.asarray(z))
        xz = np.hstack([x, z])
        yz = np.hstack([y, z])
        xyz = np.hstack([x, y, z])

        sxz, ldxz = np.linalg.slogdet(self._cov(xz))
        syz, ldyz = np.linalg.slogdet(self._cov(yz))
        sz, ldz = np.linalg.slogdet(self._cov(z))
        sxyz, ldxyz = np.linalg.slogdet(self._cov(xyz))
        if sxz <= 0 or syz <= 0 or sz <= 0 or sxyz <= 0:
            return self._unsupported(
                "decoder(gaussian): non-positive definite covariance for CMI"
            )
        cmi = 0.5 * (ldxz + ldyz - ldz - ldxyz)
        return float(max(0.0, cmi))

    def _continuous_dim_ok(self, total_dim: int) -> bool:
        return int(total_dim) <= int(self.continuous_max_dim)

    def _mi_cont_cont(self, x: np.ndarray, y: np.ndarray) -> float:
        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        total_dim = int(x.shape[1] + y.shape[1])

        if not self._continuous_dim_ok(total_dim):
            return self._unsupported(
                f"decoder: continuous MI unsupported for dim={total_dim} (max {self.continuous_max_dim}). "
                f"Use method='kraskov' for high-dim, or set decoder_continuous_strategy for low-dim."
            )

        if self.continuous_strategy in {"binned", "bin", "hist", "histogram"}:
            return float(self._binned_helper().mutual_information(x, y))
        if self.continuous_strategy in {"gaussian", "bayes"}:
            return float(self._gaussian_mi(x, y))
        return self._unsupported(
            "decoder: continuous MI disabled (decoder_continuous_strategy='none')."
        )

    @staticmethod
    def _discrete_entropy_nats(labels: np.ndarray) -> float:
        labels = np.asarray(labels).ravel().astype(int)
        if labels.size == 0:
            return 0.0
        counts = np.bincount(labels)
        counts = counts[counts > 0]
        if counts.size == 0:
            return 0.0
        p = counts.astype(float) / float(counts.sum())
        return float(-np.sum(p * np.log(p + 1e-12)))

    def _mi_cont_disc_decoder(self, x: np.ndarray, labels: np.ndarray) -> float:
        x = self._ensure_2d(np.asarray(x))
        y = KraskovEstimator._to_int_labels(np.asarray(labels).ravel())

        n_classes = int(np.unique(y).size)
        if n_classes <= 1:
            return 0.0

        H = self._discrete_entropy_nats(y)

        # Choose a feasible number of folds based on minimum class count.
        counts = np.bincount(y)
        min_count = int(counts[counts > 0].min()) if np.any(counts > 0) else 0
        n_splits = min(max(2, int(self.cv_folds)), min_count) if min_count >= 2 else 0
        if n_splits < 2:
            return 0.0

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        base_clf = LogisticRegression(
            C=self.C,
            penalty="l2",
            solver="lbfgs",
            max_iter=1000,
        )
        clf = (
            make_pipeline(StandardScaler(), base_clf) if self.standardize else base_clf
        )

        classes = np.arange(n_classes)
        total_loss = 0.0
        total_n = 0
        for train_idx, test_idx in cv.split(x, y):
            x_tr, x_te = x[train_idx], x[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            try:
                clf.fit(x_tr, y_tr)
                proba = clf.predict_proba(x_te)
            except Exception:
                return 0.0

            total_loss += float(log_loss(y_te, proba, labels=classes, normalize=False))
            total_n += int(y_te.size)

        if total_n <= 0:
            return 0.0

        H_cond = total_loss / float(total_n)
        return float(max(0.0, H - H_cond))

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        self._validate_inputs(x, y)

        x_is_discrete = x.shape[1] == 1 and KraskovEstimator._is_discrete_1d(x[:, 0])
        y_is_discrete = y.shape[1] == 1 and KraskovEstimator._is_discrete_1d(y[:, 0])

        if x_is_discrete and y_is_discrete:
            return float(
                mutual_info_score(
                    KraskovEstimator._to_int_labels(x[:, 0]),
                    KraskovEstimator._to_int_labels(y[:, 0]),
                )
            )

        if y_is_discrete and not x_is_discrete:
            return self._mi_cont_disc_decoder(x, y[:, 0])

        if x_is_discrete and not y_is_discrete:
            return self._mi_cont_disc_decoder(y, x[:, 0])

        # Continuous-continuous: only if explicitly enabled for low-dimensional cases.
        return self._mi_cont_cont(x, y)

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        x = self._ensure_2d(x)
        y = self._ensure_2d(y)
        z = self._ensure_2d(z)
        self._validate_inputs(x, y, z)

        y_is_discrete = y.shape[1] == 1 and KraskovEstimator._is_discrete_1d(y[:, 0])
        z_is_discrete = z.shape[1] == 1 and KraskovEstimator._is_discrete_1d(z[:, 0])

        if z_is_discrete:
            z_labels = z[:, 0]
            n_total = int(z_labels.size)
            if n_total <= 1:
                return 0.0
            cmi = 0.0
            for val in np.unique(z_labels):
                mask = z_labels == val
                n = int(np.sum(mask))
                if n <= 1:
                    continue
                cmi += (n / n_total) * float(self.mutual_information(x[mask], y[mask]))
            return float(max(0.0, cmi))

        if y_is_discrete:
            xz = np.hstack([x, z])
            mi_xz_y = self.mutual_information(xz, y)
            mi_z_y = self.mutual_information(z, y)
            return float(max(0.0, float(mi_xz_y) - float(mi_z_y)))

        # All-continuous: only if explicitly enabled for low-dimensional cases.
        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        z = self._ensure_2d(np.asarray(z))
        total_dim = int(x.shape[1] + y.shape[1] + z.shape[1])
        if not self._continuous_dim_ok(total_dim):
            return self._unsupported(
                f"decoder: continuous CMI unsupported for dim={total_dim} (max {self.continuous_max_dim})."
            )
        if self.continuous_strategy in {"binned", "bin", "hist", "histogram"}:
            return float(self._binned_helper().conditional_mutual_information(x, y, z))
        if self.continuous_strategy in {"gaussian", "bayes"}:
            return float(self._gaussian_cmi(x, y, z))
        return self._unsupported(
            "decoder: continuous CMI disabled (decoder_continuous_strategy='none')."
        )


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
            k = min(max(1, int(self.config.n_neighbors)), max(1, x.shape[0] - 1))
            if x.shape[1] > 1:
                return float(compute_mi_cd(x, y_labels, k))
            mi_scores = mutual_info_classif(x, y_labels, n_neighbors=k)
            return float(mi_scores[0])

        elif not x_discrete and not y_discrete:
            # Both continuous (regression)
            if x.shape[1] == 1 and y.shape[1] == 1:
                # Both univariate - use histogram-based estimation
                return self._histogram_mi(x.ravel(), y.ravel())
            else:
                # Multivariate - estimate joint I(X;Y), not mean feature-wise MI.
                k = min(max(1, int(self.config.n_neighbors)), max(1, x.shape[0] - 1))
                return float(compute_mi_cc(x, y, k))

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
        return self._chain_rule_cmi(x, y, z)

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
            _, inverse = np.unique(x, axis=0, return_inverse=True)
            return inverse.astype(int)
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


def copula_normal_scores(x: np.ndarray) -> np.ndarray:
    """Standard normal scores of each column's ranks (ties take average ranks)."""
    x = np.asarray(x, dtype=float)
    return special.ndtri(stats.rankdata(x, axis=0) / (x.shape[0] + 1))


def _half_log_det(samples: np.ndarray) -> float:
    """Bias-corrected half log-determinant of the covariance of ``samples``.

    This is the Gaussian entropy without its ``0.5 * d * log(2 pi e)`` constant,
    which cancels in every information difference. The correction removes the
    expected bias of the sample log-determinant (Misra et al., 2005).
    """
    n, d = samples.shape
    if d == 0:
        return 0.0
    if n <= d:
        raise ValueError(
            f"{n} samples cannot estimate a {d}-dimensional covariance; the "
            "Gaussian copula needs more samples than dimensions (per class when "
            "conditioning on labels)"
        )
    centered = samples - samples.mean(axis=0)
    covariance = centered.T @ centered / (n - 1)
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "the covariance of the copula normal scores is singular: one "
            "variable is a function of the others"
        ) from error
    half = float(np.sum(np.log(np.diagonal(cholesky))))
    digamma = float(np.sum(special.digamma((n - np.arange(1, d + 1)) / 2.0))) / 2.0
    return half - d * (np.log(2.0) - np.log(n - 1.0)) / 2.0 - digamma


def _varying_columns(x: np.ndarray) -> np.ndarray:
    """Drop constant columns, which carry no information."""
    return x[:, np.ptp(x, axis=0) > 0] if x.shape[1] else x


def gaussian_copula_mi(x: np.ndarray, y: np.ndarray) -> float:
    """Gaussian-copula mutual information of two continuous variables (nats)."""
    zx = copula_normal_scores(_varying_columns(x))
    zy = copula_normal_scores(_varying_columns(y))
    if zx.shape[1] == 0 or zy.shape[1] == 0:
        return 0.0
    return _half_log_det(zx) + _half_log_det(zy) - _half_log_det(np.hstack([zx, zy]))


def gaussian_copula_mi_discrete(x: np.ndarray, labels: np.ndarray) -> float:
    """Information of continuous ``x`` about discrete ``labels`` (nats).

    ``H(X) - sum_c p(c) H(X | c)`` with Gaussian entropies of the copula normal
    scores, so the labels' numeric order plays no role.
    """
    zx = copula_normal_scores(_varying_columns(x))
    if zx.shape[1] == 0:
        return 0.0
    labels = np.asarray(labels).ravel()
    conditional = sum(
        np.mean(labels == value) * _half_log_det(zx[labels == value])
        for value in np.unique(labels)
    )
    return _half_log_det(zx) - float(conditional)


def gaussian_copula_cmi(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Gaussian-copula ``I(X;Y|Z)`` of continuous variables (nats)."""
    zx = copula_normal_scores(_varying_columns(x))
    zy = copula_normal_scores(_varying_columns(y))
    zz = copula_normal_scores(_varying_columns(z))
    if zx.shape[1] == 0 or zy.shape[1] == 0:
        return 0.0
    return (
        _half_log_det(np.hstack([zx, zz]))
        + _half_log_det(np.hstack([zy, zz]))
        - _half_log_det(np.hstack([zx, zy, zz]))
        - _half_log_det(zz)
    )


class CopulaEstimator(BaseInformationEstimator):
    """Copula mutual information: Gaussian copula, or a kernel copula density.

    ``copula_type="gaussian"`` is the Gaussian-copula estimator of Ince et al.
    (Hum. Brain Mapp. 38:1541, 2017). Each variable's columns become standard
    normal scores of their ranks, and the information is that of a Gaussian
    with the scores' covariance, with the analytic bias correction of the
    sample log-determinants. It is exact for Gaussian dependence and otherwise
    a lower bound. A discrete variable (one integer-valued column, such as
    class labels) enters through class-conditional entropies, never through
    its numeric order; two discrete variables use the plug-in estimate, and a
    discrete condition averages the within-class information.

    ``copula_type="kernel"`` estimates the density of the rank-uniform copula
    of two continuous scalars with Gaussian kernels.

    Values are in nats and are not clipped at zero.
    """

    def __init__(self, config: EstimatorConfig):
        super().__init__(config)
        self.copula_type = config.copula_type

    @staticmethod
    def _discrete(values: np.ndarray) -> bool:
        return values.shape[1] == 1 and KraskovEstimator._is_discrete_1d(values[:, 0])

    def mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Copula mutual information of ``x`` and ``y`` (nats)."""
        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        self._validate_inputs(x, y)
        x_discrete, y_discrete = self._discrete(x), self._discrete(y)
        if x_discrete and y_discrete:
            return float(mutual_info_score(x.ravel(), y.ravel()))
        if self.copula_type == "kernel":
            return self._kernel_copula_mi(x, y, x_discrete or y_discrete)
        if y_discrete:
            return gaussian_copula_mi_discrete(x, y.ravel())
        if x_discrete:
            return gaussian_copula_mi_discrete(y, x.ravel())
        return gaussian_copula_mi(x, y)

    def conditional_mutual_information(
        self, x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> float:
        """Copula ``I(X;Y|Z)`` (nats).

        A discrete ``Z`` averages the within-class information; a discrete ``Y``
        (or ``X``) with continuous ``Z`` uses ``I([X,Z];Y) - I(Z;Y)``.
        """
        x = self._ensure_2d(np.asarray(x))
        y = self._ensure_2d(np.asarray(y))
        z = self._ensure_2d(np.asarray(z))
        self._validate_inputs(x, y, z)
        if self._discrete(z):
            labels = z.ravel()
            return float(
                sum(
                    np.mean(labels == value)
                    * self.mutual_information(x[labels == value], y[labels == value])
                    for value in np.unique(labels)
                )
            )
        if self.copula_type == "kernel":
            raise ValueError(
                "the kernel copula estimates the information of two continuous "
                "scalars only; use copula type 'gaussian' for conditional terms"
            )
        x_discrete, y_discrete = self._discrete(x), self._discrete(y)
        if x_discrete and y_discrete:
            raise ValueError(
                "the copula estimator has no conditional information of two "
                "discrete variables given a continuous one"
            )
        if y_discrete:
            return gaussian_copula_mi_discrete(
                np.hstack([x, z]), y.ravel()
            ) - gaussian_copula_mi_discrete(z, y.ravel())
        if x_discrete:
            return gaussian_copula_mi_discrete(
                np.hstack([y, z]), x.ravel()
            ) - gaussian_copula_mi_discrete(z, x.ravel())
        return gaussian_copula_cmi(x, y, z)

    def entropy(self, x: np.ndarray) -> float:
        """Plug-in entropy of discrete ``x`` (nats).

        The copula transform discards the marginal distributions that carry a
        continuous variable's entropy, so continuous inputs raise.
        """
        x = self._ensure_2d(np.asarray(x))
        if not all(
            KraskovEstimator._is_discrete_1d(x[:, i]) for i in range(x.shape[1])
        ):
            raise ValueError(
                "the copula estimator defines no entropy for continuous variables"
            )
        _, counts = np.unique(x, axis=0, return_counts=True)
        probabilities = counts / counts.sum()
        return float(-np.sum(probabilities * np.log(probabilities)))

    def _kernel_copula_mi(self, x: np.ndarray, y: np.ndarray, discrete: bool) -> float:
        """Kernel density estimate of the information of two continuous scalars."""
        from sklearn.neighbors import KernelDensity

        if discrete or x.shape[1] != 1 or y.shape[1] != 1:
            raise ValueError(
                "the kernel copula estimates the information of two continuous "
                "scalars only; use copula type 'gaussian' for labels or vectors"
            )
        n = x.shape[0]
        u = stats.rankdata(x[:, 0]) / (n + 1)
        v = stats.rankdata(y[:, 0]) / (n + 1)
        if np.ptp(u) == 0 or np.ptp(v) == 0:
            return 0.0
        # Scott's rule per margin; the joint kernel takes the smaller width.
        bandwidth_u = n ** (-1 / 5) * np.std(u)
        bandwidth_v = n ** (-1 / 5) * np.std(v)
        bandwidth_joint = max(min(bandwidth_u, bandwidth_v), 0.01)
        kde_u = KernelDensity(bandwidth=bandwidth_u, kernel="gaussian")
        kde_v = KernelDensity(bandwidth=bandwidth_v, kernel="gaussian")
        kde_joint = KernelDensity(bandwidth=bandwidth_joint, kernel="gaussian")
        uv = np.column_stack([u, v])
        log_pu = kde_u.fit(u[:, None]).score_samples(u[:, None])
        log_pv = kde_v.fit(v[:, None]).score_samples(v[:, None])
        log_puv = kde_joint.fit(uv).score_samples(uv)
        return float(np.mean(log_puv - log_pu - log_pv))


class InformationEstimatorFactory:
    """Factory for creating information estimators based on configuration."""

    _estimators: ClassVar[dict[EstimationMethod, type]] = {
        EstimationMethod.KRASKOV: KraskovEstimator,
        EstimationMethod.DECODER: DecoderEstimator,
        EstimationMethod.BINNED: BinnedEstimator,
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
