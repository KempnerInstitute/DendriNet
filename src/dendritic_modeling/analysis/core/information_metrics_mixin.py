"""Metric orchestration helpers for information analysis."""

from __future__ import annotations

from typing import Any

import numpy as np

from dendritic_modeling.analysis.core.information_mutual_information_mixin import (
    InformationMutualInformationMixin,
)
from dendritic_modeling.analysis.core.information_null_baseline_mixin import (
    _metric_variable_for_null_baseline,
    _parse_mi_key_for_null_baseline,
)


class InformationMetricsMixin(InformationMutualInformationMixin):
    """Orchestrates requested information metrics for one activation set."""

    def compute_metrics(
        self,
        E: np.ndarray,
        I_var: np.ndarray,
        Vout: np.ndarray,
        C: np.ndarray,
        Vinf: np.ndarray | None = None,
        Vb: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute all requested information metrics.

        Parameters
        ----------
        E : np.ndarray
            Excitatory inputs, shape (n_samples,) or (n_samples, n_features)
        I_var : np.ndarray
            Inhibitory inputs, shape (n_samples,) or (n_samples, n_features)
        Vout : np.ndarray
            Branch outputs, shape (n_samples,) or (n_samples, n_features)
        C : np.ndarray
            Class labels, shape (n_samples,)
        Vinf : np.ndarray, optional
            Soma activation (Vinf), shape (n_samples,) or (n_samples, n_features)
        Vb : np.ndarray, optional
            Branch input (upstream dendritic input), shape (n_samples,) or (n_samples, n_features)

        Returns
        -------
        dict[str, Any]
            Dictionary containing all computed metrics
        """
        # Validate inputs
        self._validate_inputs(E, I_var, Vout, C, Vinf)

        # Ensure all arrays are properly shaped
        E = self._ensure_2d(E)
        I_var = self._ensure_2d(I_var)
        Vout = self._ensure_2d(Vout)
        C = C.ravel()  # Labels are always 1D

        if Vinf is not None:
            Vinf = self._ensure_2d(Vinf)

        if Vb is not None:
            Vb = self._ensure_2d(Vb)

        results = {
            "method": self.method,
            "n_samples": len(E),
        }

        # 1. Basic mutual information with C
        if self.compute_basic_mi:
            self.logger.info("Computing basic MI metrics...")
            results["basic_mi"] = self._compute_basic_mi(E, I_var, Vout, C, Vinf, Vb)

        # 2. Pairwise mutual information
        if self.compute_pairwise_mi:
            self.logger.info("Computing pairwise MI metrics...")
            results["pairwise_mi"] = self._compute_pairwise_mi(E, I_var, Vout, Vinf, Vb)

        # 3. Conditional mutual information
        if self.compute_conditional_mi:
            self.logger.info("Computing conditional MI metrics...")
            results["conditional_mi"] = self._compute_conditional_mi(
                E, I_var, Vout, C, Vinf, Vb
            )

        # 4. Gaussian/Fisher MI proxy
        if self.compute_gaussian_fisher:
            self.logger.info("Computing Gaussian/Fisher proxy metrics...")
            results["gaussian_fisher"] = self._compute_gaussian_fisher(
                E, I_var, Vout, C, Vinf, Vb
            )

        # 4. PID decomposition
        if self.compute_pid and self.pid_available:
            self.logger.info("Computing PID metrics...")
            results["pid"] = self._compute_pid(E, I_var, Vout, C)

        return results


__all__ = [
    "InformationMetricsMixin",
    "_metric_variable_for_null_baseline",
    "_parse_mi_key_for_null_baseline",
]
