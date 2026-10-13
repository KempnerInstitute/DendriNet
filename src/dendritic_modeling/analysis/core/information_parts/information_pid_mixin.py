"""Partial-information-decomposition metrics for information analysis.

Each decomposition splits ``I(S1,S2;T)`` into redundant, unique and synergistic
parts with the minimum-mutual-information measure
(:mod:`dendritic_modeling.utils.information.pid`), estimating the three mutual
informations with the analysis' configured estimator.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dendritic_modeling.analysis.core.information_parts.information_fisher_mixin import (
    InformationFisherMetricsMixin,
)
from dendritic_modeling.utils.information.pid import partial_information_decomposition

PidArraySet = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
PidMetricSpec = tuple[np.ndarray, str, np.ndarray, str, np.ndarray, str]


class InformationPidMetricsMixin(InformationFisherMetricsMixin):
    """Computes PID metrics and PID-specific preprocessing."""

    def _add_pid_metric(
        self,
        pid_results: dict[str, Any],
        *,
        source1_array: np.ndarray,
        source1_name: str,
        source2_array: np.ndarray,
        source2_name: str,
        target_array: np.ndarray,
        target_name: str,
    ) -> None:
        """Compute one PID metric and append it to ``pid_results`` in place."""
        pid_key = f"PID({source1_name},{source2_name}->{target_name})"
        self.logger.info(f"  Computing {pid_key}...")
        start_time = time.time()
        terms = partial_information_decomposition(
            source1_array,
            source2_array,
            target_array,
            self.estimators[self.method].mutual_information,
        )
        pid_results[pid_key] = {
            "redundancy": terms.redundancy,
            f"unique_{source1_name}": terms.unique_1,
            f"unique_{source2_name}": terms.unique_2,
            "synergy": terms.synergy,
            "joint": terms.joint,
        }
        self.logger.info(
            f"  {pid_key}: redundancy={terms.redundancy:.4f}, "
            f"unique_{source1_name}={terms.unique_1:.4f}, "
            f"unique_{source2_name}={terms.unique_2:.4f}, "
            f"synergy={terms.synergy:.4f} (took {time.time() - start_time:.1f}s)"
        )

    def _pid_arrays(
        self, E: np.ndarray, I_var: np.ndarray, Vout: np.ndarray, C: np.ndarray
    ) -> PidArraySet:
        """Return the PID variables; optional binarization never touches labels.

        The class labels are already discrete, and a threshold split would merge
        classes.
        """
        self.logger.info(f"  Data preprocessing: binarize={self.pid_binarize}")
        C_pid = C.reshape(-1, 1)
        if not self.pid_binarize:
            return E, I_var, Vout, C_pid

        self.logger.info(f"  Binarizing data using method: {self.pid_binarize_method}")
        start_time = time.time()
        E_pid = self._binarize(E)
        I_pid = self._binarize(I_var)
        Vout_pid = self._binarize(Vout)
        self.logger.info(
            f"  Data binarization completed (took {time.time() - start_time:.1f}s)"
        )
        return E_pid, I_pid, Vout_pid, C_pid

    @staticmethod
    def _pid_metric_specs(
        E_pid: np.ndarray,
        I_pid: np.ndarray,
        Vout_pid: np.ndarray,
        C_pid: np.ndarray,
    ) -> tuple[PidMetricSpec, ...]:
        return (
            (E_pid, "E", I_pid, "I", C_pid, "C"),
            (E_pid, "E", Vout_pid, "Vout", C_pid, "C"),
            (I_pid, "I", Vout_pid, "Vout", C_pid, "C"),
            (E_pid, "E", I_pid, "I", Vout_pid, "Vout"),
        )

    def _compute_pid(
        self, E: np.ndarray, I_var: np.ndarray, Vout: np.ndarray, C: np.ndarray
    ) -> dict[str, Any]:
        """Compute Partial Information Decomposition."""
        self.logger.info("Computing Partial Information Decomposition (PID)...")
        pid_results = {}

        E_pid, I_pid, Vout_pid, C_pid = self._pid_arrays(E, I_var, Vout, C)

        for (
            source1_array,
            source1_name,
            source2_array,
            source2_name,
            target_array,
            target_name,
        ) in self._pid_metric_specs(E_pid, I_pid, Vout_pid, C_pid):
            self._add_pid_metric(
                pid_results,
                source1_array=source1_array,
                source1_name=source1_name,
                source2_array=source2_array,
                source2_name=source2_name,
                target_array=target_array,
                target_name=target_name,
            )

        self.logger.info("  PID computation completed.")
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

        binarized: np.ndarray = data > threshold
        return binarized.astype(int)


__all__ = ["InformationPidMetricsMixin"]
