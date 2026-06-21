"""Partial-information-decomposition metrics for information analysis."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from dendritic_modeling.analysis.core.information_fisher_mixin import (
    InformationFisherMetricsMixin,
)
from dendritic_modeling.utils.information.pid import (
    check_pid_available,
    pid_synergy_redundancy,
)


class InformationPidMetricsMixin(InformationFisherMetricsMixin):
    """Computes PID metrics and PID-specific preprocessing."""

    def _check_pid_available(self) -> None:
        """Set whether the optional PID backend is available."""
        self.pid_available = False
        if self.compute_pid:
            try:
                check_pid_available()
                self.pid_available = True
                self.logger.info("PID computation available")
            except ImportError:
                self.logger.warning(
                    "PID package not available. PID metrics will be skipped."
                )

    def _compute_pid(
        self, E: np.ndarray, I_var: np.ndarray, Vout: np.ndarray, C: np.ndarray
    ) -> dict[str, Any]:
        """Compute Partial Information Decomposition."""
        self.logger.info("Computing Partial Information Decomposition (PID)...")
        pid_results = {}

        # Prepare data for PID (may need binarization)
        self.logger.info(f"  Data preprocessing: binarize={self.pid_binarize}")
        if self.pid_binarize:
            self.logger.info(
                f"  Binarizing data using method: {self.pid_binarize_method}"
            )
            start_time = time.time()
            E_pid = self._binarize(E)
            I_pid = self._binarize(I_var)
            Vout_pid = self._binarize(Vout)
            C_pid = self._binarize(C.reshape(-1, 1))
            self.logger.info(
                f"  Data binarization completed (took {time.time() - start_time:.1f}s)"
            )
        else:
            E_pid = E
            I_pid = I_var
            Vout_pid = Vout
            C_pid = C.reshape(-1, 1)

        def _pid_helper(
            source1_array: np.ndarray,
            source1_name: str,
            source2_array: np.ndarray,
            source2_name: str,
            target_array: np.ndarray,
            target_name: str,
        ) -> None:
            """Helper function to compute PID."""
            pid_key = f"PID({source1_name},{source2_name}->{target_name})"
            self.logger.info(f"  Computing {pid_key}...")
            start_time = time.time()
            try:
                synergy, redundancy = pid_synergy_redundancy(
                    [source1_array, source2_array], target_array
                )

                # Compute unique information
                unique_1 = (
                    self.estimators[self.method].mutual_information(
                        source1_array, target_array
                    )
                    - redundancy
                )
                unique_2 = (
                    self.estimators[self.method].mutual_information(
                        source2_array, target_array
                    )
                    - redundancy
                )

                pid_results[pid_key] = {
                    "synergy": synergy,
                    "redundancy": redundancy,
                    f"unique_{source1_name}": unique_1,
                    f"unique_{source2_name}": unique_2,
                }
                self.logger.info(
                    f"  {pid_key}: synergy={synergy:.4f}, redundancy={redundancy:.4f}, unique_{source1_name}={unique_1:.4f}, unique_{source2_name}={unique_2:.4f} (took {time.time() - start_time:.1f}s)"
                )
            except Exception as e:
                self.logger.warning(
                    f"  {pid_key} computation failed: {e} (took {time.time() - start_time:.1f}s)"
                )

        # PID: (E, I) -> C
        _pid_helper(E_pid, "E", I_pid, "I", C_pid, "C")
        # PID: (E, Vout) -> C
        _pid_helper(E_pid, "E", Vout_pid, "Vout", C_pid, "C")
        # PID: (I, Vout) -> C
        _pid_helper(I_pid, "I", Vout_pid, "Vout", C_pid, "C")
        # PID: (E, I) -> Vout
        _pid_helper(E_pid, "E", I_pid, "I", Vout_pid, "Vout")

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
