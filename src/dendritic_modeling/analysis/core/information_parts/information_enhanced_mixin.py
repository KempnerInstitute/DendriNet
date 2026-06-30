"""Enhanced information-analysis result helpers."""

from typing import Any

import numpy as np


class InformationEnhancedAnalysisMixin:
    """Add optional per-layer, per-neuron, and E/I-specific analyses."""

    def _collect_enhanced_analysis_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Collect concatenated E/I/Vout arrays for enhanced analysis modes."""
        all_E_data = []
        all_I_data = []
        all_Vout_data = []

        for layer_data in self.data_dict.values():
            if "excitation" in layer_data:
                E_filtered, I_filtered = self.ensure_inhibitory_data(
                    layer_data["excitation"], layer_data["inhibition"]
                )
                all_E_data.append(E_filtered.detach().cpu().numpy())
                all_I_data.append(I_filtered.detach().cpu().numpy())
                all_Vout_data.append(layer_data["output"].detach().cpu().numpy())

        if not all_E_data:
            return None

        E_all = np.concatenate(all_E_data, axis=1)
        I_all = np.concatenate(all_I_data, axis=1)
        Vout_all = np.concatenate(all_Vout_data, axis=1)
        return E_all, I_all, Vout_all

    def _add_enhanced_analysis_results(
        self, results: dict[str, Any], C: np.ndarray
    ) -> None:
        """Add optional per-layer, per-neuron, and EI-specific analysis results."""
        if not self._enhanced_analysis_requested():
            return

        self.logger.info("Running enhanced analysis modes...")

        if self.per_layer_analysis:
            per_layer_results = self.compute_per_layer_enhanced_analysis(C)
            results["per_layer_analysis"] = per_layer_results

        if not self._enhanced_array_analysis_requested():
            return

        enhanced_arrays = self._collect_enhanced_analysis_arrays()
        if enhanced_arrays is None:
            return

        E_all, I_all, Vout_all = enhanced_arrays

        if self.per_neuron_analysis:
            per_neuron_results = self.compute_per_neuron_mi(E_all, I_all, Vout_all, C)
            results["per_neuron_analysis"] = per_neuron_results

        if self.per_einet_analysis:
            ei_specific_results = self.compute_ei_specific_mi(E_all, I_all, Vout_all, C)
            results["per_einet_analysis"] = ei_specific_results

    def _enhanced_analysis_requested(self) -> bool:
        return (
            self.per_neuron_analysis
            or self.per_einet_analysis
            or self.per_layer_analysis
        )

    def _enhanced_array_analysis_requested(self) -> bool:
        return self.per_neuron_analysis or self.per_einet_analysis


__all__ = ["InformationEnhancedAnalysisMixin"]
