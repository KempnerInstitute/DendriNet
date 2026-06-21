"""Analyzer configuration setup helpers for information analysis."""

from __future__ import annotations

from typing import Any

from dendritic_modeling.analysis.core.information_config import (
    _build_information_estimator_core_config,
    _build_information_label_shuffle_null_config,
)


class InformationConfigSetupMixin:
    """Assigns parsed information-analysis configuration groups."""

    def _configure_estimator_core(
        self,
        *,
        params: Any,
        estimator_cfg: Any,
        compute_cfg: Any,
    ) -> None:
        """Assign estimator method and core sampling parameters."""
        estimator_core_config = _build_information_estimator_core_config(
            params=params,
            estimator_cfg=estimator_cfg,
            compute_cfg=compute_cfg,
        )
        self.method = estimator_core_config.method
        self.max_samples = estimator_core_config.max_samples
        self.n_neighbors = estimator_core_config.n_neighbors
        self.n_bins = estimator_core_config.n_bins
        self.copula_type = estimator_core_config.copula_type

    def _configure_label_shuffle_null(
        self,
        *,
        params: Any,
        baselines_cfg: Any,
    ) -> None:
        """Assign label-shuffle null-baseline parameters."""
        label_shuffle_null = _build_information_label_shuffle_null_config(
            params=params,
            baselines_cfg=baselines_cfg,
        )
        self.mi_null_shuffles = label_shuffle_null.mi_null_shuffles
        self.mi_null_seed = label_shuffle_null.mi_null_seed
        self.mi_null_components = label_shuffle_null.mi_null_components


__all__ = ["InformationConfigSetupMixin"]
