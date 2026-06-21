"""Dataset materialization and activation collection for information analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dendritic_modeling.analysis.utils.runtime import materialize_dataset
from dendritic_modeling.config.analysis import EvaluationRuntimeConfig
from dendritic_modeling.models import BaseModel


@dataclass(frozen=True)
class _InformationAnalysisData:
    """Materialized inputs and labels used by information analysis."""

    inputs: torch.Tensor
    labels: torch.Tensor
    C: np.ndarray
    entropy_C: float | None


class InformationDataMixin:
    """Collect dataset tensors and recorded layer activations."""

    def _collect_analysis_data(
        self,
        *,
        model: BaseModel,
        test_dataset: torch.utils.data.Dataset,
        device: str,
        runtime: EvaluationRuntimeConfig | None,
    ) -> _InformationAnalysisData:
        """Materialize data and collect branch activations through forward hooks."""
        self.logger.info("Loading test dataset...")
        items = materialize_dataset(
            test_dataset,
            runtime,
            explicit_max_samples=self.max_samples,
            device=device,
        )
        inputs: torch.Tensor = items[0].to(device)
        labels: torch.Tensor = items[1].to(device)
        self.logger.info(
            f"Loaded dataset: {len(inputs)} samples, input shape: {inputs.shape}"
        )

        handles = self.attach_forward_hooks(model)
        try:
            with torch.no_grad():
                _ = model(inputs)
        finally:
            self.remove_forward_hooks(handles)

        C = labels.detach().cpu().numpy()
        self.logger.debug(
            f"Data collection completed. Found {len(self.data_dict)} layers."
        )

        entropy_C: float | None
        if self._is_discrete_class_labels(C):
            entropy_C = self._discrete_entropy_nats(C)
        else:
            entropy_C = None

        return _InformationAnalysisData(
            inputs=inputs,
            labels=labels,
            C=C,
            entropy_C=entropy_C,
        )


__all__ = ["InformationDataMixin", "_InformationAnalysisData"]
