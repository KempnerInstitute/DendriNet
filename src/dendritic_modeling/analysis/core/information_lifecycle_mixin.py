"""Model lifecycle helpers for information analysis."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from dendritic_modeling.analysis.utils.runtime import get_model_device
from dendritic_modeling.models import BaseModel


@dataclass(frozen=True)
class _InformationModelLifecycleState:
    """Original model state that should be restored after analysis."""

    original_device: torch.device
    was_training: bool
    moved: bool = False


class InformationLifecycleMixin:
    """Prepare and restore model state around information analysis."""

    def _capture_model_lifecycle_state(
        self, model: BaseModel
    ) -> _InformationModelLifecycleState:
        """Record model placement and train/eval state before analysis."""
        return _InformationModelLifecycleState(
            original_device=get_model_device(model),
            was_training=getattr(model, "training", False),
        )

    def _mark_model_moved(
        self, lifecycle_state: _InformationModelLifecycleState
    ) -> _InformationModelLifecycleState:
        """Return a lifecycle state indicating the model was moved for analysis."""
        return replace(lifecycle_state, moved=True)

    def _prepare_model_for_analysis(self, model: BaseModel, device: str) -> BaseModel:
        """Move a model to the analysis device and switch it to evaluation mode."""
        self.logger.info(f"Preparing data: moving model to {device}")
        model = model.to(device)
        model.eval()
        return model

    def _restore_model_after_analysis(
        self,
        *,
        model: BaseModel,
        lifecycle_state: _InformationModelLifecycleState | None = None,
        original_device: torch.device | None = None,
        was_training: bool | None = None,
        moved: bool | None = None,
    ) -> None:
        """Restore model placement and train/eval state after analysis."""
        if lifecycle_state is not None:
            original_device = lifecycle_state.original_device
            was_training = lifecycle_state.was_training
            moved = lifecycle_state.moved

        if not moved:
            return

        model.to(original_device)
        if was_training:
            model.train()


__all__ = ["InformationLifecycleMixin"]
