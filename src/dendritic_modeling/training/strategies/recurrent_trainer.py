"""
RecurrentTrainer: training strategy for recurrent dendritic networks.

Overrides both _epoch_train() and _epoch_valid() from the standard Trainer
to handle sequence data with proper temporal loss computation.

Phase 2 intentionally bypasses AMP and RegularizationManager paths.
AMP with recurrent unrolling needs careful handling (autocast context per
timestep vs. entire sequence). RegularizationManager's loss functions
(VoltageStabilization, EIBalance) are EI-specific and don't apply to
recurrent cores. Both can be added in Phase 3 after basic training is validated.
"""

import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


class RecurrentTrainer(Trainer):
    """Training strategy for recurrent dendritic networks.

    Handles sequence data ([B, T, D] inputs) with proper temporal loss.
    Uses _compute_sequence_loss for both training and validation.

    Args:
        task: "classification" or "regression" — determines loss function.
        grad_clip: Maximum gradient norm (0 to disable).
        **kwargs: All standard Trainer arguments.
    """

    def __init__(
        self,
        task: str = "classification",
        grad_clip: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.grad_clip = grad_clip
        self.task = task

        # Task-agnostic loss (not CE-only)
        if task == "classification":
            self.task_loss = nn.CrossEntropyLoss()
        elif task == "regression":
            self.task_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task for RecurrentTrainer: {task}")

        logger.info(f"RecurrentTrainer initialized: task={task}, grad_clip={grad_clip}")

    def _compute_sequence_loss(
        self, model: BaseModel, x_batch: torch.Tensor, y_batch: torch.Tensor
    ) -> torch.Tensor:
        """Shared sequence-aware loss for both train and valid.

        output_mode is the single source of truth for temporal reduction:
        - "last"/"mean": model returns [B, C] -> loss(output, target) directly
        - "all": model returns [B, T, *] -> flatten both output and target,
          then loss. Target MUST have temporal dim when output_mode="all".
        """
        output = model(x_batch)

        if output.dim() <= 2:
            # output_mode="last" or "mean" — already reduced, standard loss
            return self.task_loss(output, y_batch)

        # output_mode="all" — output is [B, T, *], target MUST have temporal dim
        B, T = output.shape[:2]

        # Explicit target-shape validation
        if self.task == "classification":
            if y_batch.dim() != 2 or y_batch.shape[0] != B or y_batch.shape[1] != T:
                raise ValueError(
                    f"output_mode='all' with classification: output is "
                    f"[{B},{T},{output.shape[2]}], target must be [{B},{T}] "
                    f"but got {list(y_batch.shape)}. "
                    f"Use output_mode='last' for many-to-one tasks."
                )
        elif self.task == "regression":
            if y_batch.dim() != 3 or y_batch.shape[0] != B or y_batch.shape[1] != T:
                raise ValueError(
                    f"output_mode='all' with regression: output is "
                    f"{list(output.shape)}, target must be [{B},{T},*] "
                    f"but got {list(y_batch.shape)}. "
                    f"Use output_mode='last' for many-to-one tasks."
                )

        # Flatten temporal dimension
        output = output.reshape(B * T, *output.shape[2:])
        y_batch = y_batch.reshape(B * T, *y_batch.shape[2:])
        return self.task_loss(output, y_batch)

    def _epoch_train(
        self, model: BaseModel, train_loader: DataLoader
    ) -> tuple[float, float, float]:
        """Returns (total, base, reg) to match Trainer._run_epoch contract."""
        model.train()
        total_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            loss = self._compute_sequence_loss(model, x_batch, y_batch)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, avg_loss, 0.0  # (total, base, reg=0)

    def _epoch_valid(self, model: BaseModel, valid_loader: DataLoader) -> float:
        """Override parent — uses same sequence-aware loss as training."""
        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self._compute_sequence_loss(model, x_batch, y_batch)
                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)
