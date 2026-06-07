"""
RecurrentTrainer: training strategy for recurrent dendritic networks.

Overrides both _epoch_train() and _epoch_valid() from the standard Trainer
to handle sequence data with proper temporal loss computation.

Supports:
- AMP (automatic mixed precision) via ``use_amp=True``
- Cosine-annealing LR schedule with linear warmup via
  ``lr_schedule="cosine"`` and ``lr_warmup_epochs=N``
"""

import logging
import math

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.distributed_samplers import (
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


class RecurrentTrainer(Trainer):
    """Training strategy for recurrent dendritic networks.

    Handles sequence data ([B, T, D] inputs) with proper temporal loss.
    Uses _compute_sequence_loss for both training and validation.

    Args:
        task: "classification" or "regression" -- determines loss function.
        grad_clip: Maximum gradient norm (0 to disable).
        use_amp: Use automatic mixed precision (default False).
        lr_schedule: LR schedule type -- "none" or "cosine" (default "none").
        lr_warmup_epochs: Number of linear warmup epochs (default 0).
        **kwargs: All standard Trainer arguments.
    """

    def __init__(
        self,
        task: str = "classification",
        grad_clip: float = 1.0,
        use_amp: bool = False,
        lr_schedule: str = "none",
        lr_warmup_epochs: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.grad_clip = grad_clip
        self.task = task

        # AMP setup
        self.recurrent_amp = use_amp and torch.cuda.is_available()
        self.recurrent_scaler = GradScaler() if self.recurrent_amp else None

        # LR schedule setup (created lazily in train() once we know n_epochs)
        self.lr_schedule = lr_schedule.lower()
        self.lr_warmup_epochs = lr_warmup_epochs
        self.scheduler = None

        # Task-agnostic loss (not CE-only)
        if task == "classification":
            self.task_loss = nn.CrossEntropyLoss()
        elif task == "regression":
            self.task_loss = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task for RecurrentTrainer: {task}")

        amp_str = "ON" if self.recurrent_amp else "OFF"
        sched_str = self.lr_schedule if self.lr_schedule != "none" else "constant"
        logger.info(
            f"RecurrentTrainer initialized: task={task}, grad_clip={grad_clip}, "
            f"amp={amp_str}, lr_schedule={sched_str}, warmup={lr_warmup_epochs}"
        )

    def _get_torch_optimizer(self):
        """Unwrap CustomWeightDecayOptimizer to get the real torch.optim.Optimizer."""
        opt = self.optimizer
        # CustomWeightDecayOptimizer stores the real optimizer as .optimizer
        if hasattr(opt, "optimizer") and isinstance(
            opt.optimizer, torch.optim.Optimizer
        ):
            return opt.optimizer
        return opt

    def _build_scheduler(self, n_epochs: int) -> None:
        """Create LR scheduler after we know the total epoch count."""
        if self.lr_schedule == "none":
            self.scheduler = None
            return

        torch_opt = self._get_torch_optimizer()

        if self.lr_schedule == "cosine":
            warmup = self.lr_warmup_epochs
            base_lrs = [pg["lr"] for pg in torch_opt.param_groups]

            def lr_lambda(epoch: int) -> float:
                if epoch < warmup:
                    return (epoch + 1) / max(warmup, 1)
                progress = (epoch - warmup) / max(n_epochs - warmup, 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(torch_opt, lr_lambda)
            logger.info(
                f"Cosine LR schedule: warmup={warmup}, total={n_epochs}, "
                f"base_lr={base_lrs}"
            )
        else:
            raise ValueError(f"Unknown lr_schedule: {self.lr_schedule}")

    def _compute_sequence_loss(
        self,
        model: BaseModel,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared sequence-aware loss for both train and valid.

        output_mode is the single source of truth for temporal reduction:
        - "last"/"mean": model returns [B, C] -> loss(output, target) directly
        - "all": model returns [B, T, *] -> flatten both output and target,
          then loss. Target MUST have temporal dim when output_mode="all".
        """
        output = model(x_batch, seq_lengths=seq_lengths)

        if output.dim() <= 2:
            # output_mode="last" or "mean" -- already reduced, standard loss
            return self.task_loss(output, y_batch)

        # output_mode="all" -- output is [B, T, *], target MUST have temporal dim
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

        for batch in train_loader:
            if len(batch) == 3:
                x_batch, y_batch, seq_lengths = batch
                seq_lengths = seq_lengths.to(self.device)
            else:
                x_batch, y_batch = batch
                seq_lengths = None
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            if self.recurrent_amp:
                amp_opt = self.optimizer
                with autocast():
                    loss = self._compute_sequence_loss(
                        model, x_batch, y_batch, seq_lengths
                    )
                self.recurrent_scaler.scale(loss).backward()
                if self.grad_clip > 0:
                    self.recurrent_scaler.unscale_(amp_opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                # Step through the configured optimizer object directly so AMP
                # skip semantics apply to wrapper logic (e.g., custom decay/rewiring).
                self.recurrent_scaler.step(amp_opt)
                self.recurrent_scaler.update()
            else:
                loss = self._compute_sequence_loss(model, x_batch, y_batch, seq_lengths)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        # Step LR scheduler once per epoch
        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, avg_loss, 0.0  # (total, base, reg=0)

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict:
        """Override to build LR scheduler before training starts."""
        self._build_scheduler(self.epochs)
        return super().train(model, train_data, valid_data)

    def _epoch_valid(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> tuple[float, int]:
        """Override parent -- return loss sum and batch count like Trainer."""
        model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in valid_loader:
                if len(batch) == 3:
                    x_batch, y_batch, seq_lengths = batch
                    seq_lengths = seq_lengths.to(self.device)
                else:
                    x_batch, y_batch = batch
                    seq_lengths = None
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                loss = self._compute_sequence_loss(model, x_batch, y_batch, seq_lengths)
                total_loss += loss.item()
                n_batches += 1

        return total_loss, n_batches

    def _epoch_valid_distributed(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> float:
        """Distributed validation using recurrent sequence-aware loss."""
        model.eval()
        weighted_loss_sum = 0.0
        total_examples = 0
        sampler = valid_loader.sampler
        if isinstance(sampler, PaddedDistributedEvalSampler):
            real_count = len(sampler.real_indices)
        else:
            real_count = None

        seen = 0
        with torch.no_grad():
            for batch in valid_loader:
                if len(batch) == 3:
                    x_batch, y_batch, seq_lengths = batch
                    seq_lengths = seq_lengths.to(self.device)
                else:
                    x_batch, y_batch = batch
                    seq_lengths = None
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                batch_size = y_batch.size(0)
                if real_count is not None:
                    real_in_batch = min(batch_size, real_count - seen)
                    if real_in_batch <= 0:
                        seen += batch_size
                        continue
                    if real_in_batch < batch_size:
                        x_batch = x_batch[:real_in_batch]
                        y_batch = y_batch[:real_in_batch]
                        if seq_lengths is not None:
                            seq_lengths = seq_lengths[:real_in_batch]
                        batch_size = real_in_batch
                    seen += batch_size

                loss = self._compute_sequence_loss(model, x_batch, y_batch, seq_lengths)
                weighted_loss_sum += loss.item() * batch_size
                total_examples += batch_size

        stats = torch.tensor(
            [weighted_loss_sum, float(total_examples)],
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats[0].item() / max(stats[1].item(), 1.0)
