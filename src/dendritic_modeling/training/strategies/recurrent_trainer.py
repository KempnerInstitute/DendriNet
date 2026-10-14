"""
RecurrentTrainer: training strategy for recurrent dendritic networks.

Overrides the standard training and validation-stat collection paths to handle
sequence data with proper temporal loss computation.

Supports:
- AMP (automatic mixed precision) via ``use_amp=True``
- Cosine-annealing LR schedule with linear warmup via
  ``lr_schedule="cosine"`` and ``lr_warmup_epochs=N``
"""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.clipping import clip_gradients
from dendritic_modeling.training.distributed_samplers import (
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.training.strategies.standard import (
    Trainer,
    _ValidationLossStats,
    _ValidationSums,
)

logger = logging.getLogger(__name__)

# Loss names that select the task loss the recurrent loop actually applies.
_RECURRENT_TASK_LOSS_NAMES = {
    "classification": frozenset({"ce", "cross_entropy"}),
    "regression": frozenset({"mse", "mean_squared_error", "ce"}),
}


class RecurrentTrainer(Trainer):
    """Training strategy for recurrent dendritic networks.

    Handles sequence data ([B, T, D] inputs) with proper temporal loss.
    Uses _compute_sequence_loss for both training and validation.

    Args:
        task: "classification" or "regression" -- determines loss function.
        grad_clip_value: Gradient clipping bound (0 or None disables it); the
            norm of all gradients by default (``grad_clip_mode``).
        use_amp: Use automatic mixed precision (default False).
        lr_schedule: LR schedule type -- "none" or "cosine" (default "none").
        lr_warmup_epochs: Number of linear warmup epochs (default 0).
        **kwargs: All standard Trainer arguments.
    """

    # The recurrent loop applies the task loss for model.task itself (see
    # _validate_loss_function), so the base check is replaced.
    checks_task_loss = False

    default_grad_clip_mode = "norm"

    # Options its own step loop does not apply (see STEP_OPTIONS).
    unapplied_options = Trainer.unapplied_options | frozenset(
        {"enable_nan_checking", "profile_dataloader"}
    )

    def __init__(
        self,
        task: str = "classification",
        use_amp: bool = False,
        lr_schedule: str = "none",
        lr_warmup_epochs: int = 0,
        **kwargs,
    ):
        configured_loss = kwargs.get("loss_function")
        super().__init__(**kwargs)
        self.task = task

        # AMP setup
        self.recurrent_amp = use_amp and torch.cuda.is_available()
        self.recurrent_scaler = GradScaler("cuda") if self.recurrent_amp else None

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
        self._validate_loss_function(configured_loss)

        amp_str = "ON" if self.recurrent_amp else "OFF"
        sched_str = self.lr_schedule if self.lr_schedule != "none" else "constant"
        logger.info(
            f"RecurrentTrainer initialized: task={task}, grad_clip_value={self.grad_clip_value}, "
            f"amp={amp_str}, lr_schedule={sched_str}, warmup={lr_warmup_epochs}"
        )

    def _validate_loss_function(self, configured_loss) -> None:
        """Reject a configured loss that the recurrent loop would not use.

        Sequence losses (``temporal_event_loss``/``sequence_prediction_loss``)
        are applied as configured. Any other loss is replaced by the task loss
        (cross-entropy for classification, MSE for regression), so only names
        selecting that loss are accepted. ``"ce"`` is also accepted for
        regression because it is the typed ``CommonTrainerConfig`` default.
        """
        if configured_loss is None:
            return
        loss = self.loss_function
        if bool(getattr(loss, "temporal_event_loss", False)) or bool(
            getattr(loss, "sequence_prediction_loss", False)
        ):
            return
        accepted = _RECURRENT_TASK_LOSS_NAMES[self.task]
        if str(configured_loss) not in accepted:
            raise ValueError(
                f"RecurrentTrainer trains task={self.task!r} with "
                f"{type(self.task_loss).__name__}; loss_function="
                f"{configured_loss!r} would not be applied. Use one of "
                f"{sorted(accepted)} or a sequence loss "
                "('temporal_event_ce', 'masked_dual_task_temporal')."
            )

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

        # Event-time prediction treats the time axis as one categorical
        # distribution.  It must be evaluated before the generic many-to-many
        # flattening below, which intentionally treats timesteps independently.
        if bool(getattr(self.loss_function, "temporal_event_loss", False)):
            return self.loss_function.from_predictions(output, y_batch)

        if bool(getattr(self.loss_function, "sequence_prediction_loss", False)):
            if (
                self._is_distributed
                and self._uses_exact_distributed_training_statistics()
            ):
                local_sums, local_counts = self.loss_function.training_sums_and_counts(
                    output, y_batch
                )
                global_counts = local_counts.detach().clone()
                dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
                gradient_objective = (
                    self.loss_function.reduce_distributed_training_sums_and_counts(
                        local_sums,
                        global_counts,
                        world_size=dist.get_world_size(),
                    )
                )
                global_sums = local_sums.detach().to(torch.float64)
                dist.all_reduce(global_sums, op=dist.ReduceOp.SUM)
                global_value = self.loss_function.reduce_validation_sums_and_counts(
                    global_sums,
                    global_counts,
                ).to(dtype=gradient_objective.dtype)
                return gradient_objective + (global_value - gradient_objective.detach())
            return self.loss_function.from_predictions(output, y_batch)

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

    def _sequence_batch(self, batch, loader: DataLoader):
        """Inputs, targets and optional per-sample lengths on the device.

        Batches are copied without blocking when the loader pins them.
        """
        non_blocking = self._loader_uses_non_blocking_transfer(loader)
        if len(batch) == 3:
            x_batch, y_batch, seq_lengths = batch
            seq_lengths = seq_lengths.to(self.device, non_blocking=non_blocking)
        else:
            x_batch, y_batch = batch
            seq_lengths = None
        return (
            x_batch.to(self.device, non_blocking=non_blocking),
            y_batch.to(self.device, non_blocking=non_blocking),
            seq_lengths,
        )

    def _epoch_train(
        self, model: BaseModel, train_loader: DataLoader
    ) -> tuple[float, float, float]:
        """Returns (total, base, reg) to match Trainer._run_epoch contract."""
        model.train()
        # float64 loss and regularization sums, read once per epoch.
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        reg_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        n_batches = 0
        # Same auxiliary regularization as the standard trainer. Objects built
        # without Trainer.__init__ (unit tests) have no manager.
        manager = getattr(self, "regularization_manager", None)
        has_regularization = manager is not None and manager.has_regularization()

        for batch in train_loader:
            x_batch, y_batch, seq_lengths = self._sequence_batch(batch, train_loader)

            self.optimizer.zero_grad()
            reg_loss = None

            if self.recurrent_amp:
                amp_opt = self.optimizer
                with autocast("cuda"):
                    loss = self._compute_sequence_loss(
                        model, x_batch, y_batch, seq_lengths
                    )
                    if has_regularization:
                        reg_loss = manager.compute_regularization_loss(
                            model,
                            x_batch,
                            y_batch,
                            param_groups=self.optimizer.param_groups,
                        )
                objective = loss if reg_loss is None else loss + reg_loss
                self.recurrent_scaler.scale(objective).backward()
                if self.grad_clip_value:
                    self.recurrent_scaler.unscale_(amp_opt)
                    clip_gradients(
                        model.parameters(), self.grad_clip_value, self.grad_clip_mode
                    )
                # Step through the configured optimizer object directly so AMP
                # skip semantics apply to wrapper logic (e.g., custom decay/rewiring).
                self.recurrent_scaler.step(amp_opt)
                self.recurrent_scaler.update()
            else:
                loss = self._compute_sequence_loss(model, x_batch, y_batch, seq_lengths)
                if has_regularization:
                    reg_loss = manager.compute_regularization_loss(
                        model,
                        x_batch,
                        y_batch,
                        param_groups=self.optimizer.param_groups,
                    )
                objective = loss if reg_loss is None else loss + reg_loss
                objective.backward()
                clip_gradients(
                    model.parameters(), self.grad_clip_value, self.grad_clip_mode
                )
                self.optimizer.step()

            loss_sum += loss.detach().double()
            if reg_loss is not None:
                reg_sum += reg_loss.detach().double()
            n_batches += 1

        # Average detached loss totals across ranks so recurrent DDP training
        # curves are rank-invariant. The exact stratified path already returns
        # the same global-batch value on every rank, which this preserves.
        if self._is_distributed:
            count = torch.full(
                (), float(n_batches), dtype=torch.float64, device=loss_sum.device
            )
            # (loss, batches[, regularization])
            parts = [loss_sum, count] + ([reg_sum] if has_regularization else [])
            statistics = torch.stack(parts)
            dist.all_reduce(statistics, op=dist.ReduceOp.SUM)
            values = statistics.tolist()
            total_loss, n_batches = values[0], int(values[1])
            reg_total = values[2] if has_regularization else 0.0
        else:
            total_loss, reg_total = torch.stack([loss_sum, reg_sum]).tolist()

        avg_loss = total_loss / max(n_batches, 1)
        if not has_regularization:
            return avg_loss, avg_loss, 0.0  # (total, base, reg=0)
        avg_reg = reg_total / max(n_batches, 1)
        return avg_loss + avg_reg, avg_loss, avg_reg

    def _uses_exact_sequence_validation_statistics(self) -> bool:
        return bool(
            getattr(
                self.loss_function,
                "exact_sequence_validation_statistics",
                False,
            )
        )

    def _uses_exact_distributed_training_statistics(self) -> bool:
        return bool(
            getattr(
                self.loss_function,
                "exact_distributed_training_statistics",
                False,
            )
        )

    def _sequence_validation_sums_and_counts(
        self,
        model: BaseModel,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        seq_lengths: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = model(x_batch, seq_lengths=seq_lengths)
        return self.loss_function.validation_sums_and_counts(output, y_batch)

    def _epoch_valid_exact_stats(
        self,
        model: BaseModel,
        valid_loader: DataLoader,
    ) -> _ValidationLossStats:
        """Accumulate exact dataset-level sequence-loss sufficient statistics."""

        model.eval()
        sums = torch.zeros(4, device=self.device, dtype=torch.float64)
        counts = torch.zeros(4, device=self.device, dtype=torch.int64)
        n_batches = 0
        n_samples = 0
        with torch.no_grad():
            for batch in valid_loader:
                x_batch, y_batch, seq_lengths = self._sequence_batch(
                    batch, valid_loader
                )
                batch_sums, batch_counts = self._sequence_validation_sums_and_counts(
                    model,
                    x_batch,
                    y_batch,
                    seq_lengths,
                )
                sums += batch_sums
                counts += batch_counts
                n_batches += 1
                n_samples += int(y_batch.shape[0])
        exact_loss = self.loss_function.reduce_validation_sums_and_counts(
            sums,
            counts,
        ).item()
        return _ValidationLossStats(
            batch_loss_sum=exact_loss * n_batches,
            batch_count=n_batches,
            sample_weighted_loss_sum=exact_loss * n_samples,
            sample_count=n_samples,
        )

    def _epoch_valid_stats(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> _ValidationLossStats:
        """Collect sequence-aware batch- and sample-level loss aggregates."""
        if self._uses_exact_sequence_validation_statistics():
            return self._epoch_valid_exact_stats(model, valid_loader)

        model.eval()
        sums = _ValidationSums(self.device)
        with torch.no_grad():
            for batch in valid_loader:
                x_batch, y_batch, seq_lengths = self._sequence_batch(
                    batch, valid_loader
                )
                loss = self._compute_sequence_loss(model, x_batch, y_batch, seq_lengths)
                sums.add(loss, y_batch.size(0))
        return sums.stats()

    def _validation_loss_reduction(self) -> str | None:
        """Use the recurrent task loss as the aggregation contract."""
        reduction = getattr(self.task_loss, "reduction", None)
        return str(reduction) if reduction is not None else None

    def _epoch_valid_distributed(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> float:
        """Distributed validation using recurrent sequence-aware loss."""
        if self._uses_exact_sequence_validation_statistics():
            return self._epoch_valid_exact_distributed(model, valid_loader)

        model.eval()
        sums = _ValidationSums(self.device)
        sampler = valid_loader.sampler
        if isinstance(sampler, PaddedDistributedEvalSampler):
            real_count = len(sampler.real_indices)
        else:
            real_count = None

        seen = 0
        with torch.no_grad():
            for batch in valid_loader:
                x_batch, y_batch, seq_lengths = self._sequence_batch(
                    batch, valid_loader
                )

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
                sums.add(loss, batch_size)

        return sums.distributed_sample_mean()

    def _epoch_valid_exact_distributed(
        self,
        model: BaseModel,
        valid_loader: DataLoader,
    ) -> float:
        """All-reduce exact four-stratum statistics without local-count gates."""

        model.eval()
        sums = torch.zeros(4, device=self.device, dtype=torch.float64)
        counts = torch.zeros(4, device=self.device, dtype=torch.int64)
        sampler = valid_loader.sampler
        real_count = (
            len(sampler.real_indices)
            if isinstance(sampler, PaddedDistributedEvalSampler)
            else None
        )
        seen = 0
        with torch.no_grad():
            for batch in valid_loader:
                x_batch, y_batch, seq_lengths = self._sequence_batch(
                    batch, valid_loader
                )
                batch_size = int(y_batch.shape[0])
                if real_count is not None:
                    real_in_batch = min(batch_size, real_count - seen)
                    seen += batch_size
                    if real_in_batch <= 0:
                        continue
                    if real_in_batch < batch_size:
                        x_batch = x_batch[:real_in_batch]
                        y_batch = y_batch[:real_in_batch]
                        if seq_lengths is not None:
                            seq_lengths = seq_lengths[:real_in_batch]
                batch_sums, batch_counts = self._sequence_validation_sums_and_counts(
                    model,
                    x_batch,
                    y_batch,
                    seq_lengths,
                )
                sums += batch_sums
                counts += batch_counts

        reduced = torch.cat((sums, counts.to(dtype=torch.float64)))
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        global_sums = reduced[:4]
        global_counts = reduced[4:].to(dtype=torch.int64)
        return self.loss_function.reduce_validation_sums_and_counts(
            global_sums,
            global_counts,
        ).item()
