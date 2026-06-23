"""
FSDP (Fully Sharded Data Parallel) training strategy.

This module provides FSDP-based distributed training functionality
for scaling to very large models across multiple GPUs and nodes.
"""

import logging
import os
import time
from collections.abc import Iterable, Iterator
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

try:  # PyTorch exposes the FSDP-aware scaler from this submodule.
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
except Exception:  # pragma: no cover - depends on installed PyTorch version.
    ShardedGradScaler = None

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.distributed_samplers import (
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.training.fsdp_utils import (
    get_fsdp_config,
    save_fsdp_checkpoint,
    wrap_model_with_fsdp,
)
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.utils.hooks import iter_modules_matching

logger = logging.getLogger(__name__)


def _optimizer_config_value(
    optimizer_config: Optional[dict], key: str, default: float
) -> Any:
    return (optimizer_config or {}).get(key, default)


def _collect_unique_parameters(
    parameters: Iterable[torch.nn.Parameter],
    used_param_ids: set[int],
) -> list[torch.nn.Parameter]:
    unique_params = []
    for param in parameters:
        param_id = id(param)
        if param_id not in used_param_ids:
            unique_params.append(param)
            used_param_ids.add(param_id)
    return unique_params


def _iter_modules_by_class_name(
    model: torch.nn.Module,
    class_name: str,
) -> Iterator[torch.nn.Module]:
    """Yield modules whose concrete class name matches ``class_name``."""
    yield from iter_modules_matching(
        model,
        lambda module: module.__class__.__name__ == class_name,
    )


def _collect_module_parameters_by_class_name(
    model: torch.nn.Module,
    class_name: str,
    used_param_ids: set[int],
) -> list[torch.nn.Parameter]:
    params = []
    for module in _iter_modules_by_class_name(model, class_name):
        params.extend(_collect_unique_parameters(module.parameters(), used_param_ids))
    return params


def _reactivation_parameter_candidates(
    module: torch.nn.Module,
) -> tuple[torch.nn.Parameter, ...]:
    if hasattr(module, "m") and hasattr(module, "b"):
        return module.m, module.b
    if hasattr(module, "log_m") and hasattr(module, "b"):
        return module.log_m, module.b
    return ()


def _has_reactivation_parameter_candidates(module: torch.nn.Module) -> bool:
    """Return whether a module exposes trainable reactivation parameters."""
    return (hasattr(module, "m") or hasattr(module, "log_m")) and hasattr(module, "b")


def _collect_reactivation_parameters(
    model: torch.nn.Module,
    used_param_ids: set[int],
) -> list[torch.nn.Parameter]:
    params = []
    for candidates in _iter_reactivation_parameter_candidates(model):
        params.extend(_collect_unique_parameters(candidates, used_param_ids))
    return params


def _iter_reactivation_parameter_candidates(
    model: torch.nn.Module,
) -> Iterator[tuple[torch.nn.Parameter, ...]]:
    """Yield non-empty reactivation parameter candidate tuples in module order."""
    for module in iter_modules_matching(model, _has_reactivation_parameter_candidates):
        yield _reactivation_parameter_candidates(module)


def _append_fsdp_param_group(
    param_groups: list[dict[str, Any]],
    params: list[torch.nn.Parameter],
    *,
    optimizer_config: Optional[dict],
    lr_key: str,
    default_lr: float,
) -> None:
    if params:
        param_groups.append(
            {
                "params": params,
                "lr": _optimizer_config_value(optimizer_config, lr_key, default_lr),
            }
        )


class FSDPTrainer(Trainer):
    """
    FSDP-enabled trainer for large dendritic models.

    This trainer extends the standard trainer with FSDP support,
    allowing training of models that don't fit on a single GPU.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        fsdp_config: Optional[dict] = None,
        checkpoint_dir: Optional[str] = None,
        save_every_n_epochs: Optional[int] = None,
        optimizer_config: Optional[dict] = None,
        analysis_manager: Optional[object] = None,
        wandb_run: Optional[object] = None,
        **kwargs,
    ):
        """
        Initialize FSDP trainer.

        Args:
            rank: Global rank of current process
            world_size: Total number of processes
            fsdp_config: FSDP configuration
            checkpoint_dir: Directory to save checkpoints
            save_every_n_epochs: Save checkpoint every N epochs (None disables periodic checkpointing)
            optimizer_config: Optimizer configuration
            **kwargs: Arguments passed to base Trainer
        """
        # Store configuration
        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.fsdp_config = fsdp_config or get_fsdp_config()
        self.save_every_n_epochs = save_every_n_epochs

        # Set up checkpoint directory structure (consistent with standard trainer)
        if checkpoint_dir is not None:
            self.checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
        else:
            self.checkpoint_dir = f"results/fsdp_run_{time.time()}/checkpoints"
        self.optimizer_config = optimizer_config or {}
        self.analysis_manager = analysis_manager
        self.wandb_run = wandb_run

        # Set device using local rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

        # Initialize parent with dummy optimizer (we'll create the real one after FSDP wrapping)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.01)
        if self.rank == 0:
            logger.info(f"Initializing FSDPTrainer with kwargs: {kwargs}")

        # Store checkpoint_dir before parent call (parent might override it)
        checkpoint_dir_backup = self.checkpoint_dir

        fsdp_kwargs = kwargs.copy()
        super().__init__(optimizer=dummy_optimizer, **fsdp_kwargs)
        # Restore checkpoint_dir after parent call
        self.checkpoint_dir = checkpoint_dir_backup

        # Create checkpoint directory
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize loss function based on loss_function parameter
        loss_fn_name = kwargs.get("loss_function", "cat_nll")
        if loss_fn_name in ["cat_nll", "cross_entropy", "ce"]:
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_fn_name == "mse":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn_name}")

        # Mixed Precision Training Support
        self.use_amp = kwargs.get("use_amp", False)
        if self.use_amp:
            try:
                # autocast and GradScaler already imported at top
                scaler_cls = ShardedGradScaler or GradScaler
                self.scaler = scaler_cls()
                if self.rank == 0:
                    logger.info("Mixed precision training enabled for FSDP")
            except ImportError:
                if self.rank == 0:
                    logger.warning(
                        "AMP not available, falling back to standard precision"
                    )
                self.use_amp = False

    def _wrap_model(self, model: BaseModel) -> BaseModel:
        """Wrap model with FSDP."""
        # Move base model to device first
        model = model.to(self.device)

        # Wrap with FSDP
        if hasattr(model, "core_network"):
            # For models with separate encoder/decoder
            model.core_network = wrap_model_with_fsdp(
                model.core_network, self.fsdp_config
            )
        elif hasattr(model, "net"):
            # For models with separate encoder/decoder
            model.net = wrap_model_with_fsdp(model.net, self.fsdp_config)

            # Optionally wrap encoder/decoder if they're large
            if hasattr(model, "encoder_network"):
                param_count = sum(p.numel() for p in model.encoder_network.parameters())
                if param_count > 10_000_000:  # > 10M parameters
                    model.encoder_network = wrap_model_with_fsdp(
                        model.encoder_network, self.fsdp_config
                    )

            if hasattr(model, "decoder_network"):
                param_count = sum(p.numel() for p in model.decoder_network.parameters())
                if param_count > 10_000_000:  # > 10M parameters
                    model.decoder_network = wrap_model_with_fsdp(
                        model.decoder_network, self.fsdp_config
                    )
        else:
            # Wrap entire model
            model = wrap_model_with_fsdp(model, self.fsdp_config)

        return model

    def create_optimizer(self, model: BaseModel) -> torch.optim.Optimizer:
        """Create optimizer with FSDP-aware parameter groups."""
        # Get all parameters first
        all_params = list(model.parameters())
        optimizer_config = self.optimizer_config or {}

        # If not using split params, use all parameters with single LR
        if not optimizer_config.get("split_params", False):
            return torch.optim.Adam(
                all_params,
                lr=_optimizer_config_value(optimizer_config, "lr", 0.001),
                weight_decay=_optimizer_config_value(
                    optimizer_config, "weight_decay", 0.0
                ),
            )

        # Otherwise, create parameter groups
        param_groups = []
        used_params = set()

        topk_params = _collect_module_parameters_by_class_name(
            model, "TopKLinear", used_params
        )
        _append_fsdp_param_group(
            param_groups,
            topk_params,
            optimizer_config=optimizer_config,
            lr_key="topk_lr",
            default_lr=0.001,
        )

        blocklinear_params = _collect_module_parameters_by_class_name(
            model, "BlockLinear", used_params
        )
        _append_fsdp_param_group(
            param_groups,
            blocklinear_params,
            optimizer_config=optimizer_config,
            lr_key="blocklinear_lr",
            default_lr=0.001,
        )

        reactivation_params = _collect_reactivation_parameters(model, used_params)
        _append_fsdp_param_group(
            param_groups,
            reactivation_params,
            optimizer_config=optimizer_config,
            lr_key="reactivation_lr",
            default_lr=0.001,
        )

        if hasattr(model, "decoder_network"):
            decoder_params = _collect_unique_parameters(
                model.decoder_network.parameters(), used_params
            )
        else:
            decoder_params = []
        _append_fsdp_param_group(
            param_groups,
            decoder_params,
            optimizer_config=optimizer_config,
            lr_key="decoder_lr",
            default_lr=0.001,
        )

        # All other parameters
        other_params = [p for p in all_params if id(p) not in used_params]
        _append_fsdp_param_group(
            param_groups,
            other_params,
            optimizer_config=optimizer_config,
            lr_key="lr",
            default_lr=0.001,
        )

        return torch.optim.Adam(
            param_groups,
            weight_decay=_optimizer_config_value(optimizer_config, "weight_decay", 0.0),
        )

    def train(
        self,
        model: BaseModel,
        train_loader: DataLoader | Dataset,
        valid_loader: Optional[DataLoader | Dataset] = None,
    ) -> dict[str, Any]:
        """
        Train model using FSDP with pre-created DataLoaders.

        Args:
            model: Model to train (will be wrapped with FSDP)
            train_loader: Training DataLoader (or training Dataset for auto-wrapping)
            valid_loader: Validation DataLoader (or validation Dataset for auto-wrapping)

        Returns:
            Training results dictionary
        """
        if isinstance(train_loader, DataLoader):
            if valid_loader is not None and not isinstance(valid_loader, DataLoader):
                raise TypeError(
                    "If train_loader is a DataLoader, valid_loader must also be a DataLoader or None."
                )
        else:
            train_loader, valid_loader = self._dataloaders(train_loader, valid_loader)

        # Wrap model with FSDP
        model = wrap_model_with_fsdp(model, self.fsdp_config)

        # Only move to device if CPU offloading is NOT enabled
        # When CPU offloading is enabled, FSDP manages device placement
        cpu_offload_enabled = self.fsdp_config.get(
            "cpu_offload"
        ) is not None and self.fsdp_config.get("cpu_offload")
        if not cpu_offload_enabled:
            model = model.to(self.device)

        # Store the FSDP-wrapped model so callers can use it for
        # checkpoint saving / full-param gathering after training.
        self.fsdp_model = model

        # Create optimizer after FSDP wrapping
        self.optimizer = self.create_optimizer(model)

        # Training loop
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        best_valid_loss = float("inf")
        patience_counter = 0

        if self.rank == 0:
            logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(self.epochs):
            if self.rank == 0:
                logger.info(f"Starting epoch {epoch}")
            epoch_start = time.time()

            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # Training
            train_loss, train_acc = self._epoch_train_fsdp(model, train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            valid_loss = None
            valid_acc = None
            if valid_loader is not None:
                if hasattr(valid_loader.sampler, "set_epoch"):
                    valid_loader.sampler.set_epoch(epoch)
                valid_loss, valid_acc = self._epoch_valid_fsdp(model, valid_loader)
                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)

                # Early stopping
                if self.early_stopping and valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    if self.checkpointing:
                        save_fsdp_checkpoint(
                            model,
                            self.optimizer,
                            epoch,
                            os.path.join(self.checkpoint_dir, "best_model.pt"),
                            rank=self.rank,
                        )
                elif self.early_stopping:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.rank == 0:
                            logger.info(f"Early stopping at epoch {epoch}")
                        break

            # Epoch-level analysis is intentionally skipped for FSDP.
            # FSDP shards model parameters across ranks; running analysis
            # on the sharded model produces incorrect results.  Final
            # analysis runs on rank 0 after gathering the full state dict
            # (see train_experiments_fsdp.py).

            # Logging
            epoch_time = time.time() - epoch_start
            if self.rank == 0 and epoch % self.print_every == 0:
                msg = f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}"
                if valid_loss is not None and valid_acc is not None:
                    msg += (
                        f" - Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_acc:.4f}"
                    )
                msg += f" - Time: {epoch_time:.2f}s"
                logger.info(msg)

            # WandB logging (only on rank 0)
            if self.rank == 0 and self.wandb_run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "train/epoch_time": epoch_time,
                }
                if valid_loss is not None and valid_acc is not None:
                    log_dict.update(
                        {
                            "valid/loss": valid_loss,
                            "valid/accuracy": valid_acc,
                        }
                    )
                self.wandb_run.log(log_dict)

            # Checkpointing
            if (
                self.save_every_n_epochs is not None
                and self.save_every_n_epochs > 0
                and (epoch + 1) % self.save_every_n_epochs == 0
            ):
                save_fsdp_checkpoint(
                    model,
                    self.optimizer,
                    epoch,
                    os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
                    rank=self.rank,
                )

        # Final save
        save_fsdp_checkpoint(
            model,
            self.optimizer,
            self.epochs - 1,
            os.path.join(self.checkpoint_dir, "final_model.pt"),
            rank=self.rank,
        )

        # Generate plots (only on rank 0)
        if self.rank == 0:
            self._loss_plotting(train_losses, valid_losses)
            # Removed redundant accuracy plotting - handled by performance option

        return {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "best_valid_loss": best_valid_loss,
            "train_accs": train_accs,
            "valid_accs": valid_accs,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_valid_loss": valid_losses[-1] if valid_losses else None,
        }

    def _epoch_train_fsdp(
        self, model: BaseModel, train_loader: DataLoader
    ) -> tuple[float, float]:
        """Training epoch with FSDP."""
        model.train()
        total_loss = 0
        total_correct = 0
        total_examples = 0

        for _batch_idx, batch in enumerate(train_loader):
            # Clear GPU cache before each forward pass
            torch.cuda.empty_cache()

            x_batch = batch[0].to(self.device)
            y_batch = batch[1].to(self.device)

            if self.use_amp:
                # Mixed precision forward pass
                with autocast():
                    outputs = model(x_batch)
                    loss = self.loss_fn(outputs, y_batch)

                # Backward pass with scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.grad_clip_value is not None:
                    self.scaler.unscale_(self.optimizer)
                    self._clip_grad_norm_fsdp(model)

                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                outputs = model(x_batch)
                loss = self.loss_fn(outputs, y_batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.grad_clip_value is not None:
                    self._clip_grad_norm_fsdp(model)

                # Optimizer step
                self.optimizer.step()

            bs = y_batch.size(0)
            total_loss += loss.item() * bs
            total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total_examples += bs

        # All-reduce exact counts across ranks.
        stats = torch.tensor(
            [total_loss, float(total_correct), float(total_examples)],
            device=self.device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        global_examples = stats[2].item()
        avg_loss = stats[0].item() / max(global_examples, 1)
        avg_acc = stats[1].item() / max(global_examples, 1)

        return avg_loss, avg_acc

    def _clip_grad_norm_fsdp(self, model: BaseModel):
        """Clip FSDP gradients through FSDP's collective-aware method."""
        if hasattr(model, "clip_grad_norm_"):
            return model.clip_grad_norm_(self.grad_clip_value)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_value)

    def _epoch_valid_fsdp(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> tuple[float, float]:
        """Validation epoch with FSDP.

        All ranks must participate (FSDP needs all-gather during forward).
        We accumulate sample-weighted loss, correct count, and total
        examples, then all-reduce with SUM for an exact global average.

        When used with ``PaddedDistributedEvalSampler``, padding indices
        (-1) may appear in the last batch.  These are sentinel values that
        index into real data (Python wraps -1 to the last element), so the
        forward pass succeeds on every rank, but we exclude them from the
        stats by tracking the known number of real samples.
        """
        model.eval()
        weighted_loss_sum = 0.0
        total_correct = 0
        total_examples = 0

        # Determine how many real (non-padding) samples this rank has.
        sampler = valid_loader.sampler
        if isinstance(sampler, PaddedDistributedEvalSampler):
            real_count = len(sampler.real_indices)
        else:
            real_count = None  # unknown — count all

        seen = 0
        with torch.no_grad():
            for batch in valid_loader:
                x_batch = batch[0].to(self.device)
                y_batch = batch[1].to(self.device)
                bs = y_batch.size(0)

                outputs = model(x_batch)
                loss = self.loss_fn(outputs, y_batch)

                if real_count is not None:
                    # Only count the real (non-padding) samples.
                    real_in_batch = min(bs, real_count - seen)
                    if real_in_batch <= 0:
                        # Entire batch is padding — still ran forward for
                        # FSDP all-gather, but skip stats.
                        seen += bs
                        continue
                    if real_in_batch < bs:
                        # Partial padding in this batch.
                        outputs = outputs[:real_in_batch]
                        y_batch = y_batch[:real_in_batch]
                        loss = self.loss_fn(outputs, y_batch)
                        bs = real_in_batch
                    seen += bs

                weighted_loss_sum += loss.item() * bs
                total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                total_examples += bs

        # All-reduce exact counts across ranks.
        stats = torch.tensor(
            [weighted_loss_sum, float(total_correct), float(total_examples)],
            device=self.device,
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        global_examples = stats[2].item()
        avg_loss = stats[0].item() / max(global_examples, 1)
        avg_acc = stats[1].item() / max(global_examples, 1)

        return avg_loss, avg_acc

    def _reduce_loss(self, loss: float) -> float:
        """Reduce loss across all processes."""
        loss_tensor = torch.tensor(loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor.item()

    def _save_checkpoint(self, model: BaseModel, epoch: int, is_final: bool = False):
        """Save FSDP checkpoint."""
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        checkpoint_name = (
            "final_checkpoint.pt" if is_final else f"checkpoint_epoch_{epoch}.pt"
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)

        save_fsdp_checkpoint(
            model=model,
            optimizer=self.optimizer,
            epoch=epoch,
            save_path=checkpoint_path,
            rank=self.rank,
        )

    # Removed _accuracy_plotting method - redundant with performance option plotting
