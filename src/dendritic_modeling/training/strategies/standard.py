"""
Base training strategy classes.

This module contains the base trainer class that provides common functionality
for all training strategies in the dendritic modeling framework.
"""

import logging
import os
import time
from copy import deepcopy
from typing import Optional

import torch
import torch.distributed as dist
import torch.utils.data
import wandb
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader, DistributedSampler

from dendritic_modeling.config.training import (
    PruningConfig,
    RegularizationConfig,
    ReportingConfig,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.plotting import plot_loss_curves
from dendritic_modeling.training.loss.functions import LossFunction, get_loss_function
from dendritic_modeling.training.regularization import RegularizationManager

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss_function: str = "mse",
        epochs: int = 100,
        early_stopping: bool = False,
        patience: int = 10,
        batch_size: int = 256,
        shuffle: bool = True,
        grad_clip_value: float = 5,
        checkpointing: bool = False,
        checkpoint_interval: int = 10,
        load_best_state_dict: bool = True,
        plot_losses: bool = False,
        save_path: Optional[str] = None,
        suppress_prints=False,
        print_every=10,
        analysis_manager: Optional[object] = None,
        **kwargs,
    ):
        self.optimizer = optimizer
        self.loss_function: LossFunction = get_loss_function(loss_function)
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.grad_clip_value = grad_clip_value
        self.checkpointing = checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.load_best_state_dict = load_best_state_dict
        self.plot_losses = plot_losses
        self.save_path = save_path
        self.suppress_prints = suppress_prints
        self.print_every = print_every
        self.analysis_manager = analysis_manager

        regularization_config = kwargs.get("regularization_config")
        pruning_config = kwargs.get("pruning_config")
        reporting_config = kwargs.get("reporting_config")

        # Initialize enhanced feature configs with safe defaults
        self.regularization_config = regularization_config or RegularizationConfig()
        self.pruning_config = pruning_config or PruningConfig()
        self.reporting_config = reporting_config or ReportingConfig()

        # Initialize RegularizationManager
        self.regularization_manager = RegularizationManager(self.regularization_config)

        # Log regularization settings
        if (
            self.regularization_config.l1_weight > 0
            or self.regularization_config.l2_weight > 0
        ):
            logger.info(
                f"L1/L2 regularization enabled: l1_weight={self.regularization_config.l1_weight}, l2_weight={self.regularization_config.l2_weight}"
            )
        if self.regularization_config.enforce_ei_weight_ratio:
            logger.info(
                f"E/I weight ratio regularization enabled: target_ratio={self.regularization_config.target_ei_weight_ratio}, loss_weight={self.regularization_config.ei_ratio_loss_weight}"
            )

        # Performance optimization flags (disabled by default for speed)
        self.enable_nan_checking = kwargs.get("enable_nan_checking", False)
        self.enable_adaptive_clipping = kwargs.get("enable_adaptive_clipping", False)

        self.enable_amp = kwargs.get("enable_amp", kwargs.get("use_amp", False))
        if self.enable_amp and torch.cuda.is_available():
            try:
                self.scaler = GradScaler()
                logger.info("Mixed precision training enabled")
            except ImportError:
                logger.warning("AMP not available, falling back to standard precision")
                self.enable_amp = False
        else:
            self.enable_amp = False

        self.enable_profiling = kwargs.get("enable_profiling", False)
        self.profiler = None

        if self.enable_profiling:
            # Auto-disable profiling for small-scale/CPU runs to improve speed
            if not torch.cuda.is_available():
                logger.info("Profiling auto-disabled for CPU run to improve speed")
                self.enable_profiling = False
            elif self.batch_size < 32:
                logger.info(
                    "Profiling auto-disabled for small batch size to reduce overhead"
                )
                self.enable_profiling = False
            else:
                try:
                    import torch.profiler as torch_profiler

                    profiling_output_dir = kwargs.get(
                        "profiling_output_dir", "outputs/profiling/"
                    )
                    os.makedirs(profiling_output_dir, exist_ok=True)

                    self.profiler = torch_profiler.profile(
                        activities=[
                            torch_profiler.ProfilerActivity.CPU,
                            torch_profiler.ProfilerActivity.CUDA,
                        ],
                        schedule=torch_profiler.schedule(
                            wait=1, warmup=1, active=3, repeat=2
                        ),
                        on_trace_ready=torch_profiler.tensorboard_trace_handler(
                            profiling_output_dir
                        ),
                        record_shapes=True,
                        with_stack=True,
                    )
                    logger.info(
                        f"Profiling enabled - output directory: {profiling_output_dir}"
                    )
                except ImportError:
                    logger.warning("torch.profiler not available, profiling disabled")
                    self.enable_profiling = False

        self.filename_prefix = "standard_"
        self._train_sampler: DistributedSampler | None = None
        for k, v in kwargs.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # DDP helpers
    # ------------------------------------------------------------------

    @property
    def _is_distributed(self) -> bool:
        return dist.is_initialized()

    @property
    def _is_main_process(self) -> bool:
        return not self._is_distributed or dist.get_rank() == 0

    @staticmethod
    def _unwrap_model(model):
        """Return the underlying model, stripping DDP wrapper if present."""
        return model.module if hasattr(model, "module") else model

    def _dataloaders(
        self, train_data: torch.utils.data.Dataset, valid_data: torch.utils.data.Dataset
    ) -> tuple[DataLoader, DataLoader]:
        # Respect scheduler CPU affinity when available (e.g., SLURM cpus-per-task).
        available_cpus = os.cpu_count() or 1
        if hasattr(os, "sched_getaffinity"):
            try:
                available_cpus = len(os.sched_getaffinity(0))
            except OSError:
                pass

        # Determine number of workers automatically or use override
        if hasattr(self, "num_workers") and self.num_workers is not None:
            num_workers = max(0, int(self.num_workers))
        else:
            if torch.cuda.is_available():
                # On GPU jobs, cap workers to avoid oversubscription warnings/freezes.
                num_workers = min(16, available_cpus)
            else:
                # On CPU-only runs, leave one core for the main process and cap to 8.
                num_workers = min(8, max(0, available_cpus - 1))

        # Never spawn more workers than visible CPUs.
        if num_workers > available_cpus:
            num_workers = available_cpus

        # Log the auto-detected value
        if not self.suppress_prints and (
            not hasattr(self, "num_workers") or self.num_workers is None
        ):
            total_cpus = os.cpu_count() or available_cpus
            logger.info(
                "Auto-detected num_workers=%s (visible CPUs: %s, total system CPUs: %s)",
                num_workers,
                available_cpus,
                total_cpus,
            )

        # Check if we should use pinned memory and persistent workers
        use_pin_memory = torch.cuda.is_available() and num_workers > 0
        use_persistent = num_workers > 0

        # Use DistributedSampler for training when running under DDP.
        # Validation is run only on rank 0 with a normal loader to avoid
        # the padding bias introduced by DistributedSampler.
        if self._is_distributed:
            self._train_sampler = DistributedSampler(train_data, shuffle=self.shuffle)
            train_shuffle = False  # sampler handles shuffling
        else:
            self._train_sampler = None
            train_shuffle = self.shuffle

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=self._train_sampler,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent,
        )
        valid_loader = DataLoader(
            valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=use_pin_memory,
            persistent_workers=use_persistent,
        )
        return train_loader, valid_loader

    def _initialize_attributes(self, model: BaseModel, total_epochs: int) -> None:
        if not hasattr(self, "device"):
            if self._is_distributed:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self.device = f"cuda:{local_rank}"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(self.device)
        if not hasattr(self, "best_state_dict"):
            self.best_state_dict = deepcopy(model.state_dict())
        if not hasattr(self, "best_loss"):
            self.best_loss = float("inf")
        if not hasattr(self, "best_epoch"):
            self.best_epoch = 1
        if not hasattr(self, "start_time"):
            self.start_time = time.time()
        if not hasattr(self, "logger_info_prefix"):
            self.logger_info_prefix = "[Standard] "
        self.epoch_counter = 0
        self.total_epochs = total_epochs
        self.patience_counter = 0

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict:
        train_loader, valid_loader = self._dataloaders(train_data, valid_data)

        train_losses = []
        valid_losses = []
        train_losses_base = []
        train_losses_reg = []

        self._initialize_attributes(model, self.epochs)
        self._save_checkpoint(model)

        train_losses, valid_losses, train_losses_base, train_losses_reg = (
            self._run_training_loop(
                n_epochs=self.epochs,
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_losses=train_losses,
                valid_losses=valid_losses,
                train_losses_base=train_losses_base,
                train_losses_reg=train_losses_reg,
                logger_info=self.logger_info_prefix,
                wandb_info={},
            )
        )

        if self._is_main_process:
            logger.info(
                f"Best epoch: {self.best_epoch}, " f"Best loss: {self.best_loss:.4f}"
            )
        if self.load_best_state_dict:
            self._unwrap_model(model).load_state_dict(self.best_state_dict)
            if self._is_main_process:
                logger.info("Final model loaded from best checkpoint")

        results = {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_state_dict": self.best_state_dict,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_losses_base": train_losses_base,
            "train_losses_reg": train_losses_reg,
        }

        self._loss_plotting(
            train_losses, valid_losses, train_losses_base, train_losses_reg
        )

        return results

    def _run_training_loop(
        self,
        n_epochs: int,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: list[float],
        valid_losses: list[float],
        train_losses_base: list[float],
        train_losses_reg: list[float],
        logger_info: str = "",
        wandb_info: dict | None = None,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        if wandb_info is None:
            wandb_info = {}
        for _ in range(1, n_epochs + 1):
            train_losses, valid_losses, train_losses_base, train_losses_reg = (
                self._run_epoch(
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    train_losses_base=train_losses_base,
                    train_losses_reg=train_losses_reg,
                    logger_info=logger_info,
                    wandb_info=wandb_info,
                )
            )

            if self._check_early_stopping():
                break

        return train_losses, valid_losses, train_losses_base, train_losses_reg

    def _run_epoch(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: list[float],
        valid_losses: list[float],
        train_losses_base: list[float],
        train_losses_reg: list[float],
        logger_info: str = "",
        wandb_info: dict | None = None,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        if wandb_info is None:
            wandb_info = {}
        self.epoch_counter += 1

        # Tell DistributedSampler which epoch we're in so each rank
        # sees a different shuffle order per epoch.
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(self.epoch_counter)

        train_loss_total, train_loss_base, train_loss_reg = self._epoch_train(
            model, train_loader
        )
        train_losses.append(train_loss_total)
        train_losses_base.append(train_loss_base)
        train_losses_reg.append(train_loss_reg)
        # Rank 0 runs validation on the full (unsharded) dataset using
        # the unwrapped model.  This avoids both the padding bias from
        # DistributedSampler and any distributed forward behaviour
        # (broadcast_buffers) from the DDP wrapper.
        if self._is_distributed:
            if self._is_main_process:
                unwrapped = self._unwrap_model(model)
                valid_loss_sum, valid_n_batches = self._epoch_valid(
                    unwrapped, valid_loader
                )
                valid_loss = valid_loss_sum / max(valid_n_batches, 1)
            else:
                valid_loss = 0.0
            vl = torch.tensor([valid_loss], device=self.device)
            dist.broadcast(vl, src=0)
            valid_loss = vl.item()
        else:
            valid_loss_sum, valid_n_batches = self._epoch_valid(model, valid_loader)
            valid_loss = valid_loss_sum / max(valid_n_batches, 1)

        valid_losses.append(valid_loss)

        self._update_best(model, valid_loss)
        self._save_checkpoint(model)
        self._analyze()
        self._tracking(
            train_loss_total,
            valid_loss,
            logger_info,
            wandb_info,
            train_loss_base=train_loss_base,
            train_loss_reg=train_loss_reg,
        )
        return train_losses, valid_losses, train_losses_base, train_losses_reg

    def _epoch_train(
        self, model: BaseModel, train_loader: DataLoader
    ) -> tuple[float, float, float]:
        model.train()
        train_loss_total = 0
        train_loss_base = 0
        train_loss_reg = 0

        # Check once if regularization is enabled to avoid repeated checks
        has_regularization = self.regularization_manager.has_regularization()

        for batch in train_loader:
            x_batch, y_batch = batch[0].to(self.device), batch[1].to(self.device)

            if self.enable_amp:
                # Mixed precision forward pass
                with autocast():
                    base_loss = self.loss_function(model, x_batch, y_batch)

                    # Add all regularization losses only if regularization is enabled
                    if has_regularization:
                        reg_loss = (
                            self.regularization_manager.compute_regularization_loss(
                                model, x_batch, y_batch
                            )
                        )
                        total_loss = base_loss + reg_loss
                        reg_loss_item = reg_loss.item()
                    else:
                        total_loss = base_loss
                        reg_loss_item = 0.0

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()

                # Optional expensive operations (configurable for performance)
                if self.enable_nan_checking:
                    nan_detected = self._check_and_handle_nan_gradients(model)
                    if nan_detected:
                        logger.warning(
                            "NaNs detected and handled in gradients during AMP training"
                        )

                # Gradient clipping with AMP
                if self.grad_clip_value:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    if self.enable_adaptive_clipping:
                        self._apply_adaptive_gradient_clipping(model)
                    else:
                        clip_grad_value_(model.parameters(), self.grad_clip_value)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                base_loss: torch.Tensor = self.loss_function(model, x_batch, y_batch)

                # Add all regularization losses only if regularization is enabled
                if has_regularization:
                    reg_loss = self.regularization_manager.compute_regularization_loss(
                        model, x_batch, y_batch
                    )
                    total_loss = base_loss + reg_loss
                    reg_loss_item = reg_loss.item()
                else:
                    total_loss = base_loss
                    reg_loss_item = 0.0

                self.optimizer.zero_grad()
                total_loss.backward()

                # Optional expensive operations (configurable for performance)
                if self.enable_nan_checking:
                    nan_detected = self._check_and_handle_nan_gradients(model)
                    if nan_detected:
                        logger.warning("NaNs detected and handled in gradients")

                # Gradient clipping: adaptive (expensive) vs simple (fast)
                if self.enable_adaptive_clipping:
                    self._apply_adaptive_gradient_clipping(model)
                else:
                    # Use simple, fast gradient clipping (default)
                    clip_grad_value_(model.parameters(), self.grad_clip_value)
                self.optimizer.step()

            train_loss_total += total_loss.item()
            train_loss_base += base_loss.item()
            train_loss_reg += reg_loss_item

        train_loss_total /= len(train_loader)
        train_loss_base /= len(train_loader)
        train_loss_reg /= len(train_loader)
        return train_loss_total, train_loss_base, train_loss_reg

    def _epoch_valid(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> tuple[float, int]:
        """Return (loss_sum, batch_count) so callers can reduce correctly."""
        model.eval()
        valid_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in valid_loader:
                x_batch: torch.Tensor = batch[0].to(self.device)
                y_batch: torch.Tensor = batch[1].to(self.device)
                loss: torch.Tensor = self.loss_function(model, x_batch, y_batch)
                valid_loss += loss.item()
                n_batches += 1
        return valid_loss, n_batches

    def _update_best(self, model: BaseModel, valid_loss: float) -> None:
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_epoch = self.epoch_counter
            self.best_state_dict = deepcopy(self._unwrap_model(model).state_dict())
            self.patience_counter = 0
            if (
                self._is_main_process
                and self.save_path
                and os.path.exists(self.save_path)
            ):
                torch.save(
                    self.best_state_dict,
                    os.path.join(
                        self.save_path, f"{self.filename_prefix}best_model.pt"
                    ),
                )
        else:
            self.patience_counter += 1

    def _save_checkpoint(self, model: BaseModel):
        if not self._is_main_process:
            return
        if self.checkpointing and (self.epoch_counter % self.checkpoint_interval == 0):
            checkpoint_path = os.path.join(
                self.save_path, f"{self.filename_prefix}checkpoints"
            )
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(
                self._unwrap_model(model).state_dict(),
                os.path.join(checkpoint_path, f"epoch_{self.epoch_counter}.pt"),
            )

    def _analyze(self):
        if not self._is_main_process:
            return
        if self.analysis_manager is not None:
            # Lazy import to avoid circular dependency
            from dendritic_modeling.analysis import AnalysisManager

            if isinstance(self.analysis_manager, AnalysisManager):
                self.analysis_manager.run_analysis(
                    filename=f"epoch{self.epoch_counter}", training=True
                )

    def _tracking(
        self,
        train_loss: float,
        valid_loss: float,
        logger_info: str = "",
        wandb_info: dict | None = None,
        train_loss_base: Optional[float] = None,
        train_loss_reg: Optional[float] = None,
    ) -> None:
        if wandb_info is None:
            wandb_info = {}
        elapsed = time.time() - self.start_time
        if not self.suppress_prints and (self.epoch_counter % self.print_every == 0):
            if train_loss_base is not None and train_loss_reg is not None:
                logger_info = (
                    logger_info
                    + f"epoch={self.epoch_counter}/{self.total_epochs}, "
                    + f"train_loss={train_loss:.4f} (base={train_loss_base:.4f}, reg={train_loss_reg:.4f}), "
                    + f"valid_loss={valid_loss:.4f}, "
                    + f"elapsed={elapsed:.1f}s"
                )
            else:
                logger_info = (
                    logger_info
                    + f"epoch={self.epoch_counter}/{self.total_epochs}, "
                    + f"train_loss={train_loss:.4f}, "
                    + f"valid_loss={valid_loss:.4f}, "
                    + f"elapsed={elapsed:.1f}s"
                )
            logger.info(logger_info)

        if wandb.run is not None:
            wandb_info["epoch"] = self.epoch_counter
            wandb_info["train_loss"] = train_loss
            wandb_info["valid_loss"] = valid_loss
            wandb_info["elapsed"] = elapsed
            if train_loss_base is not None:
                wandb_info["train_loss_base"] = train_loss_base
            if train_loss_reg is not None:
                wandb_info["train_loss_reg"] = train_loss_reg
            wandb.log(wandb_info)

    def _check_early_stopping(self) -> bool:
        if self.early_stopping and self.patience_counter >= self.patience:
            if not self.suppress_prints:
                logger.info(
                    f"Early stopping triggered " f"after {self.epoch_counter} epochs."
                )
            return True
        else:
            return False

    def _loss_plotting(
        self,
        train_losses: list[float],
        valid_losses: list[float],
        train_losses_base: Optional[list[float]] = None,
        train_losses_reg: Optional[list[float]] = None,
    ) -> None:
        if not self._is_main_process:
            return
        if self.plot_losses and len(train_losses) > 1:
            if self.save_path and os.path.exists(self.save_path):
                plot_loss_curves(
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    loss_name=self.loss_function._loss_name,
                    save_dir=self.save_path,
                    filename_prefix=self.filename_prefix,
                    train_losses_base=train_losses_base,
                    train_losses_reg=train_losses_reg,
                )

    def _check_and_handle_nan_gradients(self, model: BaseModel) -> bool:
        """
        Check for NaN values in gradients and handle them to prevent training crashes.


        Args:
            model: The model to check gradients for

        Returns:
            bool: True if NaNs were detected and handled, False otherwise
        """
        nan_detected = False

        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_detected = True
                    # Replace NaNs with zeros to prevent propagation
                    param.grad = torch.nan_to_num(
                        param.grad, nan=0.0, posinf=1.0, neginf=-1.0
                    )

                    logger.warning(f"NaNs detected in gradients for parameter: {name}")

        return nan_detected

    def _apply_adaptive_gradient_clipping(self, model: BaseModel) -> float:
        """
        Apply adaptive gradient clipping with dynamic learning rate adjustment.


        Args:
            model: The model to apply gradient clipping to

        Returns:
            float: The gradient norm before clipping
        """
        if not self.grad_clip_value:
            return 0.0

        # Calculate gradient norm for adaptive clipping (iterates all parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=self.grad_clip_value
        )
        return grad_norm


__all__ = ["Trainer"]
