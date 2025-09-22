"""
FSDP (Fully Sharded Data Parallel) training strategy.

This module provides FSDP-based distributed training functionality
for scaling to very large models across multiple GPUs and nodes.
"""

import logging
import os
import time
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.fsdp_utils import (
    get_fsdp_config,
    save_fsdp_checkpoint,
    wrap_model_with_fsdp,
)
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


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

        # Handle AMP parameter name mismatch between FSDP and standard trainer
        fsdp_kwargs = kwargs.copy()
        if "use_amp" in fsdp_kwargs:
            fsdp_kwargs["enable_amp"] = fsdp_kwargs.pop("use_amp")

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
                self.scaler = GradScaler()
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
        if hasattr(model, "net"):
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

        # If not using split params, use all parameters with single LR
        if not (self.optimizer_config or {}).get("split_params", False):
            return torch.optim.Adam(
                all_params,
                lr=(self.optimizer_config or {}).get("lr", 0.001),
                weight_decay=(self.optimizer_config or {}).get("weight_decay", 0.0),
            )

        # Otherwise, create parameter groups
        param_groups = []
        used_params = set()

        # TopK parameters
        topk_params = []
        for module in model.modules():
            if module.__class__.__name__ == "TopKLinear":
                for param in module.parameters():
                    topk_params.append(param)
                    used_params.add(id(param))
        if topk_params:
            param_groups.append(
                {
                    "params": topk_params,
                    "lr": (self.optimizer_config or {}).get("topk_lr", 0.001),
                }
            )

        # BlockLinear parameters
        blocklinear_params = []
        for module in model.modules():
            if module.__class__.__name__ == "BlockLinear":
                blocklinear_params.extend(module.parameters())
        if blocklinear_params:
            param_groups.append(
                {
                    "params": blocklinear_params,
                    "lr": (self.optimizer_config or {}).get("blocklinear_lr", 0.001),
                }
            )

        # Reactivation parameters
        reactivation_params = []
        for module in model.modules():
            if hasattr(module, "m") and hasattr(module, "b"):  # ParametricTanh
                reactivation_params.extend([module.m, module.b])
        if reactivation_params:
            param_groups.append(
                {
                    "params": reactivation_params,
                    "lr": (self.optimizer_config or {}).get("reactivation_lr", 0.001),
                }
            )

        # Decoder parameters
        decoder_params = []
        if hasattr(model, "decoder_network"):
            decoder_params = list(model.decoder_network.parameters())
        if decoder_params:
            param_groups.append(
                {
                    "params": decoder_params,
                    "lr": (self.optimizer_config or {}).get("decoder_lr", 0.001),
                }
            )

        # All other parameters
        other_params = [p for p in all_params if id(p) not in used_params]
        if other_params:
            param_groups.append(
                {
                    "params": other_params,
                    "lr": (self.optimizer_config or {}).get("lr", 0.001),
                }
            )

        return torch.optim.Adam(
            param_groups,
            weight_decay=(self.optimizer_config or {}).get("weight_decay", 0.0),
        )

    def train(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
    ) -> dict[str, Any]:
        """
        Train model using FSDP with pre-created DataLoaders.

        Args:
            model: Model to train (will be wrapped with FSDP)
            train_loader: Training DataLoader with DistributedSampler
            valid_loader: Validation DataLoader with DistributedSampler

        Returns:
            Training results dictionary
        """
        # Wrap model with FSDP
        model = wrap_model_with_fsdp(model, self.fsdp_config)

        # Only move to device if CPU offloading is NOT enabled
        # When CPU offloading is enabled, FSDP manages device placement
        cpu_offload_enabled = self.fsdp_config.get(
            "cpu_offload"
        ) is not None and self.fsdp_config.get("cpu_offload")
        if not cpu_offload_enabled:
            model = model.to(self.device)

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
                    if self.rank == 0 and self.save_checkpoints:
                        save_fsdp_checkpoint(
                            model,
                            self.optimizer,
                            epoch,
                            os.path.join(self.checkpoint_dir, "best_model.pt"),
                        )
                elif self.early_stopping:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.rank == 0:
                            logger.info(f"Early stopping at epoch {epoch}")
                        break

            # Analysis (only on rank 0 to avoid duplication)
            if self.rank == 0 and self.analysis_manager is not None:
                try:
                    # Run analysis if enabled and we have the manager
                    if self.analysis_manager is not None:
                        # Lazy import to avoid circular dependency
                        from dendritic_modeling.analysis import AnalysisManager

                        if isinstance(self.analysis_manager, AnalysisManager):
                            # Only log analysis runs every 10 epochs to reduce verbosity
                            if epoch % 10 == 0:
                                logger.info(
                                    f"Running epoch analysis for epoch {epoch + 1}..."
                                )
                            self.analysis_manager.run_analysis(
                                filename=f"epoch{epoch + 1}", training=True
                            )
                            if epoch % 10 == 0:
                                logger.info(
                                    f"Epoch analysis completed for epoch {epoch + 1}"
                                )
                        else:
                            logger.warning(
                                f"Analysis manager is not AnalysisManager instance: {type(self.analysis_manager)}"
                            )
                    else:
                        logger.warning("Analysis manager is None")
                except Exception as e:
                    logger.warning(f"Analysis failed for epoch {epoch + 1}: {e}")
                    import traceback

                    logger.warning(
                        f"Analysis error traceback: {traceback.format_exc()}"
                    )

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
                self.rank == 0
                and self.save_every_n_epochs is not None
                and self.save_every_n_epochs > 0
                and (epoch + 1) % self.save_every_n_epochs == 0
            ):
                save_fsdp_checkpoint(
                    model,
                    self.optimizer,
                    epoch,
                    os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
                )

        # Final save
        if self.rank == 0:
            save_fsdp_checkpoint(
                model,
                self.optimizer,
                self.epochs - 1,
                os.path.join(self.checkpoint_dir, "final_model.pt"),
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

        for _batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # Clear GPU cache before each forward pass
            torch.cuda.empty_cache()

            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

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
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.grad_clip_value
                    )

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
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.grad_clip_value
                    )

                # Optimizer step
                self.optimizer.step()

            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
            total_examples += y_batch.size(0)

        # Average loss across batches
        avg_loss = total_loss / (len(train_loader))
        avg_acc = total_correct / total_examples if total_examples > 0 else 0.0

        # Reduce loss across all processes
        avg_loss_tensor = torch.tensor([avg_loss, avg_acc], device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor[0].item() / self.world_size
        avg_acc = avg_loss_tensor[1].item() / self.world_size

        return avg_loss, avg_acc

    def _epoch_valid_fsdp(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> tuple[float, float]:
        """Validation epoch with FSDP."""
        model.eval()
        total_loss = 0
        total_correct = 0
        total_examples = 0

        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                outputs = model(x_batch)
                loss = self.loss_fn(outputs, y_batch)

                total_loss += loss.item()
                total_correct += (outputs.argmax(dim=1) == y_batch).sum().item()
                total_examples += y_batch.size(0)

        # Average loss across batches
        avg_loss = total_loss / (len(valid_loader))
        avg_acc = total_correct / total_examples if total_examples > 0 else 0.0

        # Reduce loss across all processes
        avg_loss_tensor = torch.tensor([avg_loss, avg_acc], device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor[0].item() / self.world_size
        avg_acc = avg_loss_tensor[1].item() / self.world_size

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
