"""
Base training strategy classes.

This module contains the base trainer class that provides common functionality
for all training strategies in the dendritic modeling framework.
"""

import os
import time
from copy import deepcopy
from typing import Optional

import torch
import torch.utils.data
import wandb
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

from dendritic_modeling import logger
from dendritic_modeling.models import BaseModel
from dendritic_modeling.plotting import plot_loss_curves
from dendritic_modeling.training.loss.functions import LossFunction, get_loss_function


class Trainer:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss_function: str,
        epochs: int = 100,
        early_stopping: bool = False,
        patience: int = 10,
        batch_size: int = 256,
        shuffle: bool = True,
        grad_clip_value=5,
        load_best_state_dict: bool = True,
        plot_losses: bool = False,
        save_path: Optional[str] = None,
        suppress_prints=False,
        print_every=10,
        analysis_manager: Optional[object] = None,
        save_every_n_epochs: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
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
        self.load_best_state_dict = load_best_state_dict
        self.plot_losses = plot_losses
        self.save_path = save_path
        self.suppress_prints = suppress_prints
        self.print_every = print_every
        self.analysis_manager = analysis_manager
        
        # Mixed Precision Training Support
        self.use_amp = kwargs.get('use_amp', False)
        if self.use_amp:
            try:
                from torch.cuda.amp import autocast, GradScaler
                self.scaler = GradScaler()
                logger.info("Mixed precision training enabled")
            except ImportError:
                logger.warning("AMP not available, falling back to standard precision")
                self.use_amp = False

        # Profiling Support
        self.enable_profiling = kwargs.get('enable_profiling', False)
        self.profiler = None
        
        if self.enable_profiling:
            try:
                import torch.profiler
                profiling_output_dir = kwargs.get('profiling_output_dir', 'outputs/profiling/')
                os.makedirs(profiling_output_dir, exist_ok=True)
                
                self.profiler = torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                    schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(profiling_output_dir),
                    record_shapes=True,
                    with_stack=True
                )
                logger.info(f"Profiling enabled - output directory: {profiling_output_dir}")
            except ImportError:
                logger.warning("torch.profiler not available, profiling disabled")
                self.enable_profiling = False

        # Checkpoint configuration
        self.save_every_n_epochs = save_every_n_epochs

        # Set up checkpoint directory structure
        if (
            checkpoint_dir is not None
            and save_every_n_epochs is not None
            and save_every_n_epochs > 0
        ):
            self.checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        else:
            self.checkpoint_dir = None

        self.filename_prefix = ""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def _dataloaders(
        self, train_data: torch.utils.data.Dataset, valid_data: torch.utils.data.Dataset
    ) -> tuple[DataLoader, DataLoader]:
        # Optimized DataLoader settings for better performance
        train_loader = DataLoader(
            train_data, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster GPU transfer
            persistent_workers=True,  # Keep workers alive between epochs
        )
        valid_loader = DataLoader(
            valid_data, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=2,  # Fewer workers for validation
            pin_memory=True,
            persistent_workers=True,
        )
        return train_loader, valid_loader

    def _initialize_attributes(self, model: BaseModel, total_epochs: int) -> None:
        if not hasattr(self, "device"):
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(self.device)
        if not hasattr(self, "best_state_dict"):
            self.best_state_dict = deepcopy(model.state_dict())
        if not hasattr(self, "best_loss"):
            self.best_loss = float("inf")
        if not hasattr(self, "best_epoch"):
            self.best_epoch = 1
        if not hasattr(self, "epoch_counter"):
            self.epoch_counter = 0
        if not hasattr(self, "start_time"):
            self.start_time = time.time()
        if not hasattr(self, "total_epochs"):
            self.total_epochs = total_epochs
        self.patience_counter = 0

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict:
        train_loader, valid_loader = self._dataloaders(train_data, valid_data)

        self.epoch_counter = 0
        train_losses = []
        valid_losses = []
        train_accs = []  # Add accuracy tracking
        valid_accs = []  # Add accuracy tracking

        self._initialize_attributes(model, self.epochs)

        # Start profiler if enabled
        if self.profiler is not None:
            self.profiler.start()

        train_losses, valid_losses, train_accs, valid_accs = self._run_training_loop(
            n_epochs=self.epochs,
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            train_losses=train_losses,
            valid_losses=valid_losses,
            train_accs=train_accs,
            valid_accs=valid_accs,
            logger_info="[Standard] ",
            wandb_info={},
        )

        # Stop profiler if enabled
        if self.profiler is not None:
            self.profiler.stop()
            logger.info("✅ Profiling complete. View in TensorBoard: tensorboard --logdir=outputs/profiling/")

        if self.best_state_dict:
            model.load_state_dict(self.best_state_dict)

        logger.info(
            f"The best validation loss was {self.best_loss} at epoch {self.best_epoch}"
        )

        results = {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_state_dict": self.best_state_dict,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_accs": train_accs,  # Add to results
            "valid_accs": valid_accs,  # Add to results
        }

        self._loss_plotting(train_losses, valid_losses)
        self._accuracy_plotting(train_accs, valid_accs)  # Add accuracy plotting

        return results

    def _run_training_loop(
        self,
        n_epochs: int,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: list[float],
        valid_losses: list[float],
        train_accs: list[float],  # Added train_accs
        valid_accs: list[float],  # Added valid_accs
        logger_info: str = "",
        wandb_info: Optional[dict] = None,
    ) -> tuple[
        list[float], list[float], list[float], list[float]
    ]:  # Added train_accs, valid_accs
        if wandb_info is None:
            wandb_info = {}
        for _ in range(1, n_epochs + 1):
            train_losses, valid_losses, train_accs, valid_accs = (
                self._run_epoch(  # Updated return
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    train_accs=train_accs,  # Passed train_accs
                    valid_accs=valid_accs,  # Passed valid_accs
                    logger_info=logger_info,
                    wandb_info=wandb_info,
                )
            )

            if self._check_early_stopping():
                break

        return (
            train_losses,
            valid_losses,
            train_accs,
            valid_accs,
        )  # Added train_accs, valid_accs

    def _run_epoch(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: list[float],
        valid_losses: list[float],
        train_accs: list[float],
        valid_accs: list[float],
        logger_info: str = "",
        wandb_info: Optional[dict] = None,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        if wandb_info is None:
            wandb_info = {}
        self.epoch_counter += 1

        train_loss, train_acc = self._epoch_train(model, train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        valid_loss, valid_acc = self._epoch_valid(model, valid_loader)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        self._update_best(model, valid_loss)
        self._analyze()
        self._save_checkpoint(model)
        self._tracking(train_loss, valid_loss, logger_info, wandb_info)
        return train_losses, valid_losses, train_accs, valid_accs

    def _epoch_train(
        self, model: BaseModel, train_loader: DataLoader
    ) -> tuple[float, float]:
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
            
            if self.use_amp:
                # Mixed precision forward pass
                from torch.cuda.amp import autocast
                with autocast():
                    loss = self.loss_function(model, x_batch, y_batch)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                if self.grad_clip_value:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    clip_grad_value_(model.parameters(), self.grad_clip_value)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                loss = self.loss_function(model, x_batch, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_value_(model.parameters(), self.grad_clip_value)
                self.optimizer.step()

            train_loss += loss.item()

            # Calculate accuracy for classification tasks
            # Only compute accuracy if y_batch contains class labels (not images for autoencoder)
            if (
                hasattr(model, "forward") and y_batch.dim() == 1
            ):  # y_batch is 1D tensor of labels
                with torch.no_grad():
                    if self.use_amp:
                        from torch.cuda.amp import autocast
                        with autocast():
                            outputs = model(x_batch)
                    else:
                        outputs = model(x_batch)
                    if (
                        outputs.dim() > 1 and outputs.size(1) > 1
                    ):  # Multi-class classification
                        predicted = outputs.argmax(dim=1)
                        correct += (predicted == y_batch).sum().item()
                        total += y_batch.size(0)

            # Step profiler if enabled
            if self.profiler is not None:
                self.profiler.step()

        avg_loss = train_loss / len(train_loader)
        accuracy = (correct / total * 100) if total > 0 else 0.0
        return avg_loss, accuracy

    def _epoch_valid(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> tuple[float, float]:
        model.eval()
        valid_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                loss = self.loss_function(model, x_batch, y_batch)
                valid_loss += loss.item()

                # Calculate accuracy for classification tasks
                # Only compute accuracy if y_batch contains class labels (not images for autoencoder)
                if y_batch.dim() == 1:  # y_batch is 1D tensor of labels
                    outputs = model(x_batch)
                    if (
                        outputs.dim() > 1 and outputs.size(1) > 1
                    ):  # Multi-class classification
                        predicted = outputs.argmax(dim=1)
                        correct += (predicted == y_batch).sum().item()
                        total += y_batch.size(0)

        avg_loss = valid_loss / len(valid_loader)
        accuracy = (correct / total * 100) if total > 0 else 0.0
        return avg_loss, accuracy

    def _update_best(self, model: BaseModel, valid_loss: float) -> None:
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_epoch = self.epoch_counter
            self.best_state_dict = deepcopy(model.state_dict())
            self.patience_counter = 0
            if self.save_path and os.path.exists(self.save_path):
                torch.save(
                    self.best_state_dict,
                    os.path.join(
                        self.save_path, f"{self.filename_prefix}best_model.pt"
                    ),
                )
        else:
            self.patience_counter += 1

    def _analyze(self):
        if self.analysis_manager is not None:
            # Conditional import to avoid circular dependency
            from dendritic_modeling.analysis import AnalysisManager

            if isinstance(self.analysis_manager, AnalysisManager):
                self.analysis_manager.run_analysis(
                    filename=f"epoch{self.epoch_counter}", training=True
                )

    def _save_checkpoint(self, model: BaseModel) -> None:
        """Save checkpoint if configured to do so."""
        if (
            self.checkpoint_dir is not None
            and self.save_every_n_epochs is not None
            and self.save_every_n_epochs > 0
            and self.epoch_counter % self.save_every_n_epochs == 0
        ):

            checkpoint = {
                "epoch": self.epoch_counter,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "best_epoch": self.best_epoch,
                "loss_function": self.loss_function._loss_name,
            }

            checkpoint_path = os.path.join(
                self.checkpoint_dir,
                f"{self.filename_prefix}checkpoint_epoch_{self.epoch_counter}.pt",
            )
            torch.save(checkpoint, checkpoint_path)

            if not self.suppress_prints:
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _tracking(
        self,
        train_loss: float,
        valid_loss: float,
        logger_info: str = "",
        wandb_info: Optional[dict] = None,
    ) -> None:
        if wandb_info is None:
            wandb_info = {}
        elapsed = time.time() - self.start_time
        if not self.suppress_prints and (self.epoch_counter % self.print_every == 0):
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
            wandb.log(wandb_info)

    def _check_early_stopping(self) -> bool:
        if self.early_stopping and self.patience_counter >= self.patience:
            if not self.suppress_prints:
                logger.info(
                    f"Early stopping triggered after {self.epoch_counter} epochs."
                )
            return True
        else:
            return False

    def _loss_plotting(
        self, train_losses: list[float], valid_losses: list[float]
    ) -> None:
        if self.plot_losses and len(train_losses) > 1:
            if self.save_path and os.path.exists(self.save_path):
                plot_loss_curves(
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    loss_name=self.loss_function._loss_name,
                    save_dir=self.save_path,
                    filename_prefix=self.filename_prefix,
                )

    def _accuracy_plotting(
        self, train_accs: list[float], valid_accs: list[float]
    ) -> None:
        """Plot accuracy curves similar to loss curves."""
        if self.plot_losses and len(train_accs) > 1:
            if self.save_path and os.path.exists(self.save_path):
                try:
                    from dendritic_modeling.plotting.training import plot_loss_curves

                    # Reuse the loss plotting function but with accuracy label
                    plot_loss_curves(
                        train_losses=train_accs,
                        valid_losses=valid_accs,
                        loss_name="Accuracy (%)",
                        save_dir=self.save_path,
                        filename_prefix=self.filename_prefix + "accuracy_",
                    )
                except Exception as e:
                    logger.warning(f"Could not plot accuracy curves: {e}")


__all__ = ["Trainer"]
