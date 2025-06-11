"""
Regression training strategy.

This module contains the TrainerRegression class which implements training 
strategies specifically designed for regression tasks with dendritic networks.
Features MSE metrics, learning rate scheduling, and early stopping.
"""

import os
import time
from copy import deepcopy
from typing import Optional
import torch
import torch.nn.functional as F
import wandb
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.analysis import AnalysisManager
from .base import BaseTrainer


class TrainerRegression(BaseTrainer):
    """
    Trainer specifically designed for regression tasks.
    
    This trainer extends BaseTrainer with regression-specific functionality including:
    - MSE evaluation metrics for monitoring training progress
    - Optional learning rate scheduling with cosine annealing
    - Support for seed setting for reproducible experiments
    - Enhanced logging with both loss and MSE tracking
    
    Args:
        optimizer: PyTorch optimizer instance
        suppress_prints: Whether to suppress print output during training
        print_every: Print frequency for training progress
        use_scheduler: Whether to use cosine annealing learning rate scheduler
        min_lr: Minimum learning rate for the scheduler
    """
    
    def __init__(self, optimizer, suppress_prints=False, print_every=10, use_scheduler=True, min_lr=1e-6):
        super().__init__(optimizer, suppress_prints, print_every)
        self.use_scheduler = use_scheduler
        self.min_lr = min_lr

    def _eval_dataset(self, model, dataset, device, batch_size=256):
        """
        Evaluate the model on a dataset, returning both loss and MSE.
        
        Args:
            model: The model to evaluate
            dataset: Dataset to evaluate on
            device: Device to run evaluation on
            batch_size: Batch size for evaluation
            
        Returns:
            Tuple of (average_loss, average_mse)
        """
        if dataset is None or len(dataset) == 0:
            return 0.0, 0.0
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        total_mse = 0.0
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                nll = model.compute_loss(xb, yb)
                total_loss += nll.item()
                pred = model.predict(xb)
                mse = F.mse_loss(pred, yb)
                total_mse += mse.item()
        return total_loss / len(loader), total_mse / len(loader)

    def train(
        self, 
        model, 
        train_data, 
        valid_data=None, 
        grad_clip_value=5, 
        epochs=100,
        early_stopping=False, 
        patience=10,
        batch_size=256, 
        shuffle=True, 
        load_best_state_dict=True, 
        best_loss=float('inf'),
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False, 
        save_path=None, 
        seed=None, 
        **kwargs
    ):
        """
        Train the model using regression-specific strategies.
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            grad_clip_value: Gradient clipping value
            epochs: Number of training epochs
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            batch_size: Training batch size
            shuffle: Whether to shuffle training data
            load_best_state_dict: Whether to load best state at the end
            best_loss: Initial best loss value
            analysis_manager: Optional analysis manager for detailed analysis
            plot_losses: Whether to plot loss curves
            save_path: Path to save checkpoints
            seed: Random seed for reproducibility
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary containing training results including best epoch, losses, etc.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if seed is not None:
            model.set_seed(seed)
        model = model.to(device)
        
        # Initialize the scheduler using the wrapped optimizer if needed
        if self.use_scheduler:
            # Get the actual optimizer from the wrapper
            actual_optimizer = getattr(self.optimizer, "optimizer", self.optimizer)
            scheduler = CosineAnnealingLR(actual_optimizer, T_max=epochs, eta_min=self.min_lr)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        train_losses = []
        valid_losses = []
        best_epoch = 1
        best_state = deepcopy(model.state_dict())
        start_time = time.time()

        patience_counter = 0
        epoch_counter = 0
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_counter += 1
            epoch_train_loss = 0.0
            epoch_train_mse = 0.0
            
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                nll = model.compute_loss(xb, yb)
                self.optimizer.zero_grad()
                nll.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
                
                # Calculate MSE for monitoring
                with torch.no_grad():
                    pred = model.predict(xb)
                    mse = F.mse_loss(pred, yb)
                    epoch_train_mse += mse.item()
            
            epoch_train_loss /= len(train_loader)
            epoch_train_mse /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            val_loss, val_mse = self._eval_dataset(model, valid_data, device, batch_size)
            valid_losses.append(val_loss)
            
            # Update scheduler (new)
            if self.use_scheduler:
                current_lr = scheduler.get_last_lr()[0]
                scheduler.step()
            
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                lr_info = f", lr={current_lr:.2e}" if self.use_scheduler else ""
                logger.info(f"[Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.4f}, "
                          f"train_mse={epoch_train_mse:.4f}, valid_loss={val_loss:.4f}, "
                          f"valid_mse={val_mse:.4f}, elapsed={elapsed:.1f}s{lr_info}")
            
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch_counter,
                    "train_loss": epoch_train_loss,
                    "train_mse": epoch_train_mse,
                    "valid_loss": val_loss,
                    "valid_mse": val_mse,
                    "best_valid_mse": best_loss
                })
                if self.use_scheduler:
                    wandb.log({
                        "learning_rate": current_lr
                    })
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch_counter
                best_state = deepcopy(model.state_dict())
                patience_counter = 0
                if save_path and os.path.exists(save_path):
                    torch.save(best_state, os.path.join(save_path, "best_model.pt"))
            else:
                patience_counter += 1
            
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{epoch}', training=True)
                
            if early_stopping and patience_counter >= patience:
                if not self.suppress:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
            
        if load_best_state_dict:
            model.load_state_dict(best_state)
        
        results = {
            "best epoch": best_epoch,
            "best loss": best_loss,
            "best state dict": best_state,
            "train losses": train_losses,
            "valid losses": valid_losses,
        }
        
        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
            else:
                results['loss curves'] = plot_NLL_loss_curves(train_losses, valid_losses, epochs)
        
        return results 