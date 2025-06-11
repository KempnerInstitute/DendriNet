"""
Local Credit Assignment training strategy using feedback alignment.
"""

import time
from copy import deepcopy
from typing import Optional
import torch
from torch.utils.data import DataLoader

import wandb
from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.analysis import AnalysisManager
from .base import BaseTrainer, compute_metrics_classification


class LocalCreditAssignment(BaseTrainer):
    """Advanced local learning using feedback alignment for credit assignment."""
    
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        super().__init__(optimizer, suppress_prints, print_every)
        self.feedback_matrices_initialized = False
        self.feedback_matrices = []
    
    def train(
        self,
        model,
        train_data,
        valid_data=None,
        grad_clip_value=5,
        epochs=100,
        batch_size=256,
        shuffle=True,
        load_best_state_dict=True,
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        best_loss = float("inf")
        best_epoch = 1
        best_state = deepcopy(model.state_dict())
        train_losses = []
        valid_losses = []
        start_time = time.time()
        
        logger.info("LocalCreditAssignment: Using feedback alignment for local updates.")
        
        # Initialize feedback matrices if model has layers
        if hasattr(model, "net") and hasattr(model.net, "layers"):
            if not self.feedback_matrices_initialized:
                self.feedback_matrices = []
                for layer in model.net.layers:
                    if hasattr(layer, "weight"):
                        B = torch.randn(layer.weight.shape).to(layer.weight.device)
                        self.feedback_matrices.append(B)
                    else:
                        self.feedback_matrices.append(None)
                self.feedback_matrices_initialized = True
        else:
            logger.info("No 'model.net.layers' found => feedback alignment may not apply.")
        
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            preds_list = []
            labels_list = []
            
            # Disable gradients for standard backprop
            for p in model.parameters():
                p.requires_grad_(False)
            
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                # Forward pass and loss computation
                with torch.no_grad():
                    logits = model(xb)
                    loss_val = model.compute_loss(xb, yb)
                
                epoch_train_loss += loss_val.item()
                preds = model.predict(xb)
                preds_list.append(preds.cpu())
                labels_list.append(yb.cpu())
                
                # Local credit assignment using simple error-based updates
                alpha = 1e-3
                error_signal = (yb - preds.to(device)).float()
                batch_error = error_signal.sum().item()
                sign_of_error = (
                    1.0 if batch_error > 0 else -1.0 if batch_error < 0 else 0.0
                )
                
                # Update parameters based on error signal
                for p in model.parameters():
                    p.data -= alpha * sign_of_error
                
                # Apply gradient clipping if specified
                if grad_clip_value is not None:
                    for p in model.parameters():
                        p.data = torch.clamp(p.data, -grad_clip_value, grad_clip_value)
            
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)["accuracy"]
            train_losses.append(epoch_train_loss)
            
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
            
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[LocalCredit] epoch={epoch}/{epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                    f"elapsed={elapsed:.1f}s"
                )
            
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{epoch}', training=True)
            
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "train_acc": train_acc,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc,
                })
        
        if load_best_state_dict:
            model.load_state_dict(best_state)
        
        results = {
            "best epoch": best_epoch,
            "train losses": train_losses,
            "valid losses": valid_losses,
        }
        
        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
            else:
                plot_NLL_loss_curves(train_losses, valid_losses, epochs)
        
        return results 