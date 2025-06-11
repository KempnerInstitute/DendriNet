"""
Specialized Training Strategies
===============================

This module contains specialized training strategies for dendritic networks,
including voltage stabilization and M-reactivation specific approaches.
"""

import os
import time
from copy import deepcopy
from typing import Optional
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

import wandb
from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves
from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.training.strategies.base import BaseTrainer, compute_metrics_classification


def voltage_warmup_loss(model, x, target_val=0.5):
    """
    Compute voltage warm-up loss to stabilize dendritic voltages.
    
    Args:
        model: The dendritic model
        x: Input data
        target_val: Target voltage value
        
    Returns:
        Loss value for voltage stabilization
    """
    voltages = []
    if hasattr(model, "net") and hasattr(model.net, "layers"):
        for layer in model.net.layers:
            if hasattr(layer, "voltage"):
                voltages.append(layer.voltage.mean())
    
    if len(voltages) == 0:
        return torch.tensor(0.0, device=x.device)
    
    voltage_tensor = torch.stack(voltages)
    target_tensor = torch.full_like(voltage_tensor, target_val)
    return F.mse_loss(voltage_tensor, target_tensor)


class VoltageStabilizationTrainer(BaseTrainer):
    """
    Voltage Stabilization Training Strategy.
    
    This trainer includes a warm-up phase that stabilizes dendritic voltages
    to target values before proceeding with normal training. This can help
    with training stability in dendritic networks.
    """
    
    def __init__(
        self,
        optimizer,
        suppress_prints=False,
        print_every=10,
        warmup_epochs=1,
        stability_target_voltage=0.5,
    ):
        """
        Initialize the voltage stabilization trainer.
        
        Args:
            optimizer: The optimizer to use
            suppress_prints: Whether to suppress print statements
            print_every: Print frequency for training logs
            warmup_epochs: Number of epochs for voltage warm-up phase
            stability_target_voltage: Target voltage value for stabilization
        """
        super().__init__(optimizer, suppress_prints, print_every)
        self.warmup_epochs = warmup_epochs
        self.stability_target_voltage = stability_target_voltage
    
    def train(
        self,
        model,
        train_data,
        valid_data=None,
        grad_clip_value=5,
        epochs=100,
        batch_size=256,
        shuffle=True,
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        """
        Train the model with voltage stabilization warm-up.
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            grad_clip_value: Gradient clipping value
            epochs: Total number of training epochs
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            analysis_manager: Optional analysis manager for periodic analysis
            plot_losses: Whether to plot loss curves
            save_path: Path to save results
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing training results and metrics
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        
        all_train_losses = []
        all_valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        start_time = time.time()
        
        # Phase 1: Voltage stabilization warm-up
        if self.warmup_epochs > 0:
            logger.info(
                f"=== VoltageStabilization: Starting warm-up phase for {self.warmup_epochs} epochs ==="
            )
            
            for epoch in range(1, self.warmup_epochs + 1):
                model.train()
                epoch_warmup_loss = 0.0
                
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    
                    # Voltage stabilization loss
                    warmup_loss = voltage_warmup_loss(model, xb, self.stability_target_voltage)
                    
                    self.optimizer.zero_grad()
                    warmup_loss.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    
                    epoch_warmup_loss += warmup_loss.item()
                    
                avg_warmup_loss = epoch_warmup_loss / len(train_loader)
                all_train_losses.append(avg_warmup_loss)
                
                # Validation
                val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
                all_valid_losses.append(val_loss)
                
                if (not self.suppress) and (epoch % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Voltage-Warmup] epoch={epoch}/{self.warmup_epochs}, "
                        f"warmup_loss={avg_warmup_loss:.4f}, valid_loss={val_loss:.4f}, "
                        f"valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )
                    
                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "phase": "voltage_warmup",
                            "warmup_loss": avg_warmup_loss,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )
        
        # Phase 2: Normal training
        normal_epochs = epochs - self.warmup_epochs
        if normal_epochs > 0:
            logger.info(
                f"=== VoltageStabilization: Starting normal training for {normal_epochs} epochs ==="
            )
            
            for epoch in range(1, normal_epochs + 1):
                actual_epoch = self.warmup_epochs + epoch
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []
                
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    
                    # Normal training loss
                    loss_val = model.compute_loss(xb, yb)
                    
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    
                    epoch_train_loss += loss_val.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())
                    
                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)["accuracy"]
                all_train_losses.append(epoch_train_loss)
                
                # Validation
                val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
                all_valid_losses.append(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())
                    
                if analysis_manager is not None:
                    analysis_manager.run_analysis(filename=f'epoch{actual_epoch}', training=True)
                
                if (not self.suppress) and (actual_epoch % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Voltage-Normal] epoch={actual_epoch}/{epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )
                    
                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": actual_epoch,
                            "phase": "normal_training",
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )
        
        # Load best state
        if best_state is not None:
            model.load_state_dict(best_state)
            
        results = {
            "best loss": best_loss,
            "train losses": all_train_losses,
            "valid losses": all_valid_losses,
        }
        
        if plot_losses and len(all_train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, epochs, save_path)
            else:
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, epochs)
                
        return results


class TrainOnlyMReactivation(BaseTrainer):
    """
    Training strategy that focuses only on M-reactivation parameters.
    
    This trainer freezes log_b parameters and trains all other parameters,
    which can be useful for specific dendritic network configurations.
    """
    
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
        """
        Train the model with M-reactivation focus.
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            grad_clip_value: Gradient clipping value
            epochs: Number of training epochs
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            load_best_state_dict: Whether to load best model state at end
            analysis_manager: Optional analysis manager for periodic analysis
            plot_losses: Whether to plot loss curves
            save_path: Path to save results
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing training results and metrics
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        logger.info(
            "TrainOnlyMReactivation: Freezing log_b and training all other parameters."
        )
        
        # Freeze log_b parameters, unfreeze all others
        for name, param in model.named_parameters():
            if "log_b" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
                
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_epoch = 1
        best_state = deepcopy(model.state_dict())
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            preds_list = []
            labels_list = []
            
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                loss_val = model.compute_loss(xb, yb)
                self.optimizer.zero_grad()
                loss_val.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += loss_val.item()
                
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
                
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)["accuracy"]
            train_losses.append(epoch_train_loss)
            
            # Validation
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TrainOnlyMReactivation] epoch={epoch}/{epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                    f"elapsed={elapsed:.1f}s"
                )
                
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{epoch}', training=True)
                
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss": epoch_train_loss,
                        "train_acc": train_acc,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )
                
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
