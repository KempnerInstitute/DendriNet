"""
Specialized training strategies.

This module implements specialized training approaches including voltage stabilization
and other advanced training techniques for dendritic networks.
"""

from __future__ import annotations

import json
import logging
import os
import time
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import wandb
from dendritic_modeling import logger
from dendritic_modeling.plotting import plot_NLL_loss_curves

if TYPE_CHECKING:
    from dendritic_modeling.analysis import AnalysisManager

from .base import BaseTrainer, compute_metrics_classification


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


def generate_warmup_report(model, data_loader, pre_state, post_state, device, save_path):
    """
    Generate a report comparing model statistics before and after voltage warm-up.
    
    Args:
        model: The dendritic model
        data_loader: DataLoader for evaluation
        pre_state: Model state before warm-up
        post_state: Model state after warm-up
        device: Device to run on
        save_path: Path to save the report
    """
    def collect_model_stats(state_dict):
        """Collect statistics from model state dictionary."""
        stats = {}
        for name, param in state_dict.items():
            if "voltage" in name.lower():
                stats[f"{name}_mean"] = param.mean().item()
                stats[f"{name}_std"] = param.std().item()
        return stats
    
    pre_stats = collect_model_stats(pre_state)
    post_stats = collect_model_stats(post_state)
    
    # Create report
    report = {
        "pre_warmup": pre_stats,
        "post_warmup": post_stats,
        "changes": {}
    }
    
    # Calculate changes
    for key in pre_stats:
        if key in post_stats:
            report["changes"][key] = post_stats[key] - pre_stats[key]
    
    if save_path and os.path.exists(save_path):
        with open(os.path.join(save_path, "warmup_report.json"), 'w') as f:
            json.dump(report, f, indent=2)
    
    return report


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
        analysis_manager: Optional["AnalysisManager"] = None,
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
        analysis_manager: Optional["AnalysisManager"] = None,
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


class InhibitoryHeadTrainer:
    """
    Custom trainer for the inhibitory head with different objective functions.
    
    This specialized trainer supports multiple training objectives including:
    - Classification: Standard classification loss
    - Diversity: Encourages inhibitory head to learn different features from excitatory head
    - Combined: Mix of classification and diversity losses
    """

    def __init__(
        self,
        model,
        optimizer,
        objective="classification",
        excit_head_model=None,
        alpha=0.5,
        suppress_prints=False,
        print_every=10,
    ):
        """
        Initialize the inhibitory head trainer.
        
        Args:
            model: The inhibitory head model to train
            optimizer: The optimizer to use
            objective: Training objective ("classification", "diversity", or "combined")
            excit_head_model: Excitatory head model for diversity loss computation
            alpha: Weight for the diversity loss component
            suppress_prints: Whether to suppress print statements
            print_every: Print frequency for training logs
        """
        self.model = model
        self.optimizer = optimizer
        self.objective = objective
        self.excit_head_model = excit_head_model
        self.alpha = alpha  # Weight for the diversity loss
        self.suppress = suppress_prints
        self.print_every = print_every

    def compute_classification_loss(self, x, y):
        """Standard classification loss"""
        return self.model.compute_loss(x, y)

    def compute_diversity_loss(self, x):
        """
        Diversity loss: encourages inhibitory head to learn features different from excitatory head
        by maximizing the cosine distance between their representations.
        
        Args:
            x: Input data
            
        Returns:
            Diversity loss value (higher when representations are similar)
        """
        if self.excit_head_model is None:
            return torch.tensor(0.0, device=x.device)

        # Get representations from both heads
        with torch.no_grad():
            if hasattr(self.excit_head_model, "net"):
                excit_repr = self.excit_head_model.net(x)
            elif hasattr(self.excit_head_model, "network"):
                excit_repr = self.excit_head_model.network(x)
            else:
                excit_repr = self.excit_head_model(x)

        if hasattr(self.model, "net"):
            inhib_repr = self.model.net(x)
        elif hasattr(self.model, "network"):
            inhib_repr = self.model.network(x)
        else:
            inhib_repr = self.model(x)

        # Normalize the representations
        excit_repr = F.normalize(excit_repr, p=2, dim=1)
        inhib_repr = F.normalize(inhib_repr, p=2, dim=1)

        # Compute cosine similarity (we want to minimize this, or maximize negative)
        similarity = torch.sum(excit_repr * inhib_repr, dim=1).mean()

        # Return negative similarity to maximize distance
        return similarity

    def train(
        self,
        model,
        train_data,
        valid_data=None,
        epochs=100,
        batch_size=256,
        grad_clip_value=5,
        shuffle=True,
        load_best_state_dict=True,
        plot_losses=False,
        save_path=None,
    ):
        """
        Train the inhibitory head with the specified objective.
        
        Args:
            model: The model to train (will use self.model if provided)
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            grad_clip_value: Gradient clipping value
            shuffle: Whether to shuffle training data
            load_best_state_dict: Whether to load best model state at end
            plot_losses: Whether to plot loss curves
            save_path: Path to save results
            
        Returns:
            Dictionary containing training results and metrics
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        if self.excit_head_model is not None:
            self.excit_head_model = self.excit_head_model.to(device)

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=shuffle
        )

        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_train_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                # Compute the appropriate loss based on the objective
                if self.objective == "classification":
                    loss = self.compute_classification_loss(xb, yb)
                elif self.objective == "diversity":
                    class_loss = self.compute_classification_loss(xb, yb)
                    div_loss = self.compute_diversity_loss(xb)
                    loss = class_loss + self.alpha * div_loss
                elif self.objective == "combined":
                    class_loss = self.compute_classification_loss(xb, yb)
                    div_loss = self.compute_diversity_loss(xb)
                    loss = class_loss + self.alpha * div_loss
                else:
                    raise ValueError(f"Unknown objective: {self.objective}")

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()

                epoch_train_loss += loss.item()

            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)

            # Evaluate on validation set
            model.eval()
            val_loss = 0.0

            if valid_data is not None:
                valid_loader = torch.utils.data.DataLoader(
                    valid_data, batch_size=batch_size
                )
                with torch.no_grad():
                    for xb, yb in valid_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)

                        # Use classification loss for validation
                        loss = self.compute_classification_loss(xb, yb)
                        val_loss += loss.item()

                val_loss /= len(valid_loader)
                valid_losses.append(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())
                    if save_path and os.path.exists(save_path):
                        torch.save(best_state, os.path.join(save_path, "best_model.pt"))

            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                if valid_data is not None:
                    logger.info(
                        f"[InhibitoryHead-{self.objective}] epoch={epoch}/{epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, valid_loss={val_loss:.4f}, "
                        f"elapsed={elapsed:.1f}s"
                    )
                else:
                    logger.info(
                        f"[InhibitoryHead-{self.objective}] epoch={epoch}/{epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, elapsed={elapsed:.1f}s"
                    )

            if wandb.run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "objective": self.objective,
                }
                if valid_data is not None:
                    log_dict["valid_loss"] = val_loss
                wandb.log(log_dict)

        if load_best_state_dict and valid_data is not None:
            model.load_state_dict(best_state)

        results = {
            "train_losses": train_losses,
            "valid_losses": valid_losses if valid_data is not None else [],
            "best_loss": best_loss if valid_data is not None else None,
        }

        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(
                    train_losses,
                    valid_losses if valid_data is not None else [],
                    epochs,
                    save_path,
                )

        return results
