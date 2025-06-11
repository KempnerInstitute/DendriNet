"""
Layer-wise training strategies for dendritic models.

This module contains training strategies that train network layers sequentially,
either from soma to input or from input to soma.
"""

import time
import logging
from copy import deepcopy
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_

from .base import BaseTrainer, compute_metrics_classification

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


class LayerWiseTrainer(BaseTrainer):
    """
    Training strategy that trains network layers sequentially.
    
    This trainer provides methods to train layers one at a time, either from
    soma to input or from input to soma, followed by full-network fine-tuning.
    
    Args:
        optimizer: PyTorch optimizer instance
        suppress_prints: Whether to suppress print statements during training
        print_every: Print training progress every N epochs
    """
    
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        super().__init__(optimizer, suppress_prints, print_every)

    def train_freeze_layers_by_layer(
        self,
        model,
        train_data,
        valid_data=None,
        epochs_per_layer=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
    ):
        """
        Train layers sequentially from soma to input (previously named train_freeze_layers).
        
        This method trains each layer of the network separately, starting from the layer
        closest to the soma and moving outward toward the input.
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            epochs_per_layer: Number of epochs to train each layer
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            grad_clip_value: Gradient clipping value
            plot_losses: Whether to plot training curves
            save_path: Path to save model and plots
            final_tune_epochs: Number of epochs for final full-network tuning
            
        Returns:
            Training results dictionary
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        if not hasattr(model, "net") or not hasattr(model.net, "layers"):
            logger.info("Model has no 'net.layers'. Falling back to normal train.")
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=epochs_per_layer,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )
            
        n_layers = len(model.net.layers)
        logger.info(
            f"train_freeze_layers_by_layer: found {n_layers} layers. Training them sequentially."
        )
        
        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = n_layers * epochs_per_layer
        epoch_counter = 0
        start_time = time.time()
        
        for layer_idx in range(n_layers):
            # Freeze all parameters
            for name, param in model.named_parameters():
                param.requires_grad = False
                
            # Unfreeze only parameters for the current layer
            target_str = f"layers.{layer_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True
                    
            logger.info(
                f"Training only layer {layer_idx} for {epochs_per_layer} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )
            
            for e in range(1, epochs_per_layer + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []
                
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())
                    
                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())
                    
                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-layers-by-layer] layer={layer_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )
                    
                if wandb and wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_layer": layer_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )
                    
        logger.info(
            "Done freeze-layers-by-layer. Unfreezing all layers and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)
        
        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model,
            train_data,
            valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )
        return final_results

    def train_freeze_layers_reversed_by_layer(
        self,
        model,
        train_data,
        valid_data=None,
        epochs_per_layer=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
    ):
        """
        Train layers sequentially from input to soma (previously named train_freeze_layers_reversed).
        
        This method trains each layer of the network separately, starting from the layer
        farthest from the soma and moving inward toward the soma.
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            epochs_per_layer: Number of epochs to train each layer
            batch_size: Batch size for training
            shuffle: Whether to shuffle training data
            grad_clip_value: Gradient clipping value
            plot_losses: Whether to plot training curves
            save_path: Path to save model and plots
            final_tune_epochs: Number of epochs for final full-network tuning
            
        Returns:
            Training results dictionary
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        if not hasattr(model, "net") or not hasattr(model.net, "layers"):
            logger.info("Model has no 'net.layers'. Falling back to normal train.")
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=epochs_per_layer,
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path,
            )
            
        n_layers = len(model.net.layers)
        logger.info(
            f"train_freeze_layers_reversed_by_layer: found {n_layers} layers. Training them sequentially from input to soma."
        )
        
        train_losses = []
        valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_epochs = n_layers * epochs_per_layer
        epoch_counter = 0
        start_time = time.time()
        
        # Iterate in reverse order (n_layers-1 to 0), training from input towards soma
        for layer_idx in range(n_layers - 1, -1, -1):
            # Freeze all parameters
            for name, param in model.named_parameters():
                param.requires_grad = False
                
            # Unfreeze only parameters for the current layer
            target_str = f"layers.{layer_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True
                    
            logger.info(
                f"Training only layer {layer_idx} (farther from soma) for {epochs_per_layer} epochs."
            )
            train_loader = DataLoader(
                train_data, batch_size=batch_size, shuffle=shuffle
            )
            
            for e in range(1, epochs_per_layer + 1):
                epoch_counter += 1
                model.train()
                epoch_train_loss = 0.0
                preds_list = []
                labels_list = []
                
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                    p = model.predict(xb)
                    preds_list.append(p.cpu())
                    labels_list.append(yb.cpu())
                    
                epoch_train_loss /= len(train_loader)
                preds_cat = torch.cat(preds_list, dim=0)
                labels_cat = torch.cat(labels_list, dim=0)
                train_acc = compute_metrics_classification(preds_cat, labels_cat)[
                    "accuracy"
                ]
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(
                    model, valid_data, device, batch_size
                )
                valid_losses.append(val_loss)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())
                    
                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(
                        f"[Freeze-layers-reversed-by-layer] layer={layer_idx}, epoch={epoch_counter}/{total_epochs}, "
                        f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                        f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                    )
                    
                if wandb and wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch_counter,
                            "freeze_layer_reversed": layer_idx,
                            "train_loss": epoch_train_loss,
                            "train_acc": train_acc,
                            "valid_loss": val_loss,
                            "valid_acc": val_acc,
                        }
                    )
                    
        logger.info(
            "Done freeze-layers-reversed-by-layer. Unfreezing all layers and loading best state."
        )
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)
        
        logger.info(
            f"Starting final full-network tuning for {final_tune_epochs} epochs."
        )
        final_results = self.train(
            model,
            train_data,
            valid_data,
            grad_clip_value=grad_clip_value,
            epochs=final_tune_epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            load_best_state_dict=True,
            plot_losses=plot_losses,
            save_path=save_path,
        )
        return final_results

    def train(
        self,
        model,
        train_data,
        valid_data=None,
        epochs=100,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        load_best_state_dict=True,
        plot_losses=False,
        save_path=None,
        **kwargs
    ):
        """
        Standard training method for LayerWiseTrainer.
        
        This is a placeholder that should be implemented based on the specific
        training approach desired (e.g., MLE, regression, etc.).
        """
        raise NotImplementedError(
            "LayerWiseTrainer.train() should be implemented based on the specific "
            "training objective (MLE, regression, etc.). Use train_freeze_layers_by_layer() "
            "or train_freeze_layers_reversed_by_layer() for layer-wise training."
        )

    # Backward compatibility aliases
    def train_freeze_layers(self, *args, **kwargs):
        """Alias for train_freeze_layers_by_layer for backward compatibility."""
        return self.train_freeze_layers_by_layer(*args, **kwargs)
        
    def train_freeze_layers_reversed(self, *args, **kwargs):
        """Alias for train_freeze_layers_reversed_by_layer for backward compatibility."""
        return self.train_freeze_layers_reversed_by_layer(*args, **kwargs)


__all__ = ["LayerWiseTrainer"] 