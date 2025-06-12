"""
Two-Step Training Strategies
============================

This module contains training strategies that implement two-step learning approaches,
where models are first pre-trained with local supervision and then fine-tuned end-to-end.
"""

from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Optional, TYPE_CHECKING
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader

import wandb
from dendritic_modeling import logger
from dendritic_modeling.plotting import plot_NLL_loss_curves

if TYPE_CHECKING:
    from dendritic_modeling.analysis import AnalysisManager

from dendritic_modeling.training.strategies.base import BaseTrainer, compute_metrics_classification
from dendritic_modeling.training.strategies.mle import TrainerMLE


class TwoStepTrainer(BaseTrainer):
    """
    Two-Step Training Strategy for dendritic networks.
    
    This trainer implements a two-phase training approach:
    1. Pre-training phase: Each layer is trained locally with local supervision heads
    2. Main training phase: End-to-end training of the full network
    
    This approach can help with gradient flow and learning in deep dendritic networks.
    """
    
    def train(
        self,
        model,
        train_data,
        valid_data=None,
        pretrain_epochs=5,
        maintrain_epochs=45,
        batch_size=256,
        grad_clip_value=5,
        shuffle=True,
        load_best_state_dict=True,
        analysis_manager: Optional["AnalysisManager"] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        """
        Train the model using two-step approach.
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            pretrain_epochs: Number of epochs for local pre-training phase
            maintrain_epochs: Number of epochs for end-to-end training phase
            batch_size: Batch size for training
            grad_clip_value: Gradient clipping value
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
        
        # Get network object
        if hasattr(model, "net"):
            net_obj = model.net
        elif hasattr(model, "network"):
            net_obj = model.network
        else:
            net_obj = model
            
        # Check if model has layers for local training
        if hasattr(net_obj, "layers"):
            if not hasattr(self, "local_heads"):
                self._initialize_local_heads(net_obj, model, device)
        else:
            logger.info("No net.layers found => falling back to MLE training.")
            trainer_mle = TrainerMLE(self.optimizer, self.suppress, self.print_every)
            return trainer_mle.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=pretrain_epochs + maintrain_epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                load_best_state_dict=load_best_state_dict,
                analysis_manager=analysis_manager,
                plot_losses=plot_losses,
                save_path=save_path,
            )
            
        # Training setup
        all_train_losses = []
        all_valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_ep = pretrain_epochs + maintrain_epochs
        ep_counter = 0
        start_time = time.time()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        
        # Phase 1: Local pretraining
        logger.info(
            f"=== TwoStepTrainer: Stage 1 (local pretraining) for {pretrain_epochs} epochs ==="
        )
        n_layers = len(net_obj.layers)
        
        for epoch in range(1, pretrain_epochs + 1):
            ep_counter += 1
            epoch_loss_accum = 0.0
            ct_batches = 0
            
            # Train each layer with its local head
            for layer_idx in range(n_layers):
                # Freeze all parameters
                for name, param in model.named_parameters():
                    param.requires_grad = False
                    
                # Unfreeze current layer
                if hasattr(net_obj.layers[layer_idx], "weight"):
                    net_obj.layers[layer_idx].weight.requires_grad = True
                if (
                    hasattr(net_obj.layers[layer_idx], "bias")
                    and net_obj.layers[layer_idx].bias is not None
                ):
                    net_obj.layers[layer_idx].bias.requires_grad = True
                    
                # Unfreeze local head
                local_head = self.local_heads[layer_idx]
                for param in local_head.parameters():
                    param.requires_grad = True
                    
                # Train on batches
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    
                    # Forward pass through layers up to current layer
                    h = xb
                    for j in range(layer_idx + 1):
                        h = net_obj.layers[j](h)
                        if isinstance(h, tuple):
                            h = h[0]
                        if hasattr(net_obj.layers[j], "activation"):
                            h = net_obj.layers[j].activation(h)
                            
                    # Local supervision
                    loss_val = F.cross_entropy(local_head(h), yb)
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    
                    epoch_loss_accum += loss_val.item()
                    ct_batches += 1
                    
            avg_loss = epoch_loss_accum / ct_batches if ct_batches > 0 else 0.0
            all_train_losses.append(avg_loss)
            
            # Validation
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            all_valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
                
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStep-Pretrain] epoch={ep_counter}/{total_ep}, train_loss={avg_loss:.4f}, "
                    f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                )
                
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{ep_counter}', training=True)
                
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": ep_counter,
                        "phase": "pretrain",
                        "train_loss": avg_loss,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )
                
        # Phase 2: End-to-end training
        logger.info(
            f"=== TwoStepTrainer: Stage 2 (end-to-end training) for {maintrain_epochs} epochs ==="
        )
        
        # Unfreeze all parameters
        for name, param in model.named_parameters():
            param.requires_grad = True
            
        for epoch in range(1, maintrain_epochs + 1):
            ep_counter += 1
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
            all_train_losses.append(epoch_train_loss)
            
            # Validation
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            all_valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
                
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStep-MainTrain] epoch={ep_counter}/{total_ep}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                    f"elapsed={elapsed:.1f}s"
                )
                
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{ep_counter}', training=True)
                
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": ep_counter,
                        "phase": "main_train",
                        "train_loss": epoch_train_loss,
                        "train_acc": train_acc,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )
                
        # Load best state
        if load_best_state_dict:
            model.load_state_dict(best_state)
            
        results = {
            "best epoch": ep_counter,
            "train losses": all_train_losses,
            "valid losses": all_valid_losses,
        }
        
        if plot_losses and len(all_train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, total_ep, save_path)
            else:
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, total_ep)
                
        return results
        
    def _initialize_local_heads(self, net_obj, model, device):
        """Initialize local supervision heads for each layer."""
        self.local_heads = []
        
        for i, layer in enumerate(net_obj.layers):
            # Determine output dimension of current layer
            out_dim = getattr(layer, "out_features", None)
            if out_dim is None:
                excit_sizes = getattr(net_obj, "excitatory_layer_sizes", None)
                if isinstance(excit_sizes, list) and i < len(excit_sizes):
                    out_dim = excit_sizes[i]
                else:
                    # Infer from dummy forward pass
                    try:
                        dummy = torch.randn(1, getattr(net_obj, "input_dim", 784)).to(device)
                    except Exception:
                        dummy = torch.randn(1, 784).to(device)
                    dummy_out = layer(dummy)
                    if isinstance(dummy_out, tuple):
                        dummy_out = dummy_out[0]
                    out_dim = dummy_out.size(1)
                    
            # Create local head
            if hasattr(layer, "weight"):
                device_for_head = layer.weight.device
            else:
                device_for_head = next(model.parameters()).device
                
            head = torch.nn.Linear(out_dim, getattr(model, "output_dim", 10)).to(device_for_head)
            self.local_heads.append(head)


class TwoStepTrainerWithKL(BaseTrainer):
    """
    Two-Step Training Strategy with KL Divergence Loss.
    
    Similar to TwoStepTrainer but uses a combined loss function during pre-training
    that includes both cross-entropy and KL divergence terms for better regularization.
    """
    
    def train(
        self,
        model,
        train_data,
        valid_data=None,
        pretrain_epochs=5,
        maintrain_epochs=45,
        batch_size=256,
        grad_clip_value=5,
        shuffle=True,
        load_best_state_dict=True,
        analysis_manager: Optional["AnalysisManager"] = None,
        plot_losses=False,
        save_path=None,
        kl_weight=0.1,
        **kwargs,
    ):
        """
        Train the model using two-step approach with KL divergence.
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)
            pretrain_epochs: Number of epochs for local pre-training phase
            maintrain_epochs: Number of epochs for end-to-end training phase
            batch_size: Batch size for training
            grad_clip_value: Gradient clipping value
            shuffle: Whether to shuffle training data
            load_best_state_dict: Whether to load best model state at end
            analysis_manager: Optional analysis manager for periodic analysis
            plot_losses: Whether to plot loss curves
            save_path: Path to save results
            kl_weight: Weight for KL divergence term in combined loss
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing training results and metrics
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Get network object
        if hasattr(model, "net"):
            net_obj = model.net
        elif hasattr(model, "network"):
            net_obj = model.network
        else:
            net_obj = model
            
        # Check if model has layers for local training
        if hasattr(net_obj, "layers"):
            if not hasattr(self, "local_heads"):
                self._initialize_local_heads(net_obj, model, device)
        else:
            logger.info("No net.layers found => falling back to MLE training.")
            trainer_mle = TrainerMLE(self.optimizer, self.suppress, self.print_every)
            return trainer_mle.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=pretrain_epochs + maintrain_epochs,
                batch_size=batch_size,
                shuffle=shuffle,
                load_best_state_dict=load_best_state_dict,
                analysis_manager=analysis_manager,
                plot_losses=plot_losses,
                save_path=save_path,
            )
            
        # Training setup
        all_train_losses = []
        all_valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        total_ep = pretrain_epochs + maintrain_epochs
        ep_counter = 0
        start_time = time.time()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        
        # Phase 1: Local pretraining with KL divergence
        logger.info(
            f"=== TwoStepTrainerWithKL: Stage 1 (local pretraining with KL) for {pretrain_epochs} epochs ==="
        )
        n_layers = len(net_obj.layers)
        
        for epoch in range(1, pretrain_epochs + 1):
            ep_counter += 1
            epoch_loss_accum = 0.0
            ct_batches = 0
            
            # Train each layer with its local head
            for layer_idx in range(n_layers):
                # Freeze all parameters
                for name, param in model.named_parameters():
                    param.requires_grad = False
                    
                # Unfreeze current layer
                if hasattr(net_obj.layers[layer_idx], "weight"):
                    net_obj.layers[layer_idx].weight.requires_grad = True
                if (
                    hasattr(net_obj.layers[layer_idx], "bias")
                    and net_obj.layers[layer_idx].bias is not None
                ):
                    net_obj.layers[layer_idx].bias.requires_grad = True
                    
                # Unfreeze local head
                local_head = self.local_heads[layer_idx]
                for param in local_head.parameters():
                    param.requires_grad = True
                    
                # Train on batches
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    
                    # Forward pass through layers up to current layer
                    h = xb
                    for j in range(layer_idx + 1):
                        h = net_obj.layers[j](h)
                        if isinstance(h, tuple):
                            h = h[0]
                        if hasattr(net_obj.layers[j], "activation"):
                            h = net_obj.layers[j].activation(h)
                            
                    # Local head predictions
                    local_logits = local_head(h)
                    
                    # Get full model predictions for KL divergence
                    with torch.no_grad():
                        full_logits = model(xb)
                        
                    # Combined loss: cross-entropy + KL divergence
                    ce_loss = F.cross_entropy(local_logits, yb)
                    kl_loss = F.kl_div(
                        F.log_softmax(local_logits, dim=1),
                        F.softmax(full_logits, dim=1),
                        reduction='batchmean'
                    )
                    loss_val = ce_loss + kl_weight * kl_loss
                    
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    
                    epoch_loss_accum += loss_val.item()
                    ct_batches += 1
                    
            avg_loss = epoch_loss_accum / ct_batches if ct_batches > 0 else 0.0
            all_train_losses.append(avg_loss)
            
            # Validation
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            all_valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
                
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStepKL-Pretrain] epoch={ep_counter}/{total_ep}, train_loss={avg_loss:.4f}, "
                    f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                )
                
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{ep_counter}', training=True)
                
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": ep_counter,
                        "phase": "pretrain_kl",
                        "train_loss": avg_loss,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )
                
        # Phase 2: End-to-end training (same as TwoStepTrainer)
        logger.info(
            f"=== TwoStepTrainerWithKL: Stage 2 (end-to-end training) for {maintrain_epochs} epochs ==="
        )
        
        # Unfreeze all parameters
        for name, param in model.named_parameters():
            param.requires_grad = True
            
        for epoch in range(1, maintrain_epochs + 1):
            ep_counter += 1
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
            all_train_losses.append(epoch_train_loss)
            
            # Validation
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            all_valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
                
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStepKL-MainTrain] epoch={ep_counter}/{total_ep}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                    f"elapsed={elapsed:.1f}s"
                )
                
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'epoch{ep_counter}', training=True)
                
            if wandb.run is not None:
                wandb.log(
                    {
                        "epoch": ep_counter,
                        "phase": "main_train_kl",
                        "train_loss": epoch_train_loss,
                        "train_acc": train_acc,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc,
                    }
                )
                
        # Load best state
        if load_best_state_dict:
            model.load_state_dict(best_state)
            
        results = {
            "best epoch": ep_counter,
            "train losses": all_train_losses,
            "valid losses": all_valid_losses,
        }
        
        if plot_losses and len(all_train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, total_ep, save_path)
            else:
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, total_ep)
                
        return results
        
    def _initialize_local_heads(self, net_obj, model, device):
        """Initialize local supervision heads for each layer."""
        self.local_heads = []
        
        for i, layer in enumerate(net_obj.layers):
            # Determine output dimension of current layer
            out_dim = getattr(layer, "out_features", None)
            if out_dim is None:
                excit_sizes = getattr(net_obj, "excitatory_layer_sizes", None)
                if isinstance(excit_sizes, list) and i < len(excit_sizes):
                    out_dim = excit_sizes[i]
                else:
                    # Infer from dummy forward pass
                    try:
                        dummy = torch.randn(1, getattr(net_obj, "input_dim", 784)).to(device)
                    except Exception:
                        dummy = torch.randn(1, 784).to(device)
                    dummy_out = layer(dummy)
                    if isinstance(dummy_out, tuple):
                        dummy_out = dummy_out[0]
                    out_dim = dummy_out.size(1)
                    
            # Create local head
            if hasattr(layer, "weight"):
                device_for_head = layer.weight.device
            else:
                device_for_head = next(model.parameters()).device
                
            head = torch.nn.Linear(out_dim, getattr(model, "output_dim", 10)).to(device_for_head)
            self.local_heads.append(head) 