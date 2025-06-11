"""
Two-step training strategies: layerwise pretraining followed by end-to-end training.
"""

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
from .base import BaseTrainer, compute_metrics_classification


class TwoStepTrainer(BaseTrainer):
    """Two-step training: layer-by-layer pretraining then end-to-end training."""
    
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        super().__init__(optimizer, suppress_prints, print_every)
        self.local_heads = []
    
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
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Get network object
        if hasattr(model, "net"):
            net_obj = model.net
        elif hasattr(model, "network"):
            net_obj = model.network
        else:
            net_obj = model
        
        # Check if model has layers for two-step training
        if hasattr(net_obj, "layers"):
            # Initialize local heads for each layer
            if not hasattr(self, "local_heads") or not self.local_heads:
                self.local_heads = []
                for i, layer in enumerate(net_obj.layers):
                    out_dim = getattr(layer, "out_features", None)
                    if out_dim is None:
                        excit_sizes = getattr(net_obj, "excitatory_layer_sizes", None)
                        if isinstance(excit_sizes, list) and i < len(excit_sizes):
                            out_dim = excit_sizes[i]
                        else:
                            try:
                                dummy = torch.randn(1, getattr(net_obj, "input_dim", 784)).to(device)
                            except Exception:
                                dummy = torch.randn(1, 784).to(device)
                            dummy_out = layer(dummy)
                            if isinstance(dummy_out, tuple):
                                dummy_out = dummy_out[0]
                            out_dim = dummy_out.size(1)
                    
                    if hasattr(layer, "weight"):
                        device_for_head = layer.weight.device
                    else:
                        device_for_head = next(model.parameters()).device
                    
                    head = torch.nn.Linear(out_dim, getattr(model, "output_dim", 10)).to(device_for_head)
                    self.local_heads.append(head)
        else:
            logger.info("No net.layers found => falling back to MLE training.")
            from .mle import TrainerMLE
            trainer_mle = TrainerMLE(self.optimizer, self.suppress, self.print_every)
            return trainer_mle.train(
                model, train_data, valid_data,
                grad_clip_value=grad_clip_value,
                epochs=pretrain_epochs + maintrain_epochs,
                batch_size=batch_size, shuffle=shuffle,
                load_best_state_dict=load_best_state_dict,
                analysis_manager=analysis_manager,
                plot_losses=plot_losses, save_path=save_path,
            )
        
        all_train_losses = []
        all_valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        ep_counter = 0
        start_time = time.time()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        
        logger.info(f"=== TwoStepTrainer: Stage 1 (local pretraining) for {pretrain_epochs} epochs ===")
        n_layers = len(net_obj.layers)
        
        # Stage 1: Layer-wise pretraining
        for epoch in range(1, pretrain_epochs + 1):
            ep_counter += 1
            epoch_loss_accum = 0.0
            ct_batches = 0
            
            for layer_idx in range(n_layers):
                # Freeze all parameters
                for name, param in model.named_parameters():
                    param.requires_grad = False
                
                # Unfreeze current layer parameters
                if hasattr(net_obj.layers[layer_idx], "weight"):
                    net_obj.layers[layer_idx].weight.requires_grad = True
                if (hasattr(net_obj.layers[layer_idx], "bias") and 
                    net_obj.layers[layer_idx].bias is not None):
                    net_obj.layers[layer_idx].bias.requires_grad = True
                
                local_head = self.local_heads[layer_idx]
                for param in local_head.parameters():
                    param.requires_grad = True
                
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
                    
                    # Compute loss with local head
                    loss_val = F.cross_entropy(local_head(h), yb)
                    
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    
                    if grad_clip_value is not None:
                        params_to_clip = list(filter(lambda p: p.requires_grad, model.parameters()))
                        if params_to_clip:
                            clip_grad_value_(params_to_clip, grad_clip_value)
                        params_local = list(local_head.parameters())
                        if params_local:
                            clip_grad_value_(params_local, grad_clip_value)
                    
                    self.optimizer.step()
                    epoch_loss_accum += loss_val.item()
                    ct_batches += 1
            
            epoch_train_loss = epoch_loss_accum / (ct_batches if ct_batches > 0 else 1)
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            all_train_losses.append(epoch_train_loss)
            all_valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStep: Stage1] epoch={epoch}/{pretrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                )
            
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'step1_epoch{epoch}', training=True)
            
            if wandb.run is not None:
                wandb.log({
                    "epoch": ep_counter,
                    "phase": "layerwise-pretrain",
                    "train_loss": epoch_train_loss,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc,
                })
        
        logger.info(f"=== TwoStepTrainer: Stage 2 (end-to-end training) for {maintrain_epochs} epochs ===")
        
        # Stage 2: End-to-end training
        for param in model.parameters():
            param.requires_grad = True
        
        for ep in range(1, maintrain_epochs + 1):
            ep_counter += 1
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
                if grad_clip_value is not None:
                    clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
            
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)["accuracy"]
            
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            all_train_losses.append(epoch_train_loss)
            all_valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStep: Stage2] epoch={ep}/{maintrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                    f"elapsed={elapsed:.1f}s"
                )
            
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'step2_epoch{ep}', training=True)
            
            if wandb.run is not None:
                wandb.log({
                    "epoch": ep_counter,
                    "phase": "main-train",
                    "train_loss": epoch_train_loss,
                    "train_acc": train_acc,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc,
                })
        
        if load_best_state_dict:
            model.load_state_dict(best_state)
        
        results = {"train losses": all_train_losses, "valid losses": all_valid_losses}
        
        if plot_losses and ep_counter > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, ep_counter, save_path)
            else:
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, ep_counter)
        
        return results


class TwoStepTrainerWithKL(BaseTrainer):
    """Two-step training with KL divergence: layerwise pretraining with KL loss then end-to-end training."""
    
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        super().__init__(optimizer, suppress_prints, print_every)
        self.local_heads = []
    
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
        analysis_manager: Optional[AnalysisManager] = None,
        plot_losses=False,
        save_path=None,
        **kwargs,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Get network object
        if hasattr(model, "net"):
            net_obj = model.net
        elif hasattr(model, "network"):
            net_obj = model.network
        else:
            net_obj = model
        
        # Check if model has layers for two-step training
        if hasattr(net_obj, "layers"):
            # Initialize local heads for each layer
            if not hasattr(self, "local_heads") or not self.local_heads:
                self.local_heads = []
                n_layers = len(net_obj.layers)
                for i, layer in enumerate(net_obj.layers):
                    out_dim = getattr(layer, "out_features", None)
                    if out_dim is None:
                        excit_sizes = getattr(net_obj, "excitatory_layer_sizes", None)
                        if isinstance(excit_sizes, list) and i < len(excit_sizes):
                            out_dim = excit_sizes[i]
                        else:
                            try:
                                dummy = torch.randn(1, getattr(net_obj, "input_dim", 784)).to(device)
                            except Exception:
                                dummy = torch.randn(1, 784).to(device)
                            dummy_out = layer(dummy)
                            if isinstance(dummy_out, tuple):
                                dummy_out = dummy_out[0]
                            out_dim = dummy_out.size(1)
                    
                    if hasattr(layer, "weight"):
                        device_for_head = layer.weight.device
                    else:
                        device_for_head = next(model.parameters()).device
                    
                    head = torch.nn.Linear(out_dim, getattr(model, "output_dim", 10)).to(device_for_head)
                    self.local_heads.append(head)
        else:
            logger.info("No net.layers found => falling back to MLE training.")
            from .mle import TrainerMLE
            trainer_mle = TrainerMLE(self.optimizer, self.suppress, self.print_every)
            return trainer_mle.train(
                model, train_data, valid_data,
                grad_clip_value=grad_clip_value,
                epochs=pretrain_epochs + maintrain_epochs,
                batch_size=batch_size, shuffle=shuffle,
                load_best_state_dict=load_best_state_dict,
                analysis_manager=analysis_manager,
                plot_losses=plot_losses, save_path=save_path,
            )
        
        all_train_losses = []
        all_valid_losses = []
        best_loss = float("inf")
        best_state = deepcopy(model.state_dict())
        ep_counter = 0
        start_time = time.time()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        
        logger.info(f"=== TwoStepTrainerWithKL: Stage 1 (local pretraining with KL) for {pretrain_epochs} epochs ===")
        n_layers = len(net_obj.layers)
        
        # Stage 1: Layer-wise pretraining with KL divergence
        for epoch in range(1, pretrain_epochs + 1):
            ep_counter += 1
            epoch_loss_accum = 0.0
            ct_batches = 0
            
            for layer_idx in range(n_layers):
                # Freeze all parameters
                for name, param in model.named_parameters():
                    param.requires_grad = False
                
                # Unfreeze current layer parameters
                if hasattr(net_obj.layers[layer_idx], "weight"):
                    net_obj.layers[layer_idx].weight.requires_grad = True
                if (hasattr(net_obj.layers[layer_idx], "bias") and 
                    net_obj.layers[layer_idx].bias is not None):
                    net_obj.layers[layer_idx].bias.requires_grad = True
                
                local_head = self.local_heads[layer_idx]
                for param in local_head.parameters():
                    param.requires_grad = True
                
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
                    
                    # Compute local and final predictions
                    local_logits = local_head(h)
                    ce_loss = F.cross_entropy(local_logits, yb)
                    
                    # Get final model predictions for KL divergence
                    with torch.no_grad():
                        final_logits = model(xb).detach()
                    
                    kl_loss = F.kl_div(
                        torch.log_softmax(local_logits, dim=-1),
                        torch.softmax(final_logits, dim=-1),
                        reduction="batchmean",
                    )
                    
                    # Weighted combination of losses
                    weight_alpha = (layer_idx + 1) / n_layers
                    weight_beta = 1.0 - weight_alpha
                    loss_val = weight_alpha * ce_loss + weight_beta * kl_loss
                    
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    
                    if grad_clip_value is not None:
                        params_to_clip = list(filter(lambda p: p.requires_grad, model.parameters()))
                        if params_to_clip:
                            clip_grad_value_(params_to_clip, grad_clip_value)
                        params_local = list(local_head.parameters())
                        if params_local:
                            clip_grad_value_(params_local, grad_clip_value)
                    
                    self.optimizer.step()
                    epoch_loss_accum += loss_val.item()
                    ct_batches += 1
            
            epoch_train_loss = epoch_loss_accum / (ct_batches if ct_batches > 0 else 1)
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            all_train_losses.append(epoch_train_loss)
            all_valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStepWithKL: Stage1] epoch={epoch}/{pretrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s"
                )
            
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'step1_epoch{epoch}', training=True)
            
            if wandb.run is not None:
                wandb.log({
                    "epoch": ep_counter,
                    "phase": "layerwise-pretrain-kl",
                    "train_loss": epoch_train_loss,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc,
                })
        
        logger.info(f"=== TwoStepTrainerWithKL: Stage 2 (end-to-end training) for {maintrain_epochs} epochs ===")
        
        # Stage 2: End-to-end training
        for param in model.parameters():
            param.requires_grad = True
        
        for ep in range(1, maintrain_epochs + 1):
            ep_counter += 1
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
                if grad_clip_value is not None:
                    clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
            
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)["accuracy"]
            
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            all_train_losses.append(epoch_train_loss)
            all_valid_losses.append(val_loss)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(
                    f"[TwoStepWithKL: Stage2] epoch={ep}/{maintrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                    f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                    f"elapsed={elapsed:.1f}s"
                )
            
            if analysis_manager is not None:
                analysis_manager.run_analysis(filename=f'step2_epoch{ep}', training=True)
            
            if wandb.run is not None:
                wandb.log({
                    "epoch": ep_counter,
                    "phase": "main-train-kl",
                    "train_loss": epoch_train_loss,
                    "train_acc": train_acc,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc,
                })
        
        if load_best_state_dict:
            model.load_state_dict(best_state)
        
        results = {"train losses": all_train_losses, "valid losses": all_valid_losses}
        
        if plot_losses and ep_counter > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, ep_counter, save_path)
            else:
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, ep_counter)
        
        return results 