"""
learning_strategies.py
======================
Houses multiple learning/training strategies:

  - TrainerMLE            (classic NLL backprop)
      * includes train_freeze_layers for freeze/unfreeze approach with a final full-network tuning stage.
  - LocalCreditAssignment (advanced local learning using feedback alignment)
  - TwoStepTrainer        (pretrain each layer with local supervision using cross-entropy loss, then full end-to-end training)
  - TwoStepTrainerWithKL  (pretrain each layer with local supervision using a combined loss (cross-entropy + KL divergence), then full end-to-end training)

A factory function get_trainer(...) returns the chosen strategy based on a string.
"""

import time
import wandb
import os
from copy import deepcopy
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_value_
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves


def compute_metrics_classification(preds, labels):
    preds_np = preds.cpu().numpy()
    labels_np = labels.cpu().numpy()
    return {
        'accuracy': accuracy_score(labels_np, preds_np),
        'precision': precision_score(labels_np, preds_np, average='weighted', zero_division=0),
        'recall': recall_score(labels_np, preds_np, average='weighted', zero_division=0),
        'f1_score': f1_score(labels_np, preds_np, average='weighted', zero_division=0)
    }


class CustomWeightDecayOptimizer():

    def __init__(self, model, optimizer, weight_decay = 0.1):
        self.model = model
        self.optimizer = optimizer
        self.weight_decay = weight_decay
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.model.decay_weights(weight_decay = lr * self.weight_decay)
        self.optimizer.step()


class BaseTrainer:
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every

    def _eval_dataset(self, model, dataset, device, batch_size=256):
        if dataset is None or len(dataset) == 0:
            return 0.0, 0.0
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        preds_list = []
        labels_list = []
        model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                nll = model.compute_loss(xb, yb)
                total_loss += nll.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
        total_loss /= len(loader)
        preds_cat = torch.cat(preds_list, dim=0)
        labels_cat = torch.cat(labels_list, dim=0)
        metrics = compute_metrics_classification(preds_cat, labels_cat)
        return total_loss, metrics["accuracy"]


class TrainerMLE(BaseTrainer):
    def train(self, model, train_data, valid_data=None, grad_clip_value=5, epochs=100,
              batch_size=256, shuffle=True, load_best_state_dict=True, best_loss = float('inf'),
              plot_losses=False, save_path=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        train_losses = []
        valid_losses = []
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
            train_acc = compute_metrics_classification(preds_cat, labels_cat)['accuracy']
            train_losses.append(epoch_train_loss)
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                if save_path and os.path.exists(save_path):
                    torch.save(best_state, os.path.join(save_path, "best_model.pt"))
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(f"[Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                            f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s")
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "train_acc": train_acc,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc
                })
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

    def train_freeze_layers(self, model, train_data, valid_data=None, epochs_per_layer=5,
                            batch_size=256, shuffle=True, grad_clip_value=5,
                            plot_losses=False, save_path=None, final_tune_epochs=10):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        if not hasattr(model, 'net') or not hasattr(model.net, 'layers'):
            logger.info("Model has no 'net.layers'. Falling back to normal train.")
            return self.train(model, train_data, valid_data, grad_clip_value=grad_clip_value,
                              epochs=epochs_per_layer, batch_size=batch_size, shuffle=shuffle,
                              plot_losses=plot_losses, save_path=save_path)
        n_layers = len(model.net.layers)
        logger.info(f"train_freeze_layers: found {n_layers} layers. Training them sequentially.")
        train_losses = []
        valid_losses = []
        best_loss = float('inf')
        best_state = deepcopy(model.state_dict())
        total_epochs = n_layers * epochs_per_layer
        epoch_counter = 0
        start_time = time.time()
        for layer_idx in range(n_layers):
            for name, param in model.named_parameters():
                param.requires_grad = False
            target_str = f"layers.{layer_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True
            logger.info(f"Training only layer {layer_idx} for {epochs_per_layer} epochs.")
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
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
                train_acc = compute_metrics_classification(preds_cat, labels_cat)['accuracy']
                train_losses.append(epoch_train_loss)
                val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
                valid_losses.append(val_loss)
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model.state_dict())
                if (not self.suppress) and (epoch_counter % self.print_every == 0):
                    elapsed = time.time() - start_time
                    logger.info(f"[Freeze-layers] layer={layer_idx}, epoch={epoch_counter}/{total_epochs}, "
                                f"train_loss={epoch_train_loss:.4f}, train_acc={train_acc:.3f}, "
                                f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s")
                if wandb.run is not None:
                    wandb.log({
                        "epoch": epoch_counter,
                        "freeze_layer": layer_idx,
                        "train_loss": epoch_train_loss,
                        "train_acc": train_acc,
                        "valid_loss": val_loss,
                        "valid_acc": val_acc
                    })
        logger.info("Done freeze-layers. Unfreezing all layers and loading best state.")
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)
        logger.info(f"Starting final full-network tuning for {final_tune_epochs} epochs.")
        final_results = self.train(model, train_data, valid_data, grad_clip_value=grad_clip_value,
                                   epochs=final_tune_epochs, batch_size=batch_size, shuffle=shuffle,
                                   load_best_state_dict=load_best_state_dict, plot_losses=plot_losses, save_path=save_path)
        return final_results


class LocalCreditAssignment(BaseTrainer):
    def train(self, model, train_data, valid_data=None, grad_clip_value=5, epochs=100,
              batch_size=256, shuffle=True, load_best_state_dict=True, plot_losses=False, save_path=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        best_loss = float('inf')
        best_epoch = 1
        best_state = deepcopy(model.state_dict())
        train_losses = []
        valid_losses = []
        start_time = time.time()
        logger.info("LocalCreditAssignment: Using feedback alignment for local updates.")
        if hasattr(model, 'net') and hasattr(model.net, 'layers'):
            if not hasattr(self, 'feedback_matrices_initialized') or not self.feedback_matrices_initialized:
                self.feedback_matrices = []
                for layer in model.net.layers:
                    if hasattr(layer, 'weight'):
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
            for p in model.parameters():
                p.requires_grad_(False)
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                with torch.no_grad():
                    logits = model(xb)
                    loss_val = model.compute_loss(xb, yb)
                epoch_train_loss += loss_val.item()
                preds = model.predict(xb)
                preds_list.append(preds.cpu())
                labels_list.append(yb.cpu())
                alpha = 1e-3
                error_signal = (yb - preds.to(device)).float()
                batch_error = error_signal.sum().item()
                sign_of_error = 1.0 if batch_error > 0 else -1.0 if batch_error < 0 else 0.0
                for p in model.parameters():
                    p.data -= alpha * sign_of_error
                if grad_clip_value is not None:
                    for p in model.parameters():
                        p.data = torch.clamp(p.data, -grad_clip_value, grad_clip_value)
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)['accuracy']
            train_losses.append(epoch_train_loss)
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(f"[LocalCredit] epoch={epoch}/{epochs}, train_loss={epoch_train_loss:.4f}, "
                            f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                            f"elapsed={elapsed:.1f}s")
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "train_acc": train_acc,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc
                })
        if load_best_state_dict:
            model.load_state_dict(best_state)
        results = {
            "best epoch": best_epoch,
            "train losses": train_losses,
            "valid losses": valid_losses
        }
        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
            else:
                plot_NLL_loss_curves(train_losses, valid_losses, epochs)
        return results


class TrainOnlyMReactivation(BaseTrainer):
    def train(self, model, train_data, valid_data=None, grad_clip_value=5, epochs=100,
              batch_size=256, shuffle=True, load_best_state_dict=True, plot_losses=False, save_path=None):
        """
        TrainOnlyMReactivation: Freeze only the log_b parameters in the activation modules,
        so that only the slope (m) is updated. This allows the reactivation to behave like a sigmoid
        (with a fixed midpoint), while still learning a proper slope.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        logger.info("TrainOnlyMReactivation: Freezing log_b and training all other parameters.")
        for name, param in model.named_parameters():
            if "log_b" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        train_losses = []
        valid_losses = []
        best_loss = float('inf')
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
            train_acc = compute_metrics_classification(preds_cat, labels_cat)['accuracy']
            train_losses.append(epoch_train_loss)
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(f"[TrainOnlyMReactivation] epoch={epoch}/{epochs}, train_loss={epoch_train_loss:.4f}, "
                            f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                            f"elapsed={elapsed:.1f}s")
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "train_acc": train_acc,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc
                })
        if load_best_state_dict:
            model.load_state_dict(best_state)
        results = {
            "best epoch": best_epoch,
            "train losses": train_losses,
            "valid losses": valid_losses
        }
        if plot_losses and len(train_losses) > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
            else:
                plot_NLL_loss_curves(train_losses, valid_losses, epochs)
        return results


class TwoStepTrainer(BaseTrainer):
    def train(self, model, train_data, valid_data=None, pretrain_epochs=5, maintrain_epochs=45,
              batch_size=256, grad_clip_value=5, shuffle=True, load_best_state_dict=True,
              plot_losses=False, save_path=None):
        """
        TwoStepTrainer: Stage 1 uses only cross-entropy loss for local pretraining;
        Stage 2 is full end-to-end training.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        if hasattr(model, 'net'):
            net_obj = model.net
        elif hasattr(model, 'network'):
            net_obj = model.network
        else:
            net_obj = model
        if hasattr(net_obj, 'layers'):
            if not hasattr(self, 'local_heads'):
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
                    if hasattr(layer, 'weight'):
                        device_for_head = layer.weight.device
                    else:
                        device_for_head = next(model.parameters()).device
                    head = torch.nn.Linear(out_dim, getattr(model, "output_dim", 10)).to(device_for_head)
                    self.local_heads.append(head)
        else:
            logger.info("No net.layers found => falling back to MLE training.")
            trainer_mle = TrainerMLE(self.optimizer, self.suppress, self.print_every)
            return trainer_mle.train(model, train_data, valid_data, grad_clip_value=grad_clip_value,
                                     epochs=pretrain_epochs+maintrain_epochs, batch_size=batch_size,
                                     shuffle=shuffle, load_best_state_dict=load_best_state_dict,
                                     plot_losses=plot_losses, save_path=save_path)
        all_train_losses = []
        all_valid_losses = []
        best_loss = float('inf')
        best_state = deepcopy(model.state_dict())
        total_ep = pretrain_epochs + maintrain_epochs
        ep_counter = 0
        start_time = time.time()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        logger.info(f"=== TwoStepTrainer: Stage 1 (local pretraining) for {pretrain_epochs} epochs ===")
        n_layers = len(net_obj.layers)
        for epoch in range(1, pretrain_epochs + 1):
            ep_counter += 1
            epoch_loss_accum = 0.0
            ct_batches = 0
            for layer_idx in range(n_layers):
                for name, param in model.named_parameters():
                    param.requires_grad = False
                if hasattr(net_obj.layers[layer_idx], 'weight'):
                    net_obj.layers[layer_idx].weight.requires_grad = True
                if hasattr(net_obj.layers[layer_idx], 'bias') and net_obj.layers[layer_idx].bias is not None:
                    net_obj.layers[layer_idx].bias.requires_grad = True
                local_head = self.local_heads[layer_idx]
                for param in local_head.parameters():
                    param.requires_grad = True
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    h = xb
                    for j in range(layer_idx + 1):
                        h = net_obj.layers[j](h)
                        if isinstance(h, tuple):
                            h = h[0]
                        if hasattr(net_obj.layers[j], 'activation'):
                            h = net_obj.layers[j].activation(h)
                    loss_val = F.cross_entropy(local_head(h), yb)
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    if grad_clip_value is not None:
                        params_to_clip = list(filter(lambda p: p.requires_grad, model.parameters()))
                        if params_to_clip:
                            torch.nn.utils.clip_grad_value_(params_to_clip, grad_clip_value)
                        params_local = list(local_head.parameters())
                        if params_local:
                            torch.nn.utils.clip_grad_value_(params_local, grad_clip_value)
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
                logger.info(f"[TwoStep: Stage1] epoch={epoch}/{pretrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                            f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s")
            if wandb.run is not None:
                wandb.log({
                    "epoch": ep_counter,
                    "phase": "layerwise-pretrain",
                    "train_loss": epoch_train_loss,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc
                })
        logger.info(f"=== TwoStepTrainer: Stage 2 (end-to-end training) for {maintrain_epochs} epochs ===")
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
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)['accuracy']
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            all_train_losses.append(epoch_train_loss)
            all_valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(f"[TwoStep: Stage2] epoch={ep}/{maintrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                            f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                            f"elapsed={elapsed:.1f}s")
            if wandb.run is not None:
                wandb.log({
                    "epoch": ep_counter,
                    "phase": "main-train",
                    "train_loss": epoch_train_loss,
                    "train_acc": train_acc,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc
                })
        if load_best_state_dict:
            model.load_state_dict(best_state)
        results = {
            "train losses": all_train_losses,
            "valid losses": all_valid_losses
        }
        if plot_losses and ep_counter > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, ep_counter, save_path)
            else:
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, ep_counter)
        return results


class TwoStepTrainerWithKL(BaseTrainer):
    def train(self, model, train_data, valid_data=None, pretrain_epochs=5, maintrain_epochs=45,
              batch_size=256, grad_clip_value=5, shuffle=True, load_best_state_dict=True,
              plot_losses=False, save_path=None, **kwargs):
        """
        TwoStepTrainerWithKL uses a combined loss during Stage 1 (local pretraining):
        a weighted sum of cross-entropy and KL divergence between the local head's output and the full model's output.
        Extra keyword arguments (e.g. 'epochs') are ignored.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        if hasattr(model, 'net'):
            net_obj = model.net
        elif hasattr(model, 'network'):
            net_obj = model.network
        else:
            net_obj = model
        if hasattr(net_obj, 'layers'):
            if not hasattr(self, 'local_heads'):
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
                    if hasattr(layer, 'weight'):
                        device_for_head = layer.weight.device
                    else:
                        device_for_head = next(model.parameters()).device
                    head = torch.nn.Linear(out_dim, getattr(model, "output_dim", 10)).to(device_for_head)
                    self.local_heads.append(head)
        else:
            logger.info("No net.layers found => falling back to MLE training.")
            trainer_mle = TrainerMLE(self.optimizer, self.suppress, self.print_every)
            return trainer_mle.train(model, train_data, valid_data, grad_clip_value=grad_clip_value,
                                     epochs=pretrain_epochs+maintrain_epochs, batch_size=batch_size,
                                     shuffle=shuffle, load_best_state_dict=load_best_state_dict,
                                     plot_losses=plot_losses, save_path=save_path)
        all_train_losses = []
        all_valid_losses = []
        best_loss = float('inf')
        best_state = deepcopy(model.state_dict())
        total_ep = pretrain_epochs + maintrain_epochs
        ep_counter = 0
        start_time = time.time()
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        logger.info(f"=== TwoStepTrainerWithKL: Stage 1 (local pretraining with KL) for {pretrain_epochs} epochs ===")
        n_layers = len(net_obj.layers)
        for epoch in range(1, pretrain_epochs + 1):
            ep_counter += 1
            epoch_loss_accum = 0.0
            ct_batches = 0
            for layer_idx in range(n_layers):
                for name, param in model.named_parameters():
                    param.requires_grad = False
                if hasattr(net_obj.layers[layer_idx], 'weight'):
                    net_obj.layers[layer_idx].weight.requires_grad = True
                if hasattr(net_obj.layers[layer_idx], 'bias') and net_obj.layers[layer_idx].bias is not None:
                    net_obj.layers[layer_idx].bias.requires_grad = True
                local_head = self.local_heads[layer_idx]
                for param in local_head.parameters():
                    param.requires_grad = True
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    h = xb
                    for j in range(layer_idx + 1):
                        h = net_obj.layers[j](h)
                        if isinstance(h, tuple):
                            h = h[0]
                        if hasattr(net_obj.layers[j], 'activation'):
                            h = net_obj.layers[j].activation(h)
                    local_logits = local_head(h)
                    ce_loss = F.cross_entropy(local_logits, yb)
                    with torch.no_grad():
                        final_logits = model(xb).detach()
                    kl_loss = F.kl_div(torch.log_softmax(local_logits, dim=-1),
                                         torch.softmax(final_logits, dim=-1),
                                         reduction='batchmean')
                    weight_alpha = (layer_idx + 1) / n_layers
                    weight_beta = 1.0 - weight_alpha
                    loss_val = weight_alpha * ce_loss + weight_beta * kl_loss
                    self.optimizer.zero_grad()
                    loss_val.backward()
                    if grad_clip_value is not None:
                        params_to_clip = list(filter(lambda p: p.requires_grad, model.parameters()))
                        if params_to_clip:
                            torch.nn.utils.clip_grad_value_(params_to_clip, grad_clip_value)
                        params_local = list(local_head.parameters())
                        if params_local:
                            torch.nn.utils.clip_grad_value_(params_local, grad_clip_value)
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
                logger.info(f"[TwoStepWithKL: Stage1] epoch={epoch}/{pretrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                            f"valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, elapsed={elapsed:.1f}s")
            if wandb.run is not None:
                wandb.log({
                    "epoch": ep_counter,
                    "phase": "layerwise-pretrain-kl",
                    "train_loss": epoch_train_loss,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc
                })
        logger.info(f"=== TwoStepTrainerWithKL: Stage 2 (end-to-end training) for {maintrain_epochs} epochs ===")
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
                    torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
                p = model.predict(xb)
                preds_list.append(p.cpu())
                labels_list.append(yb.cpu())
            epoch_train_loss /= len(train_loader)
            preds_cat = torch.cat(preds_list, dim=0)
            labels_cat = torch.cat(labels_list, dim=0)
            train_acc = compute_metrics_classification(preds_cat, labels_cat)['accuracy']
            val_loss, val_acc = self._eval_dataset(model, valid_data, device, batch_size)
            all_train_losses.append(epoch_train_loss)
            all_valid_losses.append(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = deepcopy(model.state_dict())
            if (not self.suppress) and (ep_counter % self.print_every == 0):
                elapsed = time.time() - start_time
                logger.info(f"[TwoStepWithKL: Stage2] epoch={ep}/{maintrain_epochs}, train_loss={epoch_train_loss:.4f}, "
                            f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                            f"elapsed={elapsed:.1f}s")
            if wandb.run is not None:
                wandb.log({
                    "epoch": ep_counter,
                    "phase": "main-train-kl",
                    "train_loss": epoch_train_loss,
                    "train_acc": train_acc,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc
                })
        if load_best_state_dict:
            model.load_state_dict(best_state)
        results = {
            "train losses": all_train_losses,
            "valid losses": all_valid_losses
        }
        if plot_losses and ep_counter > 1:
            if save_path and os.path.exists(save_path):
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, ep_counter, save_path)
            else:
                plot_NLL_loss_curves(all_train_losses, all_valid_losses, ep_counter)
        return results


class VoltageStabilizationTrainer(BaseTrainer):
    """
    A trainer that first uses a voltage stabilization loss to warm up the network,
    then switches to the actual task loss.
    
    This approach helps in initializing the network by encouraging stable voltage
    levels across different components (soma, dendrites, branches) before 
    training on the actual task.
    """
    
    def train(self, model, train_data, valid_data=None, 
              warmup_epochs=5, maintrain_epochs=45,
              batch_size=256, grad_clip_value=5, shuffle=True, 
              load_best_state_dict=True, plot_losses=False, save_path=None,
              target_voltage=0.5):
        """
        Train with a warm-up phase using voltage stabilization loss.
        
        Parameters
        ----------
        model : nn.Module
            The model to train.
        train_data : tuple or torch.utils.data.Dataset
            The training data.
        valid_data : tuple or torch.utils.data.Dataset, optional
            The validation data.
        warmup_epochs : int
            Number of epochs to train with voltage stabilization loss.
        maintrain_epochs : int
            Number of epochs to train with the task loss.
        batch_size : int
            The batch size.
        grad_clip_value : float
            Clip gradients to this value.
        shuffle : bool
            Whether to shuffle the data.
        load_best_state_dict : bool
            Whether to load the best state dictionary at the end.
        plot_losses : bool
            Whether to plot losses.
        save_path : str, optional
            Path to save the model.
        target_voltage : float
            Target voltage for the stabilization loss (default: 0.5).
            
        Returns
        -------
        tuple
            (train_losses, valid_losses)
        """
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
        import time
        
        device = next(model.parameters()).device
        
        # Prepare data
        if isinstance(train_data, tuple):
            X_train, y_train = train_data
            train_dataset = TensorDataset(X_train, y_train)
        else:
            train_dataset = train_data
            
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle
        )
        
        if valid_data is not None:
            if isinstance(valid_data, tuple):
                X_valid, y_valid = valid_data
                valid_dataset = TensorDataset(X_valid, y_valid)
            else:
                valid_dataset = valid_data
                
            valid_loader = DataLoader(
                valid_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            valid_loader = None
        
        # Define the task loss function
        if hasattr(model, 'log_prob'):
            # Use NLL for probabilistic models
            task_loss_fn = lambda model, X, y: -model.log_prob(X, y).mean()
        else:
            # Use cross-entropy for deterministic classification models
            task_loss_fn = lambda model, X, y: F.cross_entropy(model(X), y)
        
        # Define the voltage stabilization loss function
        def voltage_stabilization_loss(model, X, target_voltage=0.5):
            """
            Calculate voltage stabilization loss to encourage 
            stable voltage levels across the network.
            """
            # Forward pass to get all voltages
            _ = model(X)  # Run forward to populate internal states
            
            total_loss = 0.0
            voltage_values = []
            
            # Collect voltage values from each branch layer and soma
            if hasattr(model, 'layers'):  # For ExcitationInhibitionNetwork
                for layer in model.layers:
                    # Get voltages from excitatory cells
                    if hasattr(layer, 'excitatory_cells'):
                        if hasattr(layer.excitatory_cells, 'soma_layer') and hasattr(layer.excitatory_cells.soma_layer, 'voltage'):
                            voltage_values.append(layer.excitatory_cells.soma_layer.voltage)
                        
                        # Get voltages from branch layers
                        if hasattr(layer.excitatory_cells, 'branch_layers'):
                            for branch_layer in layer.excitatory_cells.branch_layers:
                                if hasattr(branch_layer, 'voltage'):
                                    voltage_values.append(branch_layer.voltage)
                    
                    # If available, get voltages from inhibitory cells
                    if hasattr(layer, 'inhibitory_cells') and layer.inhibitory_cells is not None:
                        if hasattr(layer.inhibitory_cells, 'soma_layer') and hasattr(layer.inhibitory_cells.soma_layer, 'voltage'):
                            voltage_values.append(layer.inhibitory_cells.soma_layer.voltage)
                        
                        if hasattr(layer.inhibitory_cells, 'branch_layers'):
                            for branch_layer in layer.inhibitory_cells.branch_layers:
                                if hasattr(branch_layer, 'voltage'):
                                    voltage_values.append(branch_layer.voltage)
            
            # For DendriNet models directly
            elif hasattr(model, 'branch_layers'):
                for branch_layer in model.branch_layers:
                    if hasattr(branch_layer, 'voltage'):
                        voltage_values.append(branch_layer.voltage)
                
                if hasattr(model, 'soma_layer') and hasattr(model.soma_layer, 'voltage'):
                    voltage_values.append(model.soma_layer.voltage)
            
            # Calculate MSE loss between each voltage and target
            for v in voltage_values:
                if v is not None:
                    total_loss += F.mse_loss(v, torch.ones_like(v) * target_voltage)
            
            # If no voltage values found, default to model output
            if len(voltage_values) == 0:
                outputs = model(X)
                total_loss = F.mse_loss(outputs, torch.ones_like(outputs) * target_voltage)
                
            return total_loss
        
        # Create optimizer
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters())
        
        # Initialize tracking variables
        best_valid_loss = float('inf')
        best_state_dict = None
        train_losses = []
        valid_losses = []
        
        total_epochs = warmup_epochs + maintrain_epochs
        start_time = time.time()
        
        # Training loop
        for epoch in range(total_epochs):
            model.train()
            epoch_train_loss = 0.0
            
            # Determine which loss function to use
            is_warmup_phase = epoch < warmup_epochs
            if is_warmup_phase:
                loss_fn = lambda X, y: voltage_stabilization_loss(model, X, target_voltage)
                phase = "Warmup"
            else:
                loss_fn = lambda X, y: task_loss_fn(model, X, y)
                phase = "Training"
            
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                self.optimizer.zero_grad()
                loss = loss_fn(X_batch, y_batch)
                loss.backward()
                
                # Gradient clipping
                if grad_clip_value > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
                
                self.optimizer.step()
                
                # Apply weight decay
                if hasattr(model, 'decay_weights'):
                    model.decay_weights(weight_decay=0.1)
                
                epoch_train_loss += loss.item()
            
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation
            if valid_loader is not None:
                model.eval()
                epoch_valid_loss = 0.0
                
                with torch.no_grad():
                    for X_batch, y_batch in valid_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        
                        # Always use task loss for validation, even during warmup
                        valid_loss = task_loss_fn(model, X_batch, y_batch)
                        epoch_valid_loss += valid_loss.item()
                
                epoch_valid_loss /= len(valid_loader)
                valid_losses.append(epoch_valid_loss)
                
                # Save best model
                if epoch >= warmup_epochs and epoch_valid_loss < best_valid_loss:
                    best_valid_loss = epoch_valid_loss
                    best_state_dict = model.state_dict().copy()
                    
                    if save_path:
                        torch.save(best_state_dict, save_path)
                
                if not self.suppress and (epoch + 1) % self.print_every == 0:
                    elapsed = time.time() - start_time
                    print(f"{phase} Epoch {epoch+1}/{total_epochs}, "
                          f"Train Loss: {epoch_train_loss:.6f}, "
                          f"Valid Loss: {epoch_valid_loss:.6f}, "
                          f"Elapsed: {elapsed:.1f}s")
            else:
                if not self.suppress and (epoch + 1) % self.print_every == 0:
                    elapsed = time.time() - start_time
                    print(f"{phase} Epoch {epoch+1}/{total_epochs}, "
                          f"Train Loss: {epoch_train_loss:.6f}, "
                          f"Elapsed: {elapsed:.1f}s")
            
            # Important phase transition message
            if is_warmup_phase and epoch == warmup_epochs - 1:
                if not self.suppress:
                    print(f"Switching from voltage stabilization loss to task loss")
                    
            # Log to wandb if available
            if wandb.run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "phase": phase,
                }
                if valid_loader is not None:
                    log_dict["valid_loss"] = epoch_valid_loss
                wandb.log(log_dict)
        
        # Load best model if applicable
        if load_best_state_dict and best_state_dict is not None:
            model.load_state_dict(best_state_dict)
        
        # Plotting if requested
        if plot_losses:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label='Train Loss')
            if valid_loader is not None:
                plt.plot(valid_losses, label='Valid Loss')
            plt.axvline(x=warmup_epochs, color='r', linestyle='--', 
                        label='Warmup to Task Transition')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Loss Curves')
            plt.show()
        
        # Return just the two values expected by the calling code
        return train_losses, valid_losses if valid_loader is not None else None


def get_trainer(strategy, optimizer, suppress_prints=False, print_every=10):
    """
    Get the appropriate trainer for the specified training strategy.
    
    Parameters
    ----------
    strategy : str
        The training strategy.
    optimizer : torch.optim.Optimizer
        The optimizer to use.
    suppress_prints : bool
        Whether to suppress prints.
    print_every : int
        How often to print.
        
    Returns
    -------
    BaseTrainer
        The trainer.
    """
    strategy = strategy.lower()
    if strategy == "mle":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_layers":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "local_credit_assignment":
        return LocalCreditAssignment(optimizer, suppress_prints, print_every)
    elif strategy == "only_m":
        return TrainOnlyMReactivation(optimizer, suppress_prints, print_every)
    elif strategy == "two_step":
        return TwoStepTrainer(optimizer, suppress_prints, print_every)
    elif strategy == "two_step_kl":
        return TwoStepTrainerWithKL(optimizer, suppress_prints, print_every)
    elif strategy == "voltage_stabilization":
        return VoltageStabilizationTrainer(optimizer, suppress_prints, print_every)
    else:
        logger.info(f"Unrecognized strategy '{strategy}', defaulting to 'mle'.")
        return TrainerMLE(optimizer, suppress_prints, print_every)


# Add a function to train the simple MLP
def train_simple_mlp(input_dim, hidden_dims, output_dim, train_ds, valid_ds, test_ds, 
                     epochs=20, batch_size=128, lr=0.001, run_save_path=None, device=None):
    """
    Train a simple MLP model with better performance.
    Returns the trained model and its accuracy.
    
    Args:
        input_dim (int): Input dimension
        hidden_dims (list): List of hidden layer dimensions
        output_dim (int): Output dimension (number of classes)
        train_ds, valid_ds, test_ds (Dataset): PyTorch datasets
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Learning rate
        run_save_path (str): Path to save model checkpoints
        device (str): Device to use ('cuda' or 'cpu')
        
    Returns:
        Tuple of (model, (train_acc, valid_acc, test_acc))
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Import here to avoid circular imports
    from dendritic_modeling.models import SimpleClassifier
    
    # Create the model
    model = SimpleClassifier(input_dim, hidden_dims, output_dim)
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)
    
    # Training loop
    best_valid_acc = 0.0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = model.criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += yb.size(0)
            train_correct += predicted.eq(yb).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                outputs = model(xb)
                loss = model.criterion(outputs, yb)
                
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += yb.size(0)
                valid_correct += predicted.eq(yb).sum().item()
        
        valid_loss /= len(valid_loader)
        valid_acc = valid_correct / valid_total
        
        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = deepcopy(model.state_dict())
            
            if run_save_path:
                os.makedirs(run_save_path, exist_ok=True)
                save_path = os.path.join(run_save_path, "best_simple_mlp.pt")
                torch.save(best_state, save_path)
        
        # Log progress
        logger.info(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
    
    # Evaluate on test set
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            outputs = model(xb)
            _, predicted = outputs.max(1)
            test_total += yb.size(0)
            test_correct += predicted.eq(yb).sum().item()
    
    test_acc = test_correct / test_total
    logger.info(f"Test accuracy: {test_acc:.4f}")
    
    # Return the model and its accuracy
    return model, (train_acc, best_valid_acc, test_acc)


# Function to train an MLP with a classification head for feature learning
def train_mlp_with_classification_head(feature_extractor, num_features, num_classes, 
                                     train_ds, valid_ds, test_ds, 
                                     epochs=20, batch_size=128, lr=0.001, run_save_path=None, device=None):
    """
    Train an MLP with a classification head, then return the trained feature extractor part.
    
    Args:
        feature_extractor: The feature extraction part of the MLP
        num_features: The output dimension of the feature extractor
        num_classes: The number of classes for classification
        train_ds, valid_ds, test_ds: The datasets
        epochs, batch_size, lr: Training parameters
        run_save_path: Where to save the model and logs
        device: Device to use ('cuda' or 'cpu')
        
    Returns:
        The trained feature extractor and performance metrics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Import here to avoid circular imports
    from dendritic_modeling.models import MLPWithClassificationHead
    
    # Create the model with classification head
    model = MLPWithClassificationHead(feature_extractor, num_features, num_classes)
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)
    
    # Training loop
    best_valid_acc = 0.0
    best_state = None
    best_epoch = 0
    
    # Create the save directory if it doesn't exist
    if run_save_path:
        # Ensure the main directory exists
        os.makedirs(run_save_path, exist_ok=True)
        logger.info(f"Will save best model to {run_save_path} at the end of training")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimizer.zero_grad()
            outputs = model(xb)
            loss = model.criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += yb.size(0)
            train_correct += predicted.eq(yb).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                
                outputs = model(xb)
                loss = model.criterion(outputs, yb)
                
                valid_loss += loss.item()
                _, predicted = outputs.max(1)
                valid_total += yb.size(0)
                valid_correct += predicted.eq(yb).sum().item()
        
        valid_loss /= len(valid_loader)
        valid_acc = valid_correct / valid_total
        
        # Track best model but don't save yet
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            logger.info(f"New best model found at epoch {epoch} with validation accuracy: {valid_acc:.4f}")
        
        # Log progress
        logger.info(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "mlp_train_loss": train_loss,
                "mlp_train_acc": train_acc,
                "mlp_valid_loss": valid_loss,
                "mlp_valid_acc": valid_acc,
            })
    
    # Save only the best model at the end of training
    if best_state and run_save_path:
        save_path = os.path.join(run_save_path, "best_mlp_with_classification.pt")
        torch.save(best_state, save_path)
        logger.info(f"Saved best model (from epoch {best_epoch}) to {save_path}")
    
    # Load best model
    if best_state:
        model.load_state_dict(best_state)
        logger.info(f"Loaded best model from epoch {best_epoch} with validation accuracy: {best_valid_acc:.4f}")
    
    # Evaluate on test set
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            
            outputs = model(xb)
            _, predicted = outputs.max(1)
            test_total += yb.size(0)
            test_correct += predicted.eq(yb).sum().item()
    
    test_acc = test_correct / test_total
    logger.info(f"Test accuracy with classification head: {test_acc:.4f}")
    
    # Return the feature extractor part (without the classification head)
    return model.feature_extractor, (train_acc, best_valid_acc, test_acc)


class InhibitoryHeadTrainer:
    """
    Custom trainer for the inhibitory head with different objective functions.
    """
    def __init__(self, model, optimizer, objective="classification", 
                 excit_head_model=None, alpha=0.5,
                 suppress_prints=False, print_every=10):
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
        """
        if self.excit_head_model is None:
            return torch.tensor(0.0, device=x.device)
        
        # Get representations from both heads
        with torch.no_grad():
            if hasattr(self.excit_head_model, 'net'):
                excit_repr = self.excit_head_model.net(x)
            elif hasattr(self.excit_head_model, 'network'):
                excit_repr = self.excit_head_model.network(x)
            else:
                excit_repr = self.excit_head_model(x)
        
        if hasattr(self.model, 'net'):
            inhib_repr = self.model.net(x)
        elif hasattr(self.model, 'network'):
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
    
    def train(self, model, train_data, valid_data=None, epochs=100, batch_size=256, 
              grad_clip_value=5, shuffle=True, load_best_state_dict=True, 
              plot_losses=False, save_path=None):
        """
        Train the inhibitory head with the specified objective.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        if self.excit_head_model is not None:
            self.excit_head_model = self.excit_head_model.to(device)
        
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        
        train_losses = []
        valid_losses = []
        best_loss = float('inf')
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
                valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)
                with torch.no_grad():
                    for xb, yb in valid_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        
                        # Always use task loss for validation, even during warmup
                        valid_loss = self.compute_classification_loss(xb, yb)
                        val_loss += valid_loss.item()
                
                val_loss /= len(valid_loader)
                valid_losses.append(val_loss)
                
                # Save best model
                if epoch >= warmup_epochs and epoch_valid_loss < best_valid_loss:
                    best_valid_loss = epoch_valid_loss
                    best_state = deepcopy(model.state_dict())
                    
                    if save_path and os.path.exists(save_path):
                        torch.save(best_state, os.path.join(save_path, "best_model.pt"))
            
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                if valid_data is not None:
                    logger.info(f"[Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.4f}, "
                                f"valid_loss={val_loss:.4f}, elapsed={elapsed:.1f}s")
                else:
                    logger.info(f"[Epoch {epoch}/{epochs}] train_loss={epoch_train_loss:.4f}, "
                                f"elapsed={elapsed:.1f}s")
            
            if wandb.run is not None:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
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
                from dendritic_modeling.plot_utils import plot_NLL_loss_curves
                plot_NLL_loss_curves(train_losses, valid_losses if valid_data is not None else [], 
                                     epochs, save_path)
        
        return results


def transfer_mlp_weights_to_input_net(mlp_net, input_net, head_type="excitatory"):
    """
    Transfer weights from a trained MLP to the corresponding part of an InputNetTransform.
    
    Args:
        mlp_net: The trained MLP network
        input_net: The InputNetTransform to transfer weights to
        head_type: Either "excitatory" or "inhibitory" to specify which head to update
    """
    logger.info(f"Transferring weights from MLP to {head_type} head of InputNetTransform")
    
    # Extract the MLP layers
    if hasattr(mlp_net, 'layers'):
        mlp_layers = mlp_net.layers
    elif hasattr(mlp_net, 'net') and hasattr(mlp_net.net, 'layers'):
        mlp_layers = mlp_net.net.layers
    else:
        logger.warning(f"Could not extract layers from MLP model. Using random initialization for {head_type} head.")
        return
    
    # Get the shared backbone and target head of the InputNetTransform
    shared_layers = []
    for module in input_net.shared:
        if isinstance(module, torch.nn.Linear):
            shared_layers.append(module)
    
    target_head = input_net.head_exc if head_type == "excitatory" else input_net.head_inh
    head_layers = []
    for module in target_head:
        if isinstance(module, torch.nn.Linear):
            head_layers.append(module)
    
    # Transfer weights for the shared backbone (only once, typically from excitatory head)
    if head_type == "excitatory":
        for i, shared_layer in enumerate(shared_layers):
            if i < len(mlp_layers) - 1:  # All but the last MLP layer go to shared backbone
                mlp_layer = mlp_layers[i]
                if hasattr(mlp_layer, 'excit_pre_w'):
                    # MLPExcInhLayer has a different structure
                    shared_layer.weight.data = mlp_layer.excit_pre_w.exp().t()
                    shared_layer.bias.data = mlp_layer.bias.data
                else:
                    logger.warning(f"Unexpected MLP layer structure. Using random initialization for shared layer {i}.")
    
    # Transfer weights for the head-specific part
    if len(head_layers) > 0 and len(mlp_layers) > 0:
        # The last layer of MLP goes to the head
        last_mlp_layer = mlp_layers[-1]
        if hasattr(last_mlp_layer, 'excit_pre_w'):
            head_layers[0].weight.data = last_mlp_layer.excit_pre_w.exp().t()
            head_layers[0].bias.data = last_mlp_layer.bias.data
        else:
            logger.warning(f"Unexpected MLP layer structure. Using random initialization for {head_type} head.")
    
    logger.info(f"Weight transfer completed for {head_type} head")


def extract_weights_from_simple_mlp(simple_mlp, input_net_transform):
    """
    Extract weights from a SimpleMLP model and transfer them to an InputNetTransform.
    
    Args:
        simple_mlp: The trained SimpleMLP model
        input_net_transform: The InputNetTransform to transfer weights to
    """
    logger.info("Transferring weights from SimpleMLP to InputNetTransform")
    
    # Get network modules
    mlp_modules = list(simple_mlp.net.network.children())
    mlp_linear_modules = [m for m in mlp_modules if isinstance(m, torch.nn.Linear)]
    
    # Get shared backbone modules of InputNetTransform
    shared_modules = []
    for module in input_net_transform.shared:
        if isinstance(module, torch.nn.Linear):
            shared_modules.append(module)
    
    # Get head modules
    exc_head_modules = []
    for module in input_net_transform.head_exc:
        if isinstance(module, torch.nn.Linear):
            exc_head_modules.append(module)
    
    inh_head_modules = []
    for module in input_net_transform.head_inh:
        if isinstance(module, torch.nn.Linear):
            inh_head_modules.append(module)
    
    # Transfer weights for the shared backbone (from early layers of MLP)
    for i, shared_module in enumerate(shared_modules):
        if i < len(mlp_linear_modules) - 1:  # All but the last MLP layer
            mlp_layer = mlp_linear_modules[i]
            shared_module.weight.data = mlp_layer.weight.data.clone()
            shared_module.bias.data = mlp_layer.bias.data.clone()
            logger.info(f"Transferred weights for shared layer {i}")
    
    # Transfer weights for heads (from last layer of MLP)
    if len(mlp_linear_modules) > 0 and len(exc_head_modules) > 0:
        # Use the second-to-last layer for the heads (since the last layer is the classifier)
        if len(mlp_linear_modules) >= 2:
            source_layer = mlp_linear_modules[-2]
        else:
            source_layer = mlp_linear_modules[0]
        
        # Transfer to excitatory head
        exc_head_modules[0].weight.data = source_layer.weight.data[:exc_head_modules[0].weight.shape[0], :].clone()
        exc_head_modules[0].bias.data = source_layer.bias.data[:exc_head_modules[0].bias.shape[0]].clone()
        logger.info("Transferred weights to excitatory head")
        
        # Transfer to inhibitory head
        inh_head_modules[0].weight.data = source_layer.weight.data[:inh_head_modules[0].weight.shape[0], :].clone()
        inh_head_modules[0].bias.data = source_layer.bias.data[:inh_head_modules[0].bias.shape[0]].clone()
        logger.info("Transferred weights to inhibitory head")
    
    logger.info("Weight transfer from SimpleMLP to InputNetTransform completed")