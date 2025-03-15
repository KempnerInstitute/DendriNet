"""
learning_strategies.py
======================
Houses multiple learning/training strategies:

  - TrainerMLE            (classic NLL backprop)
      * includes train_freeze_layers for freeze/unfreeze approach with a final full-network tuning stage.
  - LocalCreditAssignment (advanced local learning using feedback alignment)
  - TwoStepTrainer        (pretrain each layer with local supervision using cross-entropy loss, then full end-to-end training)
  - TwoStepTrainerWithKL  (pretrain each layer with local supervision using a combined loss (cross-entropy + KL divergence), then full end-to-end training)
  - TrainerVoltageStabilization (does a warmup with MSE to a target voltage, then CE)

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
    def __init__(self, model, optimizer, weight_decay=0.1):
        self.model = model
        self.optimizer = optimizer
        self.weight_decay = weight_decay
    
    def zero_grad(self):
        self.optimizer.zero_grad()
    
    def step(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.model.decay_weights(weight_decay=lr * self.weight_decay)
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
              batch_size=256, shuffle=True, load_best_state_dict=True, best_loss=float('inf'),
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
        final_results = self.train(model, train_data, valid_data, grad_clip_value=5,
                                   epochs=final_tune_epochs, batch_size=batch_size, shuffle=shuffle,
                                   load_best_state_dict=True, plot_losses=plot_losses, save_path=save_path)
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


def voltage_warmup_loss(model, x, target_val=0.5):
    outputs = model(x)
    to_stabilize = []
    if hasattr(model, 'soma_out'):
        to_stabilize.append(model.soma_out)
    if hasattr(model, 'dendrite_out'):
        to_stabilize.append(model.dendrite_out)
    if hasattr(model, 'dendrite_branch_out'):
        to_stabilize.append(model.dendrite_branch_out)
    if hasattr(model, 'inhibition_out'):
        to_stabilize.append(model.inhibition_out)
    if len(to_stabilize) == 0:
        if isinstance(outputs, torch.Tensor):
            to_stabilize.append(outputs)
        else:
            return torch.tensor(0.0, device=x.device, requires_grad=True)
    losses = []
    for t in to_stabilize:
        losses.append(torch.mean((t - target_val) ** 2))
    return torch.mean(torch.stack(losses))


class TrainerVoltageStabilization(BaseTrainer):
    def __init__(self, optimizer, stability_target_voltage=0.5, suppress_prints=False, print_every=10):
        super().__init__(optimizer, suppress_prints, print_every)
        self.stability_target_voltage = stability_target_voltage

    def train(self, model, train_data, valid_data=None, grad_clip_value=5, epochs=20,
              batch_size=256, shuffle=True, load_best_state_dict=True,
              warmup_epochs=5, plot_losses=False, save_path=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

        best_loss = float('inf')
        best_epoch = 1
        best_state = deepcopy(model.state_dict())
        train_losses = []
        valid_losses = []
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            preds_list = []
            labels_list = []

            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                self.optimizer.zero_grad()
                if epoch <= warmup_epochs:
                    loss_val = voltage_warmup_loss(model, xb, target_val=self.stability_target_voltage)
                else:
                    loss_val = model.compute_loss(xb, yb)
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
                if save_path and os.path.exists(save_path):
                    torch.save(best_state, os.path.join(save_path, "best_model.pt"))
            if (not self.suppress) and (epoch % self.print_every == 0):
                elapsed = time.time() - start_time
                if epoch <= warmup_epochs:
                    logger.info(f"[VoltageWarmup] epoch={epoch}/{epochs}, warmup_loss={epoch_train_loss:.4f}, "
                                f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                                f"elapsed={elapsed:.1f}s")
                else:
                    logger.info(f"[VoltageWarmup->CE] epoch={epoch}/{epochs}, train_loss={epoch_train_loss:.4f}, "
                                f"train_acc={train_acc:.3f}, valid_loss={val_loss:.4f}, valid_acc={val_acc:.3f}, "
                                f"elapsed={elapsed:.1f}s")
            if wandb.run is not None:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": epoch_train_loss,
                    "train_acc": train_acc,
                    "valid_loss": val_loss,
                    "valid_acc": val_acc,
                    "warmup_phase": (epoch <= warmup_epochs),
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

    from dendritic_modeling.models import SimpleClassifier

    model = SimpleClassifier(input_dim, hidden_dims, output_dim)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)

    best_valid_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
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

        # Validation
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

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = deepcopy(model.state_dict())
            if run_save_path:
                os.makedirs(run_save_path, exist_ok=True)
                save_path = os.path.join(run_save_path, "best_simple_mlp.pt")
                torch.save(best_state, save_path)

        logger.info(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Evaluate on test
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

    return model, (train_acc, best_valid_acc, test_acc)


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

    from dendritic_modeling.models import MLPWithClassificationHead

    model = MLPWithClassificationHead(feature_extractor, num_features, num_classes)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size)

    best_valid_acc = 0.0
    best_state = None
    best_epoch = 0

    if run_save_path:
        os.makedirs(run_save_path, exist_ok=True)
        logger.info(f"Will save best model to {run_save_path} at the end of training")

    for epoch in range(1, epochs + 1):
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

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            logger.info(f"New best model found at epoch {epoch} with validation accuracy: {valid_acc:.4f}")

        logger.info(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                    f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")

        if wandb.run is not None:
            wandb.log({
                "mlp_train_loss": train_loss,
                "mlp_train_acc": train_acc,
                "mlp_valid_loss": valid_loss,
                "mlp_valid_acc": valid_acc,
            })

    if best_state and run_save_path:
        save_path = os.path.join(run_save_path, "best_mlp_with_classification.pt")
        torch.save(best_state, save_path)
        logger.info(f"Saved best model (from epoch {best_epoch}) to {save_path}")

    if best_state:
        model.load_state_dict(best_state)
        logger.info(f"Loaded best model from epoch {best_epoch} with validation accuracy: {best_valid_acc:.4f}")

    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size)
    test_correct = 0
    test_total = 0
    model.eval()
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

    return model.feature_extractor, (train_acc, best_valid_acc, test_acc)


def transfer_mlp_weights_to_input_net(mlp_net, input_net, head_type="excitatory"):
    logger.info(f"Transferring weights from MLP to {head_type} head of InputNetTransform")
    if hasattr(mlp_net, 'layers'):
        mlp_layers = mlp_net.layers
    elif hasattr(mlp_net, 'net') and hasattr(mlp_net.net, 'layers'):
        mlp_layers = mlp_net.net.layers
    else:
        logger.warning(f"Could not extract layers from MLP model. Using random initialization for {head_type} head.")
        return
    shared_layers = []
    for module in input_net.shared:
        if isinstance(module, torch.nn.Linear):
            shared_layers.append(module)
    target_head = input_net.head_exc if head_type == "excitatory" else input_net.head_inh
    head_layers = []
    for module in target_head:
        if isinstance(module, torch.nn.Linear):
            head_layers.append(module)
    if head_type == "excitatory":
        for i, shared_layer in enumerate(shared_layers):
            if i < len(mlp_layers) - 1:
                mlp_layer = mlp_layers[i]
                if hasattr(mlp_layer, 'excit_pre_w'):
                    shared_layer.weight.data = mlp_layer.excit_pre_w.exp().t()
                    shared_layer.bias.data = mlp_layer.bias.data
                else:
                    logger.warning(f"Unexpected MLP layer structure. Using random initialization for shared layer {i}.")
    if len(head_layers) > 0 and len(mlp_layers) > 0:
        last_mlp_layer = mlp_layers[-1]
        if hasattr(last_mlp_layer, 'excit_pre_w'):
            head_layers[0].weight.data = last_mlp_layer.excit_pre_w.exp().t()
            head_layers[0].bias.data = last_mlp_layer.bias.data
        else:
            logger.warning(f"Unexpected MLP layer structure. Using random initialization for {head_type} head.")
    logger.info(f"Weight transfer completed for {head_type} head")


def extract_weights_from_simple_mlp(simple_mlp, input_net_transform):
    logger.info("Transferring weights from SimpleMLP to InputNetTransform")
    mlp_modules = list(simple_mlp.net.network.children())
    mlp_linear_modules = [m for m in mlp_modules if isinstance(m, torch.nn.Linear)]
    shared_modules = []
    for module in input_net_transform.shared:
        if isinstance(module, torch.nn.Linear):
            shared_modules.append(module)
    exc_head_modules = []
    for module in input_net_transform.head_exc:
        if isinstance(module, torch.nn.Linear):
            exc_head_modules.append(module)
    inh_head_modules = []
    for module in input_net_transform.head_inh:
        if isinstance(module, torch.nn.Linear):
            inh_head_modules.append(module)
    for i, shared_module in enumerate(shared_modules):
        if i < len(mlp_linear_modules) - 1:
            mlp_layer = mlp_linear_modules[i]
            shared_module.weight.data = mlp_layer.weight.data.clone()
            shared_module.bias.data = mlp_layer.bias.data.clone()
            logger.info(f"Transferred weights for shared layer {i}")
    if len(mlp_linear_modules) > 0 and len(exc_head_modules) > 0:
        if len(mlp_linear_modules) >= 2:
            source_layer = mlp_linear_modules[-2]
        else:
            source_layer = mlp_linear_modules[0]
        exc_head_modules[0].weight.data = source_layer.weight.data[:exc_head_modules[0].weight.shape[0], :].clone()
        exc_head_modules[0].bias.data = source_layer.bias.data[:exc_head_modules[0].bias.shape[0]].clone()
        logger.info("Transferred weights to excitatory head")
        inh_head_modules[0].weight.data = source_layer.weight.data[:inh_head_modules[0].weight.shape[0], :].clone()
        inh_head_modules[0].bias.data = source_layer.bias.data[:inh_head_modules[0].bias.shape[0]].clone()
        logger.info("Transferred weights to inhibitory head")
    logger.info("Weight transfer from SimpleMLP to InputNetTransform completed")


def get_trainer(strategy,
                optimizer,
                suppress_prints=False,
                print_every=10,
                stability_target_voltage=0.5):
    strategy = strategy.lower()
    if strategy == "mle":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_layers":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "local_credit_assignment":
        return LocalCreditAssignment(optimizer, suppress_prints, print_every)
    elif strategy == "two_step":
        return TwoStepTrainer(optimizer, suppress_prints, print_every)
    elif strategy == "two_step_kl":
        return TwoStepTrainerWithKL(optimizer, suppress_prints, print_every)
    elif strategy == "voltage_stabilization":
        return TrainerVoltageStabilization(
            optimizer,
            stability_target_voltage=stability_target_voltage,
            suppress_prints=suppress_prints,
            print_every=print_every
        )
    else:
        logger.info(f"Unrecognized strategy '{strategy}', defaulting to 'mle'.")
        return TrainerMLE(optimizer, suppress_prints, print_every)