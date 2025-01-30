"""
training.py
===========
This module contains the TrainerMLE class for training the models.

MODIFICATIONS for ensuring wandb runs and logs train accuracy each epoch:
--------------------------------------------------------------------------
1) At the start of .train(...), forcibly call wandb.init(...) if wandb.run is None.
2) Remove the condition 'if wandb.run is not None' around wandb.log(...) calls.
   Instead, we unconditionally call wandb.log(...).
3) Collect train predictions each epoch to compute train accuracy and log it.
"""

import os
import json
import time
from copy import deepcopy

import wandb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.autograd as ag
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_

from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves

def compute_metrics_classification(preds, labels):
    """
    Computes common classification metrics for a single set of predictions. 
    Both 'preds' and 'labels' should be 1D Tensors of the same shape.
    Returns a dict with keys 'accuracy', 'precision', 'recall', 'f1_score'.
    """
    assert preds.shape == labels.shape
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, average='weighted'),
        'recall': recall_score(labels, preds, average='weighted'),
        'f1_score': f1_score(labels, preds, average='weighted')
    }

class TrainerMLE(object):
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        """
        optimizer : torch.optim.Optimizer
        suppress_prints : bool
        print_every : int
        """
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every

    def train(
        self, 
        model, 
        train_data, 
        valid_data, 
        grad_clip_value=5,
        epochs=100, 
        batch_size=256, 
        shuffle=True, 
        load_best_state_dict=True,
        plot_losses=False, 
        save_path=None,
    ):
        """
        Trains the model using maximum-likelihood (NLL) approach.
        Also logs train & valid accuracy each epoch to wandb.
        """
        # ------------------------
        # 1) Force wandb init if not already running
        # ------------------------
        if not wandb.run:  # i.e., wandb.run is None
            wandb.init(
                project="dendritic_forced_project", 
                name="forced_run", 
                config={
                    "epochs": epochs, 
                    "batch_size": batch_size, 
                    "grad_clip_value": grad_clip_value
                }
            )
        # ------------------------

        save_boolean = save_path is not None and os.path.exists(save_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

        train_losses = []
        valid_losses = []
        
        stime = time.time()
        best_loss = float('inf')
        best_state_dict = deepcopy(model.state_dict())
        best_epoch = 1

        # We'll get the last branch if it exists (for conduction scale logging)
        last_branch = None
        if hasattr(model, 'net') and hasattr(model.net, 'branch_layers'):
            if len(model.net.branch_layers) > 0:
                last_branch = model.net.branch_layers[-1]

        for epoch in range(1, epochs+1):
            # 1) TRAIN
            model.train()
            epoch_train_loss = 0.0
            train_preds = []
            train_labels = []
            scale_sum = 0.0
            scale_count = 0

            for input_batch, label_batch in train_loader:
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device)

                # Forward + compute loss
                nll = model.compute_loss(input_batch, label_batch)

                # Backprop
                self.optimizer.zero_grad()
                nll.backward()

                # If using block_conductance_dynamic, the hook sets last_scale_vec
                if last_branch and hasattr(last_branch, 'branches_to_output'):
                    block = last_branch.branches_to_output
                    if hasattr(block, 'last_scale_vec'):
                        scale_mean = block.last_scale_vec.mean().item()
                        scale_sum += scale_mean
                        scale_count += 1

                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()

                epoch_train_loss += nll.item()

                # Collect predictions for accuracy
                preds = model.predict(input_batch)
                train_preds.append(preds.detach().cpu())
                train_labels.append(label_batch.detach().cpu())

            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)

            # Compute train accuracy
            train_preds = torch.cat(train_preds, dim=0)
            train_labels = torch.cat(train_labels, dim=0)
            train_metrics = compute_metrics_classification(train_preds, train_labels)
            train_acc = train_metrics['accuracy']

            # 2) VALID
            model.eval()
            epoch_valid_loss = 0.0
            valid_preds = []
            valid_labels = []

            with torch.no_grad():
                for input_batch, label_batch in valid_loader:
                    input_batch = input_batch.to(device)
                    label_batch = label_batch.to(device)

                    nll = model.compute_loss(input_batch, label_batch)
                    epoch_valid_loss += nll.item()

                    preds = model.predict(input_batch)
                    valid_preds.append(preds.cpu())
                    valid_labels.append(label_batch.cpu())

            epoch_valid_loss /= len(valid_loader)
            valid_losses.append(epoch_valid_loss)

            valid_preds = torch.cat(valid_preds, dim=0)
            valid_labels = torch.cat(valid_labels, dim=0)
            valid_metrics = compute_metrics_classification(valid_preds, valid_labels)
            valid_acc = valid_metrics['accuracy']

            # Update best model if needed
            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                if save_boolean:
                    torch.save(best_state_dict, os.path.join(save_path, 'state_dict.pt'))
                    with open(os.path.join(save_path, 'losses.json'), 'w') as f:
                        json.dump({
                            'best epoch': best_epoch, 
                            'train losses': train_losses, 
                            'valid losses': valid_losses
                            'valid acc': valid_acc
                        }, f)

            # 3) Print logs
            if not self.suppress:
                if epoch % self.print_every == 0:
                    logger.info(f" ---- Epoch {epoch} ---- ")
                    logger.info(f"training loss: {epoch_train_loss:.2f} | validation loss: {epoch_valid_loss:.2f}")
                    logger.info(f"training acc: {train_acc:.3f} | validation acc: {valid_acc:.3f}")
                    elapsed = time.time() - stime
                    pred_remaining = (elapsed / epoch) * (epochs - epoch)
                    logger.info(f"time elapsed: {elapsed:.2f}s | predicted time remaining: {pred_remaining:.2f}s")

            # 4) Log to wandb
            wandb_dict = {
                "epoch": epoch,
                "train_loss": epoch_train_loss,
                "valid_loss": epoch_valid_loss,
                "train_acc": train_acc,
                "valid_acc": valid_acc,
            }
            if scale_count > 0:
                wandb_dict["grad_scale_vec_mean"] = scale_sum / scale_count

            wandb.log(wandb_dict)

        # Done training
        results = {
            'best epoch': best_epoch,
            'train losses': train_losses,
            'valid losses': valid_losses,
            'valid acc': valid_acc
        }
        if load_best_state_dict:
            model.load_state_dict(best_state_dict)
        else:
            results['best state dict'] = best_state_dict
        
        if plot_losses and save_boolean:
            plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
        elif plot_losses:
            results['loss curves'] = plot_NLL_loss_curves(train_losses, valid_losses, epochs)

        return results

    def train_freeze_layers(
        self,
        model,
        train_data,
        valid_data,
        epochs_per_layer=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None
    ):
        """
        Freezes/unfreezes layers in sequence to train each individually.
        (Here we do not replicate the train-accuracy logic 
         but you can adapt similarly if you want.)
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        from torch.utils.data import DataLoader
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

        if not hasattr(model, 'net') or not hasattr(model.net, 'layers'):
            logger.info('Model has no "net.layers" to freeze. Doing normal train.')
            return self.train(
                model, 
                train_data, 
                valid_data, 
                grad_clip_value=grad_clip_value, 
                epochs=epochs_per_layer, 
                batch_size=batch_size, 
                shuffle=shuffle, 
                plot_losses=plot_losses, 
                save_path=save_path
            )

        n_layers = len(model.net.layers)
        logger.info(f"Freezing/unfreezing approach: found {n_layers} layers in model.net.layers.")
        train_losses = []
        valid_losses = []
        best_state_dict = deepcopy(model.state_dict())
        best_loss = float('inf')

        for layer_idx in range(n_layers):
            # Freeze everything
            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze only this layer
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}." in name:
                    param.requires_grad = True

            logger.info(f"Training only layer {layer_idx}, freezing others.")

            # standard loop
            for epoch in range(1, epochs_per_layer + 1):
                model.train()
                epoch_train_loss = 0.0
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    nll = model.compute_loss(xb, yb)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()
                epoch_train_loss /= len(train_loader)

                # valid
                model.eval()
                epoch_valid_loss = 0.0
                with torch.no_grad():
                    for xval, yval in valid_loader:
                        xval = xval.to(device)
                        yval = yval.to(device)
                        v_nll = model.compute_loss(xval, yval)
                        epoch_valid_loss += v_nll.item()
                epoch_valid_loss /= len(valid_loader)

                if epoch_valid_loss < best_loss:
                    best_loss = epoch_valid_loss
                    best_state_dict = deepcopy(model.state_dict())

                train_losses.append(epoch_train_loss)
                valid_losses.append(epoch_valid_loss)
                logger.info(f"[Freeze-layers] Layer {layer_idx}, epoch {epoch}/{epochs_per_layer} | "
                            f"Train Loss={epoch_train_loss:.4f}, Valid Loss={epoch_valid_loss:.4f}")

        logger.info("Unfreezing all layers for a final joint tune.")
        for name, param in model.named_parameters():
            param.requires_grad = True

        final_epochs = 2
        for epoch in range(1, final_epochs + 1):
            model.train()
            epoch_train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                nll = model.compute_loss(xb, yb)
                self.optimizer.zero_grad()
                nll.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
            epoch_train_loss /= len(train_loader)

            model.eval()
            epoch_valid_loss = 0.0
            with torch.no_grad():
                for xval, yval in valid_loader:
                    xval = xval.to(device)
                    yval = yval.to(device)
                    val_nll = model.compute_loss(xval, yval)
                    epoch_valid_loss += val_nll.item()
            epoch_valid_loss /= len(valid_loader)

            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_state_dict = deepcopy(model.state_dict())

            logger.info(f"[Final Joint] Epoch {epoch}/{final_epochs} "
                        f"| Train Loss={epoch_train_loss:.4f}, Valid Loss={epoch_valid_loss:.4f}")

        model.load_state_dict(best_state_dict)

        if plot_losses and save_path and os.path.exists(save_path):
            plot_NLL_loss_curves(train_losses, valid_losses, 
                                 len(train_losses), 
                                 save_path,
                                 title="Layerwise-Freeze Training Loss Curves")

        return {
            'train losses': train_losses, 
            'valid losses': valid_losses
        }