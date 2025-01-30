"""
learning_strategies.py
======================
A single file that unifies various training strategies:

    - TrainerMLE
    - Freeze-layers (in TrainerMLE.train_freeze_layers)
    - LocalCreditAssignment
    - TrainOnlyBReactivation
    - TwoStepTrainer

Use 'get_trainer(strategy, ...)' to construct the desired approach
based on a config's "learning_strategy" key.

List of recognized strategies for config:
  "mle"                   -> standard TrainerMLE
  "freeze_layers"         -> same TrainerMLE, user calls train_freeze_layers
  "local_credit_assignment"-> the LocalCreditAssignment class
  "train_only_b_react"    -> the TrainOnlyBReactivation class
  "two_step"              -> the TwoStepTrainer class
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
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_

from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves

def compute_metrics_classification(preds, labels):
    """
    Computes common classification metrics. 
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


class TrainerMLE:
    """
    A trainer that performs a standard maximum-likelihood (NLL) approach.
    Also includes a freeze-layers approach in the same class for convenience.
    """
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        """
        Parameters
        ----------
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
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=shuffle)

        train_losses = []
        valid_losses = []

        stime = time.time()
        best_loss = float('inf')
        best_state_dict = deepcopy(model.state_dict())
        best_epoch = 1

        # We'll get the last branch if it exists (for conduction hooking)
        last_branch = None
        if hasattr(model, 'net') and hasattr(model.net, 'branch_layers'):
            if len(model.net.branch_layers) > 0:
                last_branch = model.net.branch_layers[-1]

        for epoch in range(1, epochs+1):
            train_loss = 0.0
            valid_loss = 0.0
            model.train()

            scale_sum = 0.0
            scale_count = 0

            for input_batch, label_batch in train_loader:
                input_batch = input_batch.to(device)
                label_batch = label_batch.to(device)

                nll = model.compute_loss(input_batch, label_batch)
                self.optimizer.zero_grad()
                nll.backward()

                # conduction hooking
                if last_branch and hasattr(last_branch, 'branches_to_output'):
                    block = last_branch.branches_to_output
                    if hasattr(block, 'last_scale_vec'):
                        mean_val = block.last_scale_vec.mean().item()
                        scale_sum += mean_val
                        scale_count += 1

                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                train_loss += nll.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # validate
            model.eval()
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for input_batch, label_batch in valid_loader:
                    input_batch = input_batch.to(device)
                    label_batch = label_batch.to(device)
                    nll = model.compute_loss(input_batch, label_batch)
                    valid_loss += nll.item()
                    preds = model.predict(input_batch)
                    all_preds.append(preds.cpu())
                    all_labels.append(label_batch.cpu())

            valid_loss /= len(valid_loader)
            valid_losses.append(valid_loss)

            # metrics
            if len(all_preds) > 0:
                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                metrics = compute_metrics_classification(all_preds, all_labels)
                if wandb.run is not None:
                    wandb.log({
                        "val_accuracy": metrics['accuracy'],
                        "val_precision": metrics['precision'],
                        "val_recall": metrics['recall'],
                        "val_f1_score": metrics['f1_score'],
                        "epoch": epoch
                    })

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                if save_path and os.path.exists(save_path):
                    torch.save(best_state_dict, os.path.join(save_path, 'state_dict.pt'))
                    with open(os.path.join(save_path, 'losses.json'), 'w') as f:
                        json.dump({
                            'best epoch': best_epoch,
                            'train losses': train_losses,
                            'valid losses': valid_losses
                        }, f)

            if not self.suppress and epoch % self.print_every == 0:
                logger.info(f' ---- Epoch {epoch} ---- ')
                logger.info(f'training loss: {train_loss:.2f} | validation loss: {valid_loss:.2f}')
                elapsed = time.time() - stime
                pred_remaining = (elapsed / epoch) * (epochs - epoch)
                logger.info(f'time elapsed: {elapsed:.2f}s | predicted time remaining: {pred_remaining:.2f}s')

            # wandb logging
            if wandb.run is not None:
                wandb_dict = {
                    "train_loss": float(train_loss),
                    "valid_loss": float(valid_loss),
                    "epoch": epoch
                }
                if scale_count > 0:
                    wandb_dict["grad_scale_vec_mean"] = scale_sum / scale_count
                wandb.log(wandb_dict)

        results = {
            'best epoch': best_epoch,
            'train losses': train_losses,
            'valid losses': valid_losses
        }
        if load_best_state_dict:
            model.load_state_dict(best_state_dict)
        else:
            results['best state dict'] = best_state_dict

        if plot_losses and save_path and os.path.exists(save_path):
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
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

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
            for name, param in model.named_parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}." in name:
                    param.requires_grad = True

            logger.info(f"Training only layer {layer_idx}, freezing others.")

            for epoch in range(1, epochs_per_layer + 1):
                model.train()
                epoch_train_loss = 0
                for x_batch, y_batch in train_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    nll = model.compute_loss(x_batch, y_batch)
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()
                    epoch_train_loss += nll.item()

                epoch_train_loss /= len(train_loader)
                model.eval()
                epoch_valid_loss = 0
                with torch.no_grad():
                    for x_val, y_val in valid_loader:
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)
                        val_nll = model.compute_loss(x_val, y_val)
                        epoch_valid_loss += val_nll.item()
                epoch_valid_loss /= len(valid_loader)

                if epoch_valid_loss < best_loss:
                    best_loss = epoch_valid_loss
                    best_state_dict = deepcopy(model.state_dict())

                train_losses.append(epoch_train_loss)
                valid_losses.append(epoch_valid_loss)
                logger.info(f"Layer {layer_idx} | Epoch {epoch}/{epochs_per_layer} "
                            f"| Train Loss={epoch_train_loss:.4f}, Valid Loss={epoch_valid_loss:.4f}")

        logger.info("Unfreezing all layers for a final joint tune.")
        for name, param in model.named_parameters():
            param.requires_grad = True

        final_epochs = 2
        for epoch in range(1, final_epochs + 1):
            model.train()
            epoch_train_loss = 0
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                nll = model.compute_loss(x_batch, y_batch)
                self.optimizer.zero_grad()
                nll.backward()
                clip_grad_value_(model.parameters(), grad_clip_value)
                self.optimizer.step()
                epoch_train_loss += nll.item()
            epoch_train_loss /= len(train_loader)

            model.eval()
            epoch_valid_loss = 0
            with torch.no_grad():
                for x_val, y_val in valid_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    val_nll = model.compute_loss(x_val, y_val)
                    epoch_valid_loss += val_nll.item()
            epoch_valid_loss /= len(valid_loader)
            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_state_dict = deepcopy(model.state_dict())

            logger.info(f"[Final Joint] Epoch {epoch}/{final_epochs} | "
                        f"Train Loss={epoch_train_loss:.4f}, Valid Loss={epoch_valid_loss:.4f}")

        model.load_state_dict(best_state_dict)
        return {'train losses': train_losses, 'valid losses': valid_losses}


class LocalCreditAssignment:
    """
    Example class for local credit assignment approach.
    This might differ from MLE in how gradients or local updates are computed.

    Usage:
        trainer = LocalCreditAssignment(optimizer=...)
        trainer.train(model, train_data, valid_data, ...)

    Make sure your model or relevant modules handle partial local updates.
    """
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every

    def train(
        self,
        model,
        train_data,
        valid_data,
        # add any local-credit parameters
        grad_clip_value=5,
        epochs=100,
        batch_size=256,
        shuffle=True,
        # ...
    ):
        """
        Suppose this does some local-forward computations or partial gradient approach.
        We'll do a simplified version here.
        """
        # Implementation details would go here.
        # You might do a "local_loss" or partial backward approach.
        logger.info("LocalCreditAssignment: Not fully implemented. Example placeholder.")
        # Return a placeholder
        return {}


class TrainOnlyBReactivation:
    """
    A strategy that trains only the 'b' parameter in param_tanh reactivation 
    or partial submodule, ignoring other parameters, etc.
    """
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every

    def train(
        self,
        model,
        train_data,
        valid_data,
        # ...
        grad_clip_value=5,
        epochs=100,
        batch_size=256,
        shuffle=True,
    ):
        """
        Possibly freeze everything except the 'b' parameter in param_tanh.
        Example placeholder code.
        """
        logger.info("TrainOnlyBReactivation: training only 'b' parameters in reactivation.")
        # Implementation placeholder
        return {}


class TwoStepTrainer:
    """
    A two-step trainer approach. Possibly does a 'pretrain' step, 
    then a main training step, etc.
    """
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every

    def train(
        self,
        model,
        train_data,
        valid_data,
        # ...
        pretrain_epochs=5,
        maintrain_epochs=45,
        batch_size=256,
        grad_clip_value=5,
        shuffle=True,
    ):
        """
        Example two-step approach:
          1) do pre-training for 'pretrain_epochs'
          2) do main training for 'maintrain_epochs'
        """
        logger.info("TwoStepTrainer: performing pretrain, then main train.")
        # placeholder logic
        # 1) freeze some parts, do pretrain
        # 2) unfreeze, do main train
        return {}



def get_trainer(strategy, optimizer, suppress_prints=False, print_every=10):
    """
    A function to select from multiple training strategies:

    - "mle"                   -> standard TrainerMLE
    - "freeze_layers"         -> also TrainerMLE, but user calls train_freeze_layers separately
    - "local_credit_assignment"-> LocalCreditAssignment
    - "train_only_b_react"    -> TrainOnlyBReactivation
    - "two_step"              -> TwoStepTrainer

    """
    if strategy == "mle":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_layers":
        # user will call trainer.train_freeze_layers(...) themselves
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "local_credit_assignment":
        return LocalCreditAssignment(optimizer, suppress_prints, print_every)
    elif strategy == "train_only_b_react":
        return TrainOnlyBReactivation(optimizer, suppress_prints, print_every)
    elif strategy == "two_step":
        return TwoStepTrainer(optimizer, suppress_prints, print_every)
    else:
        logger.info(f"Unknown strategy '{strategy}', defaulting to 'mle'.")
        return TrainerMLE(optimizer, suppress_prints, print_every)