"""
training.py
===========
This module contains the TrainerMLE class for training the models.
"""

import os
import json
import time
from copy import deepcopy

import wandb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             f1_score)

import torch
import torch.autograd as ag
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_

from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves


class CustomWeightDecayOptimizer(object):
    """
    A simple wrapper that applies custom weight decay
    by calling model.decay_weights(...).
    """

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


def compute_metrics_classification(preds, labels):
    """
    Compute classification metrics (accuracy, precision, recall, f1).
    """
    assert preds.shape == labels.shape

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    metrics  = {
        'accuracy' : accuracy_score(labels, preds),
        'precision' : precision_score(labels, preds, average = 'weighted'),
        'recall' : recall_score(labels, preds, average = 'weighted'),
        'f1_score' : f1_score(labels, preds, average = 'weighted')
    }

    return metrics


class TrainerMLE(object):
    """
    Maximum Likelihood Estimation (MLE) trainer for training probabilistic 
    models. 

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer used to update the model parameters during training.

    suppress_prints : bool, optional
        Whether to suppress print statements during training. Default is False.

    print_every : int, optional
        The frequency with which to print training updates. Default is 10.

    Methods
    -------
    train(...)
        Standard training loop: runs backprop on all layers at once.

    train_freeze_layers(...)
        Optional method: freeze/unfreeze layers in phases to mitigate vanishing gradients.
    """
    
    def __init__(self, optimizer, suppress_prints = False, print_every = 10):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every
    

    def train(
            self, 
            model, 
            train_data, 
            valid_data, 
            grad_clip_value = 5,
            epochs = 100, 
            batch_size = 256, 
            shuffle = True, 
            load_best_state_dict = True,
            plot_losses = False, 
            save_path = None,
            ):
        """
        Trains a model using maximum likelihood estimation.

        Parameters
        ----------
        model : ProbabilisticModel or Classifier
            The model to be trained.

        train_data : torch.utils.data.Dataset
            The training data.

        valid_data : torch.utils.data.Dataset
            The validation data.

        grad_clip_value : float, optional
            The value at which to clip gradients. Default is 5.

        epochs : int, optional
            Number of epochs to train. Default is 100.

        batch_size : int, optional
            Default is 256.

        shuffle : bool, optional
            Shuffle data. Default True.

        plot_losses : bool, optional
            Plot train & valid losses. Default False.

        save_path : str, optional
            Directory for checkpoints & plots.

        Returns
        -------
        dict of training info
        """

        save_boolean = save_path is not None and os.path.exists(save_path)
        
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_properties = torch.cuda.get_device_properties(device)
            logger.debug(f"Using GPU: {gpu_properties.name}")
            logger.debug(f"GPU Memory: {gpu_properties.total_memory / 1024 / 1024 / 1024:.0f} GB")
            logger.debug(f"GPU Compute Capability: {gpu_properties.major}.{gpu_properties.minor}")
            logger.debug(f"GPU multiprocessor count: {gpu_properties.multi_processor_count}")
        else:
            device = 'cpu'
            logger.debug("Using CPU Device for Training.") 

        model = model.to(device)
        
        # create dataloader objects
        train_dataloader = DataLoader(train_data, 
                                      batch_size = batch_size, 
                                      shuffle = shuffle)
        valid_dataloader = DataLoader(valid_data, 
                                      batch_size = batch_size, 
                                      shuffle = shuffle)
        
        train_losses = []
        valid_losses = []
        
        stime = time.time()
        
        best_loss = torch.tensor(float('inf'))
        best_state_dict = deepcopy(model.state_dict())
        best_epoch = 1

        logger.debug(f"Requested number of epochs: {epochs}")
        
        for epoch in range(1, epochs + 1):        
            train_loss = 0
            valid_loss = 0
            
            model.train()
            for input_batch, label_batch in train_dataloader:
                try:
                    input_batch = ag.Variable(input_batch.to(device))
                    label_batch = ag.Variable(label_batch.to(device))
                    
                    nll = model.compute_loss(input_batch, label_batch)
                    
                    self.optimizer.zero_grad()
                    nll.backward()
                    clip_grad_value_(model.parameters(), grad_clip_value)
                    self.optimizer.step()

                    train_loss += nll.item()
                
                except Exception as e:
                    print(f"Exception: {e}")
                    if not self.suppress:
                        logger.info('exception occurred during parameter update step')
                    model.load_state_dict(best_state_dict)
                    train_loss += 10

            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                all_preds = []
                all_labels = []

                for input_batch, label_batch in valid_dataloader:
                    try:
                        nll = model.compute_loss(input_batch.to(device), label_batch.to(device))
                        valid_loss += nll.item()

                        pred = model.predict(input_batch.to(device))
                        all_preds.append(pred)
                        all_labels.append(label_batch)

                    except Exception as e:
                        print(f"Exception: {e}")
                        if not self.suppress:
                            logger.info('exception occurred during parameter validation step')
                        model.load_state_dict(best_state_dict)
                        valid_loss += 10

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

            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                if save_boolean:
                    torch.save(best_state_dict, os.path.join(save_path, 'state_dict.pt'))
                    with open(os.path.join(save_path, 'losses.json'), 'w') as f:
                        json.dump({
                            'best epoch': best_epoch,
                            'train losses': train_losses,
                            'valid losses': valid_losses
                        }, f)

            if not self.suppress and epoch % self.print_every == 0:
                logger.info(f' ----   Epoch {epoch}  ---- ')
                logger.info('training loss: %.2f | validation loss: %.2f' % (
                    train_loss, valid_loss))
                
                time_elapsed = (time.time() - stime)
                pred_time_remaining = (time_elapsed / epoch) * (epochs - epoch)
                
                logger.info('time elapsed: %.2f s | predicted time remaining: %.2f s' % (
                    time_elapsed, pred_time_remaining))

            if wandb.run is not None:
                wandb.log({"train_loss": float(train_loss), 
                           "valid_loss": float(valid_loss),
                           "epoch": epoch})

        results = {
            'best epoch': best_epoch,
            'train losses': train_losses,
            'valid losses': valid_losses
        }

        if load_best_state_dict:
            model.load_state_dict(best_state_dict)
        else:
            results['best state dict'] = best_state_dict

        # optional plotting
        if plot_losses:
            if save_boolean:
                plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
            else:
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
        A simple approach to 'freeze' layers in sequence, mitigating
        vanishing gradients in deep dendritic models.

          1) Identify the number of layers in model.net.layers
          2) For each layer, freeze everything except that layer
          3) Train for 'epochs_per_layer'
          4) Move to the next layer
          5) Optionally unfreeze all for a final pass

        Minimal example, can be extended or refined.
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        # simple assumption: ExcitationInhibitionNetwork has .layers
        if not hasattr(model, 'net') or not hasattr(model.net, 'layers'):
            logger.info("Model has no 'net.layers' to freeze. Doing normal train.")
            return self.train(
                model,
                train_data,
                valid_data,
                grad_clip_value=grad_clip_value,
                epochs=epochs_per_layer,  # or some default
                batch_size=batch_size,
                shuffle=shuffle,
                plot_losses=plot_losses,
                save_path=save_path
            )

        n_layers = len(model.net.layers)
        logger.info(f"Freezing/unfreezing approach: found {n_layers} layers in model.net.layers.")

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

        train_losses = []
        valid_losses = []

        for layer_idx in range(n_layers):
            # 1) Freeze everything
            for name, param in model.named_parameters():
                param.requires_grad = False

            # 2) Unfreeze just this layer
            for name, param in model.named_parameters():
                if f"layers.{layer_idx}." in name:
                    param.requires_grad = True

            logger.info(f"Training only layer {layer_idx}, freezing others.")

            # 3) Train for epochs_per_layer
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
                train_losses.append(epoch_train_loss)

                # validation
                model.eval()
                epoch_valid_loss = 0
                with torch.no_grad():
                    for x_val, y_val in valid_loader:
                        x_val = x_val.to(device)
                        y_val = y_val.to(device)

                        val_nll = model.compute_loss(x_val, y_val)
                        epoch_valid_loss += val_nll.item()

                epoch_valid_loss /= len(valid_loader)
                valid_losses.append(epoch_valid_loss)

                logger.info(f"Layer {layer_idx} | Epoch {epoch}/{epochs_per_layer} | "
                            f"Train Loss={epoch_train_loss:.4f}, Valid Loss={epoch_valid_loss:.4f}")

        # Optionally unfreeze all layers for a final short training
        logger.info("Unfreezing all layers for a final joint tune.")
        for name, param in model.named_parameters():
            param.requires_grad = True

        # Let's do 1 or 2 epochs final tune
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
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_nll = model.compute_loss(x_val, y_val)
                    epoch_valid_loss += val_nll.item()
            epoch_valid_loss /= len(valid_loader)

            logger.info(f"[Final Joint] Epoch {epoch}/{final_epochs} | "
                        f"Train Loss={epoch_train_loss:.4f}, Valid Loss={epoch_valid_loss:.4f}")


        if plot_losses and save_path and os.path.exists(save_path):
            plot_NLL_loss_curves(
                train_losses, 
                valid_losses, 
                len(train_losses),  # total steps
                save_path,
                title="Layerwise-Freeze Training Loss Curves"
            )

        return {
            'train losses': train_losses,
            'valid losses': valid_losses
        }