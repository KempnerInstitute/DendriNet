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
        save_boolean = save_path is not None and os.path.exists(save_path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = shuffle)
        valid_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle = shuffle)
        
        train_losses = []
        valid_losses = []
        
        stime = time.time()
        best_loss = float('inf')
        best_state_dict = deepcopy(model.state_dict())
        best_epoch = 1

        for epoch in range(1,epochs+1):        
            train_loss = 0
            valid_loss = 0
            
            model.train()
            for input_batch, label_batch in train_dataloader:
                try:
                    input_batch = input_batch.to(device)
                    label_batch = label_batch.to(device)
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
            all_preds = []
            all_labels = []
            with torch.no_grad():
                for input_batch, label_batch in valid_dataloader:
                    try:
                        input_batch = input_batch.to(device)
                        label_batch = label_batch.to(device)
                        nll = model.compute_loss(input_batch, label_batch)
                        valid_loss += nll.item()
                        pred  = model.predict(input_batch)
                        all_preds.append(pred.cpu())
                        all_labels.append(label_batch.cpu())
                    except Exception as e:
                        print(f"Exception: {e}")
                        if not self.suppress:
                            logger.info('exception occurred during parameter validation step')
                        model.load_state_dict(best_state_dict)
                        valid_loss += 10

            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)

            if len(all_preds) > 0:
                all_preds = torch.cat(all_preds, dim=0)
                all_labels = torch.cat(all_labels, dim=0)
                metrics = compute_metrics_classification(all_preds, all_labels)
                if wandb.run is not None:
                    wandb.log(
                        {"val_accuracy": metrics['accuracy'], 
                         "val_precision": metrics['precision'],
                         "val_recall": metrics['recall'],
                         "val_f1_score": metrics['f1_score'],
                         "epoch": epoch}
                    )
            
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())
                if save_boolean:
                    torch.save(best_state_dict, os.path.join(save_path,'state_dict.pt'))
                    with open(os.path.join(save_path,'losses.json'), 'w') as f:
                        json.dump({'best epoch' : best_epoch, 'train losses' : train_losses, 'valid losses': valid_losses}, f)

            if not self.suppress:
                if epoch % self.print_every == 0:
                    logger.info(f' ----   Epoch {epoch}  ---- ')
                    logger.info('training loss: %.2f | validation loss: %.2f' % (train_loss, valid_loss))
                    time_elapsed = (time.time() - stime)
                    pred_time_remaining = (time_elapsed / epoch) * (epochs-epoch)
                    logger.info('time elapsed: %.2f s | predicted time remaining: %.2f s' % (time_elapsed, pred_time_remaining))
                    
            if wandb.run is not None:
                wandb.log({"train_loss": float(train_loss), 
                           "valid_loss": float(valid_loss),
                           "epoch": epoch})

        results = {
            'best epoch' : best_epoch, 
            'train losses' : train_losses, 
            'valid losses' : valid_losses,
        }
        if load_best_state_dict:
            model.load_state_dict(best_state_dict)
        else:
            results['best state dict'] = best_state_dict
        
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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        if not hasattr(model, 'net') or not hasattr(model.net, 'layers'):
            logger.info('Model has no "net.layers" to freeze. Doing normal train.')
            return self.train(model, train_data, valid_data, grad_clip_value=grad_clip_value, epochs=epochs_per_layer, batch_size=batch_size, shuffle=shuffle, plot_losses=plot_losses, save_path=save_path)

        n_layers = len(model.net.layers)
        logger.info(f"Freezing/unfreezing approach: found {n_layers} layers in model.net.layers.")

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

        train_losses = []
        valid_losses = []

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
                train_losses.append(epoch_train_loss)

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
                logger.info(f"Layer {layer_idx} | Epoch {epoch}/{epochs_per_layer} | Train Loss={epoch_train_loss:.4f}, Valid Loss={epoch_valid_loss:.4f}")

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
                    x_val, y_val = x_val.to(device), y_val.to(device)
                    val_nll = model.compute_loss(x_val, y_val)
                    epoch_valid_loss += val_nll.item()
            epoch_valid_loss /= len(valid_loader)
            logger.info(f"[Final Joint] Epoch {epoch}/{final_epochs} | Train Loss={epoch_train_loss:.4f}, Valid Loss={epoch_valid_loss:.4f}")

        if plot_losses and save_path and os.path.exists(save_path):
            plot_NLL_loss_curves(train_losses, valid_losses, len(train_losses), save_path, title="Layerwise-Freeze Training Loss Curves")

        return {
            'train losses': train_losses,
            'valid losses': valid_losses
        }