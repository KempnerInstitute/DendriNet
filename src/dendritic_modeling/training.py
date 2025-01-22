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

    train(model, train_data, valid_data, grad_clip_value=5, epochs=100,
            batch_size=256, shuffle=True, plot_losses=False, save_path=None)
            Trains the model on the training data and validates on the 
            validation data.

    Notes
    -----
      - The model must implement a `log_prob` method that computes the log 
        probability of observing the labels given the inputs.
      - train_data: Data object containing training data.
      - valid_data: Data object containing validation data.


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

        model : ProbabilisticModel
            The model to be trained. Must implement a `log_prob` method that 
            computes the log probability of observing the labels given the 
            inputs.

        train_data : Data
            The training data.

        valid_data : Data
            The validation data.

        grad_clip_value : float, optional
            The value at which to clip the gradients to prevent exploding 
            gradients. The gradients are clipped in the range [-grad_clip_value,
            grad_clip_value]. Default is 5.

        epochs : int, optional
            The number of epochs to train the model. Default is 100.

        batch_size : int, optional
            The batch size used during training. Default is 256.

        shuffle : bool, optional    
            Whether to shuffle the data during training. Default is True.

        plot_losses : bool, optional
            Whether to plot the training and validation losses. 
            Default is False.

        save_path : str, optional
            Directory to save checkpoints and loss curves if provided.

        Returns
        -------

        dict
            A dictionary containing the best epoch, training losses, and
            validation losses. If `plot_losses` is True, the dictionary will
            also contain a figure object for the loss curves and the figure
            will be saved to `save_path` if provided.

        """


        # whether to save at checkpoints
        save_boolean = save_path is not None and os.path.exists(save_path)
        
        # send model to device
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
        
        # create dataloader objects for training data and validation data
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

        logger.debug(f"Requested number of epochs: {epochs}")
        
        for epoch in range(1,epochs+1):        
            train_loss = 0
            valid_loss = 0
            
            model.train()
            # iterate over the training data
            for input_batch, label_batch in train_dataloader:
                try:
                    input_batch = ag.Variable(input_batch.to(device))
                    label_batch = ag.Variable(label_batch.to(device))
                    
                    nll = model.compute_loss(input_batch, label_batch)
                    
                    # zero gradients
                    self.optimizer.zero_grad()
                    # backpropagate loss
                    nll.backward()
                    # prevent exploding gradients
                    
                    clip_grad_value_(model.parameters(), 
                                     clip_value = grad_clip_value)
                    # update weights
                    self.optimizer.step()
                    # aggregate training loss
                    train_loss += nll.item()
                
                except Exception as e:
                    # if NaNs encountered in gradients
                    print(f"Exception: {e}")
                    if not self.suppress:
                        logger.info('exception occurred during parameter update step')
                    model.load_state_dict(best_state_dict)
                    train_loss += 10

            
            # compute mean training loss and save to list
            train_loss /= len(train_dataloader)
            train_losses.append(train_loss)
            
            model.eval()
            with torch.no_grad():
                # iterate over validation data

                all_preds = []
                all_labels = []

                for input_batch, label_batch in valid_dataloader:
                    try:
                        # produce negative log likelihood
                        nll = model.compute_loss(input_batch.to(device), label_batch.to(device))

                        # compute and aggregate validation loss 
                        valid_loss += nll.item()

                        # compute metrics
                        pred  = model.predict(input_batch.to(device))
                        all_preds.append(pred)
                        all_labels.append(label_batch)

                    
                    except Exception as e:
                        print(f"Exception: {e}")
                        if not self.suppress:
                            logger.info('exception occurred during parameter validation step')
                        model.load_state_dict(best_state_dict)
                        valid_loss += 10

                all_preds = torch.cat(all_preds, dim = 0)
                all_labels = torch.cat(all_labels, dim = 0)

                metrics = compute_metrics_classification(all_preds, all_labels)

                if wandb.run is not None:
                    wandb.log(
                        {"val_accuracy": metrics['accuracy'], 
                         "val_precision": metrics['precision'],
                         "val_recall": metrics['recall'],
                         "val_f1_score": metrics['f1_score'],
                         "epoch": epoch}
                    )
            
            # compute mean validation loss and save to list
            valid_loss /= len(valid_dataloader)
            valid_losses.append(valid_loss)

            
            # save model that performs best on validation data
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_epoch = epoch
                best_state_dict = deepcopy(model.state_dict())

                if save_boolean:
                    torch.save(best_state_dict, save_path+'/state_dict.pt')

                    with open(save_path+'/losses.json', 'w') as f:
                        json.dump(
                            {'best epoch' : best_epoch, 
                             'train losses' : train_losses, 
                             'valid losses': valid_losses}, 
                             f
                        )

            
            if not self.suppress:
                # printing
                if epoch % self.print_every == 0:
                    logger.info(f' ----   Epoch {epoch}  ---- ')
                    logger.info('training loss: %.2f | validation loss: %.2f' % (
                        train_loss, valid_loss))
                    
                    time_elapsed = (time.time() - stime)
                    pred_time_remaining = (time_elapsed / epoch) * (epochs-epoch)
                    
                    logger.info('time elapsed: %.2f s | predicted time remaining: %.2f s' % (
                        time_elapsed, pred_time_remaining))
                    
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
        
        if save_boolean:
            with open(save_path+'/losses.json', 'w') as f:
                json.dump(results, f)

        # plot loss curves
        if plot_losses:
            if save_boolean:
                plot_NLL_loss_curves(
                    train_losses, 
                    valid_losses, 
                    epochs,
                    save_path,
                )
            else:
                results['loss curves'] = plot_NLL_loss_curves(
                    train_losses, valid_losses, epochs,
                )
            
        return results