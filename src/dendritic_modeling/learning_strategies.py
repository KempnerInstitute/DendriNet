"""
learning_strategies.py
======================
Houses multiple learning/training strategies:

    - TrainerMLE            (classic NLL backprop)
        * includes train_freeze_layers for freeze/unfreeze approach
    - LocalCreditAssignment (placeholder approach)
    - TrainOnlyBReactivation (placeholder approach)
    - TwoStepTrainer        (placeholder for pretrain + maintrain logic)

A factory function get_trainer(...) returns the chosen strategy 
based on a string (e.g. "mle", "freeze_layers", "local_credit_assignment", 
"train_only_b_react", or "two_step").
"""

import time
import json
import wandb
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_

from dendritic_modeling import logger
from dendritic_modeling.plot_utils import plot_NLL_loss_curves

def compute_metrics_classification(preds, labels):
    """
    Computes classification metrics (accuracy, etc.).
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
    A standard trainer class that performs maximum-likelihood (NLL) training.

    Also includes a 'train_freeze_layers' method for freeze/unfreeze logic.
    """

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
        valid_data=None,
        grad_clip_value=5,
        epochs=100,
        batch_size=256,
        shuffle=True,
        load_best_state_dict=True,
        plot_losses=False,
        save_path=None,
    ):
        """
        Standard NLL training loop. 
        If valid_data is None, we skip validation steps.
        """
        from torch.utils.data import DataLoader

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False) if valid_data else None

        train_losses = []
        valid_losses = []
        best_loss = float('inf')
        best_epoch = 1
        best_state = deepcopy(model.state_dict())

        stime = time.time()

        for epoch in range(1, epochs+1):
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
            train_losses.append(epoch_train_loss)

            # Validation
            epoch_valid_loss = 0.0
            if valid_loader:
                model.eval()
                with torch.no_grad():
                    for xval, yval in valid_loader:
                        xval = xval.to(device)
                        yval = yval.to(device)
                        v_nll = model.compute_loss(xval, yval)
                        epoch_valid_loss += v_nll.item()
                epoch_valid_loss /= len(valid_loader)
            else:
                epoch_valid_loss = epoch_train_loss  # or 0
            valid_losses.append(epoch_valid_loss)

            # Check best
            if epoch_valid_loss < best_loss:
                best_loss = epoch_valid_loss
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                if save_path and (epoch_valid_loss < 1e5): # sanity check
                    torch.save(best_state, f"{save_path}/best_model.pt")

            if not self.suppress and epoch % self.print_every == 0:
                logger.info(f"Epoch {epoch}/{epochs}: train_loss={epoch_train_loss:.4f}, valid_loss={epoch_valid_loss:.4f}")
                elapsed = time.time() - stime
                logger.info(f"Elapsed time: {elapsed:.1f}s")

            # wandb logging
            if wandb.run is not None:
                log_dict = {
                    "train_loss": epoch_train_loss,
                    "valid_loss": epoch_valid_loss,
                    "epoch": epoch
                }
                wandb.log(log_dict)

        if load_best_state_dict:
            model.load_state_dict(best_state)

        results = {
            'best epoch': best_epoch,
            'train losses': train_losses,
            'valid losses': valid_losses,
        }
        if plot_losses:
            if save_path:
                plot_NLL_loss_curves(train_losses, valid_losses, epochs, save_path)
            else:
                results["loss curves"] = plot_NLL_loss_curves(train_losses, valid_losses, epochs)
        return results

    def train_freeze_layers(
        self,
        model,
        train_data,
        valid_data=None,
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
        from torch.utils.data import DataLoader

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
        valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False) if valid_data else None

        if not hasattr(model, 'net') or not hasattr(model.net, 'layers'):
            logger.info("Model has no 'net.layers'. Falling back to normal train.")
            return self.train(
                model, train_data, valid_data, grad_clip_value=grad_clip_value,
                epochs=epochs_per_layer, batch_size=batch_size,
                shuffle=shuffle, plot_losses=plot_losses, save_path=save_path
            )

        n_layers = len(model.net.layers)
        logger.info(f"train_freeze_layers: found {n_layers} layers. Freezing all but one in sequence.")
        train_losses = []
        valid_losses = []
        best_loss = float('inf')
        best_state = deepcopy(model.state_dict())

        for layer_idx in range(n_layers):
            # Freeze everything
            for name, param in model.named_parameters():
                param.requires_grad = False

            # Unfreeze only this layer
            target_str = f"layers.{layer_idx}."
            for name, param in model.named_parameters():
                if target_str in name:
                    param.requires_grad = True

            logger.info(f"Training only layer {layer_idx} for {epochs_per_layer} epochs.")

            # standard loop
            for epoch in range(1, epochs_per_layer+1):
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
                train_losses.append(epoch_train_loss)

                epoch_valid_loss = 0.0
                if valid_loader:
                    model.eval()
                    with torch.no_grad():
                        for xval, yval in valid_loader:
                            xval = xval.to(device)
                            yval = yval.to(device)
                            v_nll = model.compute_loss(xval, yval)
                            epoch_valid_loss += v_nll.item()
                    epoch_valid_loss /= len(valid_loader)
                else:
                    epoch_valid_loss = epoch_train_loss

                valid_losses.append(epoch_valid_loss)

                if epoch_valid_loss < best_loss:
                    best_loss = epoch_valid_loss
                    best_state = deepcopy(model.state_dict())

                logger.info(f"[Freeze-layers] Layer {layer_idx}, epoch {epoch}/{epochs_per_layer}, train_loss={epoch_train_loss:.4f}, valid_loss={epoch_valid_loss:.4f}")

        # final unfreeze all
        logger.info("Unfreezing all layers for a final short tune.")
        for name, param in model.named_parameters():
            param.requires_grad = True
        model.load_state_dict(best_state)

        results = {
            'train losses': train_losses,
            'valid losses': valid_losses
        }
        if plot_losses:
            if save_path:
                plot_NLL_loss_curves(train_losses, valid_losses, epochs_per_layer*n_layers, save_path, title="Layerwise Freeze Training")
            else:
                results['loss curves'] = plot_NLL_loss_curves(train_losses, valid_losses, epochs_per_layer*n_layers)

        return results


class LocalCreditAssignment:
    """
    An alternative strategy that uses local learning rules or partial credit. 
    Placeholder example.
    """
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every

    def train(self, model, train_data, valid_data=None, **kwargs):
        logger.info("LocalCreditAssignment: Not implemented fully. Placeholder.")
        # You might do partial local gradients or a local loss approach here.
        return {}


class TrainOnlyBReactivation:
    """
    A strategy that only trains the 'b' offset in param_tanh reactivation layers 
    while freezing everything else. Placeholder example.
    """
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every

    def train(self, model, train_data, valid_data=None, **kwargs):
        logger.info("TrainOnlyBReactivation: Freezing everything except 'b' in param_tanh.")
        # Implementation would freeze all except param_tanh.log_b 
        return {}


class TwoStepTrainer:
    """
    A two-step approach. Possibly does a 'pretrain' with certain modules frozen,
    then a 'main train' unfreezing the entire network.
    """
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        self.optimizer = optimizer
        self.suppress = suppress_prints
        self.print_every = print_every

    def train(
        self,
        model,
        train_data,
        valid_data=None,
        pretrain_epochs=5,
        maintrain_epochs=45,
        batch_size=256,
        grad_clip_value=5,
        # ...
    ):
        logger.info("TwoStepTrainer: pretrain, then main train. Placeholder example.")
        # Step 1: freeze something, run for pretrain_epochs
        # Step 2: unfreeze all, run for maintrain_epochs
        return {}


def get_trainer(strategy, optimizer, suppress_prints=False, print_every=10):
    """
    A simple factory function that returns an instance 
    of the appropriate trainer class, given a 'strategy' string.
    Options:
      - 'mle' -> TrainerMLE
      - 'freeze_layers' -> also TrainerMLE, user calls train_freeze_layers
      - 'local_credit_assignment' -> LocalCreditAssignment
      - 'train_only_b_react' -> TrainOnlyBReactivation
      - 'two_step' -> TwoStepTrainer
    """
    if strategy == "mle":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "freeze_layers":
        return TrainerMLE(optimizer, suppress_prints, print_every)
    elif strategy == "local_credit_assignment":
        return LocalCreditAssignment(optimizer, suppress_prints, print_every)
    elif strategy == "train_only_b_react":
        return TrainOnlyBReactivation(optimizer, suppress_prints, print_every)
    elif strategy == "two_step":
        return TwoStepTrainer(optimizer, suppress_prints, print_every)
    else:
        logger.info(f"Unrecognized strategy '{strategy}' - defaulting to 'mle'.")
        return TrainerMLE(optimizer, suppress_prints, print_every)