"""
learning_strategies.py
======================
This module defines different learning rules and strategies, such as
local credit assignment, freeze-reactivation-m, and a two-step training
(for initialization + real training).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


class LocalCreditAssignment:
    """
    Example local learning rule that attempts to estimate updates locally
    via some feedback alignment or target-propagation approach.
    """

    def __init__(self, feedback_mats, local_loss_weight=0.0):
        """
        feedback_mats : list of nn.Parameter or None
            Each entry can be a feedback matrix for a layer.
        local_loss_weight : float
            How much weight to give to the local objective.
        """
        self.feedback_mats = feedback_mats
        self.local_loss_weight = local_loss_weight

    def compute_local_loss(self, layer_idx, excitatory_output, y_onehot):
        """
        Example local loss function: L = 0.5 * ||excitatory - (y_onehot * FB^T)||^2
        """
        fb = self.feedback_mats[layer_idx]
        if fb is None:
            return 0.0
        desired_acts = y_onehot @ fb.T
        local_obj = 0.5 * ((excitatory_output - desired_acts)**2).mean()
        return local_obj

    def apply_local_learning(self, model, x, y):
        """
        Aggregates local losses across layers. This can be done
        in a forward pass that returns excitatory outputs per layer.
        """
        if y is None:
            return 0.0

        device = next(model.parameters()).device
        x = x.to(device)
        y = y.to(device)
        # Assume # of classes = 10 for example
        y_onehot = torch.zeros(y.size(0), 10, device=device).scatter_(1, y.unsqueeze(1), 1.0)

        local_losses = []
        excitatory = x
        inhibitory = None

        for i, layer in enumerate(model.layers):
            excitatory, inhibitory = layer(excitatory, inhibitory)
            # compute local loss
            local_obj = self.compute_local_loss(i, excitatory, y_onehot)
            if local_obj != 0.0:
                local_losses.append(local_obj)

        if len(local_losses) == 0:
            return 0.0

        return self.local_loss_weight * torch.stack(local_losses).mean()


class TrainOnlyBReactivation:
    """
    A helper that, given a reactivation layer, can freeze log_m (m) so that only b is trained.
    """

    def __init__(self, freeze_m=True):
        """
        freeze_m : bool
            If True, we freeze the reactivation m parameter (log_m).
        """
        self.freeze_m = freeze_m

    def apply(self, reactivation_layer):
        """
        If freeze_m is True, set log_m.requires_grad=False.
        """
        if self.freeze_m:
            reactivation_layer.log_m.requires_grad = False
        else:
            reactivation_layer.log_m.requires_grad = True


class TwoStepTrainer:
    """
    Implements a two-stage training:
      1) Pre-train the network with a specialized cost to ensure
         better gradient flow (initialization).
      2) Main training with the real objective (e.g. classification).
    """

    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self._best_state_dict = None

    def pretrain(self, dataloader, epochs=5, grad_clip=5.0):
        """
        Stage 1: use an 'alignment' or 'layer-wise' cost to help
        initialization. For demonstration, we define a simple MSE
        between each layer's output and a one-hot label.

        If the model is not a simple feedforward but a DendriNet,
        we might adapt or use 'DendriNetWithOutputs' to get
        per-layer outputs.
        """
        self.model.to(self.device)
        best_loss = float('inf')
        self._best_state_dict = deepcopy(self.model.state_dict())

        for epoch in range(1, epochs+1):
            epoch_loss = 0.0
            self.model.train()

            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                y_onehot = torch.zeros(y_batch.size(0), 10, device=self.device).scatter_(1, y_batch.unsqueeze(1), 1.0)

                # capture outputs at each layer
                layer_outputs = []
                excitatory = x_batch
                inhibitory = None
                for i, layer in enumerate(self.model.layers):
                    excitatory, inhibitory = layer(excitatory, inhibitory)
                    layer_outputs.append(excitatory)

                # sum of MSE across layers to the one-hot
                losses = []
                for out in layer_outputs:
                    losses.append(F.mse_loss(out, y_onehot))

                pretrain_loss = torch.stack(losses).mean()

                self.optimizer.zero_grad()
                pretrain_loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)
                self.optimizer.step()

                epoch_loss += pretrain_loss.item()

            epoch_loss /= len(dataloader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self._best_state_dict = deepcopy(self.model.state_dict())

    def finalize_pretrain(self, load_best=True):
        """
        After pretraining, optionally load the best state dict.
        """
        if load_best and self._best_state_dict is not None:
            self.model.load_state_dict(self._best_state_dict)

    def train_main(self, dataloader, loss_fn, epochs=5, grad_clip=5.0):
        """
        Stage 2: Standard training with the real objective, e.g.
        cross-entropy or NLL for classification.
        """
        self.model.to(self.device)
        for epoch in range(1, epochs+1):
            epoch_loss = 0.0
            self.model.train()
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                out = self.model(x_batch)
                main_loss = loss_fn(out, y_batch)

                self.optimizer.zero_grad()
                main_loss.backward()
                if grad_clip is not None:
                    nn.utils.clip_grad_value_(self.model.parameters(), grad_clip)
                self.optimizer.step()

                epoch_loss += main_loss.item()

            epoch_loss /= len(dataloader)
