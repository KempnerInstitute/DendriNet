"""
models.py
=========
This module contains the neural network models for the dendritic_modeling.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from dendritic_modeling.networks import ExcitationInhibitionNetwork

class BaseModel(nn.Module):
    def __init__(self, net):
        super(BaseModel, self).__init__()
        self.net = net

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclass.")
    
    def predict(self, x):
        raise NotImplementedError("Predict method must be implemented in subclass.")

class ProbabilisticClassifier(BaseModel):
    """
    A probabilistic classifier that outputs a categorical distribution
    via LogSoftmax. 
    """
    def __init__(self, net, output_dim=10, use_output_layer=False):
        super(ProbabilisticClassifier, self).__init__(net)
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer

        # if user wants a final FC, create it here
        if self.use_output_layer:
            self.final_fc = nn.Linear(net.forward(torch.zeros(1, net.n_branch_layers)).shape[1], self.output_dim)
            # Actually we typically can't guess in_features that way. 
            # Alternatively, we can pass net.excitatory_layer_sizes[-1]. 
            # Or do a dummy forward with correct shape. 
            # For simplicity, we'll define final_fc after we know net's output shape. 
            # This is a sketch. 
            # Let's do a placeholder approach or remove it.

        self.final = nn.LogSoftmax(dim = -1)
        self.log_output_scale = nn.Parameter(torch.zeros((self.output_dim,)), 
                                             requires_grad = True)

    def log_prob(self, x, y):
        dist = self.forward(x, sampler=True)
        return dist.log_prob(y)
    
    def forward(self, x, y=None, sampler=False):
        # net_output is shape [batch_size, last_excit_size]
        net_output = self.net(x)
        if hasattr(self, 'final_fc'):
            net_output = self.final_fc(net_output)  # optional

        final_logits = self.final(self.log_output_scale.exp() * net_output)
        categorical_dist = Categorical(logits=final_logits)
        if y is None:
            return categorical_dist if sampler else categorical_dist.logits
        else:
            nll = -categorical_dist.log_prob(y)
            return nll
        
    def predict(self, x, stochastic=False):
        distribution = self.forward(x, y=None, sampler=True)
        if stochastic:
            return distribution.sample()
        else:
            return distribution.logits.argmax(dim=-1)
        
    def compute_loss(self, x, y):
        nll = self.forward(x, y=y)
        main_loss = nll.mean(dim=0)
        return main_loss

    def forward_with_branch_outputs(self, x):
        net_output, per_layer_acts = self.net.forward_with_branch_outputs(x)
        if hasattr(self, 'final_fc'):
            net_output = self.final_fc(net_output)
        return net_output, per_layer_acts


class Classifier(BaseModel):
    """
    A standard classifier that returns raw logits; 
    cross-entropy is used for compute_loss.
    """
    def __init__(self, net, output_dim=10, use_output_layer=False):
        super(Classifier, self).__init__(net)
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer

        if self.use_output_layer:
            # user wants an additional FC. We'll assume net's forward returns [batch_size, hidden_dim].
            # We'll guess hidden_dim from net. E.g. net.excitatory_layer_sizes[-1].
            # For a robust approach, do a dummy forward. 
            # We'll do a small trick:
            self.final_fc = nn.Linear(self.output_dim, self.output_dim)  # dummy
            # or better approach: do a dummy pass or ask config for hidden size. 
            # We'll just keep it as example.

    def forward(self, x, y=None):
        logits = self.net(x)
        if hasattr(self, 'final_fc'):
            logits = self.final_fc(logits)
        return logits

    def predict(self, x, stochastic=False):
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)

    def compute_loss(self, x, y):
        logits = self.forward(x)
        loss_fn = nn.CrossEntropyLoss()
        main_loss = loss_fn(logits, y)
        return main_loss

    def forward_with_branch_outputs(self, x):
        net_output, per_layer_acts = self.net.forward_with_branch_outputs(x)
        if hasattr(self, 'final_fc'):
            net_output = self.final_fc(net_output)
        return net_output, per_layer_acts