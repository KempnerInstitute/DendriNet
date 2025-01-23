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
    def __init__(self, net, output_dim):
        super(ProbabilisticClassifier, self).__init__(net)
        self.output_dim = output_dim
        self.final = nn.LogSoftmax(dim = -1)
        self.log_output_scale = nn.Parameter(torch.zeros((self.output_dim,)), 
                                             requires_grad = True)

    def log_prob(self, x, y):
        dist = self.forward(x, sampler=True)
        return dist.log_prob(y)
    
    def forward(self, x, y=None, sampler=False):
        net_output = self.net(x)
        final_logits = self.final(self.log_output_scale.exp() * net_output)
        categorical_dist = Categorical(logits=final_logits)
        if y is None:
            return categorical_dist if sampler else categorical_dist.logits
        elif y is not None:
            nll = -categorical_dist.log_prob(y)
            return nll
        else:
            raise ValueError("Invalid input-output pair for forward method.")
        
    def predict(self, x, stochastic = False):
        distribution = self.forward(x, y=None, sampler=True)
        if stochastic:
            return distribution.sample()
        else:
            return distribution.logits.argmax(dim = -1)
        
    def compute_loss(self, x, y):
        nll = self.forward(x, y=y)
        main_loss = nll.mean(dim=0)
        if hasattr(self.net, 'compute_local_loss'):
            local_obj = self.net.compute_local_loss(x, y)
            return main_loss + local_obj
        else:
            return main_loss


class Classifier(BaseModel):
    """
    A standard classifier that returns raw logits; 
    cross-entropy is used for compute_loss.
    """
    def __init__(self, net):
        super(Classifier, self).__init__(net)

    def forward(self, x, y=None):
        output = self.net(x)
        return output

    def predict(self, x, stochastic=False):
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)

    def compute_loss(self, x, y):
        logits = self.forward(x)
        loss_fn = nn.CrossEntropyLoss()
        main_loss = loss_fn(logits, y)
        if hasattr(self.net, 'compute_local_loss'):
            local_obj = self.net.compute_local_loss(x, y)
            return main_loss + local_obj
        else:
            return main_loss


class EINetClassifierNLL(BaseModel):
    """
    A negative-log-likelihood style classifier that also uses a 
    Categorical distribution under the hood. 
    This matches your older EINetClassifier logic.
    """

    def __init__(self, net, output_dim):
        super(EINetClassifierNLL, self).__init__(net)
        self.output_dim = output_dim
        self.final = nn.LogSoftmax(dim=-1)
        # learnable scale of final layer
        self.log_output_scale = nn.Parameter(torch.zeros((self.output_dim,)), requires_grad=True)

    def forward(self, x, y=None, sampler=False):
        net_output = self.net(x)
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
        if hasattr(self.net, 'compute_local_loss'):
            local_obj = self.net.compute_local_loss(x, y)
            return main_loss + local_obj
        else:
            return main_loss