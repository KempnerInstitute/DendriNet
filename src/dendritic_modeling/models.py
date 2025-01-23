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
    """
    Base class for all models in the dendritic modeling package.
    """
    def __init__(self, net):
        # IMPORTANT: call nn.Module.__init__() with no extra arguments
        super(BaseModel, self).__init__()
        # store the 'net' as an attribute
        self.net = net

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclass.")
    
    def predict(self, x):
        raise NotImplementedError("Predict method must be implemented in subclass.")


class ProbabilisticClassifier(BaseModel):
    """
    Probabilistic classification model using a categorical distribution.
    """

    def __init__(self, net, output_dim):
        # Pass 'net' up to BaseModel, NOT to nn.Module.__init__
        super(ProbabilisticClassifier, self).__init__(net)
        self.output_dim = output_dim
        self.final = nn.LogSoftmax(dim=-1)
        self.log_output_scale = nn.Parameter(torch.zeros(self.output_dim), requires_grad=True)

    def log_prob(self, x, y):
        dist = self.forward(x, sampler=True)
        return dist.log_prob(y)
    
    def forward(self, x, y=None, sampler=False):
        """
        Forward pass for the classifier. Returns either logits, a categorical
        distribution, or the negative log-likelihood loss depending on input arguments.

        Parameters
        ----------
        x : torch.Tensor
            The input data for which to compute the output.
        y : torch.Tensor, optional
            The observed labels for the input data. If provided, returns NLL.
        sampler : bool, optional
            If True and y=None, returns a categorical distribution rather than logits.

        Returns
        -------
        torch.Tensor or torch.distributions.Categorical
            Depending on arguments, returns logits, a distribution, or NLL.

        Raises
        ------
        ValueError
            If invalid arguments are provided.
        """
        net_output = self.net(x)
        scaled_logits = self.log_output_scale.exp() * net_output
        final_logits = self.final(scaled_logits)
        categorical_dist = Categorical(logits=final_logits)

        if y is None:
            # Return distribution if sampler=True, else return raw logits
            return categorical_dist if sampler else categorical_dist.logits
        elif y is not None:
            # Negative log-likelihood
            nll = -categorical_dist.log_prob(y)
            return nll
        else:
            raise ValueError("Invalid input-output pair for forward method.")
        
    def predict(self, x, stochastic=False):
        distribution = self.forward(x, y=None, sampler=True)
        if stochastic:
            return distribution.sample()
        else:
            return distribution.logits.argmax(dim=-1)
        
    def compute_loss(self, x, y):
        """
        Standard classification NLL + optional local learning objective
        if net has compute_local_loss.
        """
        nll = self.forward(x, y=y)
        main_loss = nll.mean(dim=0)

        # NEW FEATURE: local learning 
        if hasattr(self.net, 'compute_local_loss'):
            local_obj = self.net.compute_local_loss(x, y)
            return main_loss + local_obj
        else:
            return main_loss


class Classifier(BaseModel):
    """
    Classification model using argmax (standard cross-entropy).
    """
    def __init__(self, net):
        super(Classifier, self).__init__(net)
        # self.final = nn.Softmax(dim=-1)  # optional

    def forward(self, x, y=None):
        output = self.net(x)
        return output

    def predict(self, x, stochastic=False):
        """
        Predict method to return argmax of logits.
        """
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)

    def compute_loss(self, x, y):
        """
        Standard cross-entropy classification + optional local learning.
        """
        logits = self.forward(x)
        loss_fn = nn.CrossEntropyLoss()
        main_loss = loss_fn(logits, y)

        if hasattr(self.net, 'compute_local_loss'):
            local_obj = self.net.compute_local_loss(x, y)
            return main_loss + local_obj
        else:
            return main_loss