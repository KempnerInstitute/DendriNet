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
        super(BaseModel, self).__init__()
        self.net = net

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclass.")
    
    def predict(self, x):
        raise NotImplementedError("Predict method must be implemented in subclass.")


class ProbabilisticClassifier(BaseModel):
    """
    Probabilistic classification model using categorical distribution.
    """

    def __init__(self, net, output_dim):
        super(ProbabilisticClassifier, self).__init__(net)
        self.output_dim = output_dim
        self.final = nn.LogSoftmax(dim = -1)
        self.log_output_scale = nn.Parameter(torch.zeros((self.output_dim,)), 
                                             requires_grad = True)

    def log_prob(self, x, y):
        """
        Computes the log-probability for classification.
    
        Parameters
        ----------
        x : torch.Tensor
            Input features.
        y : torch.Tensor
            Labels for which log-probability is computed.
    
        Returns
        -------
        torch.Tensor
            Log-probabilities for each label.
        """
        dist = self.forward(x, sampler=True)  # Get categorical distribution
        return dist.log_prob(y)
    
    def forward(self, x, y=None, sampler=False):
        """
        Forward pass for the classifier. Returns either logits, a categorical
        distribution, or the negative log-likelihood loss depending on the
        input arguments.

        Parameters
        ----------
        x : torch.Tensor
            The input data for which to compute the output.

        y : torch.Tensor, optional
            The observed labels for the input data. If provided, the negative
            log-likelihood loss is computed. Defaults to `None`.

        sampler : bool, optional
            If `True` and `y=None`, returns the categorical distribution;
            otherwise, returns logits. Defaults to `False`.

        Returns
        -------
        torch.Tensor or torch.distributions.Categorical
            The logits, categorical distribution, or negative log-likelihood 
            loss depending on the input arguments.

        Raises
        ------
        ValueError
            If the input-output pair is invalid for the forward method.
        """
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
        return nll.mean(dim=0)


class Classifier(BaseModel):
    """
    Classification model using argmax.
    """

    def __init__(self, net):
        super().__init__(net)
        #self.final = nn.Softmax(dim=-1)

    # need to pass y=None to forward method to avoid consistency issue with 
    # ProbabilisticClassifier, TODO: refactor
    def forward(self, x, y=None):
        output = self.net(x)
        return output

    # need to pass stochastic=False to predict method to avoid consistency issue
    # with ProbabilisticClassifier, TODO: refactor
    def predict(self, x, stochastic=False):
        output = self.forward(x)
        return output.argmax(dim=-1)
    
    def compute_loss(self, x, y):
        logits = self.forward(x)
        return nn.CrossEntropyLoss()(logits, y)
