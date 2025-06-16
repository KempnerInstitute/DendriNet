"""
Probabilistic classification models.

This module contains probabilistic classifiers that output distributions.
"""

import torch
import torch.nn as nn
from torch.distributions import Categorical

from dendritic_modeling.models.base import BaseModel


class ProbabilisticClassifier(BaseModel):
    """
    A probabilistic classifier that outputs a categorical distribution
    via LogSoftmax.
    """

    def __init__(self, net, output_dim=10, use_output_layer=False):
        super(ProbabilisticClassifier, self).__init__(net)
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer

        # if we want a final FC, create it here
        if self.use_output_layer:
            self.final_fc = nn.Linear(
                net.forward(torch.zeros(1, net.n_branch_layers)).shape[1],
                self.output_dim,
            )
        else:
            self.log_output_scale = nn.Parameter(
                torch.zeros((self.output_dim,)), requires_grad=True
            )

        self.final = nn.LogSoftmax(dim=-1)

    def log_prob(self, x, y):
        dist = self.forward(x, sampler=True)
        return dist.log_prob(y)

    def forward(self, x, y=None, sampler=False):
        # net_output is shape [batch_size, last_excit_size]
        net_output = self.net(x)
        if hasattr(self, "final_fc"):
            net_output = self.final_fc(net_output)  # optional
        else:
            net_output = self.log_output_scale.exp() * net_output

        final_logits = self.final(net_output)
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
        if hasattr(self, "final_fc"):
            net_output = self.final_fc(net_output)
        return net_output, per_layer_acts


__all__ = ["ProbabilisticClassifier"] 