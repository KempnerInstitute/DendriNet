"""
models.py
=========
This module contains the neural network models for the dendritic_modeling.
"""

import torch.nn as nn
from torch.distributions import Categorical

from dendritic_modeling.dendrinet import *
from dendritic_modeling.excitation_inhibition import *



class ProbabilisticModel(nn.Module):

    def log_prob(self, x, y):
        distribution = self.forward(x)
        return distribution.log_prob(y)


class ProbabilisticClassifier(ProbabilisticModel):

    def predict(self, x, stochastic = False):
        distribution = self.forward(x)

        if stochastic:
            return distribution.sample()
        else:
            return distribution.logits.argmax(dim = -1)


class ProbabilisticRegressor(ProbabilisticModel):

    def predict(self, x, stochastic = True):
        distribution = self.forward(x)

        if stochastic:
            return distribution.sample()
        else:
            return distribution.mean


class EINetClassifier(ProbabilisticClassifier):

    def __init__(
        self, input_dim, excitatory_layer_sizes, inhibitory_layer_sizes,
        excitatory_branch_factors, inhibitory_branch_factors, 
        ee_synapses_per_branch_per_layer, ei_synapses_per_branch_per_layer, 
        ie_synapses_per_branch_per_layer, ii_synapses_per_branch_per_layer = [], 
        reactivate = False, somatic_synapses = True,
    ):
        super(EINetClassifier, self).__init__()

        self.net = ExcitationInhibitionNetwork(
            input_dim = input_dim, 
            excitatory_layer_sizes = excitatory_layer_sizes, 
            inhibitory_layer_sizes = inhibitory_layer_sizes,
            excitatory_branch_factors = excitatory_branch_factors, 
            inhibitory_branch_factors = inhibitory_branch_factors,
            ee_synapses_per_branch_per_layer = ee_synapses_per_branch_per_layer, 
            ei_synapses_per_branch_per_layer = ei_synapses_per_branch_per_layer,
            ie_synapses_per_branch_per_layer = ie_synapses_per_branch_per_layer, 
            ii_synapses_per_branch_per_layer = ii_synapses_per_branch_per_layer,
            reactivate = reactivate, somatic_synapses = somatic_synapses,
        )
        self.final = nn.LogSoftmax(dim = -1)
    
    def forward(self, x):
        return Categorical(logits = self.final(self.net(x)))


