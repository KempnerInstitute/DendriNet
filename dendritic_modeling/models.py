import torch
import torch.nn as nn
from torch.distributions import Categorical

from dendritic_modeling.excitation_inhibition import ExcitationInhibitionNetwork, Modulo10_ExcitationInhibitionNetwork


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
            input_dim = input_dim, excitatory_layer_sizes = excitatory_layer_sizes, inhibitory_layer_sizes = inhibitory_layer_sizes,
            excitatory_branch_factors = excitatory_branch_factors, inhibitory_branch_factors = inhibitory_branch_factors,
            ee_synapses_per_branch_per_layer = ee_synapses_per_branch_per_layer, ei_synapses_per_branch_per_layer = ei_synapses_per_branch_per_layer,
            ie_synapses_per_branch_per_layer = ie_synapses_per_branch_per_layer, ii_synapses_per_branch_per_layer = ii_synapses_per_branch_per_layer,
            reactivate = reactivate, somatic_synapses = somatic_synapses,
        )

        self.log_output_scale = nn.Parameter(torch.zeros(excitatory_layer_sizes[-1],), requires_grad = True)

        self.final = nn.LogSoftmax(dim = -1)
    
    def forward(self, x):
        logits = self.final(self.net(x) * self.log_output_scale.exp())
        return Categorical(logits = logits)


class Modulo10_EINetClassifier(ProbabilisticClassifier):

    def __init__(
        self, first_layer_excitatory_size, second_layer_excitatory_size,
        first_layer_inhibitory_size, second_layer_inhibitory_size,
        first_layer_excitatory_branch_factors, second_layer_excitatory_branch_factors,
        first_layer_inhibitory_branch_factors, second_layer_inhibitory_branch_factors,
        first_layer_ee_synapses_per_branch, second_layer_ee_synapses_per_branch,
        first_layer_ei_synapses_per_branch, second_layer_ei_synapses_per_branch,
        first_layer_ie_synapses_per_branch, second_layer_ie_synapses_per_branch,
        second_layer_ii_synapses_per_branch, reactivate, somatic_synapses,
    ):
        super(Modulo10_EINetClassifier, self).__init__()

        self.net = Modulo10_ExcitationInhibitionNetwork(
            first_layer_excitatory_size = first_layer_excitatory_size, second_layer_excitatory_size = second_layer_excitatory_size,
            first_layer_inhibitory_size = first_layer_inhibitory_size, second_layer_inhibitory_size = second_layer_inhibitory_size,
            first_layer_excitatory_branch_factors = first_layer_excitatory_branch_factors, second_layer_excitatory_branch_factors = second_layer_excitatory_branch_factors,
            first_layer_inhibitory_branch_factors = first_layer_inhibitory_branch_factors, second_layer_inhibitory_branch_factors = second_layer_inhibitory_branch_factors,
            first_layer_ee_synapses_per_branch = first_layer_ee_synapses_per_branch, second_layer_ee_synapses_per_branch = second_layer_ee_synapses_per_branch,
            first_layer_ei_synapses_per_branch = first_layer_ei_synapses_per_branch, second_layer_ei_synapses_per_branch = second_layer_ei_synapses_per_branch,
            first_layer_ie_synapses_per_branch = first_layer_ie_synapses_per_branch, second_layer_ie_synapses_per_branch = second_layer_ie_synapses_per_branch,
            second_layer_ii_synapses_per_branch = second_layer_ii_synapses_per_branch, reactivate = reactivate, somatic_synapses = somatic_synapses, 
        )

        self.log_output_scale = nn.Parameter(torch.zeros((second_layer_excitatory_size,)), requires_grad = True)

        self.final = nn.LogSoftmax(dim = -1)
    
    def forward(self, x):
        logits = self.final(self.net(x) * self.log_output_scale.exp())
        return Categorical(logits = logits)

