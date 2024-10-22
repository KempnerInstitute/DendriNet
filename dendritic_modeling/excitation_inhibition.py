import torch
from torch import nn
from dendritic_modeling.dendrinet import DendriNet

from copy import deepcopy


class ExcitationInhibitionLayer(nn.Module):
    def __init__(
        self, n_excitatory_cells, n_inhibitory_cells,
        excitatory_branch_factors, inhibitory_branch_factors,
        excitatory_input_dim, ee_synapses_per_branch, ei_synapses_per_branch, 
        inhibitory_input_dim = None, ie_synapses_per_branch = None, ii_synapses_per_branch = None, 
        reactivate = False, somatic_synapses = True,
    ):
        super(ExcitationInhibitionLayer, self).__init__()

        self.inhibitory_cells = DendriNet(
            n_soma = n_inhibitory_cells, branch_factors = inhibitory_branch_factors,
            excitatory_input_dim = excitatory_input_dim, excitatory_synapses_per_branch = ei_synapses_per_branch,
            inhibitory_input_dim = inhibitory_input_dim, inhibitory_synapses_per_branch = ii_synapses_per_branch,
            reactivate = reactivate, somatic_synapses = somatic_synapses, 
        )

        self.excitatory_cells = DendriNet(
            n_soma = n_excitatory_cells, branch_factors = excitatory_branch_factors,
            excitatory_input_dim = excitatory_input_dim, excitatory_synapses_per_branch = ee_synapses_per_branch,
            inhibitory_input_dim = n_inhibitory_cells, inhibitory_synapses_per_branch = ie_synapses_per_branch,
            reactivate = reactivate, somatic_synapses = somatic_synapses,
        )

    def forward(self, x, inhibitory_input = None):
        inhibitory_output = self.inhibitory_cells(x, inhibitory_input)
        excitatory_output = self.excitatory_cells(x, inhibitory_output)
        return excitatory_output, inhibitory_output


class ExcitationInhibitionNetwork(nn.Module):
    def __init__(
        self, input_dim, excitatory_layer_sizes, inhibitory_layer_sizes,
        excitatory_branch_factors, inhibitory_branch_factors, 
        ee_synapses_per_branch_per_layer, ei_synapses_per_branch_per_layer, 
        ie_synapses_per_branch_per_layer, ii_synapses_per_branch_per_layer = [], 
        reactivate = False, somatic_synapses = True,
    ):
        super(ExcitationInhibitionNetwork, self).__init__()

        self.n_layers = len(excitatory_layer_sizes)

        excitatory_layer_sizes = deepcopy(excitatory_layer_sizes)
        inhibitory_layer_sizes = deepcopy(inhibitory_layer_sizes)

        excitatory_layer_sizes.insert(0, input_dim)
        inhibitory_layer_sizes.insert(0, None)

        ii_synapses_per_branch_per_layer = deepcopy(ii_synapses_per_branch_per_layer)
        ii_synapses_per_branch_per_layer.insert(0, None)

        layers = []
        for i in range(self.n_layers):
            layers.append(ExcitationInhibitionLayer(
                n_excitatory_cells = excitatory_layer_sizes[i+1], n_inhibitory_cells = inhibitory_layer_sizes[i+1],
                excitatory_branch_factors = excitatory_branch_factors, inhibitory_branch_factors = inhibitory_branch_factors,
                excitatory_input_dim = excitatory_layer_sizes[i], 
                ee_synapses_per_branch = ee_synapses_per_branch_per_layer[i], ei_synapses_per_branch = ei_synapses_per_branch_per_layer[i],
                inhibitory_input_dim = inhibitory_layer_sizes[i],
                ie_synapses_per_branch = ie_synapses_per_branch_per_layer[i], ii_synapses_per_branch = ii_synapses_per_branch_per_layer[i],
                reactivate = reactivate, somatic_synapses = somatic_synapses,
            ))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        excitatory = x
        inhibitory = None
        for layer in self.layers:
            excitatory, inhibitory = layer(excitatory, inhibitory)
        return excitatory


class Modulo10_ExcitationInhibitionNetwork(nn.Module):

    def __init__(
        self, first_layer_excitatory_size, second_layer_excitatory_size,
        first_layer_inhibitory_size, second_layer_inhibitory_size,
        first_layer_excitatory_branch_factors, second_layer_excitatory_branch_factors,
        first_layer_inhibitory_branch_factors, second_layer_inhibitory_branch_factors,
        first_layer_ee_synapses_per_branch, second_layer_ee_synapses_per_branch,
        first_layer_ei_synapses_per_branch, second_layer_ei_synapses_per_branch,
        first_layer_ie_synapses_per_branch, second_layer_ie_synapses_per_branch,
        second_layer_ii_synapses_per_branch, reactivate = False, somatic_synapses = True,
    ):
        super(Modulo10_ExcitationInhibitionNetwork, self).__init__()

        n_excitatory_cells = [first_layer_excitatory_size] * 2 + [second_layer_excitatory_size]
        n_inhibitory_cells = [first_layer_inhibitory_size] * 2 + [second_layer_inhibitory_size]

        excitatory_branch_factors = [first_layer_excitatory_branch_factors] * 2 + [second_layer_excitatory_branch_factors]
        inhibitory_branch_factors = [first_layer_inhibitory_branch_factors] * 2 + [second_layer_inhibitory_branch_factors]

        excitatory_input_dim = [784] * 2 + [first_layer_excitatory_size * 2]
        inhibitory_input_dim = [None] * 2 + [first_layer_inhibitory_size * 2]

        ee_synapses_per_branch = [first_layer_ee_synapses_per_branch] * 2 + [second_layer_ee_synapses_per_branch]
        ei_synapses_per_branch = [first_layer_ei_synapses_per_branch] * 2 + [second_layer_ei_synapses_per_branch]
        ie_synapses_per_branch = [first_layer_ie_synapses_per_branch] * 2 + [second_layer_ie_synapses_per_branch]
        ii_synapses_per_branch = [None] * 2 + [second_layer_ii_synapses_per_branch]

        layers = []
        for i in range(3):
            layers.append(ExcitationInhibitionLayer(
                n_excitatory_cells = n_excitatory_cells[i], n_inhibitory_cells = n_inhibitory_cells[i],
                excitatory_branch_factors = excitatory_branch_factors[i], inhibitory_branch_factors = inhibitory_branch_factors[i],
                excitatory_input_dim = excitatory_input_dim[i],
                ee_synapses_per_branch = ee_synapses_per_branch[i], ei_synapses_per_branch = ei_synapses_per_branch[i],
                inhibitory_input_dim = inhibitory_input_dim[i],
                ie_synapses_per_branch = ie_synapses_per_branch[i], ii_synapses_per_branch = ii_synapses_per_branch[i],
                reactivate = reactivate, somatic_synapses = somatic_synapses,
            ))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        x0, x1 = x[...,:784], x[...,784:]

        exc0, inh0 = self.layers[0](x0)
        exc1, inh1 = self.layers[1](x1)

        exc = torch.cat([exc0, exc1], dim = -1)
        inh = torch.cat([inh0, inh1], dim = -1)

        output, _ = self.layers[2](exc, inh)
        
        return output



