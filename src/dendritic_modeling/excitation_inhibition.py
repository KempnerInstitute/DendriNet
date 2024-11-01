"""
excitation_inhibition.py
========================
This module contains the ExcitationInhibitionLayer and 
ExcitationInhibitionNetwork classes.
"""

from copy import deepcopy

from torch import nn

from dendritic_modeling.dendrinet import DendriNet
from dendritic_modeling import logger


class ExcitationInhibitionLayer(nn.Module):
    """
    A neural network layer modeling interactions between excitatory and 
    inhibitory neurons.

    This layer implements an excitation-inhibition mechanism using two 
    `DendriNet` models: one representing inhibitory neurons and the other 
    representing excitatory neurons.

    Parameters
    ----------

    n_excitatory_cells : int
        The number of excitatory neurons in the layer.
    
    n_inhibitory_cells : int
        The number of inhibitory neurons in the layer.

    excitatory_branch_factors : list of int
        The branching factor for excitatory neurons.

    inhibitory_branch_factors : list of int
        The branching factor for inhibitory neurons.

    excitatory_input_dim : int
        The number of excitatory inputs to the ExcitationInhibitionLayer.

    ee_synapses_per_branch : int
        The number of excitatory-to-excitatory synapses per branch.

    ei_synapses_per_branch : int
        The number of excitatory-to-inhibitory synapses per branch.

    inhibitory_input_dim : int, optional
        The number of inhibitory inputs to the ExcitationInhibitionLayer. 
        Defaults to `None`.

    ie_synapses_per_branch : int
        The number of inhibitory-to-excitatory synapses per branch. 
        
    ii_synapses_per_branch : int, optional
        The number of inhibitory-to-inhibitory synapses per branch. Defaults to
        `None`.
    
    reactivate : bool, optional
        If `True`, applies a reactivation function simulating dendritic/somatic 
        spiking behavior. Defaults to `False`.

    somatic_synapses : bool, optional
        If `True`, includes direct synaptic inputs to the soma layer. Defaults 
        to `True`.

    Methods
    -------

    forward(x, inhibitory_input)
        Performs a forward pass through the layer and returns excitatory and 
        inhibitory outputs.
    
    """

    def __init__(
        self, 
        n_excitatory_cells,
        n_inhibitory_cells,
        excitatory_branch_factors, 
        inhibitory_branch_factors,
        excitatory_input_dim, 
        ee_synapses_per_branch, 
        ei_synapses_per_branch, 
        inhibitory_input_dim = None, 
        ie_synapses_per_branch = None, 
        ii_synapses_per_branch = None, 
        reactivate = False, 
        somatic_synapses = True,
    ):
        super(ExcitationInhibitionLayer, self).__init__()

        self.inhibitory_cells = DendriNet(
            n_soma = n_inhibitory_cells, 
            branch_factors = inhibitory_branch_factors,
            excitatory_input_dim = excitatory_input_dim, 
            excitatory_synapses_per_branch = ei_synapses_per_branch,
            inhibitory_input_dim = inhibitory_input_dim, 
            inhibitory_synapses_per_branch = ii_synapses_per_branch,
            reactivate = reactivate, somatic_synapses = somatic_synapses, 
        )

        self.excitatory_cells = DendriNet(
            n_soma = n_excitatory_cells, 
            branch_factors = excitatory_branch_factors,
            excitatory_input_dim = excitatory_input_dim, 
            excitatory_synapses_per_branch = ee_synapses_per_branch,
            inhibitory_input_dim = n_inhibitory_cells, 
            inhibitory_synapses_per_branch = ie_synapses_per_branch,
            reactivate = reactivate, 
            somatic_synapses = somatic_synapses,
        )

    def forward(self, x, inhibitory_input = None):
        inhibitory_output = self.inhibitory_cells(x, inhibitory_input)
        excitatory_output = self.excitatory_cells(x, inhibitory_output)
        return excitatory_output, inhibitory_output


class ExcitationInhibitionNetwork(nn.Module):
    """
    A neural network module implementing an excitation-inhibition multilayer 
    architecture.

    This class constructs a network composed of TBD

    Parameters
    ----------
    input_dim : int
        The number of input features.

    excitatory_layer_sizes : list of int
        The number of excitatory neurons in each layer.
    
    inhibitory_layer_sizes : list of int
        The number of inhibitory neurons in each layer.

    excitatory_branch_factors : list of int
        The branching factor for excitatory neurons at each layer.

    inhibitory_branch_factors : list of int
        The branching factor for inhibitory neurons at each layer.

    ee_synapses_per_branch_per_layer : list of int
        The number of excitatory-to-excitatory synapses per branch at each 
        layer.

    ei_synapses_per_branch_per_layer : list of int
        The number of excitatory-to-inhibitory synapses per branch at each 
        layer.

    ie_synapses_per_branch_per_layer : list of int
        The number of inhibitory-to-excitatory synapses per branch at each 
        layer.
    
    ii_synapses_per_branch_per_layer : list of int, optional
        The number of inhibitory-to-inhibitory synapses per branch at each 
        layer. Defaults to an empty list `[]`.
    
    reactivate : bool, optional
        If `True`, applies a reactivation function simulating dendritic/somatic 
        spiking behavior. Defaults to `False`.
    
    somatic_synapses : bool, optional
        If `True`, includes direct synaptic inputs to the soma layer. Defaults 
        to `True`.

    Methods
    -------

    forward(x)
        Performs a forward pass through the network and returns an excitatory
        output.
    """
    def __init__(
        self, 
        input_dim, 
        excitatory_layer_sizes, 
        inhibitory_layer_sizes,
        excitatory_branch_factors, 
        inhibitory_branch_factors, 
        ee_synapses_per_branch_per_layer, 
        ei_synapses_per_branch_per_layer, 
        ie_synapses_per_branch_per_layer, 
        ii_synapses_per_branch_per_layer = [], 
        reactivate = False, somatic_synapses = True,
    ):
        super(ExcitationInhibitionNetwork, self).__init__()

        self.n_layers = len(excitatory_layer_sizes)
        
        logger.debug(f'Building ExcitationInhibitionNetwork with {self.n_layers} layer(s).')

        excitatory_layer_sizes = deepcopy(excitatory_layer_sizes)
        inhibitory_layer_sizes = deepcopy(inhibitory_layer_sizes)

        logger.debug(f'Excitatory layer sizes: {excitatory_layer_sizes}')
        logger.debug(f'Inhibitory layer sizes: {inhibitory_layer_sizes}')

        excitatory_layer_sizes.insert(0, input_dim)
        inhibitory_layer_sizes.insert(0, None)


        ii_synapses_per_branch_per_layer = deepcopy(
            ii_synapses_per_branch_per_layer)


        ii_synapses_per_branch_per_layer.insert(0, None)

        layers = []
        for i in range(self.n_layers):
            layers.append(ExcitationInhibitionLayer(
                n_excitatory_cells = excitatory_layer_sizes[i+1], 
                n_inhibitory_cells = inhibitory_layer_sizes[i+1],
                excitatory_branch_factors = excitatory_branch_factors, 
                inhibitory_branch_factors = inhibitory_branch_factors,
                excitatory_input_dim = excitatory_layer_sizes[i], 
                ee_synapses_per_branch = ee_synapses_per_branch_per_layer[i], 
                ei_synapses_per_branch = ei_synapses_per_branch_per_layer[i],
                inhibitory_input_dim = inhibitory_layer_sizes[i],
                ie_synapses_per_branch = ie_synapses_per_branch_per_layer[i], 
                ii_synapses_per_branch = ii_synapses_per_branch_per_layer[i],
                reactivate = reactivate, somatic_synapses = somatic_synapses,
            ))
        
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        excitatory = x
        inhibitory = None
        for layer in self.layers:
            excitatory, inhibitory = layer(excitatory, inhibitory)
        return excitatory