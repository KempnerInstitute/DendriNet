"""
excitation_inhibition.py
========================
This module contains the ExcitationInhibitionLayer and 
ExcitationInhibitionNetwork classes.
"""

from copy import deepcopy
from typing import List

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
        reactivate = True, 
        somatic_synapses = True,
        topk_init_method = "xavier_normal",
        use_shunting = True
    ):
        super(ExcitationInhibitionLayer, self).__init__()

        self.inhibitory_cells = DendriNet(
            n_soma = n_inhibitory_cells, 
            branch_factors = inhibitory_branch_factors,
            excitatory_input_dim = excitatory_input_dim, 
            excitatory_synapses_per_branch = ei_synapses_per_branch,
            inhibitory_input_dim = inhibitory_input_dim, 
            inhibitory_synapses_per_branch = ii_synapses_per_branch,
            reactivate = reactivate, 
            somatic_synapses = somatic_synapses, 
            topk_init_method = topk_init_method,
            use_shunting=use_shunting
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
            topk_init_method = topk_init_method,
            use_shunting=use_shunting
        )
    
    def decay_weights(self, weight_decay = 0.1):
        self.inhibitory_cells.decay_weights(weight_decay)
        self.excitatory_cells.decay_weights(weight_decay)

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
        reactivate = True, 
        somatic_synapses = True,
        topk_init_method = "xavier_normal",
        use_shunting = True
    ):
        super(ExcitationInhibitionNetwork, self).__init__()

        self.n_layers = len(excitatory_layer_sizes)
        
        logger.debug(f'Building ExcitationInhibitionNetwork with {self.n_layers} layer(s).')

        excitatory_layer_sizes = deepcopy(excitatory_layer_sizes)
        inhibitory_layer_sizes = deepcopy(inhibitory_layer_sizes)

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
                reactivate = reactivate, 
                somatic_synapses = somatic_synapses,
                topk_init_method = topk_init_method,
                use_shunting = use_shunting
            ))
        
        self.layers = nn.ModuleList(layers)
    
    def decay_weights(self, weight_decay = 0.1):
        for layer in self.layers:
            layer.decay_weights(weight_decay)

    def forward(self, x):
        excitatory = x
        inhibitory = None
        for layer in self.layers:
            excitatory, inhibitory = layer(excitatory, inhibitory)
        return excitatory    


class FeedforwardNN(nn.Module):
    """
    A configurable feedforward neural network as a baseline for comparison
    on the flattened MNIST dataset.
    """
    def __init__(
        self,
        input_dim: int = 784,
        hidden_layer_sizes: List[int] = [128, 64],
        activation: str = "relu",
        dropout: float = 0.2,
        output_dim: int = 10,
    ):
        super(FeedforwardNN, self).__init__()

        # Map string activation to PyTorch activation functions
        activation_mapping = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU
        }

        # Raise an error if the activation is not supported
        if activation not in activation_mapping:
            raise ValueError(f"Unsupported activation function: {activation}. "
                             f"Supported: {list(activation_mapping.keys())}")

        # Define layers dynamically
        layers = []
        current_dim = input_dim

        for layer_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_dim, layer_size))
            layers.append(activation_mapping[activation]())
            if dropout > 0:  # Apply dropout only if it's greater than 0
                layers.append(nn.Dropout(dropout))
            current_dim = layer_size

        # Add the output layer
        layers.append(nn.Linear(current_dim, output_dim))

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    
class MLPBaseline(nn.Module):
    """
    A simple feedforward MLP with one hidden layer, for baseline comparison.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPBaseline, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits