"""
networks.py
========================
This module contains the ExcitationInhibitionLayer and 
ExcitationInhibitionNetwork classes.

The conductance based gradient scaling can be activated by setting
gradient_strategy: block_conductance_dynamic
Then, from the code, we can attach a backward hook to each 
BlockLinear in the net using: 

    from dendritic_modeling.gradient_scaling import GradientScaler
    scaler = GradientScaler(strategy='block_conductance_dynamic')
    scaler.register_block_linear_dynamic(...)

"""

from copy import deepcopy
from torch import nn
from typing import List
import torch
import torch.nn.functional as F
from dendritic_modeling.dendrinet import DendriNet, DendriNetWithOutputs
from dendritic_modeling import logger

class ExcitationInhibitionLayer(nn.Module):
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
        use_shunting = True,
        gradient_strategy="none",
        reactivation_type='param_tanh',
        enable_branch_outputs=False 
    ):
        super(ExcitationInhibitionLayer, self).__init__()

        if enable_branch_outputs:
            self.inhibitory_cells = DendriNetWithOutputs(
                n_soma = n_inhibitory_cells, 
                branch_factors = inhibitory_branch_factors,
                excitatory_input_dim = excitatory_input_dim, 
                excitatory_synapses_per_branch = ei_synapses_per_branch,
                inhibitory_input_dim = inhibitory_input_dim, 
                inhibitory_synapses_per_branch = ii_synapses_per_branch,
                reactivate = reactivate, 
                somatic_synapses = somatic_synapses, 
                topk_init_method = topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
            )
            self.excitatory_cells = DendriNetWithOutputs(
                n_soma = n_excitatory_cells, 
                branch_factors = excitatory_branch_factors,
                excitatory_input_dim = excitatory_input_dim, 
                excitatory_synapses_per_branch = ee_synapses_per_branch,
                inhibitory_input_dim = n_inhibitory_cells, 
                inhibitory_synapses_per_branch = ie_synapses_per_branch,
                reactivate = reactivate, 
                somatic_synapses = somatic_synapses,
                topk_init_method = topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
            )
        else:
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
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
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
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
            )
    
    def decay_weights(self, weight_decay):
        self.inhibitory_cells.decay_weights(weight_decay)
        self.excitatory_cells.decay_weights(weight_decay)

    def forward(self, x, inhibitory_input = None):
        inhibitory_output = self.inhibitory_cells(x, inhibitory_input)
        excitatory_output = self.excitatory_cells(x, inhibitory_output)
        return excitatory_output, inhibitory_output

    def forward_with_branch_outputs(self, x, inhibitory_input=None):
        if not hasattr(self.inhibitory_cells, 'forward_with_branch_outputs'):
            inhibitory_output = self.inhibitory_cells(x, inhibitory_input)
            excitatory_output = self.excitatory_cells(x, inhibitory_output)
            return excitatory_output, inhibitory_output, [], []
        else:
            inhibitory_final, inh_acts = self.inhibitory_cells.forward_with_branch_outputs(x, inhibitory_input)
            excitatory_final, exc_acts = self.excitatory_cells.forward_with_branch_outputs(x, inhibitory_final)
            return excitatory_final, inhibitory_final, inh_acts, exc_acts


class ExcitationInhibitionNetwork(nn.Module):
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
        reactivate = False, 
        somatic_synapses = True,
        topk_init_method = "xavier_normal",
        use_shunting=True,
        gradient_strategy="none",
        weight_decay_rate=0.1,
        reactivation_type='param_tanh',
        local_loss_weight=0.0,
        enable_branch_outputs=False, 
        output_layer=False,
        output_dim=10,
        learning_strategy="mle",
    ):
        super(ExcitationInhibitionNetwork, self).__init__()
        self.weight_decay_rate = weight_decay_rate
        self.n_layers = len(excitatory_layer_sizes)
        self.gradient_strategy = gradient_strategy
        self.local_loss_weight = local_loss_weight
        
        logger.debug(f'Building ExcitationInhibitionNetwork with {self.n_layers} layer(s).')

        excitatory_layer_sizes = deepcopy(excitatory_layer_sizes)
        inhibitory_layer_sizes = deepcopy(inhibitory_layer_sizes)

        excitatory_layer_sizes.insert(0, input_dim)
        inhibitory_layer_sizes.insert(0, None)

        ee_syn = list(ee_synapses_per_branch_per_layer)
        ei_syn = list(ei_synapses_per_branch_per_layer)
        ie_syn = list(ie_synapses_per_branch_per_layer)
        ii_syn = list(ii_synapses_per_branch_per_layer)
        while len(ee_syn) < self.n_layers:
            ee_syn.append(ee_syn[-1])
        while len(ei_syn) < self.n_layers:
            ei_syn.append(ei_syn[-1])
        while len(ie_syn) < self.n_layers:
            ie_syn.append(ie_syn[-1])
        while len(ii_syn) < self.n_layers:
            ii_syn.append(None)

        layers = []
        for i in range(self.n_layers):
            layers.append(ExcitationInhibitionLayer(
                n_excitatory_cells = excitatory_layer_sizes[i+1], 
                n_inhibitory_cells = inhibitory_layer_sizes[i+1],
                excitatory_branch_factors = excitatory_branch_factors, 
                inhibitory_branch_factors = inhibitory_branch_factors,
                excitatory_input_dim = excitatory_layer_sizes[i], 
                ee_synapses_per_branch = ee_syn[i], 
                ei_synapses_per_branch = ei_syn[i],
                inhibitory_input_dim = inhibitory_layer_sizes[i],
                ie_synapses_per_branch = ie_syn[i], 
                ii_synapses_per_branch = ii_syn[i],
                reactivate = reactivate, 
                somatic_synapses = somatic_synapses,
                topk_init_method = topk_init_method,
                use_shunting=use_shunting,
                gradient_strategy = self.gradient_strategy,
                reactivation_type = reactivation_type,
                enable_branch_outputs = enable_branch_outputs
            ))
        
        self.layers = nn.ModuleList(layers)

        self.feedback_mats = nn.ParameterList()
        for i in range(self.n_layers):
            if 1==2: #self.local_learning != "none":      ####### TO DO THIS CURRENTLY IS NOT WORKING BECAUSE local_learning is removed as a config parameter
            #if self.learning_strategy.lower().startswith("local"):
                fb = nn.Parameter(torch.randn(excitatory_layer_sizes[i+1], 10)*0.01, requires_grad=False)
                self.feedback_mats.append(fb)
            else:
                self.feedback_mats.append(None)

    def decay_weights(self, weight_decay=None):
        """
        Called by the optimizer step to apply weight decay to each layer.
        If weight_decay is not provided, we default to self.weight_decay_rate.
        """
        if weight_decay is None:
            weight_decay = self.weight_decay_rate
        for layer in self.layers:
            layer.decay_weights(weight_decay)

    def forward(self, x):
        excitatory = x
        inhibitory = None
        for i, layer in enumerate(self.layers):
            excitatory, inhibitory = layer(excitatory, inhibitory)
        return excitatory

    def forward_with_branch_outputs(self, x):
        excitatory = x
        inhibitory = None
        all_acts = []
        for i, layer in enumerate(self.layers):
            excitatory, inhibitory, inh_acts, exc_acts = layer.forward_with_branch_outputs(excitatory, inhibitory)
            all_acts.extend(exc_acts)
        return excitatory, all_acts

    def compute_local_loss(self, x, y=None):
        return 0.0

    @property
    def branch_layers(self):
        """
        Provide a combined list of DendriticBranchLayer modules from all
        excitatory sub-nets so that code expecting `model.branch_layers`
        won't crash. Each item is a DendriticBranchLayer, typically
        found in excitatory_cells.branch_layers.
        """
        all_branch_layers = []
        for ei_layer in self.layers:
            all_branch_layers.extend(ei_layer.excitatory_cells.branch_layers)
        return all_branch_layers

    @property
    def n_branch_layers(self):
        """
        Provide the number of branch layers. For code referencing
        `model.n_branch_layers`, we match DendriNet's definition by counting
        how many DendriticBranchLayer modules the excitatory sub-nets hold.
        """
        return len(self.branch_layers)


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