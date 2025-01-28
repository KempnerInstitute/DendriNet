"""
networks.py
========================
This module contains the ExcitationInhibitionLayer and 
ExcitationInhibitionNetwork classes.
"""

from copy import deepcopy
from torch import nn
from typing import List
import torch
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
        init_gain=1.0,
        init_inhib_offset=0.0,
        gradient_strategy="none",
        gradient_scale_factor=1.0,
        reactivation_initial_m=1.5,
        reactivation_initial_b=1.0,
        reactivation_type='param_tanh',
        train_reactivation_b_only=False,
        enable_branch_outputs=False  # added
    ):
        super(ExcitationInhibitionLayer, self).__init__()
        self.init_gain = init_gain
        self.init_inhib_offset = init_inhib_offset
        self.gradient_strategy = gradient_strategy
        self.gradient_scale_factor = gradient_scale_factor

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
                reactivation_initial_m=reactivation_initial_m,
                reactivation_initial_b=reactivation_initial_b,
                train_reactivation_b_only=train_reactivation_b_only,
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
                reactivation_initial_m=reactivation_initial_m,
                reactivation_initial_b=reactivation_initial_b,
                train_reactivation_b_only=train_reactivation_b_only,
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
                reactivation_initial_m=reactivation_initial_m,
                reactivation_initial_b=reactivation_initial_b,
                train_reactivation_b_only=train_reactivation_b_only,
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
                reactivation_initial_m=reactivation_initial_m,
                reactivation_initial_b=reactivation_initial_b,
                train_reactivation_b_only=train_reactivation_b_only,
            )
    
    def decay_weights(self, weight_decay):
        self.inhibitory_cells.decay_weights(weight_decay)
        self.excitatory_cells.decay_weights(weight_decay)

    def forward(self, x, inhibitory_input = None):
        inhibitory_output = self.inhibitory_cells(x, inhibitory_input)
        excitatory_output = self.excitatory_cells(x, inhibitory_output)
        return excitatory_output, inhibitory_output


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
        init_gain=1.0,
        init_inhib_offset=0.0,
        gradient_strategy="none",
        gradient_scale_factor=1.0,
        weight_decay_rate=0.1,
        reactivation_initial_m=1.5,
        reactivation_initial_b=1.0,
        reactivation_type='param_tanh',
        train_reactivation_b_only=False,
        local_learning="none",
        local_loss_weight=0.0,
        enable_branch_outputs=False  # added
    ):
        super(ExcitationInhibitionNetwork, self).__init__()
        self.weight_decay_rate = weight_decay_rate
        self.n_layers = len(excitatory_layer_sizes)
        self.init_gain = init_gain
        self.init_inhib_offset = init_inhib_offset
        self.gradient_strategy = gradient_strategy
        self.gradient_scale_factor = gradient_scale_factor
        self.local_learning = local_learning
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
                init_gain = self.init_gain,
                init_inhib_offset = self.init_inhib_offset,
                gradient_strategy = self.gradient_strategy,
                gradient_scale_factor = self.gradient_scale_factor,
                reactivation_initial_m = reactivation_initial_m,
                reactivation_initial_b = reactivation_initial_b,
                reactivation_type = reactivation_type,
                train_reactivation_b_only = train_reactivation_b_only,
                enable_branch_outputs = enable_branch_outputs  # added
            ))
        
        self.layers = nn.ModuleList(layers)

        self.feedback_mats = nn.ParameterList()
        for i in range(self.n_layers):
            if self.local_learning != "none":
                fb = nn.Parameter(torch.randn(excitatory_layer_sizes[i+1], 10)*0.01, requires_grad=False)
                self.feedback_mats.append(fb)
            else:
                self.feedback_mats.append(None)

    def decay_weights(self):
        for layer in self.layers:
            layer.decay_weights(self.weight_decay_rate)

    def forward(self, x):
        excitatory = x
        inhibitory = None
        for i, layer in enumerate(self.layers):
            excitatory, inhibitory = layer(excitatory, inhibitory)
        return excitatory

    def compute_local_loss(self, x, y=None):
        return 0.0