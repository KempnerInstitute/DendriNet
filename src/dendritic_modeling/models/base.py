"""
Base model class for dendritic modeling.

This module contains the abstract BaseModel class that provides common functionality
for all model wrappers including parameter grouping and weight decay.
"""

import torch.nn as nn
from itertools import chain

from dendritic_modeling.networks import ExcitationInhibitionNetwork


class BaseModel(nn.Module):
    def __init__(self, net):
        super(BaseModel, self).__init__()
        self.net = net

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclass.")

    def predict(self, x):
        raise NotImplementedError("Predict method must be implemented in subclass.")
    
    def decay_weights(self, weight_decay):
        """
        Forward this call to self.net if it implements decay_weights.
        """
        if hasattr(self.net, 'decay_weights'):
            self.net.decay_weights(weight_decay)
    
    def get_param_groups(self, optim_dict):
        """
        Get parameter groups for optimizer.
        """
        split_params = optim_dict.get('split_params', False)
        if split_params and isinstance(self.net, ExcitationInhibitionNetwork):
            topk_params = []
            blocklinear_params = []
            reactivation_params = []
            output_layer_params = []

            for ei_layer in self.net.layers:
                exc_cells = ei_layer.excitatory_cells
                inh_cells = ei_layer.inhibitory_cells

                for dendritic_layer in exc_cells.branch_layers:
                    if dendritic_layer.branch_excitation is not None:
                        topk_params.append(dendritic_layer.branch_excitation.parameters())
                    if dendritic_layer.branch_inhibition is not None:
                        topk_params.append(dendritic_layer.branch_inhibition.parameters())
                    
                    if hasattr(dendritic_layer, 'branches_to_output'):
                        blocklinear_params.append(dendritic_layer.branches_to_output.parameters())
                    
                    if dendritic_layer.reactivate:
                        reactivation_params.append(dendritic_layer.reactivation.parameters())
                
                if inh_cells is not None:
                    for dendritic_layer in inh_cells.branch_layers:
                        if dendritic_layer.branch_excitation is not None:
                            topk_params.append(dendritic_layer.branch_excitation.parameters())
                        if dendritic_layer.branch_inhibition is not None:
                            topk_params.append(dendritic_layer.branch_inhibition.parameters())
                        
                        if hasattr(dendritic_layer, 'branches_to_output'):
                            blocklinear_params.append(dendritic_layer.branches_to_output.parameters())
                        
                        if dendritic_layer.reactivate:
                            reactivation_params.append(dendritic_layer.reactivation.parameters())
            
            if hasattr(self.net, 'output_classifier'):
                output_layer_params.append(self.net.output_classifier.parameters())
            if hasattr(self, 'final_fc'):
                output_layer_params.append(self.final_fc.parameters())
            if hasattr(self, 'log_output_scale'):
                output_layer_params.append([self.log_output_scale])

            param_groups = []
            if topk_params:
                param_groups.append({'params': chain(*topk_params), 'lr': optim_dict.get('topk_lr', 0.001)})
            if blocklinear_params:
                param_groups.append({'params': chain(*blocklinear_params), 'lr': optim_dict.get('blocklinear_lr', 0.001)})
            if reactivation_params:
                param_groups.append({'params': chain(*reactivation_params), 'lr': optim_dict.get('reactivation_lr', 0.001)})
            if hasattr(self.net, 'input_net'):
                param_groups.append({'params': self.net.input_net.parameters(), 'lr': optim_dict.get('input_net_lr', 0.001)})
            if output_layer_params:
                param_groups.append({'params': chain(*output_layer_params), 'lr': optim_dict.get('output_layer_lr', 0.001)})
            
            return param_groups
        else:
            return [{'params': self.parameters(), 'lr': optim_dict.get('lr', 0.001)}] 