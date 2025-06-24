"""
Base model class for dendritic modeling.

This module contains the abstract BaseModel class that provides common functionality
for all model wrappers including parameter grouping and weight decay.
"""

import torch
import torch.nn as nn
from itertools import chain
from typing import Union

from dendritic_modeling.networks import (
    DendriticBranchLayer,
    DendriNet,
    ExcitationInhibitionLayer,
    ExcitationInhibitionNetwork,
    BaseNetwork
)


class BaseModel(nn.Module):
    def __init__(self, net: BaseNetwork):
        super(BaseModel, self).__init__()
        self.net: Union[ExcitationInhibitionNetwork, BaseNetwork] = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method must be implemented in subclass.")

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Predict method must be implemented in subclass.")
    
    def decay_weights(self, weight_decay: float):
        """
        Forward this call to self.net if it implements decay_weights.
        """
        if hasattr(self.net, 'decay_weights'):
            self.net.decay_weights(weight_decay)
    
    def apply_rewiring(self):
        """
        Forward this call to self.net if it implements apply_rewiring.
        """
        if hasattr(self.net, 'apply_rewiring'):
            self.net.apply_rewiring()
    
    def get_param_groups(self, optim_dict: dict) -> list[dict]:
        """
        Get parameter groups for optimizer.
        """
        split_params = optim_dict.get("split_params", False)
        if split_params and isinstance(self.net, ExcitationInhibitionNetwork):
            topk_params = []
            blocklinear_params = []
            reactivation_params = []
            output_layer_params = []

            for ei_layer in self.net.layers:
                ei_layer: ExcitationInhibitionLayer
                exc_cells: DendriNet = ei_layer.excitatory_cells

                for dendritic_layer in exc_cells.branch_layers:
                    dendritic_layer: DendriticBranchLayer
                    if dendritic_layer.branch_excitation is not None:
                        topk_params.append(
                            dendritic_layer.branch_excitation.parameters()
                        )
                    if dendritic_layer.branch_inhibition is not None:
                        topk_params.append(
                            dendritic_layer.branch_inhibition.parameters()
                        )
                    
                    if hasattr(dendritic_layer, 'branches_to_output'):
                        blocklinear_params.append(
                            dendritic_layer.branches_to_output.parameters()
                        )
                    
                    if dendritic_layer.reactivate:
                        reactivation_params.append(
                            dendritic_layer.reactivation.parameters()
                        )
                
                if hasattr(ei_layer, 'inhibitory_cells'):
                    inh_cells = ei_layer.inhibitory_cells
                    for dendritic_layer in inh_cells.branch_layers:
                        if dendritic_layer.branch_excitation is not None:
                            topk_params.append(
                                dendritic_layer.branch_excitation.parameters()
                            )
                        if dendritic_layer.branch_inhibition is not None:
                            topk_params.append(
                                dendritic_layer.branch_inhibition.parameters()
                            )
                        
                        if hasattr(dendritic_layer, 'branches_to_output'):
                            blocklinear_params.append(
                                dendritic_layer.branches_to_output.parameters()
                            )
                        
                        if dendritic_layer.reactivate:
                            reactivation_params.append(
                                dendritic_layer.reactivation.parameters()
                            )
            
            if hasattr(self, 'final_fc'):
                self.final_fc: nn.Module
                output_layer_params.append(self.final_fc.parameters())
            if hasattr(self, 'log_output_scale'):
                self.log_output_scale: nn.Parameter
                output_layer_params.append([self.log_output_scale])

            param_groups = []
            if topk_params:
                param_groups.append({
                    'params': chain(*topk_params), 
                    'lr': optim_dict.get('topk_lr', 0.001)
                })
            if blocklinear_params:
                param_groups.append({
                    'params': chain(*blocklinear_params), 
                    'lr': optim_dict.get('blocklinear_lr', 0.001)
                })
            if reactivation_params:
                param_groups.append({
                    'params': chain(*reactivation_params), 
                    'lr': optim_dict.get('reactivation_lr', 0.001)
                })
            if hasattr(self.net, 'transfer_fn'):
                transfer_fn: nn.Module = self.net.transfer_fn
                param_groups.append({
                    'params': transfer_fn.parameters(), 
                    'lr': optim_dict.get('input_net_lr', 0.001)
                })
            if output_layer_params:
                param_groups.append({
                    'params': chain(*output_layer_params), 
                    'lr': optim_dict.get('output_layer_lr', 0.001)
                })
            
            return param_groups
        else:
            return [{
                'params': self.parameters(), 
                'lr': optim_dict.get('lr', 0.001)
            }]
        

__all__ = ["BaseModel"]