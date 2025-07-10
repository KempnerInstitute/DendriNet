"""
Base model class for dendritic modeling.

This module contains the abstract BaseModel class that provides common functionality
for all model wrappers including parameter grouping and weight decay.
"""

from itertools import chain
from typing import Union

import torch
import torch.nn as nn

from dendritic_modeling.config import ParamGroupsConfig
from dendritic_modeling.networks import (
    BaseNetwork,
    # TopKLinear,
    # DeepstLinear,
    # BlockLinear,
    # ParametricActivation,
    DendriticBranchLayer,
    # DendriNet,
    ExcitationInhibitionNetwork,
)


class BaseModel(nn.Module):
    def __init__(
        self,
        encoder_network: BaseNetwork,
        core_network: BaseNetwork,
        decoder_network: BaseNetwork
    ):
        super(BaseModel, self).__init__()

        self.encoder_network = encoder_network

        self.core_network: Union[
            ExcitationInhibitionNetwork, BaseNetwork
        ] = core_network

        self.decoder_network = decoder_network

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_network(x)
        x = self.core_network(x)
        x = self.decoder_network(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Predict method must be implemented in subclass.")

    def decay_weights(self, weight_decay: float):
        """
        Forward this call to self.net if it implements decay_weights.
        """
        if hasattr(self.core_network, 'decay_weights'):
            self.core_network.decay_weights(weight_decay)

    def apply_rewiring(self):
        """
        Forward this call to self.net if it implements apply_rewiring.
        """
        if hasattr(self.core_network, 'apply_rewiring'):
            self.core_network.apply_rewiring()

    def get_param_groups(self, param_groups_config: ParamGroupsConfig) -> list[Dict]:
        """
        Get parameter groups for optimizer.
        """
        split_params = getattr(param_groups_config, 'split_params', True)
        if split_params and isinstance(self.core_network, ExcitationInhibitionNetwork):
            topk_params = []
            blocklinear_params = []
            reactivation_params = []
            decoder_params = []

            for module in self.core_network.modules():
                if isinstance(module, DendriticBranchLayer):
                    if module.branch_excitation is not None:
                        topk_params.append(
                            module.branch_excitation.parameters()
                        )
                    if module.branch_inhibition is not None:
                        topk_params.append(
                            module.branch_inhibition.parameters()
                        )
                    if hasattr(module, 'branches_to_output'):
                        blocklinear_params.append(
                            module.branches_to_output.parameters()
                        )
                    if module.reactivate:
                        reactivation_params.append(
                            module.reactivation.parameters()
                        )

            decoder_params.append(self.decoder_network.parameters())
            if hasattr(self, 'log_output_scale'):
                self.log_output_scale: nn.Parameter
                decoder_params.append([self.log_output_scale])

            param_groups = []
            if topk_params:
                param_groups.append({
                    'params': chain(*topk_params),
                    'lr': getattr(param_groups_config, 'topk_lr', 0.001)
                })
            if blocklinear_params:
                param_groups.append({
                    'params': chain(*blocklinear_params),
                    'lr': getattr(param_groups_config, 'blocklinear_lr', 0.001)
                })
            if reactivation_params:
                param_groups.append({
                    'params': chain(*reactivation_params),
                    'lr': getattr(param_groups_config, 'reactivation_lr', 0.001)
                })

            # transfer_fn_params = []
            # for module in self.core_network.transfer_fn.modules():
            #     if (
            #         not isinstance(module, DendriNet)
            #         and not isinstance(module, DendriticBranchLayer)
            #         and not isinstance(module,BlockLinear)
            #         and not isinstance(module,TopKLinear)
            #         and not isinstance(module,DeepstLinear)
            #         and not isinstance(module,ParametricActivation)
            #     ):
            #         transfer_fn_params.append(module.parameters())

            # param_groups.append({
            #     'params': chain(*transfer_fn_params),
            #     'lr': optim_dict.get('transfer_fn_lr', 0.001)
            # })

            if decoder_params:
                param_groups.append({
                    'params': chain(*decoder_params),
                    'lr': getattr(param_groups_config, 'decoder_lr', 0.001)
                })

            return param_groups
        else:
            return [{
                'params': self.parameters(),
                'lr': getattr(param_groups_config, 'lr', 0.001)
            }]

__all__ = ["BaseModel"]
