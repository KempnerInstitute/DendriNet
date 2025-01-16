"""
models.py
=========
This module contains the neural network models for the dendritic_modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from dendritic_modeling.excitation_inhibition import ExcitationInhibitionNetwork

class ProbabilisticModel(nn.Module):
    """
    This class is designed to represent models that return a conditional 
    probability distribution of observing labels given their corresponding 
    inputs.
    """
    pass

    def log_prob(self, x, y):
        distribution = self.forward(x)
        return distribution.log_prob(y)


class ProbabilisticClassifier(ProbabilisticModel):
    """
    A base class for probabilistic classifiers.
    """
    def predict(self, x, stochastic=False):
        distribution = self.forward(x)
        if stochastic:
            return distribution.sample()
        else:
            return distribution.logits.argmax(dim=-1)


class ProbabilisticRegressor(ProbabilisticModel):
    """
    A base class for probabilistic regressors.
    """
    def predict(self, x, stochastic=True):
        distribution = self.forward(x)
        if stochastic:
            return distribution.sample()
        else:
            return distribution.mean


class EINetClassifier(ProbabilisticClassifier):
    """
    A probabilistic classifier based on an Excitation-Inhibition Network (EINet) 
    architecture.
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
        reactivate=False, 
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        use_shunting=True,

        # ------------------- NEW CODE: optional config fields -------------------
        distance_scaling_enabled=False,
        distance_scale_factor=1.0,
        distance_scale_trainable=False,

        reactivation_slope_scaling_enabled=False,
        reactivation_slope_factor=1.0,
        reactivation_slope_trainable=False,

        forward_hook_enabled=False,
        forward_hook_scale=1.0
    ):
        super(EINetClassifier, self).__init__()

        # (1) Possibly treat distance_scale_factor as a trainable parameter:
        if distance_scaling_enabled and distance_scale_trainable:
            self.distance_scale = nn.Parameter(torch.tensor(distance_scale_factor, dtype=torch.float32))
        else:
            self.register_buffer("distance_scale", torch.tensor(distance_scale_factor, dtype=torch.float32))
        self.distance_scaling_enabled = distance_scaling_enabled

        # (2) Possibly treat reactivation_slope_factor as a trainable parameter:
        if reactivation_slope_scaling_enabled and reactivation_slope_trainable:
            self.reactivation_slope = nn.Parameter(torch.tensor(reactivation_slope_factor, dtype=torch.float32))
        else:
            self.register_buffer("reactivation_slope", torch.tensor(reactivation_slope_factor, dtype=torch.float32))
        self.reactivation_slope_scaling_enabled = reactivation_slope_scaling_enabled

        # (3) forward_hook scale
        self.forward_hook_enabled = forward_hook_enabled
        self.forward_hook_scale = forward_hook_scale

        # Construct the underlying EINet
        self.net = ExcitationInhibitionNetwork(
            input_dim=input_dim, 
            excitatory_layer_sizes=excitatory_layer_sizes, 
            inhibitory_layer_sizes=inhibitory_layer_sizes,
            excitatory_branch_factors=excitatory_branch_factors, 
            inhibitory_branch_factors=inhibitory_branch_factors,
            ee_synapses_per_branch_per_layer=ee_synapses_per_branch_per_layer, 
            ei_synapses_per_branch_per_layer=ei_synapses_per_branch_per_layer,
            ie_synapses_per_branch_per_layer=ie_synapses_per_branch_per_layer, 
            ii_synapses_per_branch_per_layer=ii_synapses_per_branch_per_layer,
            reactivate=reactivate, 
            somatic_synapses=somatic_synapses,
            topk_init_method=topk_init_method,
            use_shunting=use_shunting,

            # pass the new scaling stuff downward if needed
            distance_scaling_enabled=self.distance_scaling_enabled,
            distance_scale=self.distance_scale,  # could be param or buffer
            reactivation_slope_scaling_enabled=self.reactivation_slope_scaling_enabled,
            reactivation_slope=self.reactivation_slope,
            forward_hook_enabled=self.forward_hook_enabled,
            forward_hook_scale=self.forward_hook_scale
        )

        # final learned scaling of the output dimension
        self.log_output_scale = nn.Parameter(
            torch.zeros((excitatory_layer_sizes[-1],)), requires_grad=True
        )
        self.final = nn.LogSoftmax(dim=-1)
    
    def decay_weights(self, weight_decay=0.1):
        self.net.decay_weights(weight_decay)

    def forward(self, x):
        # pass input forward
        net_output = self.net(x)
        # multiply by final output scale factor
        final_logits = self.final(self.log_output_scale.exp() * net_output)
        return Categorical(logits=final_logits)


class EINetClassifierNLL(EINetClassifier):
    """
    Same architecture as EINetClassifier, but returns negative log-likelihood 
    for the labeled data if y is provided.
    """
    def forward(self, x, y=None, sampler=False):
        net_output = self.net(x)
        final_logits = self.final(self.log_output_scale.exp() * net_output)
        categorical_dist = Categorical(logits=final_logits)

        if y is None:
            return categorical_dist if sampler else categorical_dist.logits
        else:
            nll = -categorical_dist.log_prob(y)
            return nll
    
    def predict(self, x, stochastic=False):
        distribution = self.forward(x, y=None, sampler=True)
        if stochastic:
            return distribution.sample()
        else:
            return distribution.logits.argmax(dim=-1)