"""
TopK Linear Layer for Dendritic Networks.

This module implements the basic TopKLinear layer that retains only the top K
strongest synaptic weights for each output neuron, simulating selective
synaptic connectivity in dendritic computation.
"""

import torch
import torch.nn as nn
from math import log

TOPK_INIT_METHODS = {
    "xavier_normal": nn.init.xavier_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "orthogonal": nn.init.orthogonal_,
    "normal": nn.init.normal_,
    "uniform": nn.init.uniform_,
    "eye": nn.init.eye_,
}

class TopKLinear(nn.Module):
    """
    A linear layer that retains only the top K strongest synaptic weights for
    in_features.

    This module implements a linear transformation with weights constrained to
    be positive. For each output neuron, only the top K weights are kept
    , and the rest are set to zero. This simulates a neuron receiving
    inputs only from its strongest synaptic connections.

    Parameters
    ----------
    in_features : int
        The number of input features.
    out_features : int
        The number of output features.
    K : int
        The number of strongest synapses to keep per dendritic branch.
    param_space : str, optional
        The parameter space for the weights. Options are 'log' and 'presigmoid'.
        Defaults to 'log'.

    Notes
    -----
    - All weights are constrained to be positive.
    - The pruning is done dynamically during the forward pass.
    """

    def __init__(
        self,
        in_features,
        out_features,
        K,
        param_space="log",
        init_method="xavier_normal",
        init_gain=1.0,
        noise_level=0.0,
    ):
        super(TopKLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method
        self.init_gain = init_gain
        self.noise_level = noise_level

        self.pre_w = nn.Parameter(
            torch.empty((out_features, in_features)), requires_grad=True
        )

        if not isinstance(K, int):
            K = int(K)  # Ensure K is an integer

        if K < 1:
            raise ValueError("K must be >= 1")

        if K > in_features:
            raise ValueError(
                f"K must be <= number of input features. (K = {K}, in_features = {in_features})"
            )

        self.K = K
        self.param_space = param_space
        self.input_scale_vec = None  # may be used for gradient scaling
        self.param_scale_vec = None  # may be used for gradient scaling

    def initialize(self):
        """Initialize the layer weights using the specified method."""
        if self.init_method in TOPK_INIT_METHODS:
            init_func = TOPK_INIT_METHODS[self.init_method]
            init_func(self.pre_w)
            self.pre_w.data.mul_(self.init_gain)  # scale
        else:
            raise ValueError(
                f"Invalid initialization method: {self.init_method}. "
                f"Choose from {list(TOPK_INIT_METHODS.keys())}"
            )

    def decay_weights(self, weight_decay=0.1, weight_boost=False):
        """Apply weight decay to active synapses and optionally boost inactive ones."""
        with torch.no_grad():
            weight_mask = self.weight_mask()
            log_decay = log(1 - weight_decay) * weight_mask
            self.pre_w.data += log_decay  # suppress active synapses
            if weight_boost:
                log_boost = log(1 + weight_decay) * (1 - weight_mask)
                self.pre_w.data -= log_boost  # boost inactive synapses

    def forward(self, x):
        """Forward pass through the TopK linear layer."""
        pruned_weight = self.pruned_weight()

        # Ensure x is a 2D tensor for matrix multiplication
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")

        # If x is a 1D tensor, unsqueeze to make it 2D [1, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        return torch.matmul(x, pruned_weight.t())

    def weight(self):
        """Return the transformed synaptic weights."""
        if self.param_space == "log":
            return self.pre_w.exp()
        elif self.param_space == "presigmoid":
            return torch.sigmoid(self.pre_w)

    def log_weight(self):
        """Return the log of the transformed weights."""
        return self.weight().log()

    def weight_mask(self):
        """Return a mask indicating the top K synaptic connections per output neuron."""
        # Determine which weights to use for topk selection
        weights_for_topk = self.pre_w
        if self.noise_level > 0:
            # Generate Gaussian noise with mean=0 and std=noise_level
            noise = torch.randn_like(self.pre_w) * self.noise_level
            # Add noise to the raw weights
            weights_for_topk = self.pre_w + noise

        # Perform topk on the (potentially noisy) weights
        topK_indices = torch.topk(
            weights_for_topk, self.K, dim=-1, largest=True, sorted=False
        )[1]

        # Create the mask based on the selected indices
        mask = torch.zeros_like(
            self.pre_w,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        mask[torch.arange(self.pre_w.shape[0])[:, None], topK_indices] = 1
        return mask

    def pruned_weight(self):
        """Return the pruned synaptic weights after applying the mask."""
        return self.weight_mask() * self.weight()

    def log_pruned_weight(self):
        """Return the log of the pruned weights."""
        return ((self.weight_mask() - 1) * 10) + self.log_weight()

    def weighted_synapses(self, cell_weights, prune=False):
        """Return weighted synapses for a given set of cell weights."""
        if prune:
            synapse_weights = self.pruned_weight()
        else:
            synapse_weights = self.weight()

        weighted_synapses = cell_weights[:, None] * synapse_weights
        return weighted_synapses.sum(dim=0)

__all__ = ["TopKLinear", "TOPK_INIT_METHODS"]
