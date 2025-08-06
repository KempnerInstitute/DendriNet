"""
Variance-based TopK Linear Layer for Dendritic Networks.

This module implements a variance-based TopK approach that tracks input
activation statistics and selects weights based on input variance.
"""

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import \
    TopKLinear


class VarianceTopKLinear(TopKLinear):
    """
    A variance-based TopK approach that tracks input activation statistics over batches
    and selects weights based on the variance of activations at each input.

    This implementation:
    1. Uses exponential moving averages to track input statistics
    2. Computes variance estimates for each input dimension
    3. Selects connections based on input variance * weight magnitude
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
        momentum=0.9,  # EMA momentum for tracking statistics
    ):
        super().__init__(
            in_features,
            out_features,
            K,
            param_space=param_space,
            init_method=init_method,
            init_gain=init_gain,
            noise_level=noise_level,
        )
        self.momentum = momentum

        # Initialize tracking buffers for activation statistics (not parameters)
        self.register_buffer("activation_mean", torch.zeros(in_features))
        self.register_buffer("activation_sq_mean", torch.zeros(in_features))
        self.register_buffer("activation_var", torch.ones(in_features))
        self.register_buffer("num_updates", torch.zeros(1))

        # Flag to track if we're in training mode
        self.stats_tracking = True

    def update_stats(self, x):
        """Update activation statistics with current batch."""
        if not self.stats_tracking:
            return

        # If this is a batched input, average over the batch
        if x.dim() > 1:
            batch_means = x.mean(dim=0)
            batch_sq_means = (x**2).mean(dim=0)
        else:
            batch_means = x
            batch_sq_means = x**2

        # Increment counter
        self.num_updates += 1

        # Apply exponential moving average updates
        if self.num_updates == 1:
            # First update, just use the batch stats
            self.activation_mean.copy_(batch_means)
            self.activation_sq_mean.copy_(batch_sq_means)
        else:
            # EMA update
            self.activation_mean.mul_(self.momentum).add_(
                batch_means * (1 - self.momentum)
            )
            self.activation_sq_mean.mul_(self.momentum).add_(
                batch_sq_means * (1 - self.momentum)
            )

        # Compute variance: E[X²] - E[X]²
        self.activation_var = self.activation_sq_mean - self.activation_mean**2

        # Ensure variance is positive (numeric stability)
        self.activation_var.clamp_(min=1e-6)

    def forward(self, x):
        # Update statistics during forward pass
        self.update_stats(x)

        # Use parent class forward implementation to avoid code duplication
        return super().forward(x)

    def weight_mask(self):
        """
        Select TopK connections based on a combination of:
        - Input activation variance
        - Weight magnitude

        The intuition is that we want to keep connections to inputs that:
        1. Have high variance (carry more information)
        2. Have strong weights (model cares about them)
        """
        # Determine weights for topk selection
        w_for_topk = self.pre_w
        if self.noise_level > 0:
            noise = torch.randn_like(self.pre_w) * self.noise_level
            w_for_topk = self.pre_w + noise

        # Calculate importance scores using weights and input variance
        # Each neuron sees the same activation variance, but has different weights
        # Expand variance to match weight shape [out_features, in_features]
        var_expanded = self.activation_var.expand_as(w_for_topk)

        # Score is weight magnitude * input variance
        if self.param_space == "log":
            # For log param space, we use exp(pre_w) * variance
            importance_scores = w_for_topk.exp() * var_expanded
        else:
            # For presigmoid, use sigmoid(pre_w) * variance
            importance_scores = torch.sigmoid(w_for_topk) * var_expanded

        # Perform topk selection per neuron
        topk_indices = torch.topk(
            importance_scores, self.K, dim=-1, largest=True, sorted=False
        )[1]

        # Create mask based on selected indices
        mask = torch.zeros_like(self.pre_w)
        mask[torch.arange(self.pre_w.shape[0])[:, None], topk_indices] = 1

        return mask

    def train(self, mode=True):
        """Override train method to control stats tracking."""
        super().train(mode)
        self.stats_tracking = mode
        return self

    def eval(self):
        """Override eval method to freeze stats in eval mode."""
        super().eval()
        self.stats_tracking = False
        return self


__all__ = ["VarianceTopKLinear"]
