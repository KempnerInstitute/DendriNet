"""
Variance-based TopK Linear Layer for Dendritic Networks.

This module implements a variance-based TopK approach that tracks input
activation statistics and selects weights based on input variance.
"""

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.utils.sparse_maintenance import (
    all_reduce_sums,
    reduces_statistics_across_ranks,
    sparse_state_frozen,
)
from dendritic_modeling.networks.utils.weight_transforms import (
    WeightTransformType,
    apply_weight_transform,
)

VARIANCE_SCORES = ("absolute", "relative")
_RELATIVE_SCORE_FLOOR = 1e-12


class VarianceTopKLinear(TopKLinear):
    """
    A variance-based TopK approach that tracks input activation statistics over batches
    and selects weights based on the variance of activations at each input.

    This implementation:
    1. Uses exponential moving averages to track input statistics
    2. Computes variance estimates for each input dimension
    3. Selects connections based on input variance * weight magnitude

    ``variance_score`` chooses the input statistic. ``"absolute"`` (default)
    ranks by the variance itself, so inputs coded on a small scale lose to
    large-scale inputs whatever their information: a one-hot context cue of
    amplitude 1 next to pixels under strong gain variation is never selected.
    ``"relative"`` ranks by the scale-free fraction of each input's second
    moment that varies, ``Var[x] / E[x^2]`` in [0, 1] (zero for inputs that
    have been constant zero), so input units no longer decide selection.
    """

    # Mask depends on running activation statistics that update per forward, so
    # it must not be reused across recurrent timesteps.
    _supports_recurrent_weight_cache = False
    # Set by the data-parallel wrappers: the statistics then cover the batches
    # of all ranks (see sparse_maintenance.mark_rank_reduced_statistics).
    reduce_statistics_across_ranks = False

    def __init__(
        self,
        in_features,
        out_features,
        K,
        param_space="log",
        weight_transform: WeightTransformType = "exp",
        init_method="xavier_normal",
        init_gain=1.0,
        noise_level=0.0,
        forbidden_input_index_per_output=None,
        connection_mask=None,
        momentum=0.9,  # EMA momentum for tracking statistics
        variance_score="absolute",
    ):
        super().__init__(
            in_features,
            out_features,
            K,
            param_space=param_space,
            weight_transform=weight_transform,
            init_method=init_method,
            init_gain=init_gain,
            noise_level=noise_level,
            forbidden_input_index_per_output=forbidden_input_index_per_output,
            connection_mask=connection_mask,
        )
        self.momentum = momentum
        if variance_score not in VARIANCE_SCORES:
            raise ValueError(
                f"variance_score must be one of {VARIANCE_SCORES}, got {variance_score!r}"
            )
        self.variance_score = variance_score

        # Initialize tracking buffers for activation statistics (not parameters)
        self.register_buffer("activation_mean", torch.zeros(in_features))
        self.register_buffer("activation_sq_mean", torch.zeros(in_features))
        self.register_buffer("activation_var", torch.ones(in_features))
        self.register_buffer("num_updates", torch.zeros(1))

        # Flag to track if we're in training mode
        self.stats_tracking = True

    @torch.no_grad()
    def update_stats(self, x):
        """Update activation statistics with current batch.

        The running statistics are buffers, not part of the training graph:
        they are updated without autograd so they never retain the input's
        graph (which also kept ``copy.deepcopy`` from working after a step).
        """
        if not self.stats_tracking or sparse_state_frozen():
            return

        x = x.detach()
        # Treat the final axis as features and reduce every leading token,
        # batch, or spatial axis. Sequence models commonly pass
        # [batch, sequence, features], not only two-dimensional matrices.
        if reduces_statistics_across_ranks(self):
            # Moments of the global batch: sums and counts over all ranks.
            rows = x.reshape(-1, x.shape[-1])
            total, total_sq, count = all_reduce_sums(
                rows.sum(0, dtype=torch.float64),
                (rows.double() ** 2).sum(0),
                torch.tensor(float(rows.shape[0]), device=x.device),
            )
            batch_means = (total / count).to(x.dtype)
            batch_sq_means = (total_sq / count).to(x.dtype)
        elif x.dim() > 1:
            reduction_dims = tuple(range(x.dim() - 1))
            batch_means = x.mean(dim=reduction_dims)
            batch_sq_means = (x**2).mean(dim=reduction_dims)
        else:
            batch_means = x
            batch_sq_means = x**2

        # Increment counter
        self.num_updates += 1

        # The first update takes the batch statistics, later ones the EMA. The
        # choice is made on the device, so no host read syncs the GPU.
        first = self.num_updates == 1
        keep, take = self.momentum, 1 - self.momentum
        self.activation_mean.copy_(
            torch.where(
                first, batch_means, self.activation_mean * keep + batch_means * take
            )
        )
        self.activation_sq_mean.copy_(
            torch.where(
                first,
                batch_sq_means,
                self.activation_sq_mean * keep + batch_sq_means * take,
            )
        )

        # Compute variance: E[X²] - E[X]²
        self.activation_var.copy_(self.activation_sq_mean - self.activation_mean**2)

        # Ensure variance is positive (numeric stability)
        self.activation_var.clamp_(min=1e-6)

    def forward_mutates_state(self) -> bool:
        """Whether a training forward updates the activation statistics."""
        return bool(self.stats_tracking)

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
        if self.noise_level > 0 and self.training:
            noise = torch.randn_like(self.pre_w) * self.noise_level
            w_for_topk = self.pre_w + noise

        # Calculate importance scores using weights and input variance
        # Each neuron sees the same activation variance, but has different weights
        # Expand variance to match weight shape [out_features, in_features]
        var_expanded = self._input_score().expand_as(w_for_topk)

        transformed = apply_weight_transform(w_for_topk, self.weight_transform)
        if self.weight_transform == "identity":
            # Signed weights are ranked by magnitude, as in TopKLinear.
            transformed = transformed.abs()
        importance_scores = transformed * var_expanded
        importance_scores = self._apply_forbidden_scores(importance_scores)

        # Perform topk selection per neuron
        topk_indices = torch.topk(
            importance_scores, self.K, dim=-1, largest=True, sorted=False
        )[1]

        # Create mask based on selected indices
        mask = torch.zeros_like(self.pre_w)
        # Scatter uses the selected device indices directly, without building
        # and transferring a CPU row-index tensor on every GPU forward.
        mask.scatter_(1, topk_indices, 1)
        mask = self._apply_forbidden_mask(mask)

        return mask

    def _input_score(self) -> torch.Tensor:
        """Per-input statistic multiplied with the transformed weight."""
        if self.variance_score == "absolute":
            return self.activation_var
        second_moment = self.activation_sq_mean
        raw_var = (second_moment - self.activation_mean**2).clamp_min(0.0)
        relative = torch.where(
            second_moment > _RELATIVE_SCORE_FLOOR,
            raw_var / second_moment.clamp_min(_RELATIVE_SCORE_FLOOR),
            torch.zeros_like(second_moment),
        )
        # Before any statistics exist every input scores one, as in the
        # absolute score's initial buffer, so selection starts by weight.
        return torch.where(self.num_updates > 0, relative, torch.ones_like(relative))

    def train(self, mode=True):
        """Track statistics in training mode only (``eval`` calls ``train(False)``)."""
        super().train(mode)
        self.stats_tracking = mode
        return self


__all__ = ["VARIANCE_SCORES", "VarianceTopKLinear"]
