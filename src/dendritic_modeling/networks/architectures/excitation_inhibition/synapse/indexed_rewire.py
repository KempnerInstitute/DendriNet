"""Adaptive rewiring indexed sparse synapses."""

from __future__ import annotations

from math import ceil

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _connection_generator,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TOPK_INIT_METHODS,
)


class IndexedRewireLinear(IndexedSparseLinear):
    """
    Memory-efficient sparse layer with local adaptive-quantile rewiring.

    The layer stores exactly ``K`` synapses per output row, like
    :class:`IndexedSparseLinear`, and periodically replaces the weakest
    fraction of those synapses with newly sampled allowed input indices. The
    rewiring is always local to the same output row, so dendritic branch
    synapse budgets stay fixed while weak synapses are given a biologically
    inspired turnover path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int,
        *,
        rewire_frequency: int = 100,
        rewire_quantile: float = 0.05,
        rewire_until_step: int | None = None,
        freeze_connectivity: bool = False,
        seed: int | None = None,
        **kwargs: object,
    ):
        if rewire_frequency < 1:
            raise ValueError("rewire_frequency must be >= 1")
        if not 0.0 <= float(rewire_quantile) <= 1.0:
            raise ValueError("rewire_quantile must be in [0, 1]")
        if rewire_until_step is not None and int(rewire_until_step) < 0:
            raise ValueError("rewire_until_step must be >= 0 when provided")

        self._rewire_seed = (
            int(seed)
            if seed is not None
            else int(torch.empty((), dtype=torch.int64).random_().item())
        )
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            K=K,
            seed=seed,
            **kwargs,
        )

        self.rewire_frequency = int(rewire_frequency)
        self.rewire_quantile = float(rewire_quantile)
        self.rewire_until_step = (
            None if rewire_until_step is None else int(rewire_until_step)
        )
        self.freeze_connectivity = bool(freeze_connectivity)
        self.register_buffer(
            "rewire_step", torch.zeros((), dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "_last_rewired_mask",
            torch.zeros_like(self.pre_w, dtype=torch.bool),
            persistent=False,
        )

    def _allowed_indices_for_row_cpu(self, row_idx: int) -> torch.Tensor:
        if self._allowed_connection_mask.numel() == 0:
            return torch.arange(self.in_features, dtype=torch.long)
        return torch.nonzero(
            self._allowed_connection_mask[row_idx].detach().cpu(), as_tuple=False
        ).flatten()

    def _new_pre_weights(self, shape: tuple[int, int]) -> torch.Tensor:
        temp = torch.empty(shape, device=self.pre_w.device, dtype=self.pre_w.dtype)
        if self.init_method not in TOPK_INIT_METHODS:
            raise ValueError(
                f"Invalid initialization method: {self.init_method}. "
                f"Choose from {list(TOPK_INIT_METHODS.keys())}"
            )
        TOPK_INIT_METHODS[self.init_method](temp)
        temp.mul_(self.init_gain)
        return temp

    def apply_rewiring(self) -> None:
        """Replace the weakest active sparse slots with new random inputs."""
        with torch.no_grad():
            self.rewire_step.add_(1)
            self._last_rewired_mask.zero_()

            step = int(self.rewire_step.item())
            if self.freeze_connectivity or self.rewire_quantile <= 0.0:
                return
            if self.rewire_until_step is not None and step > self.rewire_until_step:
                return
            if step % self.rewire_frequency != 0:
                return

            n_rewire = min(self.K, max(1, ceil(self.K * self.rewire_quantile)))
            scores = self.sparse_weight().detach().abs()
            prune_slots = torch.topk(
                scores, n_rewire, dim=1, largest=False, sorted=False
            ).indices
            fresh_weights = self._new_pre_weights((self.out_features, n_rewire))

            conn_cpu = self.connection_indices.detach().cpu()
            seed_base = self._rewire_seed + 104729 * step
            for row_idx in range(self.out_features):
                available_mask = torch.zeros(self.in_features, dtype=torch.bool)
                allowed = self._allowed_indices_for_row_cpu(row_idx)
                available_mask[allowed] = True
                available_mask[conn_cpu[row_idx]] = False
                available = torch.nonzero(available_mask, as_tuple=False).flatten()
                if available.numel() == 0:
                    continue

                n_select = min(n_rewire, int(available.numel()))
                generator = _connection_generator(seed_base + row_idx)
                selected = available[
                    torch.randperm(available.numel(), generator=generator)[:n_select]
                ].to(self.connection_indices.device)
                target_slots = prune_slots[row_idx, :n_select]
                self.connection_indices[row_idx, target_slots] = selected
                self.pre_w.data[row_idx, target_slots] = fresh_weights[
                    row_idx, :n_select
                ]
                self._last_rewired_mask[row_idx, target_slots] = True

            if self.pre_w.grad is not None and bool(self._last_rewired_mask.any()):
                self.pre_w.grad.data[self._last_rewired_mask] = 0

    def consume_rewired_mask(self) -> torch.Tensor | None:
        """Return and clear the mask of slots rewired by the last call."""
        if not bool(self._last_rewired_mask.any()):
            return None
        mask = self._last_rewired_mask.detach().clone()
        self._last_rewired_mask.zero_()
        return mask


__all__ = ["IndexedRewireLinear"]
