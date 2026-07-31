"""Contextual multi-image gain overlay dataset."""

from __future__ import annotations

import numpy as np
import torch

from .poisson import PoissonGeneratorDataset


class ContextualGainDataset(PoissonGeneratorDataset):
    """Overlay ``n_layered_images`` examples with independent gains; label by max gain.

    For each requested index, layer 0 is that example and the remaining layers are
    sampled without replacement from the rest of the corpus. Each layer receives an
    independent gain on
    ``[min_gain_factor / n, max_gain_factor / n]`` (or ``_max_gf`` in place of
    ``max_gain_factor`` when set). The returned input is the gain-weighted sum of
    layers; the label is the class of the max-gain layer (lowest layer index on ties).

    If ``poisson_sampling`` is True, returns ``poisson(x)`` with no duration/tau
    rescaling.
    """

    def __init__(
        self,
        base_inputs: torch.Tensor,
        base_labels: torch.Tensor,
        n_layered_images: int = 2,
        min_gain_factor: float = 0.01,
        max_gain_factor: float = 1.0,
        gain_sampling: str = "log_uniform",
        poisson_sampling: bool = True,
    ):
        n_layered_images = int(n_layered_images)
        if n_layered_images < 1:
            raise ValueError(
                f"n_layered_images must be >= 1, got {n_layered_images}"
            )
        n_pool = int(base_inputs.shape[0])
        if n_layered_images > n_pool:
            raise ValueError(
                f"n_layered_images ({n_layered_images}) cannot exceed dataset "
                f"size ({n_pool})"
            )

        super().__init__(
            base_inputs,
            base_labels,
            multiplicative_gain=False,
            multiscale_gain=False,
            group_components=None,
            fixed_gain_factor=None,
            min_gain_factor=float(min_gain_factor),
            max_gain_factor=float(max_gain_factor),
            uniform_gain=True,
            gain_sampling=gain_sampling,
            poisson_sampling=poisson_sampling,
        )
        self.n_layered_images = n_layered_images

    @staticmethod
    def _primary_indices(idx, n_pool: int, device: torch.device) -> torch.Tensor:
        """Normalize ``idx`` to a 1-D long tensor of primary sample indices."""
        if isinstance(idx, slice):
            return torch.arange(n_pool, device=device)[idx].long()
        if torch.is_tensor(idx):
            return idx.detach().to(device=device, dtype=torch.long).reshape(-1)
        if isinstance(idx, (list, tuple, np.ndarray)):
            return torch.as_tensor(idx, device=device, dtype=torch.long).reshape(-1)
        return torch.tensor([int(idx)], device=device, dtype=torch.long)

    def _sample_layer_indices(
        self, primary: torch.Tensor, *, n_layers: int
    ) -> torch.Tensor:
        """Return ``(B, n_layers)`` indices: column 0 = primary, rest without replacement."""
        B = int(primary.shape[0])
        n_pool = int(self.base_inputs.shape[0])
        n_others = n_layers - 1
        if n_others == 0:
            return primary.unsqueeze(1)

        # Rank all pool indices by random scores; force primary to sort last so it
        # is never selected as a companion.
        scores = torch.rand(B, n_pool, device=primary.device, dtype=torch.float32)
        scores.scatter_(1, primary.unsqueeze(1), float("inf"))
        companions = scores.argsort(dim=1)[:, :n_others]
        return torch.cat([primary.unsqueeze(1), companions], dim=1)

    def _resolve_max_gain(self) -> float:
        max_gain = self._max_gf if self._max_gf is not None else self.max_gain_factor
        if max_gain is None:
            raise ValueError(
                "ContextualGainDataset requires max_gain_factor (or _set_max_gain_factor)."
            )
        if torch.is_tensor(max_gain):
            return float(max_gain.detach().cpu().item())
        return float(max_gain)

    def __getitem__(self, idx):
        batched = self._is_batched_index(idx)
        n_pool = int(self.base_inputs.shape[0])
        device = self.base_inputs.device
        n_layers = self.n_layered_images

        primary = self._primary_indices(idx, n_pool, device)
        B = int(primary.numel())
        layer_idx = self._sample_layer_indices(primary, n_layers=n_layers)

        layer_inputs = self.base_inputs[layer_idx]  # (B, n, *features)
        layer_labels = self.base_labels[layer_idx]  # (B, n)

        min_gain = self.min_gain_factor
        if min_gain is None:
            raise ValueError("ContextualGainDataset requires min_gain_factor.")
        min_gain = float(min_gain)
        max_gain = self._resolve_max_gain()

        lo = min_gain / n_layers
        hi = max_gain / n_layers
        gains = self._sample_gain_singlestage(
            lo,
            hi,
            gain_sampling=self.gain_sampling,
            shape=(B, n_layers),
            dtype=layer_inputs.dtype,
            device=device,
        )

        feature_ndim = layer_inputs.ndim - 2
        gain_view = gains.reshape(B, n_layers, *([1] * feature_ndim))
        composed = (layer_inputs * gain_view).sum(dim=1)

        winner = gains.argmax(dim=-1)
        labels = layer_labels[torch.arange(B, device=device), winner]

        if self.poisson_sampling:
            composed = torch.poisson(composed)

        if not batched:
            return composed[0], labels[0]
        return composed, labels


__all__ = ["ContextualGainDataset"]
