"""Contextual multi-image gain overlay dataset."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .poisson import PoissonGeneratorDataset


class ContextualGainDataset(PoissonGeneratorDataset):
    """Concatenate ``n_regions`` examples with independent gains; label by max gain.

    For each requested index, region 0 is that example and the remaining regions are
    sampled without replacement from the rest of the corpus. Each region receives an
    independent gain on ``[min_gain_factor, max_gain_factor]`` (or ``_max_gf`` in place
    of ``max_gain_factor`` when set). Regions are concatenated along ``dim=-1``.

    Optional context bits are mapped to a one-hot ``context_id`` of length
    ``n_context_ids`` (``2**k`` for ``bit_id``, ``2`` for ``xor``), then
    embedded with a fixed projection ``W`` of shape
    ``(n_context_ids, context_dim)``. Every bit pattern, including all zeros,
    selects exactly one row of ``W``. Optional distractor noise
    (``context_distractor_scale`` in ``[0, 1)``) is mixed in before optional
    gain-scaling and Poisson sampling, then concatenated after the image stream.
    Labels are ``(max_gain_digit + ctx) mod n_classes`` where ``ctx`` comes from
    ``context_rule`` (``bit_id`` or ``xor``).

    Image and context Poisson are controlled separately via ``region_poisson`` and
    ``context_poisson`` (no duration/tau rescaling).
    """

    def __init__(
        self,
        base_inputs: torch.Tensor,
        base_labels: torch.Tensor,
        n_regions: int = 2,
        min_gain_factor: float = 0.01,
        max_gain_factor: float = 1.0,
        gain_sampling: str = "log_uniform",
        region_poisson: bool = True,
        n_context_bits: int = 0,
        context_dim: int = 50,
        context_rule: str = "bit_id",
        context_seed: int = 0,
        context_gain: bool = False,
        context_poisson: bool = False,
        context_distractor_scale: float = 0.0,
    ):
        n_regions = int(n_regions)
        if n_regions < 1:
            raise ValueError(
                f"n_regions must be >= 1, got {n_regions}"
            )
        n_pool = int(base_inputs.shape[0])
        if n_regions > n_pool:
            raise ValueError(
                f"n_regions ({n_regions}) cannot exceed dataset "
                f"size ({n_pool})"
            )

        n_context_bits = int(n_context_bits)
        if n_context_bits < 0:
            raise ValueError(
                f"n_context_bits must be >= 0, got {n_context_bits}"
            )
        context_rule = str(context_rule).strip().lower()
        if context_rule not in {"bit_id", "xor"}:
            raise ValueError(
                "context_rule must be 'bit_id' or 'xor', "
                f"got {context_rule!r}"
            )
        context_distractor_scale = float(context_distractor_scale)
        if not (0.0 <= context_distractor_scale < 1.0):
            raise ValueError(
                "context_distractor_scale must be in [0, 1), "
                f"got {context_distractor_scale}"
            )

        if n_context_bits > 0:
            context_dim = int(context_dim)
            if context_dim < 1:
                raise ValueError(
                    f"context_dim must be >= 1 when n_context_bits > 0, "
                    f"got {context_dim}"
                )
            if base_inputs.ndim != 2:
                raise ValueError(
                    "ContextualGainDataset with n_context_bits > 0 requires "
                    f"flattened base_inputs (ndim == 2), got ndim={base_inputs.ndim}"
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
            poisson_sampling=False,
        )
        self.n_regions = n_regions
        self.region_poisson = bool(region_poisson)
        self.n_context_bits = n_context_bits
        self.context_dim = int(context_dim)
        self.context_rule = context_rule
        self.context_seed = int(context_seed)
        self.context_gain = bool(context_gain)
        self.context_poisson = bool(context_poisson)
        self.context_distractor_scale = context_distractor_scale
        self.n_classes = int(base_labels.max().item()) + 1

        if n_context_bits > 0:
            if self.context_rule == "xor":
                self.n_context_ids = 2
            else:
                self.n_context_ids = 2 ** n_context_bits
            g = torch.Generator().manual_seed(self.context_seed)
            self.context_projection = torch.rand(
                self.n_context_ids, self.context_dim, generator=g
            )
        else:
            self.n_context_ids = 0
            self.context_projection = None

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

    def _context_scalar(self, bits: torch.Tensor) -> torch.Tensor:
        """Map bit patterns ``(B, k)`` to discrete context scalars ``(B,)``."""
        if self.context_rule == "xor":
            return bits.sum(dim=-1) % 2
        # bit_id: MSB-first integer encode
        k = bits.shape[-1]
        powers = (2 ** torch.arange(k - 1, -1, -1, device=bits.device)).to(bits.dtype)
        return (bits * powers).sum(dim=-1)

    def _context_id(self, bits: torch.Tensor) -> torch.Tensor:
        """One-hot encode bit patterns to ``(B, n_context_ids)``."""
        ctx_scalar = self._context_scalar(bits).long()
        return F.one_hot(ctx_scalar, self.n_context_ids).to(dtype=bits.dtype)

    def __getitem__(self, idx):
        batched = self._is_batched_index(idx)
        n_pool = int(self.base_inputs.shape[0])
        device = self.base_inputs.device
        n_regions = self.n_regions
        use_context = self.n_context_bits > 0
        use_context_gain = use_context and self.context_gain

        primary = self._primary_indices(idx, n_pool, device)
        B = int(primary.numel())
        layer_idx = self._sample_layer_indices(primary, n_layers=n_regions)

        layer_inputs = self.base_inputs[layer_idx]  # (B, n, *features)
        layer_labels = self.base_labels[layer_idx]  # (B, n)

        min_gain = self.min_gain_factor
        if min_gain is None:
            raise ValueError("ContextualGainDataset requires min_gain_factor.")
        min_gain = float(min_gain)
        max_gain = self._resolve_max_gain()

        n_gain_streams = n_regions + (1 if use_context_gain else 0)
        gains = self._sample_gain_singlestage(
            min_gain,
            max_gain,
            gain_sampling=self.gain_sampling,
            shape=(B, n_gain_streams),
            dtype=layer_inputs.dtype,
            device=device,
        )
        if use_context_gain:
            g_ctx = gains[..., 0]
            g_img = gains[..., 1:]
        else:
            g_ctx = None
            g_img = gains

        feature_ndim = layer_inputs.ndim - 2
        gain_view = g_img.reshape(B, n_regions, *([1] * feature_ndim))
        images = torch.cat((layer_inputs * gain_view).unbind(dim=1), dim=-1)
        if self.region_poisson:
            images = torch.poisson(images)

        winner = g_img.argmax(dim=-1)
        labels = layer_labels[torch.arange(B, device=device), winner]

        if use_context:
            bits = torch.randint(
                0, 2, (B, self.n_context_bits), device=device, dtype=layer_inputs.dtype
            )
            W = self.context_projection.to(device=device, dtype=layer_inputs.dtype)
            context_id = self._context_id(bits)
            ctx_scalar = context_id.argmax(dim=-1)
            ctx = context_id @ W
            if self.context_distractor_scale > 0:
                scale = self.context_distractor_scale
                ctx = (1.0 - scale) * ctx + scale * torch.rand_like(ctx)
            if g_ctx is not None:
                ctx = ctx * g_ctx.unsqueeze(-1)
            if self.context_poisson:
                ctx = torch.poisson(ctx)
            images = torch.cat([images, ctx], dim=-1)
            labels = (labels + ctx_scalar) % self.n_classes

        if not batched:
            return images[0], labels[0]
        return images, labels


__all__ = ["ContextualGainDataset"]
