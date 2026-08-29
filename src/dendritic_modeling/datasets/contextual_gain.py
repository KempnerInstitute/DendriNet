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

    Optional context latents (``n_context_latents >= 2``) sample a uniform integer
    id in ``0 .. n_context_latents-1``, one-hot encode it, and embed with a fixed
    projection ``W`` of shape ``(n_context_latents, context_dim)``. Values ``0`` and
    ``1`` disable the context stream (``context_projection`` is ``None``). Every id,
    including zero, selects exactly one row of ``W``. Optional distractor noise
    (``context_distractor_scale`` in ``[0, 1)``) is mixed in before optional
    gain-scaling and Poisson sampling, then concatenated after the image stream.
    Labels use the max-gain region's class ``d`` and optional 0-based region
    index ``r*`` / context-latent scalar ``c``:

    * neither: ``y = d``
    * latents only (``region_in_label=False``): ``y = (d + c) mod n_classes``
    * ``r*`` only: ``y = (d + r*) mod n_classes``
    * both (default): ``y = (-((d + r*) mod n_classes) + c) mod n_classes``

    ``r*`` is label-only (not concatenated onto the input). Image and context
    Poisson are controlled separately via ``region_poisson`` and
    ``context_poisson`` (no duration/tau rescaling).

    Optional ``compositional_ood`` holds out one ``(r*, c)`` pair per region so
    training samples ID combinations and ``_set_OOD_sampling`` can switch to the
    held-out pairs. The split is built only when ``compositional_ood`` is True,
    ``n_context_latents >= 2``, and ``n_regions >= 2``.
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
        n_context_latents: int = 0,
        context_dim: int = 50,
        context_seed: int = 0,
        context_gain: bool = False,
        context_poisson: bool = False,
        context_distractor_scale: float = 0.0,
        region_in_label: bool = True,
        compositional_ood: bool = False,
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

        n_context_latents = int(n_context_latents)
        if n_context_latents < 0:
            raise ValueError(
                f"n_context_latents must be >= 0, got {n_context_latents}"
            )
        context_distractor_scale = float(context_distractor_scale)
        if not (0.0 <= context_distractor_scale < 1.0):
            raise ValueError(
                "context_distractor_scale must be in [0, 1), "
                f"got {context_distractor_scale}"
            )

        use_context = n_context_latents >= 2
        if use_context:
            context_dim = int(context_dim)
            if context_dim < 1:
                raise ValueError(
                    f"context_dim must be >= 1 when n_context_latents >= 2, "
                    f"got {context_dim}"
                )
            if base_inputs.ndim != 2:
                raise ValueError(
                    "ContextualGainDataset with n_context_latents >= 2 requires "
                    f"flattened base_inputs (ndim == 2), got ndim={base_inputs.ndim}"
                )

        super().__init__(
            base_inputs,
            base_labels,
            multiplicative_gain=True,
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
        self.n_context_latents = n_context_latents
        self.context_dim = int(context_dim)
        self.context_seed = int(context_seed)
        self.context_gain = bool(context_gain)
        self.context_poisson = bool(context_poisson)
        self.context_distractor_scale = context_distractor_scale
        self.region_in_label = bool(region_in_label)
        self.compositional_ood = bool(compositional_ood)
        self.n_classes = int(base_labels.max().item()) + 1

        if use_context:
            g = torch.Generator().manual_seed(self.context_seed)
            self.context_projection = torch.rand(
                self.n_context_latents, self.context_dim, generator=g
            )
        else:
            self.context_projection = None

        self._ood_enabled = (
            self.compositional_ood and use_context and n_regions >= 2
        )
        self._ood_sampling = False
        self._is_ood = None
        self._ood_ctx_idx = None
        self._id_ctx_idx = None
        if self._ood_enabled:
            self._build_context_split()

    def _build_context_split(self) -> None:
        """Hold out one context id per region; remaining ids are ID for that region."""
        n_regions = self.n_regions
        n_context = self.n_context_latents
        ood_idx = (torch.arange(n_regions) % n_context).unsqueeze(1)
        is_ood = torch.zeros(n_regions, n_context, dtype=torch.bool)
        is_ood.scatter_(1, ood_idx, True)
        id_idx = (~is_ood).nonzero(as_tuple=False)[:, 1].reshape(
            n_regions, n_context - 1
        )

        if not bool(is_ood.any(dim=1).all().item()):
            raise ValueError(
                "compositional OOD split requires at least one OOD context "
                f"per region (n_regions={n_regions}, "
                f"n_context_latents={n_context})"
            )
        if not bool((~is_ood).any(dim=1).all().item()):
            raise ValueError(
                "compositional OOD split requires at least one ID context "
                f"per region (n_regions={n_regions}, "
                f"n_context_latents={n_context})"
            )
        if bool(is_ood.all(dim=0).any().item()):
            raise ValueError(
                "compositional OOD split requires every OOD context to be ID "
                f"for at least one other region (n_regions={n_regions}, "
                f"n_context_latents={n_context})"
            )

        self._is_ood = is_ood
        self._ood_ctx_idx = ood_idx
        self._id_ctx_idx = id_idx

    def _set_OOD_sampling(self, ood: bool = True) -> None:
        if self._ood_enabled:
            self._ood_sampling = bool(ood)

    def _sample_context_ids(self, winner: torch.Tensor) -> torch.Tensor:
        """Sample context ids for each sample's winning region, shape ``(B,)``."""
        B = int(winner.shape[0])
        device = winner.device
        if not self._ood_enabled:
            return torch.randint(
                0, self.n_context_latents, (B,), device=device
            )
        table = self._ood_ctx_idx if self._ood_sampling else self._id_ctx_idx
        table = table.to(device=device)
        choice = torch.randint(0, table.shape[1], (B,), device=device)
        return table[winner, choice]

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

    def _context_id(self, ids: torch.Tensor) -> torch.Tensor:
        """One-hot encode latent ids to ``(B, n_context_latents)``."""
        return F.one_hot(ids.long().reshape(-1), self.n_context_latents)

    def _compose_label(
        self,
        digit: torch.Tensor,
        region_id: torch.Tensor,
        ctx_scalar: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compose ``d``, optional ``r*``, and optional context-latent scalar ``c``.

        Nonnegative residues via ``%`` (same as ``torch.remainder`` for ``n > 0``).
        """
        n = self.n_classes
        y = digit
        if self.region_in_label:
            y = (y + region_id) % n
        if ctx_scalar is not None:
            if self.region_in_label:
                y = (-y + ctx_scalar) % n
            else:
                y = (y + ctx_scalar) % n
        return y

    def __getitem__(self, idx):
        batched = self._is_batched_index(idx)
        n_pool = int(self.base_inputs.shape[0])
        device = self.base_inputs.device
        n_regions = self.n_regions
        use_context = self.n_context_latents >= 2
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
        digit = layer_labels[torch.arange(B, device=device), winner]
        ctx_scalar = None

        if use_context:
            ctx_scalar = self._sample_context_ids(winner)
            W = self.context_projection.to(device=device, dtype=layer_inputs.dtype)
            context_id = self._context_id(ctx_scalar).to(dtype=layer_inputs.dtype)
            ctx = context_id @ W
            if self.context_distractor_scale > 0:
                scale = self.context_distractor_scale
                ctx = (1.0 - scale) * ctx + scale * torch.rand_like(ctx)
            if g_ctx is not None:
                ctx = ctx * g_ctx.unsqueeze(-1)
            if self.context_poisson:
                ctx = torch.poisson(ctx)
            images = torch.cat([images, ctx], dim=-1)

        labels = self._compose_label(digit, winner, ctx_scalar)

        if not batched:
            return images[0], labels[0]
        return images, labels


__all__ = ["ContextualGainDataset"]
