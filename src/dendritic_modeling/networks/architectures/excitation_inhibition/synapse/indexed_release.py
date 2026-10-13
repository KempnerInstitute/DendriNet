"""Indexed positive synapses with independent strength-dependent release."""

import math
from typing import ClassVar

import torch

from dendritic_modeling.networks.state_versions import (
    file_predates,
    record_legacy_state,
)
from dendritic_modeling.networks.utils.signal_rounding import (
    round_signal,
    signal_rounding_dtype,
)
from dendritic_modeling.networks.utils.sparse_maintenance import (
    all_reduce_sums,
    reduces_statistics_across_ranks,
    sparse_state_frozen,
)
from dendritic_modeling.networks.utils.weight_transforms import inverse_weight_transform

from .indexed_sparse import IndexedSparseLinear

# Registry of contact-impact statistics used by cellwise structural plasticity.
# The value names the stage at which the exponential average is updated:
# "forward" metrics need presynaptic inputs and are accumulated in training
# forwards; "step" metrics are accumulated by the rewiring controller after each
# optimizer step.
RELEASE_IMPACT_METRICS = {
    # Mean |x * w * release| over the batch (default).
    "drive": "forward",
    # Mean |x * w * release| over examples whose presynaptic source is active.
    # Rarely active inputs, such as one-hot cues, are not penalized for rarity.
    "active_drive": "forward",
    # Contact conductance w (magnitude-based plasticity).
    "weight": "step",
    # First-order loss importance |w * dL/dw|.
    "saliency": "step",
}
RELEASE_WEIGHT_TRANSFORMS = ("exp", "softplus")


class IndexedReleaseLinear(IndexedSparseLinear):
    """Stored contacts only; Bernoulli release per example/contact in training.

    p(w) = p_min + (p_max-p_min) * w / (w + 0.25 * weight_scale).
    Evaluation uses w*p(w). Training differentiates the realized conductance,
    not the Bernoulli draw. ``release_impact_metric`` selects the contact
    statistic averaged with ``release_impact_decay`` (see
    ``RELEASE_IMPACT_METRICS``); the default is absolute realized input drive,
    measured before integration. Connectivity changes are performed by the
    cellwise rewiring controller.

    Opt-in options (defaults preserve the original state, output and random
    streams exactly): ``weight_transform="softplus"`` bounds conductance growth
    to linear in the raw parameter; ``release_max_conductance`` caps each
    effective conductance (no gradient above the cap); and
    ``release_track_source_activity`` keeps a persistent average of each
    presynaptic source's activity for activity-aware regrowth.
    ``signal_rounding`` ("float16" or "bfloat16") rounds the presynaptic
    signals, the conductances, the mean-transmission probabilities and each
    output's summed drive to that format; the sums themselves are accumulated
    in the parameter dtype (see ``networks.utils.signal_rounding``).

    A provided ``connection_mask`` is checkpointed, so cellwise rewiring after
    a resume samples new sources from the original mask.
    """

    # Per-contact statistics; release draws in training forwards and the
    # cellwise impact rewiring (candidate order, regrowth slots) follow slot
    # positions.
    CONTACT_SLOT_STATE: ClassVar[tuple[str, ...]] = ("pre_w", "impact", "contact_age")
    SLOT_ORDER_MATTERS: ClassVar[bool] = True

    # Set by the data-parallel wrappers: source activity and forward impact
    # then cover the batches of all ranks.
    reduce_statistics_across_ranks = False

    _persist_allowed_connection_mask = True
    # Version 2 (2026-10-09) marks contact ages in optimizer steps. Version-1
    # files written before 2026-10-04 (1dc4f8505) counted training forwards.
    _version = 2

    def __init__(
        self,
        *args,
        release_p_min=0.5,
        release_p_max=1.0,
        release_weight_scale=None,
        release_impact_decay=0.99,
        release_impact_metric="drive",
        release_max_conductance=None,
        release_track_source_activity=False,
        signal_rounding="none",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if (
            self.weight_transform not in RELEASE_WEIGHT_TRANSFORMS
            or self.weight_norm_order is not None
        ):
            raise ValueError(
                "indexed_release requires exp or softplus weights without normalization"
            )
        if (
            self.projection_backend != "eager"
            or self.cache_transformed_weights
            or not self.persistent_indices
        ):
            raise ValueError(
                "indexed_release requires eager execution, persistent indices and no weight cache"
            )
        if not 0 <= release_p_min <= release_p_max <= 1:
            raise ValueError("require 0 <= release_p_min <= release_p_max <= 1")
        if not 0 <= release_impact_decay < 1:
            raise ValueError("release_impact_decay must be in [0, 1)")
        if release_impact_metric not in RELEASE_IMPACT_METRICS:
            raise ValueError(
                "release_impact_metric must be one of "
                f"{sorted(RELEASE_IMPACT_METRICS)}, got {release_impact_metric!r}"
            )
        if release_max_conductance is not None:
            release_max_conductance = float(release_max_conductance)
            if (
                not math.isfinite(release_max_conductance)
                or release_max_conductance <= 0
            ):
                raise ValueError("release_max_conductance must be finite and positive")
            if release_impact_metric == "saliency":
                raise ValueError(
                    "saliency impact is undefined for capped contacts; "
                    "do not combine it with release_max_conductance"
                )
        self.release_p_min, self.release_p_max = release_p_min, release_p_max
        self.release_weight_scale = (
            1 / self.K if release_weight_scale is None else float(release_weight_scale)
        )
        if (
            not math.isfinite(self.release_weight_scale)
            or self.release_weight_scale <= 0
        ):
            raise ValueError("release_weight_scale must be finite and positive")
        self.release_impact_decay = release_impact_decay
        self.release_impact_metric = release_impact_metric
        self.release_max_conductance = release_max_conductance
        self.release_track_source_activity = bool(release_track_source_activity)
        self.signal_rounding = str(signal_rounding).lower()
        self._round_dtype = signal_rounding_dtype(self.signal_rounding)
        # Log-normal law of newly initialized contacts; set by initializers that
        # define one (fan_in_lognormal) and used by "initial" regrowth weights.
        self.initial_log_mean = None
        self.initial_log_sd = None
        self.register_buffer("impact", torch.zeros_like(self.pre_w))
        self.register_buffer(
            "contact_age", torch.zeros_like(self.pre_w, dtype=torch.long)
        )
        # None preserves legacy checkpoint keys when activity tracking is off.
        self.register_buffer("source_activity", None)
        if self.release_track_source_activity:
            self.source_activity = torch.zeros(
                self.in_features, dtype=self.pre_w.dtype, device=self.pre_w.device
            )

    def sparse_weight(self) -> torch.Tensor:
        weights = super().sparse_weight()
        if self.release_max_conductance is None:
            return weights
        return weights.clamp(max=self.release_max_conductance)

    def inverse_weight(self, conductance: torch.Tensor) -> torch.Tensor:
        """Map positive conductances to the stored raw parameterization."""
        if self.weight_transform == "exp":
            return conductance.log()
        return inverse_weight_transform(conductance, self.weight_transform)

    def release_probability(self, weight):
        return self.release_p_min + (
            self.release_p_max - self.release_p_min
        ) * weight / (weight + 0.25 * self.release_weight_scale)

    def forward_mutates_state(self) -> bool:
        """A training forward tracks contact impact (and source activity)."""
        return True

    @torch.no_grad()
    def _chunk_impact_sums(self, selected, drive) -> list[torch.Tensor]:
        """Batch sums of one output chunk's forward impact metric."""
        if RELEASE_IMPACT_METRICS[self.release_impact_metric] != "forward":
            return []
        sums = [drive.detach().abs().sum(0, dtype=torch.float64)]
        if self.release_impact_metric == "active_drive":
            sums.append(selected.detach().ne(0).sum(0, dtype=torch.float64))
        return sums

    @torch.no_grad()
    def _accumulate_rank_reduced_statistics(self, flat, chunk_sums) -> None:
        """Update source activity and forward impact from the global batch.

        ``chunk_sums`` holds ``(start, stop, sums)`` of each output chunk (see
        :meth:`_chunk_impact_sums`). The batch sums and counts of every rank
        are reduced in one collective, so all ranks keep the same statistics.
        """
        rate = 1 - self.release_impact_decay
        sums = [torch.tensor(float(flat.shape[0]), device=flat.device)]
        if self.source_activity is not None:
            sums.append(flat.detach().abs().sum(0, dtype=torch.float64))
        for _start, _stop, chunk in chunk_sums:
            sums.extend(chunk)
        reduced = iter(all_reduce_sums(*sums))
        count = next(reduced)
        if self.source_activity is not None:
            activity = next(reduced) / count
            self.source_activity.lerp_(activity.to(self.source_activity.dtype), rate)
        for start, stop, chunk in chunk_sums:
            if not chunk:
                continue
            current = self.impact[start:stop]
            total = next(reduced)
            if self.release_impact_metric == "drive":
                current.lerp_((total / count).to(current.dtype), rate)
                continue
            active = next(reduced)
            value = (total / active.clamp_min(1)).to(current.dtype)
            current.copy_(torch.where(active.gt(0), current.lerp(value, rate), current))

    def _accumulate_forward_impact(self, start, stop, selected, drive):
        rate = 1 - self.release_impact_decay
        if self.release_impact_metric == "drive":
            self.impact[start:stop].lerp_(drive.detach().abs().mean(0), rate)
        elif self.release_impact_metric == "active_drive":
            active = selected.detach().ne(0).sum(0)
            value = drive.detach().abs().sum(0) / active.clamp_min(1)
            current = self.impact[start:stop]
            current.copy_(torch.where(active.gt(0), current.lerp(value, rate), current))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if prefix + "contact_age" in state_dict and file_predates(local_metadata, 2):
            record_legacy_state(
                self,
                prefix + "contact_age",
                "contact ages in files written before 2026-10-04 count training "
                "forwards, not optimizer steps; they agree only when each "
                "optimizer step runs one training forward",
            )

    @torch.no_grad()
    def accumulate_step_impact(self) -> None:
        """Advance contact ages and step-stage impact metrics after an update.

        Called once per optimizer step (by the rewiring controller), so
        ``contact_age`` counts optimizer steps, independent of how many
        training forwards a step takes (a recurrent unroll of ``T`` steps used
        to age contacts ``T`` times). Forward-stage metrics are accumulated
        during training forwards. ``saliency`` uses the parameter gradient left
        by the most recent backward pass and is skipped when no gradient is
        present.
        """
        self.contact_age.add_(1)
        metric = self.release_impact_metric
        if RELEASE_IMPACT_METRICS[metric] != "step":
            return
        rate = 1 - self.release_impact_decay
        weights = self.sparse_weight().detach()
        if metric == "weight":
            self.impact.lerp_(weights, rate)
            return
        grad = self.pre_w.grad
        if grad is None:
            return
        if self.weight_transform == "exp":
            value = grad.abs()
        else:
            value = (weights * grad / torch.sigmoid(self.pre_w)).abs()
        self.impact.lerp_(value, rate)

    def forward(self, x):
        self._require_topology()
        shape = x.shape[:-1]
        rounding = self._round_dtype
        flat = round_signal(x.reshape(-1, self.in_features), rounding)
        weights = round_signal(self.sparse_weight(), rounding)
        # Target of conductance-dynamic forward-hook scaling (inherited
        # register_forward_param_gradient_scale); retained only in that mode.
        object.__setattr__(
            self,
            "_last_forward_param_tensor",
            (
                weights
                if self._use_forward_dynamic_grad_scaling and weights.requires_grad
                else None
            ),
        )
        probabilities = self.release_probability(weights)
        track = self.training and not sparse_state_frozen()
        # Under data-parallel training the statistics are reduced over the
        # ranks once all chunks are known.
        rank_reduced = track and reduces_statistics_across_ranks(self)
        rank_chunks = []
        if track and self.source_activity is not None and not rank_reduced:
            with torch.no_grad():
                self.source_activity.lerp_(
                    flat.detach().abs().mean(0), 1 - self.release_impact_decay
                )
        result = []
        for start in range(0, self.out_features, self.output_chunk_size):
            stop = min(start + self.output_chunk_size, self.out_features)
            indices = self.connection_indices[start:stop].long()
            selected = flat[:, indices]
            weight, probability = weights[start:stop], probabilities[start:stop]
            if self.training:
                released = (
                    torch.rand(selected.shape, device=x.device) < probability.detach()
                )
                drive = selected * weight * released
                if rank_reduced:
                    rank_chunks.append(
                        (start, stop, self._chunk_impact_sums(selected, drive))
                    )
                elif track:
                    with torch.no_grad():
                        self._accumulate_forward_impact(start, stop, selected, drive)
            else:
                drive = selected * weight * round_signal(probability, rounding)
            result.append(round_signal(drive.sum(-1), rounding))
        if rank_reduced:
            self._accumulate_rank_reduced_statistics(flat, rank_chunks)
        return torch.cat(result, -1).reshape(*shape, self.out_features)

    def freeze_for_inference(self):
        raise ValueError("indexed_release does not support inference weight folding")
