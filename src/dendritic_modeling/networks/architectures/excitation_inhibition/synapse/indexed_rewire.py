"""Adaptive rewiring indexed sparse synapses."""

from __future__ import annotations

from math import ceil
from typing import ClassVar

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
from dendritic_modeling.networks.state_versions import accept_legacy_missing_key
from dendritic_modeling.networks.utils.sparse_maintenance import maintenance_applies
from dendritic_modeling.utils.reproducibility import isolated_random_seed

REWIRE_INIT_POLICIES = {"fresh", "inherit_pruned"}
# ``_connection_generator`` seeds CPU generators modulo this value.
_GENERATOR_SEED_MODULUS = 2**63 - 1


class IndexedRewireLinear(IndexedSparseLinear):
    """
    Memory-efficient sparse layer with local adaptive-quantile rewiring.

    The layer stores exactly ``K`` synapses per output row, like
    :class:`IndexedSparseLinear`, and periodically replaces the weakest
    fraction of those synapses with newly sampled allowed input indices. The
    rewiring is always local to the same output row, so dendritic branch
    synapse budgets stay fixed while weak synapses are given a biologically
    inspired turnover path.

    The rewiring random stream is seeded by ``seed`` (or, when ``seed`` is
    ``None``, by a draw from the global RNG at construction); the resolved
    seed and any provided ``connection_mask`` are checkpointed so a resumed
    run continues the same rewiring stream within the original mask.
    """

    # Rewiring prunes the slots that ``topk`` returns (ties and the pairing of
    # new inputs or fresh weights with pruned slots follow slot positions).
    CONTACT_SLOT_STATE: ClassVar[tuple[str, ...]] = ("pre_w",)
    SLOT_ORDER_MATTERS: ClassVar[bool] = True

    _persist_allowed_connection_mask = True
    _STATE_KEYS_ADDED: ClassVar[dict[int, tuple[str, ...]]] = {2: ("rewire_seed",)}

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int,
        *,
        rewire_frequency: int = 100,
        rewire_quantile: float = 0.05,
        rewire_until_step: int | None = None,
        rewire_init_policy: str = "inherit_pruned",
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
        rewire_init_policy = str(rewire_init_policy).strip().lower()
        if rewire_init_policy not in REWIRE_INIT_POLICIES:
            raise ValueError(
                "rewire_init_policy must be one of "
                f"{sorted(REWIRE_INIT_POLICIES)}, got {rewire_init_policy!r}"
            )
        if kwargs.get("persistent_indices", True) is False:
            raise ValueError(
                "IndexedRewireLinear requires persistent_indices=True because "
                "its topology changes during training"
            )

        # Rewiring generators are seeded modulo 2**63 - 1, so reducing the seed
        # here leaves every rewiring stream unchanged and makes it storable.
        self._rewire_seed = (
            int(seed)
            if seed is not None
            else int(torch.empty((), dtype=torch.int64).random_().item())
        ) % _GENERATOR_SEED_MODULUS
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            K=K,
            seed=seed,
            **kwargs,
        )

        self.rewire_frequency = int(rewire_frequency)
        self.rewire_quantile = float(rewire_quantile)
        self.rewire_init_policy = rewire_init_policy
        self.rewire_until_step = (
            None if rewire_until_step is None else int(rewire_until_step)
        )
        self.freeze_connectivity = bool(freeze_connectivity)
        self.register_buffer(
            "rewire_step", torch.zeros((), dtype=torch.long), persistent=True
        )
        # Checkpointed copy of the resolved rewiring seed (an unseeded layer
        # draws it from the global RNG, so a fresh process cannot recreate it).
        self.register_buffer(
            "rewire_seed",
            torch.tensor(self._rewire_seed, dtype=torch.long),
            persistent=True,
        )
        # Keep scheduling decisions on the host.  Reading a CUDA scalar with
        # ``rewire_step.item()`` in every optimizer step otherwise introduces
        # one device synchronization per indexed-rewire module.  The tensor
        # remains the checkpoint source of truth for backward compatibility.
        self._rewire_step_host = 0
        self._rewire_step_tensor_id = id(self.rewire_step)
        self._rewire_step_tensor_version = self.rewire_step._version
        self.register_buffer(
            "_last_rewired_mask",
            torch.zeros_like(self.pre_w, dtype=torch.bool),
            persistent=False,
        )
        # This host flag is the source of truth for whether the transient mask
        # contains any entries.  Asking CUDA ``bool(mask.any())`` after every
        # optimizer step serializes the host once per rewired projection even
        # though most steps are not rewiring boundaries.
        self._last_rewired_any_host = False

    def _sync_rewire_step_host_from_buffer(self) -> None:
        """Synchronize the non-persistent scheduler counter from model state."""
        self._rewire_step_host = int(self.rewire_step.item())
        self._rewire_step_tensor_id = id(self.rewire_step)
        self._rewire_step_tensor_version = self.rewire_step._version

    def _rewire_step_buffer_changed(self) -> bool:
        return (
            id(self.rewire_step) != self._rewire_step_tensor_id
            or self.rewire_step._version != self._rewire_step_tensor_version
        )

    def _rewire_step_value(self) -> int:
        """Advance and return the checkpointed rewiring step without a sync."""
        # ``load_state_dict`` is handled eagerly below.  The identity/version
        # check also covers direct state-tensor copies used by legacy
        # replacement-checkpoint loaders, while keeping the steady-state path
        # free of CUDA scalar reads.
        if self._rewire_step_buffer_changed():
            self._sync_rewire_step_host_from_buffer()
        self._rewire_step_host += 1
        self.rewire_step.add_(1)
        self._rewire_step_tensor_id = id(self.rewire_step)
        self._rewire_step_tensor_version = self.rewire_step._version
        return self._rewire_step_host

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Restore the host scheduler counter and seed after loading state."""
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        self._sync_rewire_step_host_from_buffer()
        if prefix + "rewire_seed" in missing_keys:
            # Checkpoints written before the seed was saved keep this
            # instance's seed: the original rewiring stream is not recoverable.
            accept_legacy_missing_key(
                self,
                "rewire_seed",
                prefix=prefix,
                local_metadata=local_metadata,
                missing_keys=missing_keys,
                version=2,
                note="the file predates the saved rewiring seed; later "
                "rewiring draws from the constructed seed",
            )
        else:
            self._rewire_seed = int(self.rewire_seed.item())

    def _apply(self, fn, recurse=True):
        """Preserve a synchronized host counter across device transfers."""
        # Legacy checkpoint helpers can copy directly into the state tensor and
        # then move the module.  Consume such a mutation before ``_apply``
        # replaces the tensor object; ordinary moves need no scalar read.
        if self._rewire_step_buffer_changed():
            self._sync_rewire_step_host_from_buffer()
        result = super()._apply(fn, recurse=recurse)
        self._rewire_step_tensor_id = id(self.rewire_step)
        self._rewire_step_tensor_version = self.rewire_step._version
        return result

    def _allowed_indices_for_row_cpu(self, row_idx: int) -> torch.Tensor:
        if self._allowed_connection_mask.numel() == 0:
            allowed = torch.arange(self.in_features, dtype=torch.long)
        else:
            allowed = torch.nonzero(
                self._allowed_connection_mask[row_idx].detach().cpu(),
                as_tuple=False,
            ).flatten()
        if self._forbidden_input_index_per_output.numel() > 0:
            forbidden = int(self._forbidden_input_index_per_output[row_idx].item())
            if forbidden >= 0:
                allowed = allowed[allowed != forbidden]
        return allowed

    def _new_pre_weights(self, shape: tuple[int, int], seed: int) -> torch.Tensor:
        temp = torch.empty(shape, device=self.pre_w.device, dtype=self.pre_w.dtype)
        if self.init_method not in TOPK_INIT_METHODS:
            raise ValueError(
                f"Invalid initialization method: {self.init_method}. "
                f"Choose from {list(TOPK_INIT_METHODS.keys())}"
            )
        # Fresh weights belong to the topology stream, independently of local
        # data/dropout draws. This also keeps synchronized DDP replicas equal.
        with isolated_random_seed(seed):
            TOPK_INIT_METHODS[self.init_method](temp)
        temp.mul_(self.init_gain)
        return temp

    def apply_rewiring(self) -> None:
        """Replace the weakest active sparse slots with new random inputs.

        Frozen layers (``pre_w.requires_grad`` is ``False``) neither advance
        their schedule nor rewire.
        """
        if not maintenance_applies(self.pre_w):
            return
        with torch.no_grad():
            self._recurrent_cached_weight = None
            self._last_rewired_mask.zero_()
            self._last_rewired_any_host = False

            step = self._rewire_step_value()
            if self.freeze_connectivity or self.rewire_quantile <= 0.0:
                return
            if self.rewire_until_step is not None and step > self.rewire_until_step:
                return
            if step % self.rewire_frequency != 0:
                return

            # DDP broadcasts buffers without invoking load_state_dict. Read
            # the authoritative seed at a rewiring event, not on every forward.
            self._rewire_seed = int(self.rewire_seed.item())
            seed_base = self._rewire_seed + 104729 * step
            n_rewire = min(self.K, max(1, ceil(self.K * self.rewire_quantile)))
            scores = self.sparse_weight().detach().abs()
            prune_slots = torch.topk(
                scores, n_rewire, dim=1, largest=False, sorted=False
            ).indices
            fresh_weights = (
                self._new_pre_weights((self.out_features, n_rewire), seed_base)
                if self.rewire_init_policy == "fresh"
                else None
            )

            # Plan the row-local random choices on CPU exactly as before, but
            # commit all disjoint row/slot writes in three batched accelerator
            # operations.  The former row loop issued up to three tiny CUDA
            # operations per output row at each rewiring boundary, which made
            # large readouts host-launch bound while doing almost no SM work.
            conn_cpu = self.connection_indices.detach().cpu()
            prune_slots_cpu = prune_slots.detach().cpu()
            allowed_mask_cpu = (
                None
                if self._allowed_connection_mask.numel() == 0
                else self._allowed_connection_mask.detach().cpu()
            )
            forbidden_cpu = (
                None
                if self._forbidden_input_index_per_output.numel() == 0
                else self._forbidden_input_index_per_output.detach().cpu()
            )
            changed_rows: list[torch.Tensor] = []
            changed_slots: list[torch.Tensor] = []
            changed_inputs: list[torch.Tensor] = []
            changed_fresh_columns: list[torch.Tensor] = []
            for row_idx in range(self.out_features):
                available_mask = torch.zeros(self.in_features, dtype=torch.bool)
                if allowed_mask_cpu is None:
                    allowed = torch.arange(self.in_features, dtype=torch.long)
                else:
                    allowed = torch.nonzero(
                        allowed_mask_cpu[row_idx], as_tuple=False
                    ).flatten()
                if forbidden_cpu is not None:
                    forbidden = int(forbidden_cpu[row_idx].item())
                    if forbidden >= 0:
                        allowed = allowed[allowed != forbidden]
                available_mask[allowed] = True
                available_mask[conn_cpu[row_idx]] = False
                available = torch.nonzero(available_mask, as_tuple=False).flatten()
                if available.numel() == 0:
                    continue

                n_select = min(n_rewire, int(available.numel()))
                generator = _connection_generator(seed_base + row_idx)
                selected = available[
                    torch.randperm(available.numel(), generator=generator)[:n_select]
                ]
                changed_rows.append(torch.full((n_select,), row_idx, dtype=torch.long))
                changed_slots.append(prune_slots_cpu[row_idx, :n_select].long())
                changed_inputs.append(selected)
                if fresh_weights is not None:
                    changed_fresh_columns.append(torch.arange(n_select))

            if changed_rows:
                row_indices = torch.cat(changed_rows).to(
                    device=self.connection_indices.device
                )
                slot_indices = torch.cat(changed_slots).to(
                    device=self.connection_indices.device
                )
                selected_inputs = torch.cat(changed_inputs).to(
                    device=self.connection_indices.device,
                    dtype=self.connection_indices.dtype,
                )
                # In-place write: bumps connection_indices._version, which
                # also invalidates the deterministic-backward CSR setup cache
                # in triton_indexed_gather_transposed.
                self.connection_indices[row_indices, slot_indices] = selected_inputs
                if fresh_weights is not None:
                    fresh_columns = torch.cat(changed_fresh_columns).to(
                        device=self.pre_w.device
                    )
                    self.pre_w.data[row_indices, slot_indices] = fresh_weights[
                        row_indices, fresh_columns
                    ]
                self._last_rewired_mask[row_indices, slot_indices] = True
                self._last_rewired_any_host = True

            if self.pre_w.grad is not None and self._last_rewired_any_host:
                self.pre_w.grad.data[self._last_rewired_mask] = 0

    def consume_rewired_mask(self) -> torch.Tensor | None:
        """Return and clear the mask of slots rewired by the last call."""
        if not self._last_rewired_any_host:
            return None
        mask = self._last_rewired_mask.detach().clone()
        self._last_rewired_mask.zero_()
        self._last_rewired_any_host = False
        return mask


__all__ = ["REWIRE_INIT_POLICIES", "IndexedRewireLinear"]
