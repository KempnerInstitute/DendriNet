"""
Stateful DendriNet with optional recurrent compartments and per-level traces.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    sample_configured_mask,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState
from dendritic_modeling.networks.architectures.recurrent.spiking_soma import (
    LIFSoma,
    SurrogateSpike,
)
from dendritic_modeling.networks.architectures.recurrent.synapse_types import (
    SynapseTypeConfig,
    build_synapse_type_set,
)


def _compute_level_sizes(n_soma: int, branch_factors: list[int]) -> list[int]:
    """Compute output dimensions per level (distal to soma order).

    For n_soma=10, branch_factors=[3,2]:
      soma=10, level_1=10*2=20, level_0=20*3=60
      Returns [60, 20, 10] (distal first, soma last).
    """
    sizes = [n_soma]
    for bf in branch_factors:
        sizes.append(sizes[-1] * bf)
    sizes.reverse()
    return sizes


def _compute_input_branch_factors(branch_factors: list[int]) -> list[Optional[int]]:
    """Compute branch aggregation factors per level.

    First (most distal) level has no branch input (None).
    Each subsequent level aggregates children from previous level.
    """
    rev_bf = list(reversed(branch_factors))
    return [None, *rev_bf]


def _compute_output_owner_index_per_level(
    level_dims: list[int], n_soma: int
) -> list[torch.Tensor]:
    """Map each branch output at a level to its parent soma-neuron index."""
    owner_indices = []
    for level_dim in level_dims:
        if level_dim % n_soma != 0:
            raise ValueError(
                f"Level dimension {level_dim} must be divisible by soma count {n_soma}"
            )
        repeats_per_soma = level_dim // n_soma
        owner_indices.append(
            torch.arange(level_dim, dtype=torch.long) // repeats_per_soma
        )
    return owner_indices


def _resolve_synapses_by_level(
    *,
    per_level: list[int],
    fallback: int,
    n_levels: int,
    field_name: str,
) -> list[int]:
    if not per_level:
        return [int(fallback)] * n_levels
    if len(per_level) != n_levels:
        raise ValueError(
            f"{field_name} length must match number of dendritic levels "
            f"({n_levels}), got {len(per_level)}"
        )
    return [int(v) for v in per_level]


class StatefulDendriNet(nn.Module):
    """Dendritic population module for feedforward and recurrent operation.

    Constructs a hierarchy of DendriticBranchLayers with all 4 synaptic
    compartments (FF E, FF I, REC E, REC I). In feedforward mode, recurrent
    compartments are not constructed (None dims). In recurrent mode, per-level
    temporal traces are maintained with geometrically spaced time constants.

    When ``store_routing`` is True, each call to :meth:`step` populates
    ``_last_routing_info`` with per-level contribution and FF/REC decomposition.
    """

    def __init__(
        self,
        pop_config: PopulationConfig,
        excitatory_input_dim: int,
        inhibitory_input_dim: Optional[int] = None,
        recurrent_excitatory_input_dim: Optional[int] = None,
        recurrent_inhibitory_input_dim: Optional[int] = None,
        recurrent_excitatory_is_self_population: bool = False,
        recurrent_inhibitory_is_self_population: bool = False,
        connection_masks_by_pathway=None,
        dt: float = 1.0,
    ):
        super().__init__()
        self.pop_config = pop_config
        self.dt = dt
        self.synapse_types = build_synapse_type_set(pop_config.synapse_types)
        self.synapse_types_enabled = bool(self.synapse_types.enabled)
        if self.synapse_types_enabled and not bool(pop_config.use_shunting):
            self._validate_additive_synapse_reversals()
        self.dynamics_mode = str(pop_config.dynamics_mode).lower()
        self.spiking_soma = None
        if self.dynamics_mode == "spike":
            self.spiking_soma = LIFSoma(
                threshold=pop_config.spike_threshold,
                reset=pop_config.spike_reset,
                tau=pop_config.spike_tau,
                dt=dt,
                refractory_steps=pop_config.spike_refractory_steps,
                surrogate_beta=pop_config.spike_surrogate_beta,
                output_mode=pop_config.spike_readout,
                readout_tau=pop_config.spike_readout_tau,
            )

        # Routing analysis flags (set externally or via EINetwork)
        self.store_routing: bool = False
        self._last_routing_info: dict = {}
        self._active_level_silencing: Optional[torch.Tensor] = None
        self._n_soma = pop_config.n_neurons
        self._level_dims = _compute_level_sizes(
            n_soma=pop_config.n_neurons,
            branch_factors=pop_config.branch_factors,
        )
        self._output_owner_index_per_level = _compute_output_owner_index_per_level(
            self._level_dims, self._n_soma
        )
        self.n_levels = len(self._level_dims)
        self.dendritic_spikes_enabled = bool(pop_config.dendritic_spikes_enabled)
        self.dendritic_spike_mode = str(pop_config.dendritic_spike_mode).lower()
        self.dendritic_spike_threshold = float(pop_config.dendritic_spike_threshold)
        self.dendritic_spike_plateau_amplitude = float(
            pop_config.dendritic_spike_plateau_amplitude
        )
        self.dendritic_spike_plateau_tau = float(pop_config.dendritic_spike_plateau_tau)
        self.dendritic_spike_refractory_steps = int(
            pop_config.dendritic_spike_refractory_steps
        )
        self.dendritic_spike_surrogate_beta = float(
            pop_config.dendritic_spike_surrogate_beta
        )
        self.dendritic_spike_propagation = str(
            pop_config.dendritic_spike_propagation
        ).lower()
        self.dendritic_spike_level_indices = self._resolve_dendritic_spike_levels(
            pop_config.dendritic_spike_levels
        )
        self.soma_feedback_enabled = bool(pop_config.soma_feedback_enabled)
        self.soma_feedback_mode = str(pop_config.soma_feedback_mode).lower()
        self.soma_feedback_source = str(pop_config.soma_feedback_source).lower()
        self.soma_feedback_per_level = bool(pop_config.soma_feedback_per_level)
        self.soma_feedback_reversal = float(pop_config.soma_feedback_reversal)
        self.soma_feedback_level_indices = self._resolve_soma_feedback_levels(
            pop_config.soma_feedback_levels
        )
        self._soma_feedback_strength_index = {
            level_idx: strength_idx
            for strength_idx, level_idx in enumerate(
                sorted(self.soma_feedback_level_indices)
            )
        }
        self.soma_feedback_projections = nn.ModuleList()
        if self.soma_feedback_enabled:
            # Keep the opt-in feedback modules from shifting the initialization
            # stream for the existing dendritic branches and downstream layers.
            with torch.random.fork_rng(devices=[]):
                for level_idx, level_dim in enumerate(self._level_dims):
                    if level_idx in self.soma_feedback_level_indices:
                        projection = nn.Linear(self._n_soma, level_dim, bias=False)
                        init_std = float(pop_config.soma_feedback_init_std)
                        if init_std == 0.0:
                            nn.init.zeros_(projection.weight)
                        else:
                            nn.init.normal_(projection.weight, mean=0.0, std=init_std)
                    else:
                        # Preserve projection indexing by level while avoiding
                        # trainable parameters for levels that are configured
                        # as feedback-disabled.
                        projection = nn.Identity()
                    self.soma_feedback_projections.append(projection)

        strength_shape = (
            len(self.soma_feedback_level_indices) if self.soma_feedback_per_level else 1
        )
        strength = torch.full(
            (strength_shape,),
            float(pop_config.soma_feedback_strength),
            dtype=torch.float32,
        )
        if self.soma_feedback_enabled and bool(
            pop_config.soma_feedback_learnable_strength
        ):
            self.soma_feedback_strength = nn.Parameter(strength)
        else:
            self.register_buffer("soma_feedback_strength", strength)

        # Compute per-level time constants
        if pop_config.level_taus:
            if len(pop_config.level_taus) != self.n_levels:
                raise ValueError(
                    "level_taus length must match number of dendritic levels "
                    f"({self.n_levels}), got {len(pop_config.level_taus)}"
                )
            level_taus = list(pop_config.level_taus)
        else:
            # Geometrically spaced taus: distal levels get longer time constants.
            # Level 0 (most distal) = tau_base * ratio^(n_levels-1)
            # Level n_levels-1 (soma) = tau_base
            ratio = pop_config.tau_ratio
            level_taus = [
                pop_config.tau_base * (ratio ** (self.n_levels - 1 - i))
                for i in range(self.n_levels)
            ]

        self.learnable_tau = bool(pop_config.learnable_tau)
        if self.learnable_tau:
            # Store log(tau) as learnable parameter; tau = exp(log_tau) > 0.
            # decay = exp(-dt / tau) is computed dynamically in step().
            self.log_tau = nn.Parameter(
                torch.tensor([math.log(tau) for tau in level_taus], dtype=torch.float32)
            )
        else:
            self.log_tau = None
            self.register_buffer(
                "decays",
                torch.tensor(
                    [math.exp(-dt / tau) for tau in level_taus], dtype=torch.float32
                ),
            )

        # Build branch layers (distal to soma)
        self.branch_layers = nn.ModuleList()
        input_branch_factors = _compute_input_branch_factors(pop_config.branch_factors)
        ff_exc_by_level = _resolve_synapses_by_level(
            per_level=pop_config.ff_excitatory_synapses_by_level,
            fallback=pop_config.ff_excitatory_synapses,
            n_levels=self.n_levels,
            field_name="ff_excitatory_synapses_by_level",
        )
        ff_inh_by_level = _resolve_synapses_by_level(
            per_level=pop_config.ff_inhibitory_synapses_by_level,
            fallback=pop_config.ff_inhibitory_synapses,
            n_levels=self.n_levels,
            field_name="ff_inhibitory_synapses_by_level",
        )
        rec_exc_by_level = _resolve_synapses_by_level(
            per_level=pop_config.rec_excitatory_synapses_by_level,
            fallback=pop_config.rec_excitatory_synapses,
            n_levels=self.n_levels,
            field_name="rec_excitatory_synapses_by_level",
        )
        rec_inh_by_level = _resolve_synapses_by_level(
            per_level=pop_config.rec_inhibitory_synapses_by_level,
            fallback=pop_config.rec_inhibitory_synapses,
            n_levels=self.n_levels,
            field_name="rec_inhibitory_synapses_by_level",
        )
        direct_connection_masks = connection_masks_by_pathway or {}

        # Common DendriticBranchLayer kwargs from PopulationConfig
        common_kwargs = {
            "topk_init_method": pop_config.topk_init_method,
            "topk_noise_level": pop_config.topk_noise_level,
            "topk_type": pop_config.topk_type,
            "topk_weight_norm_order": pop_config.topk_weight_norm_order,
            "topk_gamma": pop_config.topk_gamma,
            "topk_temperature": pop_config.topk_temperature,
            "topk_ultrafast": pop_config.topk_ultrafast,
            "topk_strategy": pop_config.topk_strategy,
            "dense_to_sparse_initial_density": pop_config.dense_to_sparse_initial_density,
            "dense_to_sparse_initial_k": pop_config.dense_to_sparse_initial_k,
            "dense_to_sparse_start_step": pop_config.dense_to_sparse_start_step,
            "dense_to_sparse_end_step": pop_config.dense_to_sparse_end_step,
            "dense_to_sparse_update_interval": pop_config.dense_to_sparse_update_interval,
            "dense_to_sparse_schedule": pop_config.dense_to_sparse_schedule,
            "dense_to_sparse_freeze_on_end": pop_config.dense_to_sparse_freeze_on_end,
            "dense_to_sparse_advance_on_forward": pop_config.dense_to_sparse_advance_on_forward,
            "dense_to_sparse_prune_metric": pop_config.dense_to_sparse_prune_metric,
            "use_shunting": pop_config.use_shunting,
            "use_additive_normalization": pop_config.use_additive_normalization,
            "reactivate": pop_config.reactivate,
            "reactivation_type": pop_config.reactivation_type,
            "reactivation_init_m": pop_config.reactivation_init_m,
            "reactivation_init_b": pop_config.reactivation_init_b,
            "reactivation_init_policy": pop_config.reactivation_init_policy,
            "reactivation_occupancy_quantile_low": pop_config.reactivation_occupancy_quantile_low,
            "reactivation_occupancy_quantile_high": pop_config.reactivation_occupancy_quantile_high,
            "reactivation_occupancy_target_low": pop_config.reactivation_occupancy_target_low,
            "reactivation_occupancy_target_high": pop_config.reactivation_occupancy_target_high,
            "reactivation_calibration_min_quantile_width": pop_config.reactivation_calibration_min_quantile_width,
            "reactivation_calibration_max_m": pop_config.reactivation_calibration_max_m,
            "reactivation_calibration_revert_on_invalid": pop_config.reactivation_calibration_revert_on_invalid,
            "reactivation_sigma_aware_k": pop_config.reactivation_sigma_aware_k,
            "reactivation_strategy": pop_config.reactivation_strategy,
            "blocklinear_strategy": pop_config.blocklinear_strategy,
            "weight_transform": pop_config.weight_transform,
            "epsilon": pop_config.epsilon,
            "dbl_init_method": pop_config.dbl_init_method,
            "efficient_blocklinear": pop_config.efficient_blocklinear,
            "print_hooks": pop_config.print_hooks,
            "excitatory_target_density": pop_config.excitatory_target_density,
            "inhibitory_target_density": pop_config.inhibitory_target_density,
            "use_noise": pop_config.use_noise,
            "sigma": pop_config.sigma,
            "rewiring_mode": pop_config.rewiring_mode,
            "rewire_frequency": pop_config.rewire_frequency,
            "synapses_per_branch": pop_config.synapses_per_branch,
            "freeze_excitatory_connectivity": pop_config.freeze_excitatory_connectivity,
            "freeze_inhibitory_connectivity": pop_config.freeze_inhibitory_connectivity,
            "init_method": pop_config.init_method,
            "weight_threshold": pop_config.weight_threshold,
            "adaptive_initialization": pop_config.adaptive_initialization,
            "adaptive_initialization_policy": pop_config.adaptive_initialization_policy,
            "adaptive_target_conductance": pop_config.adaptive_target_conductance,
            "branch_factors": pop_config.branch_factors,
        }

        # Helper to compute conditional dims
        def _exc_dim(synapses):
            return excitatory_input_dim if synapses > 0 else None

        def _exc_syn(synapses):
            return synapses if synapses > 0 else None

        def _inh_dim(dim, synapses):
            return dim if dim is not None and synapses > 0 else None

        def _inh_syn(dim, synapses):
            return synapses if dim is not None and synapses > 0 else None

        is_soma_level = self.n_levels - 1  # last index

        for level_idx, out_dim in enumerate(self._level_dims):
            recurrent_self_forbidden = None
            rec_inhibitory_self_forbidden = None
            if not pop_config.allow_self_recurrence:
                owner_idx = self._output_owner_index_per_level[level_idx]
                if recurrent_excitatory_is_self_population:
                    recurrent_self_forbidden = owner_idx
                if recurrent_inhibitory_is_self_population:
                    rec_inhibitory_self_forbidden = owner_idx
            # Soma level: optionally skip direct synaptic input
            if level_idx == is_soma_level and not pop_config.somatic_synapses:
                exc_in = None
                exc_syn = None
                inh_in = None
                inh_syn = None
                rec_in = None
                rec_syn = None
                rec_inh_in = None
                rec_inh_syn = None
            else:
                ff_exc_syn = ff_exc_by_level[level_idx]
                ff_inh_syn_count = ff_inh_by_level[level_idx]
                rec_exc_syn_count = rec_exc_by_level[level_idx]
                rec_inh_syn_count = rec_inh_by_level[level_idx]

                exc_in = _exc_dim(ff_exc_syn)
                exc_syn = _exc_syn(ff_exc_syn)
                inh_in = _inh_dim(inhibitory_input_dim, ff_inh_syn_count)
                inh_syn = _inh_syn(inhibitory_input_dim, ff_inh_syn_count)
                rec_in = _inh_dim(recurrent_excitatory_input_dim, rec_exc_syn_count)
                rec_syn = _inh_syn(recurrent_excitatory_input_dim, rec_exc_syn_count)
                rec_inh_in = _inh_dim(recurrent_inhibitory_input_dim, rec_inh_syn_count)
                rec_inh_syn = _inh_syn(
                    recurrent_inhibitory_input_dim, rec_inh_syn_count
                )

            def _structured_mask(pathway, in_dim, out_dim=out_dim, level_idx=level_idx):
                if in_dim is None:
                    return None
                direct_masks = direct_connection_masks.get(pathway)
                if direct_masks is not None:
                    if level_idx >= len(direct_masks):
                        raise ValueError(
                            f"connection_masks_by_pathway[{pathway!r}] must include "
                            f"{self.n_levels} level masks"
                        )
                    return direct_masks[level_idx]
                return sample_configured_mask(
                    pop_config.structured_connectivity,
                    pathway=pathway,
                    out_features=out_dim,
                    in_features=in_dim,
                    layer_idx=pop_config.structured_layer_idx,
                    level_idx=level_idx,
                )

            branch_layer = DendriticBranchLayer(
                output_dim=out_dim,
                excitatory_input_dim=exc_in,
                excitatory_synapses_per_branch=exc_syn,
                excitatory_connection_mask=_structured_mask(
                    pop_config.ff_excitatory_pathway,
                    exc_in,
                ),
                inhibitory_input_dim=inh_in,
                inhibitory_synapses_per_branch=inh_syn,
                inhibitory_connection_mask=_structured_mask(
                    pop_config.ff_inhibitory_pathway,
                    inh_in,
                ),
                input_branch_factor=input_branch_factors[level_idx],
                recurrent_input_dim=rec_in,
                recurrent_synapses_per_branch=rec_syn,
                recurrent_forbidden_input_index_per_output=recurrent_self_forbidden,
                recurrent_connection_mask=_structured_mask(
                    pop_config.rec_excitatory_pathway,
                    rec_in,
                ),
                rec_inhibitory_input_dim=rec_inh_in,
                rec_inhibitory_synapses_per_branch=rec_inh_syn,
                rec_inhibitory_forbidden_input_index_per_output=rec_inhibitory_self_forbidden,
                rec_inhibitory_connection_mask=_structured_mask(
                    pop_config.rec_inhibitory_pathway,
                    rec_inh_in,
                ),
                layer_idx=self.n_levels - level_idx,
                **common_kwargs,
            )
            if pop_config.topk_strategy == "conductance_dynamic":
                branch_layer.set_forward_dynamic_grad_scaling(True)
            self.branch_layers.append(branch_layer)

    @property
    def n_soma(self) -> int:
        return self._n_soma

    @property
    def output_dim(self) -> int:
        return self._n_soma

    @property
    def level_dims(self) -> list[int]:
        return self._level_dims

    def get_decays(self) -> torch.Tensor:
        """Return per-level decay factors, computing from log_tau if learnable."""
        if self.learnable_tau:
            tau = self.log_tau.exp()
            return torch.exp(-self.dt / tau)
        return self.decays

    def set_level_silencing(self, mask: Optional[torch.Tensor]) -> None:
        """Set an optional per-level lesion mask used during recurrent analysis.

        Supported shapes:
        - [n_levels]: same silencing pattern for every sample in the batch
        - [batch, n_levels]: sample-specific silencing per level
        Values are interpreted as 1=silence, 0=keep.
        """
        self._active_level_silencing = mask

    def clear_level_silencing(self) -> None:
        self._active_level_silencing = None

    def _level_keep_mask(
        self,
        level_idx: int,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if self._active_level_silencing is None:
            return None

        mask = self._active_level_silencing.to(device=device)
        if mask.dim() == 1:
            level_mask = mask[level_idx].expand(batch_size)
        elif mask.dim() == 2:
            level_mask = mask[:, level_idx]
        else:
            raise ValueError(
                "Level silencing mask must have shape [n_levels] or [batch, n_levels]"
            )

        keep = 1.0 - level_mask.to(dtype=dtype).clamp(0.0, 1.0)
        return keep.unsqueeze(-1)

    @property
    def current_taus(self) -> torch.Tensor:
        """Current tau values (useful for inspection/logging)."""
        if self.learnable_tau:
            return self.log_tau.exp().detach()
        return -self.dt / self.decays.clamp(min=1e-8).log()

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> DendriNetState:
        return DendriNetState.zeros(
            level_dims=self._level_dims,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            soma_dim=(
                self._n_soma
                if self.spiking_soma is not None or self.soma_feedback_enabled
                else None
            ),
            track_branch_voltage=(
                self.synapse_types_enabled or self.dendritic_spikes_enabled
            ),
            track_dendritic_spikes=self.dendritic_spikes_enabled,
            track_soma_output=self.soma_feedback_enabled,
        )

    def _resolve_dendritic_spike_levels(self, levels) -> set[int]:
        if not self.dendritic_spikes_enabled:
            return set()
        dendritic_levels = list(range(max(self.n_levels - 1, 0)))
        if isinstance(levels, str):
            key = levels.lower()
            if key in {"all", "dendritic", "non_soma", "non-soma"}:
                return set(dendritic_levels)
            if key == "distal":
                return {0} if dendritic_levels else set()
            if key == "proximal":
                return {dendritic_levels[-1]} if dendritic_levels else set()
            if key in {"none", ""}:
                return set()
            raise ValueError(
                "dendritic_spike_levels must be one of 'non_soma', 'all', "
                "'distal', 'proximal', 'none', or a list of non-soma level indices, "
                f"got {levels!r}"
            )
        resolved = set()
        for value in levels:
            level_idx = int(value)
            if level_idx < 0 or level_idx >= self.n_levels - 1:
                raise ValueError(
                    "dendritic_spike_levels entries are distal-to-soma non-soma "
                    f"indices in [0, {self.n_levels - 2}], got {level_idx}"
                )
            resolved.add(level_idx)
        return resolved

    def _resolve_soma_feedback_levels(self, levels) -> set[int]:
        if not self.soma_feedback_enabled:
            return set()
        dendritic_levels = list(range(max(self.n_levels - 1, 0)))
        if isinstance(levels, str):
            key = levels.lower()
            if key in {"all", "*"}:
                return set(range(self.n_levels))
            if key in {"dendritic", "non_soma", "non-soma"}:
                return set(dendritic_levels)
            if key == "distal":
                return {0} if dendritic_levels else set()
            if key == "proximal":
                return {dendritic_levels[-1]} if dendritic_levels else set()
            if key == "soma":
                return {self.n_levels - 1}
            if key in {"none", ""}:
                return set()
            raise ValueError(
                "soma_feedback_levels must be one of 'non_soma', 'all', "
                "'distal', 'proximal', 'soma', 'none', or a list of level "
                f"indices, got {levels!r}"
            )
        resolved = set()
        for value in levels:
            level_idx = int(value)
            if level_idx < 0 or level_idx >= self.n_levels:
                raise ValueError(
                    "soma_feedback_levels entries are distal-to-soma level "
                    f"indices in [0, {self.n_levels - 1}], got {level_idx}"
                )
            resolved.add(level_idx)
        return resolved

    def _previous_soma_feedback_source(
        self,
        state: DendriNetState,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if not self.soma_feedback_enabled:
            return None
        if self.soma_feedback_source == "voltage":
            source = state.v_soma if state.v_soma is not None else state.soma_output
        elif self.soma_feedback_source == "spike_readout":
            source = (
                state.spike_readout
                if self.spiking_soma is not None and state.spike_readout is not None
                else state.soma_output
            )
        else:
            source = state.soma_output
        if source is None:
            return torch.zeros(batch_size, self._n_soma, device=device, dtype=dtype)
        return source.to(device=device, dtype=dtype)

    def _soma_feedback_for_level(
        self,
        level_idx: int,
        source: torch.Tensor | None,
        reference: torch.Tensor,
    ) -> torch.Tensor | None:
        if (
            not self.soma_feedback_enabled
            or source is None
            or level_idx not in self.soma_feedback_level_indices
        ):
            return None
        strength_idx = (
            self._soma_feedback_strength_index[level_idx]
            if self.soma_feedback_per_level
            else 0
        )
        strength = self.soma_feedback_strength[strength_idx].to(
            device=reference.device,
            dtype=reference.dtype,
        )
        feedback = self.soma_feedback_projections[level_idx](source)
        return feedback.to(device=reference.device, dtype=reference.dtype) * strength

    def _validate_additive_synapse_reversals(self) -> None:
        for synapse_type in self.synapse_types.excitatory.types:
            if not math.isclose(float(synapse_type.reversal_potential), 1.0):
                raise ValueError(
                    "synapse_types with use_shunting=false require normalized "
                    "additive reversal potentials: excitatory=1.0 and "
                    "inhibitory=0.0. Set use_shunting=true for non-normalized "
                    f"reversal_potential on {synapse_type.name!r}."
                )
        for synapse_type in self.synapse_types.inhibitory.types:
            if not math.isclose(float(synapse_type.reversal_potential), 0.0):
                raise ValueError(
                    "synapse_types with use_shunting=false require normalized "
                    "additive reversal potentials: excitatory=1.0 and "
                    "inhibitory=0.0. Set use_shunting=true for non-normalized "
                    f"reversal_potential on {synapse_type.name!r}."
                )

    def _apply_soma_dynamics(
        self,
        soma_voltage: torch.Tensor,
        state: DendriNetState,
    ) -> tuple[
        torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
    ]:
        if self.spiking_soma is None:
            v_soma = soma_voltage if self.soma_feedback_enabled else state.v_soma
            return (
                soma_voltage,
                v_soma,
                state.refractory_counter,
                state.spike_readout,
            )
        return self.spiking_soma(
            soma_voltage,
            state.v_soma,
            state.refractory_counter,
            state.spike_readout,
        )[:4]

    def _previous_dendritic_spike_state(
        self,
        values: list[torch.Tensor] | None,
        level_idx: int,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if values is None or level_idx >= len(values):
            return torch.zeros_like(reference)
        return values[level_idx]

    def _apply_dendritic_spike_dynamics(
        self,
        *,
        level_idx: int,
        branch_voltage: torch.Tensor,
        state: DendriNetState,
        keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prev_plateau = self._previous_dendritic_spike_state(
            state.dendritic_spike_plateau,
            level_idx,
            branch_voltage,
        )
        prev_refractory = self._previous_dendritic_spike_state(
            state.dendritic_spike_refractory,
            level_idx,
            branch_voltage,
        )
        if keep_mask is not None:
            prev_plateau = prev_plateau * keep_mask
            prev_refractory = prev_refractory * keep_mask

        if level_idx not in self.dendritic_spike_level_indices:
            zeros = torch.zeros_like(branch_voltage)
            return branch_voltage, zeros, zeros, zeros

        active = (prev_refractory <= 0).to(dtype=branch_voltage.dtype)
        event = SurrogateSpike.apply(
            branch_voltage - self.dendritic_spike_threshold,
            self.dendritic_spike_surrogate_beta,
        )
        event = event * active

        if self.dendritic_spike_mode == "plateau":
            plateau_decay = torch.as_tensor(
                math.exp(-self.dt / self.dendritic_spike_plateau_tau),
                device=branch_voltage.device,
                dtype=branch_voltage.dtype,
            )
            plateau = plateau_decay * prev_plateau + (1 - plateau_decay) * event
        else:
            plateau = event

        next_refractory = torch.clamp(prev_refractory - 1, min=0)
        if self.dendritic_spike_refractory_steps > 0:
            next_refractory = torch.where(
                event.detach() > 0,
                torch.full_like(
                    next_refractory, float(self.dendritic_spike_refractory_steps)
                ),
                next_refractory,
            )

        amplitude = self.dendritic_spike_plateau_amplitude
        if self.dendritic_spike_propagation == "additive":
            modified_voltage = branch_voltage + amplitude * plateau
        elif self.dendritic_spike_propagation == "replace":
            modified_voltage = branch_voltage * (1 - plateau) + amplitude * plateau
        else:
            modified_voltage = branch_voltage * (1 + amplitude * plateau)

        if keep_mask is not None:
            modified_voltage = modified_voltage * keep_mask
            plateau = plateau * keep_mask
            next_refractory = next_refractory * keep_mask
            event = event * keep_mask

        return modified_voltage, plateau, next_refractory, event

    def forward(
        self,
        x: torch.Tensor,
        inhibitory_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Feedforward pass, using one stateful step when opt-in dynamics need state."""
        if (
            self.synapse_types_enabled
            or self.spiking_soma is not None
            or self.dendritic_spikes_enabled
            or self.soma_feedback_enabled
        ):
            output, _ = self.step(x=x, inhibitory_input=inhibitory_input)
            return output
        branch_output = None
        for layer in self.branch_layers:
            branch_output = layer(
                x=x,
                inhibitory_input=inhibitory_input,
                branch_input=branch_output,
                recurrent_input=None,
                rec_inhibitory_input=None,
            )
        return branch_output

    def _ensure_tensor(
        self,
        value,
        batch_size: int,
        level_idx: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Ensure value is a tensor; if scalar 0, create zeros tensor."""
        if torch.is_tensor(value):
            return value
        return torch.zeros(
            batch_size, self._level_dims[level_idx], device=device, dtype=dtype
        )

    def _typed_trace_key(self, compartment: str, synapse_type: str) -> str:
        return f"{compartment}:{synapse_type}"

    def _previous_typed_trace(
        self,
        traces: dict[str, list[torch.Tensor]] | None,
        key: str,
        level_idx: int,
        reference: torch.Tensor,
        default: float = 0.0,
    ) -> torch.Tensor:
        if traces is None or key not in traces or level_idx >= len(traces[key]):
            return torch.full_like(reference, float(default))
        return traces[key][level_idx]

    def _decay_for_tau(
        self, tau: float | None, legacy_decay: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        if tau is None:
            return legacy_decay.to(device=reference.device, dtype=reference.dtype)
        return torch.exp(
            torch.as_tensor(
                -self.dt / tau, device=reference.device, dtype=reference.dtype
            )
        )

    def _synapse_decay(
        self,
        synapse_type: SynapseTypeConfig,
        legacy_decay: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if synapse_type.tau_decay is None:
            return legacy_decay
        return torch.exp(
            torch.as_tensor(
                -self.dt / synapse_type.tau_decay, device=device, dtype=dtype
            )
        )

    def _apply_synapse_nonlinearity(
        self, raw: torch.Tensor, synapse_type: SynapseTypeConfig
    ) -> torch.Tensor:
        mode = synapse_type.nonlinearity
        if mode == "relu":
            raw = torch.relu(raw)
        elif mode == "sigmoid":
            raw = torch.sigmoid(raw)
        elif mode == "tanh":
            raw = torch.tanh(raw)
        elif mode == "softplus":
            raw = torch.nn.functional.softplus(raw)

        power = float(synapse_type.conductance_power)
        if power != 1.0:
            magnitude = raw.abs()
            if power < 1.0 and torch.is_floating_point(magnitude):
                zero = magnitude == 0
                safe_magnitude = torch.where(
                    zero, torch.ones_like(magnitude), magnitude
                )
                powered = safe_magnitude.pow(power)
                powered = torch.where(zero, torch.zeros_like(powered), powered)
            else:
                powered = magnitude.pow(power)
            raw = raw.sign() * powered
        return raw

    def _apply_short_term_plasticity(
        self,
        *,
        raw: torch.Tensor,
        prev_typed_traces: dict[str, list[torch.Tensor]] | None,
        new_typed_traces: dict[str, list[torch.Tensor]],
        key: str,
        level_idx: int,
        synapse_type: SynapseTypeConfig,
    ) -> torch.Tensor:
        if not synapse_type.stp_enabled:
            return raw

        u_key = f"{key}:stp_u"
        x_key = f"{key}:stp_x"
        prev_u = self._previous_typed_trace(
            prev_typed_traces,
            u_key,
            level_idx,
            raw,
            default=synapse_type.stp_u0,
        )
        prev_x = self._previous_typed_trace(
            prev_typed_traces, x_key, level_idx, raw, default=1.0
        )
        # Aggregate STP uses the local raw current as a presynaptic-rate proxy.
        activity_source = raw if synapse_type.stp_differentiable_state else raw.detach()
        activity = torch.sigmoid(activity_source)
        dt_over_tau_u = self.dt / float(synapse_type.stp_tau_u)
        dt_over_tau_x = self.dt / float(synapse_type.stp_tau_x)
        next_u = prev_u + dt_over_tau_u * (synapse_type.stp_u0 - prev_u)
        next_u = (
            next_u + float(synapse_type.stp_facilitation) * (1.0 - prev_u) * activity
        )
        next_x = prev_x + dt_over_tau_x * (1.0 - prev_x)
        next_x = (
            next_x - float(synapse_type.stp_depression) * next_u * prev_x * activity
        )
        next_u = next_u.clamp(0.0, 1.0)
        next_x = next_x.clamp(0.0, 1.0)
        new_typed_traces.setdefault(u_key, []).append(next_u)
        new_typed_traces.setdefault(x_key, []).append(next_x)
        return raw * next_u * next_x

    def _apply_voltage_gate(
        self,
        trace: torch.Tensor,
        synapse_type: SynapseTypeConfig,
        voltage_reference: torch.Tensor,
    ) -> torch.Tensor:
        gate_mode = synapse_type.voltage_gate
        if gate_mode == "none":
            return trace
        v_ref = voltage_reference.to(device=trace.device, dtype=trace.dtype)
        v_gate = v_ref * float(synapse_type.voltage_scale_mv) + float(
            synapse_type.mg_v_offset
        )
        if gate_mode == "nmda_magnesium":
            exponent = -v_gate / float(synapse_type.mg_slope)
            gate = 1.0 / (
                1.0
                + float(synapse_type.mg_concentration)
                * torch.exp(exponent)
                / float(synapse_type.mg_scale)
            )
            return trace * gate
        if gate_mode == "sigmoid":
            gate = torch.sigmoid(v_gate / float(synapse_type.mg_slope))
            return trace * gate
        return trace

    def _integrate_single_or_double_exponential(
        self,
        *,
        raw: torch.Tensor,
        prev_typed_traces: dict[str, list[torch.Tensor]] | None,
        new_typed_traces: dict[str, list[torch.Tensor]],
        key: str,
        level_idx: int,
        synapse_type: SynapseTypeConfig,
        legacy_decay: torch.Tensor,
    ) -> torch.Tensor:
        if synapse_type.tau_rise is None:
            prev_trace = self._previous_typed_trace(
                prev_typed_traces, key, level_idx, raw
            )
            decay = self._synapse_decay(
                synapse_type,
                legacy_decay,
                device=raw.device,
                dtype=raw.dtype,
            )
            return decay * prev_trace + (1 - decay) * (
                raw * float(synapse_type.fraction)
            )

        drive = raw * float(synapse_type.fraction)
        rise_key = f"{key}:rise"
        decay_key = f"{key}:decay"
        prev_rise = self._previous_typed_trace(
            prev_typed_traces, rise_key, level_idx, raw
        )
        prev_decay = self._previous_typed_trace(
            prev_typed_traces, decay_key, level_idx, raw
        )
        rise_decay = self._decay_for_tau(
            synapse_type.tau_rise, legacy_decay, reference=raw
        )
        decay_decay = self._decay_for_tau(
            synapse_type.tau_decay, legacy_decay, reference=raw
        )
        rise_trace = rise_decay * prev_rise + drive
        decay_trace = decay_decay * prev_decay + drive
        new_typed_traces.setdefault(rise_key, []).append(rise_trace)
        new_typed_traces.setdefault(decay_key, []).append(decay_trace)
        return (decay_trace - rise_trace).clamp_min(0.0)

    def _integrate_typed_compartment(
        self,
        *,
        raw: torch.Tensor,
        prev_typed_traces: dict[str, list[torch.Tensor]] | None,
        new_typed_traces: dict[str, list[torch.Tensor]],
        level_idx: int,
        compartment: str,
        synapse_types: list[SynapseTypeConfig],
        legacy_decay: torch.Tensor,
        voltage_reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        conductance = torch.zeros_like(raw)
        shunting_current = torch.zeros_like(raw)
        additive_drive = torch.zeros_like(raw)

        for synapse_type in synapse_types:
            key = self._typed_trace_key(compartment, synapse_type.name)
            raw_type = self._apply_synapse_nonlinearity(raw, synapse_type)
            raw_type = self._apply_short_term_plasticity(
                raw=raw_type,
                prev_typed_traces=prev_typed_traces,
                new_typed_traces=new_typed_traces,
                key=key,
                level_idx=level_idx,
                synapse_type=synapse_type,
            )
            trace = self._integrate_single_or_double_exponential(
                raw=raw_type,
                prev_typed_traces=prev_typed_traces,
                new_typed_traces=new_typed_traces,
                key=key,
                level_idx=level_idx,
                synapse_type=synapse_type,
                legacy_decay=legacy_decay,
            )
            trace = self._apply_voltage_gate(trace, synapse_type, voltage_reference)
            new_typed_traces.setdefault(key, []).append(trace)
            conductance = conductance + trace
            reversal = float(synapse_type.reversal_potential)
            shunting_current = shunting_current + trace * reversal
            if synapse_type.polarity == "inhibitory":
                additive_drive = additive_drive + trace * (1.0 - reversal)
            else:
                additive_drive = additive_drive + trace * reversal

        return conductance, shunting_current, additive_drive

    def _branch_conductance_tensor(
        self,
        layer: DendriticBranchLayer,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not layer.input_branches or not hasattr(layer, "branches_to_output"):
            return torch.zeros_like(reference)
        conductance = layer.branches_to_output.sum_conductances()
        return conductance.to(device=reference.device, dtype=reference.dtype).unsqueeze(
            0
        )

    def _maybe_update_dynamic_grad_scales(
        self,
        layer: DendriticBranchLayer,
        denominator: torch.Tensor | None,
    ) -> None:
        if not self.training:
            return
        if "conductance_dynamic" not in [
            layer.topk_strategy,
            layer.blocklinear_strategy,
        ]:
            return
        layer.compute_grad_scales(
            g_total=denominator if layer.use_shunting else None,
            use_forward_hooks=True,
        )

    def _apply_typed_voltage(
        self,
        layer: DendriticBranchLayer,
        *,
        trace_E: torch.Tensor,
        trace_E_rec: torch.Tensor,
        trace_I: torch.Tensor,
        trace_I_rec: torch.Tensor,
        current_E: torch.Tensor,
        current_E_rec: torch.Tensor,
        current_I: torch.Tensor,
        current_I_rec: torch.Tensor,
        drive_E: torch.Tensor,
        drive_E_rec: torch.Tensor,
        drive_I: torch.Tensor,
        drive_I_rec: torch.Tensor,
        trace_branch: torch.Tensor,
        branch_conductance: torch.Tensor,
    ) -> torch.Tensor:
        denominator = None
        if layer.use_shunting:
            numerator = current_E + current_E_rec + current_I + current_I_rec
            numerator = numerator + trace_branch
            denominator = 1 + trace_E + trace_E_rec + trace_I + trace_I_rec
            denominator = denominator + branch_conductance
            voltage = numerator / (denominator + layer.epsilon)
        else:
            voltage = drive_E + drive_E_rec - drive_I - drive_I_rec + trace_branch
            voltage = layer.normalize_additive_voltage(voltage)
        self._maybe_update_dynamic_grad_scales(layer, denominator)
        return layer.reactivation(voltage)

    def step(
        self,
        x: torch.Tensor,
        inhibitory_input: Optional[torch.Tensor] = None,
        recurrent_input: Optional[torch.Tensor] = None,
        rec_inhibitory_input: Optional[torch.Tensor] = None,
        state: Optional[DendriNetState] = None,
    ) -> tuple[torch.Tensor, DendriNetState]:
        """Recurrent step with per-level trace integration.

        Args:
            x: FF excitatory input [batch, excitatory_input_dim].
            inhibitory_input: FF inhibitory input [batch, inhibitory_input_dim].
            recurrent_input: Recurrent excitatory input [batch, rec_e_dim].
            rec_inhibitory_input: Recurrent inhibitory input [batch, rec_i_dim].
            state: Previous DendriNetState (auto-initialized if None).

        Returns:
            output: Soma-level voltage [batch, n_soma].
            new_state: Updated DendriNetState with integrated traces.
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        if state is None:
            state = self.init_state(batch_size=batch_size, device=device, dtype=dtype)

        # Precompute all decay factors (differentiable when learnable_tau=True)
        all_decays = self.get_decays()

        # Build new trace lists in-place (avoids allocating zeros that are overwritten)
        new_trace_E = []
        new_trace_E_rec = []
        new_trace_I = []
        new_trace_I_rec = []
        new_trace_branch = []
        new_typed_traces = {} if self.synapse_types_enabled else None
        new_dendritic_spike_plateau = [] if self.dendritic_spikes_enabled else None
        new_dendritic_spike_refractory = [] if self.dendritic_spikes_enabled else None
        new_dendritic_spike_events = [] if self.dendritic_spikes_enabled else None

        branch_voltage = None
        level_voltage_cache: list[torch.Tensor] = []
        soma_feedback_source = self._previous_soma_feedback_source(
            state,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        for level_idx, layer in enumerate(self.branch_layers):
            # 1. Compute raw synaptic currents from all 4 compartments
            raw_E, raw_E_rec, raw_I, raw_I_rec = layer.compute_raw_currents(
                x=x,
                inhibitory_input=inhibitory_input,
                recurrent_input=recurrent_input,
                rec_inhibitory_input=rec_inhibitory_input,
            )
            raw_E = self._ensure_tensor(raw_E, batch_size, level_idx, device, dtype)
            raw_E_rec = self._ensure_tensor(
                raw_E_rec, batch_size, level_idx, device, dtype
            )
            raw_I = self._ensure_tensor(raw_I, batch_size, level_idx, device, dtype)
            raw_I_rec = self._ensure_tensor(
                raw_I_rec, batch_size, level_idx, device, dtype
            )

            # 2. Branch aggregation from child level
            if layer.input_branches and branch_voltage is not None:
                raw_branch = layer.branches_to_output(branch_voltage)
            else:
                raw_branch = torch.zeros(
                    batch_size, self._level_dims[level_idx], device=device, dtype=dtype
                )
            branch_conductance = self._branch_conductance_tensor(layer, raw_branch)

            keep_mask = self._level_keep_mask(
                level_idx=level_idx,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            prev_trace_E = state.trace_E[level_idx]
            prev_trace_E_rec = state.trace_E_rec[level_idx]
            prev_trace_I = state.trace_I[level_idx]
            prev_trace_I_rec = state.trace_I_rec[level_idx]
            prev_trace_branch = state.trace_branch[level_idx]
            prev_branch_voltage = (
                state.branch_voltage[level_idx]
                if state.branch_voltage is not None
                and level_idx < len(state.branch_voltage)
                else prev_trace_branch
            )
            if keep_mask is not None:
                raw_E = raw_E * keep_mask
                raw_E_rec = raw_E_rec * keep_mask
                raw_I = raw_I * keep_mask
                raw_I_rec = raw_I_rec * keep_mask
                raw_branch = raw_branch * keep_mask
                branch_conductance = branch_conductance * keep_mask
                prev_trace_E = prev_trace_E * keep_mask
                prev_trace_E_rec = prev_trace_E_rec * keep_mask
                prev_trace_I = prev_trace_I * keep_mask
                prev_trace_I_rec = prev_trace_I_rec * keep_mask
                prev_trace_branch = prev_trace_branch * keep_mask
                prev_branch_voltage = prev_branch_voltage * keep_mask

            # 3. Leaky-integrate all 5 trace types
            decay = all_decays[level_idx].to(device=device, dtype=dtype)
            one_minus_decay = 1 - decay

            new_trace_branch.append(
                decay * prev_trace_branch + one_minus_decay * raw_branch
            )
            soma_feedback = self._soma_feedback_for_level(
                level_idx,
                soma_feedback_source,
                raw_branch,
            )
            if soma_feedback is not None and keep_mask is not None:
                soma_feedback = soma_feedback * keep_mask
            effective_trace_branch = new_trace_branch[level_idx]
            effective_branch_conductance = branch_conductance
            soma_feedback_gate = None
            if soma_feedback is not None:
                if self.soma_feedback_mode == "additive":
                    effective_trace_branch = effective_trace_branch + soma_feedback
                elif self.soma_feedback_mode == "shunting":
                    soma_feedback_conductance = torch.relu(soma_feedback)
                    effective_trace_branch = (
                        effective_trace_branch
                        + soma_feedback_conductance * self.soma_feedback_reversal
                    )
                    effective_branch_conductance = (
                        effective_branch_conductance + soma_feedback_conductance
                    )
                else:
                    soma_feedback_gate = torch.tanh(soma_feedback)

            if self.synapse_types_enabled:
                assert new_typed_traces is not None
                trace_E, current_E, drive_E = self._integrate_typed_compartment(
                    raw=raw_E,
                    prev_typed_traces=state.typed_traces,
                    new_typed_traces=new_typed_traces,
                    level_idx=level_idx,
                    compartment="ff_exc",
                    synapse_types=self.synapse_types.excitatory.types,
                    legacy_decay=decay,
                    voltage_reference=prev_branch_voltage,
                )
                trace_E_rec, current_E_rec, drive_E_rec = (
                    self._integrate_typed_compartment(
                        raw=raw_E_rec,
                        prev_typed_traces=state.typed_traces,
                        new_typed_traces=new_typed_traces,
                        level_idx=level_idx,
                        compartment="rec_exc",
                        synapse_types=self.synapse_types.excitatory.types,
                        legacy_decay=decay,
                        voltage_reference=prev_branch_voltage,
                    )
                )
                trace_I, current_I, drive_I = self._integrate_typed_compartment(
                    raw=raw_I,
                    prev_typed_traces=state.typed_traces,
                    new_typed_traces=new_typed_traces,
                    level_idx=level_idx,
                    compartment="ff_inh",
                    synapse_types=self.synapse_types.inhibitory.types,
                    legacy_decay=decay,
                    voltage_reference=prev_branch_voltage,
                )
                trace_I_rec, current_I_rec, drive_I_rec = (
                    self._integrate_typed_compartment(
                        raw=raw_I_rec,
                        prev_typed_traces=state.typed_traces,
                        new_typed_traces=new_typed_traces,
                        level_idx=level_idx,
                        compartment="rec_inh",
                        synapse_types=self.synapse_types.inhibitory.types,
                        legacy_decay=decay,
                        voltage_reference=prev_branch_voltage,
                    )
                )
                new_trace_E.append(trace_E)
                new_trace_E_rec.append(trace_E_rec)
                new_trace_I.append(trace_I)
                new_trace_I_rec.append(trace_I_rec)

                # 4. Shunting/additive computation from typed conductances.
                branch_voltage = self._apply_typed_voltage(
                    layer,
                    trace_E=trace_E,
                    trace_E_rec=trace_E_rec,
                    trace_I=trace_I,
                    trace_I_rec=trace_I_rec,
                    current_E=current_E,
                    current_E_rec=current_E_rec,
                    current_I=current_I,
                    current_I_rec=current_I_rec,
                    drive_E=drive_E,
                    drive_E_rec=drive_E_rec,
                    drive_I=drive_I,
                    drive_I_rec=drive_I_rec,
                    trace_branch=effective_trace_branch,
                    branch_conductance=effective_branch_conductance,
                )
            else:
                new_trace_E.append(decay * prev_trace_E + one_minus_decay * raw_E)
                new_trace_E_rec.append(
                    decay * prev_trace_E_rec + one_minus_decay * raw_E_rec
                )
                new_trace_I.append(decay * prev_trace_I + one_minus_decay * raw_I)
                new_trace_I_rec.append(
                    decay * prev_trace_I_rec + one_minus_decay * raw_I_rec
                )

                # 4. Shunting/additive computation from integrated traces.
                voltage, denominator = layer.voltage_from_currents(
                    trace_E=new_trace_E[level_idx],
                    trace_E_rec=new_trace_E_rec[level_idx],
                    trace_I=new_trace_I[level_idx],
                    trace_I_rec=new_trace_I_rec[level_idx],
                    trace_branch=effective_trace_branch,
                    branch_conductance=effective_branch_conductance,
                )
                self._maybe_update_dynamic_grad_scales(layer, denominator)
                branch_voltage = layer.reactivation(voltage)
            if soma_feedback_gate is not None:
                branch_voltage = branch_voltage * (1.0 + soma_feedback_gate)
            if keep_mask is not None:
                branch_voltage = branch_voltage * keep_mask
            if self.dendritic_spikes_enabled:
                branch_voltage, plateau, refractory, event = (
                    self._apply_dendritic_spike_dynamics(
                        level_idx=level_idx,
                        branch_voltage=branch_voltage,
                        state=state,
                        keep_mask=keep_mask,
                    )
                )
                assert new_dendritic_spike_plateau is not None
                assert new_dendritic_spike_refractory is not None
                assert new_dendritic_spike_events is not None
                new_dendritic_spike_plateau.append(plateau)
                new_dendritic_spike_refractory.append(refractory)
                new_dendritic_spike_events.append(event)
            level_voltage_cache.append(branch_voltage)

        output, v_soma, refractory_counter, spike_readout = self._apply_soma_dynamics(
            branch_voltage,
            state,
        )

        new_state = DendriNetState(
            trace_E=new_trace_E,
            trace_E_rec=new_trace_E_rec,
            trace_I=new_trace_I,
            trace_I_rec=new_trace_I_rec,
            trace_branch=new_trace_branch,
            branch_voltage=(
                level_voltage_cache
                if self.synapse_types_enabled or self.dendritic_spikes_enabled
                else None
            ),
            typed_traces=new_typed_traces,
            soma_output=output if self.soma_feedback_enabled else None,
            v_soma=v_soma,
            refractory_counter=refractory_counter,
            spike_readout=spike_readout,
            dendritic_spike_plateau=new_dendritic_spike_plateau,
            dendritic_spike_refractory=new_dendritic_spike_refractory,
            dendritic_spike_events=new_dendritic_spike_events,
        )

        # Collect routing info when enabled
        if self.store_routing:
            eps = 1e-8
            level_voltages = []
            ff_rec_ratios = []
            for l_idx in range(self.n_levels):
                # Per-level contribution: L2 norm of voltage (computed from traces)
                v = level_voltage_cache[l_idx]
                level_voltages.append(v.norm(dim=-1))  # [batch]

                # FF / REC ratio: fraction of signal from feedforward compartments
                ff_mag = new_trace_E[l_idx].abs().sum(dim=-1) + new_trace_I[
                    l_idx
                ].abs().sum(dim=-1)
                rec_mag = new_trace_E_rec[l_idx].abs().sum(dim=-1) + new_trace_I_rec[
                    l_idx
                ].abs().sum(dim=-1)
                ff_rec_ratios.append(ff_mag / (ff_mag + rec_mag + eps))  # [batch]

            self._last_routing_info = {
                "level_contributions": torch.stack(
                    level_voltages, dim=-1
                ),  # [batch, n_levels]
                "ff_rec_ratios": torch.stack(
                    ff_rec_ratios, dim=-1
                ),  # [batch, n_levels]
            }

        return output, new_state
