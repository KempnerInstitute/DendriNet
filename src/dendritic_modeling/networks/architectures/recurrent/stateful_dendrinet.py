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
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    sample_configured_mask,
)
from dendritic_modeling.networks.architectures.recurrent.dendritic_geometry import (
    compute_input_branch_factors,
    compute_level_sizes,
    compute_output_owner_index_per_level,
    resolve_synapses_by_level,
)
from dendritic_modeling.networks.architectures.recurrent.dendritic_spikes import (
    DendriticSpikeDynamics,
    previous_dendritic_spike_state,
    resolve_dendritic_spike_levels,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState
from dendritic_modeling.networks.architectures.recurrent.spiking_soma import (
    LIFSoma,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_typed_dynamics import (
    StatefulTypedDynamicsMixin,
)
from dendritic_modeling.networks.architectures.recurrent.synapse_types import (
    build_synapse_type_set,
)


class StatefulDendriNet(StatefulTypedDynamicsMixin, nn.Module):
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
        self._level_dims = compute_level_sizes(
            n_soma=pop_config.n_neurons,
            branch_factors=pop_config.branch_factors,
        )
        self._output_owner_index_per_level = compute_output_owner_index_per_level(
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
        self.dendritic_spike_level_indices = resolve_dendritic_spike_levels(
            self.dendritic_spikes_enabled,
            pop_config.dendritic_spike_levels,
            self.n_levels,
        )
        self.dendritic_spike_dynamics = DendriticSpikeDynamics(
            enabled=self.dendritic_spikes_enabled,
            level_indices=self.dendritic_spike_level_indices,
            mode=self.dendritic_spike_mode,
            threshold=self.dendritic_spike_threshold,
            plateau_amplitude=self.dendritic_spike_plateau_amplitude,
            plateau_tau=self.dendritic_spike_plateau_tau,
            refractory_steps=self.dendritic_spike_refractory_steps,
            surrogate_beta=self.dendritic_spike_surrogate_beta,
            propagation=self.dendritic_spike_propagation,
            dt=self.dt,
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
        input_branch_factors = compute_input_branch_factors(pop_config.branch_factors)
        ff_exc_by_level = resolve_synapses_by_level(
            per_level=pop_config.ff_excitatory_synapses_by_level,
            fallback=pop_config.ff_excitatory_synapses,
            n_levels=self.n_levels,
            field_name="ff_excitatory_synapses_by_level",
        )
        ff_inh_by_level = resolve_synapses_by_level(
            per_level=pop_config.ff_inhibitory_synapses_by_level,
            fallback=pop_config.ff_inhibitory_synapses,
            n_levels=self.n_levels,
            field_name="ff_inhibitory_synapses_by_level",
        )
        rec_exc_by_level = resolve_synapses_by_level(
            per_level=pop_config.rec_excitatory_synapses_by_level,
            fallback=pop_config.rec_excitatory_synapses,
            n_levels=self.n_levels,
            field_name="rec_excitatory_synapses_by_level",
        )
        rec_inh_by_level = resolve_synapses_by_level(
            per_level=pop_config.rec_inhibitory_synapses_by_level,
            fallback=pop_config.rec_inhibitory_synapses,
            n_levels=self.n_levels,
            field_name="rec_inhibitory_synapses_by_level",
        )
        direct_connection_masks = connection_masks_by_pathway or {}

        common_synapse_config = DendriticSynapseConfig.from_population_config(
            pop_config
        )

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
                synapse_config=common_synapse_config,
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

    def _apply_dendritic_spike_dynamics(
        self,
        *,
        level_idx: int,
        branch_voltage: torch.Tensor,
        state: DendriNetState,
        keep_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prev_plateau = previous_dendritic_spike_state(
            state.dendritic_spike_plateau,
            level_idx,
            branch_voltage,
        )
        prev_refractory = previous_dendritic_spike_state(
            state.dendritic_spike_refractory,
            level_idx,
            branch_voltage,
        )
        return self.dendritic_spike_dynamics.apply(
            level_idx=level_idx,
            branch_voltage=branch_voltage,
            prev_plateau=prev_plateau,
            prev_refractory=prev_refractory,
            keep_mask=keep_mask,
        )

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
