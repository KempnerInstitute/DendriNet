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
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState


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
        dt: float = 1.0,
    ):
        super().__init__()
        self.pop_config = pop_config
        self.dt = dt

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

            branch_layer = DendriticBranchLayer(
                output_dim=out_dim,
                excitatory_input_dim=exc_in,
                excitatory_synapses_per_branch=exc_syn,
                inhibitory_input_dim=inh_in,
                inhibitory_synapses_per_branch=inh_syn,
                input_branch_factor=input_branch_factors[level_idx],
                recurrent_input_dim=rec_in,
                recurrent_synapses_per_branch=rec_syn,
                recurrent_forbidden_input_index_per_output=recurrent_self_forbidden,
                rec_inhibitory_input_dim=rec_inh_in,
                rec_inhibitory_synapses_per_branch=rec_inh_syn,
                rec_inhibitory_forbidden_input_index_per_output=rec_inhibitory_self_forbidden,
                layer_idx=self.n_levels - level_idx,
                **common_kwargs,
            )
            self.branch_layers.append(branch_layer)

    @property
    def n_soma(self) -> int:
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
        )

    def forward(
        self,
        x: torch.Tensor,
        inhibitory_input: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Feedforward pass -- identical to DendriNet.forward()."""
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

        branch_voltage = None
        level_voltage_cache: list[torch.Tensor] = []

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
            if keep_mask is not None:
                raw_E = raw_E * keep_mask
                raw_E_rec = raw_E_rec * keep_mask
                raw_I = raw_I * keep_mask
                raw_I_rec = raw_I_rec * keep_mask
                raw_branch = raw_branch * keep_mask
                prev_trace_E = prev_trace_E * keep_mask
                prev_trace_E_rec = prev_trace_E_rec * keep_mask
                prev_trace_I = prev_trace_I * keep_mask
                prev_trace_I_rec = prev_trace_I_rec * keep_mask
                prev_trace_branch = prev_trace_branch * keep_mask

            # 3. Leaky-integrate all 5 trace types
            decay = all_decays[level_idx].to(device=device, dtype=dtype)
            one_minus_decay = 1 - decay

            new_trace_E.append(decay * prev_trace_E + one_minus_decay * raw_E)
            new_trace_E_rec.append(
                decay * prev_trace_E_rec + one_minus_decay * raw_E_rec
            )
            new_trace_I.append(decay * prev_trace_I + one_minus_decay * raw_I)
            new_trace_I_rec.append(
                decay * prev_trace_I_rec + one_minus_decay * raw_I_rec
            )
            new_trace_branch.append(
                decay * prev_trace_branch + one_minus_decay * raw_branch
            )

            # 4. Shunting from integrated traces
            branch_voltage = layer.shunt_from_currents(
                trace_E=new_trace_E[level_idx] + new_trace_branch[level_idx],
                trace_E_rec=new_trace_E_rec[level_idx],
                trace_I=new_trace_I[level_idx],
                trace_I_rec=new_trace_I_rec[level_idx],
            )
            if keep_mask is not None:
                branch_voltage = branch_voltage * keep_mask
            level_voltage_cache.append(branch_voltage)

        new_state = DendriNetState(
            trace_E=new_trace_E,
            trace_E_rec=new_trace_E_rec,
            trace_I=new_trace_I,
            trace_I_rec=new_trace_I_rec,
            trace_branch=new_trace_branch,
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

        return branch_voltage, new_state
