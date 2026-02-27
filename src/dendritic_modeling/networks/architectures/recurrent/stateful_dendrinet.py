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
    """Compute output dimensions per level (distal → soma order).

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
        dt: float = 1.0,
    ):
        super().__init__()
        self.pop_config = pop_config
        self.dt = dt

        # Routing analysis flags (set externally or via EINetwork)
        self.store_routing: bool = False
        self._last_routing_info: dict = {}
        self._n_soma = pop_config.n_neurons
        self._level_dims = _compute_level_sizes(
            n_soma=pop_config.n_neurons,
            branch_factors=pop_config.branch_factors,
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
        self.register_buffer(
            "decays",
            torch.tensor(
                [math.exp(-dt / tau) for tau in level_taus], dtype=torch.float32
            ),
        )

        # Build branch layers (distal → soma)
        self.branch_layers = nn.ModuleList()
        input_branch_factors = _compute_input_branch_factors(pop_config.branch_factors)

        # Common DendriticBranchLayer kwargs from PopulationConfig
        common_kwargs = {
            "topk_init_method": pop_config.topk_init_method,
            "use_shunting": pop_config.use_shunting,
            "reactivate": pop_config.reactivate,
            "reactivation_type": pop_config.reactivation_type,
            "reactivation_init_m": pop_config.reactivation_init_m,
            "reactivation_init_b": pop_config.reactivation_init_b,
            "weight_transform": pop_config.weight_transform,
            "epsilon": pop_config.epsilon,
            "dbl_init_method": pop_config.dbl_init_method,
            "efficient_blocklinear": pop_config.efficient_blocklinear,
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
                exc_in = _exc_dim(pop_config.ff_excitatory_synapses)
                exc_syn = _exc_syn(pop_config.ff_excitatory_synapses)
                inh_in = _inh_dim(
                    inhibitory_input_dim, pop_config.ff_inhibitory_synapses
                )
                inh_syn = _inh_syn(
                    inhibitory_input_dim, pop_config.ff_inhibitory_synapses
                )
                rec_in = _inh_dim(
                    recurrent_excitatory_input_dim, pop_config.rec_excitatory_synapses
                )
                rec_syn = _inh_syn(
                    recurrent_excitatory_input_dim, pop_config.rec_excitatory_synapses
                )
                rec_inh_in = _inh_dim(
                    recurrent_inhibitory_input_dim, pop_config.rec_inhibitory_synapses
                )
                rec_inh_syn = _inh_syn(
                    recurrent_inhibitory_input_dim, pop_config.rec_inhibitory_synapses
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
                rec_inhibitory_input_dim=rec_inh_in,
                rec_inhibitory_synapses_per_branch=rec_inh_syn,
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
        """Feedforward pass — identical to DendriNet.forward()."""
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

            # 3. Leaky-integrate all 5 trace types
            decay = self.decays[level_idx].to(device=device, dtype=dtype)
            one_minus_decay = 1 - decay

            new_trace_E.append(
                decay * state.trace_E[level_idx] + one_minus_decay * raw_E
            )
            new_trace_E_rec.append(
                decay * state.trace_E_rec[level_idx] + one_minus_decay * raw_E_rec
            )
            new_trace_I.append(
                decay * state.trace_I[level_idx] + one_minus_decay * raw_I
            )
            new_trace_I_rec.append(
                decay * state.trace_I_rec[level_idx] + one_minus_decay * raw_I_rec
            )
            new_trace_branch.append(
                decay * state.trace_branch[level_idx] + one_minus_decay * raw_branch
            )

            # 4. Shunting from integrated traces
            branch_voltage = layer.shunt_from_currents(
                trace_E=new_trace_E[level_idx] + new_trace_branch[level_idx],
                trace_E_rec=new_trace_E_rec[level_idx],
                trace_I=new_trace_I[level_idx],
                trace_I_rec=new_trace_I_rec[level_idx],
            )
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
