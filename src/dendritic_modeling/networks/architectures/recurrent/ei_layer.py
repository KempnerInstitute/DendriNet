"""
Unified E-I layer: feedforward special case and recurrent mode in one class.
"""

from dataclasses import replace
from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.recurrent.ei_config import EILayerConfig
from dendritic_modeling.networks.architectures.recurrent.ei_state import EIState
from dendritic_modeling.networks.architectures.recurrent.population_common import (
    population_requires_temporal_state,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)
from dendritic_modeling.networks.utils.forward_caches import detach_forward_caches
from dendritic_modeling.utils.stable_hash import stable_seed_offset


def _layer_population_config(population, layer_index: int, population_name: str):
    """Give one layer population its own indexed-topology stream.

    Layers and their E and I populations share one configured base seed; the
    offset by layer index and role keeps their sampled contacts independent,
    as the population-network builder does for named populations.
    """
    if population is None or population.indexed_seed is None:
        return population
    return replace(
        population,
        indexed_seed=(
            int(population.indexed_seed)
            + stable_seed_offset("ei_layer", int(layer_index), population_name)
        )
        % ((1 << 63) - 1),
    )


def _reject_recurrent_populations(config: EILayerConfig, layer_index: int) -> None:
    """Reject recurrent synapses and stateful dynamics in a non-recurrent layer.

    A non-recurrent layer passes no recurrent inputs, so recurrent synapses
    would not be built, and it runs stateful dynamics as a single step from
    a zero state; both need ``recurrent: true``.
    """
    for role in ("excitatory", "inhibitory"):
        population = getattr(config, role, None)
        if population is None:
            continue
        features = [
            name
            for name in (
                "rec_excitatory_synapses",
                "rec_inhibitory_synapses",
                "rec_excitatory_synapses_by_level",
                "rec_inhibitory_synapses_by_level",
            )
            if any(
                value > 0
                for value in (
                    getattr(population, name)
                    if isinstance(getattr(population, name), (list, tuple))
                    else [getattr(population, name)]
                )
            )
        ]
        if population_requires_temporal_state(population):
            features.append(
                "stateful dynamics (spiking soma, synapse_types, dendritic "
                "spikes or soma feedback)"
            )
        if features:
            raise ValueError(
                f"unified_ei layer {layer_index} is not recurrent, but its {role} "
                f"population sets {', '.join(features)}, which need recurrent "
                "state; set recurrent: true on the layer"
            )


class EILayer(nn.Module):
    """Unified E-I layer with optional recurrence."""

    def __init__(self, config: EILayerConfig, layer_index: int = 0):
        super().__init__()
        for population_name in ("excitatory", "inhibitory"):
            population = getattr(config, population_name, None)
            if (
                population is not None
                and population.initialization_seed is not None
                and not population.initialization_namespace
            ):
                population.initialization_namespace = (
                    f"ei.layer.{population.structured_layer_idx}.{population_name}"
                )
        self.config = config
        self.recurrent = config.recurrent
        if not self.recurrent:
            _reject_recurrent_populations(config, layer_index)
        self._last_routing_info: dict = {}

        self.n_exc = config.excitatory.n_neurons
        self.n_inh = config.inhibitory.n_neurons if config.inhibitory is not None else 0
        self.direct_ff_inhibitory_to_excitatory = bool(
            config.direct_ff_inhibitory_to_excitatory
        )

        rec_e_dim = self.n_exc if self.recurrent else None
        rec_i_dim = self.n_inh if self.recurrent and self.n_inh > 0 else None

        self.i_population: Optional[StatefulDendriNet]
        if config.inhibitory is not None and self.n_inh > 0:
            self.i_population = StatefulDendriNet(
                pop_config=_layer_population_config(
                    config.inhibitory, layer_index, "inhibitory"
                ),
                excitatory_input_dim=config.excitatory_input_dim,
                inhibitory_input_dim=config.inhibitory_input_dim,
                recurrent_excitatory_input_dim=rec_e_dim,
                recurrent_inhibitory_input_dim=rec_i_dim,
                recurrent_excitatory_is_self_population=False,
                recurrent_inhibitory_is_self_population=True,
                dt=config.dt,
            )
        else:
            self.i_population = None

        # E receives FF inhibitory from same-layer I population (if present).
        e_ff_inhibitory_dim = (
            self.n_inh
            if self.n_inh > 0
            else (
                config.inhibitory_input_dim
                if self.direct_ff_inhibitory_to_excitatory
                else None
            )
        )
        self.e_population = StatefulDendriNet(
            pop_config=_layer_population_config(
                config.excitatory, layer_index, "excitatory"
            ),
            excitatory_input_dim=config.excitatory_input_dim,
            inhibitory_input_dim=e_ff_inhibitory_dim,
            recurrent_excitatory_input_dim=rec_e_dim,
            recurrent_inhibitory_input_dim=rec_i_dim,
            recurrent_excitatory_is_self_population=True,
            recurrent_inhibitory_is_self_population=False,
            dt=config.dt,
        )
        if self.recurrent:  # forward() steps both populations
            for population in (self.i_population, self.e_population):
                if population is not None:
                    population.prepare_stepped_gradient_scaling()

        self.output_dim = self.n_exc

    # Tensors of the last forward pass kept for analyses; copies and pickles
    # hold them detached (networks.utils.forward_caches).
    _FORWARD_CACHES = ("_last_routing_info",)

    def __getstate__(self):
        return detach_forward_caches(super().__getstate__(), self._FORWARD_CACHES)

    @property
    def store_routing(self) -> bool:
        return self.e_population.store_routing

    @store_routing.setter
    def store_routing(self, value: bool) -> None:
        self.e_population.store_routing = value
        if self.i_population is not None:
            self.i_population.store_routing = value

    def decay_weights(self, weight_decay, weight_boosting=False):
        if self.i_population is not None:
            self.i_population.decay_weights(weight_decay, weight_boosting)
        self.e_population.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self):
        if self.i_population is not None:
            self.i_population.apply_rewiring()
        self.e_population.apply_rewiring()

    def set_population_level_silencing(
        self,
        population: str,
        mask: Optional[torch.Tensor],
    ) -> None:
        if population == "excitatory":
            self.e_population.set_level_silencing(mask)
            return
        if population == "inhibitory":
            if self.i_population is None:
                raise ValueError(
                    "Cannot silence inhibitory levels: no inhibitory population"
                )
            self.i_population.set_level_silencing(mask)
            return
        if population == "both":
            self.e_population.set_level_silencing(mask)
            if self.i_population is not None:
                self.i_population.set_level_silencing(mask)
            return
        raise ValueError(f"Unknown population '{population}'")

    def clear_population_level_silencing(self) -> None:
        self.e_population.clear_level_silencing()
        if self.i_population is not None:
            self.i_population.clear_level_silencing()

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> EIState:
        e_state = self.e_population.init_state(
            batch_size=batch_size, device=device, dtype=dtype
        )
        i_state = (
            self.i_population.init_state(
                batch_size=batch_size, device=device, dtype=dtype
            )
            if self.i_population is not None
            else None
        )
        s_E = torch.zeros(batch_size, self.n_exc, device=device, dtype=dtype)
        h_I = (
            torch.zeros(batch_size, self.n_inh, device=device, dtype=dtype)
            if self.n_inh > 0
            else None
        )
        return EIState(e_state=e_state, i_state=i_state, s_E=s_E, h_I=h_I)

    def forward(
        self,
        x_ff: torch.Tensor,
        ff_inhibitory: Optional[torch.Tensor] = None,
        state: Optional[EIState] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[EIState]]:
        """Forward pass through the E-I layer.

        Args:
            x_ff: FF excitatory input from upstream [batch, excitatory_input_dim].
            ff_inhibitory: FF inhibitory input from upstream (for I pop) [batch, inhibitory_input_dim].
            state: Recurrent state (only used when recurrent=True).

        Returns:
            s_E: Excitatory output [batch, n_exc].
            h_I: Inhibitory output [batch, n_inh] or None.
            new_state: Updated EIState (None when recurrent=False).
        """
        if not self.recurrent:
            h_I = (
                self.i_population(x_ff, ff_inhibitory)
                if self.i_population is not None
                else None
            )
            e_ff_inhibitory = (
                h_I
                if h_I is not None
                else (
                    ff_inhibitory if self.direct_ff_inhibitory_to_excitatory else None
                )
            )
            s_E = self.e_population(x_ff, e_ff_inhibitory)
            return s_E, h_I, None

        batch_size = x_ff.shape[0]
        device = x_ff.device
        dtype = x_ff.dtype

        if state is None:
            state = self.init_state(batch_size=batch_size, device=device, dtype=dtype)

        s_E_prev = state.s_E
        h_I_prev = state.h_I

        # Update I first (faster dynamics).
        if self.i_population is not None:
            h_I, i_state = self.i_population.step(
                x=x_ff,
                inhibitory_input=ff_inhibitory,
                recurrent_input=s_E_prev,
                rec_inhibitory_input=h_I_prev,
                state=state.i_state,
            )
        else:
            h_I, i_state = None, None

        # E uses current same-layer I as FF inhibitory and previous I for REC inhibitory.
        e_ff_inhibitory = (
            h_I
            if h_I is not None
            else (ff_inhibitory if self.direct_ff_inhibitory_to_excitatory else None)
        )
        s_E, e_state = self.e_population.step(
            x=x_ff,
            inhibitory_input=e_ff_inhibitory,
            recurrent_input=s_E_prev,
            rec_inhibitory_input=h_I_prev,
            state=state.e_state,
        )

        new_state = EIState(e_state=e_state, i_state=i_state, s_E=s_E, h_I=h_I)

        if self.e_population.store_routing:
            self._last_routing_info = {
                "excitatory": self.e_population._last_routing_info,
            }
            if self.i_population is not None:
                self._last_routing_info["inhibitory"] = (
                    self.i_population._last_routing_info
                )

        return s_E, h_I, new_state
