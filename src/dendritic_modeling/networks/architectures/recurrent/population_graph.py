"""Backward-compatible population-graph API.

``population_network`` is the canonical backend for named populations.  This
module keeps the older single-layer ``population_graph`` config/class working
by lowering it into a one-layer recurrent ``PopulationNetwork``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState
from dendritic_modeling.networks.architectures.recurrent.population_common import (
    CellTypeConfig,
    as_cell_type_config,
    population_config_from_cell,
)
from dendritic_modeling.networks.architectures.recurrent.population_network import (
    PopulationLayerConfig,
    PopulationLayerState,
    PopulationNetwork,
    PopulationNetworkConfig,
    PopulationProjectionConfig,
)
from dendritic_modeling.networks.base import BaseNetwork


@dataclass
class PopulationConnectionConfig(BaseConfig):
    """Optional source-target recurrent connection rule between cell types."""

    source: str = ""
    target: str = ""
    enabled: bool = True
    probability: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.source = str(self.source)
        self.target = str(self.target)
        if not self.source:
            raise ValueError("connection source must be non-empty")
        if not self.target:
            raise ValueError("connection target must be non-empty")
        if self.probability is not None:
            self.probability = float(self.probability)
            if self.probability < 0.0 or self.probability > 1.0:
                raise ValueError(
                    f"connection probability must be in [0, 1], got {self.probability}"
                )


@dataclass
class PopulationGraphConfig(BaseConfig):
    """Configuration for a legacy named population-graph recurrent core."""

    cell_types: list[CellTypeConfig | dict[str, Any]] = field(default_factory=list)
    connections: list[PopulationConnectionConfig | dict[str, Any]] = field(
        default_factory=list
    )
    input_dim: int = 64
    readout_population: str | None = None
    output_mode: str = "last"  # last | mean | all
    dt: float = 1.0
    input_projection_dims: list[int] = field(default_factory=list)
    population_defaults: dict[str, Any] = field(default_factory=dict)
    connection_seed: int | None = None
    store_routing: bool = False

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if self.output_mode not in {"last", "mean", "all"}:
            raise ValueError(
                f"output_mode must be one of ('last', 'mean', 'all'), got {self.output_mode!r}"
            )
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        for hidden_dim in self.input_projection_dims:
            if int(hidden_dim) <= 0:
                raise ValueError(
                    "input_projection_dims entries must be positive, "
                    f"got {hidden_dim}"
                )


@dataclass
class PopulationGraphState:
    """Runtime state for a legacy population graph layer."""

    population_states: dict[str, DendriNetState]
    outputs: dict[str, torch.Tensor]

    def detach(self) -> PopulationGraphState:
        return PopulationGraphState(
            population_states={
                name: state.detach() for name, state in self.population_states.items()
            },
            outputs={name: output.detach() for name, output in self.outputs.items()},
        )

    def clone(self) -> PopulationGraphState:
        return PopulationGraphState(
            population_states={
                name: state.clone() for name, state in self.population_states.items()
            },
            outputs={name: output.clone() for name, output in self.outputs.items()},
        )

    def to(self, device: torch.device) -> PopulationGraphState:
        return PopulationGraphState(
            population_states={
                name: state.to(device) for name, state in self.population_states.items()
            },
            outputs={name: output.to(device) for name, output in self.outputs.items()},
        )


def _as_connection_config(
    connection: PopulationConnectionConfig | dict[str, Any],
) -> PopulationConnectionConfig:
    if isinstance(connection, PopulationConnectionConfig):
        return connection
    if isinstance(connection, dict):
        payload = dict(connection)
        if "probability" not in payload and "p_connect" in payload:
            payload["probability"] = payload.pop("p_connect")
        return PopulationConnectionConfig(**payload)
    raise TypeError(
        "connections entries must be PopulationConnectionConfig or dict, "
        f"got {type(connection).__name__}"
    )


def _has_positive_synapses(
    pop_config: PopulationConfig,
    scalar_field: str,
    per_level_field: str,
) -> bool:
    if int(getattr(pop_config, scalar_field, 0)) > 0:
        return True
    return any(int(value) > 0 for value in getattr(pop_config, per_level_field, []))


def _probability_for_population_network(
    spec: PopulationConnectionConfig | None,
) -> float | None:
    if spec is None:
        return None
    if not spec.enabled:
        return 0.0
    return spec.probability


def _projection_from_graph_spec(
    *,
    source: str,
    target: str,
    pathway: str,
    spec: PopulationConnectionConfig | None,
) -> PopulationProjectionConfig:
    return PopulationProjectionConfig(
        source=source,
        target=target,
        pathway=pathway,
        timing="delayed",
        enabled=True,
        probability=_probability_for_population_network(spec),
        seed=None if spec is None else spec.seed,
    )


def _lower_population_graph_config(
    config: PopulationGraphConfig,
    cell_types: list[CellTypeConfig],
    connection_specs: list[PopulationConnectionConfig],
    readout_population: str,
) -> PopulationNetworkConfig:
    """Translate the legacy graph into the canonical population-network schema."""

    connection_by_pair: dict[tuple[str, str], PopulationConnectionConfig] = {
        (spec.source, spec.target): spec for spec in connection_specs
    }
    excitatory_names = [
        cell.name for cell in cell_types if cell.polarity == "excitatory"
    ]
    inhibitory_names = [
        cell.name for cell in cell_types if cell.polarity == "inhibitory"
    ]

    defaults = dict(config.population_defaults)
    projections: list[PopulationProjectionConfig] = [
        PopulationProjectionConfig(source="input", target=cell.name)
        for cell in cell_types
    ]

    for idx, target_cell in enumerate(cell_types):
        pop_cfg = population_config_from_cell(
            target_cell,
            defaults,
            layer_idx=idx,
        )
        if _has_positive_synapses(
            pop_cfg,
            "rec_excitatory_synapses",
            "rec_excitatory_synapses_by_level",
        ):
            projections.extend(
                _projection_from_graph_spec(
                    source=source,
                    target=target_cell.name,
                    pathway="rec_excitatory",
                    spec=connection_by_pair.get((source, target_cell.name)),
                )
                for source in excitatory_names
            )
        if _has_positive_synapses(
            pop_cfg,
            "rec_inhibitory_synapses",
            "rec_inhibitory_synapses_by_level",
        ):
            projections.extend(
                _projection_from_graph_spec(
                    source=source,
                    target=target_cell.name,
                    pathway="rec_inhibitory",
                    spec=connection_by_pair.get((source, target_cell.name)),
                )
                for source in inhibitory_names
            )

    return PopulationNetworkConfig(
        input_dim=config.input_dim,
        input_projection_dims=list(config.input_projection_dims),
        readout_layer="population_graph",
        readout_population=readout_population,
        output_mode=config.output_mode,
        store_routing=config.store_routing,
        layers=[
            PopulationLayerConfig(
                name="population_graph",
                populations=cell_types,
                connections=projections,
                population_defaults=defaults,
                readout_population=readout_population,
                recurrent=any(proj.timing == "delayed" for proj in projections),
                dt=config.dt,
                connection_seed=config.connection_seed,
                store_routing=config.store_routing,
            )
        ],
    )


class PopulationGraphNetwork(BaseNetwork):
    """Legacy single-layer named-cell core backed by ``PopulationNetwork``."""

    def __init__(self, config: PopulationGraphConfig):
        super().__init__()
        self.config = config
        self.output_mode = config.output_mode

        self.cell_types = [
            as_cell_type_config(cell, field_name="cell_types")
            for cell in config.cell_types
        ]
        if not self.cell_types:
            raise ValueError("PopulationGraphNetwork requires at least one cell type")
        names = [cell.name for cell in self.cell_types]
        if len(set(names)) != len(names):
            raise ValueError(f"cell type names must be unique, got {names}")

        self.excitatory_names = [
            cell.name for cell in self.cell_types if cell.polarity == "excitatory"
        ]
        self.inhibitory_names = [
            cell.name for cell in self.cell_types if cell.polarity == "inhibitory"
        ]
        if not self.excitatory_names:
            raise ValueError("PopulationGraphNetwork requires an excitatory population")

        self.readout_population = config.readout_population or self.excitatory_names[0]
        if self.readout_population not in names:
            raise ValueError(
                f"readout_population {self.readout_population!r} is not in cell_types"
            )

        self.connection_specs = [
            _as_connection_config(connection) for connection in config.connections
        ]
        self._connection_specs_by_pair: dict[
            tuple[str, str], PopulationConnectionConfig
        ] = {}
        known_names = set(names)
        for spec in self.connection_specs:
            if spec.source not in known_names:
                raise ValueError(
                    f"connection source {spec.source!r} is not a cell type"
                )
            if spec.target not in known_names:
                raise ValueError(
                    f"connection target {spec.target!r} is not a cell type"
                )
            key = (spec.source, spec.target)
            if key in self._connection_specs_by_pair:
                raise ValueError(
                    f"duplicate connection spec for {spec.source}->{spec.target}"
                )
            self._connection_specs_by_pair[key] = spec

        network_config = _lower_population_graph_config(
            config,
            self.cell_types,
            self.connection_specs,
            self.readout_population,
        )
        self._network = PopulationNetwork(network_config)
        self._layer = self._network.layers[0]

        self._cell_dims = dict(self._layer._cell_dims)
        self._exc_rec_dim = sum(self._cell_dims[name] for name in self.excitatory_names)
        self._inh_rec_dim = sum(self._cell_dims[name] for name in self.inhibitory_names)
        self.output_dim = self._network.output_dim

    @property
    def populations(self):
        return self._layer.populations

    @property
    def population_configs(self) -> dict[str, PopulationConfig]:
        return self._layer.population_configs

    @property
    def input_projection(self):
        return self._network.input_projection

    @property
    def proj_output_dim(self) -> int:
        return self._network.proj_output_dim

    @property
    def branch_layers(self) -> list[torch.nn.Module]:
        return self._network.branch_layers

    @property
    def is_recurrent(self) -> bool:
        return self._network.is_recurrent

    @property
    def store_routing(self) -> bool:
        return self._network.store_routing

    @store_routing.setter
    def store_routing(self, value: bool) -> None:
        self._network.store_routing = bool(value)

    @property
    def _routing_info(self) -> list[dict[str, dict]]:
        flattened = []
        for timestep in self._network._routing_info:
            if "population_graph" in timestep and isinstance(
                timestep["population_graph"], dict
            ):
                flattened.append(timestep["population_graph"])
            else:
                flattened.append(timestep)
        return flattened

    def decay_weights(self, weight_decay: float, weight_boosting: bool = False) -> None:
        self._network.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self) -> None:
        self._network.apply_rewiring()

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> PopulationGraphState:
        layer_state = self._layer.init_state(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return self._layer_state_to_graph_state(layer_state)

    def _graph_state_to_layer_state(
        self,
        state: PopulationGraphState | PopulationLayerState,
    ) -> PopulationLayerState:
        if isinstance(state, PopulationLayerState):
            return state
        return PopulationLayerState(
            population_states=state.population_states,
            outputs=state.outputs,
        )

    def _layer_state_to_graph_state(
        self,
        state: PopulationLayerState,
    ) -> PopulationGraphState:
        return PopulationGraphState(
            population_states=state.population_states,
            outputs=state.outputs,
        )

    def _reduce_outputs(
        self,
        outputs: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.output_mode == "last":
            if seq_lengths is not None:
                batch_size, seq_len, _ = outputs.shape
                idx = (seq_lengths - 1).long().clamp(0, seq_len - 1)
                return outputs[torch.arange(batch_size, device=outputs.device), idx]
            return outputs[:, -1, :]
        if self.output_mode == "mean":
            if seq_lengths is not None:
                _batch_size, seq_len, _ = outputs.shape
                lengths = seq_lengths.to(device=outputs.device).long().clamp(0, seq_len)
                mask = torch.arange(seq_len, device=outputs.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
                masked = outputs * mask.unsqueeze(-1).to(dtype=outputs.dtype)
                denom = lengths.clamp_min(1).unsqueeze(1).to(dtype=outputs.dtype)
                return masked.sum(dim=1) / denom
            return outputs.mean(dim=1)
        if self.output_mode == "all":
            if seq_lengths is not None:
                _batch_size, seq_len, _ = outputs.shape
                lengths = seq_lengths.to(device=outputs.device).long().clamp(0, seq_len)
                mask = torch.arange(seq_len, device=outputs.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
                return outputs * mask.unsqueeze(-1).to(dtype=outputs.dtype)
            return outputs
        raise ValueError(f"Unknown output_mode: {self.output_mode}")

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
        # Accept checkpoints saved by the previous direct PopulationGraphNetwork.
        remapped = {}
        old_pop_prefix = prefix + "populations."
        new_pop_prefix = prefix + "_network.layers.0.populations."
        old_proj_prefix = prefix + "input_projection."
        new_proj_prefix = prefix + "_network.input_projection."
        for key, value in list(state_dict.items()):
            if key.startswith(old_pop_prefix):
                remapped[new_pop_prefix + key[len(old_pop_prefix) :]] = value
            elif key.startswith(old_proj_prefix):
                remapped[new_proj_prefix + key[len(old_proj_prefix) :]] = value
        for key, value in remapped.items():
            state_dict.setdefault(key, value)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: PopulationGraphState | PopulationLayerState | None = None,
        return_hidden: bool = False,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, PopulationGraphState | None]:
        was_sequence = x.dim() == 3
        network_hidden = None
        if hidden is not None:
            network_hidden = [self._graph_state_to_layer_state(hidden)]

        out, next_hidden = self._network(
            x,
            hidden=network_hidden,
            return_hidden=True,
            seq_lengths=seq_lengths,
        )
        if not self.is_recurrent and was_sequence and out.dim() == 3:
            out = self._reduce_outputs(out, seq_lengths=seq_lengths)

        graph_hidden = None
        if next_hidden:
            graph_hidden = self._layer_state_to_graph_state(next_hidden[0])
        if return_hidden:
            return out, graph_hidden
        return out


__all__ = [
    "CellTypeConfig",
    "PopulationConnectionConfig",
    "PopulationGraphConfig",
    "PopulationGraphNetwork",
    "PopulationGraphState",
]
