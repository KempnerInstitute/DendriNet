"""Population-network feedforward/recurrent dendritic layers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.transfer import (
    TransferLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    sample_probability_mask,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState
from dendritic_modeling.networks.architectures.recurrent.population_common import (
    CellTypeConfig,
    as_cell_type_config as _as_cell_type_config,
    population_config_from_cell as _population_config_from_cell,
    population_level_dims as _population_level_dims,
    population_requires_temporal_state as _population_requires_temporal_state,
)
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)
from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.utils.stable_hash import stable_seed_offset

_INPUT_SOURCE = "input"
_INPUT_E_SOURCE = "input_e"
_INPUT_I_SOURCE = "input_i"
_EXTERNAL_INPUT_SOURCES = {_INPUT_SOURCE, _INPUT_E_SOURCE, _INPUT_I_SOURCE}
_EXTERNAL_INPUT_ALIASES = {
    "input": _INPUT_SOURCE,
    "input_e": _INPUT_E_SOURCE,
    "input_exc": _INPUT_E_SOURCE,
    "input_excitatory": _INPUT_E_SOURCE,
    "input_i": _INPUT_I_SOURCE,
    "input_inh": _INPUT_I_SOURCE,
    "input_inhibitory": _INPUT_I_SOURCE,
}
_SAME_STEP = "same_step"
_DELAYED = "delayed"
_FF_EXC = "ff_excitatory"
_FF_INH = "ff_inhibitory"
_REC_EXC = "rec_excitatory"
_REC_INH = "rec_inhibitory"


@dataclass
class PopulationProjectionConfig(BaseConfig):
    """Directed connection between one source and one target population."""

    source: str = _INPUT_SOURCE
    target: str = ""
    pathway: str = ""  # inferred from source polarity when omitted
    timing: str = _SAME_STEP  # same_step | delayed
    enabled: bool = True
    probability: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        self.source = str(self.source)
        self.target = str(self.target)
        self.pathway = str(self.pathway).lower()
        self.timing = str(self.timing).lower()
        if not self.source:
            raise ValueError("connection source must be non-empty")
        if not self.target:
            raise ValueError("connection target must be non-empty")
        if self.timing not in {_SAME_STEP, _DELAYED}:
            raise ValueError(
                "connection timing must be 'same_step' or 'delayed', "
                f"got {self.timing!r}"
            )
        if self.probability is not None:
            self.probability = float(self.probability)
            if self.probability < 0.0 or self.probability > 1.0:
                raise ValueError(
                    f"connection probability must be in [0, 1], got {self.probability}"
                )


@dataclass
class PopulationLayerConfig(BaseConfig):
    """One layer containing multiple named dendritic populations."""

    name: str = "layer0"
    populations: list[CellTypeConfig | dict[str, Any]] = field(default_factory=list)
    connections: list[PopulationProjectionConfig | dict[str, Any]] = field(
        default_factory=list
    )
    population_defaults: dict[str, Any] = field(default_factory=dict)
    readout_population: str | None = None
    recurrent: bool = False
    dt: float = 1.0
    connection_seed: int | None = None
    store_routing: bool = False

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise ValueError("layer name must be non-empty")
        if not self.populations:
            raise ValueError("population-network layer requires populations")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")


@dataclass
class PopulationNetworkConfig(BaseConfig):
    """Stack of population-network dendritic layers."""

    layers: list[PopulationLayerConfig | dict[str, Any]] = field(default_factory=list)
    input_dim: int = 64
    use_transfer: bool = False
    transfer_params: dict[str, Any] = field(default_factory=dict)
    input_projection_dims: list[int] = field(default_factory=list)
    readout_layer: str | None = None
    readout_population: str | None = None
    output_mode: str = "last"  # recurrent only: last | mean | all
    store_routing: bool = False

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be > 0, got {self.input_dim}")
        if not self.layers:
            raise ValueError("population network requires at least one layer")
        for hidden_dim in self.input_projection_dims:
            if int(hidden_dim) <= 0:
                raise ValueError(
                    "input_projection_dims entries must be positive, "
                    f"got {hidden_dim}"
                )
        if self.output_mode not in {"last", "mean", "all"}:
            raise ValueError(
                f"output_mode must be one of ('last', 'mean', 'all'), got {self.output_mode!r}"
            )


@dataclass
class PopulationLayerState:
    """Runtime state for one population-network layer."""

    population_states: dict[str, DendriNetState]
    outputs: dict[str, torch.Tensor]

    def detach(self) -> PopulationLayerState:
        return PopulationLayerState(
            population_states={
                name: state.detach() for name, state in self.population_states.items()
            },
            outputs={name: output.detach() for name, output in self.outputs.items()},
        )

    def clone(self) -> PopulationLayerState:
        return PopulationLayerState(
            population_states={
                name: state.clone() for name, state in self.population_states.items()
            },
            outputs={name: output.clone() for name, output in self.outputs.items()},
        )

    def to(self, device: torch.device) -> PopulationLayerState:
        return PopulationLayerState(
            population_states={
                name: state.to(device) for name, state in self.population_states.items()
            },
            outputs={name: output.to(device) for name, output in self.outputs.items()},
        )


def _as_projection_config(
    connection: PopulationProjectionConfig | dict[str, Any],
) -> PopulationProjectionConfig:
    if isinstance(connection, PopulationProjectionConfig):
        return connection
    if isinstance(connection, dict):
        payload = dict(connection)
        if "probability" not in payload and "p_connect" in payload:
            payload["probability"] = payload.pop("p_connect")
        return PopulationProjectionConfig(**payload)
    raise TypeError(
        "connections entries must be PopulationProjectionConfig or dict, "
        f"got {type(connection).__name__}"
    )


def _concat_or_none(values: list[torch.Tensor]) -> torch.Tensor | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return torch.cat(values, dim=-1)


def _canonical_external_source(source: str) -> str | None:
    return _EXTERNAL_INPUT_ALIASES.get(str(source).lower())


def _split_qualified_source(source: str) -> tuple[str, str] | None:
    if "." not in source:
        return None
    layer_name, population_name = source.split(".", 1)
    if not layer_name or not population_name:
        raise ValueError(
            "qualified population sources must use 'layer.population', "
            f"got {source!r}"
        )
    return layer_name, population_name


class PopulationLayer(nn.Module):
    """A directional population-network layer.

    Same-step population-to-population edges must form a DAG. Delayed edges use
    previous-timestep outputs and may be cyclic.
    """

    def __init__(
        self,
        config: PopulationLayerConfig,
        input_dim: int | dict[str, int],
        external_source_polarities: dict[str, str] | None = None,
    ):
        super().__init__()
        self.config = config
        self.external_input_dims = self._normalize_external_input_dims(input_dim)
        self.external_source_polarities = self._normalize_external_source_polarities(
            external_source_polarities or {}
        )
        self.input_dim = self.external_input_dims[_INPUT_SOURCE]
        self.recurrent = bool(config.recurrent)
        self._store_routing = bool(config.store_routing)
        self._last_routing_info: dict[str, dict] = {}
        self._last_outputs: dict[str, torch.Tensor] = {}

        self.cell_types = [_as_cell_type_config(cell) for cell in config.populations]
        names = [cell.name for cell in self.cell_types]
        if len(set(names)) != len(names):
            raise ValueError(f"population names must be unique per layer, got {names}")
        self._cell_by_name = {cell.name: cell for cell in self.cell_types}
        self._cell_dims = {cell.name: int(cell.n_neurons) for cell in self.cell_types}
        self.excitatory_names = [
            cell.name for cell in self.cell_types if cell.polarity == "excitatory"
        ]
        self.inhibitory_names = [
            cell.name for cell in self.cell_types if cell.polarity == "inhibitory"
        ]
        if not self.excitatory_names:
            raise ValueError(
                "population-network layer requires an excitatory population"
            )

        self.readout_population = config.readout_population or self.excitatory_names[0]
        if self.readout_population not in self._cell_by_name:
            raise ValueError(
                f"readout_population {self.readout_population!r} is not a population"
            )

        raw_connections = [_as_projection_config(conn) for conn in config.connections]
        self.connections = [
            self._normalize_connection(conn) for conn in raw_connections
        ]
        self._validate_connections()
        self._same_step_order = self._topological_same_step_order()

        self._sources_by_target_pathway: dict[tuple[str, str, str], list[str]] = {}
        self._connection_by_target_pathway_source: dict[
            tuple[str, str, str, str], PopulationProjectionConfig
        ] = {}
        for conn in self.connections:
            if not conn.enabled:
                continue
            key = (conn.target, conn.timing, conn.pathway)
            self._sources_by_target_pathway.setdefault(key, []).append(conn.source)
            self._connection_by_target_pathway_source[
                (conn.target, conn.timing, conn.pathway, conn.source)
            ] = conn
        for cell in self.cell_types:
            if not self._stream_sources(cell.name, _SAME_STEP, _FF_EXC):
                raise ValueError(
                    f"population {cell.name!r} has no same-step ff_excitatory input"
                )

        self.population_configs: dict[str, PopulationConfig] = {}
        self.populations = nn.ModuleDict()
        self._has_temporal_population_state = False
        defaults = dict(config.population_defaults)
        for idx, cell in enumerate(self.cell_types):
            pop_cfg = _population_config_from_cell(cell, defaults, layer_idx=idx)
            self.population_configs[cell.name] = pop_cfg
            self._has_temporal_population_state = (
                self._has_temporal_population_state
                or _population_requires_temporal_state(pop_cfg)
            )
            connection_masks = self._build_connection_masks_for_cell(cell, pop_cfg)
            self.populations[cell.name] = StatefulDendriNet(
                pop_config=pop_cfg,
                excitatory_input_dim=self._stream_dim(cell.name, _SAME_STEP, _FF_EXC),
                inhibitory_input_dim=self._stream_dim(cell.name, _SAME_STEP, _FF_INH),
                recurrent_excitatory_input_dim=self._stream_dim(
                    cell.name, _DELAYED, _REC_EXC
                ),
                recurrent_inhibitory_input_dim=self._stream_dim(
                    cell.name, _DELAYED, _REC_INH
                ),
                recurrent_excitatory_is_self_population=False,
                recurrent_inhibitory_is_self_population=False,
                connection_masks_by_pathway=connection_masks,
                dt=config.dt,
            )

        self.output_dim = self._cell_dims[self.readout_population]
        self._is_recurrent = (
            self.recurrent
            or self._has_temporal_population_state
            or any(conn.timing == _DELAYED for conn in self.connections if conn.enabled)
        )

    @property
    def is_recurrent(self) -> bool:
        return self._is_recurrent

    @property
    def store_routing(self) -> bool:
        return self._store_routing

    @store_routing.setter
    def store_routing(self, value: bool) -> None:
        self._store_routing = bool(value)
        for population in self.populations.values():
            population.store_routing = self._store_routing

    def _normalize_connection(
        self,
        conn: PopulationProjectionConfig,
    ) -> PopulationProjectionConfig:
        source = _canonical_external_source(conn.source) or conn.source
        source_is_external = source in self.external_input_dims
        if not source_is_external and source not in self._cell_by_name:
            raise ValueError(f"connection source {source!r} is not a population")
        if conn.target not in self._cell_by_name:
            raise ValueError(f"connection target {conn.target!r} is not a population")
        if source in _EXTERNAL_INPUT_SOURCES and conn.timing == _DELAYED:
            raise ValueError(
                f"source={source!r} cannot use timing='delayed'; delayed edges "
                "must come from a population output"
            )

        pathway = conn.pathway
        if not pathway:
            source_polarity = self._source_polarity(source)
            if conn.timing == _SAME_STEP:
                pathway = _FF_INH if source_polarity == "inhibitory" else _FF_EXC
            else:
                pathway = _REC_INH if source_polarity == "inhibitory" else _REC_EXC
        if conn.timing == _SAME_STEP and pathway not in {_FF_EXC, _FF_INH}:
            raise ValueError(
                "same_step connections must use pathway 'ff_excitatory' "
                f"or 'ff_inhibitory', got {pathway!r}"
            )
        if conn.timing == _DELAYED and pathway not in {_REC_EXC, _REC_INH}:
            raise ValueError(
                "delayed connections must use pathway 'rec_excitatory' "
                f"or 'rec_inhibitory', got {pathway!r}"
            )

        payload = dict(conn.__dict__)
        payload["source"] = source
        payload["pathway"] = pathway
        return PopulationProjectionConfig(**payload)

    def _validate_connections(self) -> None:
        seen = set()
        for conn in self.connections:
            key = (conn.source, conn.target, conn.timing, conn.pathway)
            if key in seen:
                raise ValueError(
                    "duplicate population-network connection "
                    f"{conn.source}->{conn.target} ({conn.timing}, {conn.pathway})"
                )
            seen.add(key)
            if conn.timing == _DELAYED and not self.recurrent:
                raise ValueError(
                    "delayed population connections require layer.recurrent=true"
                )

    def _topological_same_step_order(self) -> list[str]:
        incoming = dict.fromkeys(self._cell_by_name, 0)
        outgoing = {name: [] for name in self._cell_by_name}
        for conn in self.connections:
            if not conn.enabled or conn.timing != _SAME_STEP:
                continue
            if conn.source in self.external_input_dims:
                continue
            outgoing[conn.source].append(conn.target)
            incoming[conn.target] += 1

        queue = [name for name in self._cell_by_name if incoming[name] == 0]
        order = []
        while queue:
            name = queue.pop(0)
            order.append(name)
            for target in outgoing[name]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    queue.append(target)

        if len(order) != len(self._cell_by_name):
            cyclic = [name for name, degree in incoming.items() if degree > 0]
            raise ValueError(
                "Same-step within-layer population connections must be acyclic; "
                f"cycle involves {cyclic}"
            )
        return order

    def _source_dim(self, source: str) -> int:
        if source in self.external_input_dims:
            return self.external_input_dims[source]
        return self._cell_dims[source]

    def _source_polarity(self, source: str) -> str:
        if source in self.external_input_dims:
            return self.external_source_polarities.get(source, "excitatory")
        return self._cell_by_name[source].polarity

    def _normalize_external_input_dims(
        self,
        input_dim: int | dict[str, int],
    ) -> dict[str, int]:
        if isinstance(input_dim, dict):
            dims: dict[str, int] = {}
            for source, dim in input_dim.items():
                canonical = _canonical_external_source(source) or str(source)
                if not canonical:
                    raise ValueError("external input source names must be non-empty")
                dims[canonical] = int(dim)
            if _INPUT_SOURCE not in dims:
                raise ValueError("external input dimensions must include 'input'")
        else:
            dims = {_INPUT_SOURCE: int(input_dim)}

        for source in _EXTERNAL_INPUT_SOURCES:
            dims.setdefault(source, dims[_INPUT_SOURCE])
        for source, dim in dims.items():
            if int(dim) <= 0:
                raise ValueError(
                    f"external input dimension for {source!r} must be > 0, got {dim}"
                )
            dims[source] = int(dim)
        return dims

    def _normalize_external_source_polarities(
        self,
        external_source_polarities: dict[str, str],
    ) -> dict[str, str]:
        polarities = {
            _INPUT_SOURCE: "excitatory",
            _INPUT_E_SOURCE: "excitatory",
            _INPUT_I_SOURCE: "inhibitory",
        }
        for source, polarity in external_source_polarities.items():
            canonical = _canonical_external_source(source) or str(source)
            polarity = str(polarity).lower()
            if polarity not in {"excitatory", "inhibitory"}:
                raise ValueError(
                    "external source polarity must be 'excitatory' or "
                    f"'inhibitory', got {polarity!r} for {canonical!r}"
                )
            polarities[canonical] = polarity
        for source in self.external_input_dims:
            polarities.setdefault(source, "excitatory")
        return polarities

    def _normalize_external_inputs(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if isinstance(x, dict):
            external_inputs: dict[str, torch.Tensor] = {}
            for source, value in x.items():
                canonical = _canonical_external_source(source) or str(source)
                if canonical not in self.external_input_dims:
                    raise ValueError(
                        "external input tensors may only be provided for configured "
                        f"sources {sorted(self.external_input_dims)}, got {source!r}"
                    )
                external_inputs[canonical] = value
            if _INPUT_SOURCE in external_inputs:
                for source in _EXTERNAL_INPUT_SOURCES:
                    external_inputs.setdefault(source, external_inputs[_INPUT_SOURCE])
        else:
            external_inputs = dict.fromkeys(_EXTERNAL_INPUT_SOURCES, x)

        reference_shape = None
        for source, value in external_inputs.items():
            expected = self.external_input_dims[source]
            if value.shape[-1] != expected:
                raise ValueError(
                    "PopulationLayer expected external source "
                    f"{source!r} to have last dimension {expected}, "
                    f"got {value.shape[-1]}"
                )
            prefix_shape = tuple(value.shape[:-1])
            if reference_shape is None:
                reference_shape = prefix_shape
            elif prefix_shape != reference_shape:
                raise ValueError(
                    "PopulationLayer external sources must share batch/time "
                    f"dimensions, got {reference_shape} and {prefix_shape}"
                )
        return external_inputs

    def _stream_sources(self, target: str, timing: str, pathway: str) -> list[str]:
        return self._sources_by_target_pathway.get((target, timing, pathway), [])

    def _stream_dim(self, target: str, timing: str, pathway: str) -> int | None:
        sources = self._stream_sources(target, timing, pathway)
        if not sources:
            return None
        return sum(self._source_dim(source) for source in sources)

    def _level_dims(self, cell: CellTypeConfig) -> list[int]:
        return _population_level_dims(cell.n_neurons, cell.branch_factors)

    def _build_connection_masks_for_cell(
        self,
        cell: CellTypeConfig,
        pop_cfg: PopulationConfig,
    ) -> dict[str, list[torch.Tensor]]:
        masks_by_pathway = {}
        for timing, pathway, pop_pathway in (
            (_SAME_STEP, _FF_EXC, pop_cfg.ff_excitatory_pathway),
            (_SAME_STEP, _FF_INH, pop_cfg.ff_inhibitory_pathway),
            (_DELAYED, _REC_EXC, pop_cfg.rec_excitatory_pathway),
            (_DELAYED, _REC_INH, pop_cfg.rec_inhibitory_pathway),
        ):
            masks = self._build_masks_for_stream(
                target=cell.name,
                timing=timing,
                pathway=pathway,
                level_dims=self._level_dims(cell),
            )
            if masks is not None:
                masks_by_pathway[pop_pathway] = masks
        return masks_by_pathway

    def _build_masks_for_stream(
        self,
        *,
        target: str,
        timing: str,
        pathway: str,
        level_dims: list[int],
    ) -> list[torch.Tensor] | None:
        sources = self._stream_sources(target, timing, pathway)
        if not sources:
            return None
        masks = [
            self._build_mask_level(
                target=target,
                timing=timing,
                pathway=pathway,
                sources=sources,
                out_dim=out_dim,
                level_idx=level_idx,
            )
            for level_idx, out_dim in enumerate(level_dims)
        ]
        if all(bool(mask.all()) for mask in masks):
            return None
        return masks

    def _build_mask_level(
        self,
        *,
        target: str,
        timing: str,
        pathway: str,
        sources: list[str],
        out_dim: int,
        level_idx: int,
    ) -> torch.Tensor:
        blocks = []
        for source in sources:
            conn = self._connection_by_target_pathway_source[
                (target, timing, pathway, source)
            ]
            probability = self._connection_probability(conn)
            source_dim = self._source_dim(source)
            blocks.append(
                self._sample_connection_block(
                    source=source,
                    target=target,
                    timing=timing,
                    pathway=pathway,
                    probability=probability,
                    out_dim=out_dim,
                    source_dim=source_dim,
                    level_idx=level_idx,
                    seed=conn.seed,
                )
            )
        return torch.cat(blocks, dim=1)

    def _connection_probability(self, conn: PopulationProjectionConfig) -> float:
        if not conn.enabled:
            return 0.0
        if conn.probability is None:
            return 1.0
        return float(conn.probability)

    def _sample_connection_block(
        self,
        *,
        source: str,
        target: str,
        timing: str,
        pathway: str,
        probability: float,
        out_dim: int,
        source_dim: int,
        level_idx: int,
        seed: int | None,
    ) -> torch.Tensor:
        if probability <= 0.0:
            return torch.zeros(out_dim, source_dim, dtype=torch.bool)
        if probability >= 1.0:
            return torch.ones(out_dim, source_dim, dtype=torch.bool)

        base_seed = seed if seed is not None else self.config.connection_seed
        if base_seed is None:
            base_seed = 0
        generator = torch.Generator()
        generator.manual_seed(
            (
                int(base_seed)
                + stable_seed_offset(
                    self.config.name,
                    source,
                    target,
                    timing,
                    pathway,
                    level_idx,
                    out_dim,
                    source_dim,
                )
            )
            % ((1 << 63) - 1)
        )
        return sample_probability_mask(
            out_features=out_dim,
            in_features=source_dim,
            probability=float(probability),
            generator=generator,
        )

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> PopulationLayerState:
        population_states = {
            name: population.init_state(
                batch_size=batch_size, device=device, dtype=dtype
            )
            for name, population in self.populations.items()
        }
        outputs = {
            name: torch.zeros(
                batch_size,
                self._cell_dims[name],
                device=device,
                dtype=dtype,
            )
            for name in self.populations
        }
        return PopulationLayerState(
            population_states=population_states,
            outputs=outputs,
        )

    def _collect_stream(
        self,
        *,
        target: str,
        timing: str,
        pathway: str,
        external_inputs: dict[str, torch.Tensor],
        delayed_external_inputs: dict[str, torch.Tensor] | None,
        current_outputs: dict[str, torch.Tensor],
        previous_outputs: dict[str, torch.Tensor],
    ) -> torch.Tensor | None:
        values = []
        source_outputs = current_outputs if timing == _SAME_STEP else previous_outputs
        for source in self._stream_sources(target, timing, pathway):
            if source in self.external_input_dims:
                source_inputs = (
                    delayed_external_inputs if timing == _DELAYED else external_inputs
                )
                if source_inputs is None:
                    source_inputs = {}
                if source not in source_inputs:
                    raise ValueError(
                        f"missing external input source {source!r} for "
                        f"population {target!r}"
                    )
                values.append(source_inputs[source])
            else:
                values.append(source_outputs[source])
        return _concat_or_none(values)

    def forward(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        state: PopulationLayerState | None = None,
        delayed_external_inputs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, PopulationLayerState | None]:
        if self.is_recurrent:
            return self.step(
                x,
                state=state,
                delayed_external_inputs=delayed_external_inputs,
            )

        external_inputs = self._normalize_external_inputs(x)
        current_outputs: dict[str, torch.Tensor] = {}
        previous_outputs: dict[str, torch.Tensor] = {}
        for name in self._same_step_order:
            population = self.populations[name]
            x_ff = self._collect_stream(
                target=name,
                timing=_SAME_STEP,
                pathway=_FF_EXC,
                external_inputs=external_inputs,
                delayed_external_inputs=None,
                current_outputs=current_outputs,
                previous_outputs=previous_outputs,
            )
            ff_inh = self._collect_stream(
                target=name,
                timing=_SAME_STEP,
                pathway=_FF_INH,
                external_inputs=external_inputs,
                delayed_external_inputs=None,
                current_outputs=current_outputs,
                previous_outputs=previous_outputs,
            )
            if x_ff is None:
                raise ValueError(
                    f"population {name!r} has no same-step ff_excitatory input"
                )
            current_outputs[name] = population(x_ff, ff_inh)

        self._last_outputs = current_outputs
        return current_outputs[self.readout_population], None

    def step(
        self,
        x: torch.Tensor | dict[str, torch.Tensor],
        state: PopulationLayerState | None = None,
        delayed_external_inputs: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, PopulationLayerState]:
        external_inputs = self._normalize_external_inputs(x)
        delayed_external_inputs = self._normalize_external_inputs(
            delayed_external_inputs or {}
        )
        reference_input = next(iter(external_inputs.values()))
        if state is None:
            state = self.init_state(
                batch_size=reference_input.shape[0],
                device=reference_input.device,
                dtype=reference_input.dtype,
            )

        current_outputs: dict[str, torch.Tensor] = {}
        new_population_states: dict[str, DendriNetState] = {}
        for name in self._same_step_order:
            population = self.populations[name]
            x_ff = self._collect_stream(
                target=name,
                timing=_SAME_STEP,
                pathway=_FF_EXC,
                external_inputs=external_inputs,
                delayed_external_inputs=delayed_external_inputs,
                current_outputs=current_outputs,
                previous_outputs=state.outputs,
            )
            ff_inh = self._collect_stream(
                target=name,
                timing=_SAME_STEP,
                pathway=_FF_INH,
                external_inputs=external_inputs,
                delayed_external_inputs=delayed_external_inputs,
                current_outputs=current_outputs,
                previous_outputs=state.outputs,
            )
            rec_exc = self._collect_stream(
                target=name,
                timing=_DELAYED,
                pathway=_REC_EXC,
                external_inputs=external_inputs,
                delayed_external_inputs=delayed_external_inputs,
                current_outputs=current_outputs,
                previous_outputs=state.outputs,
            )
            rec_inh = self._collect_stream(
                target=name,
                timing=_DELAYED,
                pathway=_REC_INH,
                external_inputs=external_inputs,
                delayed_external_inputs=delayed_external_inputs,
                current_outputs=current_outputs,
                previous_outputs=state.outputs,
            )
            if x_ff is None:
                raise ValueError(
                    f"population {name!r} has no same-step ff_excitatory input"
                )
            out, new_state = population.step(
                x=x_ff,
                inhibitory_input=ff_inh,
                recurrent_input=rec_exc,
                rec_inhibitory_input=rec_inh,
                state=state.population_states[name],
            )
            current_outputs[name] = out
            new_population_states[name] = new_state

        new_layer_state = PopulationLayerState(
            population_states=new_population_states,
            outputs=current_outputs,
        )
        self._last_outputs = current_outputs
        if self.store_routing:
            self._last_routing_info = {
                name: population._last_routing_info
                for name, population in self.populations.items()
            }
        return current_outputs[self.readout_population], new_layer_state


class PopulationNetwork(BaseNetwork):
    """Stack of population-network dendritic layers."""

    def __init__(self, config: PopulationNetworkConfig):
        super().__init__()
        self.config = config
        self.input_dim = int(config.input_dim)
        self.output_mode = config.output_mode
        self._store_routing = bool(config.store_routing)
        self._routing_info: list[dict[str, dict]] = []
        self._use_transfer = bool(config.use_transfer)
        self.transfer_fn: TransferLayer | None = None

        input_dim = self.input_dim
        if self._use_transfer:
            self.transfer_fn = TransferLayer(
                input_dim=config.input_dim,
                transfer_params=config.transfer_params,
            )
            input_dim = self.transfer_fn.excitatory_dim

        proj_layers = []
        prev_dim = input_dim
        for hidden_dim in config.input_projection_dims:
            proj_layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            proj_layers.append(nn.ReLU())
            prev_dim = int(hidden_dim)
        self.input_projection = (
            nn.Sequential(*proj_layers) if proj_layers else nn.Identity()
        )
        self.proj_output_dim = prev_dim

        self.layer_configs = [
            (
                layer
                if isinstance(layer, PopulationLayerConfig)
                else PopulationLayerConfig(**layer)
            )
            for layer in config.layers
        ]
        layer_names = [layer.name for layer in self.layer_configs]
        if len(set(layer_names)) != len(layer_names):
            raise ValueError(
                f"population-network layer names must be unique, got {layer_names}"
            )

        self.readout_layer = config.readout_layer or layer_names[-1]
        if self.readout_layer not in layer_names:
            raise ValueError(f"readout_layer {self.readout_layer!r} is not a layer")

        self._layer_index_by_name = {
            layer_name: idx for idx, layer_name in enumerate(layer_names)
        }
        self._population_dims_by_layer, self._population_polarities_by_layer = (
            self._build_population_metadata()
        )
        self._qualified_sources_by_layer_timing = self._resolve_qualified_sources()

        self.layers = nn.ModuleList()
        self.layer_names = layer_names
        current_external_dims = self._initial_external_input_dims()
        self._is_recurrent = self.output_mode in {"all", "mean"}
        for layer_idx, base_layer_cfg in enumerate(self.layer_configs):
            layer_cfg = base_layer_cfg
            updates: dict[str, Any] = {}
            if self._store_routing:
                updates["store_routing"] = True
            if (
                base_layer_cfg.name == self.readout_layer
                and config.readout_population is not None
            ):
                if (
                    base_layer_cfg.readout_population is not None
                    and base_layer_cfg.readout_population != config.readout_population
                ):
                    raise ValueError(
                        "network-level readout_population conflicts with the "
                        f"selected layer's readout_population "
                        f"({config.readout_population!r} vs "
                        f"{base_layer_cfg.readout_population!r})"
                    )
                updates["readout_population"] = config.readout_population
            if updates:
                layer_cfg = replace(base_layer_cfg, **updates)
            layer_external_dims = self._layer_external_dims(
                layer_idx,
                current_external_dims,
            )
            layer = PopulationLayer(
                layer_cfg,
                input_dim=layer_external_dims,
                external_source_polarities=self._layer_external_polarities(layer_idx),
            )
            self.layers.append(layer)
            current_external_dims = self._single_stream_external_dims(layer.output_dim)
            self._is_recurrent = self._is_recurrent or layer.is_recurrent

        readout_idx = self.layer_names.index(self.readout_layer)
        readout_layer = self.layers[readout_idx]
        self.output_dim = readout_layer.output_dim

    @property
    def is_recurrent(self) -> bool:
        return self._is_recurrent

    @property
    def store_routing(self) -> bool:
        return self._store_routing

    @store_routing.setter
    def store_routing(self, value: bool) -> None:
        self._store_routing = bool(value)
        for layer in self.layers:
            layer.store_routing = self._store_routing

    @property
    def branch_layers(self) -> list[nn.Module]:
        branch_layers: list[nn.Module] = []
        for layer in self.layers:
            for population in layer.populations.values():
                branch_layers.extend(list(getattr(population, "branch_layers", [])))
        return branch_layers

    def decay_weights(self, weight_decay: float, weight_boosting: bool = False) -> None:
        for layer in self.layers:
            for population in layer.populations.values():
                if hasattr(population, "decay_weights"):
                    population.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self) -> None:
        for layer in self.layers:
            for population in layer.populations.values():
                if hasattr(population, "apply_rewiring"):
                    population.apply_rewiring()

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> list[PopulationLayerState]:
        return [
            layer.init_state(batch_size=batch_size, device=device, dtype=dtype)
            for layer in self.layers
        ]

    def _build_population_metadata(
        self,
    ) -> tuple[dict[str, dict[str, int]], dict[str, dict[str, str]]]:
        dims_by_layer: dict[str, dict[str, int]] = {}
        polarities_by_layer: dict[str, dict[str, str]] = {}
        for layer_cfg in self.layer_configs:
            if "." in layer_cfg.name:
                raise ValueError(
                    "population-network layer names may not contain '.', "
                    f"got {layer_cfg.name!r}"
                )
            cells = [_as_cell_type_config(cell) for cell in layer_cfg.populations]
            dims: dict[str, int] = {}
            polarities: dict[str, str] = {}
            for cell in cells:
                if "." in cell.name:
                    raise ValueError(
                        "population names may not contain '.' when qualified "
                        f"sources are enabled, got {cell.name!r}"
                    )
                if cell.name in dims:
                    raise ValueError(
                        f"population names must be unique per layer, got {cell.name!r}"
                    )
                dims[cell.name] = int(cell.n_neurons)
                polarities[cell.name] = cell.polarity
            dims_by_layer[layer_cfg.name] = dims
            polarities_by_layer[layer_cfg.name] = polarities
        return dims_by_layer, polarities_by_layer

    def _resolve_qualified_sources(self) -> list[dict[str, set[str]]]:
        sources_by_layer: list[dict[str, set[str]]] = [
            {_SAME_STEP: set(), _DELAYED: set()} for _ in self.layer_configs
        ]
        for target_layer_idx, layer_cfg in enumerate(self.layer_configs):
            for raw_conn in layer_cfg.connections:
                conn = _as_projection_config(raw_conn)
                source = _canonical_external_source(conn.source) or conn.source
                qualified = _split_qualified_source(source)
                if qualified is None:
                    continue
                source_layer, source_population = qualified
                if source_layer not in self._layer_index_by_name:
                    raise ValueError(
                        f"qualified source {source!r} references unknown layer "
                        f"{source_layer!r}"
                    )
                if (
                    source_population
                    not in self._population_dims_by_layer[source_layer]
                ):
                    raise ValueError(
                        f"qualified source {source!r} references unknown "
                        f"population {source_population!r} in layer {source_layer!r}"
                    )
                source_layer_idx = self._layer_index_by_name[source_layer]
                if source_layer_idx == target_layer_idx:
                    raise ValueError(
                        f"qualified source {source!r} points to its own layer; "
                        "use the unqualified population name for same-layer edges"
                    )
                if conn.timing == _SAME_STEP and source_layer_idx > target_layer_idx:
                    raise ValueError(
                        "same_step qualified population sources must come from "
                        f"earlier layers, got {source!r} for target layer "
                        f"{layer_cfg.name!r}"
                    )
                sources_by_layer[target_layer_idx][conn.timing].add(source)
        return sources_by_layer

    def _qualified_source_dim(self, source: str) -> int:
        layer_name, population_name = _split_qualified_source(source) or ("", "")
        return self._population_dims_by_layer[layer_name][population_name]

    def _qualified_source_polarity(self, source: str) -> str:
        layer_name, population_name = _split_qualified_source(source) or ("", "")
        return self._population_polarities_by_layer[layer_name][population_name]

    def _layer_external_dims(
        self,
        layer_idx: int,
        base_dims: dict[str, int],
    ) -> dict[str, int]:
        dims = dict(base_dims)
        for timing in (_SAME_STEP, _DELAYED):
            for source in self._qualified_sources_by_layer_timing[layer_idx][timing]:
                dims[source] = self._qualified_source_dim(source)
        return dims

    def _layer_external_polarities(self, layer_idx: int) -> dict[str, str]:
        polarities = {
            _INPUT_SOURCE: "excitatory",
            _INPUT_E_SOURCE: "excitatory",
            _INPUT_I_SOURCE: "inhibitory",
        }
        for timing in (_SAME_STEP, _DELAYED):
            for source in self._qualified_sources_by_layer_timing[layer_idx][timing]:
                polarities[source] = self._qualified_source_polarity(source)
        return polarities

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

    def _single_stream_external_dims(self, dim: int) -> dict[str, int]:
        return {source: int(dim) for source in _EXTERNAL_INPUT_SOURCES}

    def _single_stream_external_inputs(
        self,
        x: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return dict.fromkeys(_EXTERNAL_INPUT_SOURCES, x)

    def _initial_external_input_dims(self) -> dict[str, int]:
        if self.transfer_fn is None:
            return self._single_stream_external_dims(self.proj_output_dim)
        return {
            _INPUT_SOURCE: self.proj_output_dim,
            _INPUT_E_SOURCE: self.proj_output_dim,
            _INPUT_I_SOURCE: self.transfer_fn.inhibitory_dim,
        }

    def _project_external_inputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "PopulationNetwork expected last input dimension "
                f"{self.input_dim}, got {x.shape[-1]}"
            )
        if self.transfer_fn is not None:
            x_e, x_i = self.transfer_fn(x)
            projected_e = self.input_projection(x_e)
            return {
                _INPUT_SOURCE: projected_e,
                _INPUT_E_SOURCE: projected_e,
                _INPUT_I_SOURCE: x_i,
            }
        projected = self.input_projection(x)
        return self._single_stream_external_inputs(projected)

    def _qualified_tensor(
        self,
        source: str,
        layer_outputs: dict[str, dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        layer_name, population_name = _split_qualified_source(source) or ("", "")
        try:
            return layer_outputs[layer_name][population_name]
        except KeyError as exc:
            raise RuntimeError(
                f"qualified source {source!r} was not available at runtime"
            ) from exc

    def _same_step_external_inputs_for_layer(
        self,
        layer_idx: int,
        default_inputs: dict[str, torch.Tensor],
        current_layer_outputs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        external_inputs = dict(default_inputs)
        for source in self._qualified_sources_by_layer_timing[layer_idx][_SAME_STEP]:
            external_inputs[source] = self._qualified_tensor(
                source,
                current_layer_outputs,
            )
        return external_inputs

    def _delayed_external_inputs_for_layer(
        self,
        layer_idx: int,
        previous_layer_outputs: dict[str, dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        delayed_inputs: dict[str, torch.Tensor] = {}
        for source in self._qualified_sources_by_layer_timing[layer_idx][_DELAYED]:
            delayed_inputs[source] = self._qualified_tensor(
                source,
                previous_layer_outputs,
            )
        return delayed_inputs

    def forward(
        self,
        x: torch.Tensor,
        hidden: list[PopulationLayerState] | None = None,
        return_hidden: bool = False,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, list[PopulationLayerState] | None]:
        if self.is_recurrent:
            if x.dim() == 2:
                x = x.unsqueeze(1)
            out, hidden = self._forward_recurrent(
                x,
                hidden=hidden,
                seq_lengths=seq_lengths,
            )
        else:
            out = self._forward_feedforward(x)
            hidden = None

        if return_hidden:
            return out, hidden
        return out

    def _forward_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        default_inputs = self._project_external_inputs(x)
        layer_outputs: dict[str, dict[str, torch.Tensor]] = {}
        readout = None
        for layer_idx, layer in enumerate(self.layers):
            external_inputs = self._same_step_external_inputs_for_layer(
                layer_idx,
                default_inputs,
                layer_outputs,
            )
            h, _ = layer(external_inputs)
            layer_outputs[layer.config.name] = layer._last_outputs
            if layer.config.name == self.readout_layer:
                readout = h
            default_inputs = self._single_stream_external_inputs(h)
        return h if readout is None else readout

    def _forward_recurrent(
        self,
        x: torch.Tensor,
        hidden: list[PopulationLayerState] | None = None,
        seq_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[PopulationLayerState]]:
        batch_size, seq_len, _ = x.shape
        if hidden is None:
            hidden = self.init_state(
                batch_size=batch_size,
                device=x.device,
                dtype=x.dtype,
            )

        outputs = []
        self._routing_info = []
        for t in range(seq_len):
            default_inputs = self._project_external_inputs(x[:, t, :])
            previous_layer_outputs = {
                layer.config.name: hidden[layer_idx].outputs
                for layer_idx, layer in enumerate(self.layers)
            }
            current_layer_outputs: dict[str, dict[str, torch.Tensor]] = {}
            timestep_hidden = []
            readout = None
            timestep_info = {}
            for layer_idx, layer in enumerate(self.layers):
                external_inputs = self._same_step_external_inputs_for_layer(
                    layer_idx,
                    default_inputs,
                    current_layer_outputs,
                )
                delayed_external_inputs = self._delayed_external_inputs_for_layer(
                    layer_idx,
                    previous_layer_outputs,
                )
                h, new_layer_state = layer(
                    external_inputs,
                    hidden[layer_idx],
                    delayed_external_inputs=delayed_external_inputs,
                )
                if new_layer_state is None:
                    new_layer_state = PopulationLayerState(
                        population_states=hidden[layer_idx].population_states,
                        outputs=layer._last_outputs,
                    )
                timestep_hidden.append(new_layer_state)
                current_layer_outputs[layer.config.name] = new_layer_state.outputs
                if layer.config.name == self.readout_layer:
                    readout = h
                if self._store_routing:
                    timestep_info[layer.config.name] = layer._last_routing_info
                default_inputs = self._single_stream_external_inputs(h)
            hidden = timestep_hidden
            outputs.append(h if readout is None else readout)
            if self._store_routing:
                self._routing_info.append(timestep_info)

        stacked = torch.stack(outputs, dim=1)
        return self._reduce_outputs(stacked, seq_lengths=seq_lengths), hidden


__all__ = [
    "PopulationLayer",
    "PopulationLayerConfig",
    "PopulationLayerState",
    "PopulationNetwork",
    "PopulationNetworkConfig",
    "PopulationProjectionConfig",
]
