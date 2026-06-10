"""Named-cell recurrent dendritic network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.conversion import (
    deep_merge_dicts,
    has_enabled_synapse_types,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    sample_probability_mask,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.ei_state import DendriNetState
from dendritic_modeling.networks.architectures.recurrent.stateful_dendrinet import (
    StatefulDendriNet,
)
from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.utils.stable_hash import stable_seed_offset


@dataclass
class CellTypeConfig(BaseConfig):
    """Configuration for one named cell type/population."""

    name: str = "pyr"
    polarity: str = "excitatory"  # excitatory | inhibitory
    n_neurons: int = 64
    branch_factors: list[int] = field(default_factory=lambda: [1])
    population: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        if not self.name:
            raise ValueError("cell type name must be non-empty")
        self.polarity = str(self.polarity).lower()
        if self.polarity not in {"excitatory", "inhibitory"}:
            raise ValueError(
                f"polarity must be 'excitatory' or 'inhibitory', got {self.polarity!r}"
            )
        if self.n_neurons <= 0:
            raise ValueError(f"n_neurons must be > 0, got {self.n_neurons}")
        if not self.branch_factors:
            raise ValueError("branch_factors must be non-empty")


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
    """Configuration for a named population graph recurrent core."""

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


@dataclass
class PopulationGraphState:
    """Runtime state for a population graph layer."""

    population_states: dict[str, DendriNetState]
    outputs: dict[str, torch.Tensor]


def _as_cell_type_config(cell_type: CellTypeConfig | dict[str, Any]) -> CellTypeConfig:
    if isinstance(cell_type, CellTypeConfig):
        return cell_type
    if isinstance(cell_type, dict):
        return CellTypeConfig(**cell_type)
    raise TypeError(
        "cell_types entries must be CellTypeConfig or dict, "
        f"got {type(cell_type).__name__}"
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


def _population_level_dims(n_soma: int, branch_factors: list[int]) -> list[int]:
    sizes = [int(n_soma)]
    for branch_factor in branch_factors:
        sizes.append(sizes[-1] * int(branch_factor))
    sizes.reverse()
    return sizes


def _deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dict-valued population config overrides."""
    return deep_merge_dicts(base, override)


def _population_config_from_cell(
    cell: CellTypeConfig,
    defaults: dict[str, Any],
    *,
    layer_idx: int,
) -> PopulationConfig:
    payload = _deep_merge_dicts(defaults, dict(cell.population))
    payload["n_neurons"] = int(cell.n_neurons)
    payload["branch_factors"] = list(cell.branch_factors)
    payload.setdefault("structured_layer_idx", layer_idx)
    return PopulationConfig(**payload)


def _has_positive_synapses(
    pop_config: PopulationConfig, base: str, per_level: str
) -> bool:
    if int(getattr(pop_config, base, 0)) > 0:
        return True
    return any(int(value) > 0 for value in getattr(pop_config, per_level, []))


def _population_requires_temporal_state(pop_config: PopulationConfig) -> bool:
    """Return whether population dynamics need recurrent state over time."""
    return (
        str(pop_config.dynamics_mode).lower() == "spike"
        or has_enabled_synapse_types(pop_config.synapse_types)
        or bool(pop_config.dendritic_spikes_enabled)
        or bool(pop_config.soma_feedback_enabled)
    )


class PopulationGraphNetwork(BaseNetwork):
    """Opt-in recurrent core with named excitatory/inhibitory cell types.

    Each named population is implemented by ``StatefulDendriNet``. At every
    timestep, previous outputs from all excitatory populations are concatenated
    into the recurrent excitatory stream, and previous outputs from all
    inhibitory populations are concatenated into the recurrent inhibitory
    stream. Updates are synchronous, so population order does not introduce
    hidden same-timestep dependencies.
    """

    def __init__(self, config: PopulationGraphConfig):
        super().__init__()
        self.config = config
        self.output_mode = config.output_mode
        self._store_routing = bool(config.store_routing)

        self.cell_types = [_as_cell_type_config(cell) for cell in config.cell_types]
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
        known_names = set(names)
        self._connection_specs_by_pair: dict[
            tuple[str, str], PopulationConnectionConfig
        ] = {}
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

        proj_layers = []
        prev_dim = config.input_dim
        for hidden_dim in config.input_projection_dims:
            proj_layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            proj_layers.append(nn.ReLU())
            prev_dim = int(hidden_dim)
        self.input_projection = (
            nn.Sequential(*proj_layers) if proj_layers else nn.Identity()
        )
        self.proj_output_dim = prev_dim

        self._cell_dims = {cell.name: int(cell.n_neurons) for cell in self.cell_types}
        self._exc_rec_dim = sum(self._cell_dims[name] for name in self.excitatory_names)
        self._inh_rec_dim = sum(self._cell_dims[name] for name in self.inhibitory_names)

        self.population_configs: dict[str, PopulationConfig] = {}
        self._population_has_rec_exc: dict[str, bool] = {}
        self._population_has_rec_inh: dict[str, bool] = {}
        self._population_has_soma_feedback: dict[str, bool] = {}
        self._population_has_temporal_state: dict[str, bool] = {}
        self.populations = nn.ModuleDict()
        defaults = dict(config.population_defaults)
        for idx, cell in enumerate(self.cell_types):
            pop_cfg = _population_config_from_cell(cell, defaults, layer_idx=idx)
            self.population_configs[cell.name] = pop_cfg
            has_rec_exc = _has_positive_synapses(
                pop_cfg,
                "rec_excitatory_synapses",
                "rec_excitatory_synapses_by_level",
            )
            has_rec_inh = _has_positive_synapses(
                pop_cfg,
                "rec_inhibitory_synapses",
                "rec_inhibitory_synapses_by_level",
            )
            self._population_has_rec_exc[cell.name] = has_rec_exc
            self._population_has_rec_inh[cell.name] = has_rec_inh
            self._population_has_soma_feedback[cell.name] = bool(
                pop_cfg.soma_feedback_enabled
            )
            self._population_has_temporal_state[cell.name] = (
                _population_requires_temporal_state(pop_cfg)
            )
            connection_masks = self._build_connection_masks_for_cell(cell, pop_cfg)
            self.populations[cell.name] = StatefulDendriNet(
                pop_config=pop_cfg,
                excitatory_input_dim=self.proj_output_dim,
                inhibitory_input_dim=None,
                recurrent_excitatory_input_dim=(
                    self._exc_rec_dim if has_rec_exc else None
                ),
                recurrent_inhibitory_input_dim=(
                    self._inh_rec_dim if has_rec_inh else None
                ),
                recurrent_excitatory_is_self_population=False,
                recurrent_inhibitory_is_self_population=False,
                connection_masks_by_pathway=connection_masks,
                dt=config.dt,
            )

        self._has_recurrent_connections = any(
            self._population_has_rec_exc.values()
        ) or any(self._population_has_rec_inh.values())
        self._has_soma_feedback = any(self._population_has_soma_feedback.values())
        self._has_temporal_population_state = any(
            self._population_has_temporal_state.values()
        )
        self._is_recurrent = (
            self._has_recurrent_connections
            or self._has_soma_feedback
            or self._has_temporal_population_state
            or self.output_mode in {"all", "mean"}
        )
        self.output_dim = self._cell_dims[self.readout_population]
        self._routing_info: list[dict[str, dict]] = []
        self.store_routing = self._store_routing

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

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> PopulationGraphState:
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
        return PopulationGraphState(
            population_states=population_states,
            outputs=outputs,
        )

    def _concat_outputs(
        self,
        outputs: dict[str, torch.Tensor],
        names: list[str],
    ) -> torch.Tensor | None:
        if not names:
            return None
        return torch.cat([outputs[name] for name in names], dim=-1)

    def _build_connection_masks_for_cell(
        self,
        cell: CellTypeConfig,
        pop_cfg: PopulationConfig,
    ) -> dict[str, list[torch.Tensor]]:
        level_dims = _population_level_dims(cell.n_neurons, cell.branch_factors)
        masks_by_pathway = {}
        if self._population_has_rec_exc.get(cell.name, False):
            rec_exc_masks = self._build_recurrent_masks(
                target=cell.name,
                source_names=self.excitatory_names,
                level_dims=level_dims,
            )
            if rec_exc_masks is not None:
                masks_by_pathway[pop_cfg.rec_excitatory_pathway] = rec_exc_masks

        if self._population_has_rec_inh.get(cell.name, False):
            rec_inh_masks = self._build_recurrent_masks(
                target=cell.name,
                source_names=self.inhibitory_names,
                level_dims=level_dims,
            )
            if rec_inh_masks is not None:
                masks_by_pathway[pop_cfg.rec_inhibitory_pathway] = rec_inh_masks

        return masks_by_pathway

    def _build_recurrent_masks(
        self,
        *,
        target: str,
        source_names: list[str],
        level_dims: list[int],
    ) -> list[torch.Tensor] | None:
        if not source_names:
            return None

        masks = [
            self._build_recurrent_mask_level(
                target=target,
                source_names=source_names,
                out_dim=out_dim,
                level_idx=level_idx,
            )
            for level_idx, out_dim in enumerate(level_dims)
        ]
        if all(bool(mask.all()) for mask in masks):
            return None
        return masks

    def _build_recurrent_mask_level(
        self,
        *,
        target: str,
        source_names: list[str],
        out_dim: int,
        level_idx: int,
    ) -> torch.Tensor:
        blocks = []
        for source in source_names:
            spec = self._connection_specs_by_pair.get((source, target))
            probability = self._connection_probability(spec)
            source_dim = self._cell_dims[source]
            blocks.append(
                self._sample_connection_block(
                    source=source,
                    target=target,
                    probability=probability,
                    out_dim=out_dim,
                    source_dim=source_dim,
                    level_idx=level_idx,
                    seed=None if spec is None else spec.seed,
                )
            )
        return torch.cat(blocks, dim=1)

    def _connection_probability(
        self,
        spec: PopulationConnectionConfig | None,
    ) -> float:
        if spec is None:
            return 1.0
        if not spec.enabled:
            return 0.0
        if spec.probability is None:
            return 1.0
        return spec.probability

    def _sample_connection_block(
        self,
        *,
        source: str,
        target: str,
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
                + stable_seed_offset(source, target, level_idx, out_dim, source_dim)
            )
            % ((1 << 63) - 1)
        )
        return sample_probability_mask(
            out_features=out_dim,
            in_features=source_dim,
            probability=float(probability),
            generator=generator,
        )

    def _reduce_outputs(
        self,
        outputs: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.output_mode == "last":
            if seq_lengths is not None:
                B, T, _ = outputs.shape
                idx = (seq_lengths - 1).long().clamp(0, T - 1)
                return outputs[torch.arange(B, device=outputs.device), idx]
            return outputs[:, -1, :]
        if self.output_mode == "mean":
            if seq_lengths is not None:
                _B, T, _ = outputs.shape
                lengths = seq_lengths.to(device=outputs.device).long().clamp(0, T)
                mask = torch.arange(T, device=outputs.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
                masked = outputs * mask.unsqueeze(-1).to(dtype=outputs.dtype)
                denom = lengths.clamp_min(1).unsqueeze(1).to(dtype=outputs.dtype)
                return masked.sum(dim=1) / denom
            return outputs.mean(dim=1)
        if self.output_mode == "all":
            if seq_lengths is not None:
                _B, T, _ = outputs.shape
                lengths = seq_lengths.to(device=outputs.device).long().clamp(0, T)
                mask = torch.arange(T, device=outputs.device).unsqueeze(
                    0
                ) < lengths.unsqueeze(1)
                return outputs * mask.unsqueeze(-1).to(dtype=outputs.dtype)
            return outputs
        raise ValueError(f"Unknown output_mode: {self.output_mode}")

    def forward(
        self,
        x: torch.Tensor,
        hidden: PopulationGraphState | None = None,
        return_hidden: bool = False,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, PopulationGraphState]:
        squeeze_time = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_time = True
        if x.dim() != 3:
            raise ValueError(
                "PopulationGraphNetwork expects [batch, seq_len, features] "
                f"or [batch, features], got shape {tuple(x.shape)}"
            )

        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        if hidden is None:
            hidden = self.init_state(batch_size=batch_size, device=device, dtype=dtype)

        outputs = []
        self._routing_info = []
        for t in range(seq_len):
            h = self.input_projection(x[:, t, :])
            rec_exc = self._concat_outputs(hidden.outputs, self.excitatory_names)
            rec_inh = self._concat_outputs(hidden.outputs, self.inhibitory_names)

            new_population_states = {}
            new_outputs = {}
            for name, population in self.populations.items():
                out, new_state = population.step(
                    x=h,
                    inhibitory_input=None,
                    recurrent_input=(
                        rec_exc if self._population_has_rec_exc[name] else None
                    ),
                    rec_inhibitory_input=(
                        rec_inh if self._population_has_rec_inh[name] else None
                    ),
                    state=hidden.population_states[name],
                )
                new_population_states[name] = new_state
                new_outputs[name] = out

            hidden = PopulationGraphState(
                population_states=new_population_states,
                outputs=new_outputs,
            )
            outputs.append(new_outputs[self.readout_population])

            if self._store_routing:
                self._routing_info.append(
                    {
                        name: population._last_routing_info
                        for name, population in self.populations.items()
                    }
                )

        stacked = torch.stack(outputs, dim=1)
        if squeeze_time:
            reduced = stacked[:, -1, :]
        else:
            reduced = self._reduce_outputs(stacked, seq_lengths=seq_lengths)
        if return_hidden:
            return reduced, hidden
        return reduced


__all__ = [
    "CellTypeConfig",
    "PopulationConnectionConfig",
    "PopulationGraphConfig",
    "PopulationGraphNetwork",
    "PopulationGraphState",
]
