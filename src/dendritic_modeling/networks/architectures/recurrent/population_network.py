"""Population-network feedforward/recurrent dendritic layers."""

from __future__ import annotations

import logging
from dataclasses import replace
from numbers import Integral
from typing import Any

import torch
import torch.nn as nn

from dendritic_modeling.config.keys import config_from_payload
from dendritic_modeling.networks.architectures.excitation_inhibition.input_transform.transfer import (
    TransferLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    apply_recurrent_weight_cache,
)
from dendritic_modeling.networks.architectures.recurrent.population_common import (
    PopulationDefinitionConfig,
    as_population_definition_config as _as_population_definition_config,
)
from dendritic_modeling.networks.architectures.recurrent.population_configs import (
    PopulationLayerConfig,
    PopulationNetworkConfig,
    PopulationProjectionConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_constants import (
    _DELAYED,
    _EXTERNAL_INPUT_SOURCES,
    _INPUT_E_SOURCE,
    _INPUT_I_SOURCE,
    _INPUT_SOURCE,
    _SAME_STEP,
)
from dendritic_modeling.networks.architectures.recurrent.population_layer import (
    PopulationLayer,
)
from dendritic_modeling.networks.architectures.recurrent.population_sources import (
    _as_projection_config,
    _canonical_external_source,
    _split_qualified_source,
)
from dendritic_modeling.networks.architectures.recurrent.population_state import (
    PopulationLayerState,
)
from dendritic_modeling.networks.architectures.replacement.adapters import (
    NonNegativeInputAdapter,
    transformed_feature_dim,
)
from dendritic_modeling.networks.base import BaseNetwork
from dendritic_modeling.networks.utils.forward_caches import detach_forward_caches
from dendritic_modeling.networks.utils.nonnegative_linear import NonnegativeLinear
from dendritic_modeling.networks.utils.weight_transforms import (
    POSITIVE_WEIGHT_TRANSFORMS,
)

logger = logging.getLogger(__name__)


class PopulationNetwork(BaseNetwork):
    """Stack of population-network dendritic layers."""

    def __init__(self, config: PopulationNetworkConfig):
        super().__init__()
        self.config = config
        self.input_dim = int(config.input_dim)
        self.input_transform = str(config.input_transform)
        self.input_adapter = NonNegativeInputAdapter(
            self.input_dim,
            self.input_transform,
        )
        self.adapted_input_dim = transformed_feature_dim(
            self.input_dim,
            self.input_transform,
        )
        self.output_mode = config.output_mode
        self._store_routing = bool(config.store_routing)
        self._routing_info: list[dict[str, dict]] = []
        self._use_transfer = bool(config.use_transfer)
        self.transfer_fn: TransferLayer | None = None

        input_dim = self.adapted_input_dim
        if self._use_transfer:
            self.transfer_fn = TransferLayer(
                input_dim=input_dim,
                transfer_params=config.transfer_params,
            )
            input_dim = self.transfer_fn.excitatory_dim

        proj_layers = []
        prev_dim = input_dim
        for hidden_dim in config.input_projection_dims:
            projection_type = (
                NonnegativeLinear
                if config.input_projection_weight_transform == "softplus"
                else nn.Linear
            )
            proj_layers.append(projection_type(prev_dim, int(hidden_dim)))
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
                else config_from_payload(
                    PopulationLayerConfig,
                    layer,
                    section=f"model.core.population_network.layers[{index}]",
                )
            )
            for index, layer in enumerate(config.layers)
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
        self._reject_transferred_streams_after_first_layer()

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
                layer_index=layer_idx,
            )
            self.layers.append(layer)
            current_external_dims = self._single_stream_external_dims(layer.output_dim)
            self._is_recurrent = self._is_recurrent or layer.is_recurrent

        # First-layer shunting populations with positive weights that read the
        # external input; signed input can make their conductances negative.
        first = self.layers[0]
        self._shunting_external_readers = sorted(
            {
                conn.target
                for conn in first.connections
                if conn.enabled
                and _canonical_external_source(conn.source) is not None
                and first.population_configs[conn.target].use_shunting
                and str(first.population_configs[conn.target].weight_transform).lower()
                in POSITIVE_WEIGHT_TRANSFORMS
            }
        )
        self._signed_input_checked = not self._shunting_external_readers

        # Opt-in compiled layer steps. Module.compile compiles each layer's own
        # call; copies and pickles of the network run the eager layers.
        self.compile_step = bool(config.compile_step)
        if self.compile_step:
            # torch.compile does not compile the conductance-scaled gradient
            # hooks of shunting layers: their first training step fails.
            scaled = sorted(
                {
                    f"{kind}_gradient_scaling"
                    for layer in self.branch_layers
                    if layer.use_shunting
                    for kind in ("topk", "blocklinear")
                    if getattr(layer, f"{kind}_strategy") == "conductance_dynamic"
                }
            )
            if scaled:
                raise ValueError(
                    "compile_step does not support conductance_dynamic "
                    f"{' or '.join(scaled)} in shunting populations: compiled "
                    "steps do not apply its gradient scales; set compile_step: "
                    "false"
                )
            for layer in self.layers:
                layer.compile()

        readout_idx = self.layer_names.index(self.readout_layer)
        readout_layer = self.layers[readout_idx]
        self.output_dim = readout_layer.output_dim

    # Tensors of the last forward pass kept for analyses; copies and pickles
    # hold them detached (networks.utils.forward_caches).
    _FORWARD_CACHES = ("_routing_info",)

    def __getstate__(self):
        return detach_forward_caches(super().__getstate__(), self._FORWARD_CACHES)

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

    @property
    def n_branch_layers(self) -> int:
        """Return the total number of dendritic branch layers across populations."""
        return len(self.branch_layers)

    def get_effective_params(self) -> int:
        """Calculate active/effective parameters after TopK-style sparsity.

        This mirrors the legacy E/I network API used by matched MLP baselines.
        Every non-synaptic parameter counts in full. Sparse synaptic candidate
        banks are replaced by the number of entries in their realized masks.
        This includes feedforward and recurrent E/I pathways and respects
        structural masks that can realize fewer than ``out_features * K``
        contacts.
        """
        effective = sum(p.numel() for p in self.parameters())
        seen_synapses: set[int] = set()
        for branch_layer in self.branch_layers:
            for attr in (
                "branch_excitation",
                "branch_inhibition",
                "branch_recurrent",
                "branch_rec_inhibition",
            ):
                synapse = getattr(branch_layer, attr, None)
                if synapse is None or id(synapse) in seen_synapses:
                    continue
                seen_synapses.add(id(synapse))
                candidate_params = sum(p.numel() for p in synapse.parameters())
                counts_fn = getattr(synapse, "connectivity_resource_counts", None)
                if callable(counts_fn):
                    counts = counts_fn()
                    active_synapses = counts.get("active_synapses")
                    if isinstance(active_synapses, bool) or not isinstance(
                        active_synapses, Integral
                    ):
                        raise TypeError(
                            "connectivity_resource_counts.active_synapses must "
                            "be an integer"
                        )
                    active_params = int(active_synapses)
                    if not 0 <= active_params <= candidate_params:
                        raise ValueError(
                            "connectivity_resource_counts.active_synapses lies "
                            "outside the stored synapse parameter count"
                        )
                    effective += active_params - candidate_params
                    continue
                mask_fn = getattr(synapse, "weight_mask", None)
                if not callable(mask_fn):
                    continue
                active_params = int(mask_fn().ne(0).sum().item())
                effective += active_params - candidate_params

        return int(effective)

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
            population_defs = [
                _as_population_definition_config(population)
                for population in layer_cfg.populations
            ]
            dims: dict[str, int] = {}
            polarities: dict[str, str] = {}
            for population in population_defs:
                if "." in population.name:
                    raise ValueError(
                        "population names may not contain '.' when qualified "
                        f"sources are enabled, got {population.name!r}"
                    )
                if population.name in dims:
                    raise ValueError(
                        "population names must be unique per layer, "
                        f"got {population.name!r}"
                    )
                dims[population.name] = int(population.n_neurons)
                polarities[population.name] = population.polarity
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

    def _reject_transferred_streams_after_first_layer(self) -> None:
        """Fail when a later layer names the network input's E or I stream.

        ``input_e`` and ``input_i`` are the transfer layer's excitatory and
        inhibitory streams of the network input and reach only the first
        layer. In a later layer they would silently mean the previous layer's
        readout population, whatever its polarity.
        """
        for layer_cfg in self.layer_configs[1:]:
            for raw_conn in layer_cfg.connections:
                conn = _as_projection_config(raw_conn)
                source = _canonical_external_source(conn.source)
                if source in {_INPUT_E_SOURCE, _INPUT_I_SOURCE}:
                    raise ValueError(
                        f"layer {layer_cfg.name!r} connects source {conn.source!r} "
                        f"to {conn.target!r}, but {source!r} is a stream of the "
                        "network input and reaches only the first layer. Use "
                        "'input' for the previous layer's readout population, or "
                        "a qualified 'layer.population' source."
                    )

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
        if layer_idx == 0:
            polarities = {
                _INPUT_SOURCE: "excitatory",
                _INPUT_E_SOURCE: "excitatory",
                _INPUT_I_SOURCE: "inhibitory",
            }
        else:
            # A later layer's "input" is the previous layer's readout
            # population, so its synapses take that population's sign.
            previous = self.layers[layer_idx - 1]
            readout_polarity = previous._population_by_name[
                previous.readout_population
            ].polarity
            polarities = dict.fromkeys(_EXTERNAL_INPUT_SOURCES, readout_polarity)
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
        x = self.input_adapter(x)
        if self.transfer_fn is not None:
            x_e, x_i = self.transfer_fn(x)
            projected_e = self.input_projection(x_e)
            inputs = {
                _INPUT_SOURCE: projected_e,
                _INPUT_E_SOURCE: projected_e,
                _INPUT_I_SOURCE: x_i,
            }
        else:
            inputs = self._single_stream_external_inputs(self.input_projection(x))
        if not self._signed_input_checked:
            self._warn_on_signed_shunting_input(inputs)
        return inputs

    @torch.compiler.disable
    def _warn_on_signed_shunting_input(self, inputs: dict[str, torch.Tensor]) -> None:
        """Warn once when shunting populations receive signed external input.

        Shunting conductances with positive weights assume non-negative input
        (rates); a signed input can drive the shunting denominator toward zero.
        Checked on the first batch only, to avoid a device sync per step.
        """
        self._signed_input_checked = True
        minimum = min(float(tensor.detach().min()) for tensor in inputs.values())
        if minimum < -1e-5:
            logger.warning(
                "PopulationNetwork shunting populations %s read signed external "
                "input (min %.4g); their conductances can turn negative and the "
                "shunting denominator approach zero. Use input_transform: relu "
                "or signed_split, a transfer layer with a non-negative "
                "output_activation, input_projection_dims, or "
                "data.processing.normalize: false.",
                self._shunting_external_readers,
                minimum,
            )

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

    def _feedforward_layers(self, x: torch.Tensor):
        """Yield ``(layer_idx, layer, output)`` of a feed-forward pass in order."""
        default_inputs = self._project_external_inputs(x)
        layer_outputs: dict[str, dict[str, torch.Tensor]] = {}
        for layer_idx, layer in enumerate(self.layers):
            external_inputs = self._same_step_external_inputs_for_layer(
                layer_idx,
                default_inputs,
                layer_outputs,
            )
            h, _ = layer(external_inputs)
            layer_outputs[layer.config.name] = layer._last_outputs
            yield layer_idx, layer, h
            default_inputs = self._single_stream_external_inputs(h)

    def _forward_feedforward(self, x: torch.Tensor) -> torch.Tensor:
        readout = None
        for _, layer, h in self._feedforward_layers(x):
            if layer.config.name == self.readout_layer:
                readout = h
        return h if readout is None else readout

    def forward_until(self, x: torch.Tensor, layer_index: int) -> torch.Tensor:
        """Readout output of layer ``layer_index``, computed as ``forward`` does.

        The layers before it run with the same external projections and
        same-step routing as a feed-forward ``forward``; later layers do not
        run. Recurrent networks raise.
        """
        if self.is_recurrent:
            raise ValueError(
                "forward_until runs a feed-forward pass; this PopulationNetwork "
                "is recurrent"
            )
        if not 0 <= int(layer_index) < len(self.layers):
            raise IndexError(
                f"layer_index {layer_index} outside the {len(self.layers)} layers"
            )
        for layer_idx, _, h in self._feedforward_layers(x):
            if layer_idx == layer_index:
                return h
        raise AssertionError("unreachable")

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
        # Reuse deterministic TopK selection across timesteps of this unroll
        # instead of recomputing topk selection per step.
        # Always release it, including on shape/config errors.
        apply_recurrent_weight_cache(self, True)
        try:
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
        finally:
            apply_recurrent_weight_cache(self, False)
        stacked = torch.stack(outputs, dim=1)
        return self._reduce_outputs(stacked, seq_lengths=seq_lengths), hidden


__all__ = [
    "PopulationDefinitionConfig",
    "PopulationLayer",
    "PopulationLayerConfig",
    "PopulationLayerState",
    "PopulationNetwork",
    "PopulationNetworkConfig",
    "PopulationProjectionConfig",
]
