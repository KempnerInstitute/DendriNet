"""Modern typed configuration validation.

This module validates structural contracts that should fail at config-load time
instead of surfacing later as shape errors in factories or forward passes.
It intentionally stays conservative: scientific/optimization warnings remain in
the training-stability validator, while this layer rejects only impossible or
internally inconsistent model definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.core_sections import (
    SPATIAL_DENDRITIC_CONV_TYPES,
    SPATIAL_DENDRITIC_TYPES,
    SPATIAL_PATCH_POINT_TYPES,
)
from dendritic_modeling.config.population_routing import (
    INPUT_ALIASES as _EXTERNAL_ALIASES,
    INPUT_POLARITIES as _EXTERNAL_POLARITIES,
    canonical_input_source as _canonical_external_source,
    population_readout,
    projection_pathway,
)
from dendritic_modeling.config.recurrence import _STRUCTURED_EI_TYPES

# Controls whose matched E/I reference network the structured sections describe.
_STRUCTURED_REFERENCE_CONTROLS = {
    "point_mlp",
    "direct_active_matched_point",
    "sparse_active_matched_point",
    "total_param_mlp",
    "active_param_mlp",
    "ss_mlp",
    "ss_mlp_flat",
}
# Core types built from the structured ``architecture`` and ``connectivity``
# sections: the structured E/I types, the spatial cores that run one on a
# flattened map or on every patch, and the matched controls. ``ei_net`` and
# ``unified_einet`` are retired aliases checked here since before.
_LEGACY_EI_TYPES = (
    set(_STRUCTURED_EI_TYPES)
    | {"ei_net", "unified_einet"}
    | SPATIAL_DENDRITIC_TYPES
    | SPATIAL_DENDRITIC_CONV_TYPES
    | SPATIAL_PATCH_POINT_TYPES
    | _STRUCTURED_REFERENCE_CONTROLS
)
_POPULATION_NETWORK_TYPES = {"population_network"}
_REMOVED_STRUCTURED_TYPES = {
    "population_graph": "Use core.type: population_network with population_network.layers[].populations.",
    "cell_type_graph": "Use core.type: population_network with population_network.layers[].populations.",
    "cell_types": "Use core.type: population_network with population_network.layers[].populations.",
}
_SAME_STEP = "same_step"
_DELAYED = "delayed"
_FF_EXC = "ff_excitatory"
_FF_INH = "ff_inhibitory"
_REC_EXC = "rec_excitatory"
_REC_INH = "rec_inhibitory"


@dataclass
class ConfigValidationError(ValueError):
    """Raised when a loaded config violates structural model contracts."""

    errors: list[str]

    def __str__(self) -> str:
        details = "\n".join(f"- {error}" for error in self.errors)
        return f"Invalid configuration:\n{details}"


@dataclass
class _PopulationNetworkValidationIndex:
    """Layer and population lookup tables built during structural validation."""

    layer_names: list[str]
    population_dims: dict[str, dict[str, int]]
    population_polarities: dict[str, dict[str, str]]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _core_type(core: Any) -> str:
    return str(_get(core, "type", "") or "").lower()


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _is_enabled(payload: dict[str, Any]) -> bool:
    return bool(payload.get("enabled", True))


def _validate_positive_ints(
    values: list[Any],
    *,
    name: str,
    errors: list[str],
    allow_empty: bool = False,
    allow_zero: bool = False,
) -> None:
    if not values:
        if not allow_empty:
            errors.append(f"{name} must be non-empty")
        return
    for idx, value in enumerate(values):
        try:
            intval = int(value)
        except (TypeError, ValueError):
            errors.append(f"{name}[{idx}] must be an integer, got {value!r}")
            continue
        if intval < 0 or (intval == 0 and not allow_zero):
            bound = ">= 0" if allow_zero else "> 0"
            errors.append(f"{name}[{idx}] must be {bound}, got {intval}")


def _validate_nonnegative_ints(
    values: list[Any],
    *,
    name: str,
    errors: list[str],
    allow_empty: bool = False,
) -> None:
    if not values:
        if not allow_empty:
            errors.append(f"{name} must be non-empty")
        return
    for idx, value in enumerate(values):
        try:
            intval = int(value)
        except (TypeError, ValueError):
            errors.append(f"{name}[{idx}] must be an integer, got {value!r}")
            continue
        if intval < 0:
            errors.append(f"{name}[{idx}] must be >= 0, got {intval}")


def _validate_layer_list_length(
    values: list[Any],
    *,
    name: str,
    n_layers: int,
    errors: list[str],
    allow_empty: bool = False,
) -> None:
    if not values:
        if not allow_empty:
            errors.append(f"{name} must be non-empty")
        return
    if len(values) > n_layers:
        errors.append(
            f"{name} has {len(values)} entries for {n_layers} layer(s); "
            "use one value to repeat or one value per layer"
        )


def _validate_legacy_ei_core(core: Any, errors: list[str]) -> None:
    arch = _get(core, "architecture", {})
    conn = _get(core, "connectivity", {})
    transfer = _get(core, "transfer", {})

    excitatory_sizes = _as_list(_get(arch, "excitatory_layer_sizes", []))
    inhibitory_sizes = _as_list(_get(arch, "inhibitory_layer_sizes", []))
    input_projection_dims = _as_list(_get(arch, "input_projection_dims", []))
    n_layers = len(excitatory_sizes)

    _validate_positive_ints(
        excitatory_sizes,
        name="architecture.excitatory_layer_sizes",
        errors=errors,
    )
    # 0 is a layer without inhibitory cells.
    _validate_positive_ints(
        inhibitory_sizes,
        name="architecture.inhibitory_layer_sizes",
        errors=errors,
        allow_empty=True,
        allow_zero=True,
    )
    _validate_positive_ints(
        input_projection_dims,
        name="architecture.input_projection_dims",
        errors=errors,
        allow_empty=True,
    )
    if inhibitory_sizes and n_layers and len(inhibitory_sizes) > n_layers:
        errors.append(
            "architecture.inhibitory_layer_sizes has more entries than "
            "architecture.excitatory_layer_sizes"
        )

    for field_name in (
        "ee_synapses_per_branch_per_layer",
        "ei_synapses_per_branch_per_layer",
        "ie_synapses_per_branch_per_layer",
        "ii_synapses_per_branch_per_layer",
    ):
        values = _as_list(_get(conn, field_name, []))
        _validate_nonnegative_ints(
            values,
            name=f"connectivity.{field_name}",
            errors=errors,
        )
        _validate_layer_list_length(
            values,
            name=f"connectivity.{field_name}",
            n_layers=max(n_layers, 1),
            errors=errors,
        )

    for field_name in (
        "rec_ee_synapses_per_branch",
        "rec_ie_synapses_per_branch",
        "rec_ei_synapses_per_branch",
        "rec_ii_synapses_per_branch",
    ):
        values = _as_list(_get(conn, field_name, []))
        _validate_nonnegative_ints(
            values,
            name=f"connectivity.{field_name}",
            errors=errors,
            allow_empty=True,
        )
        _validate_layer_list_length(
            values,
            name=f"connectivity.{field_name}",
            n_layers=max(n_layers, 1),
            errors=errors,
            allow_empty=True,
        )

    for field_name in ("excitatory_dim", "inhibitory_dim"):
        value = _get(transfer, field_name, None)
        if value is None:
            continue
        try:
            dim = int(value)
        except (TypeError, ValueError):
            errors.append(f"transfer.{field_name} must be an integer, got {value!r}")
            continue
        if dim <= 0:
            errors.append(f"transfer.{field_name} must be > 0 when set, got {value}")


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


def _validate_population_entry(
    population: dict[str, Any],
    *,
    context: str,
    errors: list[str],
) -> tuple[str, str, int] | None:
    name = str(population.get("name", ""))
    if not name:
        errors.append(f"{context}.name must be non-empty")
        return None
    polarity = str(population.get("polarity", "excitatory")).lower()
    if polarity not in {"excitatory", "inhibitory"}:
        errors.append(
            f"{context}.polarity must be 'excitatory' or 'inhibitory', got {polarity!r}"
        )
    try:
        n_neurons = int(population.get("n_neurons", 0))
    except (TypeError, ValueError):
        errors.append(f"{context}.n_neurons must be an integer")
        n_neurons = 0
    if n_neurons <= 0:
        errors.append(f"{context}.n_neurons must be > 0, got {n_neurons}")
    branch_factors = _as_list(population.get("branch_factors", []))
    _validate_positive_ints(
        branch_factors,
        name=f"{context}.branch_factors",
        errors=errors,
        allow_empty=True,
    )
    return name, polarity, n_neurons


def _validate_probability(
    value: Any,
    *,
    context: str,
    errors: list[str],
) -> None:
    if value is None:
        return
    try:
        probability = float(value)
    except (TypeError, ValueError):
        errors.append(f"{context} probability must be numeric, got {value!r}")
        return
    if probability < 0.0 or probability > 1.0:
        errors.append(f"{context} probability must be in [0, 1], got {probability}")


def _connection_pathway(
    connection: dict[str, Any],
    *,
    source_polarity: str,
) -> str:
    pathway = str(connection.get("pathway", "") or "").lower()
    timing = str(connection.get("timing", _SAME_STEP)).lower()
    return projection_pathway(timing, source_polarity, pathway)


def _validate_same_step_acyclic(
    *,
    layer_name: str,
    population_names: set[str],
    connections: list[dict[str, Any]],
    errors: list[str],
) -> None:
    incoming = dict.fromkeys(population_names, 0)
    outgoing = {name: [] for name in population_names}
    for conn in connections:
        if not _is_enabled(conn):
            continue
        if str(conn.get("timing", _SAME_STEP)).lower() != _SAME_STEP:
            continue
        source = _canonical_external_source(conn.get("source", "input")) or str(
            conn.get("source", "input")
        )
        target = str(conn.get("target", ""))
        if source not in population_names or target not in population_names:
            continue
        outgoing[source].append(target)
        incoming[target] += 1

    queue = [name for name, degree in incoming.items() if degree == 0]
    visited = []
    while queue:
        name = queue.pop(0)
        visited.append(name)
        for target in outgoing[name]:
            incoming[target] -= 1
            if incoming[target] == 0:
                queue.append(target)
    if len(visited) != len(population_names):
        cyclic = [name for name, degree in incoming.items() if degree > 0]
        errors.append(
            f"population_network layer {layer_name!r} has a same-step cycle "
            f"involving {cyclic}; use timing: delayed for recurrent cycles"
        )


def _validate_population_network_core(core: Any, errors: list[str]) -> None:
    config = to_plain_dict(_get(core, "population_network", {}))
    layers = [to_plain_dict(layer) for layer in config.get("layers", []) or []]
    if not layers:
        errors.append("population_network.layers must contain at least one layer")
        return

    validation_index = _validate_population_network_layers(layers, errors)
    _validate_population_network_readout(config, validation_index, errors)
    _validate_population_network_connections(
        layers, validation_index, errors, config=config
    )


def _validate_population_network_layers(
    layers: list[dict[str, Any]],
    errors: list[str],
) -> _PopulationNetworkValidationIndex:
    """Validate layer/population declarations and build lookup tables."""
    layer_names: list[str] = []
    population_dims: dict[str, dict[str, int]] = {}
    population_polarities: dict[str, dict[str, str]] = {}
    for layer_idx, layer in enumerate(layers):
        layer_name = str(layer.get("name", f"layer{layer_idx}"))
        if not layer_name:
            errors.append(
                f"population_network.layers[{layer_idx}].name must be non-empty"
            )
        if "." in layer_name:
            errors.append(
                f"population_network layer names may not contain '.', got {layer_name!r}"
            )
        if layer_name in layer_names:
            errors.append(f"duplicate population_network layer name {layer_name!r}")
        layer_names.append(layer_name)

        populations = [
            to_plain_dict(population)
            for population in layer.get("populations", []) or []
        ]
        if not populations:
            errors.append(
                f"population_network layer {layer_name!r} requires populations"
            )
            continue
        dims: dict[str, int] = {}
        polarities: dict[str, str] = {}
        for pop_idx, population in enumerate(populations):
            parsed = _validate_population_entry(
                population,
                context=f"population_network.layers[{layer_idx}].populations[{pop_idx}]",
                errors=errors,
            )
            if parsed is None:
                continue
            pop_name, polarity, n_neurons = parsed
            if "." in pop_name:
                errors.append(f"population names may not contain '.', got {pop_name!r}")
            if pop_name in dims:
                errors.append(
                    f"duplicate population name {pop_name!r} in layer {layer_name!r}"
                )
            dims[pop_name] = n_neurons
            polarities[pop_name] = polarity
        if "excitatory" not in set(polarities.values()):
            errors.append(
                f"population_network layer {layer_name!r} needs an excitatory population"
            )
        readout = layer.get("readout_population")
        if readout is not None and str(readout) not in dims:
            errors.append(
                f"readout_population {readout!r} is not in layer {layer_name!r}"
            )
        population_dims[layer_name] = dims
        population_polarities[layer_name] = polarities

    return _PopulationNetworkValidationIndex(
        layer_names=layer_names,
        population_dims=population_dims,
        population_polarities=population_polarities,
    )


def _validate_population_network_readout(
    config: dict[str, Any],
    validation_index: _PopulationNetworkValidationIndex,
    errors: list[str],
) -> None:
    """Validate network-level readout layer and population references."""
    layer_names = validation_index.layer_names
    population_dims = validation_index.population_dims
    layer_index = {name: idx for idx, name in enumerate(layer_names)}
    network_readout_layer = config.get("readout_layer")
    if (
        network_readout_layer is not None
        and str(network_readout_layer) not in layer_index
    ):
        errors.append(
            f"population_network.readout_layer {network_readout_layer!r} is not a layer"
        )
    network_readout_population = config.get("readout_population")
    if network_readout_population is not None:
        selected_layer = str(network_readout_layer or layer_names[-1])
        if str(network_readout_population) not in population_dims.get(
            selected_layer, {}
        ):
            errors.append(
                "population_network.readout_population "
                f"{network_readout_population!r} is not in readout layer {selected_layer!r}"
            )


def _validate_population_network_connections(
    layers: list[dict[str, Any]],
    validation_index: _PopulationNetworkValidationIndex,
    errors: list[str],
    *,
    config: dict[str, Any],
) -> None:
    """Validate per-layer population connection declarations."""
    layer_names = validation_index.layer_names
    population_dims = validation_index.population_dims
    population_polarities = validation_index.population_polarities
    layer_index = {name: idx for idx, name in enumerate(layer_names)}
    readout_layer = config.get("readout_layer") or layer_names[-1]
    previous_readout_polarity = None

    for layer_idx, layer in enumerate(layers):
        layer_name = layer_names[layer_idx]
        local_names = set(population_dims.get(layer_name, {}))
        local_polarities = population_polarities.get(layer_name, {})
        input_polarity = previous_readout_polarity
        override = (
            config.get("readout_population") if layer_name == readout_layer else None
        )
        readout = population_readout(
            local_polarities, override or layer.get("readout_population")
        )
        previous_readout_polarity = local_polarities.get(readout)
        connections = [
            to_plain_dict(connection)
            for connection in layer.get("connections", []) or []
        ]
        seen_connections: set[tuple[str, str, str, str]] = set()
        has_ff_exc_input = dict.fromkeys(local_names, False)
        for conn_idx, connection in enumerate(connections):
            source_raw = str(connection.get("source", "input"))
            source = _canonical_external_source(source_raw) or source_raw
            target = str(connection.get("target", ""))
            timing = str(connection.get("timing", _SAME_STEP)).lower()
            if timing not in {_SAME_STEP, _DELAYED}:
                errors.append(
                    f"population_network layer {layer_name!r} connection {conn_idx} "
                    f"has invalid timing {timing!r}"
                )
                continue
            if target not in local_names:
                errors.append(
                    f"population_network layer {layer_name!r} connection target "
                    f"{target!r} is not a population"
                )
                continue
            if timing == _DELAYED and not bool(layer.get("recurrent", False)):
                errors.append(
                    f"population_network layer {layer_name!r} delayed connection "
                    "requires recurrent: true"
                )

            source_polarity = _EXTERNAL_POLARITIES.get(source, "excitatory")
            if input_polarity and source in _EXTERNAL_POLARITIES:
                source_polarity = input_polarity
            qualified = None
            if source not in _EXTERNAL_ALIASES.values() and source not in local_names:
                try:
                    qualified = _split_qualified_source(source)
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                if qualified is None:
                    errors.append(
                        f"population_network layer {layer_name!r} connection source "
                        f"{source!r} is neither external nor a known population"
                    )
                    continue
                source_layer, source_population = qualified
                if source_layer not in layer_index:
                    errors.append(
                        f"qualified source {source!r} references unknown layer "
                        f"{source_layer!r}"
                    )
                    continue
                if source_population not in population_dims[source_layer]:
                    errors.append(
                        f"qualified source {source!r} references unknown population "
                        f"{source_population!r}"
                    )
                    continue
                if layer_index[source_layer] == layer_idx:
                    errors.append(
                        f"qualified source {source!r} points to its own layer; "
                        "use the unqualified population name for same-layer edges"
                    )
                if timing == _SAME_STEP and layer_index[source_layer] > layer_idx:
                    errors.append(
                        "same_step qualified population sources must come from "
                        f"earlier layers, got {source!r} for {layer_name!r}"
                    )
                source_polarity = population_polarities[source_layer][source_population]
            elif source in local_names:
                source_polarity = local_polarities[source]

            probability = connection.get("probability", connection.get("p_connect"))
            _validate_probability(
                probability,
                context=(
                    f"population_network layer {layer_name!r} connection "
                    f"{source}->{target}"
                ),
                errors=errors,
            )
            pathway = _connection_pathway(connection, source_polarity=source_polarity)
            if timing == _SAME_STEP and pathway not in {_FF_EXC, _FF_INH}:
                errors.append(
                    f"same_step connection {source}->{target} in layer {layer_name!r} "
                    f"must use ff_excitatory or ff_inhibitory, got {pathway!r}"
                )
            if timing == _DELAYED and pathway not in {_REC_EXC, _REC_INH}:
                errors.append(
                    f"delayed connection {source}->{target} in layer {layer_name!r} "
                    f"must use rec_excitatory or rec_inhibitory, got {pathway!r}"
                )
            key = (source, target, timing, pathway)
            if key in seen_connections:
                errors.append(
                    f"duplicate population_network connection {source}->{target} "
                    f"({timing}, {pathway}) in layer {layer_name!r}"
                )
            seen_connections.add(key)
            if _is_enabled(connection) and timing == _SAME_STEP and pathway == _FF_EXC:
                has_ff_exc_input[target] = True

        for population_name, has_input in has_ff_exc_input.items():
            if not has_input:
                errors.append(
                    f"population {population_name!r} in layer {layer_name!r} has no "
                    "same-step ff_excitatory input"
                )
        _validate_same_step_acyclic(
            layer_name=layer_name,
            population_names=local_names,
            connections=connections,
            errors=errors,
        )


def validate_loaded_config(config: Any) -> None:
    """Raise ``ConfigValidationError`` if a loaded config is structurally invalid."""

    model = _get(config, "model", {})
    core = _get(model, "core", {})
    core_type = _core_type(core)
    errors: list[str] = []

    # An enabled extension section with its own selection step builds the
    # executable core later, so model.core is then a placeholder and is not
    # validated here.
    selector_compiles_core = any(
        bool(_get(_get(section, "selection", {}), "enabled", False))
        for section in (
            _get(model, "transformer_replacement", {}),
            _get(model, "vision_replacement", {}),
            _get(model, "pretrained_replacement", {}),
        )
        if bool(_get(section, "enabled", False))
    )

    if selector_compiles_core:
        pass
    elif core_type in _LEGACY_EI_TYPES:
        _validate_legacy_ei_core(core, errors)
    elif core_type in _POPULATION_NETWORK_TYPES:
        _validate_population_network_core(core, errors)
    elif core_type == "feature_readout":
        from dendritic_modeling.config.feature_readout import FeatureReadoutConfig

        payload = _get(core, "feature_readout", None)
        if payload is None:
            errors.append("feature_readout requires its configuration section")
        else:
            try:
                FeatureReadoutConfig(**to_plain_dict(payload)).validate()
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        main = _get(_get(config, "training", {}), "main", {})
        if _get(main, "strategy", "standard") != "standard":
            errors.append("feature_readout supports only standard backpropagation")
    elif core_type == "routed_ei":
        from dendritic_modeling.config.routed_ei import RoutedEIConfig

        payload = _get(core, "routed_ei", None)
        if payload is None:
            errors.append("routed_ei requires its configuration section")
        else:
            try:
                RoutedEIConfig(**to_plain_dict(payload)).validate()
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        main = _get(_get(config, "training", {}), "main", {})
        if _get(main, "strategy", "standard") != "standard":
            errors.append(
                "routed_ei supports only the standard backpropagation trainer"
            )
        if _get(_get(main, "common", {}), "use_amp", False):
            errors.append(
                "routed_ei computes in its parameter dtype; set "
                "training.main.common.use_amp: false"
            )
    elif core_type == "shared_projection_shunting":
        from dendritic_modeling.config.shared_projection import (
            SharedProjectionShuntingConfig,
        )

        payload = _get(core, "shared_projection_shunting", None)
        if payload is None:
            errors.append(
                "shared_projection_shunting requires its configuration section"
            )
        else:
            try:
                SharedProjectionShuntingConfig(**to_plain_dict(payload)).validate()
            except (TypeError, ValueError) as exc:
                errors.append(str(exc))
        main = _get(_get(config, "training", {}), "main", {})
        if _get(main, "strategy", "standard") != "standard":
            errors.append(
                "shared_projection_shunting supports only the standard backpropagation trainer"
            )
    elif core_type in _REMOVED_STRUCTURED_TYPES:
        errors.append(
            f"core.type {core_type!r} has been removed. "
            f"{_REMOVED_STRUCTURED_TYPES[core_type]}"
        )

    if errors:
        raise ConfigValidationError(errors)


__all__ = ["ConfigValidationError", "validate_loaded_config"]
