"""Configuration inspection; all model semantics come from DendriNet."""

from __future__ import annotations

import copy
import dataclasses
import json
from typing import Any

import yaml

from dendritic_modeling import __version__ as core_version
from dendritic_modeling.config import config_from_mapping
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.keys import config_from_payload
from dendritic_modeling.config.population_routing import (
    INPUT_POLARITIES,
    canonical_input_source,
    population_readout,
    projection_pathway,
)
from dendritic_modeling.examples import example_names, example_text
from dendritic_modeling.networks.architectures.recurrent.dendritic_geometry import (
    resolve_synapses_by_level,
)
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_common import (
    as_population_definition_config,
    population_config_from_definition,
    population_level_dims,
)
from dendritic_modeling.networks.architectures.recurrent.population_configs import (
    PopulationLayerConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_sources import (
    as_population_projection_config,
)


def parse_document(text: str) -> dict[str, Any]:
    """Accept a standalone config, retaining fields outside the visual editor."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML: {error}") from error
    if not isinstance(raw, dict):
        raise ValueError("A configuration must be a YAML mapping")
    # Relative composition belongs to the file loader; pasted/uploaded text has
    # no trustworthy base directory. Never silently drop a parent config.
    if "extends" in raw or "base_config" in raw:
        raise ValueError(
            "Import a composed, standalone experiment YAML, not an extends/sweep file"
        )
    json.dumps(raw, allow_nan=False)
    return raw


def schema() -> dict[str, Any]:
    """Defaults are read from core, never duplicated in JavaScript."""
    defaults = dataclasses.asdict(PopulationConfig())
    fields = {
        key: {"default": defaults[key], "type": str(field.type)}
        for key, field in ((f.name, f) for f in dataclasses.fields(PopulationConfig))
        if key in defaults
    }
    return {
        "core_version": core_version,
        "examples": list(example_names()),
        "population_fields": fields,
        "model_validation": "Configuration preview does not build or calibrate a model. Check model before training.",
    }


def example(name: str) -> dict[str, Any]:
    return parse_document(example_text(name))


def inspect_document(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate configuration keys and describe the configured population graph.

    This does not download data, build weights, calibrate gates, or consume the
    random streams of a training run. Counts describe morphology, not a sampled
    connectivity mask or a trained network.
    """
    original = copy.deepcopy(raw)
    cfg = config_from_mapping(copy.deepcopy(raw))
    core = cfg.model.core
    if core.type != "population_network":
        raise ValueError(
            "The visual editor currently supports core.type: population_network; use the core CLI for other families"
        )
    network = to_plain_dict(core.population_network)
    nodes, edges, layers = [], [], []
    signs = dict(INPUT_POLARITIES)
    definitions = [
        config_from_payload(
            PopulationLayerConfig, item, section="population_network.layers"
        )
        for item in network["layers"]
    ]
    selected_layer = network.get("readout_layer") or definitions[-1].name
    if len({layer.name for layer in definitions}) != len(definitions):
        raise ValueError("Population layer names must be unique")
    for li, layer in enumerate(definitions):
        polarities = {}
        for pi, item in enumerate(layer.populations):
            definition = as_population_definition_config(item)
            population = population_config_from_definition(
                definition,
                layer.population_defaults,
                layer_idx=pi,
                network_layer_idx=li,
            )
            levels = list(reversed(population_level_dims(1, population.branch_factors)))
            for stream in (
                "ff_excitatory",
                "ff_inhibitory",
                "rec_excitatory",
                "rec_inhibitory",
            ):
                field = stream + "_synapses"
                resolve_synapses_by_level(
                    per_level=getattr(population, field + "_by_level"),
                    fallback=getattr(population, field),
                    n_levels=len(levels),
                    field_name=field + "_by_level",
                )
            node_id = f"{layer.name}.{definition.name}"
            signs[node_id] = polarities[definition.name] = definition.polarity
            nodes.append(
                {
                    "id": node_id,
                    "layer": li,
                    "population": pi,
                    "name": definition.name,
                    "layer_name": layer.name,
                    "polarity": definition.polarity,
                    "neurons": definition.n_neurons,
                    "branch_factors": definition.branch_factors,
                    "levels_including_soma": levels,
                    "compartments_per_neuron": sum(levels),
                    "settings": dataclasses.asdict(population),
                }
            )
        if "excitatory" not in polarities.values():
            raise ValueError(f"Layer {layer.name!r} requires an excitatory population")
        override = (
            network.get("readout_population") if layer.name == selected_layer else None
        )
        if (
            override
            and layer.readout_population
            and override != layer.readout_population
        ):
            raise ValueError(f"Conflicting readout populations in layer {layer.name!r}")
        readout = population_readout(polarities, override or layer.readout_population)
        if not readout or readout not in polarities:
            raise ValueError(
                f"Layer {layer.name!r} requires a valid readout population"
            )
        layers.append({"name": layer.name, "readout": readout})

    # Resolve qualified delayed sources after all populations are known.
    for li, layer in enumerate(definitions):
        for item in layer.connections:
            connection = to_plain_dict(as_population_projection_config(item))
            declared = connection["source"]
            external = canonical_input_source(declared)
            source = external or declared
            if external and li:
                if external != "input":
                    raise ValueError(
                        f"{declared!r} is only available in the first layer"
                    )
                previous = layers[li - 1]
                origin = f"{previous['name']}.{previous['readout']}"
            elif external or "." in source:
                origin = source
            else:
                origin = f"{layer.name}.{source}"
            timing = connection.get("timing", "same_step").lower()
            edges.append(
                {
                    **connection,
                    "layer": li,
                    "layer_name": layer.name,
                    "source": source,
                    "source_id": origin,
                    "target_id": f"{layer.name}.{connection['target']}",
                    "timing": timing,
                    "pathway": projection_pathway(
                        timing, signs[origin], (connection.get("pathway") or "").lower()
                    ),
                }
            )
    return {
        "config": original,
        "yaml": yaml.safe_dump(original, sort_keys=False),
        "resolved_config": to_plain_dict(cfg),
        "layers": layers,
        "nodes": nodes,
        "edges": edges,
        "biological_neuron": core.biological_neuron,
        "compartments_total": sum(
            n["neurons"] * n["compartments_per_neuron"] for n in nodes
        ),
        "validation": "Config checked; construction and data-dependent calibration not yet checked",
    }
