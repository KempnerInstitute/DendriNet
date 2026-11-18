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
from dendritic_modeling.examples import example_names, example_text
from dendritic_modeling.networks.architectures.recurrent.ei_config import (
    PopulationConfig,
)
from dendritic_modeling.networks.architectures.recurrent.population_common import (
    as_population_definition_config,
    population_config_from_definition,
    population_level_dims,
)


def parse_document(text: str) -> dict[str, Any]:
    """Accept a standalone config, retaining fields outside the visual editor."""
    raw = yaml.safe_load(text)
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
    nodes, edges = [], []
    for li, layer in enumerate(network["layers"]):
        for pi, item in enumerate(layer["populations"]):
            definition = as_population_definition_config(item)
            population = population_config_from_definition(
                definition,
                layer.get("population_defaults", {}),
                layer_idx=pi,
                network_layer_idx=li,
            )
            levels = list(reversed(population_level_dims(1, population.branch_factors)))
            nodes.append(
                {
                    "id": f"{layer['name']}.{definition.name}",
                    "layer": li,
                    "population": pi,
                    "name": definition.name,
                    "layer_name": layer["name"],
                    "polarity": definition.polarity,
                    "neurons": definition.n_neurons,
                    "branch_factors": definition.branch_factors,
                    "levels_including_soma": levels,
                    "compartments_per_neuron": sum(levels),
                    "settings": dataclasses.asdict(population),
                }
            )
        for connection in layer["connections"]:
            edges.append({"layer": li, "layer_name": layer["name"], **connection})
    return {
        "config": original,
        "yaml": yaml.safe_dump(original, sort_keys=False),
        "resolved_config": to_plain_dict(cfg),
        "nodes": nodes,
        "edges": edges,
        "biological_neuron": core.biological_neuron,
        "compartments_total": sum(
            n["neurons"] * n["compartments_per_neuron"] for n in nodes
        ),
        "validation": "Config checked; construction and data-dependent calibration not yet checked",
    }
