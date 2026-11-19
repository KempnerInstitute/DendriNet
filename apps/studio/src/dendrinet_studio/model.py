"""Configuration inspection; all model semantics come from DendriNet.

``inspect_document`` validates a configuration with the library's strict
loader and describes it for the editor: the resolved population graph with
per-level contact counts for population networks, and a block overview with
the morphology of the structured dendritic family for the other model families.
Nothing here builds weights, downloads data or touches a random stream.
"""

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
from dendritic_modeling.config.recurrence import is_recurrent_core_config
from dendritic_modeling.examples import example_names, example_text
from dendritic_modeling.networks.architectures.recurrent.dendritic_geometry import (
    resolve_synapses_by_level,
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

STREAMS = ("ff_excitatory", "ff_inhibitory", "rec_excitatory", "rec_inhibitory")


def parse_document(text: str) -> dict[str, Any]:
    """Accept a standalone config, retaining fields outside the visual editor."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as error:
        line = getattr(getattr(error, "problem_mark", None), "line", None)
        problem = getattr(error, "problem", None) or "the YAML could not be read"
        where = f" (line {line + 1})" if line is not None else ""
        raise ValueError(f"Invalid YAML{where}: {problem}") from error
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


def parse_value(text: str) -> Any:
    """Read one YAML value (a mapping, list or scalar) typed into a form."""
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as error:
        line = getattr(getattr(error, "problem_mark", None), "line", None)
        problem = getattr(error, "problem", None) or "the YAML could not be read"
        where = f" (line {line + 1})" if line is not None else ""
        raise ValueError(f"Invalid YAML{where}: {problem}") from error
    json.dumps(value, allow_nan=False)
    return value


def dump_document(raw: Any) -> str:
    """The YAML text of a configuration, in the order its keys were written."""
    if not isinstance(raw, dict):
        raise ValueError("A configuration must be a mapping")
    return yaml.safe_dump(raw, sort_keys=False)


def _example_header(text: str) -> tuple[str, str]:
    """Title and description from an example's leading comment block."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("#"):
            break
        content = stripped.lstrip("#").strip()
        if content and not set(content) <= set("=-"):
            lines.append(content)
    if not lines:
        return "", ""
    title, _, rest = lines[0].partition(". ")
    description = " ".join([rest, *lines[1:]]).strip() if rest else " ".join(lines[1:])
    return title.rstrip("."), description


# The order the Studio lists the examples in: the quickest tutorial first.
EXAMPLE_ORDER = (
    "synthetic_population",
    "structured_tree_synthetic",
    "recurrent_decision_population",
    "adding_problem_gru",
    "mnist_population",
)


def example_catalog() -> list[dict[str, Any]]:
    """The installed examples with what each demonstrates."""
    from dendrinet_studio.schema import dataset_groups

    groups = dataset_groups()
    downloads = set(groups.get("Image (downloads once)", [])) | set(
        groups.get("Image variants", [])
    )
    catalog = []
    for name in example_names():
        text = example_text(name)
        raw = yaml.safe_load(text) or {}
        model = raw.get("model", {}) or {}
        core = model.get("core", {}) or {}
        title, description = _example_header(text)
        catalog.append(
            {
                "name": name,
                "title": title or name.replace("_", " "),
                "description": description,
                "family": core.get("type", ""),
                "task": model.get("task", "classification"),
                "dataset": (raw.get("data", {}) or {}).get("dataset_name", ""),
                "recurrent": bool(is_recurrent_core_config(core)),
                "download": (raw.get("data", {}) or {}).get("dataset_name")
                in downloads,
            }
        )
    rank = {name: index for index, name in enumerate(EXAMPLE_ORDER)}
    catalog.sort(key=lambda item: (rank.get(item["name"], len(rank)), item["name"]))
    return catalog


def example(name: str) -> dict[str, Any]:
    return parse_document(example_text(name))


def _levels_soma_first(branch_factors: list[int]) -> list[int]:
    return list(reversed(population_level_dims(1, branch_factors)))


def _population_graph(network: dict[str, Any]) -> dict[str, Any]:
    nodes, edges, layers = [], [], []
    signs = dict(INPUT_POLARITIES)
    definitions = [
        config_from_payload(
            PopulationLayerConfig, item, section="population_network.layers"
        )
        for item in network["layers"]
    ]
    if not definitions:
        raise ValueError("A population network needs at least one layer")
    if len({layer.name for layer in definitions}) != len(definitions):
        raise ValueError("Population layer names must be unique")
    selected_layer = network.get("readout_layer") or definitions[-1].name
    settings: dict[str, Any] = {}
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
            levels = _levels_soma_first(population.branch_factors)
            per_level = {}
            for stream in STREAMS:
                field = stream + "_synapses"
                distal_first = resolve_synapses_by_level(
                    per_level=getattr(population, field + "_by_level"),
                    fallback=getattr(population, field),
                    n_levels=len(levels),
                    field_name=field + "_by_level",
                )
                per_level[stream] = list(reversed(distal_first))
            node_id = f"{layer.name}.{definition.name}"
            signs[node_id] = polarities[definition.name] = definition.polarity
            settings[node_id] = population
            raw_item = item if isinstance(item, dict) else {}
            override = raw_item.get("population", {}) or {}
            nodes.append(
                {
                    "id": node_id,
                    "layer": li,
                    "population": pi,
                    "name": definition.name,
                    "layer_name": layer.name,
                    "polarity": definition.polarity,
                    "neurons": definition.n_neurons,
                    "branch_factors": list(definition.branch_factors),
                    "levels_including_soma": levels,
                    "compartments_per_neuron": sum(levels),
                    "configured_per_level": per_level,
                    "overridden": sorted(override),
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
        layers.append(
            {
                "name": layer.name,
                "readout": readout,
                "recurrent": bool(layer.recurrent),
                "dt": layer.dt,
                "is_readout_layer": layer.name == selected_layer,
            }
        )

    # Resolve qualified delayed sources after all populations are known.
    reached: dict[str, dict[str, set[int] | None]] = {}
    for li, layer in enumerate(definitions):
        for item in layer.connections:
            projection = as_population_projection_config(item)
            connection = dataclasses.asdict(projection)
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
            target_id = f"{layer.name}.{connection['target']}"
            if origin not in signs:
                raise ValueError(
                    f"Unknown connection source {declared!r} in layer {layer.name!r}"
                )
            if target_id not in settings:
                raise ValueError(
                    f"Unknown connection target {connection['target']!r} "
                    f"in layer {layer.name!r}"
                )
            timing = str(connection.get("timing") or "same_step").lower()
            if external and timing == "delayed":
                raise ValueError(
                    f"Delayed source {declared!r} must name a population output"
                )
            pathway = projection_pathway(
                timing, signs[origin], str(connection.get("pathway") or "").lower()
            )
            n_levels = len(_levels_soma_first(settings[target_id].branch_factors))
            depths = projection.target_depths(n_levels)
            if connection.get("enabled", True):
                current = reached.setdefault(target_id, {}).get(pathway, set())
                if depths is None or current is None:
                    reached[target_id][pathway] = None
                else:
                    reached[target_id][pathway] = current | depths
            edges.append(
                {
                    **connection,
                    "layer": li,
                    "layer_name": layer.name,
                    "source": source,
                    "source_id": origin,
                    "target_id": target_id,
                    "timing": timing,
                    "pathway": pathway,
                    "target_depths": sorted(depths) if depths is not None else None,
                }
            )

    # Contacts a neuron actually receives: a stream needs a connection, and the
    # soma carries contacts only with somatic_synapses, as the builder rules.
    for node in nodes:
        population = settings[node["id"]]
        n_levels = len(node["levels_including_soma"])
        received = {}
        for stream in STREAMS:
            allowed = reached.get(node["id"], {}).get(stream, set())
            counts = []
            for depth, count in enumerate(node["configured_per_level"][stream]):
                ok = allowed is None or depth in allowed
                if depth == 0 and not population.somatic_synapses:
                    ok = False
                counts.append(count if ok else 0)
            received[stream] = counts
        node["received_per_level"] = received
        node["synapses_per_neuron"] = sum(
            received[s][d] * node["levels_including_soma"][d]
            for s in STREAMS
            for d in range(n_levels)
        )
        node["synapses_total"] = node["synapses_per_neuron"] * node["neurons"]
    return {
        "layers": layers,
        "nodes": nodes,
        "edges": edges,
        "compartments_total": sum(
            n["neurons"] * n["compartments_per_neuron"] for n in nodes
        ),
        "synapses_total": sum(n["synapses_total"] for n in nodes),
    }


def _structured_overview(core: dict[str, Any]) -> dict[str, Any]:
    """Morphology of the structured dendritic family, read from its sections."""
    architecture = core.get("architecture", {}) or {}
    connectivity = core.get("connectivity", {}) or {}
    morphology = core.get("morphology", {}) or {}
    return {
        "excitatory_layer_sizes": architecture.get("excitatory_layer_sizes") or [],
        "inhibitory_layer_sizes": architecture.get("inhibitory_layer_sizes") or [],
        "branch_factors": list(architecture.get("excitatory_branch_factors") or []),
        "inhibitory_branch_factors": list(
            architecture.get("inhibitory_branch_factors") or []
        ),
        "ee_per_branch": connectivity.get("ee_synapses_per_branch_per_layer") or [],
        "ie_per_branch": connectivity.get("ie_synapses_per_branch_per_layer") or [],
        "use_shunting": morphology.get("use_shunting"),
        "somatic_synapses": morphology.get("somatic_synapses"),
    }


def _payload_name(core: dict[str, Any]) -> str | None:
    try:
        from dendritic_modeling.config.core_sections import core_family

        return core_family(core.get("type"))
    except Exception:
        return None


def inspect_document(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a configuration and describe it for the editor.

    This does not download data, build weights, calibrate gates, or consume the
    random streams of a training run. Counts describe the configured morphology
    and connections, not a sampled mask or a trained network.
    """
    original = copy.deepcopy(raw)
    cfg = config_from_mapping(copy.deepcopy(raw))
    plain = to_plain_dict(cfg)
    core = plain["model"]["core"]
    family = str(core.get("type") or "")
    result: dict[str, Any] = {
        "config": original,
        "yaml": yaml.safe_dump(original, sort_keys=False),
        "resolved_config": plain,
        "core_version": core_version,
        "family": family,
        "payload": _payload_name(core),
        "task": cfg.model.task,
        "dataset": cfg.data.dataset_name,
        "strategy": cfg.training.main.strategy,
        "recurrent": bool(is_recurrent_core_config(core)),
        "biological_neuron": core.get("biological_neuron"),
        "validation": "Configuration valid. Run a model check to build it.",
    }
    if family == "population_network":
        result.update(_population_graph(core["population_network"]))
    else:
        encoder = plain["model"].get("encoder", {}) or {}
        decoder = plain["model"].get("decoder", {}) or {}
        result["overview"] = {
            "encoder": encoder.get("type"),
            "input_dim": (encoder.get("params") or {}).get("input_dim"),
            "decoder": decoder.get("type"),
            "decoder_hidden": (decoder.get("params") or {}).get("hidden_dims"),
            "output_dim": (decoder.get("params") or {}).get("output_dim"),
            "structured": _structured_overview(core),
        }
    return result
