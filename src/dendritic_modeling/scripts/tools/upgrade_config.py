"""Rewrite experiment or sweep YAML files written for earlier library versions.

The current loader rejects keys no builder reads, warns about retired keys and
renamed fields, and loads old names under their new ones. This tool writes a
copy of a config in the current form, so it loads without those warnings and
errors while describing the same run:

* top-level keys that are not configuration sections (study annotations such
  as ``paper_contract`` or ``schema_version``) move under ``metadata``;
* renamed fields take their new names (``weight_decay_rate`` becomes
  ``synapse_decay_rate``, population ``topk_strategy`` becomes
  ``topk_gradient_scaling``, ...), including dotted sweep override paths;
* retired keys whose value the current library treats as the old default are
  removed; retired keys with other values are left in place and reported,
  because the run they describe cannot be reproduced by this version;
  retired entries of list options that were never read (the correlation
  ``stratified_categories`` entry ``across_layer``) are removed;
* the retired ``transfer.input_mode`` routing becomes explicit counts: a
  feedforward EINet with ``input_mode: 1`` (the former default) built none of
  its declared inhibitory cells, so their sizes become 0; a recurrent EINet
  with ``input_mode: 0`` or ``inhibitory_mode: none`` gets the first-layer
  ``ie`` count it ran with (0). Routing the current rule cannot express is
  reported;
* a ``local_ca`` run on a core with forward features the local rules do not
  model (typed synapses, spiking, dendritic spikes, soma feedback, parallel
  readout) gets ``learning_strategy_config.allow_unmodeled_forward_features:
  true``, the explicit form of what it did, and one with a path-based error
  mode gets ``allow_scalar_fallback: true`` (it used the scalar error where no
  path existed);
* annotation blocks inside ``analysis`` move under ``metadata.analysis``;
  morphology values a core-type alias replaced are removed; a sweep's
  ``ei-sum``/``ei-ratio`` filter gets the tolerance generation used (0, 0.1);
* settings the library ignored are removed: the retired
  ``model.learned_output_scale`` (``output_scale_mode`` selected the scale; a
  lone ``false`` becomes ``output_scale_mode: fixed``), encoder/decoder params
  their type does not read, population sparse-synapse options their synapse
  type does not read, additive-integration options of shunting layers,
  ``recurrent_ei`` settings of feedforward EINets, a learned output-scale mode
  without an identity decoder (now ``fixed``), ``quantile`` gate updates
  without recalibration (now ``frozen``), ``data.processing`` flags the dataset's loader
  does not apply, step options such as ``enable_nan_checking`` that
  the selected training loop does not apply, population-network or unified
  E/I ``input_dim``
  values (set from the encoder), and a layer's
  ``population_defaults.indexed_seed`` that its indexed populations replaced
  with the experiment topology stream, and ``model.core`` sections the core
  type never reads (for example ``morphology`` on a population network);
* keys rewritten by the configuration migrations of installed plugins (for
  example a legacy key of a plugin's own section);
* synapse counts the library now requires to be explicit get the value that
  was used: a delayed population-network connection whose target never set its
  recurrent count gets ``rec_*_synapses: 0`` (it stays a lesion); a unified
  E/I layer without an ``inhibitory`` entry gets the default population it
  built; a short
  structured ``inhibitory_layer_sizes`` list gets one size per layer (the
  recurrent EINet repeated its last entry, the feedforward EINet built no
  inhibitory cells past its end); and structured E/I layers without inhibitory
  cells get zero counts for synapses that had no source or target;
* stochastic Top-K settings left out get the values that applied before
  2026-10-08: ``model.core.sparsity`` of type stochastic gets
  ``temperature: 0.0``, population options get ``topk_temperature: 0.0`` or
  ``topk_ultrafast: false``, and stochastic selection at the former
  temperature 0 without noise, which selected exactly the standard Top-K,
  becomes ``selection: standard`` for indexed dynamic synapses and is reported
  for the stochastic layer type;
* a top-level ``fsdp`` block identical to ``distributed.fsdp``, or with
  ``use_fsdp: false`` (a run without FSDP), is removed; any other is reported,
  since whether it was read depended on the library version.

Those rewrites apply to files written before ``config_version: 2``. Files
written for version 2 or earlier also get the version-3 rewrites:

* a ``model.core.connectivity.structured`` block whose ``distance`` rules
  have no kernel gets ``distance_kernel: none`` (version 3 reads an omitted
  kernel as gaussian); the second spelling of a key (``kernel``, ``sigma``,
  ``p_connect``, ``in_degree``) that the root's key shadowed in a pathway
  entry, and the second of two spellings or pathway entries written for one
  key or pathway, are removed (version 3 reads the entry's value, or raises);
* a population-network or unified E/I structured rule without a seed gets
  ``seed: global_rng`` (its masks drew from the global random stream; version
  3 seeds it from the topology seed), or ``seed: 0`` for spatial contacts.
  The rewritten rules are compared with their version-2 reading for every
  population and pathway, and a difference is reported;
* inhibitory counts of unified E/I populations that had no input become 0,
  and the recurrent baselines' ``input_dim`` (set from the encoder) and the
  ``baseline_rnn`` ``cell_type`` (set by the core type) are removed.

The upgraded file starts with ``config_version`` (see
:data:`~dendritic_modeling.config.CONFIG_VERSION`). A file that already has
the current version is left unchanged, so upgrading twice changes nothing.
What the tool cannot rewrite is reported and also listed in the copy under
``unresolved_migration``: the loader refuses such a partly upgraded file, and
a later run reports the same items again without rewriting anything. Resolve
each item by hand, then delete the key.

Sweep files are upgraded section by section (``base_config``, sweep axes and
variant overrides). Comments and key order of the original YAML are not kept.

Usage::

    dendritic-upgrade-config old.yaml --output-dir upgraded/ [--report report.json]

Copies keep their paths relative to the common directory of the inputs, and
the tool stops before writing when two inputs would be written to one file.

Load the upgraded file with ``dendritic-train --validate-only`` (or compare a
short run with the original at the old library revision) before relying on it.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import sys
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import yaml

from dendritic_modeling.config.config import CONFIG_VERSION, UNRESOLVED_MIGRATION_KEY
from dendritic_modeling.config.keys import RETIRED_KEYS, RETIRED_VALUES

# Dotted section paths (``*`` matches every list item or mapping value) and the
# config class whose RENAMED_FIELDS apply there.
_RENAMED_SECTIONS: tuple[tuple[str, str], ...] = (
    ("training.main.common", "dendritic_modeling.config.training:CommonTrainerConfig"),
    (
        "training.main.strategies.recurrent",
        "dendritic_modeling.config.training:RecurrentStrategyConfig",
    ),
    (
        "training.main.strategies.multi_stage",
        "dendritic_modeling.config.multi_stage_training:MultiStageStrategyConfig",
    ),
    (
        "model.core.population_network.layers.*.population_defaults",
        "dendritic_modeling.networks.architectures.recurrent.ei_config:PopulationConfig",
    ),
    (
        "model.core.population_network.layers.*.populations.*.population",
        "dendritic_modeling.networks.architectures.recurrent.ei_config:PopulationConfig",
    ),
    (
        "model.core.unified_ei.layers.*.excitatory",
        "dendritic_modeling.networks.architectures.recurrent.ei_config:PopulationConfig",
    ),
    (
        "model.core.unified_ei.layers.*.inhibitory",
        "dendritic_modeling.networks.architectures.recurrent.ei_config:PopulationConfig",
    ),
)


def _load_class(spec: str) -> type:
    module, name = spec.split(":")
    return getattr(__import__(module, fromlist=[name]), name)


def _renamed_fields(spec: str) -> dict[str, str]:
    renamed = getattr(_load_class(spec), "RENAMED_FIELDS", None) or {}
    return {old: new for old, (new, _date) in renamed.items()}


def _sections(node: Any, parts: list[str], path: str) -> Iterable[tuple[str, dict]]:
    """Yield (path, mapping) for every mapping matching the dotted pattern."""
    if not parts:
        if isinstance(node, dict):
            yield path, node
        return
    head, rest = parts[0], parts[1:]
    if head == "*":
        items = (
            enumerate(node)
            if isinstance(node, list)
            else node.items() if isinstance(node, dict) else ()
        )
        for key, value in items:
            yield from _sections(value, rest, f"{path}[{key}]")
    elif isinstance(node, dict) and head in node:
        yield from _sections(node[head], rest, f"{path}.{head}" if path else head)


def _rename_in_place(
    section: dict, renamed: dict[str, str], path: str, notes: list[str]
):
    for old, new in renamed.items():
        if old not in section:
            continue
        value = section.pop(old)
        if new in section and section[new] != value:
            notes.append(
                f"{path}: kept {new}={section[new]!r} and dropped the conflicting "
                f"old {old}={value!r}"
            )
            continue
        section[new] = value
        notes.append(f"{path}.{old} -> {path}.{new}")


# Retired keys that only configured logging to an external service: removing
# them leaves the run itself unchanged, whatever their value.
_LOGGING_ONLY = frozenset({("", "wandb")})


def _drop_retired(
    data: dict,
    notes: list[str],
    unresolved: list[str],
    *,
    skip: frozenset[tuple[str, str]] = frozenset(),
) -> None:
    for section_path, retired in RETIRED_KEYS.items():
        parts = section_path.split(".") if section_path else []
        for path, section in _sections(data, parts, ""):
            for key, entry in retired.items():
                if key not in section or (section_path, key) in skip:
                    continue
                full = f"{path}.{key}" if path else key
                if entry.accepted(section[key]) or (section_path, key) in _LOGGING_ONLY:
                    del section[key]
                    notes.append(f"removed retired {full} ({entry.reason})")
                else:
                    unresolved.append(
                        f"{full}={section[key]!r}: {entry.reason}; this version cannot "
                        "reproduce it"
                    )
    # Retired entries of list options had no effect, so dropping them is exact.
    for section_path, retired_values in RETIRED_VALUES.items():
        for path, section in _sections(data, section_path.split("."), ""):
            for key, value_entry in retired_values.items():
                values = section.get(key)
                if not isinstance(values, list):
                    continue
                dropped = [value for value in values if value in value_entry.values]
                if not dropped:
                    continue
                section[key] = [
                    value for value in values if value not in value_entry.values
                ]
                notes.append(
                    f"removed retired {', '.join(map(repr, dropped))} from "
                    f"{path}.{key} ({value_entry.reason})"
                )


def _write_unset_recurrent_counts(data: dict, notes: list[str]) -> None:
    """Make delayed population connections without synapses explicit lesions."""
    network = data.get("model", {}).get("core", {}).get("population_network")
    if not isinstance(network, dict):
        return
    for layer_idx, layer in enumerate(network.get("layers") or []):
        if not isinstance(layer, dict):
            continue
        populations = {
            str(pop.get("name")): pop
            for pop in layer.get("populations") or []
            if isinstance(pop, dict)
        }
        defaults = layer.get("population_defaults") or {}
        for connection in layer.get("connections") or []:
            if not isinstance(connection, dict) or connection.get("enabled") is False:
                continue
            if connection.get("timing", "same_step") != "delayed":
                continue
            target = populations.get(str(connection.get("target")))
            if target is None:
                continue
            pathway = connection.get("pathway")
            if not pathway:
                source = populations.get(str(connection.get("source")), {})
                polarity = str(source.get("polarity", "excitatory")).lower()
                pathway = (
                    "rec_inhibitory" if polarity == "inhibitory" else "rec_excitatory"
                )
            if pathway not in ("rec_excitatory", "rec_inhibitory"):
                continue
            overrides = target.get("population") or {}
            written = set(defaults) | set(overrides)
            if {f"{pathway}_synapses", f"{pathway}_synapses_by_level"} & written:
                continue
            target["population"] = {**overrides, f"{pathway}_synapses": 0}
            notes.append(
                f"model.core.population_network.layers[{layer_idx}] population "
                f"{target.get('name')}: wrote {pathway}_synapses: 0 (the delayed "
                f"connection from {connection.get('source')} reached no synapses)"
            )


def _builds_recurrent_einet(core: dict) -> bool | None:
    """Whether a structured core builds the recurrent EINet (None if unknown)."""
    from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_features import (
        _detect_recurrent_feature_flags,
        _resolve_recurrent_synapse_counts,
        _should_build_structured_recurrent_einet,
    )
    from dendritic_modeling.networks.architectures.recurrent.structured_recurrent_sections import (
        _parse_structured_recurrent_sections,
    )

    try:
        sections = _parse_structured_recurrent_sections(copy.deepcopy(core))
        counts = _resolve_recurrent_synapse_counts(
            connectivity=sections.connectivity,
            recurrent_cfg=sections.recurrent_cfg,
        )
        flags = _detect_recurrent_feature_flags(sections, counts)
    except (TypeError, ValueError):
        return None
    return bool(_should_build_structured_recurrent_einet(sections, flags))


def _layer_list(values: Any, n_layers: int) -> list:
    """Expand a per-layer setting (lists repeat their last entry) to n_layers."""
    if not isinstance(values, list):
        return [values] * n_layers
    if not values:
        return [0] * n_layers
    return [values[min(i, len(values) - 1)] for i in range(n_layers)]


def _reproduce_inhibitory_routing(
    data: dict, notes: list[str], unresolved: list[str]
) -> None:
    """Write the wiring the retired ``transfer.input_mode`` options selected.

    Before 2026-10-08 a feedforward EINet with ``input_mode: 1`` (the default)
    built none of the inhibitory cells it declared: first-layer excitatory
    cells read the transferred inhibitory stream and later layers had no
    inhibition. A recurrent EINet with ``input_mode: 0`` or
    ``inhibitory_mode: none`` ran first-layer excitatory cells without their
    ``ie`` synapses when that layer had no inhibitory cells. The rewritten
    config builds those networks under the current wiring rule, and the
    routing keys are removed; configurations the rule cannot express are
    reported.
    """
    from dendritic_modeling.config.inhibitory_wiring import (
        RETIRED_ROUTING_KEYS,
        check_retired_inhibitory_routing,
    )

    core = data.get("model", {}).get("core", {})
    if not isinstance(core, dict):
        return
    architecture = core.get("architecture")
    connectivity = core.get("connectivity")
    if not isinstance(architecture, dict) or not isinstance(connectivity, dict):
        return
    transfer = core.get("transfer")
    transfer = transfer if isinstance(transfer, dict) else {}
    routing = {key: transfer[key] for key in RETIRED_ROUTING_KEYS if key in transfer}
    recurrent = _builds_recurrent_einet(core)
    if recurrent is None:
        return
    n_layers = len(architecture.get("excitatory_layer_sizes") or [])
    mode = routing.get("input_mode", 1)
    source = (
        f"transfer.input_mode={mode}"
        if "input_mode" in routing
        else "the former default transfer.input_mode=1"
    )
    if not recurrent:
        builds = routing.get(
            "input_mode1_build_inhibitory_population",
            routing.get("build_inhibitory_population_for_input_mode1", False),
        )
        if mode == 1 and not builds:
            vision = core.get("vision")
            if str(core.get("type", "")).lower() == "dendritic_alexnet":
                if isinstance(vision, dict) and float(
                    vision.get("inhibitory_ratio", 0) or 0
                ):
                    notes.append(
                        f"model.core.vision.inhibitory_ratio: "
                        f"{vision['inhibitory_ratio']} -> 0 ({source} built no "
                        "inhibitory cells)"
                    )
                    vision["inhibitory_ratio"] = 0.0
                    # Every block is one layer whose cells no longer exist.
                    for name in (
                        "ei_synapses_per_branch_per_layer",
                        "ii_synapses_per_branch_per_layer",
                    ):
                        if any(
                            int(v or 0)
                            for v in _layer_list(connectivity.get(name, []), 1)
                        ):
                            notes.append(f"model.core.connectivity.{name}: -> [0]")
                            connectivity[name] = [0]
            sizes = list(architecture.get("inhibitory_layer_sizes") or [])
            if any(int(size or 0) > 0 for size in sizes):
                architecture["inhibitory_layer_sizes"] = [0] * n_layers
                notes.append(
                    "model.core.architecture.inhibitory_layer_sizes: "
                    f"{sizes} -> {[0] * n_layers} ({source} built no inhibitory "
                    "cells in a feedforward EINet)"
                )
    elif routing:
        inhibitory_mode = str(routing.get("inhibitory_mode", "first")).lower()
        recurrent_cfg = core.get("recurrent_ei")
        use_transfer = not (
            isinstance(recurrent_cfg, dict)
            and recurrent_cfg.get("use_transfer") is False
        )
        sizes = list(architecture.get("inhibitory_layer_sizes") or [])
        first_has_cells = bool(sizes) and int(sizes[0] or 0) > 0
        ie_name = "ie_synapses_per_branch_per_layer"
        ie = _layer_list(connectivity.get(ie_name, []), n_layers)
        if (
            use_transfer
            and n_layers
            and (mode == 0 or inhibitory_mode == "none")
            and not first_has_cells
            and int(ie[0] or 0)
        ):
            connectivity[ie_name] = [0, *ie[1:]]
            notes.append(
                f"model.core.connectivity.{ie_name}[0]: {ie[0]} -> 0 "
                f"({source}, inhibitory_mode={inhibitory_mode!r} ran first-layer "
                "excitatory cells without these synapses)"
            )
    if not routing:
        return
    try:
        check_retired_inhibitory_routing(core, recurrent=recurrent)
    except ValueError as error:
        unresolved.append(f"model.core.transfer {routing}: {error}")
        return
    for key in routing:
        del transfer[key]
    if not transfer and "transfer" in core:
        del core["transfer"]
    notes.append(
        f"removed retired model.core.transfer {routing} (the wiring is now written "
        "out in the synapse counts)"
    )


# Core types built with synapse_mode "mlp" (see the architecture factory).
_MLP_SYNAPSE_MODE_TYPES = frozenset({"dendritic_mlp", "flat_mlp"})


def _zero_inert_inhibitory_counts(data: dict, notes: list[str]) -> None:
    """Zero structured E/I inhibitory counts of layers without inhibitory cells."""
    core = data.get("model", {}).get("core", {})
    architecture = core.get("architecture")
    connectivity = core.get("connectivity")
    if not isinstance(architecture, dict) or not isinstance(connectivity, dict):
        return
    excitatory = list(architecture.get("excitatory_layer_sizes") or [])
    inhibitory = list(architecture.get("inhibitory_layer_sizes") or [])
    n_layers = len(excitatory)
    if not n_layers or len(inhibitory) > n_layers:
        return
    if inhibitory and len(inhibitory) < n_layers:
        # The recurrent EINet repeated the last entry for later layers; the
        # feedforward EINet built no inhibitory cells past the end of the list.
        recurrent = _builds_recurrent_einet(core)
        if recurrent is None:
            return
        fill = inhibitory[-1] if recurrent else 0
        inhibitory = inhibitory + [fill] * (n_layers - len(inhibitory))
        architecture["inhibitory_layer_sizes"] = inhibitory
        notes.append(
            f"model.core.architecture.inhibitory_layer_sizes: wrote {inhibitory} "
            "(one size per layer, as the "
            + ("recurrent" if recurrent else "feedforward")
            + " EINet built them)"
        )
    fields = {
        "ei": "ei_synapses_per_branch_per_layer",
        "ii": "ii_synapses_per_branch_per_layer",
        "ie": "ie_synapses_per_branch_per_layer",
        "rec_ei": "rec_ei_synapses_per_branch",
        "rec_ii": "rec_ii_synapses_per_branch",
        "rec_ie": "rec_ie_synapses_per_branch",
    }
    # The mlp synapse mode merges ee and ie into one signed bank of ee + ie
    # contacts, which exists whether or not the layer has inhibitory input.
    mlp_mode = str(core.get("type", "")).lower() in _MLP_SYNAPSE_MODE_TYPES
    for name, field_name in fields.items():
        values = connectivity.get(field_name)
        if not isinstance(values, list) or not values:
            continue
        # Lists repeat their last entry for later layers.
        expanded = [values[min(i, len(values) - 1)] for i in range(n_layers)]
        changed = False
        for layer in range(n_layers):
            has_cells = bool(inhibitory) and int(inhibitory[layer] or 0) > 0
            if has_cells or not expanded[layer]:
                continue
            if name == "ie" and layer == 0:
                continue  # a transfer layer may feed the first layer's stream
            if name == "ie" and mlp_mode:
                ee_name = "ee_synapses_per_branch_per_layer"
                ee = _layer_list(connectivity.get(ee_name, []), n_layers)
                ee[layer] = int(ee[layer] or 0) + int(expanded[layer])
                connectivity[ee_name] = ee
                notes.append(
                    f"model.core.connectivity.{field_name}[{layer}] moved into "
                    f"{ee_name}[{layer}] (the mlp synapse mode merges them into one "
                    "bank, and the layer has no inhibitory input)"
                )
            else:
                notes.append(
                    f"model.core.connectivity.{field_name}[{layer}]: wrote 0 (layer "
                    f"{layer} has no inhibitory cells, so these synapses had no "
                    "source or target)"
                )
            expanded[layer] = 0
            changed = True
        if changed:
            connectivity[field_name] = expanded


def _drop_derived_input_dims(
    data: dict,
    notes: list[str],
    sections: tuple[str, ...] = ("population_network", "unified_ei"),
) -> None:
    """Remove core input dimensions that are always set from the encoder."""
    core = data.get("model", {}).get("core", {})
    if not isinstance(core, dict):
        return
    for section in sections:
        payload = core.get(section)
        if isinstance(payload, dict) and "input_dim" in payload:
            value = payload.pop("input_dim")
            notes.append(
                f"removed model.core.{section}.input_dim={value!r} (it is set from "
                "the encoder output dimension)"
            )


def _drop_replaced_recurrent_baseline_settings(data: dict, notes: list[str]) -> None:
    """Remove recurrent-baseline settings that the build replaced.

    The payload ``input_dim`` is set from the encoder output dimension, and
    the ``baseline_rnn`` cell from the core type; since 2026-10-13 a written
    value that differs raises instead of being replaced.
    """
    _drop_derived_input_dims(
        data,
        notes,
        sections=("baseline_rnn", "heterogeneous_leak_ctrnn", "legendre_memory"),
    )
    core = data.get("model", {}).get("core", {})
    if not isinstance(core, dict):
        return
    baseline = core.get("baseline_rnn")
    if isinstance(baseline, dict) and "cell_type" in baseline:
        value = baseline.pop("cell_type")
        notes.append(
            f"removed model.core.baseline_rnn.cell_type={value!r} (model.core.type "
            "selects the cell)"
        )


def _fold_learned_output_scale(data: dict, notes: list[str]) -> None:
    """Replace the retired ``model.learned_output_scale`` by the mode it selected."""
    model = data.get("model")
    if not isinstance(model, dict) or "learned_output_scale" not in model:
        return
    learned = model.pop("learned_output_scale")
    mode = model.get("output_scale_mode")
    if mode is not None:
        notes.append(
            f"removed model.learned_output_scale={learned!r} "
            f"(model.output_scale_mode={mode!r} selected the output scale)"
        )
    elif not learned:
        model["output_scale_mode"] = "fixed"
        notes.append(
            "model.learned_output_scale=False -> model.output_scale_mode: fixed"
        )
    else:
        notes.append(
            "removed model.learned_output_scale=True (the default output scale)"
        )


def _drop_unread_io_params(data: dict, notes: list[str]) -> None:
    """Remove encoder/decoder params that their type does not read."""
    from dendritic_modeling.config.io_params import is_saved_record, unread_params
    from dendritic_modeling.config.keys import field_default
    from dendritic_modeling.config.model import (
        DecoderConfig,
        DecoderParamsConfig,
        EncoderConfig,
        EncoderParamsConfig,
    )

    model = data.get("model")
    if not isinstance(model, dict):
        return
    for kind, section_cls, params_cls in (
        ("encoder", EncoderConfig, EncoderParamsConfig),
        ("decoder", DecoderConfig, DecoderParamsConfig),
    ):
        section = model.get(kind)
        if not isinstance(section, dict) or not isinstance(section.get("params"), dict):
            continue
        params = section["params"]
        if is_saved_record(params, params_cls):
            continue  # a saved record lists every field; loading only warns
        type_name = section.get("type", field_default(section_cls, "type"))
        for name in unread_params(kind, type_name, params, params_cls):
            value = params.pop(name)
            notes.append(
                f"removed model.{kind}.params.{name}={value!r} (type "
                f"{type_name!r} does not read it)"
            )


def _drop_replaced_layer_indexed_seeds(data: dict, notes: list[str]) -> None:
    """Remove layer ``indexed_seed`` values that each population replaced.

    Before 2026-10-08 every population of a population-network layer drew its
    indexed topology from the experiment topology stream, whatever
    ``population_defaults.indexed_seed`` said; the current library uses that
    seed. Removing it keeps the topology the run had.
    """
    network = data.get("model", {}).get("core", {}).get("population_network")
    if not isinstance(network, dict):
        return
    for layer_idx, layer in enumerate(network.get("layers") or []):
        if not isinstance(layer, dict):
            continue
        defaults = layer.get("population_defaults")
        if not isinstance(defaults, dict) or "indexed_seed" not in defaults:
            continue
        populations = [p for p in layer.get("populations") or [] if isinstance(p, dict)]
        replaced = False
        for population in populations:
            overrides = population.get("population")
            overrides = overrides if isinstance(overrides, dict) else {}
            topk_type = str(overrides.get("topk_type", defaults.get("topk_type", "")))
            if (
                topk_type.lower().startswith("indexed")
                and "indexed_seed" not in overrides
            ):
                replaced = True
        if replaced:
            value = defaults.pop("indexed_seed")
            notes.append(
                f"removed model.core.population_network.layers[{layer_idx}]."
                f"population_defaults.indexed_seed={value!r} (its populations drew "
                "their topology from the experiment topology stream)"
            )


def _drop_unapplied_step_options(data: dict, notes: list[str]) -> None:
    """Remove step options (``use_amp``, ...) a strategy's loop never applied."""
    from dendritic_modeling.training.factory import (
        DEPRECATED_STRATEGY_ALIASES,
        TRAINER_STRATEGY_REGISTRY,
        _resolve_trainer_class,
    )
    from dendritic_modeling.training.strategies.recurrent_trainer import (
        RecurrentTrainer,
    )
    from dendritic_modeling.training.strategies.standard import STEP_OPTIONS

    training = data.get("training")
    if not isinstance(training, dict):
        return
    for phase_name, phase in training.items():
        if not isinstance(phase, dict) or not isinstance(phase.get("common"), dict):
            continue
        strategy = str(phase.get("strategy") or "standard").lower()
        strategy = DEPRECATED_STRATEGY_ALIASES.get(strategy, strategy)
        if strategy == "recurrent":
            trainer_cls = RecurrentTrainer
        elif strategy in TRAINER_STRATEGY_REGISTRY:
            trainer_cls = _resolve_trainer_class(
                TRAINER_STRATEGY_REGISTRY[strategy]["class"]
            )
        else:
            continue
        unapplied = getattr(trainer_cls, "unapplied_options", frozenset())
        common = phase["common"]
        for name in STEP_OPTIONS:
            if name in unapplied and common.get(name):
                value = common.pop(name)
                notes.append(
                    f"removed training.{phase_name}.common.{name}={value!r} (the "
                    f"{strategy} training loop does not apply it)"
                )


def _drop_unapplied_processing(data: dict, notes: list[str]) -> None:
    """Remove ``flatten``/``normalize: true`` that the dataset's loader ignored."""
    from dendritic_modeling.datasets.standard_datasets import _applied_processing

    section = data.get("data")
    if not isinstance(section, dict) or not section.get("dataset_name"):
        return
    name = str(section["dataset_name"])
    try:
        applied = _applied_processing(name)
    except (KeyError, ValueError):
        return
    if applied is None:
        return
    params = section.get("dataset_params")
    blocks = [("data.processing", section.get("processing"))]
    if isinstance(params, dict):
        blocks.append((f"data.dataset_params.{name}", params.get(name)))
    for path, block in blocks:
        if not isinstance(block, dict):
            continue
        for key in ("flatten", "normalize"):
            if block.get(key) is True and key not in applied:
                del block[key]
                notes.append(
                    f"removed {path}.{key}=True (the {name} loader does not apply it)"
                )


def _write_default_unified_inhibitory(data: dict, notes: list[str]) -> None:
    """Write the inhibitory population a unified E/I layer got by default.

    A layer without an ``inhibitory`` entry built EILayerConfig's default
    population before 2026-10-08; the current library asks for the entry.
    """
    from dendritic_modeling.networks.architectures.recurrent.ei_config import (
        EILayerConfig,
        PopulationConfig,
    )

    core = data.get("model", {}).get("core", {})
    if not isinstance(core, dict):
        return
    field = EILayerConfig.__dataclass_fields__["inhibitory"]
    if field.default_factory is dataclasses.MISSING:
        return
    default = field.default_factory()
    written = (
        None
        if default is None
        else {
            item.name: getattr(default, item.name)
            for item in dataclasses.fields(PopulationConfig)
            if getattr(default, item.name) != getattr(PopulationConfig(), item.name)
        }
    )
    for section in ("unified_ei", "ei_unified"):
        payload = core.get(section)
        if not isinstance(payload, dict):
            continue
        for layer_idx, layer in enumerate(payload.get("layers") or []):
            if isinstance(layer, dict) and "inhibitory" not in layer:
                layer["inhibitory"] = copy.deepcopy(written)
                notes.append(
                    f"model.core.{section}.layers[{layer_idx}].inhibitory: wrote "
                    f"{written} (the default population a layer without the "
                    "entry built)"
                )


def _zero_unsourced_unified_counts(data: dict, notes: list[str]) -> None:
    """Zero unified E/I inhibitory counts that had no input, so built no synapses.

    The library raises for them since 2026-10-13.
    """
    from dendritic_modeling.networks.architectures.factory_unified import (
        unsourced_unified_counts,
    )

    core = data.get("model", {}).get("core", {})
    if not isinstance(core, dict):
        return
    for section in ("unified_ei", "ei_unified"):
        payload = core.get(section)
        if not isinstance(payload, dict):
            continue
        layers = payload.get("layers") or []
        if not all(
            isinstance(layer, dict) and isinstance(layer.get("excitatory"), dict)
            for layer in layers
        ):
            continue  # the shorthand layer form is rewritten when it is built
        for index, role, key, reason in unsourced_unified_counts(
            layers, use_transfer=bool(payload.get("use_transfer", False))
        ):
            population = layers[index][role]
            value = population[key]
            population[key] = [0] * len(value) if isinstance(value, list) else 0
            notes.append(
                f"model.core.{section}.layers[{index}].{role}.{key}: wrote "
                f"{population[key]} instead of {value!r} ({reason}, so the run "
                "built no such synapses)"
            )


def _freeze_quantile_gates_without_recalibration(data: dict, notes: list[str]) -> None:
    """``quantile`` gate updates without recalibration kept the gates frozen."""
    from dendritic_modeling.config.reactivation import (
        normalize_reactivation_update_mode,
    )

    training = data.get("training")
    if not isinstance(training, dict):
        return
    for phase_name, phase in training.items():
        common = phase.get("common") if isinstance(phase, dict) else None
        if not isinstance(common, dict) or "reactivation_update_mode" not in common:
            continue
        try:
            mode = normalize_reactivation_update_mode(
                common["reactivation_update_mode"]
            )
        except ValueError:
            continue
        if mode == "quantile" and not int(
            common.get("recalibrate_reactivation_every") or 0
        ):
            common["reactivation_update_mode"] = "frozen"
            notes.append(
                f"training.{phase_name}.common.reactivation_update_mode: quantile -> "
                "frozen (without recalibrate_reactivation_every the gates never "
                "changed)"
            )


def _fix_output_scale_without_identity_decoder(data: dict, notes: list[str]) -> None:
    """A learned output-scale mode gave a non-identity decoder no scale."""
    from dendritic_modeling.config.model import resolve_output_scale_mode

    model = data.get("model")
    if not isinstance(model, dict) or model.get("output_scale_mode") is None:
        return
    decoder = model.get("decoder")
    decoder_type = decoder.get("type") if isinstance(decoder, dict) else None
    if str(decoder_type or "").strip().lower() in ("identity", "none"):
        return
    try:
        mode = resolve_output_scale_mode(model["output_scale_mode"])
    except ValueError:
        return
    if decoder_type is not None and mode != "fixed":
        notes.append(
            f"model.output_scale_mode: {mode} -> fixed (a {decoder_type} decoder "
            "gets no learned output scale)"
        )
        model["output_scale_mode"] = "fixed"


def _drop_feedforward_recurrent_settings(data: dict, notes: list[str]) -> None:
    """Remove ``recurrent_ei`` settings that a feedforward EINet never read."""
    core = data.get("model", {}).get("core", {})
    if not isinstance(core, dict) or not isinstance(core.get("recurrent_ei"), dict):
        return
    if not isinstance(core.get("architecture"), dict) or not isinstance(
        core.get("connectivity"), dict
    ):
        return
    if _builds_recurrent_einet(core) is not False:
        return
    recurrent = core["recurrent_ei"]
    for key in sorted(recurrent):
        value = recurrent[key]
        if value in (None, [], {}) or (key == "enabled" and not value):
            continue
        del recurrent[key]
        notes.append(
            f"removed model.core.recurrent_ei.{key}={value!r} (a feedforward EINet "
            "does not read it)"
        )


def _drop_additive_options_of_shunting_layers(data: dict, notes: list[str]) -> None:
    """Remove additive-integration options that shunting layers ignored."""
    from dendritic_modeling.config.model_aliases import (
        get_core_morphology_alias_overrides,
    )

    def inert(key: str, value: Any) -> bool:
        if key == "additive_mode":
            return value is not None and str(value).strip().lower() != "raw"
        return bool(value)

    core = data.get("model", {}).get("core", {})
    if not isinstance(core, dict):
        return
    morphology = core.get("morphology")
    if isinstance(morphology, dict) and isinstance(core.get("architecture"), dict):
        aliases = get_core_morphology_alias_overrides(str(core.get("type", "")))
        shunting = aliases.get("use_shunting", morphology.get("use_shunting", True))
        for key in ("additive_mode", "use_additive_normalization"):
            if shunting is not False and inert(key, morphology.get(key)):
                value = morphology.pop(key)
                notes.append(
                    f"removed model.core.morphology.{key}={value!r} (shunting "
                    "layers do not use it)"
                )
    network = core.get("population_network")
    if not isinstance(network, dict):
        return
    for layer_idx, layer in enumerate(network.get("layers") or []):
        if not isinstance(layer, dict):
            continue
        defaults = layer.get("population_defaults")
        defaults = defaults if isinstance(defaults, dict) else {}
        populations = [p for p in layer.get("populations") or [] if isinstance(p, dict)]
        for key, neutral in (
            ("additive_mode", "raw"),
            ("use_additive_normalization", False),
        ):
            for population in populations:
                overrides = population.get("population")
                overrides = overrides if isinstance(overrides, dict) else {}
                shunting = overrides.get(
                    "use_shunting", defaults.get("use_shunting", True)
                )
                value = overrides.get(key, defaults.get(key))
                if shunting is False or not inert(key, value):
                    continue
                if key in overrides:
                    del overrides[key]
                else:  # the layer default; other populations may use it
                    overrides[key] = neutral
                    population["population"] = overrides
                notes.append(
                    f"model.core.population_network.layers[{layer_idx}] population "
                    f"{population.get('name')}: {key}={value!r} is not used by a "
                    f"shunting population (now {overrides.get(key, 'unset')!r})"
                )


_UNMODELED_FEATURE_FIELDS = frozenset(
    {
        "cross_level_mode",
        "dynamics_mode",
        "dendritic_spikes_enabled",
        "soma_feedback_enabled",
        "synapse_types",
    }
)


def _unmodeled_local_features(core: dict) -> list[str]:
    """Forward features of a core that the local rules do not model."""
    populations: list[dict] = []
    network = core.get("population_network")
    if isinstance(network, dict):
        for layer in network.get("layers") or []:
            if not isinstance(layer, dict):
                continue
            defaults = layer.get("population_defaults")
            defaults = defaults if isinstance(defaults, dict) else {}
            for population in layer.get("populations") or []:
                overrides = (
                    population.get("population")
                    if isinstance(population, dict)
                    else None
                )
                populations.append(
                    {**defaults, **(overrides if isinstance(overrides, dict) else {})}
                )
    for section in ("unified_ei", "ei_unified"):
        payload = core.get(section)
        for layer in (payload.get("layers") or []) if isinstance(payload, dict) else []:
            for kind in ("excitatory", "inhibitory"):
                if isinstance(layer, dict) and isinstance(layer.get(kind), dict):
                    populations.append(layer[kind])
    features = set()
    for population in populations:
        mode = str(population.get("cross_level_mode", "hierarchical")).lower()
        if mode in ("parallel_readout", "all_active_star"):
            features.add(f"cross_level_mode {mode!r}")
        if str(population.get("dynamics_mode", "rate")).lower() == "spike":
            features.add("a spiking soma")
        if population.get("dendritic_spikes_enabled"):
            features.add("dendritic spikes")
        if population.get("soma_feedback_enabled"):
            features.add("soma feedback")
        types = population.get("synapse_types")
        if isinstance(types, dict) and types.get("enabled"):
            features.add("typed synapses")
    dynamics = core.get("dynamics", core.get("spiking"))
    if (
        isinstance(dynamics, dict)
        and str(dynamics.get("mode", "rate")).lower() == "spike"
    ):
        features.add("a spiking soma")
    for section, label in (
        ("dendritic_spikes", "dendritic spikes"),
        ("soma_feedback", "soma feedback"),
        ("synapse_types", "typed synapses"),
    ):
        block = core.get(section)
        if isinstance(block, dict) and block.get("enabled"):
            features.add(label)
    return sorted(features)


def _allow_unmodeled_local_features(data: dict, notes: list[str]) -> None:
    """local_ca once applied its rules to forward features they do not model."""
    training = data.get("training")
    core = data.get("model", {}).get("core")
    if not isinstance(training, dict) or not isinstance(core, dict):
        return
    features = None
    for phase_name, phase in training.items():
        if not isinstance(phase, dict):
            continue
        if str(phase.get("strategy") or "").lower() != "local_ca":
            continue
        if features is None:
            features = _unmodeled_local_features(core)
        if not features:
            return
        rules = phase.get("learning_strategy_config")
        rules = rules if isinstance(rules, dict) else {}
        if "allow_unmodeled_forward_features" in rules:
            continue
        rules["allow_unmodeled_forward_features"] = True
        phase["learning_strategy_config"] = rules
        notes.append(
            f"training.{phase_name}.learning_strategy_config."
            "allow_unmodeled_forward_features: true (local_ca applied its rules to "
            f"a core with {', '.join(features)}, which they do not model)"
        )


def _drop_unread_population_sparse_options(data: dict, notes: list[str]) -> None:
    """Remove sparse-synapse options that a population's synapse type ignored."""
    from dendritic_modeling.config.conversion import normalize_sparsity_type
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_synapses import (
        _DENSE_TO_SPARSE_TYPES,
        _INDEXED_EXECUTION_FIELDS,
        _INDEXED_FIELD_TYPES,
    )
    from dendritic_modeling.networks.architectures.recurrent.ei_config import (
        PopulationConfig,
    )

    defaults_of = {item.name: item for item in dataclasses.fields(PopulationConfig)}

    def readers(key: str):
        if key.startswith("dense_to_sparse_"):
            return _DENSE_TO_SPARSE_TYPES
        field = key[len("indexed_") :] if key.startswith("indexed_") else key
        if field in _INDEXED_EXECUTION_FIELDS:
            return None  # execution settings are only logged
        return _INDEXED_FIELD_TYPES.get(field)

    def is_default(key: str, value: Any) -> bool:
        item = defaults_of.get(key)
        if item is None or item.default is dataclasses.MISSING:
            return False
        return value == item.default

    network = data.get("model", {}).get("core", {}).get("population_network")
    if not isinstance(network, dict):
        return
    for layer_idx, layer in enumerate(network.get("layers") or []):
        if not isinstance(layer, dict):
            continue
        defaults = layer.get("population_defaults")
        defaults = defaults if isinstance(defaults, dict) else {}
        for population in layer.get("populations") or []:
            if not isinstance(population, dict):
                continue
            overrides = population.get("population")
            overrides = overrides if isinstance(overrides, dict) else {}
            merged = {**defaults, **overrides}
            try:
                topk_type = normalize_sparsity_type(merged.get("topk_type", "standard"))
            except ValueError:
                continue
            for key, value in merged.items():
                types = readers(str(key))
                if types is None or topk_type in types or is_default(key, value):
                    continue
                if key in overrides:
                    del overrides[key]
                else:  # a layer default that other populations may read
                    overrides[key] = defaults_of[key].default
                    population["population"] = overrides
                notes.append(
                    f"model.core.population_network.layers[{layer_idx}] population "
                    f"{population.get('name')}: {key}={value!r} is not read by "
                    f"synapse type {topk_type!r} (now the default)"
                )


def _allow_path_scalar_fallback(data: dict, notes: list[str]) -> None:
    """Path-based local errors once fell back to the scalar error silently."""
    training = data.get("training")
    if not isinstance(training, dict):
        return
    for phase_name, phase in training.items():
        if not isinstance(phase, dict):
            continue
        if str(phase.get("strategy") or "").lower() != "local_ca":
            continue
        rules = phase.get("learning_strategy_config")
        if not isinstance(rules, dict) or "allow_scalar_fallback" in rules:
            continue
        mode = str(rules.get("error_broadcast_mode", "scalar")).lower()
        morphology = rules.get("morphology_aware")
        propagation = isinstance(morphology, dict) and bool(
            morphology.get("use_path_propagation")
        )
        if mode in ("path_transport", "pathway_vector") or propagation:
            rules["allow_scalar_fallback"] = True
            notes.append(
                f"training.{phase_name}.learning_strategy_config."
                "allow_scalar_fallback: true (path-based errors used the scalar "
                "error where a model has no path to one soma)"
            )


def _drop_morphology_overridden_by_alias(data: dict, notes: list[str]) -> None:
    """Remove morphology values that the core-type alias replaced."""
    from dendritic_modeling.config.model_aliases import (
        get_core_morphology_alias_overrides,
    )

    core = data.get("model", {}).get("core", {})
    if not isinstance(core, dict) or not isinstance(core.get("morphology"), dict):
        return
    core_type = str(core.get("type", "")).lower()
    morphology = core["morphology"]
    for key, alias_value in get_core_morphology_alias_overrides(core_type).items():
        written = morphology.get(key)
        if written is not None and written != alias_value:
            del morphology[key]
            notes.append(
                f"removed model.core.morphology.{key}={written!r} (core type "
                f"{core_type!r} used {alias_value!r})"
            )


def _drop_unread_core_sections(data: dict, notes: list[str]) -> None:
    """Remove ``model.core`` sections that the selected core type never reads.

    Population networks, unified E/I cores and the other payload families read
    only their own section; structured EINet cores ignore the payload sections
    of the other families. Such a section never changed the built model, and
    the loader rejects it when it holds non-default values.
    """
    from dendritic_modeling.config import CoreConfig
    from dendritic_modeling.config.core_sections import (
        COMMON_CORE_FIELDS,
        EINET_FOREIGN_PAYLOADS,
        core_family,
    )
    from dendritic_modeling.config.recurrence import _STRUCTURED_EI_TYPES

    core = data.get("model", {}).get("core")
    if not isinstance(core, dict):
        return
    core_type = str(core.get("type") or CoreConfig().type).strip().lower()
    family = core_family(core_type)
    if family is None and core_type not in _STRUCTURED_EI_TYPES:
        return  # other builders read sections that are not declared per type
    fields = {field.name for field in dataclasses.fields(CoreConfig)}
    for name in sorted(set(core) & fields - COMMON_CORE_FIELDS):
        if family is not None:
            unread = name != family
        else:
            unread = name in EINET_FOREIGN_PAYLOADS
        if unread:
            del core[name]
            notes.append(
                f"removed model.core.{name} (model.core.type={core_type!r} "
                "does not read it)"
            )


def _core_setting_names() -> frozenset[str]:
    """Every field name that can appear under ``model.core``, at any depth."""
    from dendritic_modeling.config import CoreConfig

    names: set[str] = set()

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                names.add(str(key))
                walk(item)

    walk(dataclasses.asdict(CoreConfig()))
    return frozenset(names)


def _move_analysis_annotations(
    data: dict, notes: list[str], unresolved: list[str]
) -> None:
    """Move annotation blocks written inside ``analysis`` under ``metadata``.

    A key that is no analyzer, analysis field or alias was ignored; one that
    nearly matches an analyzer name may be a typo and is reported instead.
    """
    import difflib

    from dendritic_modeling.config.analysis import AnalysisConfig
    from dendritic_modeling.config.config import _analysis_field_mapping

    analysis = data.get("analysis")
    if not isinstance(analysis, dict):
        return
    from dendritic_modeling.config.extensions import registered_named_sections
    from dendritic_modeling.plugins import load_plugins

    load_plugins()
    valid = (
        set(_analysis_field_mapping())
        | {item.name for item in dataclasses.fields(AnalysisConfig)}
        | set(registered_named_sections("analysis"))
        # Sections of the replacement plugin, kept even where it is not installed.
        | {
            "runtime",
            "vision_boundary_diagnostics",
            "vision_boundary_diagnostics_analysis",
        }
    )
    for key in [key for key in analysis if str(key) not in valid]:
        if str(key).startswith("_"):
            continue
        near = difflib.get_close_matches(str(key), sorted(valid), n=1, cutoff=0.85)
        if near:
            unresolved.append(
                f"analysis.{key}: not an analysis section; did you mean {near[0]!r}?"
            )
            continue
        metadata = data.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        block = metadata.setdefault("analysis", {})
        if not isinstance(block, dict) or key in block:
            unresolved.append(f"analysis.{key}: metadata.analysis already holds it")
            continue
        block[key] = analysis.pop(key)
        data["metadata"] = metadata
        notes.append(
            f"moved analysis.{key} under metadata.analysis (no analyzer reads it)"
        )


def _config_sections() -> set[str]:
    from dendritic_modeling.config.config import Config

    return {item.name for item in dataclasses.fields(Config)}


# Rules whose rewrite depends on other settings: (rule, sections it reads,
# the setting names within them that it reads). A sweep axis or variant that
# overrides one of those settings can make the rewrite of base_config wrong for
# some generated configs, so the rule is then reported instead of applied.
_POPULATION_FIELDS = frozenset(
    {
        "layers",
        "populations",
        "population",
        "population_defaults",
        "connections",
        "name",
        "polarity",
        "source",
        "target",
        "timing",
        "pathway",
        "enabled",
        "rec_excitatory_synapses",
        "rec_inhibitory_synapses",
        "rec_excitatory_synapses_by_level",
        "rec_inhibitory_synapses_by_level",
    }
)
_STRUCTURED_FIELDS = frozenset(
    {
        "type",
        "architecture",
        "connectivity",
        "excitatory_layer_sizes",
        "inhibitory_layer_sizes",
        "ee_synapses_per_branch_per_layer",
        "ei_synapses_per_branch_per_layer",
        "ie_synapses_per_branch_per_layer",
        "ii_synapses_per_branch_per_layer",
        "rec_ee_synapses_per_branch",
        "rec_ei_synapses_per_branch",
        "rec_ie_synapses_per_branch",
        "rec_ii_synapses_per_branch",
        "recurrent_ei",
        "enabled",
        "use_transfer",
        "dynamics",
        "spiking",
        "mode",
        "soma_feedback",
        "dendritic_spikes",
        "synapse_types",
    }
)


# Settings the structured-connectivity rule reads (config_version 2 to 3).
_STRUCTURED_RULE_FIELDS = frozenset(
    {
        "structured",
        "structured_connectivity",
        "layers",
        "population_defaults",
        "populations",
        "population",
        "excitatory",
        "inhibitory",
        "enabled",
        "pathways",
        "method",
        "seed",
        "probability",
        "p_connect",
        "indegree",
        "in_degree",
        "distance_kernel",
        "kernel",
        "distance_sigma",
        "sigma",
        "base_probability",
        "pre_positions",
        "post_positions",
        "spatial",
        "ee",
        "ie",
        "ei",
        "ii",
        "rec_ee",
        "rec_ie",
        "rec_ei",
        "rec_ii",
        "e_to_e",
        "exc_to_exc",
        "excitatory_to_excitatory",
        "i_to_e",
        "inh_to_exc",
        "inhibitory_to_excitatory",
        "e_to_i",
        "exc_to_inh",
        "excitatory_to_inhibitory",
        "i_to_i",
        "inh_to_inh",
        "inhibitory_to_inhibitory",
        "recurrent_ee",
        "rec_e_to_e",
        "recurrent_ie",
        "rec_i_to_e",
        "recurrent_ei",
        "rec_e_to_i",
        "recurrent_ii",
        "rec_i_to_i",
    }
)


def _rule_reads(key: str, sections: tuple[str, ...], fields: frozenset[str]) -> bool:
    """Whether overriding the dotted ``key`` changes a setting a rule reads."""
    for section in sections:
        if section == key or section.startswith(f"{key}."):
            return True  # the override replaces the whole section
        if key.startswith(f"{section}."):
            last = key.rsplit(".", 1)[-1]
            if last in fields or last.isdigit():
                return True
    return False


# Largest number of sweep conditions a base_config rewrite is checked on.
_MAX_CHECKED_ARMS = 4096


def _set_dotted(data: Any, key: str, value: Any) -> None:
    """Set a sweep override (numeric components index lists), as sweeps do."""
    parts = key.split(".")
    target = data
    for index, part in enumerate(parts[:-1]):
        if isinstance(target, list):
            target = target[int(part)]
            continue
        if target.get(part) is None:
            target[part] = [] if parts[index + 1].isdigit() else {}
        target = target[part]
    if isinstance(target, list):
        target[int(parts[-1])] = copy.deepcopy(value)
    else:
        target[parts[-1]] = copy.deepcopy(value)


def _sweep_conditions(
    sweep: dict, relevant: Callable[[str], bool]
) -> list[list[tuple[str, Any]]] | None:
    """Override lists of the sweep's conditions, restricted to relevant keys.

    A variant's overrides apply first and the Cartesian axes on top, as in
    sweep generation. Returns None when there are too many to check.
    """
    import itertools
    import math

    from dendritic_modeling.scripts.sweeps.unified.sweep_types.unified_sweep import (
        expand_axis,
    )

    axes = {
        str(key): expand_axis(value, name=str(key))
        for key, value in (sweep.get("sweep_config") or {}).items()
        if relevant(str(key))
    }
    variants = [
        [
            (str(key), value)
            for key, value in (variant.get("overrides") or {}).items()
            if relevant(str(key))
        ]
        for variant in sweep.get("sweep_variants") or []
        if isinstance(variant, dict)
    ] or [[]]
    count = len(variants) * math.prod(len(values) for values in axes.values())
    if count > _MAX_CHECKED_ARMS:
        return None
    return [
        [*variant, *zip(axes, combination)]
        for variant in variants
        for combination in itertools.product(*axes.values())
    ]


def _rewrite_holds(
    rule: Callable, before: dict, after: dict, conditions: list[list[tuple[str, Any]]]
) -> bool:
    """Whether rewriting base_config equals rewriting every generated config."""
    for overrides in conditions:
        expected, produced = copy.deepcopy(before), copy.deepcopy(after)
        try:
            for key, value in overrides:
                _set_dotted(expected, key, value)
                _set_dotted(produced, key, value)
            _run_rule(rule, expected)
        except (IndexError, KeyError, TypeError, ValueError, AttributeError):
            return False
        if expected != produced:
            return False
    return True


def _run_rule(rule: Callable, data: dict) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    unresolved: list[str] = []
    if rule in (
        _reproduce_inhibitory_routing,
        _move_analysis_annotations,
        _write_historical_stochastic_defaults,
        _write_version2_structured_meanings,
    ):
        rule(data, notes, unresolved)
    else:
        rule(data, notes)
    return notes, unresolved


def _stochastic_population_options(data: dict) -> Iterable[tuple[str, dict, dict]]:
    """Population option mappings: (path, options, layer defaults)."""
    core = data.get("model", {}).get("core", {})
    if not isinstance(core, dict):
        return
    network = core.get("population_network")
    if isinstance(network, dict):
        for index, layer in enumerate(network.get("layers") or []):
            if not isinstance(layer, dict):
                continue
            defaults = layer.get("population_defaults")
            defaults = defaults if isinstance(defaults, dict) else {}
            base = f"model.core.population_network.layers[{index}]"
            yield f"{base}.population_defaults", defaults, {}
            for position, entry in enumerate(layer.get("populations") or []):
                if isinstance(entry, dict) and isinstance(
                    entry.get("population"), dict
                ):
                    yield (
                        f"{base}.populations[{position}].population",
                        entry["population"],
                        defaults,
                    )
    for name in ("unified_ei", "ei_unified"):
        unified = core.get(name)
        if not isinstance(unified, dict):
            continue
        for index, layer in enumerate(unified.get("layers") or []):
            if not isinstance(layer, dict):
                continue
            for role in ("excitatory", "inhibitory"):
                if isinstance(layer.get(role), dict):
                    yield f"model.core.{name}.layers[{index}].{role}", layer[role], {}


# Structured-connectivity rules (config_version 2 to 3) ----------------------

_V2_RANDOM_MASK_METHODS = frozenset(
    {
        "bernoulli",
        "probability",
        "probabilistic",
        "fixed_indegree",
        "fixed_in_degree",
        "indegree",
        "distance",
        "distance_kernel",
    }
)
_V2_SPATIAL_METHODS = frozenset({"spatial_morphology", "morphology_spatial"})
_V2_PATHWAYS = ("ee", "ie", "ei", "ii", "rec_ee", "rec_ie", "rec_ei", "rec_ii")
# The root a typed block always had, before and after config_version 3.
_TYPED_ROOT_V2 = {
    "method": "bernoulli",
    "probability": None,
    "indegree": None,
    "distance_kernel": "none",
    "distance_sigma": 1.0,
    "seed": None,
    "spatial": {},
}
_TYPED_ROOT_V3 = {**_TYPED_ROOT_V2, "distance_kernel": "gaussian"}


def _v2_rule(block: dict, pathway: str) -> tuple[dict, dict] | None:
    """The merged rule and pathway entry of one pathway, read as version 2 did."""
    from dendritic_modeling.config.structured_connectivity import (
        MASK_RULE_KEYS,
        normalize_pathway_name,
    )

    if not bool(block.get("enabled", False)):
        return None
    root = {k: v for k, v in block.items() if k not in {"pathways", "enabled"}}
    entry: dict = {}
    for name, value in (block.get("pathways") or {}).items():
        if normalize_pathway_name(name) == pathway:
            entry = dict(value or {})
            break
    root_rule = any(root.get(key) is not None for key in MASK_RULE_KEYS) or bool(
        root.get("spatial") or {}
    )
    if not entry and not root_rule:
        return None
    return {**root, **entry}, entry


def _seed_meaning(seed: Any, *, from_entry: bool, filled: bool, spatial: bool):
    """What a structured seed makes of the masks of a training run."""
    if seed is None:
        if filled and not from_entry:
            return "topology seed"
        return 0 if spatial else "global random stream"
    if seed == "global_rng":
        return "global random stream"
    return int(seed)


def _rule_meaning(rule: dict, entry: dict, *, version: int, typed: bool) -> tuple:
    """The settings one pathway's masks follow, as ``version`` reads them."""
    method = str(rule.get("method", "bernoulli")).lower()
    if method in {"none", "dense", "all"}:
        return ("dense",)
    if method in {"index_manifest", "manifest"}:
        return ("manifest", rule.get("manifest_path"))
    if method in {"level_feature_blocks", "level_blocks"}:
        return ("level blocks", rule.get("ranges_by_level"))
    if method in {"inventory_feature_blocks", "inventory_blocks"}:
        keys = ("inventory_counts", "feature_ranges", "tier_groups", "owner_count")
        return ("inventory blocks", *(rule.get(key) for key in keys))
    # A typed root is seeded by training in every version; population rules
    # are from version 3 on. A pathway entry's own seed is never filled.
    from_entry = "seed" in entry
    seed = _seed_meaning(
        rule.get("seed"),
        from_entry=from_entry,
        filled=typed or version >= 3,
        spatial=method in _V2_SPATIAL_METHODS,
    )
    if method in _V2_SPATIAL_METHODS:
        return ("spatial", rule.get("spatial"), seed)
    if version >= 3:
        rule = {**rule}  # canonical names, see resolve_pathway_rule
        probability, indegree = rule.get("probability"), rule.get("indegree")
        kernel = rule.get("distance_kernel", "gaussian")
        sigma = rule.get("distance_sigma", 1.0)
        base_key = "base_probability" if "base_probability" in rule else "probability"
        base = rule.get(base_key, 1.0)
    else:
        probability = rule.get("probability", rule.get("p_connect"))
        indegree = rule.get("indegree", rule.get("in_degree"))
        kernel = rule.get("distance_kernel", rule.get("kernel", "gaussian"))
        sigma = rule.get("distance_sigma", rule.get("sigma", 1.0))
        base = rule.get("base_probability", rule.get("probability", 1.0))
    if method in {"bernoulli", "probability", "probabilistic"}:
        return ("bernoulli", probability, seed)
    if method in {"fixed_indegree", "fixed_in_degree", "indegree"}:
        return ("fixed indegree", indegree, seed)
    if method in {"distance", "distance_kernel"}:
        positions = (rule.get("pre_positions"), rule.get("post_positions"))
        return ("distance", positions, str(kernel), sigma, base, seed)
    return ("unknown method", method)


def _pathway_meanings(block: dict, *, version: int, typed: bool) -> dict:
    """The meaning of every pathway of a structured block in ``version``."""
    from dendritic_modeling.config.structured_connectivity import (
        RULE_KEY_ALIASES,
        normalize_pathway_name,
        resolve_pathway_rule,
    )

    if typed:
        # The typed root carries its defaults; a root alias (which version 2
        # rejected) loads as its key.
        defaults = _TYPED_ROOT_V3 if version >= 3 else _TYPED_ROOT_V2
        block = {
            **defaults,
            **{RULE_KEY_ALIASES.get(key, key): value for key, value in block.items()},
        }
    meanings = {}
    for pathway in _V2_PATHWAYS:
        if version >= 3:
            rule = resolve_pathway_rule(block, pathway)
            if not rule:
                meanings[pathway] = None
                continue
            entry = {}
            for name, value in (block.get("pathways") or {}).items():
                if normalize_pathway_name(name) == pathway:
                    entry = dict(value or {})
                    break
        else:
            resolved = _v2_rule(block, pathway)
            if resolved is None:
                meanings[pathway] = None
                continue
            rule, entry = resolved
        meanings[pathway] = _rule_meaning(rule, entry, version=version, typed=typed)
    return meanings


def _drop_second_spellings(rule: dict, path: str, notes: list[str]) -> None:
    """Drop the alias of a key a mapping also writes canonically (version 2 read that)."""
    from dendritic_modeling.config.structured_connectivity import RULE_KEY_ALIASES

    for alias, canonical in RULE_KEY_ALIASES.items():
        if alias in rule and canonical in rule:
            value = rule.pop(alias)
            notes.append(
                f"{path}.{alias}: removed {value!r} (the mapping also sets "
                f"{canonical}, which was read)"
            )


def _clean_structured_block(block: dict, path: str, notes: list[str]) -> None:
    """Rewrite what one written block repeats: spellings and pathway entries."""
    from dendritic_modeling.config.structured_connectivity import normalize_pathway_name

    _drop_second_spellings(block, path, notes)
    pathways = block.get("pathways")
    if not isinstance(pathways, dict):
        return
    first: dict[str, str] = {}
    for name in list(pathways):
        canonical = normalize_pathway_name(name)
        if canonical in first:
            pathways.pop(name)
            notes.append(
                f"{path}.pathways.{name}: removed (it names pathway {canonical!r}, "
                f"whose first entry {first[canonical]!r} was read)"
            )
            continue
        first[canonical] = str(name)
        if isinstance(pathways[name], dict):
            _drop_second_spellings(pathways[name], f"{path}.pathways.{name}", notes)


def _upgrade_typed_structured_block(block: dict, path: str, notes: list[str]) -> None:
    """Write the version-2 meaning into a ``model.core.connectivity.structured`` block.

    Its root always had ``distance_kernel: none``, ``distance_sigma: 1.0``,
    ``probability: null`` and ``indegree: null``, which the second spellings
    in a pathway entry never overrode.
    """
    from dendritic_modeling.config.structured_connectivity import (
        DISTANCE_METHODS,
        MASK_RULE_KEYS,
        RULE_KEY_ALIASES,
    )

    _clean_structured_block(block, path, notes)
    root_method = str(block.get("method", "bernoulli")).lower()
    root_rule = any(block.get(key) is not None for key in MASK_RULE_KEYS) or bool(
        block.get("spatial") or {}
    )
    root_kernel = root_method in DISTANCE_METHODS and root_rule
    pathways = block.get("pathways")
    for name, entry in (pathways if isinstance(pathways, dict) else {}).items():
        if not isinstance(entry, dict):
            continue
        for alias, canonical in RULE_KEY_ALIASES.items():
            if alias in entry:
                value = entry.pop(alias)
                notes.append(
                    f"{path}.pathways.{name}.{alias}: removed {value!r} (the "
                    f"block's {canonical}="
                    f"{block.get(canonical, _TYPED_ROOT_V2[canonical])!r} was read)"
                )
        method = str(entry.get("method", root_method)).lower()
        if method in DISTANCE_METHODS and "distance_kernel" not in entry:
            root_kernel = True
    if root_kernel and "distance_kernel" not in block:
        block["distance_kernel"] = "none"
        notes.append(
            f"{path}.distance_kernel: wrote 'none' (distance rules without a "
            "kernel used it before config_version 3)"
        )


def _upgrade_population_structured_blocks(
    blocks: list[tuple[str, dict]],
    populations: list[tuple[str, tuple[int | None, int | None]]],
    notes: list[str],
) -> None:
    """Write the version-2 meaning into the structured rules of some populations.

    ``blocks`` are the written rule mappings with their paths; each entry of
    ``populations`` names the rule it reads as indices into ``blocks``: the
    layer default (``None`` when absent) and its own rule. A pathway alias
    that a canonical root key shadowed is removed where it is written, and a
    rule without a seed gets ``seed: global_rng`` (its random masks drew from
    the global random stream) or ``seed: 0`` (spatial contacts used seed 0)
    where its seed is read from.
    """
    from dendritic_modeling.config.conversion import deep_merge_dicts
    from dendritic_modeling.config.structured_connectivity import RULE_KEY_ALIASES

    for path, block in blocks:
        if isinstance(block, dict):
            _clean_structured_block(block, path, notes)
    seeds: dict[int, set] = {}
    for _, (default_index, own_index) in populations:
        default = blocks[default_index][1] if default_index is not None else None
        own = blocks[own_index][1] if own_index is not None else None
        merged = deep_merge_dicts(default or {}, own or {})
        if not merged.get("enabled", False):
            continue
        root = {k: v for k, v in merged.items() if k not in {"pathways", "enabled"}}
        for name, entry in (merged.get("pathways") or {}).items():
            if not isinstance(entry, dict):
                continue
            for alias, canonical in RULE_KEY_ALIASES.items():
                if alias not in entry or canonical in entry or canonical not in root:
                    continue
                # Version 2 read the root's key; remove the alias where written.
                for index in (own_index, default_index):
                    holder = blocks[index][1] if index is not None else None
                    written = ((holder or {}).get("pathways") or {}).get(name)
                    if isinstance(written, dict) and alias in written:
                        value = written.pop(alias)
                        notes.append(
                            f"{blocks[index][0]}.pathways.{name}.{alias}: removed "
                            f"{value!r} (the rule's {canonical}={root[canonical]!r} "
                            "was read)"
                        )
                        break
        if root.get("seed") is not None:
            continue
        needs = set()
        for pathway in _V2_PATHWAYS:
            resolved = _v2_rule(merged, pathway)
            if resolved is None:
                continue
            rule, entry = resolved
            method = str(rule.get("method", "bernoulli")).lower()
            if "seed" in entry:
                continue
            if method in _V2_RANDOM_MASK_METHODS:
                needs.add("global_rng")
            elif method in _V2_SPATIAL_METHODS:
                needs.add(0)
        if not needs:
            continue
        if own is not None and ("seed" in own or default is None):
            source = own_index
        else:
            source = default_index
        seeds.setdefault(source, set()).update(needs)
    for index, needs in seeds.items():
        path, block = blocks[index]
        if len(needs) == 1:
            value = next(iter(needs))
            block["seed"] = value
            notes.append(
                f"{path}.seed: wrote {value!r} ("
                + (
                    "its masks drew from the global random stream"
                    if value == "global_rng"
                    else "its spatial contacts used seed 0"
                )
                + " before config_version 3)"
            )


def _structured_rule_holders(core: dict) -> list:
    """The structured rules of a core: (typed blocks, population groups)."""
    typed = []
    connectivity = core.get("connectivity")
    if isinstance(connectivity, dict):
        for key in ("structured", "structured_connectivity"):
            if isinstance(connectivity.get(key), dict):
                typed.append((f"model.core.connectivity.{key}", connectivity[key]))
    groups = []
    network = core.get("population_network")
    for li, layer in enumerate(
        (network.get("layers") or []) if isinstance(network, dict) else []
    ):
        if not isinstance(layer, dict):
            continue
        prefix = f"model.core.population_network.layers[{li}]"
        defaults = layer.get("population_defaults")
        blocks: list = []
        default_index = None
        if isinstance(defaults, dict) and isinstance(
            defaults.get("structured_connectivity"), dict
        ):
            blocks.append(
                (
                    f"{prefix}.population_defaults.structured_connectivity",
                    defaults["structured_connectivity"],
                )
            )
            default_index = 0
        members = []
        for pi, population in enumerate(layer.get("populations") or []):
            own = (
                (population.get("population") or {})
                if isinstance(population, dict)
                else {}
            )
            own_index = None
            if isinstance(own, dict) and isinstance(
                own.get("structured_connectivity"), dict
            ):
                blocks.append(
                    (
                        f"{prefix}.populations[{pi}].population.structured_connectivity",
                        own["structured_connectivity"],
                    )
                )
                own_index = len(blocks) - 1
            members.append((f"{prefix}.populations[{pi}]", (default_index, own_index)))
        groups.append((blocks, members))
    for key in ("unified_ei", "ei_unified"):
        unified = core.get(key)
        for li, layer in enumerate(
            (unified.get("layers") or []) if isinstance(unified, dict) else []
        ):
            for role in ("excitatory", "inhibitory"):
                population = layer.get(role) if isinstance(layer, dict) else None
                if isinstance(population, dict) and isinstance(
                    population.get("structured_connectivity"), dict
                ):
                    path = f"model.core.{key}.layers[{li}].{role}"
                    groups.append(
                        (
                            [
                                (
                                    f"{path}.structured_connectivity",
                                    population["structured_connectivity"],
                                )
                            ],
                            [(path, (None, 0))],
                        )
                    )
    return [typed, groups]


def _structured_meanings(core: dict, *, version: int) -> dict:
    """The meaning of every structured rule of a core, by population and pathway."""
    from dendritic_modeling.config.conversion import deep_merge_dicts

    typed, groups = _structured_rule_holders(core)
    meanings = {}
    for path, block in typed:
        meanings[path] = _pathway_meanings(block, version=version, typed=True)
    for blocks, members in groups:
        for path, (default_index, own_index) in members:
            default = blocks[default_index][1] if default_index is not None else {}
            own = blocks[own_index][1] if own_index is not None else {}
            merged = deep_merge_dicts(default or {}, own or {})
            if merged:
                meanings[path] = _pathway_meanings(merged, version=version, typed=False)
    return meanings


def _write_version2_structured_meanings(
    data: dict, notes: list[str], unresolved: list[str]
) -> None:
    """Keep what the structured-connectivity rules of a version-2 file meant.

    Version 3 reads a typed ``distance`` rule without a kernel as gaussian
    (version 2: ``none``), lets a pathway entry's second spelling of a key
    (``kernel``, ``sigma``, ``p_connect``, ``in_degree``) override the root's
    key, rejects a key or pathway written twice, and seeds unseeded population
    and unified E/I rules from the topology seed. The rewrite is checked
    against the version-2 reading of every population and pathway; what it
    cannot keep is reported.
    """
    core = data.get("model", {}).get("core")
    if not isinstance(core, dict):
        return
    try:
        before = _structured_meanings(copy.deepcopy(core), version=2)
    except (TypeError, ValueError) as error:
        unresolved.append(
            f"model.core structured connectivity: {error} (version 2 could not "
            "read it either)"
        )
        return
    typed, groups = _structured_rule_holders(core)
    for path, block in typed:
        _upgrade_typed_structured_block(block, path, notes)
    for blocks, members in groups:
        _upgrade_population_structured_blocks(blocks, members, notes)
    try:
        after = _structured_meanings(core, version=3)
    except (TypeError, ValueError) as error:
        unresolved.append(f"model.core structured connectivity: {error}")
        return
    for path, pathways in before.items():
        for pathway, meaning in pathways.items():
            if after.get(path, {}).get(pathway) != meaning:
                unresolved.append(
                    f"{path} structured connectivity, pathway {pathway}: version 2 "
                    f"read {meaning}, and the upgraded rule reads "
                    f"{after.get(path, {}).get(pathway)}; write the version-2 "
                    "values by hand"
                )


def _write_historical_stochastic_defaults(
    data: dict, notes: list[str], unresolved: list[str]
) -> None:
    """Write the stochastic Top-K settings that omitted keys meant before 2026-10-08.

    Stochastic selection took temperature 0 when it was omitted (now 0.5), and
    population options took ``topk_ultrafast: false`` (now true).
    ``model.core.sparsity`` with type stochastic used the noisy top-K with
    noise std ``0.01 * (1 + noise_level)``, which ``temperature: 0.0`` keeps.
    Stochastic selection at temperature 0 without the ultrafast noise selected
    exactly the standard top-K, which the current library rejects: for
    indexed_dynamic the selection becomes ``standard`` (the same computation),
    while the stochastic layer type is reported, because the standard layer
    draws its initial weights once where the stochastic one drew them twice.
    """
    core = data.get("model", {}).get("core", {})
    if not isinstance(core, dict):
        return
    sparsity = core.get("sparsity")
    if isinstance(sparsity, dict) and "temperature" not in sparsity:
        kind = str(sparsity.get("type", "standard")).strip().lower()
        indexed = sparsity.get("indexed")
        indexed = indexed if isinstance(indexed, dict) else {}
        path = "model.core.sparsity"
        if kind == "stochastic":
            if sparsity.get("ultrafast", True) is False:
                unresolved.append(
                    f"{path}: type stochastic with ultrafast false and the former "
                    "default temperature 0 selected exactly the standard Top-K, "
                    "which this library rejects; choose type standard (initial "
                    "weights are drawn differently) or a positive temperature"
                )
            else:
                sparsity["temperature"] = 0.0
                notes.append(
                    f"{path}.temperature: wrote 0.0 (the former default; it is now 0.5)"
                )
        elif kind == "indexed_dynamic" and str(
            indexed.get("selection", "")
        ).lower() in {
            "stochastic",
            "rank_probabilistic",
        }:
            indexed["selection"] = "standard"
            notes.append(
                f"{path}.indexed.selection: stochastic -> standard (the former "
                "default temperature 0 selected exactly the standard candidates)"
            )
    for path, options, defaults in _stochastic_population_options(data):
        merged = {**defaults, **options}
        kind = str(merged.get("topk_type", "standard")).strip().lower()
        selection = str(merged.get("indexed_selection", "standard")).strip().lower()
        stochastic = kind == "stochastic" or (
            kind == "indexed_dynamic"
            and selection in {"stochastic", "rank_probabilistic"}
        )
        if not stochastic or (
            "topk_temperature" in merged and "topk_ultrafast" in merged
        ):
            continue
        temperature = merged.get("topk_temperature")
        if temperature is None and kind == "indexed_dynamic":
            options["indexed_selection"] = "standard"
            notes.append(
                f"{path}.indexed_selection: stochastic -> standard (the former "
                "default topk_temperature 0 selected exactly the standard candidates)"
            )
            continue
        if temperature is None and merged.get("topk_ultrafast") is True:
            options["topk_temperature"] = 0.0
            notes.append(
                f"{path}.topk_temperature: wrote 0.0 (the former default; it is now 0.5)"
            )
            continue
        if temperature is not None and float(temperature) > 0:
            options["topk_ultrafast"] = False
            notes.append(
                f"{path}.topk_ultrafast: wrote false (the former default; it is now true)"
            )
            continue
        unresolved.append(
            f"{path}: topk_type stochastic with the former defaults (topk_ultrafast "
            "false, topk_temperature 0) selected exactly the standard Top-K, which "
            "this library rejects; choose topk_type standard (initial weights are "
            "drawn differently) or a positive temperature"
        )


def _report_top_level_fsdp(data: dict, notes: list[str], unresolved: list[str]) -> None:
    """Remove a top-level ``fsdp`` block that repeats ``distributed.fsdp`` or is off.

    A block with ``use_fsdp: false`` (the retired switch) describes a run
    without FSDP and is removed too. Any other top-level block is reported:
    early versions read FSDP settings there, while later ones ignored the
    block next to a ``distributed`` section, so which settings a run used
    depends on the library version that ran it.
    """
    if "fsdp" not in data:
        return
    distributed = data.get("distributed")
    distributed = distributed if isinstance(distributed, dict) else {}
    if distributed.get("fsdp") == data["fsdp"]:
        del data["fsdp"]
        notes.append("removed top-level fsdp (identical to distributed.fsdp)")
        return
    if isinstance(data["fsdp"], dict) and data["fsdp"].get("use_fsdp") is False:
        del data["fsdp"]
        notes.append(
            "removed top-level fsdp (use_fsdp: false; the run did not use FSDP)"
        )
        return
    unresolved.append(
        "fsdp: a top-level block is not read; move the settings the run used "
        "under distributed.fsdp (with distributed.mode: fsdp) and remove it"
    )


def upgrade_experiment(
    data: dict,
    *,
    overridden: Iterable[str] = (),
    sweep: dict | None = None,
) -> tuple[dict, list[str], list[str]]:
    """Return the upgraded experiment mapping, change notes and unresolved keys.

    ``overridden`` lists dotted paths that sweep axes or variants set on top
    of ``data``, the base_config of ``sweep``. A rule whose rewrite depends on
    them is applied only when it rewrites every generated config the same
    way; otherwise it is reported instead.
    """
    from dendritic_modeling.config.extensions import apply_config_migrations
    from dendritic_modeling.plugins import load_plugins

    load_plugins()
    data, notes = apply_config_migrations(copy.deepcopy(data))
    unresolved: list[str] = []
    if UNRESOLVED_MIGRATION_KEY in data:
        # A partial migration: its rewrites are done and must not run again,
        # and what is left stays reported until it is resolved by hand.
        recorded = data[UNRESOLVED_MIGRATION_KEY]
        recorded = recorded if isinstance(recorded, list) else [recorded]
        unresolved = [str(item) for item in recorded] or [
            f"{UNRESOLVED_MIGRATION_KEY} is empty; delete it"
        ]
        return data, notes, unresolved
    version = data.get("config_version")
    if version is not None:
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or not 1 <= version <= CONFIG_VERSION
        ):
            unresolved.append(
                f"config_version: {version!r} is not a configuration format this "
                f"library knows (1 to {CONFIG_VERSION}); the file was left unchanged"
            )
            return data, notes, unresolved
        # Each rule rewrites the files of the versions before the one whose
        # meaning it records; a file of the current version is left as is.
        if version == CONFIG_VERSION:
            return data, notes, unresolved
    # A file without a version is read as written for the first format.
    written_for = 1 if version is None else version
    from dendritic_modeling.config.inhibitory_wiring import RETIRED_ROUTING_KEYS

    if written_for < 2:
        sections = _config_sections()
        retired_top = set(RETIRED_KEYS.get("", {}))
        extra = [
            key
            for key in list(data)
            if key not in sections
            and key not in retired_top
            and key not in {"extends", "fsdp"}
        ]
        _report_top_level_fsdp(data, notes, unresolved)
        if extra:
            metadata = data.get("metadata")
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            for key in extra:
                if key in metadata and metadata[key] != data[key]:
                    unresolved.append(
                        f"{key}: also present in metadata with another value"
                    )
                    continue
                metadata[key] = data.pop(key)
                notes.append(f"moved top-level {key} under metadata")
            data["metadata"] = metadata
        for pattern, spec in _RENAMED_SECTIONS:
            renamed = _renamed_fields(spec)
            for path, section in list(_sections(data, pattern.split("."), "")):
                _rename_in_place(section, renamed, path, notes)
    # A file of any earlier version can hold keys retired since it was written;
    # those it still loads with have no effect and are removed. The routing
    # keys are resolved with the synapse counts they wired.
    _drop_retired(
        data,
        notes,
        unresolved,
        skip=frozenset(("model.core.transfer", key) for key in RETIRED_ROUTING_KEYS),
    )
    overridden = list(overridden)
    # (rule, sections and fields it reads, version): each rule rewrites the
    # files written for a format before that version.
    rules = (
        (
            _reproduce_inhibitory_routing,
            ("model.core",),
            _STRUCTURED_FIELDS
            | {"transfer", "vision", "inhibitory_ratio", *RETIRED_ROUTING_KEYS},
            2,
        ),
        (
            _fold_learned_output_scale,
            ("model",),
            frozenset({"learned_output_scale", "output_scale_mode"}),
            2,
        ),
        (
            _drop_unread_io_params,
            ("model.encoder", "model.decoder"),
            frozenset({"type", "params"}),
            2,
        ),
        (_drop_derived_input_dims, (), frozenset(), 2),
        (_drop_replaced_recurrent_baseline_settings, (), frozenset(), 3),
        (_move_analysis_annotations, ("analysis",), frozenset({"analysis"}), 2),
        (
            _drop_morphology_overridden_by_alias,
            ("model.core",),
            frozenset(
                {"type", "morphology", "use_shunting", "use_additive_normalization"}
            ),
            2,
        ),
        (
            _allow_path_scalar_fallback,
            ("training",),
            frozenset(
                {
                    "strategy",
                    "learning_strategy_config",
                    "error_broadcast_mode",
                    "morphology_aware",
                    "use_path_propagation",
                }
            ),
            2,
        ),
        (
            _drop_unread_population_sparse_options,
            ("model.core.population_network",),
            _POPULATION_FIELDS | {"topk_type"},
            2,
        ),
        (
            _allow_unmodeled_local_features,
            ("model.core", "training"),
            _POPULATION_FIELDS
            | _UNMODELED_FEATURE_FIELDS
            | {
                "strategy",
                "learning_strategy_config",
                "unified_ei",
                "ei_unified",
                "excitatory",
                "inhibitory",
                "dynamics",
                "spiking",
                "mode",
                "dendritic_spikes",
                "soma_feedback",
                "enabled",
            },
            2,
        ),
        (
            _freeze_quantile_gates_without_recalibration,
            ("training",),
            frozenset(
                {"common", "reactivation_update_mode", "recalibrate_reactivation_every"}
            ),
            2,
        ),
        (
            _fix_output_scale_without_identity_decoder,
            ("model",),
            frozenset({"output_scale_mode", "decoder", "type"}),
            2,
        ),
        (_drop_feedforward_recurrent_settings, ("model.core",), _STRUCTURED_FIELDS, 2),
        (
            _drop_additive_options_of_shunting_layers,
            ("model.core",),
            _POPULATION_FIELDS
            | {
                "type",
                "morphology",
                "use_shunting",
                "additive_mode",
                "use_additive_normalization",
            },
            2,
        ),
        (
            _write_default_unified_inhibitory,
            ("model.core.unified_ei", "model.core.ei_unified"),
            frozenset({"layers", "inhibitory"}),
            2,
        ),
        (
            _zero_unsourced_unified_counts,
            ("model.core.unified_ei", "model.core.ei_unified"),
            frozenset(
                {
                    "layers",
                    "excitatory",
                    "inhibitory",
                    "recurrent",
                    "use_transfer",
                    "direct_ff_inhibitory_to_excitatory",
                    "ff_inhibitory_synapses",
                    "ff_inhibitory_synapses_by_level",
                    "rec_inhibitory_synapses",
                    "rec_inhibitory_synapses_by_level",
                }
            ),
            3,
        ),
        (
            _drop_unapplied_processing,
            ("data",),
            frozenset(
                {"dataset_name", "processing", "dataset_params", "flatten", "normalize"}
            ),
            2,
        ),
        (
            _drop_unapplied_step_options,
            ("training",),
            frozenset({"strategy", "common", "use_amp", "enable_nan_checking"}),
            2,
        ),
        (
            _drop_replaced_layer_indexed_seeds,
            ("model.core.population_network",),
            frozenset(
                {
                    "layers",
                    "population_defaults",
                    "populations",
                    "population",
                    "indexed_seed",
                    "topk_type",
                }
            ),
            2,
        ),
        (
            _write_unset_recurrent_counts,
            ("model.core.population_network",),
            _POPULATION_FIELDS,
            2,
        ),
        (_zero_inert_inhibitory_counts, ("model.core",), _STRUCTURED_FIELDS, 2),
        (
            _write_historical_stochastic_defaults,
            ("model.core",),
            _POPULATION_FIELDS
            | {
                "sparsity",
                "type",
                "temperature",
                "ultrafast",
                "indexed",
                "selection",
                "unified_ei",
                "ei_unified",
                "excitatory",
                "inhibitory",
                "topk_type",
                "topk_temperature",
                "topk_ultrafast",
                "indexed_selection",
            },
            2,
        ),
        (_drop_unread_core_sections, ("model.core",), _core_setting_names(), 2),
        (
            _write_version2_structured_meanings,
            (
                "model.core.connectivity",
                "model.core.population_network",
                "model.core.unified_ei",
                "model.core.ei_unified",
            ),
            _STRUCTURED_RULE_FIELDS,
            3,
        ),
    )
    for rule, sections, fields, before in rules:
        if written_for >= before:
            continue
        candidate = copy.deepcopy(data)
        trial, trial_unresolved = _run_rule(rule, candidate)
        if not trial and not trial_unresolved:
            continue

        def reads(key: str, sections=sections, fields=fields) -> bool:
            return _rule_reads(key, sections, fields)

        blocking = [key for key in overridden if key != "extends" and reads(key)]
        if blocking and sweep is not None:
            conditions = _sweep_conditions(sweep, reads)
            if conditions is not None and _rewrite_holds(
                rule, data, candidate, conditions
            ):
                blocking = []
        if "extends" in data or "extends" in overridden:
            blocking.append("extends")  # the parent file may set what it reads
        if blocking:
            unresolved += [
                f"not applied, because {', '.join(blocking)} also sets what it "
                f"depends on (upgrade the generated or composed configs): {note}"
                for note in trial + trial_unresolved
            ]
            continue
        data = candidate
        notes += trial
        unresolved += trial_unresolved
    # The version marks the rewrite as done, so a second pass leaves the file
    # unchanged instead of reading keys this pass removed as omitted. What
    # could not be rewritten is listed in the file, which then does not load
    # and is reported again by every later pass until it is resolved.
    head: dict[str, Any] = {"config_version": CONFIG_VERSION}
    if unresolved:
        head[UNRESOLVED_MIGRATION_KEY] = list(unresolved)
    data = {**head, **{key: value for key, value in data.items() if key not in head}}
    notes.append(
        f"config_version: wrote {CONFIG_VERSION} (every key now has this "
        "library's meaning)"
    )
    if unresolved:
        notes.append(
            f"{UNRESOLVED_MIGRATION_KEY}: listed {len(unresolved)} item(s) to "
            "resolve by hand before the file loads"
        )
    return data, notes, unresolved


def _rename_dotted(path: str, notes: list[str]) -> str:
    for pattern, spec in _RENAMED_SECTIONS:
        renamed = _renamed_fields(spec)
        parts, pattern_parts = path.split("."), pattern.split(".")
        if len(parts) != len(pattern_parts) + 1:
            continue
        if all(p == "*" or p == q for p, q in zip(pattern_parts, parts[:-1])):
            new = renamed.get(parts[-1])
            if new is not None:
                updated = ".".join([*parts[:-1], new])
                notes.append(f"sweep path {path} -> {updated}")
                return updated
    return path


def upgrade_sweep(data: dict) -> tuple[dict, list[str], list[str]]:
    """Upgrade a sweep file's base config and override paths."""
    data = copy.deepcopy(data)
    notes: list[str] = []
    unresolved: list[str] = []
    if isinstance(data.get("sweep_config"), dict):
        data["sweep_config"] = {
            _rename_dotted(str(key), notes): value
            for key, value in data["sweep_config"].items()
        }
    overridden = list(data.get("sweep_config") or {})
    filters = data.get("filter_config")
    if isinstance(filters, dict) and "tolerance" not in filters:
        # Generation used these tolerances when none was written.
        default = {"ei-sum": 0, "ei-ratio": 0.1}.get(
            str(filters.get("mode", "")).lower()
        )
        if default is not None:
            filters["tolerance"] = default
            notes.append(f"filter_config.tolerance: wrote {default} (the value used)")
    for variant in data.get("sweep_variants") or []:
        if isinstance(variant, dict) and isinstance(variant.get("overrides"), dict):
            variant["overrides"] = {
                _rename_dotted(str(key), notes): value
                for key, value in variant["overrides"].items()
            }
            overridden += list(variant["overrides"])
    base = data.get("base_config")
    if isinstance(base, dict):
        if "extends" in data:
            overridden.append("extends")
        base, base_notes, base_unresolved = upgrade_experiment(
            base, overridden=overridden, sweep=data
        )
        data["base_config"] = base
        notes += [f"base_config: {note}" for note in base_notes]
        unresolved += [f"base_config: {item}" for item in base_unresolved]
    return data, notes, unresolved


def upgrade_mapping(data: Any) -> tuple[Any, list[str], list[str]]:
    """Upgrade an experiment or sweep mapping; other documents are returned as is."""
    if not isinstance(data, dict):
        return data, [], []
    if "base_config" in data:
        return upgrade_sweep(data)
    if "model" in data or "training" in data or "experiment" in data:
        return upgrade_experiment(data)
    return data, [], []


def upgrade_file(
    path: Path, destination: Path, *, dump: Callable[[Any, Any], None] | None = None
) -> dict[str, Any]:
    """Write the upgraded form of ``path`` to ``destination``; return a record."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    upgraded, notes, unresolved = upgrade_mapping(data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        (dump or (lambda d, h: yaml.safe_dump(d, h, sort_keys=False)))(upgraded, handle)
    return {
        "source": str(path),
        "output": str(destination),
        "changes": notes,
        "unresolved": unresolved,
    }


def _destinations(
    configs: list[Path], output_dir: Path | None, *, in_place: bool
) -> list[Path]:
    """Output paths: the inputs' paths below their common directory.

    Raises ``SystemExit`` before anything is written when two inputs would
    be written to the same file.
    """
    if in_place:
        destinations = list(configs)
    else:
        resolved = [path.resolve() for path in configs]
        root = Path(os.path.commonpath([path.parent for path in resolved]))
        destinations = [output_dir / path.relative_to(root) for path in resolved]
    seen: dict[Path, Path] = {}
    for source, destination in zip(configs, destinations, strict=True):
        key = destination.resolve()
        if key in seen:
            raise SystemExit(
                f"{source} and {seen[key]} would both be written to {destination}"
            )
        seen[key] = source
    return destinations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "configs", nargs="+", type=Path, help="configuration files to upgrade"
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "write upgraded copies here, at their paths relative to the "
            "inputs' common directory"
        ),
    )
    target.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite the files (comments are lost)",
    )
    parser.add_argument("--report", type=Path, help="write the change records as JSON")
    args = parser.parse_args(argv)
    destinations = _destinations(args.configs, args.output_dir, in_place=args.in_place)
    records = []
    for path, destination in zip(args.configs, destinations, strict=True):
        records.append(upgrade_file(path, destination))
        record = records[-1]
        print(
            f"{path}: {len(record['changes'])} changes"
            + (
                f", {len(record['unresolved'])} unresolved"
                if record["unresolved"]
                else ""
            )
        )
        for item in record["unresolved"]:
            print(f"  unresolved: {item}", file=sys.stderr)
    if args.report:
        args.report.write_text(json.dumps(records, indent=1))
    return 1 if any(record["unresolved"] for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
