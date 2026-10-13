"""Where the inhibitory synapses of a structured EINet read their input.

Feedforward and recurrent structured EINets (``model.core.architecture`` and
``model.core.connectivity``) follow one rule:

* The first layer's feedforward inhibitory input is the transfer layer's
  inhibitory stream; a later layer's is the previous layer's inhibitory cells.
* Inhibitory cells read their layer's feedforward inhibitory input on their
  ``ii`` synapses.
* Excitatory cells read their own layer's inhibitory cells on their ``ie``
  synapses. In the first layer, when it has no inhibitory cells, they read the
  transfer layer's inhibitory stream directly.

A synapse count with no source raises, as do inhibitory cells without input
synapses and synapse counts for inhibitory cells a layer does not have; these
settings were silently ignored before 2026-10-08.
:func:`check_retired_inhibitory_routing` rejects the retired
``transfer.input_mode`` options in the few configurations whose wiring they
changed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from dendritic_modeling.config.conversion import layer_value, to_plain_dict
from dendritic_modeling.config.keys import check_retired_keys

# transfer keys that selected inhibitory routing before 2026-10-08.
RETIRED_ROUTING_KEYS = (
    "input_mode",
    "inhibitory_mode",
    "allow_direct_inhibitory_stream",
    "allow_direct_inhibitory_input",
    "require_explicit_direct_inhibitory_stream",
    "input_mode1_build_inhibitory_population",
    "build_inhibitory_population_for_input_mode1",
)

_COUNTS = {
    "ee": "ee_synapses_per_branch_per_layer",
    "ei": "ei_synapses_per_branch_per_layer",
    "ie": "ie_synapses_per_branch_per_layer",
    "ii": "ii_synapses_per_branch_per_layer",
    "rec_ei": "rec_ei_synapses_per_branch",
    "rec_ie": "rec_ie_synapses_per_branch",
    "rec_ii": "rec_ii_synapses_per_branch",
}


def _count(connectivity: Mapping[str, Any], name: str, layer: int) -> int:
    return int(layer_value(connectivity.get(_COUNTS[name], []), layer, default=0) or 0)


def check_inhibitory_wiring(
    architecture: Mapping[str, Any],
    connectivity: Mapping[str, Any],
    *,
    has_transfer_stream: bool,
) -> None:
    """Raise when an inhibitory synapse count or population has no counterpart.

    Synapse lists repeat their last entry for later layers, so a one-entry
    ``ie`` list also gives later layers inhibitory synapses.
    """
    excitatory_sizes = list(architecture.get("excitatory_layer_sizes") or [])
    inhibitory_sizes = list(architecture.get("inhibitory_layer_sizes") or [])
    n_layers = len(excitatory_sizes)
    if inhibitory_sizes and len(inhibitory_sizes) != n_layers:
        raise ValueError(
            f"architecture.inhibitory_layer_sizes={inhibitory_sizes} must be empty "
            f"or give one size per layer ({n_layers} layers); write 0 for a layer "
            "without inhibitory cells"
        )
    problems = []
    previous_has_cells = False
    for layer in range(n_layers):
        size = int(inhibitory_sizes[layer] or 0) if inhibitory_sizes else 0
        counts = {name: _count(connectivity, name, layer) for name in _COUNTS}
        cell_inputs = [
            name for name in ("ei", "ii", "rec_ei", "rec_ii") if counts[name]
        ]
        has_cells = size > 0
        if has_cells and not cell_inputs:
            problems.append(
                f"layer {layer} declares {size} inhibitory cells but gives them no "
                "input synapses (ei, ii, rec_ei and rec_ii are 0); set its "
                "inhibitory_layer_sizes entry to 0 or give the cells synapses"
            )
        if not has_cells and cell_inputs:
            problems.append(
                f"layer {layer} has no inhibitory cells, but sets "
                + ", ".join(f"{name}={counts[name]}" for name in cell_inputs)
                + " synapses for them; set these to 0 for this layer or add "
                "inhibitory cells"
            )
        incoming = has_transfer_stream if layer == 0 else previous_has_cells
        if has_cells and counts["ii"] and not incoming:
            source = (
                "the network has no transfer layer"
                if layer == 0
                else f"layer {layer - 1} has no inhibitory cells"
            )
            problems.append(
                f"layer {layer} inhibitory cells have ii={counts['ii']} synapses "
                f"but no feedforward inhibitory input: {source}"
            )
        if not has_cells and counts["ie"]:
            if layer > 0:
                problems.append(
                    f"layer {layer} excitatory cells have ie={counts['ie']} synapses "
                    "but the layer has no inhibitory cells, and only the first layer "
                    "reads the transferred inhibitory stream; set ie to 0 for this "
                    "layer or add inhibitory cells"
                )
            elif not has_transfer_stream:
                problems.append(
                    f"first-layer excitatory cells have ie={counts['ie']} synapses "
                    "but the layer has no inhibitory cells and the network has no "
                    "transfer layer to provide an inhibitory stream"
                )
        if not has_cells and counts["rec_ie"]:
            problems.append(
                f"layer {layer} excitatory cells have rec_ie={counts['rec_ie']} "
                "recurrent inhibitory synapses but the layer has no inhibitory cells"
            )
        previous_has_cells = has_cells and bool(cell_inputs)
    if problems:
        raise ValueError(
            "model.core inhibitory synapses without a source or target:\n- "
            + "\n- ".join(problems)
        )


def check_retired_inhibitory_routing(core: Any, *, recurrent: bool) -> None:
    """Reject retired routing options whose wiring the current rule differs from.

    The retired ``model.core.transfer`` keys load with a warning when the
    network they selected is the one the current rule builds. They raise for
    the configurations they built differently:

    * feedforward ``input_mode: 1`` without
      ``input_mode1_build_inhibitory_population`` ran without the inhibitory
      cells it declares;
    * feedforward ``inhibitory_mode`` other than ``first`` never built;
    * recurrent ``input_mode: 0`` or ``inhibitory_mode: none`` ran first-layer
      excitatory cells without inhibitory cells and without their ``ie``
      synapses;
    * recurrent ``inhibitory_mode: all`` fed the transferred stream to the
      inhibitory cells of later layers.
    """
    core = to_plain_dict(core)
    transfer = to_plain_dict(core.get("transfer") or {})
    if not set(transfer) & set(RETIRED_ROUTING_KEYS):
        return
    architecture = to_plain_dict(core.get("architecture") or {})
    connectivity = to_plain_dict(core.get("connectivity") or {})
    inhibitory_sizes = list(architecture.get("inhibitory_layer_sizes") or [])
    n_layers = len(architecture.get("excitatory_layer_sizes") or [])

    def has_cells(layer: int) -> bool:
        return layer < len(inhibitory_sizes) and int(inhibitory_sizes[layer] or 0) > 0

    mode = transfer.get("input_mode")
    inhibitory_mode = str(transfer.get("inhibitory_mode", "first")).lower()
    replay = (
        " Replay this config at the library revision that produced it, or "
        "remove the retired transfer keys and set the synapse counts to the "
        "network you want."
    )
    if not recurrent:
        if inhibitory_mode != "first":
            raise ValueError(
                f"transfer.inhibitory_mode={inhibitory_mode!r} applied only to "
                "recurrent EINets; feedforward EINets never built with it." + replay
            )
        builds = transfer.get(
            "input_mode1_build_inhibitory_population",
            transfer.get("build_inhibitory_population_for_input_mode1", False),
        )
        driven = [
            layer
            for layer in range(n_layers)
            if has_cells(layer)
            and (_count(connectivity, "ei", layer) or _count(connectivity, "ii", layer))
        ]
        if mode == 1 and not builds and driven:
            raise ValueError(
                "transfer.input_mode=1 ran this feedforward EINet without the "
                f"inhibitory cells it declares in layers {driven}; the current "
                "library builds them. Set those inhibitory_layer_sizes entries and "
                "their ei and ii synapses to 0 to reproduce the run." + replay
            )
        return
    recurrent_ei = to_plain_dict(core.get("recurrent_ei") or {})
    if not recurrent_ei.get("use_transfer", True):
        return
    if (mode == 0 or inhibitory_mode == "none") and n_layers:
        if not has_cells(0) and _count(connectivity, "ie", 0):
            raise ValueError(
                "transfer.input_mode=0 or inhibitory_mode='none' ran this recurrent "
                "EINet's first-layer excitatory cells without their ie synapses; "
                "they now read the transferred inhibitory stream. Set ie for the "
                "first layer to 0 to reproduce the run." + replay
            )
    if mode != 0 and inhibitory_mode == "all":
        later = [
            layer
            for layer in range(1, n_layers)
            if has_cells(layer) and _count(connectivity, "ii", layer)
        ]
        if later:
            raise ValueError(
                "transfer.inhibitory_mode='all' fed the transferred inhibitory "
                f"stream to the inhibitory cells of layers {later}; they now read "
                "the previous layer's inhibitory cells." + replay
            )


def strip_retired_routing_keys(transfer: Mapping[str, Any]) -> dict[str, Any]:
    """Return ``transfer`` without the retired routing keys.

    Each key is checked against the retired-key table, which raises for a value
    that never loaded and warns once otherwise.
    """
    kept = {k: v for k, v in transfer.items() if k not in RETIRED_ROUTING_KEYS}
    retired = {k: v for k, v in transfer.items() if k in RETIRED_ROUTING_KEYS}
    if retired:
        check_retired_keys(retired, section="model.core.transfer")
    return kept


__all__ = [
    "RETIRED_ROUTING_KEYS",
    "check_inhibitory_wiring",
    "check_retired_inhibitory_routing",
    "strip_retired_routing_keys",
]
