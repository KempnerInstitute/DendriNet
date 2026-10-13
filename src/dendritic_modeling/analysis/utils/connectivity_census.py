"""Census of the synaptic contacts a model realizes.

A contact is realized when its synapse bank's active mask selects it and its
effective weight is nonzero. Pruning zeroes stored weights, and a Top-K mask
can still select a zeroed entry, so counting mask entries alone would
overstate connectivity after pruning. Masks are read in evaluation mode,
where every selector is deterministic.
"""

from __future__ import annotations

from typing import Any

import torch

from dendritic_modeling.analysis.utils.dendritic_depth import (
    soma_relative_dendritic_depth,
)
from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.analysis.utils.runtime import (
    analysis_device_context,
    get_model_device,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.structure import inhibitory_branch_layer_ids
from dendritic_modeling.utils.hooks import iter_named_modules_of_type

# Synapse banks of a branch layer, by pathway label.
SYNAPSE_BANKS = {
    "exc": "branch_excitation",
    "inh": "branch_inhibition",
    "rec_exc": "branch_recurrent",
    "rec_inh": "branch_rec_inhibition",
}


def connectivity_census(model: torch.nn.Module) -> dict[str, Any]:
    """Count the realized contacts of every synapse bank of every branch layer.

    Returns, per branch-layer module, its population label, soma-relative
    depth and, per pathway, the bank's size (``outputs`` branches by
    ``inputs``), the contacts its mask selects, the realized contacts and the
    realized contacts of each branch. ``totals`` sums the selected and
    realized contacts per pathway and over all banks (``all``).
    """
    inhibitory = inhibitory_branch_layer_ids(model)
    modules: dict[str, dict[str, Any]] = {}
    totals = {
        group: {"selected": 0, "realized": 0} for group in (*SYNAPSE_BANKS, "all")
    }
    with analysis_device_context(model, get_model_device(model)), torch.no_grad():
        for name, layer in iter_named_modules_of_type(model, DendriticBranchLayer):
            pathways: dict[str, dict[str, Any]] = {}
            for pathway, attribute in SYNAPSE_BANKS.items():
                bank = getattr(layer, attribute, None)
                if bank is None:
                    continue
                snapshot = effective_synapse_snapshot(bank)
                selected = snapshot.active_mask.ne(0)
                realized = selected & snapshot.effective_weight.ne(0)
                counts = {
                    "outputs": int(selected.shape[0]),
                    "inputs": int(selected.shape[1]),
                    "selected": int(selected.sum()),
                    "realized": int(realized.sum()),
                    "realized_per_branch": realized.sum(dim=1).cpu().tolist(),
                }
                pathways[pathway] = counts
                for group in (pathway, "all"):
                    totals[group]["selected"] += counts["selected"]
                    totals[group]["realized"] += counts["realized"]
            if pathways:
                modules[name] = {
                    "population": (
                        "explicit_inhibitory"
                        if id(layer) in inhibitory
                        else "excitatory"
                    ),
                    "soma_relative_depth": int(soma_relative_dendritic_depth(layer)),
                    "pathways": pathways,
                }
    present = {pathway for module in modules.values() for pathway in module["pathways"]}
    return {
        "modules": modules,
        "totals": {
            group: totals[group]
            for group in (*SYNAPSE_BANKS, "all")
            if group == "all" or group in present
        },
    }


def census_changes(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Compare two censuses of the same model, per module, pathway and total.

    ``removed`` counts contacts realized before but not after, and ``added``
    the reverse; both are differences of counts, so they are exact for an
    operation that only removes (pruning) or only adds contacts.
    """
    if before["modules"].keys() != after["modules"].keys():
        raise ValueError("the two censuses describe different branch layers")

    def _delta(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
        realized_before = int(old["realized"])
        realized_after = int(new["realized"])
        return {
            "realized_before": realized_before,
            "realized_after": realized_after,
            "removed": max(realized_before - realized_after, 0),
            "added": max(realized_after - realized_before, 0),
            "removed_fraction": (
                max(realized_before - realized_after, 0) / realized_before
                if realized_before
                else 0.0
            ),
        }

    modules = {
        name: {
            pathway: _delta(counts, after["modules"][name]["pathways"][pathway])
            for pathway, counts in module["pathways"].items()
        }
        for name, module in before["modules"].items()
    }
    totals = {
        group: _delta(counts, after["totals"][group])
        for group, counts in before["totals"].items()
    }
    return {"modules": modules, "totals": totals}


__all__ = ["SYNAPSE_BANKS", "census_changes", "connectivity_census"]
