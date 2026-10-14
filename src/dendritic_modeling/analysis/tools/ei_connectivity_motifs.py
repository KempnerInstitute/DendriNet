"""Connectivity-motif analysis for recurrent E/I dendritic networks.

This module extends the recurrent inhibitory-specialization tooling to a more
general view of dendritic routing motifs across the local E/I circuit. It reads
trained synaptic weights from branch-layer checkpoints and summarizes how each
route distributes its weight across distal, middle, and somatic compartments of
the target population.

The core structural quantity is a route-level targeting index for each
source-target pair::

    B(src, tgt) = f_soma(src, tgt) - f_distal(src, tgt)

where ``f_level`` is the within-pair fraction of realized synaptic weight mass
delivered to that dendritic level (see
``dendritic_modeling.analysis.utils.compartment_targeting``, shared with the
inhibitory-specialization analysis). Positive values indicate soma-biased
targeting; negative values indicate distal-biased targeting. Pairs without a
realized synapse are disconnected and excluded from the statistics; a source
with no connected target is classed ``disconnected``.

This is a structural routing descriptor. It does not by itself imply causal
dynamic gating, but it can reveal motifs that are consistent with output-gain
control, distal context modulation, or mixed target-dependent inhibition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dendritic_modeling.analysis.utils.compartment_targeting import (
    TargetingClassLabels,
    apply_weight_transform,
    classify_sources,
    level_weight_mass,
    pair_level_stats,
    pair_stats_by_class as shared_pair_stats_by_class,
    population_fraction_summary,
    reshape_by_compartment,
    targeting_index,
)

TARGETING_CLASS_ORDER = [
    "soma-biased",
    "mixed-targeting",
    "diffuse",
    "distal-biased",
    "disconnected",
]
_CLASS_LABELS = TargetingClassLabels(
    soma="soma-biased",
    distal="distal-biased",
    mixed="mixed-targeting",
)


@dataclass(frozen=True)
class ConnectivityRouteSpec:
    """Description of one dendritic connectivity route."""

    key: str
    population: str
    synapse_attr: str
    label: str
    source_label: str
    target_label: str
    temporal_role: str


ROUTE_SPECS: dict[str, ConnectivityRouteSpec] = {
    "e_ff_exc": ConnectivityRouteSpec(
        key="e_ff_exc",
        population="e_population",
        synapse_attr="branch_excitation",
        label="input→E",
        source_label="input",
        target_label="E",
        temporal_role="feedforward excitatory",
    ),
    "e_ff_inh": ConnectivityRouteSpec(
        key="e_ff_inh",
        population="e_population",
        synapse_attr="branch_inhibition",
        label="ff inh→E",
        source_label="inh_ff",
        target_label="E",
        temporal_role="feedforward inhibitory",
    ),
    "e_rec_exc": ConnectivityRouteSpec(
        key="e_rec_exc",
        population="e_population",
        synapse_attr="branch_recurrent",
        label="E(t-1)→E",
        source_label="E",
        target_label="E",
        temporal_role="recurrent excitatory",
    ),
    "e_rec_inh": ConnectivityRouteSpec(
        key="e_rec_inh",
        population="e_population",
        synapse_attr="branch_rec_inhibition",
        label="I(t-1)→E",
        source_label="I",
        target_label="E",
        temporal_role="recurrent inhibitory",
    ),
    "i_ff_exc": ConnectivityRouteSpec(
        key="i_ff_exc",
        population="i_population",
        synapse_attr="branch_excitation",
        label="input→I",
        source_label="input",
        target_label="I",
        temporal_role="feedforward excitatory",
    ),
    "i_ff_inh": ConnectivityRouteSpec(
        key="i_ff_inh",
        population="i_population",
        synapse_attr="branch_inhibition",
        label="ff inh→I",
        source_label="inh_ff",
        target_label="I",
        temporal_role="feedforward inhibitory",
    ),
    "i_rec_exc": ConnectivityRouteSpec(
        key="i_rec_exc",
        population="i_population",
        synapse_attr="branch_recurrent",
        label="E(t-1)→I",
        source_label="E",
        target_label="I",
        temporal_role="recurrent excitatory",
    ),
    "i_rec_inh": ConnectivityRouteSpec(
        key="i_rec_inh",
        population="i_population",
        synapse_attr="branch_rec_inhibition",
        label="I(t-1)→I",
        source_label="I",
        target_label="I",
        temporal_role="recurrent inhibitory",
    ),
}


LOCAL_RECURRENT_ROUTE_KEYS = [
    "e_rec_exc",
    "e_rec_inh",
    "i_rec_exc",
    "i_rec_inh",
]


def pair_stats_by_class(
    bias: torch.Tensor,
    classes: list[str],
    hi: float = 0.1,
    connected: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    """Per-class summary of the connected pairs' targeting index."""
    return shared_pair_stats_by_class(
        bias, classes, TARGETING_CLASS_ORDER, hi=hi, connected=connected
    )


def extract_compartment_weights_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    synapse_attr: str,
    n_target: int,
    weight_transform: str = "exp",
    active_masks: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Extract realized per-level weights as ``(n_target, K_l, n_source)``.

    A state dict stores candidate parameters, so callers must supply the exact
    active masks from the analyzed model/forward realization.
    """
    if active_masks is None:
        raise ValueError(
            "active_masks are required: checkpoint pre_w tensors are dense "
            "candidate parameters, not realized synapses"
        )
    level_tensors: list[torch.Tensor] = []
    level = 0
    while True:
        key = f"{prefix}.branch_layers.{level}.{synapse_attr}.pre_w"
        if key not in state_dict:
            break
        pre_w = state_dict[key].detach().cpu().float()
        if level >= len(active_masks):
            raise ValueError(f"Missing active mask for route level {level}")
        mask = active_masks[level].detach().cpu().to(dtype=pre_w.dtype)
        if mask.shape != pre_w.shape:
            raise ValueError(
                f"Mask shape at level {level} is {tuple(mask.shape)}, expected "
                f"{tuple(pre_w.shape)}"
            )
        w = apply_weight_transform(pre_w, weight_transform) * mask
        compartment_w = reshape_by_compartment(w, n_target=n_target)
        if compartment_w is None:
            raise ValueError(
                f"Could not reshape weights at level {level} for n_target={n_target}"
            )
        level_tensors.append(compartment_w)
        level += 1
    if not level_tensors:
        raise ValueError(
            f"No weights found under prefix '{prefix}' and synapse '{synapse_attr}'"
        )
    return level_tensors


def summarize_compartment_targeting_from_compartment_weights(
    compartment_weights: list[torch.Tensor],
    hi: float = 0.1,
    sigma: float = 0.15,
) -> dict[str, Any]:
    """Summarize soma-versus-distal targeting from compartment weights."""
    if not compartment_weights:
        raise ValueError("Expected at least one level of compartment weights")
    n_target = int(compartment_weights[0].shape[0])
    n_source = int(compartment_weights[0].shape[2])
    out = targeting_index(level_weight_mass(compartment_weights))
    if out is None:
        raise ValueError("Need at least two levels to compute targeting index")
    bias, level_fracs, connected = out
    mean_bias, std_bias, classes = classify_sources(
        bias, hi=hi, sigma=sigma, labels=_CLASS_LABELS, connected=connected
    )
    frac_by_level = [lvl.T for lvl in level_fracs]
    return {
        "n_target": n_target,
        "n_source": n_source,
        "n_levels": len(level_fracs),
        "targeting_index": bias,
        "connected_pairs": connected,
        "mean_bias": mean_bias,
        "std_bias": std_bias,
        "classes": classes,
        "class_counts": {c: int(classes.count(c)) for c in TARGETING_CLASS_ORDER},
        "pair_level": pair_level_stats(bias, hi=hi, connected=connected),
        "level_fracs": level_fracs,
        "frac_by_level_src_target": frac_by_level,
        "frac_distal": frac_by_level[0],
        "frac_middle": (
            torch.stack(frac_by_level[1:-1], dim=0).mean(dim=0)
            if len(frac_by_level) > 2
            else torch.zeros_like(frac_by_level[0])
        ),
        "frac_soma": frac_by_level[-1],
        **population_fraction_summary(level_fracs, connected=connected.T),
    }


def analyze_ei_connectivity_motifs_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    layer_prefix: str,
    n_E: int,
    n_I: int,
    weight_transform: str = "exp",
    route_keys: list[str] | None = None,
    hi: float = 0.1,
    sigma: float = 0.15,
    active_masks_by_route: dict[str, list[torch.Tensor]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Analyze all requested routes present in a checkpoint state dict."""
    if active_masks_by_route is None:
        raise ValueError(
            "active_masks_by_route is required for realized connectivity analysis"
        )
    if route_keys is None:
        route_keys = list(ROUTE_SPECS.keys())

    summaries: dict[str, dict[str, Any]] = {}
    for route_key in route_keys:
        spec = ROUTE_SPECS[route_key]
        n_target = n_E if spec.target_label == "E" else n_I
        prefix = f"{layer_prefix}.{spec.population}"
        try:
            route_masks = active_masks_by_route.get(route_key)
            if route_masks is None:
                continue
            compartment_weights = extract_compartment_weights_from_state_dict(
                state_dict,
                prefix=prefix,
                synapse_attr=spec.synapse_attr,
                n_target=n_target,
                weight_transform=weight_transform,
                active_masks=route_masks,
            )
        except ValueError:
            continue
        summary = summarize_compartment_targeting_from_compartment_weights(
            compartment_weights,
            hi=hi,
            sigma=sigma,
        )
        summaries[route_key] = {
            "route_key": route_key,
            "label": spec.label,
            "source_label": spec.source_label,
            "target_label": spec.target_label,
            "temporal_role": spec.temporal_role,
            "n_compartments_per_level": [int(w.shape[1]) for w in compartment_weights],
            **summary,
        }
    return summaries


__all__ = [
    "LOCAL_RECURRENT_ROUTE_KEYS",
    "ROUTE_SPECS",
    "TARGETING_CLASS_ORDER",
    "ConnectivityRouteSpec",
    "analyze_ei_connectivity_motifs_from_state_dict",
    "extract_compartment_weights_from_state_dict",
    "pair_stats_by_class",
    "summarize_compartment_targeting_from_compartment_weights",
]
