"""
Analyze compartment-specific recurrent inhibitory structure in unified E-I RNNs.

This script inspects the trained recurrent inhibitory pathway from inhibitory
neurons to the excitatory population (`I -> E`) in `ei_unified` models. It
does not need to instantiate the model; instead it reads the saved config and
checkpoint, reconstructs the per-level TopK masks, and summarizes whether
inhibitory neurons preferentially target soma, dendrites, or both.

Key outputs:
  - Whether somatic synapses are enabled for the analyzed excitatory population
  - Per-level recurrent inhibitory synapse counts
  - Rule-based groups of inhibitory neurons:
      * `soma_only`
      * `dendrite_only`
      * `mixed`
      * `none`
  - A stronger "compartment-selective" flag for inhibitory neurons that target
    some excitatory neurons only at soma and other excitatory neurons only at
    dendrites
  - Pairwise `I -> E_soma_owner` relation counts:
      * `soma_only`
      * `dendrite_only`
      * `both`
      * `none`

Usage:
    PYTHONPATH=src:$PYTHONPATH python -m \
      dendritic_modeling.scripts.analysis.recurrent_inhibitory_structure \
      --config_path /path/to/config.json \
      --checkpoint_path /path/to/final_model.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed."
            ) from exc
        with path.open("r") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
        return data

    raise ValueError(f"Unsupported config format: {path}")


def _get_unified_layer_cfg(config: dict[str, Any], layer_idx: int) -> dict[str, Any]:
    core_cfg = config.get("model", {}).get("core", {})
    if core_cfg.get("type") not in {
        "ei_unified",
        "unified_ei",
        "ei_net",
        "unified_einet",
        # Recurrent / unified-EI aliases (rnn_dendritic_*, rnn_flat_*) all
        # dispatch through the same factory branch and produce an EINetwork.
        "rnn_dendritic_shunting",
        "rnn_dendritic_additive",
        "rnn_dendritic_normalized_additive",
        "rnn_flat_shunting",
        "rnn_flat_additive",
        "rnn_flat_normalized_additive",
    }:
        raise ValueError(
            "This script currently supports only unified E-I recurrent models."
        )

    unified_cfg = core_cfg.get("unified_ei", core_cfg.get("ei_unified", {}))
    layers = unified_cfg.get("layers", [])
    if not layers:
        raise ValueError("No unified_ei.layers found in config.")
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError(
            f"layer_idx={layer_idx} is out of range for {len(layers)} layers"
        )
    return layers[layer_idx]


def _compute_level_dims(n_soma: int, branch_factors: list[int]) -> list[int]:
    dims = [n_soma]
    for branch_factor in branch_factors:
        dims.append(dims[-1] * branch_factor)
    dims.reverse()
    return dims


def _compute_owner_indices(level_dims: list[int], n_soma: int) -> list[torch.Tensor]:
    owner_indices: list[torch.Tensor] = []
    for level_dim in level_dims:
        repeats = level_dim // n_soma
        owner_indices.append(torch.arange(level_dim, dtype=torch.long) // repeats)
    return owner_indices


def _weight_mask_from_preweights(
    pre_w: torch.Tensor,
    *,
    k: int,
    weight_transform: str,
) -> torch.Tensor:
    if k <= 0:
        return torch.zeros_like(pre_w, dtype=torch.bool)

    if pre_w.shape[1] < k:
        k = pre_w.shape[1]

    scores = pre_w.abs() if weight_transform == "identity" else pre_w
    topk_indices = torch.topk(scores, k, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(pre_w, dtype=torch.bool)
    mask.scatter_(1, topk_indices, True)
    return mask


def _level_name(level_idx: int, n_levels: int) -> str:
    if level_idx == n_levels - 1:
        return "soma"
    return f"dendrite_{level_idx}"


def analyze_recurrent_inhibitory_structure(
    *,
    config: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    layer_idx: int = 0,
) -> dict[str, Any]:
    layer_cfg = _get_unified_layer_cfg(config, layer_idx=layer_idx)
    exc_cfg = layer_cfg["excitatory"]
    inh_cfg = layer_cfg.get("inhibitory")
    if not inh_cfg:
        raise ValueError("Selected layer has no inhibitory population.")

    n_exc = int(exc_cfg["n_neurons"])
    n_inh = int(inh_cfg["n_neurons"])
    branch_factors = list(exc_cfg.get("branch_factors", [1]))
    level_dims = _compute_level_dims(n_exc, branch_factors)
    owner_indices = _compute_owner_indices(level_dims, n_exc)
    n_levels = len(level_dims)
    soma_level_idx = n_levels - 1

    somatic_synapses = bool(exc_cfg.get("somatic_synapses", True))
    rec_inh_k = int(exc_cfg.get("rec_inhibitory_synapses", 0))
    weight_transform = str(exc_cfg.get("weight_transform", "exp")).lower()

    has_soma = torch.zeros(n_inh, n_exc, dtype=torch.bool)
    has_dend = torch.zeros(n_inh, n_exc, dtype=torch.bool)
    synapse_counts_by_level = torch.zeros(n_inh, n_levels, dtype=torch.long)
    owner_target_counts_by_level = torch.zeros(n_inh, n_levels, dtype=torch.long)
    level_summaries: list[dict[str, Any]] = []

    for level_idx, level_dim in enumerate(level_dims):
        key = (
            f"core_network.layers.{layer_idx}.e_population.branch_layers."
            f"{level_idx}.branch_rec_inhibition.pre_w"
        )
        if key not in state_dict:
            level_summaries.append(
                {
                    "level_idx": level_idx,
                    "level_name": _level_name(level_idx, n_levels),
                    "level_dim": level_dim,
                    "present_in_checkpoint": False,
                    "total_active_synapses": 0,
                }
            )
            continue

        pre_w = state_dict[key]
        mask = _weight_mask_from_preweights(
            pre_w, k=rec_inh_k, weight_transform=weight_transform
        )
        synapse_counts_by_level[:, level_idx] = mask.sum(dim=0)
        owners = owner_indices[level_idx]

        owner_hit = torch.zeros(n_inh, n_exc, dtype=torch.bool)
        for row_idx in range(mask.shape[0]):
            active_inhibitory = torch.nonzero(mask[row_idx], as_tuple=False).squeeze(-1)
            if active_inhibitory.numel() == 0:
                continue
            owner = int(owners[row_idx].item())
            owner_hit[active_inhibitory, owner] = True

        owner_target_counts_by_level[:, level_idx] = owner_hit.sum(dim=1)
        if level_idx == soma_level_idx:
            has_soma |= owner_hit
        else:
            has_dend |= owner_hit

        level_summaries.append(
            {
                "level_idx": level_idx,
                "level_name": _level_name(level_idx, n_levels),
                "level_dim": level_dim,
                "present_in_checkpoint": True,
                "total_active_synapses": int(mask.sum().item()),
                "active_synapses_per_inhibitory_mean": float(
                    synapse_counts_by_level[:, level_idx].float().mean().item()
                ),
                "target_owners_per_inhibitory_mean": float(
                    owner_target_counts_by_level[:, level_idx].float().mean().item()
                ),
            }
        )

    inhibitory_neurons: list[dict[str, Any]] = []
    group_counts = {"soma_only": 0, "dendrite_only": 0, "mixed": 0, "none": 0}
    pair_relation_counts = {"soma_only": 0, "dendrite_only": 0, "both": 0, "none": 0}

    for inh_idx in range(n_inh):
        soma_targets = has_soma[inh_idx]
        dend_targets = has_dend[inh_idx]

        n_soma_only = int((soma_targets & ~dend_targets).sum().item())
        n_dend_only = int((dend_targets & ~soma_targets).sum().item())
        n_both = int((soma_targets & dend_targets).sum().item())
        n_none = int((~soma_targets & ~dend_targets).sum().item())

        if bool(soma_targets.any()) and bool(dend_targets.any()):
            group = "mixed"
        elif bool(soma_targets.any()):
            group = "soma_only"
        elif bool(dend_targets.any()):
            group = "dendrite_only"
        else:
            group = "none"
        group_counts[group] += 1

        pair_relation_counts["soma_only"] += n_soma_only
        pair_relation_counts["dendrite_only"] += n_dend_only
        pair_relation_counts["both"] += n_both
        pair_relation_counts["none"] += n_none

        inhibitory_neurons.append(
            {
                "inhibitory_index": inh_idx,
                "group": group,
                "compartment_selective": bool(n_soma_only > 0 and n_dend_only > 0),
                "n_e_targets_soma_only": n_soma_only,
                "n_e_targets_dendrite_only": n_dend_only,
                "n_e_targets_both": n_both,
                "n_e_targets_none": n_none,
                "n_e_targets_any_soma": int(soma_targets.sum().item()),
                "n_e_targets_any_dendrite": int(dend_targets.sum().item()),
                "synapse_counts_by_level": [
                    int(value.item()) for value in synapse_counts_by_level[inh_idx]
                ],
                "owner_target_counts_by_level": [
                    int(value.item()) for value in owner_target_counts_by_level[inh_idx]
                ],
            }
        )

    compartment_selective = [
        neuron for neuron in inhibitory_neurons if neuron["compartment_selective"]
    ]
    compartment_selective.sort(
        key=lambda item: (
            item["n_e_targets_soma_only"] + item["n_e_targets_dendrite_only"],
            item["n_e_targets_both"],
        ),
        reverse=True,
    )

    return {
        "metadata": {
            "layer_idx": layer_idx,
            "n_excitatory": n_exc,
            "n_inhibitory": n_inh,
            "branch_factors": branch_factors,
            "n_levels": n_levels,
            "level_dims_distal_to_soma": level_dims,
            "somatic_synapses": somatic_synapses,
            "rec_inhibitory_synapses_per_branch": rec_inh_k,
            "weight_transform": weight_transform,
        },
        "per_level_summary": level_summaries,
        "inhibitory_group_counts": {
            **group_counts,
            "compartment_selective": len(compartment_selective),
        },
        "pair_relation_counts": pair_relation_counts,
        "top_compartment_selective_inhibitory_neurons": compartment_selective[:10],
        "inhibitory_neurons": inhibitory_neurons,
    }


def _print_summary(results: dict[str, Any]) -> None:
    meta = results["metadata"]
    groups = results["inhibitory_group_counts"]
    pair_counts = results["pair_relation_counts"]

    print(
        json.dumps(
            {
                "metadata": meta,
                "inhibitory_group_counts": groups,
                "pair_relation_counts": pair_counts,
                "top_compartment_selective_inhibitory_neurons": results[
                    "top_compartment_selective_inhibitory_neurons"
                ],
            },
            indent=2,
        )
    )


def _save_pair_relation_plot(
    results: dict[str, Any],
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    pair_counts = results["pair_relation_counts"]
    labels = ["soma_only", "dendrite_only", "both", "none"]
    counts = [int(pair_counts[label]) for label in labels]
    total = sum(counts)
    fractions = [count / total if total else 0.0 for count in counts]
    colors = ["#C75146", "#2D6A4F", "#4F772D", "#8D99AE"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    fig.suptitle(
        title or "Recurrent inhibitory I -> E compartment structure", fontsize=13
    )

    bars = axes[0].bar(labels, counts, color=colors)
    axes[0].set_title("Pairwise I -> E_owner relation counts")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.25)
    for bar, count, frac in zip(bars, counts, fractions):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{count}\n({frac:.1%})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    pie_labels = [
        f"{label}\n{count} ({frac:.1%})"
        for label, count, frac in zip(labels, counts, fractions)
    ]
    axes[1].pie(
        counts,
        labels=pie_labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
        textprops={"fontsize": 9},
    )
    axes[1].set_title("Share of all I -> E_owner pairs")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze recurrent inhibitory soma-vs-dendrite structure in ei_unified checkpoints."
    )
    parser.add_argument("--config_path", required=True, type=Path)
    parser.add_argument("--checkpoint_path", required=True, type=Path)
    parser.add_argument("--layer_idx", type=int, default=0)
    parser.add_argument(
        "--output_json",
        type=Path,
        default=None,
        help="Optional path to save the full JSON results.",
    )
    parser.add_argument(
        "--plot_path",
        type=Path,
        default=None,
        help="Optional path to save a plot of pairwise I -> E_owner relations.",
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        default=None,
        help="Optional title override for the saved plot.",
    )
    args = parser.parse_args()

    config = _load_config(args.config_path)
    state_dict = torch.load(args.checkpoint_path, map_location="cpu")
    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Expected a state_dict-like checkpoint, got {type(state_dict).__name__}"
        )

    results = analyze_recurrent_inhibitory_structure(
        config=config,
        state_dict=state_dict,
        layer_idx=args.layer_idx,
    )
    _print_summary(results)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2))
    if args.plot_path is not None:
        _save_pair_relation_plot(
            results,
            args.plot_path,
            title=args.plot_title,
        )


if __name__ == "__main__":
    main()
