"""
Inhibitory specialization analyzer for recurrent E-I dendritic models.

For each E-I layer with a local I pool, this analyzer reads the
``branch_rec_inhibition`` connection weights of the excitatory population
and summarizes, for every (I, E) pair, how recurrent inhibition is
*routed* across compartments. The routing is captured by a per-pair
gate index::

    G(i, e) = f_soma(i, e) - f_distal(i, e)

where ``f_level(i, e)`` is the fraction of i's realized inhibitory synaptic
weight mass onto e that lands on compartments at that dendritic level,
normalized so the per-level fractions sum to 1 within each (i, e) pair
(see ``dendritic_modeling.analysis.utils.compartment_targeting``). Soma = last
branch level; distal = first branch level; middle levels are reported
separately but not used in the scalar index.

Two complementary views are produced:

*Per-I view.* Aggregating G over the E partners of each i, we compute
``mean G_i`` and ``std G_i`` and classify i into one of four functional
classes using thresholds (mean 0.1, std 0.15 by default):

- **gate-specialist** : ``mean_G > hi`` and ``std_G < sigma`` — i
  systematically targets the soma of its E partners, a structural motif
  consistent with broad output-gain modulation.
- **context-specialist** : ``mean_G < -hi`` and ``std_G < sigma`` — i
  systematically targets distal compartments, a structural motif
  consistent with dendritic context modulation.
- **mixed-conditional** : ``std_G > sigma`` — i is soma-like for some
  partners and distal-like for others.
- **diffuse** : ``|mean_G| <= hi`` and ``std_G < sigma`` — i has no
  strong compartment preference.

*Per-pair view.* Classifying each (i, e) pair directly by ``G(i, e)``,
we report the fraction of pairs that are soma-dominant, distal-dominant,
or balanced. This is the right view for "what fraction of individual
I→E targeting motifs are soma-biased versus distal-biased" and
reconciles with older per-pair dominance analyses that bypass the
per-I std threshold.

For models without an I population or without recurrent inhibition
(feedforward-only, or baseline RNN cores), the analyzer returns an
empty result and warns.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
from dendritic_modeling.analysis.utils.compartment_targeting import (
    TargetingClassLabels,
    apply_weight_transform,
    classify_sources,
    level_weight_mass,
    level_weight_per_compartment,
    pair_level_stats,
    pair_stats_by_class as shared_pair_stats_by_class,
    population_fraction_summary,
    reshape_by_compartment,
    targeting_index,
)
from dendritic_modeling.analysis.utils.effective_synapses import (
    effective_synapse_snapshot,
)
from dendritic_modeling.analysis.utils.recurrent_introspection import (
    iter_recurrent_populations,
)
from dendritic_modeling.config.analysis import InhibitorySpecializationAnalysisParams
from dendritic_modeling.models import BaseModel
from dendritic_modeling.utils import save_dict

logger = logging.getLogger(__name__)


CLASS_ORDER = [
    "gate-specialist",
    "mixed-conditional",
    "diffuse",
    "context-specialist",
    "disconnected",
]


_CLASS_LABELS = TargetingClassLabels(
    soma="gate-specialist",
    distal="context-specialist",
    mixed="mixed-conditional",
)


def _compartment_weights_per_level(
    e_population: torch.nn.Module,
    n_E: int,
) -> list[torch.Tensor | None] | None:
    """Return realized recurrent I->E weights as ``(n_E, K_l, n_I)`` per level.

    Levels without recurrent inhibition are ``None``; ``None`` is returned
    when the population has no branch layers or a level cannot be reshaped.
    """
    branch_layers = getattr(e_population, "branch_layers", None)
    if branch_layers is None or len(branch_layers) == 0:
        return None

    per_level: list[torch.Tensor | None] = []
    for lyr in branch_layers:
        rec_inh = getattr(lyr, "branch_rec_inhibition", None)
        if rec_inh is None:
            per_level.append(None)
            continue
        with torch.no_grad():
            w = effective_synapse_snapshot(rec_inh).effective_weight.cpu()
        compartment_w = reshape_by_compartment(w, n_target=n_E)
        if compartment_w is None:
            return None
        per_level.append(compartment_w)
    return per_level


def _summarize(
    compartment_weights: list[torch.Tensor | None],
    hi: float,
    sigma: float,
) -> dict[str, Any] | None:
    """Summarize targeting from per-level weights; ``None`` below two levels."""
    present = [w for w in compartment_weights if w is not None]
    if not present:
        return None
    out = targeting_index(level_weight_mass(compartment_weights))
    if out is None:
        return None
    G, level_fracs, connected = out
    mean_G, std_G, classes = classify_sources(
        G, hi=hi, sigma=sigma, labels=_CLASS_LABELS, connected=connected
    )
    frac_by_level = [lvl.T for lvl in level_fracs]
    return {
        "n_E": int(present[0].shape[0]),
        "n_I": int(present[0].shape[2]),
        "n_levels": len(level_fracs),
        "G": G,
        "connected_pairs": connected,
        "mean_G": mean_G,
        "std_G": std_G,
        "classes": classes,
        "class_counts": {c: int(classes.count(c)) for c in CLASS_ORDER},
        "pair_level": pair_level_stats(G, hi=hi, connected=connected),
        "level_fracs": level_fracs,
        "frac_by_level_ie": frac_by_level,
        "frac_distal": frac_by_level[0],
        "frac_middle": (
            torch.stack(frac_by_level[1:-1], dim=0).mean(dim=0)
            if len(frac_by_level) > 2
            else torch.zeros_like(frac_by_level[0])
        ),
        "frac_soma": frac_by_level[-1],
        **population_fraction_summary(level_fracs, connected=connected.T),
        "per_compartment_control": _per_compartment_control(compartment_weights),
    }


def _per_compartment_control(
    compartment_weights: list[torch.Tensor | None],
) -> dict[str, float]:
    """Level fractions of each level's mass divided by its compartment count."""
    out = targeting_index(level_weight_per_compartment(compartment_weights))
    if out is None:
        return {}
    _G, level_fracs, connected = out
    return population_fraction_summary(level_fracs, connected=connected.T)


def _analyze_layer(
    e_population: torch.nn.Module,
    n_E: int,
    n_I: int,
    hi: float,
    sigma: float,
) -> dict[str, Any] | None:
    compartment_weights = _compartment_weights_per_level(e_population, n_E)
    if compartment_weights is None:
        return None
    summary = _summarize(compartment_weights, hi=hi, sigma=sigma)
    if summary is None:
        return None
    return {
        "n_E": int(n_E),
        "n_I": int(n_I),
        "n_levels": summary["n_levels"],
        "G": summary["G"].tolist(),
        "connected_pairs": summary["connected_pairs"].tolist(),
        "mean_G": summary["mean_G"].tolist(),
        "std_G": summary["std_G"].tolist(),
        "classes": summary["classes"],
        "class_counts": summary["class_counts"],
        "pair_level": summary["pair_level"],
        "frac_distal_pop": summary["frac_distal_pop"],
        "frac_soma_pop": summary["frac_soma_pop"],
        "frac_middle_pop": summary["frac_middle_pop"],
        "per_compartment_control": summary["per_compartment_control"],
    }


def _population_network_source_groups(
    core: torch.nn.Module,
    record_key: str,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    _, _, suffix = record_key.partition("/")
    layer_name, sep, _population_name = suffix.partition(".")
    if not sep:
        return {"excitatory": [], "inhibitory": []}, {}

    for layer in getattr(core, "layers", []) or []:
        config = getattr(layer, "config", None)
        if str(getattr(config, "name", "")) != layer_name:
            continue
        groups = {"excitatory": [], "inhibitory": []}
        dims: dict[str, int] = {}
        for population in getattr(layer, "population_definitions", []) or []:
            name = getattr(population, "name", None)
            polarity = str(getattr(population, "polarity", "")).lower()
            if name is None or polarity not in groups:
                continue
            name_str = str(name)
            groups[polarity].append(name_str)
            dims[name_str] = int(
                getattr(layer, "_population_dims", {}).get(name_str, 0)
            )
        return groups, dims

    return {"excitatory": [], "inhibitory": []}, {}


def _record_source_summary(core: torch.nn.Module, record_key: str) -> dict[str, Any]:
    if record_key.startswith("population_network/"):
        groups, dims = _population_network_source_groups(core, record_key)
    else:
        return {}

    if not groups["inhibitory"]:
        return {}
    return {
        "source_populations": groups["inhibitory"],
        "source_population_dims": dims,
    }


def extract_recurrent_inhibitory_compartment_weights_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    n_E: int,
    weight_transform: str = "exp",
    active_masks: list[torch.Tensor] | None = None,
) -> list[torch.Tensor]:
    """Extract recurrent I→E weights as ``(n_E, K_l, n_I)`` tensors per level.

    Parameters
    ----------
    state_dict:
        Checkpoint state dict containing ``branch_rec_inhibition.pre_w`` tensors.
    prefix:
        Path prefix up to the excitatory population, e.g.
        ``"core_network.layers.0.e_population"``.
    n_E:
        Number of excitatory soma units.
    weight_transform:
        Effective transform applied to ``pre_w``. Supported values are
        ``"exp"``, ``"softplus"``, ``"relu"``, and ``"identity"``.
    active_masks:
        Exact dense masks used by the analyzed forward, one per level. A raw
        state dict contains candidate parameters but not enough information to
        recover every structured/dynamic/stochastic realization. The masks are
        therefore required rather than silently treating candidates as active.
    """
    if active_masks is None:
        raise ValueError(
            "active_masks are required: checkpoint pre_w tensors are dense "
            "candidate parameters, not realized synapses"
        )
    level_tensors: list[torch.Tensor] = []
    level = 0
    while True:
        key = f"{prefix}.branch_layers.{level}.branch_rec_inhibition.pre_w"
        if key not in state_dict:
            break
        pre_w = state_dict[key].detach().cpu().float()
        if level >= len(active_masks):
            raise ValueError(f"Missing active mask for inhibitory level {level}")
        mask = active_masks[level].detach().cpu().to(dtype=pre_w.dtype)
        if mask.shape != pre_w.shape:
            raise ValueError(
                f"Mask shape at level {level} is {tuple(mask.shape)}, expected "
                f"{tuple(pre_w.shape)}"
            )
        w = apply_weight_transform(pre_w, weight_transform) * mask
        compartment_w = reshape_by_compartment(w, n_target=n_E)
        if compartment_w is None:
            raise ValueError(
                f"Could not reshape inhibitory weights at level {level} for n_E={n_E}"
            )
        level_tensors.append(compartment_w)
        level += 1
    if not level_tensors:
        raise ValueError(
            f"No recurrent inhibitory weights found under prefix '{prefix}'"
        )
    return level_tensors


def summarize_inhibitory_specialization_from_compartment_weights(
    compartment_weights: list[torch.Tensor],
    hi: float = 0.1,
    sigma: float = 0.15,
) -> dict[str, Any]:
    """Summarize inhibitory specialization from per-compartment weight tensors.

    ``compartment_weights[l]`` must have shape ``(n_E, K_l, n_I)``. The
    analyzer computes its JSON with the same function, so the two agree; this
    helper keeps the tensors in memory for downstream analysis.
    """
    if not compartment_weights:
        raise ValueError("Expected at least one level of compartment weights")
    summary = _summarize(list(compartment_weights), hi=hi, sigma=sigma)
    if summary is None:
        raise ValueError("Need at least two valid inhibitory levels to compute G")
    return summary


def pair_stats_by_class(
    G: torch.Tensor,
    classes: list[str],
    hi: float = 0.1,
    connected: torch.Tensor | None = None,
) -> dict[str, dict[str, float]]:
    """Per-class summary of the pair-level gate-index distribution."""
    return shared_pair_stats_by_class(
        G, classes, CLASS_ORDER, hi=hi, connected=connected, count_key="n_I"
    )


class InhibitorySpecializationAnalyzer(AbstractAnalyzer):
    """Static-weight analysis of inhibitory-to-excitatory compartment targeting.

    Runs on any recurrent E-I dendritic model that carries at least one
    layer whose excitatory population has a local I pool wired through
    ``branch_rec_inhibition``. For each such layer the analyzer writes:

    - a JSON report with ``G``, per-I classification, per-class counts
      and the per-pair dominance percentages, and
    - a single-figure PDF with three panels (G heatmap sorted by
      ``mean_G``, ``(mean_G, std_G)`` scatter coloured by class, and a
      horizontal bar chart of class counts).

    The analyzer only reads trained weights — no forward pass through
    the data — so it is cheap and deterministic, suitable to run at the
    end of every training job by default.
    """

    def __init__(self, params: InhibitorySpecializationAnalysisParams | None = None):
        super().__init__("InhibitorySpecializationAnalyzer")
        if params is None:
            params = InhibitorySpecializationAnalysisParams()
        self.params = params

    def analyze(
        self,
        model: BaseModel,
        data: torch.utils.data.Dataset | None = None,
        device: str = "cpu",
        save_path: str | None = None,
        filename: str = "inhibitory_specialization",
        **kwargs: Any,
    ) -> dict[str, Any]:
        self.log_analysis_start("inhibitory_specialization")

        core = getattr(model, "core_network", model)
        records = list(iter_recurrent_populations(core))
        if not records:
            self.logger.warning(
                "Skipping inhibitory specialization: no recurrent populations found"
            )
            return {}

        hi = float(self.params.gate_threshold)
        sigma = float(self.params.std_threshold)

        total_inhibitory_by_layer = self._inhibitory_counts_by_layer(records)
        per_layer = self._analyze_excitatory_records(
            core=core,
            records=records,
            total_inhibitory_by_layer=total_inhibitory_by_layer,
            hi=hi,
            sigma=sigma,
        )

        if not per_layer:
            self.logger.warning(
                "Skipping inhibitory specialization: no E-I layer with "
                "recurrent inhibition found"
            )
            return {}

        results: dict[str, Any] = {
            "thresholds": {"gate_threshold": hi, "std_threshold": sigma},
            "class_order": CLASS_ORDER,
            "layers": per_layer,
        }

        if save_path is not None:
            self._save_results(results, save_path, filename)

        self.log_analysis_end("inhibitory_specialization", num_results=len(per_layer))
        return results

    @staticmethod
    def _inhibitory_counts_by_layer(records: list[Any]) -> dict[int | None, int]:
        total_inhibitory_by_layer: dict[int | None, int] = {}
        for record in records:
            if record.polarity != "inhibitory":
                continue
            total_inhibitory_by_layer[record.layer_index] = (
                total_inhibitory_by_layer.get(record.layer_index, 0) + record.n_neurons
            )
        return total_inhibitory_by_layer

    def _analyze_excitatory_records(
        self,
        core: torch.nn.Module,
        records: list[Any],
        total_inhibitory_by_layer: dict[int | None, int],
        hi: float,
        sigma: float,
    ) -> dict[str, dict[str, Any]]:
        per_layer: dict[str, dict[str, Any]] = {}
        for record in records:
            if record.polarity != "excitatory":
                continue
            entry = self._analyze_excitatory_record(
                core=core,
                record=record,
                n_I=total_inhibitory_by_layer.get(record.layer_index, 0),
                hi=hi,
                sigma=sigma,
            )
            if entry is not None:
                per_layer[self._record_result_key(record)] = entry
        return per_layer

    @staticmethod
    def _analyze_excitatory_record(
        core: torch.nn.Module,
        record: Any,
        n_I: int,
        hi: float,
        sigma: float,
    ) -> dict[str, Any] | None:
        n_E = record.n_neurons
        if n_E == 0 or n_I == 0:
            return None
        entry = _analyze_layer(record.population, n_E=n_E, n_I=n_I, hi=hi, sigma=sigma)
        if entry is not None:
            entry.update(_record_source_summary(core, record.key))
        return entry

    @staticmethod
    def _record_result_key(record: Any) -> str:
        if record.layer_index is not None and record.population_name == "excitatory":
            return f"layer_{record.layer_index}"
        return record.key

    def _save_results(
        self,
        results: dict[str, Any],
        save_path: str,
        filename: str,
    ) -> None:
        os.makedirs(save_path, exist_ok=True)
        save_dict(results, save_path, f"{filename}.json")
        if not self.params.save_figures:
            return
        try:
            self._save_figures(results, save_path, filename)
        except Exception as e:  # pragma: no cover - plotting is best-effort
            self.logger.warning(
                "Figure generation failed for inhibitory specialization: %s",
                e,
            )

    def _save_figures(
        self,
        results: dict[str, Any],
        save_path: str,
        filename: str,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        class_colors = {
            "gate-specialist": "#D62728",
            "mixed-conditional": "#E8943A",
            "diffuse": "#888888",
            "context-specialist": "#1F77B4",
            "disconnected": "#D9D9D9",
        }
        # Marker shapes repeat the class, so the scatter reads without colour.
        class_markers = {
            "gate-specialist": "^",
            "mixed-conditional": "s",
            "diffuse": "o",
            "context-specialist": "v",
            "disconnected": "X",
        }

        layers = results["layers"]
        n_layers = len(layers)
        fig, axes = plt.subplots(
            n_layers,
            3,
            figsize=(9.0, 2.3 * max(n_layers, 1)),
            gridspec_kw={
                "width_ratios": [2.0, 1.0, 0.9],
                "wspace": 0.42,
                "hspace": 0.55,
            },
            squeeze=False,
        )

        for li, (layer_key, entry) in enumerate(layers.items()):
            G = np.asarray(entry["G"])
            mean_G = np.asarray(entry["mean_G"])
            std_G = np.asarray(entry["std_G"])
            classes = entry["classes"]

            i_order = np.argsort(mean_G)[::-1]
            e_order = np.argsort(G.mean(axis=0))[::-1]
            G_s = G[np.ix_(i_order, e_order)]

            ax_h, ax_sc, ax_c = axes[li]

            im = ax_h.imshow(G_s, aspect="auto", cmap="RdBu_r", vmin=-0.6, vmax=0.6)
            ax_h.set_xlabel("E-neuron (sorted)", fontsize=7)
            ax_h.set_ylabel("I-neuron (sorted)", fontsize=7)
            ax_h.set_title(
                f"{layer_key}: G(i, e)  "
                f"n_E={entry['n_E']}, n_I={entry['n_I']}, "
                f"L={entry['n_levels']}",
                fontsize=8,
            )
            cbar = fig.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02)
            cbar.ax.tick_params(labelsize=6)
            if li == 0:
                cbar.set_label("G", fontsize=7)

            for cname in CLASS_ORDER:
                members = [k for k, c in enumerate(classes) if c == cname]
                if not members:
                    continue
                ax_sc.scatter(
                    mean_G[members],
                    std_G[members],
                    s=18,
                    marker=class_markers[cname],
                    color=class_colors[cname],
                    edgecolors="black",
                    linewidths=0.3,
                    alpha=0.85,
                    label=cname,
                )
            ax_sc.legend(fontsize=5, loc="upper right", frameon=False)
            ax_sc.axvline(x=0, color="#999", ls="--", lw=0.4)
            ax_sc.axhline(
                y=results["thresholds"]["std_threshold"],
                color="#999",
                ls=":",
                lw=0.4,
            )
            ax_sc.set_xlim(-0.6, 0.6)
            ax_sc.set_ylim(0, max(0.4, float(std_G.max()) * 1.1))
            ax_sc.set_xlabel(r"mean $G_i$", fontsize=7)
            ax_sc.set_ylabel(r"std $G_i$", fontsize=7)
            ax_sc.set_title("per-I class scatter", fontsize=8)

            counts = [entry["class_counts"].get(c, 0) for c in CLASS_ORDER]
            ax_c.barh(
                range(len(CLASS_ORDER)),
                counts,
                color=[class_colors[c] for c in CLASS_ORDER],
                edgecolor="white",
                linewidth=0.4,
            )
            ax_c.set_yticks(range(len(CLASS_ORDER)))
            ax_c.set_yticklabels(["gate", "mixed", "diff", "ctx", "none"], fontsize=6)
            ax_c.invert_yaxis()
            ax_c.set_xlabel("# I", fontsize=7)
            for i, v in enumerate(counts):
                ax_c.text(v + 0.3, i, str(v), va="center", fontsize=6)

        fig.suptitle("Inhibitory specialization (per-E-I layer)", fontsize=9, y=0.995)
        out_path = os.path.join(save_path, f"{filename}.pdf")
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)


__all__ = [
    "CLASS_ORDER",
    "InhibitorySpecializationAnalyzer",
    "extract_recurrent_inhibitory_compartment_weights_from_state_dict",
    "pair_stats_by_class",
    "summarize_inhibitory_specialization_from_compartment_weights",
]
