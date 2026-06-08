"""
Inhibitory specialization analyzer for recurrent E-I dendritic models.

For each E-I layer with a local I pool, this analyzer reads the
``branch_rec_inhibition`` connection weights of the excitatory population
and summarizes, for every (I, E) pair, how recurrent inhibition is
*routed* across compartments. The routing is captured by a per-pair
gate index::

    G(i, e) = f_soma(i, e) - f_distal(i, e)

where ``f_level(i, e)`` is the fraction of i's inhibitory synaptic weight
onto e that lands on compartments at that dendritic level, normalized so
the per-level fractions sum to 1 within each (i, e) pair. Soma = last
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
empty result and logs a skip message.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

from dendritic_modeling.analysis.core.base import AbstractAnalyzer
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
]


def _reshape_weight_matrix_by_compartment(
    w: torch.Tensor, n_E: int
) -> torch.Tensor | None:
    """Reshape ``(n_E * K_l, n_I)`` weights to ``(n_E, K_l, n_I)``."""
    out_features, _n_I = w.shape
    if n_E <= 0 or out_features % n_E != 0:
        return None
    branch_factor = out_features // n_E
    if branch_factor == 0:
        return None
    return w.reshape(n_E, branch_factor, -1)


def _compartment_mean_level_weights(
    compartment_weights: list[torch.Tensor | None],
) -> list[torch.Tensor | None]:
    """Average per-compartment weights into comparable per-level weights."""
    per_level: list[torch.Tensor | None] = []
    for w in compartment_weights:
        if w is None:
            per_level.append(None)
            continue
        per_level.append(w.sum(dim=1) / float(w.shape[1]))
    return per_level


def _fractions_per_level(
    e_population: torch.nn.Module,
    n_E: int,
) -> list[torch.Tensor] | None:
    """Return per-level inhibition fractions ``f_l`` of shape ``(n_E, n_I)``.

    For each branch layer of the excitatory population that carries a
    ``branch_rec_inhibition`` TopK layer, we read the effective weight
    ``W_l`` of shape ``(n_E * branch_factor_l, n_I)``, reshape it to
    ``(n_E, branch_factor_l, n_I)`` and sum over the branching axis to
    get total weight per (E-neuron, I-neuron) pair. Dividing by the
    per-level branch factor normalizes the per-level sums to a
    weight-per-synapse quantity so the levels are comparable across a
    wide dendrite and a narrow one.

    Returns ``None`` if the excitatory population has no recurrent
    inhibition at any branch level.
    """
    branch_layers = getattr(e_population, "branch_layers", None)
    if branch_layers is None or len(branch_layers) == 0:
        return None

    per_level: list[torch.Tensor] = []
    for lyr in branch_layers:
        rec_inh = getattr(lyr, "branch_rec_inhibition", None)
        if rec_inh is None:
            per_level.append(None)  # type: ignore[arg-type]
            continue
        with torch.no_grad():
            w = rec_inh.weight().detach().cpu()
        compartment_w = _reshape_weight_matrix_by_compartment(w, n_E=n_E)
        if compartment_w is None:
            return None
        per_level.append(compartment_w.sum(dim=1) / float(compartment_w.shape[1]))
    return per_level


def _compute_gate_index(
    per_level: list[torch.Tensor | None],
    eps: float = 1e-12,
) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
    """Turn per-level ``f_l`` tensors into ``G(i, e)`` and level fractions.

    ``G`` is returned in the ``(n_I, n_E)`` convention used by the
    downstream plotting code (rows are I-neurons, columns are E-neurons).
    ``level_fracs[l]`` is ``f_l / sum_l f_l`` in the same shape as the
    inputs, i.e. ``(n_E, n_I)``.
    """
    present = [f for f in per_level if f is not None]
    if len(present) < 2:
        return None
    stacked = torch.stack(present, dim=0)  # (L, n_E, n_I)
    total = stacked.sum(dim=0).clamp(min=eps)
    level_fracs = [stacked[i] / total for i in range(stacked.shape[0])]
    f_distal = level_fracs[0]
    f_soma = level_fracs[-1]
    G = (f_soma - f_distal).T  # (n_I, n_E)
    return G, level_fracs


def _classify_i_neurons(
    G: torch.Tensor,
    hi: float,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Compute ``mean_G_i``, ``std_G_i``, and the per-I class label."""
    mean_G = G.mean(dim=1)
    std_G = G.std(dim=1, unbiased=False)
    classes: list[str] = []
    for m, s in zip(mean_G.tolist(), std_G.tolist()):
        if s > sigma:
            classes.append("mixed-conditional")
        elif m > hi:
            classes.append("gate-specialist")
        elif m < -hi:
            classes.append("context-specialist")
        else:
            classes.append("diffuse")
    return mean_G, std_G, classes


def _pair_level_stats(G: torch.Tensor, hi: float) -> dict[str, float]:
    """Classify each (i, e) pair by its per-pair gate index directly.

    Returns percentages for: strong-soma (``G >= hi``), strong-distal
    (``G <= -hi``), balanced (``|G| < hi``), and the raw any-soma-bias
    tally (``G > 0``) that ignores the margin.
    """
    n = G.numel()
    if n == 0:
        return {
            "strong_soma_pct": 0.0,
            "strong_distal_pct": 0.0,
            "balanced_pct": 0.0,
            "any_soma_bias_pct": 0.0,
            "n_pairs": 0,
        }
    return {
        "strong_soma_pct": float((G >= hi).float().mean().item() * 100.0),
        "strong_distal_pct": float((G <= -hi).float().mean().item() * 100.0),
        "balanced_pct": float(((G > -hi) & (G < hi)).float().mean().item() * 100.0),
        "any_soma_bias_pct": float((G > 0).float().mean().item() * 100.0),
        "n_pairs": int(n),
    }


def _analyze_layer(
    e_population: torch.nn.Module,
    n_E: int,
    n_I: int,
    hi: float,
    sigma: float,
) -> dict[str, Any] | None:
    per_level = _fractions_per_level(e_population, n_E)
    if per_level is None:
        return None
    out = _compute_gate_index(per_level)
    if out is None:
        return None
    G, level_fracs = out
    mean_G, std_G, classes = _classify_i_neurons(G, hi=hi, sigma=sigma)
    class_counts = {c: int(classes.count(c)) for c in CLASS_ORDER}

    population_fracs: dict[str, float] = {}
    if len(level_fracs) >= 2:
        population_fracs["frac_distal_pop"] = float(level_fracs[0].mean().item())
        population_fracs["frac_soma_pop"] = float(level_fracs[-1].mean().item())
        if len(level_fracs) > 2:
            middle = torch.stack(level_fracs[1:-1], dim=0).mean(dim=0)
            population_fracs["frac_middle_pop"] = float(middle.mean().item())
        else:
            population_fracs["frac_middle_pop"] = 0.0

    return {
        "n_E": int(n_E),
        "n_I": int(n_I),
        "n_levels": len(level_fracs),
        "G": G.tolist(),
        "mean_G": mean_G.tolist(),
        "std_G": std_G.tolist(),
        "classes": classes,
        "class_counts": class_counts,
        "pair_level": _pair_level_stats(G, hi=hi),
        **population_fracs,
    }


def _population_graph_source_groups(core: torch.nn.Module) -> dict[str, list[str]]:
    groups = {"excitatory": [], "inhibitory": []}
    for cell in getattr(core, "cell_types", []) or []:
        name = getattr(cell, "name", None)
        polarity = str(getattr(cell, "polarity", "")).lower()
        if name is not None and polarity in groups:
            groups[polarity].append(str(name))
    return groups


def _record_source_summary(core: torch.nn.Module, record_key: str) -> dict[str, Any]:
    if not record_key.startswith("population_graph/"):
        return {}
    groups = _population_graph_source_groups(core)
    if not groups["inhibitory"]:
        return {}
    dims = {
        name: int(getattr(core, "_cell_dims", {}).get(name, 0))
        for name in groups["inhibitory"]
    }
    return {
        "source_populations": groups["inhibitory"],
        "source_population_dims": dims,
    }


def extract_recurrent_inhibitory_compartment_weights_from_state_dict(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
    n_E: int,
    weight_transform: str = "exp",
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
    """
    transform = weight_transform.lower()
    level_tensors: list[torch.Tensor] = []
    level = 0
    while True:
        key = f"{prefix}.branch_layers.{level}.branch_rec_inhibition.pre_w"
        if key not in state_dict:
            break
        pre_w = state_dict[key].detach().cpu().float()
        if transform == "exp":
            w = torch.exp(pre_w)
        elif transform == "softplus":
            w = torch.nn.functional.softplus(pre_w)
        elif transform == "relu":
            w = torch.relu(pre_w)
        elif transform == "identity":
            w = pre_w
        else:
            raise ValueError(f"Unsupported weight_transform: {weight_transform}")
        compartment_w = _reshape_weight_matrix_by_compartment(w, n_E=n_E)
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
    summary mirrors the analyzer JSON but keeps tensors in memory for
    downstream paper scripts.
    """
    if not compartment_weights:
        raise ValueError("Expected at least one level of compartment weights")
    n_E = int(compartment_weights[0].shape[0])
    n_I = int(compartment_weights[0].shape[2])
    per_level = _compartment_mean_level_weights(compartment_weights)
    out = _compute_gate_index(per_level)
    if out is None:
        raise ValueError("Need at least two valid inhibitory levels to compute G")
    G, level_fracs = out
    mean_G, std_G, classes = _classify_i_neurons(G, hi=hi, sigma=sigma)
    class_counts = {c: int(classes.count(c)) for c in CLASS_ORDER}
    population_fracs: dict[str, float] = {
        "frac_distal_pop": float(level_fracs[0].mean().item()),
        "frac_soma_pop": float(level_fracs[-1].mean().item()),
        "frac_middle_pop": (
            float(torch.stack(level_fracs[1:-1], dim=0).mean().item())
            if len(level_fracs) > 2
            else 0.0
        ),
    }
    frac_by_level = [lvl.T for lvl in level_fracs]
    return {
        "n_E": n_E,
        "n_I": n_I,
        "n_levels": len(level_fracs),
        "G": G,
        "mean_G": mean_G,
        "std_G": std_G,
        "classes": classes,
        "class_counts": class_counts,
        "pair_level": _pair_level_stats(G, hi=hi),
        "level_fracs": level_fracs,
        "frac_by_level_ie": frac_by_level,
        "frac_distal": frac_by_level[0],
        "frac_middle": (
            torch.stack(frac_by_level[1:-1], dim=0).mean(dim=0)
            if len(frac_by_level) > 2
            else torch.zeros_like(frac_by_level[0])
        ),
        "frac_soma": frac_by_level[-1],
        **population_fracs,
    }


def pair_stats_by_class(
    G: torch.Tensor,
    classes: list[str],
    hi: float = 0.1,
) -> dict[str, dict[str, float]]:
    """Per-class summary of the pair-level gate-index distribution."""
    out: dict[str, dict[str, float]] = {}
    for cname in CLASS_ORDER:
        idx = [i for i, c in enumerate(classes) if c == cname]
        if not idx:
            out[cname] = {"n_I": 0, "n_pairs": 0}
            continue
        vals = G[idx].reshape(-1)
        out[cname] = {
            "n_I": len(idx),
            "n_pairs": int(vals.numel()),
            "mean": float(vals.mean().item()),
            "std": float(vals.std(unbiased=False).item()),
            "q10": float(torch.quantile(vals, 0.10).item()),
            "q50": float(torch.quantile(vals, 0.50).item()),
            "q90": float(torch.quantile(vals, 0.90).item()),
            "strong_soma_pct": float((vals >= hi).float().mean().item() * 100.0),
            "strong_distal_pct": float((vals <= -hi).float().mean().item() * 100.0),
            "balanced_pct": float(
                ((vals > -hi) & (vals < hi)).float().mean().item() * 100.0
            ),
            "bimodality": float(
                2.0
                * min(
                    float((vals >= hi).float().mean().item()),
                    float((vals <= -hi).float().mean().item()),
                )
            ),
        }
    return out


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
            self.logger.info(
                "Skipping inhibitory specialization: no recurrent populations found"
            )
            return {}

        hi = float(self.params.gate_threshold)
        sigma = float(self.params.std_threshold)

        per_layer: dict[str, dict[str, Any]] = {}
        total_inhibitory_by_layer: dict[int | None, int] = {}
        for record in records:
            if record.polarity != "inhibitory":
                continue
            total_inhibitory_by_layer[record.layer_index] = (
                total_inhibitory_by_layer.get(record.layer_index, 0) + record.n_neurons
            )

        for record in records:
            if record.polarity != "excitatory":
                continue
            n_E = record.n_neurons
            n_I = total_inhibitory_by_layer.get(record.layer_index, 0)
            if record.key.startswith("population_graph/"):
                n_I = int(getattr(core, "_inh_rec_dim", n_I))
            if n_E == 0 or n_I == 0:
                continue
            entry = _analyze_layer(
                record.population, n_E=n_E, n_I=n_I, hi=hi, sigma=sigma
            )
            if entry is not None:
                entry.update(_record_source_summary(core, record.key))
                key = (
                    f"layer_{record.layer_index}"
                    if record.layer_index is not None
                    and record.population_name == "excitatory"
                    else record.key
                )
                per_layer[key] = entry

        if not per_layer:
            self.logger.info(
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
            os.makedirs(save_path, exist_ok=True)
            save_dict(results, save_path, f"{filename}.json")
            if self.params.save_figures:
                try:
                    self._save_figures(results, save_path, filename)
                except Exception as e:  # pragma: no cover - plotting is best-effort
                    self.logger.warning(
                        "Figure generation failed for inhibitory specialization: %s",
                        e,
                    )

        self.log_analysis_end("inhibitory_specialization", num_results=len(per_layer))
        return results

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

            for m, s, c in zip(mean_G, std_G, classes):
                ax_sc.scatter(
                    m,
                    s,
                    s=18,
                    color=class_colors.get(c, "#888888"),
                    edgecolors="white",
                    linewidths=0.3,
                    alpha=0.85,
                )
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
            ax_c.set_yticklabels(["gate", "mixed", "diff", "ctx"], fontsize=6)
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
