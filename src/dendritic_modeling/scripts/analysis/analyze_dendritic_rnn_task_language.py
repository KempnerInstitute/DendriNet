"""Assemble the complete four-cell C1-C4 delayed-comparison evidence gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from dendritic_modeling.analysis.statistics import bootstrap_mean_interval
from dendritic_modeling.scripts.analysis.analyze_dendritic_rnn_extreme_cube import (
    N_REPLICATES,
    exact_two_sided_sign_flip,
    holm_adjust,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_counterfactual_bank import (
    ROLE_CELLS,
    _resolve_repo_path,
    _sha256,
    _strict_json,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_stimulus_blanks import (
    POLICIES,
    load_study_config as load_blank_study_config,
)

STUDY_SCHEMA = "dendritic-rnn-task-language-study-v1"
STATISTICS_SCHEMA = "dendritic-rnn-task-language-statistics-v1"
MARKER_SCHEMA = "dendritic-rnn-task-language-analysis-marker-v1"
FIGURE_SOURCE_SCHEMA = "dendritic-rnn-task-language-figure-source-v1"
COMPONENTS = ("C1", "C2", "C3", "C4")
LANGUAGE_GATES = ("both_stimulus_identities", "two_stimulus_comparison")


class TaskLanguageAnalysisError(RuntimeError):
    """Raised when the exact forty-row task-language evidence is incomplete."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _subseed(label: str, analysis_seed: int) -> int:
    encoded = f"task-language-v1\0{analysis_seed}\0{label}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def load_study_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve(strict=True)
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise TaskLanguageAnalysisError("Invalid task-language YAML") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != STUDY_SCHEMA:
        raise TaskLanguageAnalysisError("Unsupported task-language study schema")
    analysis = payload.get("analysis")
    sources = payload.get("sources")
    if not isinstance(analysis, Mapping) or not isinstance(sources, Mapping):
        raise TaskLanguageAnalysisError("Task-language study config is incomplete")
    roles = {role: tuple(cell) for role, cell in analysis.get("roles", {}).items()}
    if roles != ROLE_CELLS:
        raise TaskLanguageAnalysisError("Task-language role design changed")
    if analysis.get("proposal_first_rank_oracle") != 0.875:
        raise TaskLanguageAnalysisError("Proposal first-rank oracle changed")
    return payload, path


def _finite_bank_first_rank_oracle(index: Mapping[str, Any]) -> float:
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    trials = index.get("trials")
    if not isinstance(trials, list) or len(trials) != 8192:
        raise TaskLanguageAnalysisError("Ordinary index lacks 8,192 trials")
    for row in trials:
        counts[int(row["frequency1_index"])][int(row["target"])] += 1
    return sum(max(values) for values in counts.values()) / len(trials)


def _component_summary(
    values: list[float],
    *,
    label: str,
    analysis_seed: int,
    bootstrap_draws: int,
) -> dict[str, Any]:
    bootstrap = bootstrap_mean_interval(
        values,
        draws=bootstrap_draws,
        seed=_subseed(label, analysis_seed),
        require_n=N_REPLICATES,
    ).to_dict()
    sign_flip = exact_two_sided_sign_flip(values)
    positive = bool(
        bootstrap["mean"] > 0.0
        and bootstrap["lower"] > 0.0
        and sign_flip["p_value"] <= 0.05
    )
    return {
        "values_by_replicate": values,
        "mean": bootstrap["mean"],
        "standard_deviation": bootstrap["standard_deviation"],
        "bootstrap_95_percent_ci": [bootstrap["lower"], bootstrap["upper"]],
        "bootstrap": bootstrap,
        "exact_two_sided_sign_flip": sign_flip,
        "support_criteria": {
            "positive_mean": bool(bootstrap["mean"] > 0.0),
            "strictly_positive_bootstrap_lower": bool(bootstrap["lower"] > 0.0),
            "raw_p_at_most_0.05": bool(sign_flip["p_value"] <= 0.05),
            "unadjusted_positive_support": positive,
        },
    }


def analyze_components(
    rows_by_role: Mapping[str, list[Mapping[str, float]]],
    *,
    analysis_seed: int,
    bootstrap_draws: int,
    family_alpha: float,
) -> dict[str, Any]:
    """Apply component support and the one authoritative eight-member Holm gate."""

    if set(rows_by_role) != set(ROLE_CELLS):
        raise TaskLanguageAnalysisError("The exact four roles are required")
    roles = {}
    raw_language_p = {}
    for role in ROLE_CELLS:
        rows = rows_by_role[role]
        if len(rows) != N_REPLICATES:
            raise TaskLanguageAnalysisError(f"{role} requires ten seeds")
        components = {}
        for component in COMPONENTS:
            values = [float(row[component]) for row in rows]
            components[component] = _component_summary(
                values,
                label=f"{role}:{component}",
                analysis_seed=analysis_seed,
                bootstrap_draws=bootstrap_draws,
            )
        p = {
            component: components[component]["exact_two_sided_sign_flip"]["p_value"]
            for component in COMPONENTS
        }
        component_support = {
            component: components[component]["support_criteria"][
                "unadjusted_positive_support"
            ]
            for component in COMPONENTS
        }
        raw_language_p[f"{role}:both_stimulus_identities"] = max(p["C1"], p["C2"])
        raw_language_p[f"{role}:two_stimulus_comparison"] = max(
            p["C1"], p["C2"], p["C3"]
        )
        roles[role] = {
            "cell": list(ROLE_CELLS[role]),
            "components": components,
            "unadjusted_component_conjunctions": {
                "both_stimulus_identities": bool(
                    component_support["C1"] and component_support["C2"]
                ),
                "two_stimulus_comparison": bool(
                    component_support["C1"]
                    and component_support["C2"]
                    and component_support["C3"]
                ),
            },
            "C4_role": "out_of_distribution_deletion_diagnostic_non_authorizing",
        }
    holm = holm_adjust(raw_language_p)
    for role in ROLE_CELLS:
        decisions = {}
        for gate in LANGUAGE_GATES:
            label = f"{role}:{gate}"
            components_pass = roles[role]["unadjusted_component_conjunctions"][gate]
            decisions[gate] = {
                "raw_intersection_union_p_value": raw_language_p[label],
                "holm_adjusted_p_value": holm[label]["adjusted_p_value"],
                "required_components_have_positive_support": components_pass,
                "holm_rejects_at_family_alpha": holm[label]["reject_at_family_alpha"],
                "authorized": bool(
                    components_pass and holm[label]["adjusted_p_value"] <= family_alpha
                ),
            }
        roles[role]["language_decisions"] = decisions
    return {
        "schema_version": STATISTICS_SCHEMA,
        "created_utc": _utc_now(),
        "analysis_plan": {
            "independent_unit": "trained_model_seed",
            "n_independent_units_per_cell": N_REPLICATES,
            "component_definitions": {
                "C1": "ordinary accuracy - 0.5",
                "C2": "joint-supported counterfactual central difference",
                "C3": "ordinary accuracy - max(0.875, finite-bank first-rank oracle)",
                "C4": "ordinary accuracy - second-stimulus-blank accuracy",
            },
            "authoritative_family": (
                "eight cell-by-language intersection-union p-values: four cells "
                "times two language statements"
            ),
            "family_alpha": family_alpha,
            "C4_in_authoritative_family": False,
            "bootstrap_draws": bootstrap_draws,
            "analysis_seed": analysis_seed,
        },
        "roles": roles,
        "holm_8_family": {
            "alpha": family_alpha,
            "members": holm,
        },
        "completeness": {
            "expected_cell_seed_rows": 40,
            "validated_cell_seed_rows": 40,
            "complete": True,
            "trial_resampling_used": False,
        },
    }


def _collect_rows(
    task_config: Mapping[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], list[dict[str, Any]]]:
    source_config_path = _resolve_repo_path(
        str(task_config["sources"]["stimulus_blank_config"])
    )
    blank_config, _ = load_blank_study_config(source_config_path)
    blank_root = _resolve_repo_path(str(blank_config["study"]["output_root"]))
    c2_path = _resolve_repo_path(
        str(task_config["sources"]["counterfactual_statistics"])
    )
    c2 = _strict_json(c2_path)
    if (
        c2.get("schema_version")
        != "dendritic-rnn-counterfactual-psychometric-statistics-v1"
    ):
        raise TaskLanguageAnalysisError("Unsupported counterfactual statistics")
    rows_by_role = {role: [] for role in ROLE_CELLS}
    provenance = []
    for replicate in range(N_REPLICATES):
        for role in ROLE_CELLS:
            role_dir = blank_root / f"r{replicate:03d}" / role
            marker = _strict_json(role_dir / "COMPLETE.json")
            result_path = Path(str(marker["result"]["path"]))
            if result_path != role_dir / "result.json":
                raise TaskLanguageAnalysisError("Blank public result path mismatch")
            if _sha256(result_path) != marker["result"]["sha256"]:
                raise TaskLanguageAnalysisError("Blank result hash mismatch")
            result = _strict_json(result_path)
            if (
                result.get("role") != role
                or result.get("replicate") != f"r{replicate:03d}"
                or result.get("audits", {}).get("first_blank_pair_identity")
                != "PASSED_BITWISE"
                or result.get("audits", {}).get("state_immutability") != "PASSED"
            ):
                raise TaskLanguageAnalysisError("Blank result audit failed")
            index_path = Path(result["bank"]["index_path"])
            if _sha256(index_path) != result["bank"]["index_sha256"]:
                raise TaskLanguageAnalysisError("Blank bank index hash changed")
            index = _strict_json(index_path)
            finite_oracle = _finite_bank_first_rank_oracle(index)
            threshold = max(
                float(task_config["analysis"]["proposal_first_rank_oracle"]),
                finite_oracle,
            )
            ordinary = float(result["ordinary"]["accuracy"])
            first_blank = float(result["policies"][POLICIES[0]]["accuracy"])
            second_blank = float(result["policies"][POLICIES[1]]["accuracy"])
            if first_blank != 0.5:
                raise TaskLanguageAnalysisError(
                    "First-blank accuracy must be exactly 0.5"
                )
            c2_value = float(c2["roles"][role]["values_by_replicate"][replicate])
            row = {
                "C1": ordinary - 0.5,
                "C2": c2_value,
                "C3": ordinary - threshold,
                "C4": ordinary - second_blank,
            }
            rows_by_role[role].append(row)
            provenance.append(
                {
                    "replicate": f"r{replicate:03d}",
                    "role": role,
                    "run_id": result["run_id"],
                    "blank_result": {
                        "path": str(result_path),
                        "sha256": _sha256(result_path),
                    },
                    "ordinary_accuracy": ordinary,
                    "finite_bank_first_rank_lookup_accuracy": finite_oracle,
                    "conservative_first_rank_threshold": threshold,
                    "first_stimulus_blank_accuracy": first_blank,
                    "second_stimulus_blank_accuracy": second_blank,
                    "components": row,
                }
            )
    return rows_by_role, c2, provenance


def _atomic_figure(fig: Any, path: Path, **kwargs: Any) -> None:
    temporary = path.with_name(f".{path.stem}.{os.getpid()}{path.suffix}")
    fig.savefig(temporary, **kwargs)
    temporary.replace(path)


def render_figures(
    statistics: Mapping[str, Any],
    c2: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    titles = {
        "hierarchical_aligned": "Hierarchical aligned",
        "hierarchical_reverse": "Hierarchical reverse",
        "reversed_aligned": "Reversed aligned",
        "reversed_reverse": "Reversed reverse",
    }
    component_titles = {
        "C1": "Delayed first-stimulus memory",
        "C2": "Counterfactual second-stimulus use",
        "C3": "Beyond first-rank-only rule",
        "C4": "Second-stimulus deletion",
    }

    fig8, axes = plt.subplots(2, 2, figsize=(8.3, 5.8), constrained_layout=True)
    aligned = statistics["roles"]["hierarchical_aligned"]
    for ax, component in zip(axes.flat, COMPONENTS, strict=True):
        row = aligned["components"][component]
        values = np.asarray(row["values_by_replicate"])
        interval = np.asarray(row["bootstrap_95_percent_ci"])
        ax.scatter(values, np.linspace(-0.12, 0.12, 10), color="#5277a8", alpha=0.75)
        ax.plot(interval, [0, 0], color="black", linewidth=3)
        ax.scatter(
            [row["mean"]], [0], marker="D", s=55, color="#d17a17", edgecolor="black"
        )
        ax.axvline(0, color="0.4", linestyle="--")
        ax.set_yticks([])
        ax.set_title(f"{component}: {component_titles[component]}", fontsize=9)
        ax.grid(axis="x", alpha=0.2)
        ax.text(
            0.98,
            0.08,
            f"mean {row['mean']:.3g}\np={row['exact_two_sided_sign_flip']['p_value']:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
        )
    fig8.suptitle("Aligned hierarchical task-language components", fontweight="bold")
    fig8.text(
        0.5,
        -0.02,
        "n=10 trained-model seeds. C4 is an out-of-distribution deletion diagnostic and does not authorize language.",
        ha="center",
        fontsize=10,
    )
    fig8_pdf = output_dir / "figure_8_task_language_gate.pdf"
    fig8_png = output_dir / "figure_8_task_language_gate.png"
    _atomic_figure(fig8, fig8_pdf, bbox_inches="tight")
    _atomic_figure(fig8, fig8_png, bbox_inches="tight", dpi=220)
    plt.close(fig8)

    fig = plt.figure(figsize=(12.0, 7.8), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1.0))
    surface_grid = grid[0, 0].subgridspec(2, 2)
    colors = {2: "#3366a8", 3: "#c84b31"}
    for index, role in enumerate(ROLE_CELLS):
        ax = fig.add_subplot(surface_grid[index // 2, index % 2])
        surfaces = c2["roles"][role]["psychometric_surface_by_seed"]
        for first_rank in (2, 3):
            curves = []
            for surface in surfaces:
                curve = [
                    row["mean_class1_probability"]
                    for row in surface
                    if row["receiver_first_rank"] == first_rank
                ]
                curves.append(curve)
            ax.plot(
                range(1, 5),
                np.mean(curves, axis=0),
                color=colors[first_rank],
                marker="o",
                linewidth=2,
                label=f"f1 rank {first_rank}",
            )
        ax.set_title(titles[role], fontsize=10)
        ax.set_xticks(range(1, 5))
        ax.set_ylim(-0.03, 1.03)
        if index // 2 == 1:
            ax.set_xlabel("Counterfactual f2 rank")
        if index % 2 == 0:
            ax.set_ylabel("P(class 1)")
        if index == 0:
            ax.legend(frameon=False, fontsize=9)
            ax.text(
                -0.22, 1.15, "A", transform=ax.transAxes, fontsize=13, fontweight="bold"
            )
        ax.grid(alpha=0.2)

    ax_c2 = fig.add_subplot(grid[0, 1])
    positions = np.arange(4)[::-1]
    for y, role in zip(positions, ROLE_CELLS, strict=True):
        row = statistics["roles"][role]["components"]["C2"]
        values = np.asarray(row["values_by_replicate"])
        ax_c2.scatter(values, y + np.linspace(-0.1, 0.1, 10), alpha=0.7)
        ax_c2.plot(row["bootstrap_95_percent_ci"], [y, y], color="black", linewidth=3)
        ax_c2.scatter([row["mean"]], [y], marker="D", s=55, edgecolor="black")
    ax_c2.axvline(0, color="0.4", linestyle="--")
    ax_c2.set_yticks(positions, [titles[role] for role in ROLE_CELLS], fontsize=9)
    ax_c2.set_xlabel("C2 central difference")
    ax_c2.set_title("Counterfactual second-stimulus evidence")
    ax_c2.grid(axis="x", alpha=0.2)
    ax_c2.text(
        -0.13, 1.08, "B", transform=ax_c2.transAxes, fontsize=13, fontweight="bold"
    )

    ax_table = fig.add_subplot(grid[1, 0])
    ax_table.axis("off")
    table_rows = []
    cell_colors = []
    for role in ROLE_CELLS:
        values = []
        colors_row = []
        for component in COMPONENTS:
            row = statistics["roles"][role]["components"][component]
            supported = row["support_criteria"]["unadjusted_positive_support"]
            p_value = row["exact_two_sided_sign_flip"]["p_value"]
            values.append(
                f"{row['mean']:.3g}\n"
                f"p={p_value:.2g} {'pass' if supported else 'fail'}"
            )
            colors_row.append("#d9ead3" if supported else "#f4cccc")
        table_rows.append(values)
        cell_colors.append(colors_row)
    table = ax_table.table(
        cellText=table_rows,
        rowLabels=[titles[role] for role in ROLE_CELLS],
        colLabels=list(COMPONENTS),
        cellColours=cell_colors,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.0, 2.35)
    ax_table.set_title("Component means and unadjusted support", pad=18)
    ax_table.text(
        -0.08, 1.06, "C", transform=ax_table.transAxes, fontsize=13, fontweight="bold"
    )

    ax_gate = fig.add_subplot(grid[1, 1])
    labels = []
    adjusted = []
    authorized = []
    for role in ROLE_CELLS:
        for gate in LANGUAGE_GATES:
            row = statistics["roles"][role]["language_decisions"][gate]
            labels.append(f"{titles[role]}\n{gate.replace('_', ' ')}")
            adjusted.append(row["holm_adjusted_p_value"])
            authorized.append(row["authorized"])
    y = np.arange(8)[::-1]
    ax_gate.scatter(
        adjusted,
        y,
        c=["#3b8d5a" if value else "#b44b4b" for value in authorized],
        marker="s",
        s=55,
    )
    ax_gate.axvline(
        statistics["analysis_plan"]["family_alpha"], color="0.3", linestyle="--"
    )
    ax_gate.set_xscale("log")
    ax_gate.set_yticks(y, labels, fontsize=9.5)
    ax_gate.set_xlabel("Authoritative Holm-adjusted p-value")
    ax_gate.set_title("Eight-member language family")
    ax_gate.grid(axis="x", which="both", alpha=0.2)
    ax_gate.text(
        0.98,
        0.03,
        "40/40 cell-seed rows validated",
        transform=ax_gate.transAxes,
        ha="right",
        fontsize=9,
        color="0.30",
    )
    ax_gate.text(
        -0.13, 1.06, "D", transform=ax_gate.transAxes, fontsize=13, fontweight="bold"
    )
    fig.suptitle(
        "Task-language certification across the primary placement-timescale quartet",
        fontsize=12,
        fontweight="bold",
    )
    fig9_pdf = output_dir / "figure_9_quartet_task_language.pdf"
    fig9_png = output_dir / "figure_9_quartet_task_language.png"
    _atomic_figure(fig, fig9_pdf, bbox_inches="tight")
    _atomic_figure(fig, fig9_png, bbox_inches="tight", dpi=220)
    plt.close(fig)
    return {
        "figure8_pdf": str(fig8_pdf),
        "figure8_png": str(fig8_png),
        "figure9_pdf": str(fig9_pdf),
        "figure9_png": str(fig9_png),
    }


def run_analysis(*, study_config_path: Path) -> dict[str, Any]:
    config, study_config_path = load_study_config(study_config_path)
    output_dir = _resolve_repo_path(str(config["analysis"]["output_root"])).resolve()
    if "/drafts/" in str(output_dir):
        raise TaskLanguageAnalysisError("Draft-owned analysis output is forbidden")
    if (output_dir / "COMPLETE.json").exists():
        raise TaskLanguageAnalysisError("Task-language output is already complete")
    rows, c2, provenance = _collect_rows(config)
    statistics = analyze_components(
        rows,
        analysis_seed=int(config["analysis"]["analysis_seed"]),
        bootstrap_draws=int(config["analysis"]["bootstrap_draws"]),
        family_alpha=float(config["analysis"]["family_alpha"]),
    )
    statistics["study_config"] = {
        "path": str(study_config_path),
        "sha256": _sha256(study_config_path),
    }
    statistics["source_rows"] = provenance
    statistics_path = output_dir / "statistics.json"
    _atomic_json(statistics_path, statistics)
    figures = render_figures(statistics, c2, output_dir)
    figure_source = {
        "schema_version": FIGURE_SOURCE_SCHEMA,
        "created_utc": _utc_now(),
        "statistics": {
            "path": str(statistics_path),
            "sha256": _sha256(statistics_path),
        },
        "figures": {
            label: {"path": path, "sha256": _sha256(Path(path))}
            for label, path in figures.items()
        },
    }
    figure_source_path = output_dir / "figure_source.json"
    _atomic_json(figure_source_path, figure_source)
    marker = {
        "schema_version": MARKER_SCHEMA,
        "state": "COMPLETE",
        "artifacts": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in (
                statistics_path,
                figure_source_path,
                *(Path(path) for path in figures.values()),
            )
        },
    }
    _atomic_json(output_dir / "COMPLETE.json", marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(study_config_path=args.study_config), indent=2, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "TaskLanguageAnalysisError",
    "analyze_components",
    "render_figures",
    "run_analysis",
]
