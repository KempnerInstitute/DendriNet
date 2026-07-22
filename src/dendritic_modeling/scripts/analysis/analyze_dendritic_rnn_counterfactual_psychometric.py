"""Analyze the four-cell frozen counterfactual psychometric evaluation.

The model seed is the independent unit.  Receiver substitutions are averaged
within each trained model and never counted as replications.  The principal
endpoint here is the prospectively defined jointly supported same-noise
central difference (C2); this artifact does not manufacture the separate C1,
C3, or stimulus-blank C4 evidence required by the manuscript's task-language
gate.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dendritic_modeling.analysis.statistics import bootstrap_mean_interval
from dendritic_modeling.scripts.analysis.analyze_dendritic_rnn_extreme_cube import (
    N_REPLICATES,
    exact_two_sided_sign_flip,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_counterfactual_bank import (
    ROLE_CELLS,
    _sha256,
    _strict_json,
    load_counterfactual_arrays,
    load_study_config,
)

STATISTICS_SCHEMA = "dendritic-rnn-counterfactual-psychometric-statistics-v1"
MARKER_SCHEMA = "dendritic-rnn-counterfactual-psychometric-analysis-marker-v1"
FIGURE_SOURCE_SCHEMA = "dendritic-rnn-counterfactual-psychometric-figure-source-v1"
ANALYSIS_SEED = 20_260_722
BOOTSTRAP_DRAWS = 10_000
SUPPORTED_TARGETS = {2: (1, 3), 3: (2, 4)}


class CounterfactualAnalysisError(RuntimeError):
    """Raised when a result cannot support the frozen seed-first analysis."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _subseed(label: str, *, analysis_seed: int) -> int:
    import hashlib

    encoded = f"counterfactual-c2-v1\0{analysis_seed}\0{label}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def _class1_probability(margins: np.ndarray) -> np.ndarray:
    probabilities = np.empty_like(margins, dtype=np.float64)
    positive = margins >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-margins[positive]))
    exponent = np.exp(margins[~positive])
    probabilities[~positive] = exponent / (1.0 + exponent)
    return probabilities


def summarize_run(
    logits: np.ndarray,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Reduce 4,096 substitutions to one trained-model C2 endpoint."""

    logits = np.asarray(logits)
    if logits.shape != (4096, 2) or logits.dtype != np.float32:
        raise CounterfactualAnalysisError("Counterfactual logits metadata changed")
    if not np.isfinite(logits).all():
        raise CounterfactualAnalysisError("Counterfactual logits are non-finite")
    margins = logits[:, 1].astype(np.float64) - logits[:, 0].astype(np.float64)
    probabilities = _class1_probability(margins)
    first = np.asarray(arrays["receiver_first_ranks"], dtype=np.int64)
    second = np.asarray(arrays["target_second_ranks"], dtype=np.int64)
    targets = np.asarray(arrays["targets"], dtype=np.int64)
    defined = np.asarray(arrays["comparison_defined"], dtype=bool)
    receiver = np.asarray(arrays["receiver_indices"], dtype=np.int64)
    if any(
        array.shape != (4096,) for array in (first, second, targets, defined, receiver)
    ):
        raise CounterfactualAnalysisError("Counterfactual metadata shape changed")

    margin_matrix = margins.reshape(1024, 4)
    first_by_receiver = first.reshape(1024, 4)[:, 0]
    c2 = np.empty(1024, dtype=np.float64)
    for first_rank, (below, above) in SUPPORTED_TARGETS.items():
        selected = first_by_receiver == first_rank
        c2[selected] = (
            margin_matrix[selected, above - 1] - margin_matrix[selected, below - 1]
        ) / 2.0
    if not np.isfinite(c2).all():
        raise CounterfactualAnalysisError("C2 receiver reductions are invalid")

    predictions = logits.argmax(axis=1)
    supported = np.zeros(4096, dtype=bool)
    for first_rank, target_ranks in SUPPORTED_TARGETS.items():
        supported |= (first == first_rank) & np.isin(second, target_ranks)
    if int(supported.sum()) != 2048 or int(defined.sum()) != 3072:
        raise CounterfactualAnalysisError("Frozen support counts changed")

    surface = []
    for first_rank in (2, 3):
        for second_rank in (1, 2, 3, 4):
            mask = (first == first_rank) & (second == second_rank)
            if int(mask.sum()) != 512:
                raise CounterfactualAnalysisError("Psychometric surface count changed")
            if second_rank == first_rank:
                support_class = "equality"
            elif second_rank in SUPPORTED_TARGETS[first_rank]:
                support_class = "joint_supported_unequal"
            else:
                support_class = "marginal_only_joint_unsupported"
            surface.append(
                {
                    "receiver_first_rank": first_rank,
                    "target_second_rank": second_rank,
                    "support_class": support_class,
                    "n_rows": int(mask.sum()),
                    "mean_class1_probability": float(
                        np.mean(probabilities[mask], dtype=np.float64)
                    ),
                }
            )
    return {
        "receiver_count": 1024,
        "counterfactual_row_count": 4096,
        "mean_joint_supported_central_difference": float(np.mean(c2, dtype=np.float64)),
        "receiver_c2_standard_deviation": float(np.std(c2, ddof=1)),
        "fraction_receivers_positive_c2": float(np.mean(c2 > 0.0)),
        "joint_supported_unequal_accuracy": float(
            np.mean(predictions[supported] == targets[supported])
        ),
        "full_unequal_accuracy": float(
            np.mean(predictions[defined] == targets[defined])
        ),
        "psychometric_surface": surface,
    }


def analyze_seed_summaries(
    summaries: Mapping[str, list[Mapping[str, Any]]],
    *,
    analysis_seed: int = ANALYSIS_SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Summarize C2 over ten independent trained-model seeds per cell."""

    if set(summaries) != set(ROLE_CELLS):
        raise CounterfactualAnalysisError("The exact four cell roles are required")
    role_statistics = {}
    by_role_values = {}
    for role in ROLE_CELLS:
        rows = summaries[role]
        if len(rows) != N_REPLICATES:
            raise CounterfactualAnalysisError(f"{role} does not contain ten seeds")
        values = [float(row["mean_joint_supported_central_difference"]) for row in rows]
        bootstrap = bootstrap_mean_interval(
            values,
            draws=bootstrap_draws,
            seed=_subseed(role, analysis_seed=analysis_seed),
            require_n=N_REPLICATES,
        ).to_dict()
        sign_flip = exact_two_sided_sign_flip(values)
        lower = float(bootstrap["lower"])
        supported = bool(
            bootstrap["mean"] > 0.0 and lower > 0.0 and sign_flip["p_value"] <= 0.05
        )
        role_statistics[role] = {
            "cell": list(ROLE_CELLS[role]),
            "values_by_replicate": values,
            "mean": bootstrap["mean"],
            "standard_deviation": bootstrap["standard_deviation"],
            "bootstrap_95_percent_ci": [bootstrap["lower"], bootstrap["upper"]],
            "bootstrap": bootstrap,
            "exact_two_sided_sign_flip": sign_flip,
            "support_gate": {
                "positive_mean": bool(bootstrap["mean"] > 0.0),
                "strictly_positive_bootstrap_lower": bool(lower > 0.0),
                "raw_p_at_most_0.05": bool(sign_flip["p_value"] <= 0.05),
                "unadjusted_positive_support": supported,
                "task_language_authorized_by_c2_alone": False,
            },
            "psychometric_surface_by_seed": [
                row["psychometric_surface"] for row in rows
            ],
            "descriptive_accuracy": {
                "joint_supported_unequal_by_seed": [
                    row["joint_supported_unequal_accuracy"] for row in rows
                ],
                "full_unequal_by_seed": [row["full_unequal_accuracy"] for row in rows],
            },
        }
        by_role_values[role] = values

    crossover = [
        by_role_values["hierarchical_aligned"][seed]
        - by_role_values["hierarchical_reverse"][seed]
        - by_role_values["reversed_aligned"][seed]
        + by_role_values["reversed_reverse"][seed]
        for seed in range(N_REPLICATES)
    ]
    crossover_bootstrap = bootstrap_mean_interval(
        crossover,
        draws=bootstrap_draws,
        seed=_subseed("descriptive-c2-crossover", analysis_seed=analysis_seed),
        require_n=N_REPLICATES,
    ).to_dict()
    return {
        "schema_version": STATISTICS_SCHEMA,
        "created_utc": _utc_now(),
        "analysis_plan": {
            "independent_unit": "paired_trained_model_seed",
            "n_independent_units_per_cell": N_REPLICATES,
            "receiver_role": "within-model_measurement_only",
            "c2_definition": (
                "mean receiver-wise (class1-margin above - class1-margin below) / 2 "
                "for frozen jointly supported rank substitutions"
            ),
            "bootstrap_draws": bootstrap_draws,
            "analysis_seed": analysis_seed,
            "task_language_boundary": (
                "C2 is necessary but cannot authorize task-consistent use of both "
                "stimulus identities without the separately bound C1 and C3 evidence"
            ),
        },
        "roles": role_statistics,
        "descriptive_c2_crossover": {
            "role": "descriptive_not_a_registered_behavioral_endpoint",
            "values_by_replicate": crossover,
            "mean": crossover_bootstrap["mean"],
            "standard_deviation": crossover_bootstrap["standard_deviation"],
            "bootstrap_95_percent_ci": [
                crossover_bootstrap["lower"],
                crossover_bootstrap["upper"],
            ],
            "null_test": None,
            "can_rescue_accuracy_crossover": False,
        },
    }


def _load_all_results(config: Mapping[str, Any]) -> tuple[dict, list[dict[str, Any]]]:
    output_root = Path(str(config["study"]["output_root"]))
    if not output_root.is_absolute():
        from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_counterfactual_bank import (
            REPO_ROOT,
        )

        output_root = REPO_ROOT / output_root
    summaries = {role: [] for role in ROLE_CELLS}
    provenance = []
    for replicate in range(N_REPLICATES):
        replicate_dir = output_root / f"r{replicate:03d}"
        marker_path = replicate_dir / "COMPLETE.json"
        marker = _strict_json(marker_path)
        if (
            marker.get("state") != "COMPLETE"
            or marker.get("replicate") != f"r{replicate:03d}"
        ):
            raise CounterfactualAnalysisError("Replicate completion marker mismatch")
        for role in ROLE_CELLS:
            role_marker = _strict_json(replicate_dir / role / "COMPLETE.json")
            result_path = Path(str(role_marker["result"]["path"]))
            logits_path = Path(str(role_marker["logits"]["path"]))
            if result_path != replicate_dir / role / "result.json":
                raise CounterfactualAnalysisError("Result public path mismatch")
            if logits_path != replicate_dir / role / "logits.npz":
                raise CounterfactualAnalysisError("Logits public path mismatch")
            if _sha256(result_path) != role_marker["result"]["sha256"]:
                raise CounterfactualAnalysisError("Result hash mismatch")
            if _sha256(logits_path) != role_marker["logits"]["sha256"]:
                raise CounterfactualAnalysisError("Logits hash mismatch")
            result = _strict_json(result_path)
            audits = result.get("audits", {})
            if (
                result.get("role") != role
                or result.get("replicate") != f"r{replicate:03d}"
                or audits.get("identity_bridge") != "PASSED_BITWISE"
                or audits.get("rotated_partition") != "PASSED_BITWISE"
                or audits.get("state_immutability") != "PASSED"
            ):
                raise CounterfactualAnalysisError("Required evaluation audit failed")
            bank_path = Path(result["counterfactual_bank"]["bank_path"])
            if _sha256(bank_path) != result["counterfactual_bank"]["bank_sha256"]:
                raise CounterfactualAnalysisError("Counterfactual bank hash changed")
            arrays = load_counterfactual_arrays(bank_path)
            with np.load(logits_path, allow_pickle=False) as archive:
                if set(archive.files) != {"counterfactual_logits", "identity_logits"}:
                    raise CounterfactualAnalysisError("Logits artifact members changed")
                logits = np.array(archive["counterfactual_logits"], copy=True)
                identity = np.array(archive["identity_logits"], copy=True)
            if identity.shape != (1024, 2) or identity.dtype != np.float32:
                raise CounterfactualAnalysisError("Identity logits metadata changed")
            summary = summarize_run(logits, arrays)
            summaries[role].append(summary)
            provenance.append(
                {
                    "replicate": f"r{replicate:03d}",
                    "role": role,
                    "run_id": result["run_id"],
                    "result": {
                        "path": str(result_path),
                        "sha256": _sha256(result_path),
                    },
                    "logits": {
                        "path": str(logits_path),
                        "sha256": _sha256(logits_path),
                    },
                    **summary,
                }
            )
    return summaries, provenance


def _atomic_figure(fig: Any, path: Path, **kwargs: Any) -> None:
    temporary = path.with_name(f".{path.stem}.{os.getpid()}{path.suffix}")
    fig.savefig(temporary, **kwargs)
    temporary.replace(path)


def render_figure(statistics: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.5, 6.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1.0, 1.0, 1.15))
    titles = {
        "hierarchical_aligned": "Hierarchical, FF prox / REC dist",
        "hierarchical_reverse": "Hierarchical, FF dist / REC prox",
        "reversed_aligned": "Reversed, FF prox / REC dist",
        "reversed_reverse": "Reversed, FF dist / REC prox",
    }
    colors = {2: "#3366a8", 3: "#c84b31"}
    for index, role in enumerate(ROLE_CELLS):
        ax = fig.add_subplot(grid[index // 2, index % 2])
        surfaces = statistics["roles"][role]["psychometric_surface_by_seed"]
        for first_rank in (2, 3):
            seed_curves = []
            for surface in surfaces:
                values = [
                    row["mean_class1_probability"]
                    for row in surface
                    if row["receiver_first_rank"] == first_rank
                ]
                seed_curves.append(values)
                ax.plot(range(1, 5), values, color=colors[first_rank], alpha=0.13)
            mean_curve = np.mean(np.asarray(seed_curves), axis=0)
            ax.plot(
                range(1, 5),
                mean_curve,
                color=colors[first_rank],
                linewidth=2.4,
                marker="o",
                label=f"first rank {first_rank}",
            )
        ax.set_title(titles[role], fontsize=9)
        ax.set_xticks(range(1, 5))
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Counterfactual second-stimulus rank")
        ax.set_ylabel("Class-1 probability")
        ax.grid(alpha=0.20)
        if index == 0:
            ax.legend(frameon=False, fontsize=8)
            ax.text(
                -0.18, 1.08, "A", transform=ax.transAxes, fontweight="bold", fontsize=13
            )

    ax = fig.add_subplot(grid[:, 2])
    positions = np.arange(4)[::-1]
    role_colors = ["#4869a5", "#7798c5", "#c46d4d", "#dc9a7e"]
    for y, role, color in zip(positions, ROLE_CELLS, role_colors, strict=True):
        row = statistics["roles"][role]
        values = np.asarray(row["values_by_replicate"])
        interval = np.asarray(row["bootstrap_95_percent_ci"])
        jitter = np.linspace(-0.12, 0.12, N_REPLICATES)
        ax.scatter(values, y + jitter, color=color, s=25, alpha=0.70)
        ax.plot(interval, [y, y], color="black", linewidth=3)
        ax.scatter([row["mean"]], [y], marker="D", s=60, color=color, edgecolor="black")
    ax.axvline(0.0, color="0.35", linestyle="--")
    ax.set_yticks(positions, [titles[role] for role in ROLE_CELLS], fontsize=8)
    ax.set_xlabel("C2: supported central difference in class-1 margin")
    ax.set_title("Seed-first C2 evidence", fontsize=10)
    ax.grid(axis="x", alpha=0.20)
    ax.text(-0.12, 1.04, "B", transform=ax.transAxes, fontweight="bold", fontsize=13)
    fig.suptitle(
        "Frozen counterfactual second-stimulus response across the primary quartet",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.02,
        "Thin curves/dots: ten trained-model seeds. Thick curves/diamonds: means; black bars: "
        "paired-seed bootstrap 95% intervals. Receiver substitutions are not replications. "
        "C2 alone does not authorize the manuscript task-language gate.",
        ha="center",
        fontsize=8,
        color="0.30",
    )
    pdf = output_dir / "counterfactual_psychometric_c2_src_v1.pdf"
    png = output_dir / "counterfactual_psychometric_c2_src_v1.png"
    _atomic_figure(
        fig,
        pdf,
        bbox_inches="tight",
        metadata={
            "Creator": "dendritic_modeling",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    _atomic_figure(fig, png, bbox_inches="tight", dpi=240)
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def run_analysis(*, study_config_path: Path, output_dir: Path) -> dict[str, Any]:
    config, study_config_path = load_study_config(study_config_path)
    output_dir = output_dir.resolve()
    if "/drafts/" in str(output_dir):
        raise CounterfactualAnalysisError("Draft-owned analysis output is forbidden")
    if (output_dir / "COMPLETE.json").exists():
        raise CounterfactualAnalysisError("Analysis output is already complete")
    summaries, provenance = _load_all_results(config)
    statistics = analyze_seed_summaries(summaries)
    statistics["study_config"] = {
        "path": str(study_config_path),
        "sha256": _sha256(study_config_path),
    }
    statistics["source_rows"] = provenance
    statistics_path = output_dir / "statistics.json"
    _atomic_json(statistics_path, statistics)
    figures = render_figure(statistics, output_dir)
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
                Path(figures["pdf"]),
                Path(figures["png"]),
            )
        },
    }
    _atomic_json(output_dir / "COMPLETE.json", marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    marker = run_analysis(
        study_config_path=args.study_config,
        output_dir=args.output_dir,
    )
    print(json.dumps(marker, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "CounterfactualAnalysisError",
    "analyze_seed_summaries",
    "render_figure",
    "run_analysis",
    "summarize_run",
]
