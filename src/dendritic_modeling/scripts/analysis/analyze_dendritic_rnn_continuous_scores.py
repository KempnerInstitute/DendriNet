"""Render non-inferential continuous-score diagnostics for the RNN cube.

The authoritative endpoint of the extreme-placement experiment is frozen-bank
accuracy.  This module does not replace or retest that endpoint.  It reuses the
same four primary crossover cells and the same ordered prediction artifacts to
show whether ceiling accuracy hides changes in target log score or binary
target margin.  Each run is reduced to one float64 mean before paired
trained-seed effects are formed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dendritic_modeling.analysis.statistics import bootstrap_mean_interval
from dendritic_modeling.scripts.analysis.analyze_dendritic_rnn_extreme_cube import (
    COLLECTION_SCHEMA,
    N_REPLICATES,
)

STATISTICS_SCHEMA = "dendritic-rnn-continuous-score-statistics-v1"
FIGURE_SOURCE_SCHEMA = "dendritic-rnn-continuous-score-figure-source-v1"
MARKER_SCHEMA = "dendritic-rnn-continuous-score-marker-v1"
ANALYSIS_SEED = 20_260_722
BOOTSTRAP_DRAWS = 10_000
PRIMARY_CROSSOVER_CELLS = (
    ("hierarchical", "proximal", "distal"),
    ("hierarchical", "distal", "proximal"),
    ("reversed", "proximal", "distal"),
    ("reversed", "distal", "proximal"),
)
CONTRAST_COEFFICIENTS = (1.0, -1.0, -1.0, 1.0)


class ContinuousScoreAnalysisError(RuntimeError):
    """Raised when a source artifact violates the diagnostic contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ContinuousScoreAnalysisError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ContinuousScoreAnalysisError(f"JSON root must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _subseed(label: str, *, analysis_seed: int) -> int:
    encoded = f"continuous-score-v1\0{analysis_seed}\0{label}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def summarize_predictions(payload: Mapping[str, Any]) -> dict[str, float | int]:
    """Reduce one prediction artifact to accuracy and two continuous scores."""

    predictions = payload.get("predictions")
    expected_n = payload.get("n_trials")
    if (
        not isinstance(predictions, list)
        or not isinstance(expected_n, int)
        or expected_n <= 0
        or len(predictions) != expected_n
    ):
        raise ContinuousScoreAnalysisError("Prediction artifact has invalid length")

    margins = np.empty(expected_n, dtype=np.float64)
    correct_count = 0
    for index, row in enumerate(predictions):
        if not isinstance(row, Mapping):
            raise ContinuousScoreAnalysisError(f"Invalid prediction row {index}")
        target = row.get("target")
        predicted = row.get("prediction")
        correct = row.get("correct")
        logits = row.get("logits")
        if target not in (0, 1) or predicted not in (0, 1):
            raise ContinuousScoreAnalysisError(f"Invalid class in row {index}")
        if type(correct) is not bool or correct != (predicted == target):
            raise ContinuousScoreAnalysisError(f"Invalid correctness in row {index}")
        if (
            not isinstance(logits, list)
            or len(logits) != 2
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in logits
            )
        ):
            raise ContinuousScoreAnalysisError(f"Invalid logits in row {index}")
        margins[index] = float(logits[target]) - float(logits[1 - target])
        correct_count += int(correct)

    log_scores = -np.logaddexp(0.0, -margins)
    return {
        "n_trials": expected_n,
        "correct_count": correct_count,
        "accuracy": correct_count / expected_n,
        "target_log_score": float(np.mean(log_scores, dtype=np.float64)),
        "target_margin": float(np.mean(margins, dtype=np.float64)),
    }


def _effect_vector(
    run_scores: Mapping[tuple[int, tuple[str, str, str]], Mapping[str, float]],
    metric: str,
) -> list[float]:
    values = []
    for replicate in range(N_REPLICATES):
        value = sum(
            coefficient * float(run_scores[replicate, cell][metric])
            for cell, coefficient in zip(
                PRIMARY_CROSSOVER_CELLS,
                CONTRAST_COEFFICIENTS,
                strict=True,
            )
        )
        values.append(float(value))
    return values


def analyze_scores(
    run_scores: Mapping[tuple[int, tuple[str, str, str]], Mapping[str, float]],
    *,
    analysis_seed: int = ANALYSIS_SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Form the same difference-of-crossovers for three descriptive metrics."""

    expected = {
        (replicate, cell)
        for replicate in range(N_REPLICATES)
        for cell in PRIMARY_CROSSOVER_CELLS
    }
    if set(run_scores) != expected:
        raise ContinuousScoreAnalysisError("The exact 10 paired quartets are required")

    metrics: dict[str, Any] = {}
    for metric in ("accuracy", "target_log_score", "target_margin"):
        values = _effect_vector(run_scores, metric)
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise ContinuousScoreAnalysisError(f"Non-finite {metric} effect")
        bootstrap = bootstrap_mean_interval(
            values,
            draws=bootstrap_draws,
            seed=_subseed(metric, analysis_seed=analysis_seed),
            require_n=N_REPLICATES,
        ).to_dict()
        metrics[metric] = {
            "values_by_replicate": values,
            "mean": bootstrap["mean"],
            "standard_deviation": bootstrap["standard_deviation"],
            "bootstrap_95_percent_ci": [bootstrap["lower"], bootstrap["upper"]],
            "bootstrap": bootstrap,
            "role": "descriptive_noninferential",
        }

    return {
        "schema_version": STATISTICS_SCHEMA,
        "created_utc": _utc_now(),
        "analysis_plan": {
            "independent_unit": "paired_trained_model_seed",
            "n_independent_units": N_REPLICATES,
            "run_reduction": "float64 mean over the exact ordered frozen trials",
            "contrast": (
                "hierarchical(proximal,distal) - "
                "hierarchical(distal,proximal) - "
                "reversed(proximal,distal) + reversed(distal,proximal)"
            ),
            "bootstrap": "paired trained-seed percentile 95% interval",
            "bootstrap_draws": bootstrap_draws,
            "analysis_seed": analysis_seed,
            "null_tests": [],
            "multiplicity_family": None,
            "claim_role": "descriptive_non_rescuing_diagnostic",
        },
        "metrics": metrics,
        "limitations": [
            "Accuracy remains the authoritative prespecified endpoint.",
            "Target margin depends on decoder logit scale.",
            "Target log score reflects both discrimination and calibration.",
            "The two continuous scores are deterministic transforms of the same binary logits and are not independent replications.",
            "No interval or sign from this diagnostic can rescue or redefine the failed accuracy crossover.",
        ],
    }


def collect_scores(collection: Mapping[str, Any]) -> tuple[dict, list[dict[str, Any]]]:
    """Load and hash-check the 40 prediction files bound by a cube collection."""

    if collection.get("schema_version") != COLLECTION_SCHEMA:
        raise ContinuousScoreAnalysisError("Unsupported cube collection schema")
    rows = collection.get("runs")
    if not isinstance(rows, list) or len(rows) != 80:
        raise ContinuousScoreAnalysisError("Cube collection must contain 80 runs")

    selected: dict[tuple[int, tuple[str, str, str]], Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ContinuousScoreAnalysisError("Invalid collection row")
        cell = (str(row.get("tau")), str(row.get("ff")), str(row.get("rec")))
        if cell not in PRIMARY_CROSSOVER_CELLS:
            continue
        replicate_text = row.get("replicate")
        if not isinstance(replicate_text, str) or not replicate_text.startswith("r"):
            raise ContinuousScoreAnalysisError("Invalid replicate label")
        replicate = int(replicate_text[1:])
        key = replicate, cell
        if key in selected:
            raise ContinuousScoreAnalysisError(f"Duplicate selected run: {key}")
        selected[key] = row

    expected = {
        (replicate, cell)
        for replicate in range(N_REPLICATES)
        for cell in PRIMARY_CROSSOVER_CELLS
    }
    if set(selected) != expected:
        raise ContinuousScoreAnalysisError("Cube lacks a complete primary quartet")

    scores: dict[tuple[int, tuple[str, str, str]], dict[str, float]] = {}
    provenance = []
    for key in sorted(selected):
        row = selected[key]
        path = Path(str(row.get("prediction_artifact"))).resolve(strict=True)
        if not path.is_file() or path.is_symlink() or "/drafts/" in str(path):
            raise ContinuousScoreAnalysisError(f"Invalid prediction path: {path}")
        observed_hash = _sha256(path)
        if observed_hash != row.get("prediction_sha256"):
            raise ContinuousScoreAnalysisError(f"Prediction hash mismatch: {path}")
        payload = _strict_json(path)
        if payload.get("run_id") != row.get("run_id"):
            raise ContinuousScoreAnalysisError(f"Prediction run mismatch: {path}")
        summary = summarize_predictions(payload)
        if (
            summary["n_trials"] != row.get("n_trials")
            or summary["correct_count"] != row.get("correct_count")
            or summary["accuracy"] != row.get("accuracy")
        ):
            raise ContinuousScoreAnalysisError(f"Accuracy audit mismatch: {path}")
        scores[key] = {
            metric: float(summary[metric])
            for metric in ("accuracy", "target_log_score", "target_margin")
        }
        provenance.append(
            {
                "replicate": f"r{key[0]:03d}",
                "tau": key[1][0],
                "ff": key[1][1],
                "rec": key[1][2],
                "run_id": row["run_id"],
                "prediction_artifact": str(path),
                "prediction_sha256": observed_hash,
                **summary,
            }
        )
    return scores, provenance


def _atomic_figure(fig: Any, path: Path, **kwargs: Any) -> None:
    temporary = path.with_name(f".{path.stem}.{os.getpid()}{path.suffix}")
    fig.savefig(temporary, **kwargs)
    temporary.replace(path)


def render_figure(statistics: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    """Render the three frozen diagnostic panels used by the manuscript."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    panels = (
        ("accuracy", "A", "Accuracy crossover", "C (percentage points)", 100.0),
        ("target_log_score", "B", "Target log score", r"$C_{\log}$", 1.0),
        ("target_margin", "C", "Target margin", r"$C_{\mathrm{margin}}$", 1.0),
    )
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), constrained_layout=True)
    color = "#8056a8"
    for ax, (metric, panel, title, xlabel, scale) in zip(axes, panels, strict=True):
        row = statistics["metrics"][metric]
        values = np.asarray(row["values_by_replicate"], dtype=np.float64) * scale
        interval = np.asarray(row["bootstrap_95_percent_ci"], dtype=np.float64) * scale
        mean = float(row["mean"]) * scale
        y = np.linspace(-0.18, 0.18, N_REPLICATES)
        ax.scatter(values, y, s=34, color=color, alpha=0.72, edgecolor="white")
        ax.plot(
            interval, [0.0, 0.0], color="black", linewidth=4, solid_capstyle="round"
        )
        ax.scatter(
            [mean],
            [0.0],
            marker="D",
            s=72,
            color="#d47a20",
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )
        ax.axvline(0.0, color="0.35", linestyle="--", linewidth=1)
        ax.set_ylim(-0.28, 0.28)
        ax.set_yticks([])
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.20)
        ax.text(
            -0.10, 1.08, panel, transform=ax.transAxes, fontsize=13, fontweight="bold"
        )
        ax.text(
            0.98,
            0.05,
            f"mean {mean:.4g}\n95% CI [{interval[0]:.4g}, {interval[1]:.4g}]",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.5,
            color="0.25",
        )
    fig.suptitle(
        "Continuous-score views of the frozen placement-timescale crossover",
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.035,
        "n=10 paired trained-model seeds; dots are seed effects, orange diamonds are means, "
        "and black bars are descriptive paired-seed bootstrap 95% intervals. No null tests.",
        ha="center",
        fontsize=11,
        color="0.30",
    )
    pdf = output_dir / "continuous_score_crossover_diagnostic_v2.pdf"
    png = output_dir / "continuous_score_crossover_diagnostic_v2.png"
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
    _atomic_figure(
        fig,
        png,
        bbox_inches="tight",
        dpi=240,
        metadata={"Software": "dendritic_modeling"},
    )
    plt.close(fig)
    return {"pdf": str(pdf), "png": str(png)}


def run_analysis(
    *,
    collection_path: Path,
    output_dir: Path,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    source_sha256 = _sha256(source_path)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ContinuousScoreAnalysisError("Analysis source changed after invocation")
    collection_path = collection_path.resolve(strict=True)
    output_dir = output_dir.resolve()
    if "/drafts/" in str(collection_path) or "/drafts/" in str(output_dir):
        raise ContinuousScoreAnalysisError("Draft-owned analysis paths are forbidden")
    marker_path = output_dir / "COMPLETE.json"
    if marker_path.exists():
        raise ContinuousScoreAnalysisError(
            f"Analysis is already complete: {output_dir}"
        )

    collection = _strict_json(collection_path)
    scores, run_summaries = collect_scores(collection)
    statistics = analyze_scores(scores)
    statistics["collection_artifact"] = {
        "path": str(collection_path),
        "sha256": _sha256(collection_path),
    }
    statistics["analysis_source"] = {
        "path": str(source_path),
        "sha256": source_sha256,
    }
    statistics["run_summaries"] = run_summaries

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
        "analysis_source": statistics["analysis_source"],
        "figures": {
            label: {"path": path, "sha256": _sha256(Path(path))}
            for label, path in figures.items()
        },
        "panel_contract": {
            "A": "authoritative accuracy C reused as a nonindependent reference",
            "B": "target-log-score C from the identical trials and seed quartets",
            "C": "binary-target-margin C from the identical trials and seed quartets",
        },
    }
    figure_source_path = output_dir / "figure_source.json"
    _atomic_json(figure_source_path, figure_source)
    marker = {
        "schema_version": MARKER_SCHEMA,
        "state": "COMPLETE",
        "created_utc": _utc_now(),
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
    _atomic_json(marker_path, marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    marker = run_analysis(
        collection_path=args.collection,
        output_dir=args.output_dir,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(json.dumps(marker, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "ContinuousScoreAnalysisError",
    "analyze_scores",
    "collect_scores",
    "render_figure",
    "run_analysis",
    "summarize_predictions",
]
