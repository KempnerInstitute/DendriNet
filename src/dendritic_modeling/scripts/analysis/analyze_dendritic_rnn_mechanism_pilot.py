"""Analyze the three-seed fixed-soma dendritic-RNN mechanism pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_counterfactual_bank import (
    _atomic_json,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_frozen_bank import (
    _sha256,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_mechanism_pilot import (
    DEFAULT_CONFIG,
    EXPECTED_CONDITIONS,
    REPO_ROOT,
    RESULT_SCHEMA,
    _resolve,
    _strict_json,
    load_config,
)

STATISTICS_SCHEMA = "dendritic-rnn-mechanism-pilot-statistics-v1"
ANALYSIS_CONTRACT_SCHEMA = "dendritic-rnn-mechanism-pilot-analysis-contract-v1"
DEFAULT_ANALYSIS_CONTRACT = (
    REPO_ROOT
    / "configs/dendritic_rnn/mechanism_identification_pilot_v1/analysis_contract.json"
)
METRICS = ("accuracy", "target_log_score", "target_margin")
EXPECTED_DELAY_STRATA = ((240, 260), (261, 280), (281, 300), (301, 320))
SLOW_PROX, SLOW_DIST, FAST_PROX, FAST_DIST = EXPECTED_CONDITIONS
SHORT_LABELS = {
    SLOW_PROX: "slow distal $\\tau$\nREC proximal",
    SLOW_DIST: "slow distal $\\tau$\nREC distal",
    FAST_PROX: "fast distal $\\tau$\nREC proximal",
    FAST_DIST: "fast distal $\\tau$\nREC distal",
}


class MechanismPilotAnalysisError(RuntimeError):
    """Raised when pilot evaluation results are incomplete or inconsistent."""


def _load_analysis_contract(
    path: Path = DEFAULT_ANALYSIS_CONTRACT,
) -> dict[str, Any]:
    """Validate the analysis source and delay plan frozen before pilot outcomes."""

    contract = _strict_json(path.resolve(strict=True))
    if contract.get("schema_version") != ANALYSIS_CONTRACT_SCHEMA:
        raise MechanismPilotAnalysisError("Unsupported pilot analysis contract")
    if _sha256(Path(__file__).resolve()) != contract.get("analysis_source_sha256"):
        raise MechanismPilotAnalysisError("Pilot analysis source changed after freeze")
    if tuple(tuple(values) for values in contract.get("delay_strata", ())) != (
        EXPECTED_DELAY_STRATA
    ):
        raise MechanismPilotAnalysisError("Pilot delay-stratum plan changed")
    if tuple(contract.get("aggregate_gate_metrics", ())) != METRICS:
        raise MechanismPilotAnalysisError("Pilot aggregate gate metrics changed")
    if contract.get("delay_strata_in_aggregate_gate") is not False:
        raise MechanismPilotAnalysisError("Delay strata entered the aggregate gate")
    return contract


def _result_paths(config: dict[str, Any]) -> list[Path]:
    root = _resolve(config["study"]["output_root"])
    return [root / group / "result.json" for group in ("r000", "r001", "r002")]


def _load_results(
    config: dict[str, Any], config_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows = []
    provenance = []
    config_sha256 = _sha256(config_path)
    for path in _result_paths(config):
        payload = _strict_json(path.resolve(strict=True))
        if (
            payload.get("schema_version") != RESULT_SCHEMA
            or payload.get("state") != "COMPLETE"
        ):
            raise MechanismPilotAnalysisError(f"Incomplete evaluation result: {path}")
        if payload.get("config_sha256") != config_sha256:
            raise MechanismPilotAnalysisError(f"Evaluation/config mismatch: {path}")
        conditions = payload.get("conditions")
        if (
            not isinstance(conditions, list)
            or tuple(row.get("condition") for row in conditions) != EXPECTED_CONDITIONS
        ):
            raise MechanismPilotAnalysisError(f"Condition grid changed: {path}")
        for condition in conditions:
            metrics = condition.get("metrics", {})
            if any(
                metric not in metrics or not np.isfinite(metrics[metric])
                for metric in METRICS
            ):
                raise MechanismPilotAnalysisError(
                    f"Missing or non-finite metric: {path}"
                )
            delay_strata = metrics.get("delay_strata")
            if not isinstance(delay_strata, list) or len(delay_strata) != len(
                EXPECTED_DELAY_STRATA
            ):
                raise MechanismPilotAnalysisError(f"Missing delay strata: {path}")
            for stratum, (expected_low, expected_high) in zip(
                delay_strata, EXPECTED_DELAY_STRATA, strict=True
            ):
                if (
                    stratum.get("delay_low") != expected_low
                    or stratum.get("delay_high") != expected_high
                    or int(stratum.get("n_trials", 0)) <= 0
                    or any(
                        metric not in stratum or not np.isfinite(stratum[metric])
                        for metric in METRICS
                    )
                ):
                    raise MechanismPilotAnalysisError(
                        f"Invalid delay stratum in evaluation result: {path}"
                    )
            rows.append(
                {
                    "pairing_group": payload["pairing_group"],
                    "condition": condition["condition"],
                    **{metric: float(metrics[metric]) for metric in METRICS},
                    "delay_strata": [
                        {
                            "delay_low": int(stratum["delay_low"]),
                            "delay_high": int(stratum["delay_high"]),
                            "n_trials": int(stratum["n_trials"]),
                            **{metric: float(stratum[metric]) for metric in METRICS},
                        }
                        for stratum in delay_strata
                    ],
                    "best_epoch": None,
                    "best_validation_loss": None,
                }
            )
        provenance.append({"path": str(path), "sha256": _sha256(path)})
    if len(rows) != 12:
        raise MechanismPilotAnalysisError("Expected exactly twelve evaluated cell rows")
    return rows, provenance


def _nested(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        result.setdefault(row["pairing_group"], {})[row["condition"]] = {
            metric: float(row[metric]) for metric in METRICS
        }
    if any(set(cells) != set(EXPECTED_CONDITIONS) for cells in result.values()):
        raise MechanismPilotAnalysisError("Paired quartet is incomplete")
    return result


def _summarize_metric(
    nested: dict[str, dict[str, dict[str, float]]], metric: str
) -> dict[str, Any]:
    seeds = sorted(nested)
    condition_values = {
        condition: [nested[seed][condition][metric] for seed in seeds]
        for condition in EXPECTED_CONDITIONS
    }
    slow_effect = [
        nested[seed][SLOW_DIST][metric] - nested[seed][SLOW_PROX][metric]
        for seed in seeds
    ]
    fast_effect = [
        nested[seed][FAST_DIST][metric] - nested[seed][FAST_PROX][metric]
        for seed in seeds
    ]
    contrast = [slow - fast for slow, fast in zip(slow_effect, fast_effect)]
    return {
        "condition_values": condition_values,
        "condition_means": {
            condition: float(np.mean(values, dtype=np.float64))
            for condition, values in condition_values.items()
        },
        "recurrent_distal_minus_proximal": {
            "slow_distal_tau": slow_effect,
            "fast_distal_tau": fast_effect,
            "slow_mean": float(np.mean(slow_effect, dtype=np.float64)),
            "fast_mean": float(np.mean(fast_effect, dtype=np.float64)),
        },
        "primary_contrast_seed_values": contrast,
        "primary_contrast_mean": float(np.mean(contrast, dtype=np.float64)),
        "primary_contrast_positive_seed_count": int(np.sum(np.asarray(contrast) > 0)),
    }


def _summarize_delay_strata(
    rows: list[dict[str, Any]], metric: str = "accuracy"
) -> dict[str, Any]:
    """Compute the paired interaction separately in each frozen delay bin."""

    nested = {
        (row["pairing_group"], row["condition"]): row["delay_strata"] for row in rows
    }
    seeds = sorted({row["pairing_group"] for row in rows})
    summaries = []
    for stratum_index, (low, high) in enumerate(EXPECTED_DELAY_STRATA):
        slow_effect = []
        fast_effect = []
        contrast = []
        n_trials_by_seed = []
        for seed in seeds:
            values = {
                condition: float(nested[(seed, condition)][stratum_index][metric])
                for condition in EXPECTED_CONDITIONS
            }
            slow = values[SLOW_DIST] - values[SLOW_PROX]
            fast = values[FAST_DIST] - values[FAST_PROX]
            slow_effect.append(slow)
            fast_effect.append(fast)
            contrast.append(slow - fast)
            n_trials_by_seed.append(
                int(nested[(seed, SLOW_PROX)][stratum_index]["n_trials"])
            )
        summaries.append(
            {
                "delay_low": low,
                "delay_high": high,
                "delay_midpoint": 0.5 * (low + high),
                "n_trials_by_seed": n_trials_by_seed,
                "slow_distal_tau_effect": slow_effect,
                "fast_distal_tau_effect": fast_effect,
                "primary_contrast_seed_values": contrast,
                "primary_contrast_mean": float(np.mean(contrast, dtype=np.float64)),
            }
        )
    first = np.asarray(summaries[0]["primary_contrast_seed_values"])
    last = np.asarray(summaries[-1]["primary_contrast_seed_values"])
    return {
        "metric": metric,
        "strata": summaries,
        "longest_minus_shortest_seed_values": (last - first).tolist(),
        "longest_minus_shortest_mean": float(np.mean(last - first)),
        "inferential_status": "descriptive_only_not_part_of_go_no_go",
    }


def compute_statistics(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Validate the complete grid and compute descriptive paired-seed effects."""

    analysis_contract = _load_analysis_contract()
    config, config_path = load_config(config_path)
    rows, provenance = _load_results(config, config_path)
    nested = _nested(rows)
    metrics = {metric: _summarize_metric(nested, metric) for metric in METRICS}
    delay_strata_accuracy = _summarize_delay_strata(rows, "accuracy")
    gate = config["go_no_go"]
    cell_accuracy_means = list(metrics["accuracy"]["condition_means"].values())
    checks = {
        "positive_mean_all_metrics": all(
            metrics[metric]["primary_contrast_mean"] > 0 for metric in METRICS
        ),
        "positive_accuracy_seed_count": metrics["accuracy"][
            "primary_contrast_positive_seed_count"
        ]
        >= int(gate["require_positive_accuracy_contrast_seeds"]),
        "minimum_accuracy_effect": metrics["accuracy"]["primary_contrast_mean"]
        >= float(gate["minimum_mean_accuracy_contrast"]),
        "not_ceiling_limited": max(cell_accuracy_means)
        <= float(gate["maximum_cell_mean_accuracy"]),
        "above_chance_floor": min(cell_accuracy_means)
        >= float(gate["minimum_cell_mean_accuracy"]),
    }
    expand = all(checks.values())
    return {
        "schema_version": STATISTICS_SCHEMA,
        "inferential_status": "descriptive_three_seed_pilot_no_null_tests",
        "replication_unit": "paired_trained_model_seed",
        "n_replicates": 3,
        "n_cell_rows": 12,
        "primary_contrast": config["study"]["primary_contrast"],
        "metrics": metrics,
        "delay_strata_accuracy": delay_strata_accuracy,
        "go_no_go": {
            "checks": checks,
            "expand_to_confirmatory_seed_count": expand,
            "decision": "EXPAND" if expand else "DO_NOT_EXPAND_FOR_DIRECTIONAL_SUPPORT",
            "interpretation": "A negative decision does not make the pilot uninformative; it only declines a larger directional-support run.",
        },
        "source_results": provenance,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "analysis_contract": {
            "path": str(DEFAULT_ANALYSIS_CONTRACT),
            "sha256": _sha256(DEFAULT_ANALYSIS_CONTRACT),
            "frozen_utc": analysis_contract["frozen_utc"],
        },
        "rows": rows,
    }


def _mean_marker(ax: plt.Axes, x: float, values: list[float]) -> None:
    ax.scatter(
        [x],
        [float(np.mean(values, dtype=np.float64))],
        marker="D",
        s=52,
        color="#e07a1f",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )


def render_figure(statistics: dict[str, Any], output: Path) -> None:
    """Render absolute accuracy plus all three paired interaction endpoints."""

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8})
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0), constrained_layout=True)
    seeds = ("r000", "r001", "r002")
    colors = ("#2b6cb0", "#5b8e3e", "#8a5aa6")
    accuracy = statistics["metrics"]["accuracy"]
    x = np.arange(4)
    for seed_index, (seed, color) in enumerate(zip(seeds, colors)):
        values = [
            accuracy["condition_values"][condition][seed_index]
            for condition in EXPECTED_CONDITIONS
        ]
        axes[0, 0].plot(x, values, color=color, alpha=0.7, linewidth=1.0)
        axes[0, 0].scatter(x, values, color=color, s=24, label=seed, zorder=3)
    means = [
        accuracy["condition_means"][condition] for condition in EXPECTED_CONDITIONS
    ]
    axes[0, 0].scatter(
        x,
        means,
        marker="D",
        s=52,
        color="#e07a1f",
        edgecolor="black",
        linewidth=0.6,
        zorder=5,
    )
    axes[0, 0].set_xticks(
        x, [SHORT_LABELS[condition] for condition in EXPECTED_CONDITIONS]
    )
    axes[0, 0].set_ylabel("held-out accuracy")
    axes[0, 0].set_ylim(0.45, 1.01)
    axes[0, 0].set_title(
        "a  Absolute performance at delays 240--320", loc="left", fontweight="bold"
    )
    axes[0, 0].legend(frameon=False, ncol=3, loc="lower left")

    panel_specs = (
        (axes[0, 1], "accuracy", "accuracy difference", "b"),
        (axes[1, 0], "target_log_score", "target-log-score difference", "d"),
        (axes[1, 1], "target_margin", "target-margin difference", "e"),
    )
    for ax, metric, ylabel, letter in panel_specs:
        summary = statistics["metrics"][metric]
        slow = summary["recurrent_distal_minus_proximal"]["slow_distal_tau"]
        fast = summary["recurrent_distal_minus_proximal"]["fast_distal_tau"]
        for seed_index, color in enumerate(colors):
            ax.plot(
                [0, 1],
                [fast[seed_index], slow[seed_index]],
                color=color,
                alpha=0.75,
                linewidth=1.0,
            )
            ax.scatter(
                [0, 1],
                [fast[seed_index], slow[seed_index]],
                color=color,
                s=24,
                zorder=3,
            )
        _mean_marker(ax, 0, fast)
        _mean_marker(ax, 1, slow)
        ax.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
        ax.set_xticks([0, 1], ["fast distal $\\tau$", "slow distal $\\tau$"])
        ax.set_ylabel("REC distal - proximal\n" + ylabel)
        contrast = summary["primary_contrast_mean"]
        ax.set_title(
            f"{letter}  Placement effect; interaction = {contrast:+.4g}",
            loc="left",
            fontweight="bold",
        )

    delay_summary = statistics["delay_strata_accuracy"]
    delay_ax = axes[0, 2]
    midpoints = [row["delay_midpoint"] for row in delay_summary["strata"]]
    for seed_index, color in enumerate(colors):
        values = [
            row["primary_contrast_seed_values"][seed_index]
            for row in delay_summary["strata"]
        ]
        delay_ax.plot(midpoints, values, color=color, alpha=0.75, linewidth=1.0)
        delay_ax.scatter(midpoints, values, color=color, s=22, zorder=3)
    delay_means = [row["primary_contrast_mean"] for row in delay_summary["strata"]]
    delay_ax.plot(midpoints, delay_means, color="#e07a1f", linewidth=1.3, zorder=4)
    delay_ax.scatter(
        midpoints,
        delay_means,
        marker="D",
        s=42,
        color="#e07a1f",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
    )
    delay_ax.axhline(0, color="0.35", linewidth=0.8, linestyle="--")
    delay_ax.set_xticks(midpoints, ["240-260", "261-280", "281-300", "301-320"])
    delay_ax.tick_params(axis="x", rotation=25)
    delay_ax.set_xlabel("held-out delay steps")
    delay_ax.set_ylabel("accuracy interaction")
    delay_ax.set_title(
        "c  Retention prediction across delay",
        loc="left",
        fontweight="bold",
    )

    design_ax = axes[1, 2]
    design_ax.axis("off")
    decision = statistics["go_no_go"]["decision"].replace("_", " ")
    checks = statistics["go_no_go"]["checks"]
    check_lines = "\n".join(
        f"{'PASS' if passed else 'FAIL'}  {name.replace('_', ' ')}"
        for name, passed in checks.items()
    )
    design_ax.set_title(
        "f  Controlled pilot and frozen gate", loc="left", fontweight="bold"
    )
    design_ax.text(
        0.0,
        0.96,
        "Changed axis\n"
        "  distal τ: 5 versus 125\n\n"
        "Held fixed\n"
        "  middle τ = 25; soma τ = 5\n"
        "  FF profile = [1, 1, 50]\n"
        "  REC contacts = 4,288\n"
        "  decoder, optimizer, seeds, banks\n\n"
        f"Decision: {decision}\n{check_lines}\n\n"
        "Delay panel is descriptive and\nnot part of the aggregate gate.",
        va="top",
        ha="left",
        linespacing=1.25,
    )
    fig.suptitle(
        "Fixed-soma mechanism pilot (n=3 paired trained seeds; descriptive only)",
        fontsize=12,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def analyze(
    config_path: Path = DEFAULT_CONFIG, output_dir: Path | None = None
) -> dict[str, Any]:
    config, _ = load_config(config_path)
    destination = output_dir or (
        _resolve(config["study"]["output_root"]).parent / "analysis_v1"
    )
    statistics_path = destination / "statistics.json"
    figure_path = destination / "figure_mechanism_identification_pilot_v1.pdf"
    if statistics_path.exists() or figure_path.exists():
        raise MechanismPilotAnalysisError("Analysis output already exists")
    statistics = compute_statistics(config_path)
    _atomic_json(statistics_path, statistics, readonly=True)
    render_figure(statistics, figure_path)
    return {
        "statistics": str(statistics_path),
        "figure": str(figure_path),
        "figure_png": str(figure_path.with_suffix(".png")),
        "decision": statistics["go_no_go"]["decision"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.config, args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["analyze", "compute_statistics", "render_figure"]
