"""Analyze the frozen-checkpoint causal-current pilot without null tests."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml

from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_causal_current import (
    MARKER_SCHEMA,
    REPLICATE_MARKER_SCHEMA,
    RESULT_SCHEMA,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_counterfactual_bank import (
    N_REPLICATES,
    ROLE_CELLS,
    _regular_file,
    _resolve_repo_path,
    _sha256,
    _strict_json,
)

ANALYSIS_CONFIG_SCHEMA = "dendritic-rnn-causal-current-analysis-v1"
STATISTICS_SCHEMA = "dendritic-rnn-causal-current-statistics-v1"
COMPLETE_SCHEMA = "dendritic-rnn-causal-current-analysis-marker-v1"
ENDPOINTS = ("ff_write", "rec_late_delay")
DEPTHS = (0, 1, 2)
ROLE_TITLES = {
    "hierarchical_aligned": "Hierarchical aligned",
    "hierarchical_reverse": "Hierarchical reverse",
    "reversed_aligned": "Reversed aligned",
    "reversed_reverse": "Reversed reverse",
}
ROLE_COLORS = {
    "hierarchical_aligned": "#377eb8",
    "hierarchical_reverse": "#4daf4a",
    "reversed_aligned": "#984ea3",
    "reversed_reverse": "#e377c2",
}


class CausalCurrentAnalysisError(RuntimeError):
    """Raised when source completeness or analysis provenance is invalid."""


def paired_bootstrap_interval(
    values: np.ndarray,
    *,
    draws: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    """Descriptive seed bootstrap interval for a one-dimensional vector."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise CausalCurrentAnalysisError("Bootstrap values must be finite vector")
    if draws < 1 or not 0.0 < confidence_level < 1.0:
        raise CausalCurrentAnalysisError("Invalid bootstrap settings")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1, dtype=np.float64)
    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lower), float(upper)


def configured_depth_selectivity(
    values_by_depth: Mapping[int, np.ndarray], *, placement: str
) -> np.ndarray:
    """Configured extreme-depth effect minus the opposite extreme-depth effect."""

    if set(values_by_depth) != set(DEPTHS):
        raise CausalCurrentAnalysisError("Depth profile must contain depths 0, 1, 2")
    proximal = np.asarray(values_by_depth[0], dtype=np.float64)
    distal = np.asarray(values_by_depth[2], dtype=np.float64)
    if proximal.shape != distal.shape or proximal.ndim != 1:
        raise CausalCurrentAnalysisError("Depth effects must be paired seed vectors")
    if placement == "proximal":
        return proximal - distal
    if placement == "distal":
        return distal - proximal
    raise CausalCurrentAnalysisError(f"Unknown placement: {placement}")


def _load_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = _regular_file(path, "causal-current analysis config")
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise CausalCurrentAnalysisError("Invalid analysis YAML") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != ANALYSIS_CONFIG_SCHEMA
    ):
        raise CausalCurrentAnalysisError("Unsupported analysis config schema")
    sources = payload.get("sources")
    analysis = payload.get("analysis")
    if not isinstance(sources, Mapping) or not isinstance(analysis, Mapping):
        raise CausalCurrentAnalysisError("Analysis config is incomplete")
    if analysis.get("inferential_status") != "descriptive_no_null_tests":
        raise CausalCurrentAnalysisError("Pilot must remain descriptive")
    return payload, path


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _load_sources(result_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    effects: dict[str, dict[str, dict[int, list[float]]]] = {
        role: {endpoint: {depth: [] for depth in DEPTHS} for endpoint in ENDPOINTS}
        for role in ROLE_CELLS
    }
    records = []
    for replicate in range(N_REPLICATES):
        replicate_dir = result_root / f"r{replicate:03d}"
        marker_path = _regular_file(replicate_dir / "COMPLETE.json", "replicate marker")
        marker = _strict_json(marker_path)
        if (
            marker.get("schema_version") != REPLICATE_MARKER_SCHEMA
            or marker.get("state") != "COMPLETE"
            or set(marker.get("roles", {})) != set(ROLE_CELLS)
        ):
            raise CausalCurrentAnalysisError("Replicate marker is incomplete")
        for role in ROLE_CELLS:
            role_marker = marker["roles"][role]
            if (
                role_marker.get("schema_version") != MARKER_SCHEMA
                or role_marker.get("state") != "COMPLETE"
            ):
                raise CausalCurrentAnalysisError("Role marker is incomplete")
            result_record = role_marker.get("result")
            if not isinstance(result_record, Mapping):
                raise CausalCurrentAnalysisError("Role marker lacks result binding")
            result_path = _regular_file(
                Path(str(result_record["path"])), "causal-current result"
            )
            if _sha256(result_path) != result_record.get("sha256"):
                raise CausalCurrentAnalysisError("Causal-current result hash changed")
            result = _strict_json(result_path)
            if (
                result.get("schema_version") != RESULT_SCHEMA
                or result.get("role") != role
                or result.get("replicate") != f"r{replicate:03d}"
            ):
                raise CausalCurrentAnalysisError(
                    "Causal-current result identity changed"
                )
            audits = result.get("audits", {})
            expected_audits = {
                "same_batch_direct_manual_noop": "PASSED_BITWISE",
                "prior_partition_bridge": ("PASSED_BOUNDED_WITH_PREDICTION_IDENTITY"),
                "state_immutability": "PASSED",
            }
            if audits != expected_audits:
                raise CausalCurrentAnalysisError("A causal-current audit did not pass")
            summaries = result["intervention"]["summaries"]
            for endpoint in ENDPOINTS:
                for depth in DEPTHS:
                    row = summaries[endpoint]["depths"][str(depth)]
                    direct = row["direct_intervention_audit"]
                    if direct["directly_changed_trial_timesteps"] <= 0:
                        raise CausalCurrentAnalysisError(
                            "Intervention changed no current"
                        )
                    value = float(row["mean_target_margin_drop"])
                    if not np.isfinite(value):
                        raise CausalCurrentAnalysisError("Non-finite causal effect")
                    effects[role][endpoint][depth].append(value)
            records.append(
                {
                    "replicate": f"r{replicate:03d}",
                    "role": role,
                    "result_path": str(result_path),
                    "result_sha256": _sha256(result_path),
                    "prior_max_abs_difference": float(
                        result["prior_counterfactual_logits"]["max_abs_difference"]
                    ),
                }
            )
    arrays = {
        role: {
            endpoint: {
                depth: np.asarray(values, dtype=np.float64)
                for depth, values in by_depth.items()
            }
            for endpoint, by_depth in by_endpoint.items()
        }
        for role, by_endpoint in effects.items()
    }
    return arrays, records


def _summarize(
    effects: Mapping[str, Any], *, draws: int, confidence: float, seed: int
) -> dict[str, Any]:
    roles = {}
    offset = 0
    for role, cell in ROLE_CELLS.items():
        role_summary: dict[str, Any] = {"cell": list(cell), "endpoints": {}}
        for endpoint_index, endpoint in enumerate(ENDPOINTS):
            profile = {}
            for depth in DEPTHS:
                vector = effects[role][endpoint][depth]
                interval = paired_bootstrap_interval(
                    vector,
                    draws=draws,
                    confidence_level=confidence,
                    seed=seed + offset,
                )
                offset += 1
                profile[str(depth)] = {
                    "seed_values": vector.tolist(),
                    "mean": float(vector.mean(dtype=np.float64)),
                    "bootstrap_interval": list(interval),
                }
            placement = cell[1] if endpoint == "ff_write" else cell[2]
            selectivity = configured_depth_selectivity(
                effects[role][endpoint], placement=placement
            )
            selectivity_interval = paired_bootstrap_interval(
                selectivity,
                draws=draws,
                confidence_level=confidence,
                seed=seed + 100 + offset + endpoint_index,
            )
            role_summary["endpoints"][endpoint] = {
                "placement": placement,
                "profile": profile,
                "configured_depth_selectivity": {
                    "definition": "configured_extreme_minus_opposite_extreme",
                    "seed_values": selectivity.tolist(),
                    "mean": float(selectivity.mean(dtype=np.float64)),
                    "bootstrap_interval": list(selectivity_interval),
                },
            }
        roles[role] = role_summary
    return roles


def _render_figure(statistics: Mapping[str, Any], output: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2))
    depth_offsets = np.linspace(-0.18, 0.18, len(ROLE_CELLS))
    for panel, endpoint, title in (
        (axes[0, 0], "ff_write", "Feedforward current during stimulus 1"),
        (axes[0, 1], "rec_late_delay", "Recurrent current during late delay"),
    ):
        for role_index, role in enumerate(ROLE_CELLS):
            rows = statistics["roles"][role]["endpoints"][endpoint]["profile"]
            x = np.asarray(DEPTHS, dtype=np.float64) + depth_offsets[role_index]
            means = np.asarray([rows[str(depth)]["mean"] for depth in DEPTHS])
            intervals = np.asarray(
                [rows[str(depth)]["bootstrap_interval"] for depth in DEPTHS]
            )
            panel.errorbar(
                x,
                means,
                yerr=np.vstack((means - intervals[:, 0], intervals[:, 1] - means)),
                marker="o",
                linewidth=1.7,
                capsize=3,
                color=ROLE_COLORS[role],
                label=ROLE_TITLES[role],
            )
        panel.axhline(0.0, color="#555555", linestyle="--", linewidth=1.2)
        panel.set_xticks(DEPTHS, ["soma (0)", "middle (1)", "distal (2)"])
        panel.set_ylabel("target-margin drop")
        panel.set_title(title)
        panel.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, loc="best")

    role_y = np.arange(len(ROLE_CELLS))
    for panel, endpoint, title in (
        (axes[1, 0], "ff_write", "FF configured-depth selectivity"),
        (axes[1, 1], "rec_late_delay", "REC configured-depth selectivity"),
    ):
        for row_index, role in enumerate(ROLE_CELLS):
            summary = statistics["roles"][role]["endpoints"][endpoint][
                "configured_depth_selectivity"
            ]
            values = np.asarray(summary["seed_values"], dtype=np.float64)
            mean = float(summary["mean"])
            low, high = summary["bootstrap_interval"]
            jitter = np.linspace(-0.12, 0.12, len(values))
            panel.scatter(
                values,
                row_index + jitter,
                s=28,
                alpha=0.55,
                color=ROLE_COLORS[role],
            )
            panel.errorbar(
                mean,
                row_index,
                xerr=[[mean - low], [high - mean]],
                fmt="D",
                markersize=7,
                capsize=3,
                color="black",
                markerfacecolor=ROLE_COLORS[role],
            )
        panel.axvline(0.0, color="#555555", linestyle="--", linewidth=1.2)
        panel.set_yticks(role_y, [ROLE_TITLES[role] for role in ROLE_CELLS])
        panel.invert_yaxis()
        panel.set_xlabel("configured extreme - opposite extreme")
        panel.set_title(title)
        panel.grid(axis="x", alpha=0.2)
    for label, panel in zip("ABCD", axes.flat, strict=True):
        panel.text(
            -0.13,
            1.05,
            label,
            transform=panel.transAxes,
            fontsize=15,
            fontweight="bold",
        )
    fig.suptitle(
        "Same-checkpoint, phase-specific current substitutions",
        fontsize=17,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.01,
        (
            "n=10 trained-model seeds. Positive drop means fixed-mean substitution "
            "reduced the correct-class margin; negative means it increased the margin. "
            "Intervals are descriptive seed bootstraps; no null tests."
        ),
        ha="center",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.99, 0.95))
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_analysis(*, analysis_config_path: Path) -> dict[str, Any]:
    config, config_path = _load_config(analysis_config_path)
    sources = config["sources"]
    settings = config["analysis"]
    study_config = _regular_file(
        _resolve_repo_path(str(sources["study_config"])), "causal-current study config"
    )
    result_root = _resolve_repo_path(str(sources["result_root"])).resolve()
    output_root = _resolve_repo_path(str(settings["output_root"])).resolve()
    if any("/drafts/" in str(path) for path in (result_root, output_root)):
        raise CausalCurrentAnalysisError("Draft-owned analysis paths are forbidden")
    if output_root.exists():
        raise CausalCurrentAnalysisError(
            "Causal-current analysis output already exists"
        )
    output_root.mkdir(parents=True)
    effects, records = _load_sources(result_root)
    draws = int(settings["bootstrap_draws"])
    confidence = float(settings["confidence_level"])
    seed = int(settings["analysis_seed"])
    statistics = {
        "schema_version": STATISTICS_SCHEMA,
        "inferential_status": "descriptive_no_null_tests",
        "estimand": (
            "ordinary target margin minus target margin after a feature-wise "
            "fixed-mean raw-pathway-current substitution"
        ),
        "replication_unit": "paired trained-model seed",
        "n_replicates": N_REPLICATES,
        "n_complete_cell_seed_rows": len(records),
        "bootstrap": {
            "draws": draws,
            "confidence_level": confidence,
            "seed": seed,
        },
        "roles": _summarize(effects, draws=draws, confidence=confidence, seed=seed),
        "audits": {
            "all_source_rows_complete": len(records) == 40,
            "same_batch_direct_manual_bitwise_passes": 40,
            "state_immutability_passes": 40,
            "maximum_prior_partition_bridge_difference": max(
                row["prior_max_abs_difference"] for row in records
            ),
            "source_records": records,
        },
        "provenance": {
            "analysis_config": {
                "path": str(config_path),
                "sha256": _sha256(config_path),
            },
            "study_config": {
                "path": str(study_config),
                "sha256": _sha256(study_config),
            },
        },
    }
    statistics_path = output_root / "statistics.json"
    _atomic_json(statistics_path, statistics)
    figure_path = output_root / "figure_causal_current_pilot_v1.pdf"
    _render_figure(statistics, figure_path)
    marker = {
        "schema_version": COMPLETE_SCHEMA,
        "state": "COMPLETE",
        "statistics": {
            "path": str(statistics_path),
            "sha256": _sha256(statistics_path),
        },
        "figure": {"path": str(figure_path), "sha256": _sha256(figure_path)},
        "figure_png": {
            "path": str(figure_path.with_suffix(".png")),
            "sha256": _sha256(figure_path.with_suffix(".png")),
        },
    }
    _atomic_json(output_root / "COMPLETE.json", marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-config", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run_analysis(analysis_config_path=args.analysis_config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CausalCurrentAnalysisError",
    "configured_depth_selectivity",
    "paired_bootstrap_interval",
    "run_analysis",
]
