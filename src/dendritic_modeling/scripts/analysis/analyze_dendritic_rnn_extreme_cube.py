"""Collect, analyze, and render the dendritic-RNN extreme placement cube.

The eight-cell cube crosses feedforward placement, recurrent placement, and
dendritic time-constant ordering. Six cells come from the immutable primary
confirmatory manifest; two same-side/reversed cells come from the repository-
owned extension manifest. All inference is paired over the ten trained-model
seeds. Frozen-bank trials are used only to compute each trained model's OOD
accuracy and are never treated as independent inferential units.

This module is deliberately independent of ``drafts``. It reads standard run
artifacts, writes compact provenance-bound results under ``results``, and
renders the paper-facing PDF/PNG from those results.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dendritic_modeling.analysis.statistics import bootstrap_mean_interval

TASK = "romo_delay_comparison_positive"
PREDICTION_SCHEMA = "dendritic-rnn-evaluation-predictions-v1"
COLLECTION_SCHEMA = "dendritic-rnn-extreme-cube-collection-v2"
STATISTICS_SCHEMA = "dendritic-rnn-extreme-cube-statistics-v2"
FIGURE_SOURCE_SCHEMA = "dendritic-rnn-extreme-cube-figure-source-v2"
ANALYSIS_MARKER_SCHEMA = "dendritic-rnn-extreme-cube-analysis-marker-v2"
N_REPLICATES = 10
N_TRIALS = 8192
BOOTSTRAP_DRAWS = 10_000
ANALYSIS_SEED = 20_260_718
FAMILY_ALPHA = 0.05
CONTRAST_ORDER = ("ff_by_tau", "rec_by_tau", "ff_by_rec_by_tau")
BASE_PARTITION = "kempner_h100"
PRIORITY_PARTITION = "kempner_h100_priority"
CELL_ORDER = tuple(
    (tau, ff, rec)
    for tau in ("hierarchical", "reversed")
    for ff in ("distal", "proximal")
    for rec in ("distal", "proximal")
)
PRIMARY_CELLS = {
    *(cell for cell in CELL_ORDER if cell[0] == "hierarchical"),
    ("reversed", "distal", "proximal"),
    ("reversed", "proximal", "distal"),
}
EXTENSION_CELLS = {
    ("reversed", "distal", "distal"),
    ("reversed", "proximal", "proximal"),
}
CONDITION_PATTERN = re.compile(
    r"^ff_(distal|proximal)__rec_(distal|proximal)__"
    r"tau_(hierarchical|reversed)__standard$"
)


class ExtremeCubeAnalysisError(RuntimeError):
    """Raised when an artifact violates the frozen analysis contract."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ExtremeCubeAnalysisError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise ExtremeCubeAnalysisError(f"JSON root must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _regular_file(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ExtremeCubeAnalysisError(f"Missing {label}: {path}") from exc
    if path.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise ExtremeCubeAnalysisError(f"{label} must be a regular file: {path}")
    return resolved


def _condition_cell(condition: str) -> tuple[str, str, str]:
    match = CONDITION_PATTERN.fullmatch(condition)
    if match is None:
        raise ExtremeCubeAnalysisError(f"Unexpected cube condition: {condition!r}")
    ff, rec, tau = match.groups()
    return tau, ff, rec


def _condition_name(cell: tuple[str, str, str]) -> str:
    tau, ff, rec = cell
    return f"ff_{ff}__rec_{rec}__tau_{tau}__standard"


def _subseed(label: str, *, analysis_seed: int = ANALYSIS_SEED) -> int:
    payload = f"paired-statistics-v1\0{analysis_seed}\0{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def contrast_coefficients(cell: tuple[str, str, str]) -> dict[str, float]:
    """Return the prespecified coefficients for one cell of the 2x2x2 cube."""

    tau, ff, rec = cell
    if cell not in CELL_ORDER:
        raise ValueError(f"Not an extreme-cube cell: {cell}")
    tau_sign = 1.0 if tau == "hierarchical" else -1.0
    ff_sign = 1.0 if ff == "proximal" else -1.0
    rec_sign = 1.0 if rec == "proximal" else -1.0
    return {
        "ff_by_tau": 0.5 * tau_sign * ff_sign,
        "rec_by_tau": 0.5 * tau_sign * rec_sign,
        "ff_by_rec_by_tau": tau_sign * ff_sign * rec_sign,
    }


def exact_two_sided_sign_flip(values: Sequence[float]) -> dict[str, Any]:
    """Enumerate the exact paired-seed sign-flip null distribution."""

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Sign-flip values must be a non-empty finite vector")
    observed = abs(float(array.mean()))
    tolerance = np.finfo(np.float64).eps * max(1.0, observed) * 16.0
    extreme = 0
    for assignment in itertools.product((-1.0, 1.0), repeat=array.size):
        statistic = abs(float(np.mean(array * np.asarray(assignment))))
        extreme += int(statistic >= observed - tolerance)
    assignments = 2 ** int(array.size)
    return {
        "method": "exact_two_sided_paired_seed_sign_flip",
        "observed_absolute_mean": observed,
        "extreme_assignment_count": extreme,
        "assignment_count": assignments,
        "p_value": extreme / assignments,
        "numerical_tolerance": tolerance,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, dict[str, Any]]:
    """Holm step-down adjustment with cumulative maximum and cap at one."""

    if not p_values:
        raise ValueError("The Holm family cannot be empty")
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in p_values.values()
    ):
        raise ValueError("Holm p-values must be finite and within [0, 1]")
    ordered = sorted(p_values, key=lambda label: (p_values[label], label))
    family_size = len(ordered)
    cumulative = 0.0
    adjusted: dict[str, dict[str, Any]] = {}
    for rank, label in enumerate(ordered, start=1):
        candidate = min(1.0, (family_size - rank + 1) * p_values[label])
        cumulative = max(cumulative, candidate)
        adjusted[label] = {
            "rank": rank,
            "raw_p_value": float(p_values[label]),
            "adjusted_p_value": cumulative,
            "reject_at_family_alpha": bool(cumulative <= FAMILY_ALPHA),
        }
    return adjusted


def _normalized_seeds(row: Mapping[str, Any], *, source: str) -> dict[str, int]:
    seeds = row.get("seeds")
    if not isinstance(seeds, Mapping):
        raise ExtremeCubeAnalysisError(f"{source} row lacks seeds")
    if source == "primary":
        mapping = {
            "seed": "base_seed",
            "dataset_seed": "dataset_seed",
            "split_seed": "split_seed",
            "model_seed": "model_init_seed",
            "topology_seed": "topology_seed",
            "loader_seed": "loader_seed",
            "evaluation_seed": "evaluation_seed",
            "probe_seed": "probe_seed",
        }
    else:
        mapping = {
            name: name
            for name in (
                "seed",
                "dataset_seed",
                "split_seed",
                "model_seed",
                "topology_seed",
                "loader_seed",
                "evaluation_seed",
                "probe_seed",
            )
        }
    try:
        return {
            target: int(seeds[source_name]) for target, source_name in mapping.items()
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ExtremeCubeAnalysisError(
            f"Invalid seed record for {row.get('run_id')}"
        ) from exc


def _reference_extension_rows(
    reference_manifest: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    rows = reference_manifest.get("runs")
    if not isinstance(rows, list):
        raise ExtremeCubeAnalysisError("Extension reference manifest lacks runs")
    selected = {
        row["run_id"]: row
        for row in rows
        if isinstance(row, dict) and row.get("run_id", "").startswith("extreme-cube-")
    }
    if len(selected) != 20:
        raise ExtremeCubeAnalysisError(
            f"Expected 20 extension reference rows, found {len(selected)}"
        )
    return selected


def _validate_prediction_artifact(
    *,
    run_row: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest_sha256: str,
    require_evaluation_marker: bool,
    expected_runtime_source_sha256: str | None,
) -> dict[str, Any]:
    run_id = str(run_row["run_id"])
    run_dir = Path(str(run_row["output_dir"])).resolve()
    if "/drafts/" in str(run_dir):
        raise ExtremeCubeAnalysisError(
            f"Draft-owned result path is forbidden: {run_dir}"
        )
    complete_path = _regular_file(run_dir / "COMPLETE.json", "training COMPLETE marker")
    complete = _strict_json(complete_path)
    if complete.get("state") != "COMPLETE" or complete.get("run_id") != run_id:
        raise ExtremeCubeAnalysisError(f"Training marker mismatch for {run_id}")
    slurm = complete.get("slurm")
    if not isinstance(slurm, Mapping):
        raise ExtremeCubeAnalysisError(
            f"Training marker lacks Slurm provenance for {run_id}"
        )
    partition = slurm.get("partition", slurm.get("SLURM_JOB_PARTITION"))
    if partition not in {BASE_PARTITION, PRIORITY_PARTITION}:
        raise ExtremeCubeAnalysisError(
            f"Unexpected training partition for {run_id}: {partition!r}"
        )

    prediction_path = _regular_file(
        run_dir
        / str(contract.get("predictions_artifact", "evaluation/predictions.json")),
        "prediction artifact",
    )
    prediction_sha256 = _sha256(prediction_path)
    if require_evaluation_marker:
        marker_path = _regular_file(
            prediction_path.parent / "COMPLETE.json", "evaluation COMPLETE marker"
        )
        marker = _strict_json(marker_path)
        expected_marker = {
            "state": "COMPLETE",
            "run_id": run_id,
            "prediction_sha256": prediction_sha256,
            "n_trials": int(contract["n_trials"]),
            "bank_id": contract["bank_id"],
        }
        if expected_runtime_source_sha256 is not None:
            expected_marker["evaluation_runtime_source_sha256"] = (
                expected_runtime_source_sha256
            )
        if any(
            marker.get(key) != expected for key, expected in expected_marker.items()
        ):
            raise ExtremeCubeAnalysisError(f"Evaluation marker mismatch for {run_id}")

    payload = _strict_json(prediction_path)
    expected_header = {
        "schema_version": PREDICTION_SCHEMA,
        "run_id": run_id,
        "manifest_sha256": manifest_sha256,
        "bank_id": contract["bank_id"],
        "bank_sha256": contract["bank"]["sha256"],
        "index_sha256": contract["index"]["sha256"],
        "evaluation_seed": int(contract["evaluation_seed"]),
        "trial_ids_sha256": contract["trial_ids_sha256"],
        "trial_metadata_sha256": contract["trial_metadata_sha256"],
        "n_trials": int(contract["n_trials"]),
    }
    if expected_runtime_source_sha256 is not None:
        expected_header["evaluation_runtime_source_sha256"] = (
            expected_runtime_source_sha256
        )
    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in expected_header.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise ExtremeCubeAnalysisError(
            f"Prediction header mismatch for {run_id}: {mismatches}"
        )
    if payload.get("population_noop_audit", {}).get("status") != "PASSED":
        raise ExtremeCubeAnalysisError(
            f"Population no-op audit did not pass for {run_id}"
        )

    index_path = _regular_file(Path(str(contract["index"]["path"])), "bank index")
    if _sha256(index_path) != contract["index"]["sha256"]:
        raise ExtremeCubeAnalysisError(f"Frozen index hash mismatch for {run_id}")
    index = _strict_json(index_path)
    trial_ids = index.get("trial_ids")
    trials = index.get("trials")
    if not isinstance(trial_ids, list) or not isinstance(trials, list):
        raise ExtremeCubeAnalysisError(f"Frozen index is incomplete for {run_id}")
    if len(trial_ids) != N_TRIALS or len(trials) != N_TRIALS:
        raise ExtremeCubeAnalysisError(f"Frozen index length mismatch for {run_id}")
    if _sha256_json(trial_ids) != contract["trial_ids_sha256"]:
        raise ExtremeCubeAnalysisError(f"Frozen trial-ID hash mismatch for {run_id}")
    if _sha256_json(trials) != contract["trial_metadata_sha256"]:
        raise ExtremeCubeAnalysisError(f"Frozen metadata hash mismatch for {run_id}")

    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != N_TRIALS:
        raise ExtremeCubeAnalysisError(f"Prediction length mismatch for {run_id}")
    correct_count = 0
    observed_ids = []
    for position, (prediction, metadata) in enumerate(
        zip(predictions, trials, strict=True)
    ):
        if not isinstance(prediction, Mapping):
            raise ExtremeCubeAnalysisError(
                f"Invalid prediction row {position} for {run_id}"
            )
        trial_id = prediction.get("trial_id")
        observed_ids.append(trial_id)
        target = prediction.get("target")
        predicted = prediction.get("prediction")
        correct = prediction.get("correct")
        if trial_id != trial_ids[position] or trial_id != metadata.get("trial_id"):
            raise ExtremeCubeAnalysisError(f"Trial order mismatch for {run_id}")
        if target != metadata.get("target") or not isinstance(target, int):
            raise ExtremeCubeAnalysisError(f"Target mismatch for {run_id}")
        if not isinstance(predicted, int) or predicted not in (0, 1):
            raise ExtremeCubeAnalysisError(f"Invalid prediction class for {run_id}")
        if type(correct) is not bool or correct != (predicted == target):
            raise ExtremeCubeAnalysisError(f"Incorrect correctness flag for {run_id}")
        logits = prediction.get("logits")
        if (
            not isinstance(logits, list)
            or len(logits) != 2
            or any(
                not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in logits
            )
        ):
            raise ExtremeCubeAnalysisError(f"Invalid logits for {run_id}")
        correct_count += int(correct)
    if _sha256_json(observed_ids) != contract["trial_ids_sha256"]:
        raise ExtremeCubeAnalysisError(
            f"Prediction trial-ID hash mismatch for {run_id}"
        )

    return {
        "run_id": run_id,
        "prediction_artifact": str(prediction_path),
        "prediction_sha256": prediction_sha256,
        "training_complete_marker": str(complete_path),
        "training_complete_marker_sha256": _sha256(complete_path),
        "training_partition": partition,
        "bank_id": contract["bank_id"],
        "bank_sha256": contract["bank"]["sha256"],
        "index_sha256": contract["index"]["sha256"],
        "trial_ids_sha256": contract["trial_ids_sha256"],
        "trial_metadata_sha256": contract["trial_metadata_sha256"],
        "n_trials": N_TRIALS,
        "correct_count": correct_count,
        "accuracy": correct_count / N_TRIALS,
        "population_noop_status": "PASSED",
    }


def collect_cube(
    *,
    primary_manifest_path: Path,
    extension_manifest_path: Path,
    extension_reference_manifest_path: Path,
    extension_runtime_source_sha256: str,
) -> dict[str, Any]:
    """Collect the strict 80-run cube without retaining trial-level payloads."""

    paths = {
        "primary": _regular_file(primary_manifest_path, "primary manifest"),
        "extension": _regular_file(extension_manifest_path, "extension manifest"),
        "extension_reference": _regular_file(
            extension_reference_manifest_path, "extension reference manifest"
        ),
    }
    if any("/drafts/" in str(path) for path in paths.values()):
        raise ExtremeCubeAnalysisError("Draft-owned manifest paths are forbidden")
    manifests = {name: _strict_json(path) for name, path in paths.items()}
    manifest_hashes = {name: _sha256(path) for name, path in paths.items()}
    primary = manifests["primary"]
    extension = manifests["extension"]
    if primary.get("schema_version") != "dendritic-rnn-revision-run-manifest-v1":
        raise ExtremeCubeAnalysisError("Unsupported primary manifest schema")
    if extension.get("schema_version") != "dendritic-rnn-standard-training-manifest-v1":
        raise ExtremeCubeAnalysisError("Unsupported extension manifest schema")
    if len(primary.get("runs", [])) != 140 or len(extension.get("runs", [])) != 20:
        raise ExtremeCubeAnalysisError("Manifest run counts changed")
    reference_rows = _reference_extension_rows(manifests["extension_reference"])

    selected: dict[tuple[int, tuple[str, str, str]], dict[str, Any]] = {}
    for source, manifest, expected_cells in (
        ("primary", primary, PRIMARY_CELLS),
        ("extension", extension, EXTENSION_CELLS),
    ):
        for row in manifest["runs"]:
            condition = row.get("condition")
            if (
                not isinstance(condition, str)
                or CONDITION_PATTERN.fullmatch(condition) is None
            ):
                continue
            cell = _condition_cell(condition)
            if cell not in expected_cells:
                continue
            if source == "primary":
                if (
                    row.get("task") != TASK
                    or row.get("uses_ood_evaluation") is not True
                ):
                    raise ExtremeCubeAnalysisError(
                        f"Primary scientific contract changed for {row.get('run_id')}"
                    )
                match = re.fullmatch(
                    r"romo-confirmatory-replicate-(\d{3})",
                    str(row.get("pairing_group")),
                )
                contract = row.get("evaluation_bank")
            else:
                match = re.fullmatch(r"r(\d{3})", str(row.get("pairing_group")))
                reference = reference_rows.get(str(row.get("run_id")))
                if reference is None:
                    raise ExtremeCubeAnalysisError(
                        f"Missing extension reference for {row.get('run_id')}"
                    )
                contract = reference.get("evaluation_bank")
            if match is None or not isinstance(contract, Mapping):
                raise ExtremeCubeAnalysisError(
                    f"Invalid pairing/bank contract for {row.get('run_id')}"
                )
            replicate = int(match.group(1))
            if not 0 <= replicate < N_REPLICATES:
                raise ExtremeCubeAnalysisError(
                    f"Unexpected replicate for {row.get('run_id')}"
                )
            key = replicate, cell
            if key in selected:
                raise ExtremeCubeAnalysisError(f"Duplicate cube cell: {key}")
            artifact = _validate_prediction_artifact(
                run_row=row,
                contract=contract,
                manifest_sha256=manifest_hashes[source],
                require_evaluation_marker=(source == "extension"),
                expected_runtime_source_sha256=(
                    extension_runtime_source_sha256 if source == "extension" else None
                ),
            )
            selected[key] = {
                "replicate": f"r{replicate:03d}",
                "condition": condition,
                "tau": cell[0],
                "ff": cell[1],
                "rec": cell[2],
                "source": source,
                "seeds": _normalized_seeds(row, source=source),
                **artifact,
            }

    expected_keys = {
        (replicate, cell) for replicate in range(N_REPLICATES) for cell in CELL_ORDER
    }
    if set(selected) != expected_keys:
        missing = sorted(expected_keys - set(selected))
        extra = sorted(set(selected) - expected_keys)
        raise ExtremeCubeAnalysisError(
            f"Cube is incomplete; missing={missing}, extra={extra}"
        )

    for replicate in range(N_REPLICATES):
        rows = [selected[replicate, cell] for cell in CELL_ORDER]
        reference_seeds = rows[0]["seeds"]
        if any(row["seeds"] != reference_seeds for row in rows[1:]):
            raise ExtremeCubeAnalysisError(f"Seed pairing broke for r{replicate:03d}")
        bank_fields = (
            "bank_id",
            "bank_sha256",
            "index_sha256",
            "trial_ids_sha256",
            "trial_metadata_sha256",
            "n_trials",
        )
        reference_bank = {field: rows[0][field] for field in bank_fields}
        if any(
            {field: row[field] for field in bank_fields} != reference_bank
            for row in rows[1:]
        ):
            raise ExtremeCubeAnalysisError(
                f"Frozen bank pairing broke for r{replicate:03d}"
            )

    route_counts = {
        BASE_PARTITION: sum(
            row["training_partition"] == BASE_PARTITION for row in selected.values()
        ),
        PRIORITY_PARTITION: sum(
            row["training_partition"] == PRIORITY_PARTITION for row in selected.values()
        ),
    }
    if route_counts != {BASE_PARTITION: 48, PRIORITY_PARTITION: 32}:
        raise ExtremeCubeAnalysisError(
            f"Cube training-route composition changed: {route_counts}"
        )
    source_route_counts = {
        source: {
            partition: sum(
                row["source"] == source and row["training_partition"] == partition
                for row in selected.values()
            )
            for partition in (BASE_PARTITION, PRIORITY_PARTITION)
        }
        for source in ("primary", "extension")
    }
    expected_source_route_counts = {
        "primary": {BASE_PARTITION: 48, PRIORITY_PARTITION: 12},
        "extension": {BASE_PARTITION: 0, PRIORITY_PARTITION: 20},
    }
    if source_route_counts != expected_source_route_counts:
        raise ExtremeCubeAnalysisError(
            f"Cube source-by-route composition changed: {source_route_counts}"
        )

    ordered_rows = [
        selected[replicate, cell]
        for replicate in range(N_REPLICATES)
        for cell in CELL_ORDER
    ]
    return {
        "schema_version": COLLECTION_SCHEMA,
        "created_utc": _utc_now(),
        "task": TASK,
        "design": {
            "factors": {
                "tau": ["hierarchical", "reversed"],
                "ff": ["distal", "proximal"],
                "rec": ["distal", "proximal"],
            },
            "cell_order": [list(cell) for cell in CELL_ORDER],
            "replicates": [f"r{index:03d}" for index in range(N_REPLICATES)],
            "independent_unit": "paired_trained_model_seed",
            "trial_role": "within-model_accuracy_measurement_only",
            "n_cells": 8,
            "n_replicates": N_REPLICATES,
            "n_runs": 80,
            "trials_per_run": N_TRIALS,
            "route_audit": {
                "status": "PASSED",
                "observed_counts": route_counts,
                "observed_source_by_route_counts": source_route_counts,
                "expected_counts": {BASE_PARTITION: 48, PRIORITY_PARTITION: 32},
                "expected_source_by_route_counts": expected_source_route_counts,
                "interpretation": (
                    "Training route and calendar are descriptive nuisance variables; "
                    "they do not add replicate credit or alter any inferential gate."
                ),
            },
        },
        "manifests": {
            name: {"path": str(paths[name]), "sha256": manifest_hashes[name]}
            for name in paths
        },
        "runs": ordered_rows,
    }


def analyze_collection(
    collection: Mapping[str, Any],
    *,
    analysis_seed: int = ANALYSIS_SEED,
    bootstrap_draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    if collection.get("schema_version") != COLLECTION_SCHEMA:
        raise ExtremeCubeAnalysisError("Unsupported collection schema")
    rows = collection.get("runs")
    if not isinstance(rows, list) or len(rows) != 80:
        raise ExtremeCubeAnalysisError("Collection does not contain the exact 80 runs")
    accuracy = {
        (int(row["replicate"][1:]), (row["tau"], row["ff"], row["rec"])): float(
            row["accuracy"]
        )
        for row in rows
    }
    if len(accuracy) != 80 or not all(
        math.isfinite(value) for value in accuracy.values()
    ):
        raise ExtremeCubeAnalysisError(
            "Collection accuracies are incomplete or non-finite"
        )

    cell_summaries: dict[str, Any] = {}
    for cell in CELL_ORDER:
        values = [accuracy[replicate, cell] for replicate in range(N_REPLICATES)]
        summary = bootstrap_mean_interval(
            values,
            draws=bootstrap_draws,
            seed=_subseed(
                f"extreme-cube:{TASK}:cell:{_condition_name(cell)}",
                analysis_seed=analysis_seed,
            ),
            require_n=N_REPLICATES,
        ).to_dict()
        cell_summaries[_condition_name(cell)] = {
            "tau": cell[0],
            "ff": cell[1],
            "rec": cell[2],
            "values_by_replicate": values,
            "bootstrap_95_percent_ci": [summary["lower"], summary["upper"]],
            **{
                key: summary[key]
                for key in (
                    "mean",
                    "standard_deviation",
                    "n_units",
                    "draws",
                    "seed",
                    "method",
                )
            },
        }

    contrast_vectors: dict[str, list[float]] = {name: [] for name in CONTRAST_ORDER}
    for replicate in range(N_REPLICATES):
        for name in CONTRAST_ORDER:
            value = sum(
                contrast_coefficients(cell)[name] * accuracy[replicate, cell]
                for cell in CELL_ORDER
            )
            contrast_vectors[name].append(float(value))

    partitions_by_replicate = {
        replicate: {
            str(row["training_partition"])
            for row in rows
            if int(row["replicate"][1:]) == replicate
        }
        for replicate in range(N_REPLICATES)
    }
    mixed_route_replicates = [
        replicate
        for replicate, partitions in partitions_by_replicate.items()
        if partitions == {BASE_PARTITION, PRIORITY_PARTITION}
    ]
    all_priority_replicates = [
        replicate
        for replicate, partitions in partitions_by_replicate.items()
        if partitions == {PRIORITY_PARTITION}
    ]
    if mixed_route_replicates != list(range(8)) or all_priority_replicates != [8, 9]:
        raise ExtremeCubeAnalysisError(
            "Expected eight mixed-route blocks followed by two all-priority blocks; "
            f"observed mixed={mixed_route_replicates}, priority={all_priority_replicates}"
        )

    route_sensitivity = {}
    for group, replicate_indices in (
        ("mixed_route", mixed_route_replicates),
        ("all_priority", all_priority_replicates),
    ):
        route_sensitivity[group] = {
            "replicates": [f"r{replicate:03d}" for replicate in replicate_indices],
            "n_units": len(replicate_indices),
            "contrasts": {
                name: {
                    "values_by_replicate": [
                        contrast_vectors[name][replicate]
                        for replicate in replicate_indices
                    ],
                    "mean": float(
                        np.mean(
                            [
                                contrast_vectors[name][replicate]
                                for replicate in replicate_indices
                            ],
                            dtype=np.float64,
                        )
                    ),
                }
                for name in CONTRAST_ORDER
            },
        }

    contrast_summaries: dict[str, Any] = {}
    raw_p_values: dict[str, float] = {}
    for name in CONTRAST_ORDER:
        label = f"extreme-cube:{TASK}:{name}"
        bootstrap = bootstrap_mean_interval(
            contrast_vectors[name],
            draws=bootstrap_draws,
            seed=_subseed(label, analysis_seed=analysis_seed),
            require_n=N_REPLICATES,
        ).to_dict()
        sign_flip = exact_two_sided_sign_flip(contrast_vectors[name])
        raw_p_values[name] = sign_flip["p_value"]
        contrast_summaries[name] = {
            "definition": {
                _condition_name(cell): contrast_coefficients(cell)[name]
                for cell in CELL_ORDER
            },
            "values_by_replicate": contrast_vectors[name],
            "mean": bootstrap["mean"],
            "standard_deviation": bootstrap["standard_deviation"],
            "bootstrap_95_percent_ci": [bootstrap["lower"], bootstrap["upper"]],
            "bootstrap": bootstrap,
            "sign_flip_test": sign_flip,
        }

    holm = holm_adjust(raw_p_values)
    for name in CONTRAST_ORDER:
        lower, upper = contrast_summaries[name]["bootstrap_95_percent_ci"]
        excludes_zero = bool(lower > 0.0 or upper < 0.0)
        contrast_summaries[name]["holm"] = holm[name]
        contrast_summaries[name]["support_gate"] = {
            "bootstrap_interval_excludes_zero": excludes_zero,
            "holm_rejects_at_family_alpha": holm[name]["reject_at_family_alpha"],
            "supported": bool(excludes_zero and holm[name]["reject_at_family_alpha"]),
        }

    direct = []
    component = []
    residual = []
    for replicate in range(N_REPLICATES):
        direct_value = (
            accuracy[replicate, ("hierarchical", "proximal", "distal")]
            - accuracy[replicate, ("hierarchical", "distal", "proximal")]
            - accuracy[replicate, ("reversed", "proximal", "distal")]
            + accuracy[replicate, ("reversed", "distal", "proximal")]
        )
        component_value = (
            contrast_vectors["ff_by_tau"][replicate]
            - contrast_vectors["rec_by_tau"][replicate]
        )
        direct.append(float(direct_value))
        component.append(float(component_value))
        residual.append(float(direct_value - component_value))
    max_cell = max(abs(value) for value in accuracy.values())
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, max_cell)
    max_residual = max(abs(value) for value in residual)
    if max_residual > tolerance:
        raise ExtremeCubeAnalysisError(
            f"Algebraic bridge failed: {max_residual} > {tolerance}"
        )

    return {
        "schema_version": STATISTICS_SCHEMA,
        "created_utc": _utc_now(),
        "collection_sha256": _sha256_json(collection),
        "task": TASK,
        "analysis_plan": {
            "independent_unit": "paired_trained_model_seed",
            "n_independent_units": N_REPLICATES,
            "bootstrap": {
                "method": "paired_trained_seed_percentile",
                "resamples": bootstrap_draws,
                "confidence": 0.95,
                "analysis_seed": analysis_seed,
                "subseed_derivation": "sha256 first 8 bytes big-endian of paired-statistics-v1\\0seed\\0label",
            },
            "hypothesis_test": "exact_two_sided_paired_seed_sign_flip",
            "sign_flip_assignments_per_contrast": 2**N_REPLICATES,
            "multiplicity": "Holm step-down with cumulative maximum and cap at one",
            "family": list(CONTRAST_ORDER),
            "family_alpha": FAMILY_ALPHA,
            "support_rule": "Holm-adjusted p <= 0.05 and paired-seed bootstrap 95% CI excludes zero",
        },
        "cell_summaries": cell_summaries,
        "contrasts": contrast_summaries,
        "holm_family": {
            "family": list(CONTRAST_ORDER),
            "alpha": FAMILY_ALPHA,
            "members": holm,
        },
        "route_sensitivity": {
            "role": "descriptive_noninferential_nuisance_check",
            "can_rescue_or_modify_any_claim": False,
            "route_calendar_alias_remains": True,
            "groups": route_sensitivity,
            "route_audit": collection["design"]["route_audit"],
        },
        "algebraic_bridge": {
            "definition": "C = hierarchical(proximal,distal) - hierarchical(distal,proximal) - reversed(proximal,distal) + reversed(distal,proximal)",
            "component_identity": "C = ff_by_tau - rec_by_tau",
            "direct_values_by_replicate": direct,
            "component_values_by_replicate": component,
            "residuals_by_replicate": residual,
            "max_absolute_residual": max_residual,
            "tolerance": tolerance,
            "status": "PASSED",
        },
        "limitations": [
            "The independent unit is the trained-model seed (n=10), not any of the 8,192 frozen-bank trials.",
            "The two extension conditions share seeds and frozen banks with the primary cube cells; this is a paired extension, not an independent replication.",
            "The extension used the same H100 80GB model class but a new optimized CPU/memory/walltime envelope and a later scheduler batch, so this analysis does not establish platform or calendar-batch invariance.",
            "The analysis estimates effects for the fixed architecture, task, training recipe, and extreme placement definitions represented by these manifests.",
            "OOD validation banks remain separate from the sealed test split; no test data enter this analysis.",
        ],
    }


def _atomic_figure(fig: Any, path: Path, **kwargs: Any) -> None:
    temporary = path.with_name(f".{path.stem}.{os.getpid()}{path.suffix}")
    fig.savefig(temporary, **kwargs)
    temporary.replace(path)


def render_figure(statistics: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    """Render a four-panel science/audit figure from frozen statistics."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(12.0, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2)
    cell_grid = grid[0, 0].subgridspec(1, 2, wspace=0.10)
    rec_colors = {"distal": "#3366a8", "proximal": "#c84b31"}
    ff_x = {"distal": 0.0, "proximal": 1.0}
    seed_offsets = np.linspace(-0.035, 0.035, N_REPLICATES)

    cell_axes = []
    for tau_index, tau in enumerate(("hierarchical", "reversed")):
        ax = fig.add_subplot(
            cell_grid[0, tau_index], sharey=cell_axes[0] if cell_axes else None
        )
        cell_axes.append(ax)
        for rec in ("distal", "proximal"):
            values_by_ff = []
            for ff in ("distal", "proximal"):
                row = statistics["cell_summaries"][_condition_name((tau, ff, rec))]
                values_by_ff.append(np.asarray(row["values_by_replicate"]) * 100.0)
            for seed_index in range(N_REPLICATES):
                ax.plot(
                    [
                        ff_x["distal"] + seed_offsets[seed_index],
                        ff_x["proximal"] + seed_offsets[seed_index],
                    ],
                    [values_by_ff[0][seed_index], values_by_ff[1][seed_index]],
                    color=rec_colors[rec],
                    alpha=0.20,
                    linewidth=0.8,
                    zorder=1,
                )
                ax.scatter(
                    [
                        ff_x["distal"] + seed_offsets[seed_index],
                        ff_x["proximal"] + seed_offsets[seed_index],
                    ],
                    [values_by_ff[0][seed_index], values_by_ff[1][seed_index]],
                    color=rec_colors[rec],
                    alpha=0.55,
                    s=11,
                    linewidths=0,
                    zorder=2,
                )
            means = [float(values.mean()) for values in values_by_ff]
            ax.plot(
                [0, 1],
                means,
                color=rec_colors[rec],
                linewidth=2.6,
                marker="D",
                markersize=5,
                zorder=3,
            )
        ax.set_title(f"$\\tau$: {tau}")
        ax.set_xticks([0, 1], ["Distal", "Proximal"])
        ax.set_xlabel("Feedforward placement")
        ax.grid(axis="y", alpha=0.20)
        if tau_index == 0:
            ax.set_ylabel("Frozen OOD accuracy (%)")
            ax.text(
                -0.18, 1.08, "A", transform=ax.transAxes, fontsize=13, fontweight="bold"
            )
        else:
            ax.tick_params(labelleft=False)
    handles = [
        Line2D(
            [0], [0], color=rec_colors[rec], marker="D", linewidth=2, label=f"REC {rec}"
        )
        for rec in ("distal", "proximal")
    ]
    cell_axes[1].legend(handles=handles, frameon=False, loc="best", fontsize=8)

    labels = {
        "ff_by_tau": r"FF $\times$ $\tau$",
        "rec_by_tau": r"REC $\times$ $\tau$",
        "ff_by_rec_by_tau": r"FF $\times$ REC $\times$ $\tau$",
    }
    contrast_colors = {
        "ff_by_tau": "#5b4ba8",
        "rec_by_tau": "#188977",
        "ff_by_rec_by_tau": "#d17a17",
    }
    ax_effect = fig.add_subplot(grid[0, 1])
    y_positions = np.arange(len(CONTRAST_ORDER))[::-1]
    for y, name in zip(y_positions, CONTRAST_ORDER, strict=True):
        row = statistics["contrasts"][name]
        values = np.asarray(row["values_by_replicate"]) * 100.0
        interval = np.asarray(row["bootstrap_95_percent_ci"]) * 100.0
        mean = float(row["mean"]) * 100.0
        jitter = np.linspace(-0.12, 0.12, N_REPLICATES)
        ax_effect.scatter(
            values,
            y + jitter,
            s=22,
            color=contrast_colors[name],
            alpha=0.65,
            edgecolor="white",
            linewidth=0.35,
        )
        ax_effect.plot(
            interval, [y, y], color="black", linewidth=3.0, solid_capstyle="round"
        )
        ax_effect.scatter(
            [mean],
            [y],
            marker="D",
            s=55,
            color=contrast_colors[name],
            edgecolor="black",
            linewidth=0.7,
            zorder=4,
        )
    ax_effect.axvline(0.0, color="0.35", linestyle="--", linewidth=1)
    ax_effect.set_yticks(y_positions, [labels[name] for name in CONTRAST_ORDER])
    ax_effect.set_xlabel("Paired contrast (percentage points)")
    ax_effect.set_title("Seed effects and paired-seed bootstrap 95% CI", loc="left")
    ax_effect.grid(axis="x", alpha=0.20)
    ax_effect.text(
        -0.10, 1.08, "B", transform=ax_effect.transAxes, fontsize=13, fontweight="bold"
    )

    ax_p = fig.add_subplot(grid[1, 0])
    for y, name in zip(y_positions, CONTRAST_ORDER, strict=True):
        row = statistics["contrasts"][name]["holm"]
        raw = max(float(row["raw_p_value"]), np.finfo(float).tiny)
        adjusted = max(float(row["adjusted_p_value"]), np.finfo(float).tiny)
        ax_p.plot(
            [raw, adjusted],
            [y, y],
            color=contrast_colors[name],
            linewidth=2.0,
            alpha=0.75,
        )
        ax_p.scatter(
            raw,
            y,
            marker="o",
            s=40,
            facecolor="white",
            edgecolor=contrast_colors[name],
            linewidth=1.5,
            zorder=3,
        )
        ax_p.scatter(
            adjusted, y, marker="s", s=42, color=contrast_colors[name], zorder=3
        )
        decision = (
            "supported"
            if statistics["contrasts"][name]["support_gate"]["supported"]
            else "not supported"
        )
        ax_p.text(min(1.8, adjusted * 1.45), y, decision, va="center", fontsize=8)
    ax_p.axvline(
        FAMILY_ALPHA,
        color="0.25",
        linestyle="--",
        linewidth=1,
        label=r"family $\alpha=0.05$",
    )
    ax_p.set_xscale("log")
    ax_p.set_xlim(5e-4, 2.5)
    ax_p.set_yticks(y_positions, [labels[name] for name in CONTRAST_ORDER])
    ax_p.set_xlabel("Exact sign-flip p-value")
    ax_p.set_title("Prespecified three-member Holm family", loc="left")
    ax_p.grid(axis="x", which="both", alpha=0.20)
    ax_p.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                markerfacecolor="white",
                markeredgecolor="0.2",
                linewidth=0,
                label="raw p",
            ),
            Line2D([0], [0], marker="s", color="0.2", linewidth=0, label="Holm p"),
            Line2D(
                [0], [0], color="0.25", linestyle="--", label=r"family $\alpha=0.05$"
            ),
        ],
        frameon=False,
        fontsize=8,
        loc="upper left",
    )
    ax_p.text(
        -0.13, 1.08, "C", transform=ax_p.transAxes, fontsize=13, fontweight="bold"
    )

    ax_route = fig.add_subplot(grid[1, 1])
    route_groups = statistics["route_sensitivity"]["groups"]
    group_styles = {
        "mixed_route": {
            "label": "mixed route (n=8)",
            "marker": "o",
            "offset": 0.10,
        },
        "all_priority": {
            "label": "all priority (n=2)",
            "marker": "^",
            "offset": -0.10,
        },
    }
    for name, y in zip(CONTRAST_ORDER, y_positions, strict=True):
        for group, style in group_styles.items():
            row = route_groups[group]["contrasts"][name]
            values = np.asarray(row["values_by_replicate"]) * 100.0
            y_value = y + style["offset"]
            jitter = np.linspace(-0.035, 0.035, len(values))
            ax_route.scatter(
                values,
                y_value + jitter,
                s=20,
                marker=style["marker"],
                facecolor="white",
                edgecolor=contrast_colors[name],
                alpha=0.65,
                linewidth=0.8,
            )
            ax_route.scatter(
                [float(row["mean"]) * 100.0],
                [y_value],
                s=58,
                marker=style["marker"],
                color=contrast_colors[name],
                edgecolor="black",
                linewidth=0.6,
                zorder=4,
            )
    ax_route.axvline(0.0, color="0.35", linestyle="--", linewidth=1)
    ax_route.set_yticks(y_positions, [labels[name] for name in CONTRAST_ORDER])
    ax_route.set_xlabel("Paired contrast (percentage points)")
    ax_route.set_title("Route/calendar sensitivity (descriptive only)", loc="left")
    ax_route.grid(axis="x", alpha=0.20)
    ax_route.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                color="0.25",
                markerfacecolor="0.25",
                linewidth=0,
                label=style["label"],
            )
            for style in group_styles.values()
        ],
        frameon=False,
        fontsize=8,
        loc="best",
    )
    ax_route.text(
        -0.12, 1.08, "D", transform=ax_route.transAxes, fontsize=13, fontweight="bold"
    )

    fig.suptitle(
        "Extreme dendritic placement cube: paired H100 trained-seed analysis",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.015,
        "n=10 paired trained-model seeds; 8,192 frozen OOD-validation trials/model. "
        "Trials are not inferential units. Extension is paired, not an independent replication. "
        "Routes: 48 base + 12 migrated priority + 20 new priority; route view is descriptive.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="0.30",
    )
    pdf = output_dir / "extreme_placement_cube_src_v2.pdf"
    png = output_dir / "extreme_placement_cube_src_v2.png"
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
    primary_manifest_path: Path,
    extension_manifest_path: Path,
    extension_reference_manifest_path: Path,
    evaluation_receipt_path: Path,
    runtime_attestation_path: Path,
    output_dir: Path,
    expected_source_sha256: str | None = None,
) -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    source_sha256 = _sha256(source_path)
    if expected_source_sha256 is not None and source_sha256 != expected_source_sha256:
        raise ExtremeCubeAnalysisError("Analysis source changed after job submission")
    output_dir = output_dir.resolve()
    if "/drafts/" in str(output_dir):
        raise ExtremeCubeAnalysisError("Draft-owned analysis output is forbidden")
    marker_path = output_dir / "COMPLETE.json"
    if marker_path.exists():
        raise ExtremeCubeAnalysisError(f"Analysis is already complete: {output_dir}")
    evaluation_receipt_path = _regular_file(
        evaluation_receipt_path, "evaluation receipt"
    )
    receipt = _strict_json(evaluation_receipt_path)
    if (
        receipt.get("schema_version")
        != "dendritic-rnn-standard-evaluation-submission-receipt-v1"
    ):
        raise ExtremeCubeAnalysisError("Unsupported evaluation receipt")
    if receipt.get("dry_run") is not False or len(receipt.get("jobs", [])) != 20:
        raise ExtremeCubeAnalysisError(
            "Analysis requires one real exact-20 evaluation receipt"
        )
    if (
        Path(str(receipt.get("manifest"))).resolve()
        != extension_manifest_path.resolve()
    ):
        raise ExtremeCubeAnalysisError(
            "Evaluation receipt points to a different extension manifest"
        )
    if receipt.get("manifest_sha256") != _sha256(extension_manifest_path):
        raise ExtremeCubeAnalysisError(
            "Extension manifest changed after evaluation submission"
        )
    runtime_attestation_path = _regular_file(
        runtime_attestation_path, "evaluation runtime attestation"
    )
    attestation = _strict_json(runtime_attestation_path)
    if (
        attestation.get("schema_version")
        != "dendritic-rnn-evaluation-runtime-attestation-v1"
        or attestation.get("status") != "AUDITED_ANALYSIS_CONFIGURATION_ONLY_DRIFT"
        or attestation.get("training_runtime_source_sha256")
        != receipt.get("training_runtime_source_sha256")
        or attestation.get("evaluation_runtime_source_sha256")
        != receipt.get("evaluation_runtime_source_sha256")
    ):
        raise ExtremeCubeAnalysisError(
            "Runtime attestation does not match the frozen evaluation receipt"
        )

    collection = collect_cube(
        primary_manifest_path=primary_manifest_path,
        extension_manifest_path=extension_manifest_path,
        extension_reference_manifest_path=extension_reference_manifest_path,
        extension_runtime_source_sha256=receipt["evaluation_runtime_source_sha256"],
    )
    collection["analysis_source"] = {"path": str(source_path), "sha256": source_sha256}
    collection["evaluation_receipt"] = {
        "path": str(evaluation_receipt_path),
        "sha256": _sha256(evaluation_receipt_path),
    }
    collection["runtime_source_attestation"] = {
        "artifact": {
            "path": str(runtime_attestation_path),
            "sha256": _sha256(runtime_attestation_path),
        },
        **attestation,
    }
    collection_path = output_dir / "collection.json"
    _atomic_json(collection_path, collection)

    statistics = analyze_collection(collection)
    statistics["collection_artifact"] = {
        "path": str(collection_path),
        "sha256": _sha256(collection_path),
    }
    statistics["analysis_source"] = {"path": str(source_path), "sha256": source_sha256}
    statistics["runtime_source_attestation"] = collection["runtime_source_attestation"]
    statistics["limitations"].extend(attestation["limitations"])
    statistics_path = output_dir / "statistics.json"
    _atomic_json(statistics_path, statistics)
    figure_paths = render_figure(statistics, output_dir)

    figure_source = {
        "schema_version": FIGURE_SOURCE_SCHEMA,
        "created_utc": _utc_now(),
        "statistics": {
            "path": str(statistics_path),
            "sha256": _sha256(statistics_path),
        },
        "analysis_source": {"path": str(source_path), "sha256": source_sha256},
        "figures": {
            label: {"path": path, "sha256": _sha256(Path(path))}
            for label, path in figure_paths.items()
        },
        "panel_contract": {
            "A": "all eight cell accuracies with paired trained-seed trajectories",
            "B": "all ten paired-seed values, mean, and paired-seed bootstrap 95% CI for three prespecified contrasts",
            "C": "raw exact sign-flip p-values, Holm-adjusted p-values, and frozen support decision",
            "D": "descriptive mixed-route n=8 and all-priority n=2 contrast values and means; no inferential or claim-changing role",
        },
    }
    figure_source_path = output_dir / "figure_source.json"
    _atomic_json(figure_source_path, figure_source)
    marker = {
        "schema_version": ANALYSIS_MARKER_SCHEMA,
        "state": "COMPLETE",
        "created_utc": _utc_now(),
        "artifacts": {
            path.name: {"path": str(path), "sha256": _sha256(path)}
            for path in (
                collection_path,
                statistics_path,
                figure_source_path,
                Path(figure_paths["pdf"]),
                Path(figure_paths["png"]),
            )
        },
    }
    _atomic_json(marker_path, marker)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-manifest", type=Path, required=True)
    parser.add_argument("--extension-manifest", type=Path, required=True)
    parser.add_argument("--extension-reference-manifest", type=Path, required=True)
    parser.add_argument("--evaluation-receipt", type=Path, required=True)
    parser.add_argument("--runtime-attestation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    marker = run_analysis(
        primary_manifest_path=args.primary_manifest,
        extension_manifest_path=args.extension_manifest,
        extension_reference_manifest_path=args.extension_reference_manifest,
        evaluation_receipt_path=args.evaluation_receipt,
        runtime_attestation_path=args.runtime_attestation,
        output_dir=args.output_dir,
        expected_source_sha256=args.expected_source_sha256,
    )
    print(
        json.dumps(
            {
                "state": marker["state"],
                "artifact_names": sorted(marker["artifacts"]),
                "artifact_hashes": {
                    name: row["sha256"] for name, row in marker["artifacts"].items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ExtremeCubeAnalysisError",
    "analyze_collection",
    "collect_cube",
    "contrast_coefficients",
    "exact_two_sided_sign_flip",
    "holm_adjust",
    "render_figure",
    "run_analysis",
]
