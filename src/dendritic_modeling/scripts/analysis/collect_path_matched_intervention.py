"""Validate and collect frozen-checkpoint path-matched interventions.

The collector consumes the JSONL declarations used by
``dendritic-analyze-checkpoint``.  It binds every result to the declared
configuration, checkpoint, and analysis profile; verifies the fixed-count
support and record grid; keeps individual support draws in an audit table;
and averages technical support draws within each trained checkpoint for
downstream inference.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity

SCHEMA_VERSION = "path_matched_intervention_collection_v1"
RESULT_SUBDIRECTORY = "path_matched_intervention"
RESULT_ANALYSIS_TYPE = "path_matched_depth_intervention"
METRIC_FIELDS = (
    "baseline_accuracy",
    "accuracy",
    "accuracy_drop",
    "chance_accuracy",
    "normalized_above_chance_accuracy_loss",
    "baseline_mean_cross_entropy",
    "mean_cross_entropy",
    "mean_cross_entropy_increase",
)
IDENTITY_FIELDS = (
    "soma_relative_depth",
    "target",
    "method",
)


def _sha256_file(path: Path, *, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _csv_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({str(key) for row in rows for key in row})
    if "job_index" in fieldnames:
        fieldnames.remove("job_index")
        fieldnames.insert(0, "job_index")
    if not fieldnames:
        fieldnames = ["job_index"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_jobs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{path}:{line_number} must contain an object")
                required = {"output_dir", "config", "checkpoint", "analysis_config"}
                missing = sorted(required - set(payload))
                if missing:
                    raise ValueError(
                        f"{path}:{line_number} lacks required fields: {missing}"
                    )
                jobs.append(payload)
    output_dirs = [str(Path(str(job["output_dir"])).resolve()) for job in jobs]
    if not jobs:
        raise ValueError("No checkpoint-analysis jobs were declared")
    if len(output_dirs) != len(set(output_dirs)):
        raise ValueError("Job declarations contain duplicate output directories")
    return jobs


def _base_row(job_index: int, job: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"job_index": int(job_index)}
    for key, value in sorted(job.items()):
        row[f"job_{key}"] = _csv_cell(value)
    return row


def _validate_path_and_hash(
    *,
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
    field: str,
    hash_cache: dict[Path, str],
) -> None:
    job_path = Path(str(job[field])).resolve()
    provenance_path = Path(str(provenance.get(f"{field}_path", ""))).resolve()
    if job_path != provenance_path:
        raise ValueError(f"{field} path differs between job and provenance")
    if not job_path.is_file():
        raise FileNotFoundError(job_path)
    digest = hash_cache.get(job_path)
    if digest is None:
        digest = _sha256_file(job_path)
        hash_cache[job_path] = digest
    if digest != provenance.get(f"{field}_sha256"):
        raise ValueError(f"{field} hash differs from checkpoint-run provenance")


def _validate_run_provenance(
    *,
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
    hash_cache: dict[Path, str],
) -> None:
    if provenance.get("training_performed") is not False:
        raise ValueError("Checkpoint analysis unexpectedly performed training")
    if provenance.get("strict_checkpoint_loading") is not True:
        raise ValueError("Checkpoint loading was not strict")
    if provenance.get("missing_keys") or provenance.get("unexpected_keys"):
        raise ValueError("Checkpoint incompatibilities were recorded")
    if (
        Path(str(provenance.get("output_dir", ""))).resolve()
        != Path(str(job["output_dir"])).resolve()
    ):
        raise ValueError("Output directory differs between job and provenance")
    if "path_matched_intervention" not in {
        str(value) for value in provenance.get("enabled_analyzers", [])
    }:
        raise ValueError("Path-matched analyzer is absent from run provenance")
    for field in ("config", "checkpoint", "analysis_config"):
        _validate_path_and_hash(
            job=job,
            provenance=provenance,
            field=field,
            hash_cache=hash_cache,
        )


def _finite(value: Any, *, context: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"Non-finite value in {context}: {value!r}")
    return converted


def _validate_result(
    result: Mapping[str, Any],
    *,
    expected_support_hash: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if result.get("schema_version") != 1:
        raise ValueError("Unexpected path-matched result schema")
    if result.get("analysis_type") != RESULT_ANALYSIS_TYPE:
        raise ValueError("Unexpected path-matched analysis type")
    if result.get("reference_split") != "training":
        raise ValueError("Fixed references were not calibrated on training data")
    if result.get("evaluation_split") != "test":
        raise ValueError("Interventions were not evaluated on held-out test data")

    tree = result.get("tree")
    support = result.get("support_plan")
    shuffle = result.get("shuffle")
    if not isinstance(tree, dict) or not isinstance(support, dict):
        raise ValueError("Result lacks tree or support-plan metadata")
    if not isinstance(shuffle, dict):
        raise ValueError("Result lacks shuffle metadata")
    if support.get("technical_draws_are_model_replicates") is not False:
        raise ValueError("Support draws are incorrectly marked as model replicates")
    selected_support_hash = str(support.get("selected_index_supports_sha256", ""))
    if not selected_support_hash:
        raise ValueError("Result lacks selected-index support hash")
    if expected_support_hash and selected_support_hash != expected_support_hash:
        raise ValueError("Selected-index support differs from the declared contract")

    support_depths = sorted(int(value) for value in support.get("depths", {}))
    draws = int(support.get("draw_count", 0))
    n_somas = int(support.get("n_somas", 0))
    if not support_depths or draws < 1 or n_somas < 1:
        raise ValueError("Invalid support-plan dimensions")
    if int(tree.get("n_somas", -1)) != n_somas:
        raise ValueError("Tree and support-plan soma counts differ")
    if list(tree.get("branch_factors", [])) != list(support.get("branch_factors", [])):
        raise ValueError("Tree and support-plan morphologies differ")
    if int(support.get("coordinates_per_depth_per_draw", -1)) != n_somas:
        raise ValueError("Support-plan coordinate count is not fixed to soma count")

    support_entries: dict[tuple[int, int], Mapping[str, Any]] = {}
    for depth in support_depths:
        entries = support["depths"].get(str(depth))
        if not isinstance(entries, list) or len(entries) != draws:
            raise ValueError(f"Incomplete support entries at depth {depth}")
        for entry in entries:
            draw = int(entry.get("support_draw", -1))
            indices = entry.get("selected_indices", [])
            if draw < 0 or draw >= draws or len(indices) != n_somas:
                raise ValueError(f"Invalid support draw at depth {depth}")
            if len({int(value) for value in indices}) != n_somas:
                raise ValueError(f"Duplicate support coordinates at depth {depth}")
            support_entries[(depth, draw)] = entry
    if len(support_entries) != len(support_depths) * draws:
        raise ValueError("Support depth/draw identities are duplicated")

    raw_records = result.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("Result contains no intervention records")
    depths = sorted(
        {int(record.get("soma_relative_depth", -1)) for record in raw_records}
    )
    if not depths or not set(depths).issubset(support_depths):
        raise ValueError("Intervention records refer to undeclared support depths")
    targets = sorted({str(record.get("target")) for record in raw_records})
    methods = sorted({str(record.get("method")) for record in raw_records})
    expected_grid = {
        (depth, target, method, draw)
        for depth in depths
        for target in targets
        for method in methods
        for draw in range(draws)
    }
    observed_grid: set[tuple[int, str, str, int]] = set()
    records: list[dict[str, Any]] = []
    baseline = result.get("baseline", {})
    for record_number, raw in enumerate(raw_records):
        record = dict(raw)
        identity = (
            int(record.get("soma_relative_depth", -1)),
            str(record.get("target")),
            str(record.get("method")),
            int(record.get("support_draw", -1)),
        )
        if identity in observed_grid:
            raise ValueError(f"Duplicate intervention identity: {identity}")
        observed_grid.add(identity)
        depth, _target, _method, draw = identity
        support_entry = support_entries.get((depth, draw))
        if support_entry is None:
            raise ValueError(f"Record refers to undeclared support: {identity}")
        if record.get("technical_draw_is_model_replicate") is not False:
            raise ValueError("An intervention draw is marked as a model replicate")
        if record.get("intervention_scope") != (
            "path_matched_fixed_count_compartment_coordinates"
        ):
            raise ValueError("Record has the wrong intervention scope")
        if int(record.get("selected_compartment_coordinate_count", -1)) != n_somas:
            raise ValueError("Intervention coordinate count changes with depth")
        if str(record.get("selected_indices_sha256", "")) != str(
            support_entry.get("selected_indices_sha256", "")
        ):
            raise ValueError("Record and support selected-index hashes differ")
        signal_count = len(record.get("target_signals", []))
        if (
            signal_count < 1
            or int(record.get("replaced_tensor_coordinate_count", -1))
            != n_somas * signal_count
        ):
            raise ValueError("Replaced tensor-coordinate count is inconsistent")
        if int(record.get("producer_stream_count", -1)) != signal_count:
            raise ValueError("Producer-stream count is inconsistent")
        for field in METRIC_FIELDS:
            record[field] = _finite(
                record.get(field), context=f"record {record_number} {field}"
            )
        if not math.isclose(
            record["baseline_accuracy"],
            _finite(baseline.get("accuracy"), context="baseline accuracy"),
            abs_tol=1e-12,
        ):
            raise ValueError("Record baseline accuracy differs from run baseline")
        if not math.isclose(
            record["baseline_mean_cross_entropy"],
            _finite(
                baseline.get("mean_cross_entropy"), context="baseline cross entropy"
            ),
            abs_tol=1e-12,
        ):
            raise ValueError("Record baseline cross entropy differs from run baseline")
        records.append(record)
    if observed_grid != expected_grid:
        raise ValueError("Intervention depth/target/method/draw grid is incomplete")

    run_metadata = {
        "support_plan_sha256": str(support.get("support_plan_sha256", "")),
        "selected_index_supports_sha256": selected_support_hash,
        "shuffle_permutation_sha256": str(shuffle.get("permutation_sha256", "")),
        "support_draw_count": draws,
        "n_somas": n_somas,
        "branch_factors": list(support.get("branch_factors", [])),
        "depths": depths,
        "support_plan_depths": support_depths,
        "targets": targets,
        "methods": methods,
        "reference_sample_count": int(result.get("reference_sample_count", -1)),
        "evaluation_sample_count": int(baseline.get("sample_count", -1)),
    }
    if (
        not run_metadata["support_plan_sha256"]
        or not run_metadata["shuffle_permutation_sha256"]
    ):
        raise ValueError("Result lacks support or shuffle provenance hashes")
    if (
        min(
            run_metadata["reference_sample_count"],
            run_metadata["evaluation_sample_count"],
        )
        < 1
    ):
        raise ValueError("Result contains invalid dataset sample counts")
    return records, run_metadata


def _technical_row(
    job_index: int,
    job: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    row = _base_row(job_index, job)
    row.update({str(key): _csv_cell(value) for key, value in record.items()})
    return row


def _checkpoint_rows(
    *,
    job_index: int,
    job: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in IDENTITY_FIELDS)].append(record)
    rows: list[dict[str, Any]] = []
    for identity, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        row = _base_row(job_index, job)
        row.update(dict(zip(IDENTITY_FIELDS, identity, strict=True)))
        row["technical_support_draw_count"] = len(group)
        row["technical_draws_are_model_replicates"] = False
        stable_fields = (
            "target_signals",
            "producer_stream_count",
            "selected_compartment_coordinate_count",
            "replaced_tensor_coordinate_count",
            "available_compartment_coordinate_count",
            "intervention_scope",
        )
        for field in stable_fields:
            values = {_csv_cell(record.get(field)) for record in group}
            if len(values) != 1:
                raise ValueError(f"{field} changes across technical support draws")
            row[field] = values.pop()
        for field in METRIC_FIELDS:
            values = np.asarray([float(record[field]) for record in group], dtype=float)
            row[field] = float(np.mean(values))
            row[f"{field}_technical_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
        rows.append(row)
    return rows


def collect_path_matched_interventions(
    *,
    jobs_paths: Sequence[str | Path],
    output_dir: str | Path,
    expected_count: int | None = None,
    expected_support_hash: str | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate declared runs and write audit plus checkpoint-level tables."""

    resolved_jobs_paths = [Path(path).resolve() for path in jobs_paths]
    jobs = _read_jobs(resolved_jobs_paths)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(
            f"Declared job count {len(jobs)} does not match expected {expected_count}"
        )

    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    skipped: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_summary: list[dict[str, Any]] = []
    run_contracts: list[dict[str, Any]] = []

    for job_index, job in enumerate(jobs):
        output = Path(str(job["output_dir"])).resolve()
        filename = str(job.get("filename", "final"))
        result_path = output / RESULT_SUBDIRECTORY / f"{filename}.json"
        provenance_path = output / "checkpoint_analysis_provenance.json"
        missing = [
            str(path) for path in (result_path, provenance_path) if not path.is_file()
        ]
        if missing:
            if not allow_incomplete:
                raise FileNotFoundError(missing[0])
            skipped.append(
                {"job_index": job_index, "output_dir": str(output), "missing": missing}
            )
            continue

        result = json.loads(result_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        _validate_run_provenance(job=job, provenance=provenance, hash_cache=hash_cache)
        records, contract = _validate_result(
            result, expected_support_hash=expected_support_hash
        )
        run_contracts.append(contract)
        support_rows.extend(
            _technical_row(job_index, job, record) for record in records
        )
        averaged = _checkpoint_rows(job_index=job_index, job=job, records=records)
        checkpoint_rows.extend(averaged)

        summary = _base_row(job_index, job)
        summary.update({key: _csv_cell(value) for key, value in contract.items()})
        summary.update(
            {
                "config_sha256": provenance["config_sha256"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "analysis_config_sha256": provenance["analysis_config_sha256"],
                "strict_checkpoint_loading": True,
                "technical_record_count": len(records),
                "checkpoint_record_count": len(averaged),
                "run_code_identity": _csv_cell(provenance.get("code_identity")),
            }
        )
        checkpoint_summary.append(summary)
        source_artifacts.append(
            {
                "job_index": job_index,
                "result_path": str(result_path),
                "result_sha256": _sha256_file(result_path),
                "provenance_path": str(provenance_path),
                "provenance_sha256": _sha256_file(provenance_path),
            }
        )

    if expected_count is not None and not allow_incomplete:
        if len(checkpoint_summary) != int(expected_count):
            raise ValueError(
                f"Collected {len(checkpoint_summary)} runs; expected {expected_count}"
            )
    if checkpoint_summary:
        contract_fields = (
            "support_plan_sha256",
            "selected_index_supports_sha256",
            "shuffle_permutation_sha256",
            "support_draw_count",
            "n_somas",
            "branch_factors",
            "depths",
            "support_plan_depths",
            "targets",
            "methods",
        )
        for field in contract_fields:
            values = {_csv_cell(contract[field]) for contract in run_contracts}
            if len(values) != 1:
                raise ValueError(f"Run contract differs across checkpoints: {field}")

    tables = {
        "checkpoint_summary": checkpoint_summary,
        "support_draw_records": support_rows,
        "checkpoint_records": checkpoint_rows,
    }
    output_artifacts: list[dict[str, Any]] = []
    for name, rows in tables.items():
        path = output_root / f"{name}.csv"
        _write_csv(path, rows)
        output_artifacts.append(
            {"path": str(path), "sha256": _sha256_file(path), "row_count": len(rows)}
        )

    collection = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_files": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in resolved_jobs_paths
        ],
        "declared_job_count": len(jobs),
        "expected_job_count": expected_count,
        "collected_job_count": len(checkpoint_summary),
        "allow_incomplete": bool(allow_incomplete),
        "expected_support_hash": expected_support_hash,
        "skipped_jobs": skipped,
        "source_artifacts": source_artifacts,
        "output_artifacts": output_artifacts,
        "collector_code_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
    }
    provenance_path = output_root / "collection_provenance.json"
    provenance_path.write_text(
        json.dumps(collection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return collection


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        action="append",
        required=True,
        help="JSONL job declaration; repeat for pilot and production files",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-support-hash")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Collect completed declarations while recording missing outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    return collect_path_matched_interventions(
        jobs_paths=args.jobs,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
        expected_support_hash=args.expected_support_hash,
        allow_incomplete=args.allow_incomplete,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
