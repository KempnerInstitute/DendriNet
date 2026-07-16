"""Validate and collect frozen-checkpoint representation analyses.

The collector consumes the JSONL job declarations used to run
``dendritic-analyze-checkpoint``. It verifies every declared result against its
run provenance, keeps technical projection/probe repeats within checkpoints,
and writes stable CSV tables for downstream statistics and plotting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity

SCHEMA_VERSION = "representation_accessibility_collection_v1"
RESULT_SUBDIRECTORY = "representation_accessibility"


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


def _base_row(job_index: int, job: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"job_index": int(job_index)}
    for key, value in sorted(job.items()):
        row[f"job_{key}"] = _csv_cell(value)
    return row


def _record_row(
    job_index: int,
    job: Mapping[str, Any],
    record: Mapping[str, Any],
) -> dict[str, Any]:
    row = _base_row(job_index, job)
    for key, value in record.items():
        row[str(key)] = _csv_cell(value)
    return row


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
                if not payload.get("output_dir"):
                    raise ValueError(f"{path}:{line_number} lacks output_dir")
                jobs.append(payload)
    output_dirs = [str(Path(job["output_dir"]).resolve()) for job in jobs]
    if len(output_dirs) != len(set(output_dirs)):
        raise ValueError("Job declarations contain duplicate output directories")
    return jobs


def _validate_path_and_hash(
    *,
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
    field: str,
    hash_cache: dict[Path, str],
) -> None:
    job_path = Path(str(job[field])).resolve()
    provenance_path = Path(str(provenance[f"{field}_path"])).resolve()
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


def _validate_run(
    *,
    job: Mapping[str, Any],
    final_path: Path,
    provenance_path: Path,
    hash_cache: dict[Path, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(final_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if result.get("analysis_type") != "representation_accessibility":
        raise ValueError(f"Unexpected analysis type in {final_path}")
    if provenance.get("training_performed") is not False:
        raise ValueError(
            f"Checkpoint analysis unexpectedly trained in {provenance_path}"
        )
    if provenance.get("strict_checkpoint_loading") is not True:
        raise ValueError(f"Non-strict checkpoint loading in {provenance_path}")
    if provenance.get("missing_keys") or provenance.get("unexpected_keys"):
        raise ValueError(f"Checkpoint incompatibilities recorded in {provenance_path}")
    if (
        Path(str(provenance.get("output_dir"))).resolve()
        != Path(str(job["output_dir"])).resolve()
    ):
        raise ValueError(f"Output directory differs in {provenance_path}")
    for field in ("config", "checkpoint", "analysis_config"):
        if field not in job:
            raise ValueError(f"Job declaration lacks {field}")
        _validate_path_and_hash(
            job=job,
            provenance=provenance,
            field=field,
            hash_cache=hash_cache,
        )
    return result, provenance


def _metric_summary(values: Iterable[float], prefix: str) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=float)
    if array.size == 0:
        return {}
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
    }


def _technical_summary(
    *,
    job_index: int,
    job: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    grouping_fields: Sequence[str],
    metric_fields: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(_csv_cell(record.get(field)) for field in grouping_fields)].append(
            record
        )

    rows: list[dict[str, Any]] = []
    for group_key, group_records in sorted(
        groups.items(), key=lambda item: repr(item[0])
    ):
        row = _base_row(job_index, job)
        row.update(dict(zip(grouping_fields, group_key)))
        row["technical_record_count"] = len(group_records)
        row["projection_draw_count"] = len(
            {record.get("projection_draw") for record in group_records}
        )
        row["probe_draw_count"] = len(
            {record.get("probe_draw") for record in group_records}
        )
        for field in metric_fields:
            values = [
                record[field]
                for record in group_records
                if record.get(field) is not None
            ]
            row.update(_metric_summary(values, field))
        rows.append(row)
    return rows


def collect_representation_accessibility(
    *,
    jobs_paths: Sequence[str | Path],
    output_dir: str | Path,
    expected_count: int | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate declared runs and write analysis-ready CSV tables."""

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
    tables: dict[str, list[dict[str, Any]]] = {
        "checkpoint_summary": [],
        "inventory": [],
        "missing_representations": [],
        "probe_records": [],
        "probe_technical_summary": [],
        "joint_information_records": [],
        "joint_information_technical_summary": [],
        "per_soma_information_records": [],
        "per_soma_information_summary": [],
    }

    for job_index, job in enumerate(jobs):
        job_output = Path(str(job["output_dir"])).resolve()
        filename = str(job.get("filename", "final"))
        final_path = job_output / RESULT_SUBDIRECTORY / f"{filename}.json"
        provenance_path = job_output / "checkpoint_analysis_provenance.json"
        missing = [path for path in (final_path, provenance_path) if not path.is_file()]
        if missing:
            if not allow_incomplete:
                raise FileNotFoundError(missing[0])
            skipped.append(
                {
                    "job_index": job_index,
                    "output_dir": str(job_output),
                    "missing_paths": [str(path) for path in missing],
                }
            )
            continue

        result, provenance = _validate_run(
            job=job,
            final_path=final_path,
            provenance_path=provenance_path,
            hash_cache=hash_cache,
        )
        source_artifacts.append(
            {
                "job_index": job_index,
                "final_path": str(final_path),
                "final_sha256": _sha256_file(final_path),
                "provenance_path": str(provenance_path),
                "provenance_sha256": _sha256_file(provenance_path),
            }
        )

        checkpoint_row = _base_row(job_index, job)
        checkpoint_row.update(
            {
                "config_sha256": provenance["config_sha256"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "analysis_config_sha256": provenance["analysis_config_sha256"],
                "strict_checkpoint_loading": provenance["strict_checkpoint_loading"],
                "valid_model_accuracy": result["validation_summary"]["model_accuracy"],
                "test_model_accuracy": result["test_summary"]["model_accuracy"],
                "inventory_count": len(result.get("inventory", [])),
                "missing_representation_count": len(
                    result.get("missing_representations", [])
                ),
                "probe_record_count": len(result.get("probe_records", [])),
                "joint_information_record_count": len(
                    result.get("joint_information_records", [])
                ),
                "per_soma_information_record_count": len(
                    result.get("per_soma_information_records", [])
                ),
                "run_code_identity": _csv_cell(provenance.get("code_identity")),
            }
        )
        tables["checkpoint_summary"].append(checkpoint_row)

        for table_name in (
            "inventory",
            "missing_representations",
            "probe_records",
            "joint_information_records",
            "per_soma_information_records",
            "per_soma_information_summary",
        ):
            tables[table_name].extend(
                _record_row(job_index, job, record)
                for record in result.get(table_name, [])
            )

        tables["probe_technical_summary"].extend(
            _technical_summary(
                job_index=job_index,
                job=job,
                records=result.get("probe_records", []),
                grouping_fields=(
                    "network_layer_index",
                    "network_layer_number",
                    "representation",
                    "population_names",
                    "native_feature_count",
                    "projection_name",
                    "decoded_feature_count",
                    "probe_name",
                    "probe_type",
                    "probe_family",
                ),
                metric_fields=(
                    "class_accessibility_above_null",
                    "heldout_test_balanced_accuracy",
                    "heldout_test_label_shuffle_balanced_accuracy",
                ),
            )
        )
        tables["joint_information_technical_summary"].extend(
            _technical_summary(
                job_index=job_index,
                job=job,
                records=result.get("joint_information_records", []),
                grouping_fields=(
                    "network_layer_index",
                    "network_layer_number",
                    "representation",
                    "population_names",
                    "native_feature_count",
                    "projection_name",
                    "decoded_feature_count",
                ),
                metric_fields=(
                    "observed_mi_bits",
                    "label_shuffle_mi_mean_bits",
                    "excess_mi_over_label_shuffle_bits",
                ),
            )
        )

    if expected_count is not None and not allow_incomplete:
        collected = len(tables["checkpoint_summary"])
        if collected != int(expected_count):
            raise ValueError(f"Collected {collected} runs; expected {expected_count}")

    output_artifacts: list[dict[str, Any]] = []
    for name, rows in tables.items():
        path = output_root / f"{name}.csv"
        _write_csv(path, rows)
        output_artifacts.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "row_count": len(rows),
            }
        )

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_files": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in resolved_jobs_paths
        ],
        "declared_job_count": len(jobs),
        "expected_job_count": expected_count,
        "collected_job_count": len(tables["checkpoint_summary"]),
        "allow_incomplete": bool(allow_incomplete),
        "skipped_jobs": skipped,
        "source_artifacts": source_artifacts,
        "output_artifacts": output_artifacts,
        "collector_code_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
    }
    provenance_path = output_root / "collection_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


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
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Collect completed declarations while recording missing outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    return collect_representation_accessibility(
        jobs_paths=args.jobs,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
        allow_incomplete=args.allow_incomplete,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
