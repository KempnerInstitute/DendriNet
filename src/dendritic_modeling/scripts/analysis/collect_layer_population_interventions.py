"""Validate and collect frozen local-I population intervention analyses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity

SCHEMA_VERSION = "local_i_population_intervention_collection_v1"
RESULT_SUBDIRECTORY = "local_inhibitory_population_intervention"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _read_jobs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    jobs = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("output_dir"):
                raise ValueError(f"invalid job declaration at {path}:{line_number}")
            jobs.append(value)
    outputs = [str(Path(job["output_dir"]).resolve()) for job in jobs]
    if len(outputs) != len(set(outputs)):
        raise ValueError("job declarations contain duplicate output directories")
    return jobs


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if "job_index" in fields:
        fields.remove("job_index")
        fields.insert(0, "job_index")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["job_index"])
        writer.writeheader()
        writer.writerows(rows)


def _job_row(index: int, job: Mapping[str, Any]) -> dict[str, Any]:
    row = {"job_index": int(index)}
    row.update({f"job_{key}": _csv_cell(value) for key, value in sorted(job.items())})
    return row


def _validate_provenance(
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
    hash_cache: dict[Path, str],
) -> None:
    if provenance.get("training_performed") is not False:
        raise ValueError("checkpoint analysis unexpectedly trained")
    if provenance.get("strict_checkpoint_loading") is not True:
        raise ValueError("checkpoint loading was not strict")
    if provenance.get("missing_keys") or provenance.get("unexpected_keys"):
        raise ValueError("checkpoint incompatibilities were recorded")
    if provenance.get("schema_version") != "checkpoint_analysis_run_v1":
        raise ValueError("unexpected checkpoint-analysis provenance schema")
    enabled = provenance.get("enabled_analyzers")
    if not isinstance(enabled, list) or (
        "local_inhibitory_population_intervention" not in enabled
    ):
        raise ValueError("local-I analyzer is absent from run provenance")
    if (
        Path(str(provenance.get("output_dir"))).resolve()
        != Path(str(job["output_dir"])).resolve()
    ):
        raise ValueError("output directory differs between job and provenance")
    for field in ("config", "checkpoint", "analysis_config"):
        job_path = Path(str(job[field])).resolve()
        recorded_path = Path(str(provenance[f"{field}_path"])).resolve()
        if job_path != recorded_path or not job_path.is_file():
            raise ValueError(f"{field} path differs between job and provenance")
        digest = hash_cache.setdefault(job_path, _sha256(job_path))
        if digest != provenance.get(f"{field}_sha256"):
            raise ValueError(f"{field} hash differs from provenance")


def _validate_result(result: Mapping[str, Any], path: Path) -> None:
    if result.get("analysis_type") != "local_inhibitory_population_intervention":
        raise ValueError(f"unexpected analysis type in {path}")
    if result.get("schema_version") != 1:
        raise ValueError(f"unexpected local-I result schema in {path}")
    records = result.get("records")
    paired = result.get("paired_global_minus_within_class_records")
    if not isinstance(records, list) or not records:
        raise ValueError(f"missing intervention records in {path}")
    if not isinstance(paired, list):
        raise ValueError(f"missing paired donor controls in {path}")
    layers = sorted({int(record["network_layer_number"]) for record in records})
    if layers != list(range(1, max(layers) + 1)):
        raise ValueError(f"non-contiguous network positions in {path}")
    draws = int(result["permutation_draws"])
    methods = {str(record["method"]) for record in records}
    required = {"global_test_shuffle", "within_class_test_shuffle", "zero"}
    if methods != required:
        raise ValueError(f"local-I method set changed in {path}: {methods}")
    expected_records = len(layers) * (2 * draws + 1)
    if len(records) != expected_records or len(paired) != len(layers) * draws:
        raise ValueError(f"local-I record factorial is incomplete in {path}")
    identities = [
        (
            int(record["network_layer_number"]),
            str(record["source_population"]),
            str(record["target_population"]),
            str(record["method"]),
            record.get("permutation_draw"),
        )
        for record in records
    ]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate local-I record identities in {path}")
    expected_identities = {
        (layer, method, draw)
        for layer in layers
        for method in ("global_test_shuffle", "within_class_test_shuffle")
        for draw in range(draws)
    } | {(layer, "zero", None) for layer in layers}
    observed_identities = {
        (layer, method, draw) for layer, _source, _target, method, draw in identities
    }
    if observed_identities != expected_identities:
        raise ValueError(f"local-I layer/method/draw grid changed in {path}")
    paired_identities = [
        (int(record["network_layer_number"]), int(record["permutation_draw"]))
        for record in paired
    ]
    if len(paired_identities) != len(set(paired_identities)) or set(
        paired_identities
    ) != {(layer, draw) for layer in layers for draw in range(draws)}:
        raise ValueError(f"paired local-I donor grid changed in {path}")
    if any(
        record.get("route") != "same_step_ff_inhibitory_local_I_to_E"
        for record in records
    ):
        raise ValueError(f"local-I route semantics changed in {path}")
    baseline = result.get("baseline", {})
    baseline_metrics = (
        baseline.get("accuracy"),
        baseline.get("mean_cross_entropy"),
        baseline.get("sample_count"),
    )
    if not all(
        value is not None and np.isfinite(float(value)) for value in baseline_metrics
    ):
        raise ValueError(f"non-finite local-I baseline metrics in {path}")
    raw_metrics = (
        "baseline_accuracy",
        "accuracy",
        "accuracy_drop",
        "baseline_mean_cross_entropy",
        "mean_cross_entropy",
        "mean_cross_entropy_increase",
        "normalized_above_chance_accuracy_loss",
    )
    if any(
        record.get(metric) is None or not np.isfinite(float(record[metric]))
        for record in records
        for metric in raw_metrics
    ):
        raise ValueError(f"non-finite local-I intervention metric in {path}")
    paired_metrics = (
        "class_aligned_mean_cross_entropy_excess",
        "class_aligned_accuracy_drop_excess",
    )
    if any(
        record.get(metric) is None or not np.isfinite(float(record[metric]))
        for record in paired
        for metric in paired_metrics
    ):
        raise ValueError(f"non-finite paired local-I metric in {path}")


def _technical_means(
    job_index: int,
    job: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    paired: bool,
) -> list[dict[str, Any]]:
    group_fields = ["network_layer_index", "network_layer_number"]
    if not paired:
        group_fields.append("method")
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[tuple(record[field] for field in group_fields)].append(record)
    metric_fields = (
        (
            "class_aligned_mean_cross_entropy_excess",
            "class_aligned_accuracy_drop_excess",
        )
        if paired
        else (
            "accuracy_drop",
            "mean_cross_entropy_increase",
            "normalized_above_chance_accuracy_loss",
        )
    )
    rows = []
    for key, values in sorted(groups.items()):
        row = _job_row(job_index, job)
        row.update(dict(zip(group_fields, key, strict=True)))
        row["technical_record_count"] = len(values)
        for metric in metric_fields:
            numeric = [
                float(value[metric])
                for value in values
                if value.get(metric) is not None
            ]
            row[f"{metric}_mean"] = float(np.mean(numeric)) if numeric else None
            row[f"{metric}_sd"] = (
                float(np.std(numeric, ddof=1)) if len(numeric) > 1 else 0.0
            )
        rows.append(row)
    return rows


def collect(
    *,
    jobs_paths: Sequence[str | Path],
    output_dir: str | Path,
    expected_count: int,
) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in jobs_paths]
    jobs = _read_jobs(paths)
    if len(jobs) != int(expected_count):
        raise ValueError(f"expected {expected_count} jobs, found {len(jobs)}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    tables: dict[str, list[dict[str, Any]]] = {
        "checkpoint_summary": [],
        "intervention_records": [],
        "intervention_technical_summary": [],
        "paired_donor_contrasts": [],
        "paired_donor_technical_summary": [],
    }
    artifacts = []
    for job_index, job in enumerate(jobs):
        result_path = (
            Path(str(job["output_dir"]))
            / RESULT_SUBDIRECTORY
            / f"{job.get('filename', 'final')}.json"
        ).resolve()
        provenance_path = (
            Path(str(job["output_dir"])) / "checkpoint_analysis_provenance.json"
        ).resolve()
        if not result_path.is_file() or not provenance_path.is_file():
            raise FileNotFoundError(
                result_path if not result_path.is_file() else provenance_path
            )
        result = json.loads(result_path.read_text())
        provenance = json.loads(provenance_path.read_text())
        _validate_provenance(job, provenance, hash_cache)
        _validate_result(result, result_path)
        base = _job_row(job_index, job)
        base.update(
            {
                "config_sha256": provenance["config_sha256"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "analysis_config_sha256": provenance["analysis_config_sha256"],
                "baseline_accuracy": result["baseline"]["accuracy"],
                "baseline_mean_cross_entropy": result["baseline"]["mean_cross_entropy"],
                "sample_count": result["baseline"]["sample_count"],
                "network_layer_count": len(
                    {record["network_layer_number"] for record in result["records"]}
                ),
            }
        )
        tables["checkpoint_summary"].append(base)
        raw_records = result["records"]
        paired_records = result["paired_global_minus_within_class_records"]
        for name, records in (
            ("intervention_records", raw_records),
            ("paired_donor_contrasts", paired_records),
        ):
            for record in records:
                row = _job_row(job_index, job)
                row.update(
                    {str(key): _csv_cell(value) for key, value in record.items()}
                )
                tables[name].append(row)
        tables["intervention_technical_summary"].extend(
            _technical_means(job_index, job, raw_records, paired=False)
        )
        tables["paired_donor_technical_summary"].extend(
            _technical_means(job_index, job, paired_records, paired=True)
        )
        artifacts.append(
            {
                "result_path": str(result_path),
                "result_sha256": _sha256(result_path),
                "provenance_path": str(provenance_path),
                "provenance_sha256": _sha256(provenance_path),
            }
        )
    output_tables = {}
    for name, rows in tables.items():
        path = output / f"{name}.csv"
        _write_csv(path, rows)
        output_tables[name] = {
            "path": str(path),
            "sha256": _sha256(path),
            "row_count": len(rows),
        }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_files": [str(path) for path in paths],
        "jobs_file_sha256": {str(path): _sha256(path) for path in paths},
        "checkpoint_count": len(jobs),
        "source_artifacts": artifacts,
        "output_tables": output_tables,
        "collector_source_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
        "inference_unit": "independently trained checkpoint",
        "technical_draws_are_model_replicates": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args(argv)
    manifest = collect(
        jobs_paths=args.jobs,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
