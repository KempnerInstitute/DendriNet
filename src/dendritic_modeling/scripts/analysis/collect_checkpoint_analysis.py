"""Collect generic registered-analyzer outputs from frozen checkpoints.

This collector is deliberately schema-agnostic. It validates the checkpoint-run
provenance, preserves each complete result dictionary in JSONL, and emits a long
table of finite numeric leaves. It performs no across-checkpoint inference and
never treats list elements, branches, somas, or technical draws as replicates.
Analyzers with dedicated technical-draw collectors should continue to use them.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dendritic_modeling.analysis.registry import get_registered_analyzers
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity

SCHEMA_VERSION = "checkpoint_analysis_collection_v1"
RUN_SCHEMA_VERSION = "checkpoint_analysis_run_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path, *, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jobs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    required = {"output_dir", "config", "checkpoint", "analysis_config"}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict) or not required.issubset(payload):
                    raise ValueError(f"invalid job declaration at {path}:{line_number}")
                jobs.append(payload)
    outputs = [str(Path(str(job["output_dir"])).resolve()) for job in jobs]
    if not jobs or len(outputs) != len(set(outputs)):
        raise ValueError("jobs must be nonempty with unique output directories")
    return jobs


def _validate_provenance(
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
    hash_cache: dict[Path, str],
) -> None:
    if provenance.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unexpected checkpoint-analysis provenance schema")
    if provenance.get("training_performed") is not False:
        raise ValueError("checkpoint analysis unexpectedly performed training")
    if provenance.get("strict_checkpoint_loading") is not True:
        raise ValueError("checkpoint loading was not strict")
    if provenance.get("missing_keys") or provenance.get("unexpected_keys"):
        raise ValueError("checkpoint incompatibilities were recorded")
    if (
        Path(str(provenance.get("output_dir", ""))).resolve()
        != Path(str(job["output_dir"])).resolve()
    ):
        raise ValueError("output directory differs between job and provenance")

    for field in ("config", "checkpoint", "analysis_config"):
        path = Path(str(job[field])).resolve()
        recorded = Path(str(provenance.get(f"{field}_path", ""))).resolve()
        if path != recorded or not path.is_file():
            raise ValueError(f"{field} path differs between job and provenance")
        digest = hash_cache.get(path)
        if digest is None:
            digest = _sha256(path)
            hash_cache[path] = digest
        if digest != provenance.get(f"{field}_sha256"):
            raise ValueError(f"{field} hash differs from provenance")


def _json_path(parent: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    return f"{parent}.{key}" if parent else key


def _numeric_leaves(value: Any, *, path: str = "") -> list[tuple[str, float]]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite numeric result at {path}")
        return [(path, number)]
    if isinstance(value, Mapping):
        leaves: list[tuple[str, float]] = []
        for key in sorted(value, key=str):
            leaves.extend(_numeric_leaves(value[key], path=_json_path(path, str(key))))
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(_numeric_leaves(item, path=_json_path(path, index)))
        return leaves
    return []


def _csv_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _job_fields(job_index: int, job: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"job_index": int(job_index)}
    row.update({f"job_{key}": _csv_cell(value) for key, value in sorted(job.items())})
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if "job_index" in fields:
        fields.remove("job_index")
        fields.insert(0, "job_index")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["job_index"])
        writer.writeheader()
        writer.writerows(rows)


def _result_path(run_dir: Path, save_subdir: str) -> Path:
    """Resolve supported final-result layouts without guessing further."""
    candidates = [
        run_dir / save_subdir / "final.json",
        run_dir / save_subdir / "final",
        run_dir / save_subdir / "epochs" / "final.json",
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        if not existing:
            raise FileNotFoundError(candidates[0])
        raise ValueError(f"ambiguous analyzer result files: {existing}")
    return existing[0]


def collect_checkpoint_analysis(
    *,
    jobs_paths: Sequence[Path],
    output_dir: Path,
    required_analyzers: Sequence[str] = (),
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Validate and collect generic checkpoint-level analyzer results."""
    jobs = _read_jobs(jobs_paths)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(f"declared {len(jobs)} jobs; expected {expected_count}")

    registry = get_registered_analyzers()
    required = tuple(dict.fromkeys(str(name) for name in required_analyzers))
    unknown = sorted(set(required) - set(registry))
    if unknown:
        raise ValueError(f"unknown required analyzers: {unknown}")

    output_dir.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    file_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    jsonl_records: list[dict[str, Any]] = []

    for job_index, job in enumerate(jobs):
        run_dir = Path(str(job["output_dir"])).resolve()
        provenance_path = run_dir / "checkpoint_analysis_provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(provenance_path)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        _validate_provenance(job, provenance, hash_cache)

        enabled = tuple(str(name) for name in provenance.get("enabled_analyzers", []))
        missing_enabled = sorted(set(required) - set(enabled))
        if missing_enabled:
            raise ValueError(
                f"required analyzers absent from run provenance: {missing_enabled}"
            )
        selected = required or tuple(name for name in enabled if name in registry)
        if not selected:
            raise ValueError("run provenance contains no registered analyzers")

        for analyzer_name in selected:
            spec = registry[analyzer_name]
            result_path = _result_path(run_dir, spec.save_subdir)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ValueError(f"analyzer result is not a mapping: {result_path}")
            digest = _sha256(result_path)
            base = _job_fields(job_index, job)
            file_rows.append(
                {
                    **base,
                    "analyzer": analyzer_name,
                    "result_path": str(result_path),
                    "result_sha256": digest,
                    "numeric_leaf_count": len(_numeric_leaves(result)),
                    "provenance_path": str(provenance_path),
                    "provenance_sha256": _sha256(provenance_path),
                }
            )
            for metric_path, metric_value in _numeric_leaves(result):
                scalar_rows.append(
                    {
                        **base,
                        "analyzer": analyzer_name,
                        "metric_path": metric_path,
                        "value": metric_value,
                        "value_role": "raw_saved_numeric_leaf",
                        "independent_replicate": False,
                    }
                )
            jsonl_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "job_index": job_index,
                    "job": job,
                    "analyzer": analyzer_name,
                    "result_path": str(result_path),
                    "result_sha256": digest,
                    "result": result,
                }
            )

    _write_csv(output_dir / "checkpoint_analysis_files.csv", file_rows)
    _write_csv(output_dir / "checkpoint_scalar_metrics.csv", scalar_rows)
    (output_dir / "checkpoint_analysis_records.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in jsonl_records),
        encoding="utf-8",
    )
    release = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_paths": [str(path.resolve()) for path in jobs_paths],
        "jobs_sha256": [_sha256(path.resolve()) for path in jobs_paths],
        "collected_job_count": len(jobs),
        "collected_result_count": len(jsonl_records),
        "required_analyzers": list(required),
        "aggregation_policy": (
            "none; values are raw numeric leaves and are not independent replicates"
        ),
        "dedicated_collectors_required_for_technical_draw_inference": True,
        "collector_code_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
        "outputs": {
            name: _sha256(output_dir / name)
            for name in (
                "checkpoint_analysis_files.csv",
                "checkpoint_scalar_metrics.csv",
                "checkpoint_analysis_records.jsonl",
            )
        },
    }
    (output_dir / "collection_provenance.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--require-analyzer", action="append", default=[])
    parser.add_argument("--expected-count", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    return collect_checkpoint_analysis(
        jobs_paths=args.jobs,
        output_dir=args.output_dir,
        required_analyzers=args.require_analyzer,
        expected_count=args.expected_count,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
