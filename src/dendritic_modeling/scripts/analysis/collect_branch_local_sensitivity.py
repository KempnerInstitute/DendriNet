"""Validate and collect frozen branch-local sensitivity analyses.

The collector binds every record to the declared checkpoint-analysis job and
its strict-loading provenance. It preserves integration mode, soma-relative
depth, and branch component as explicit columns, so summaries never depend on
JSON list ordering.
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

import yaml

from dendritic_modeling.analysis.managers.analysis_status import describe_incomplete
from dendritic_modeling.scripts.analysis.collect_checkpoint_analysis import (
    RUN_SCHEMA_VERSION,
    incomplete_recorded_analyses,
)
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity
from dendritic_modeling.utils.logging_config import setup_logging

SCHEMA_VERSION = "branch_local_sensitivity_collection_v1"
CONFIG_SCHEMA_VERSION = "branch_local_sensitivity_collection_config_v1"
RESULT_SUBDIRECTORY = "branch_local_sensitivity"
RESULT_ANALYSIS_TYPE = "branch_local_sensitivity"
METRIC_FIELDS = (
    "component_mean",
    "component_std",
    "mean_signed_doutput_dcomponent",
    "mean_abs_doutput_dcomponent",
    "rms_doutput_dcomponent",
    "mean_abs_first_order_contribution",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path, *, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if "job_index" in fields:
        fields.remove("job_index")
        fields.insert(0, "job_index")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields or ["job_index"], extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)


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


def _base_row(job_index: int, job: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"job_index": int(job_index)}
    row.update({f"job_{key}": _csv_cell(value) for key, value in sorted(job.items())})
    return row


def _validate_provenance(
    *, job: Mapping[str, Any], provenance: Mapping[str, Any], hashes: dict[Path, str]
) -> None:
    if provenance.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unexpected checkpoint-analysis provenance schema")
    if provenance.get("training_performed") is not False:
        raise ValueError("checkpoint analysis unexpectedly performed training")
    if provenance.get("strict_checkpoint_loading") is not True:
        raise ValueError("checkpoint loading was not strict")
    if provenance.get("missing_keys") or provenance.get("unexpected_keys"):
        raise ValueError("checkpoint incompatibilities were recorded")
    if RESULT_ANALYSIS_TYPE not in {
        str(value) for value in provenance.get("enabled_analyzers", [])
    }:
        raise ValueError("branch-local sensitivity analyzer is absent from provenance")
    if (
        Path(str(provenance.get("output_dir", ""))).resolve()
        != Path(str(job["output_dir"])).resolve()
    ):
        raise ValueError("output directory differs between job and provenance")
    for field in ("config", "checkpoint", "analysis_config"):
        path = Path(str(job[field])).resolve()
        if path != Path(str(provenance.get(f"{field}_path", ""))).resolve():
            raise ValueError(f"{field} path differs between job and provenance")
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashes.setdefault(path, _sha256(path))
        if digest != provenance.get(f"{field}_sha256"):
            raise ValueError(f"{field} hash differs from provenance")


def _validate_result(
    result: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if result.get("schema_version") != 1:
        raise ValueError("unexpected branch-sensitivity result schema")
    if result.get("analysis_type") != RESULT_ANALYSIS_TYPE:
        raise ValueError("unexpected branch-sensitivity analysis type")
    if result.get("evaluation_split") != "test":
        raise ValueError("branch sensitivities were not evaluated on test")
    if result.get("depth_reference") != "soma_relative":
        raise ValueError("branch sensitivities do not use soma-relative depth")
    tree = result.get("tree")
    records = result.get("records")
    if not isinstance(tree, dict) or not isinstance(records, list) or not records:
        raise ValueError("result lacks tree metadata or sensitivity records")
    n_somas = int(tree.get("n_somas", 0))
    branch_factors = [int(value) for value in tree.get("branch_factors", [])]
    module_names = tree.get("module_names_by_depth")
    if n_somas < 1 or not branch_factors or not isinstance(module_names, dict):
        raise ValueError("invalid tree metadata")
    observed_depths = sorted(int(value) for value in module_names)
    branch_depths = list(range(1, len(branch_factors) + 1))
    soma_and_branch_depths = list(range(len(branch_factors) + 1))
    if observed_depths == branch_depths:
        expected_depths = branch_depths
    elif observed_depths == soma_and_branch_depths:
        expected_depths = soma_and_branch_depths
    else:
        raise ValueError("tree module map does not cover every soma-relative depth")

    validated: list[dict[str, Any]] = []
    identities: set[tuple[int, str]] = set()
    modes: set[str] = set()
    component_sets: dict[int, set[str]] = {depth: set() for depth in expected_depths}
    for number, raw in enumerate(records):
        record = dict(raw)
        depth = int(record.get("soma_relative_depth", -1))
        component = str(record.get("component", ""))
        mode = str(record.get("integration_mode", ""))
        identity = (depth, component)
        if depth not in component_sets or not component or identity in identities:
            raise ValueError(f"invalid or duplicate sensitivity identity: {identity}")
        identities.add(identity)
        modes.add(mode)
        component_sets[depth].add(component)
        if str(record.get("module_name", "")) != str(module_names[str(depth)]):
            raise ValueError("record module differs from declared tree module")
        coordinate_count = int(record.get("coordinate_count", 0))
        sample_coordinate_count = int(record.get("sample_coordinate_count", 0))
        if coordinate_count < n_somas or sample_coordinate_count < coordinate_count:
            raise ValueError("invalid sensitivity coordinate counts")
        record["coordinate_count"] = coordinate_count
        record["sample_coordinate_count"] = sample_coordinate_count
        for field in METRIC_FIELDS:
            value = float(record.get(field))
            if not math.isfinite(value):
                raise ValueError(f"non-finite {field} in record {number}")
            record[field] = value
        validated.append(record)
    if len(modes) != 1 or not next(iter(modes)):
        raise ValueError("integration mode changes within one checkpoint result")
    reference_components = component_sets[expected_depths[0]]
    if not reference_components or any(
        components != reference_components for components in component_sets.values()
    ):
        raise ValueError("component coverage changes across tree depth")
    contract = {
        "n_somas": n_somas,
        "branch_factors": branch_factors,
        "depths": expected_depths,
        "components": sorted(reference_components),
        "integration_mode": next(iter(modes)),
        "module_name_prefix": str(result.get("module_name_prefix", "")),
        "estimand": str(result.get("estimand", "")),
    }
    return validated, contract


def collect_branch_local_sensitivities(
    *,
    jobs_paths: Sequence[str | Path],
    output_dir: str | Path,
    expected_count: int | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate declared runs and write checkpoint-level sensitivity tables."""

    resolved_jobs = [Path(path).resolve() for path in jobs_paths]
    jobs = _read_jobs(resolved_jobs)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError("declared job count does not match expected_count")
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    hashes: dict[Path, str] = {}
    records_table: list[dict[str, Any]] = []
    summary_table: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
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
            skipped.append({"job_index": job_index, "missing": missing})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        _validate_provenance(job=job, provenance=provenance, hashes=hashes)
        incomplete = incomplete_recorded_analyses(
            provenance, [RESULT_ANALYSIS_TYPE], run_dir=output
        )
        if incomplete:
            reason = describe_incomplete(incomplete)
            if not allow_incomplete:
                raise ValueError(f"the run recorded incomplete analyses: {reason}")
            skipped.append({"job_index": job_index, "incomplete": reason})
            continue
        records, contract = _validate_result(result)
        contracts.append(contract)
        for record in records:
            row = _base_row(job_index, job)
            row.update({key: _csv_cell(value) for key, value in record.items()})
            records_table.append(row)
        summary = _base_row(job_index, job)
        summary.update({key: _csv_cell(value) for key, value in contract.items()})
        summary.update(
            {
                "record_count": len(records),
                "config_sha256": provenance["config_sha256"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "analysis_config_sha256": provenance["analysis_config_sha256"],
                "strict_checkpoint_loading": True,
                "run_code_identity": _csv_cell(provenance.get("code_identity")),
            }
        )
        summary_table.append(summary)
        sources.append(
            {
                "job_index": job_index,
                "result_path": str(result_path),
                "result_sha256": _sha256(result_path),
                "provenance_path": str(provenance_path),
                "provenance_sha256": _sha256(provenance_path),
            }
        )
    if expected_count is not None and not allow_incomplete:
        if len(summary_table) != int(expected_count):
            raise ValueError("collected job count does not match expected_count")
    if contracts:
        for field in (
            "n_somas",
            "branch_factors",
            "depths",
            "components",
            "module_name_prefix",
        ):
            if len({_csv_cell(contract[field]) for contract in contracts}) != 1:
                raise ValueError(f"run contract differs across checkpoints: {field}")
    artifacts: list[dict[str, Any]] = []
    for name, rows in (
        ("checkpoint_records", records_table),
        ("checkpoint_summary", summary_table),
    ):
        path = output_root / f"{name}.csv"
        _write_csv(path, rows)
        artifacts.append(
            {"path": str(path), "sha256": _sha256(path), "row_count": len(rows)}
        )
    collection = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in resolved_jobs
        ],
        "declared_job_count": len(jobs),
        "expected_job_count": expected_count,
        "collected_job_count": len(summary_table),
        "allow_incomplete": bool(allow_incomplete),
        "skipped_jobs": skipped,
        "source_artifacts": sources,
        "output_artifacts": artifacts,
        "collector_code_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
    }
    (output_root / "collection_provenance.json").write_text(
        json.dumps(collection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return collection


def _resolve(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_collection_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported branch-sensitivity collection config: {path}")
    if not isinstance(payload.get("jobs"), list) or not payload["jobs"]:
        raise ValueError("collection config jobs must be a nonempty list")
    if "output_dir" not in payload:
        raise ValueError("collection config lacks output_dir")
    base = path.resolve().parent
    return {
        "jobs_paths": [_resolve(value, base=base) for value in payload["jobs"]],
        "output_dir": _resolve(payload["output_dir"], base=base),
        "expected_count": payload.get("expected_count"),
        "allow_incomplete": bool(payload.get("allow_incomplete", False)),
    }


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="Collection configuration file"
    )
    args = parser.parse_args(argv)
    release = collect_branch_local_sensitivities(**_load_collection_config(args.config))
    print("Collected " f"{release['collected_job_count']} branch-sensitivity jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
