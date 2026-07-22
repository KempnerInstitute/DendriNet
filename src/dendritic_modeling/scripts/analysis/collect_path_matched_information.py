"""Validate and collect frozen path-matched dendritic information analyses.

The collector binds every result to the checkpoint-analysis JSONL declaration
and strict run provenance.  It preserves anatomical support draws in an audit
table and averages those technical draws within each trained checkpoint before
downstream model-level inference.
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
import yaml

from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity

SCHEMA_VERSION = "path_matched_information_collection_v1"
CONFIG_SCHEMA_VERSION = "path_matched_information_collection_config_v1"
RESULT_SUBDIRECTORY = "path_matched_information"
RESULT_SCHEMA_VERSION = "path_matched_information_v1"
METRIC_FIELDS = (
    "observed_mi_bits",
    "label_shuffle_mi_mean_bits",
    "label_shuffle_mi_std_bits",
    "excess_mi_over_label_shuffle_bits",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path, *, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _csv_cell(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _base_row(job_index: int, job: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"job_index": int(job_index)}
    row.update({f"job_{key}": _csv_cell(value) for key, value in sorted(job.items())})
    return row


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if "job_index" in fields:
        fields.remove("job_index")
        fields.insert(0, "job_index")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields or ["job_index"],
            extrasaction="raise",
        )
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
                required = {"output_dir", "config", "checkpoint", "analysis_config"}
                if not isinstance(payload, dict) or not required.issubset(payload):
                    raise ValueError(f"invalid job declaration at {path}:{line_number}")
                jobs.append(payload)
    outputs = [str(Path(str(job["output_dir"])).resolve()) for job in jobs]
    if not jobs or len(outputs) != len(set(outputs)):
        raise ValueError("jobs must be nonempty with unique output directories")
    return jobs


def _validate_run_provenance(
    *,
    job: Mapping[str, Any],
    provenance: Mapping[str, Any],
    hash_cache: dict[Path, str],
) -> None:
    if provenance.get("training_performed") is not False:
        raise ValueError("checkpoint analysis unexpectedly performed training")
    if provenance.get("strict_checkpoint_loading") is not True:
        raise ValueError("checkpoint loading was not strict")
    if provenance.get("missing_keys") or provenance.get("unexpected_keys"):
        raise ValueError("checkpoint incompatibilities were recorded")
    if "path_matched_information" not in {
        str(value) for value in provenance.get("enabled_analyzers", [])
    }:
        raise ValueError("path-matched information analyzer is absent from provenance")
    if (
        Path(str(provenance.get("output_dir", ""))).resolve()
        != Path(str(job["output_dir"])).resolve()
    ):
        raise ValueError("output directory differs between job and provenance")
    for field in ("config", "checkpoint", "analysis_config"):
        job_path = Path(str(job[field])).resolve()
        recorded = Path(str(provenance.get(f"{field}_path", ""))).resolve()
        if job_path != recorded or not job_path.is_file():
            raise ValueError(f"{field} path differs between job and provenance")
        digest = hash_cache.get(job_path)
        if digest is None:
            digest = _sha256(job_path)
            hash_cache[job_path] = digest
        if digest != provenance.get(f"{field}_sha256"):
            raise ValueError(f"{field} hash differs from provenance")


def _finite(value: Any, *, context: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value in {context}: {value!r}")
    return number


def _validate_result(
    result: Mapping[str, Any],
    *,
    expected_support_hash: str | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("unexpected path-information result schema")
    if result.get("analysis_split") != "validation":
        raise ValueError("path information was not estimated on validation")
    if result.get("test_examples_or_labels_used") is not False:
        raise ValueError("path information used test data")
    if result.get("feature_selection_uses_labels_or_activations") is not False:
        raise ValueError("path-information feature selection used data or labels")
    if result.get("technical_draws_are_model_replicates") is not False:
        raise ValueError("technical supports are marked as model replicates")
    support = result.get("support_plan")
    tree = result.get("tree")
    available = result.get("available_depths_by_signal")
    if not isinstance(support, dict) or not isinstance(tree, dict):
        raise ValueError("result lacks tree/support metadata")
    if not isinstance(available, dict) or not available:
        raise ValueError("result lacks structural signal availability")
    support_hash = str(support.get("selected_index_supports_sha256", ""))
    if not support_hash:
        raise ValueError("result lacks selected-index support hash")
    if expected_support_hash and support_hash != expected_support_hash:
        raise ValueError("selected-index support differs from declared contract")
    n_somas = int(support.get("n_somas", 0))
    draws = int(support.get("draw_count", 0))
    n_shuffles: int | None = None
    if n_somas < 1 or draws < 1:
        raise ValueError("invalid path-information support dimensions")
    if int(tree.get("n_somas", -1)) != n_somas:
        raise ValueError("tree and support soma counts differ")

    support_entries: dict[tuple[int, int], Mapping[str, Any]] = {}
    somatic_support = result.get("somatic_support")
    if not isinstance(somatic_support, dict):
        raise ValueError("result lacks the declared somatic support")
    support_depths = support.get("depths")
    if not isinstance(support_depths, dict):
        raise ValueError("support plan lacks non-somatic depth entries")
    entries_by_depth: dict[int, list[Mapping[str, Any]]] = {0: [somatic_support]}
    for depth_text, entries in support_depths.items():
        depth = int(depth_text)
        if depth < 1 or not isinstance(entries, list) or len(entries) != draws:
            raise ValueError(f"invalid support entries at dendritic depth {depth}")
        entries_by_depth[depth] = entries
    for depth, entries in entries_by_depth.items():
        expected_draws = 1 if depth == 0 else draws
        if len(entries) != expected_draws:
            raise ValueError(f"support draw count changed at depth {depth}")
        for entry in entries:
            draw = int(entry.get("support_draw", -1))
            if draw < 0 or draw >= expected_draws or (depth, draw) in support_entries:
                raise ValueError(
                    f"duplicate or invalid support identity {(depth, draw)}"
                )
            indices = np.asarray(entry.get("selected_indices", []), dtype=np.int64)
            if indices.shape != (n_somas,) or np.unique(indices).size != n_somas:
                raise ValueError(
                    f"invalid selected support at depth {depth}, draw {draw}"
                )
            digest = _sha256_array(indices)
            if digest != str(entry.get("selected_indices_sha256", "")):
                raise ValueError(
                    f"selected-support digest differs at depth {depth}, draw {draw}"
                )
            support_entries[(depth, draw)] = entry

    raw_records = result.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("path-information result contains no records")
    expected_grid = {
        (str(signal), int(depth), draw)
        for signal, depths in available.items()
        for depth in depths
        for draw in (range(draws) if int(depth) > 0 else range(1))
    }
    observed_grid: set[tuple[str, int, int]] = set()
    records: list[dict[str, Any]] = []
    for record_number, raw in enumerate(raw_records):
        record = dict(raw)
        identity = (
            str(record.get("signal")),
            int(record.get("soma_relative_depth", -1)),
            int(record.get("support_draw", -1)),
        )
        if identity in observed_grid:
            raise ValueError(f"duplicate path-information record: {identity}")
        observed_grid.add(identity)
        signal, depth, _draw = identity
        support_entry = support_entries.get((depth, _draw))
        if support_entry is None:
            raise ValueError(f"record refers to an undeclared support: {identity}")
        if record.get("technical_draw_role") != (
            "coordinate-support sensitivity; not an independent trained-model replicate"
        ):
            raise ValueError("path-information draw semantics changed")
        selected_count = int(record.get("selected_coordinate_count_per_component", -1))
        if selected_count != n_somas:
            raise ValueError("path-information coordinate count is not width matched")
        expected_joint_width = n_somas * (2 if signal == "joint_EI_currents" else 1)
        if int(record.get("joint_feature_count", -1)) != expected_joint_width:
            raise ValueError("path-information joint feature width changed")
        if str(record.get("selected_index_supports_sha256", "")) != support_hash:
            raise ValueError("record and support-plan hashes differ")
        if str(record.get("support_plan_sha256", "")) != str(
            support.get("support_plan_sha256", "")
        ):
            raise ValueError("record and support-plan identity digests differ")
        selected_features = [
            int(value) for value in record.get("selected_features", [])
        ]
        declared_features = [
            int(value) for value in support_entry.get("selected_indices", [])
        ]
        if selected_features != declared_features:
            raise ValueError("record selected features differ from its exact support")
        selected_digest = str(record.get("selected_features_sha256", ""))
        if selected_digest != str(support_entry.get("selected_indices_sha256", "")):
            raise ValueError("record selected-feature digest differs from its support")
        if record.get("feature_selection_policy") != (
            "shared_nested_anatomical_coordinate_support"
        ):
            raise ValueError("path-information feature-selection policy changed")
        if record.get("preprocessing_uses_labels") is not False:
            raise ValueError("path-information preprocessing used labels")
        for field in METRIC_FIELDS:
            record[field] = _finite(
                record.get(field),
                context=f"record {record_number} {field}",
            )
        null_values = np.asarray(record.get("label_shuffle_mi_bits", []), dtype=float)
        record_shuffles = int(record.get("n_label_shuffles", -1))
        if (
            record_shuffles < 1
            or null_values.size != record_shuffles
            or not np.isfinite(null_values).all()
        ):
            raise ValueError("path-information null distribution is incomplete")
        if n_shuffles is None:
            n_shuffles = record_shuffles
        elif n_shuffles != record_shuffles:
            raise ValueError("label-shuffle count changes across records")
        if depth == 0 and int(record.get("support_draw", -1)) != 0:
            raise ValueError("somatic information must use its single full support")
        records.append(record)
    if observed_grid != expected_grid:
        raise ValueError("path-information signal/depth/support grid is incomplete")

    contract = {
        "support_plan_sha256": str(support.get("support_plan_sha256", "")),
        "selected_index_supports_sha256": support_hash,
        "n_somas": n_somas,
        "branch_factors": list(support.get("branch_factors", [])),
        "support_draw_count": draws,
        "available_depths_by_signal": available,
        "n_label_shuffles": int(n_shuffles or 0),
        "sample_count": int(result.get("sample_count", -1)),
        "information_seed_effective": int(result.get("information_seed_effective", -1)),
    }
    if contract["sample_count"] < 1 or not contract["support_plan_sha256"]:
        raise ValueError("path-information run contract is incomplete")
    return records, contract


def _checkpoint_rows(
    *,
    job_index: int,
    job: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["signal"]), int(record["soma_relative_depth"]))].append(
            record
        )
    rows: list[dict[str, Any]] = []
    for (signal, depth), group in sorted(groups.items()):
        row = _base_row(job_index, job)
        row.update(
            {
                "signal": signal,
                "soma_relative_depth": depth,
                "technical_support_draw_count": len(group),
                "technical_draws_are_model_replicates": False,
                "selected_coordinate_count_per_component": int(
                    group[0]["selected_coordinate_count_per_component"]
                ),
                "joint_feature_count": int(group[0]["joint_feature_count"]),
            }
        )
        for field in METRIC_FIELDS:
            values = np.asarray([float(record[field]) for record in group], dtype=float)
            row[field] = float(values.mean())
            row[f"{field}_technical_std"] = (
                float(values.std(ddof=1)) if values.size > 1 else 0.0
            )
        rows.append(row)
    return rows


def collect_path_matched_information(
    *,
    jobs_paths: Sequence[str | Path],
    output_dir: str | Path,
    expected_count: int | None = None,
    expected_support_hash: str | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Validate declared runs and write support- and checkpoint-level tables."""

    paths = [Path(path).resolve() for path in jobs_paths]
    jobs = _read_jobs(paths)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(f"declared {len(jobs)} jobs; expected {expected_count}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    support_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    checkpoint_summary: list[dict[str, Any]] = []
    contracts: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for job_index, job in enumerate(jobs):
        run_output = Path(str(job["output_dir"])).resolve()
        filename = str(job.get("filename", "final"))
        result_path = run_output / RESULT_SUBDIRECTORY / f"{filename}.json"
        provenance_path = run_output / "checkpoint_analysis_provenance.json"
        missing = [
            path for path in (result_path, provenance_path) if not path.is_file()
        ]
        if missing:
            if not allow_incomplete:
                raise FileNotFoundError(missing[0])
            skipped.append(
                {
                    "job_index": job_index,
                    "output_dir": str(run_output),
                    "missing": [str(path) for path in missing],
                }
            )
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        _validate_run_provenance(
            job=job,
            provenance=provenance,
            hash_cache=hash_cache,
        )
        records, contract = _validate_result(
            result,
            expected_support_hash=expected_support_hash,
        )
        contracts.append(contract)
        for record in records:
            row = _base_row(job_index, job)
            row.update({str(key): _csv_cell(value) for key, value in record.items()})
            support_rows.append(row)
        averaged = _checkpoint_rows(
            job_index=job_index,
            job=job,
            records=records,
        )
        checkpoint_rows.extend(averaged)
        summary = _base_row(job_index, job)
        summary.update({key: _csv_cell(value) for key, value in contract.items()})
        summary.update(
            {
                "config_sha256": provenance["config_sha256"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "analysis_config_sha256": provenance["analysis_config_sha256"],
                "technical_record_count": len(records),
                "checkpoint_record_count": len(averaged),
                "run_code_identity": _csv_cell(provenance.get("code_identity")),
            }
        )
        checkpoint_summary.append(summary)
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
        if len(checkpoint_summary) != int(expected_count):
            raise ValueError(
                f"collected {len(checkpoint_summary)} runs; expected {expected_count}"
            )
    if contracts:
        for field in (
            "selected_index_supports_sha256",
            "n_somas",
            "branch_factors",
            "support_draw_count",
            "available_depths_by_signal",
            "n_label_shuffles",
            "sample_count",
        ):
            if len({_csv_cell(contract[field]) for contract in contracts}) != 1:
                raise ValueError(f"path-information contract differs: {field}")

    tables = {
        "checkpoint_summary": checkpoint_summary,
        "support_draw_records": support_rows,
        "checkpoint_records": checkpoint_rows,
    }
    output_tables = []
    for name, rows in tables.items():
        path = output / f"{name}.csv"
        _write_csv(path, rows)
        output_tables.append(
            {"path": str(path), "sha256": _sha256(path), "row_count": len(rows)}
        )
    release = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_files": [{"path": str(path), "sha256": _sha256(path)} for path in paths],
        "declared_job_count": len(jobs),
        "expected_job_count": expected_count,
        "collected_job_count": len(checkpoint_summary),
        "allow_incomplete": bool(allow_incomplete),
        "expected_support_hash": expected_support_hash,
        "skipped_jobs": skipped,
        "source_artifacts": sources,
        "output_artifacts": output_tables,
        "collector_code_identity": discover_source_identity(
            _repo_root(),
            source_files=(Path(__file__),),
        ),
        "inference_unit": "independently trained checkpoint",
        "technical_draws_are_model_replicates": False,
    }
    release_path = output / "collection_provenance.json"
    release_path.write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release


def _resolve_config_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_collection_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported path-information collection config: {path}")
    missing = {"jobs", "output_dir"} - set(payload)
    if missing:
        raise ValueError(f"collection config lacks required fields: {sorted(missing)}")
    jobs = payload["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("collection config jobs must be a nonempty list")
    base = path.resolve().parent
    return {
        "jobs_paths": [_resolve_config_path(value, base=base) for value in jobs],
        "output_dir": _resolve_config_path(payload["output_dir"], base=base),
        "expected_count": payload.get("expected_count"),
        "expected_support_hash": payload.get("expected_support_hash"),
        "allow_incomplete": bool(payload.get("allow_incomplete", False)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--jobs", action="append")
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--expected-support-hash")
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _parser().parse_args(argv)
    if args.config is not None:
        explicit = any(
            (
                args.jobs,
                args.output_dir,
                args.expected_count is not None,
                args.expected_support_hash,
                args.allow_incomplete,
            )
        )
        if explicit:
            raise ValueError("--config cannot be combined with explicit options")
        return collect_path_matched_information(**_load_collection_config(args.config))
    if not args.jobs or args.output_dir is None:
        raise ValueError("--jobs and --output-dir are required without --config")
    return collect_path_matched_information(
        jobs_paths=args.jobs,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
        expected_support_hash=args.expected_support_hash,
        allow_incomplete=args.allow_incomplete,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
