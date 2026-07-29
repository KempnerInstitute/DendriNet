"""Collect exact sparse synaptic-support records from frozen checkpoints.

The registered weight analyzer emits one scalar record per soma, dendritic
depth, and pathway, plus NPZ arrays containing the corresponding source
profiles. This collector validates both representations and preserves their
structural coordinates. It performs no across-checkpoint aggregation and never
treats somas, depths, pathways, or source coordinates as independent replicates.
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

import numpy as np
import yaml

from dendritic_modeling.scripts.analysis.collect_checkpoint_analysis import (
    _job_fields,
    _read_jobs,
    _sha256,
    _validate_provenance,
)
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity

SCHEMA_VERSION = "synaptic_source_support_collection_v1"
CONFIG_SCHEMA_VERSION = "synaptic_source_support_collection_config_v1"
RESULT_SUBDIRECTORY = Path("weight_analysis")
RESULT_FILENAME = "final"

_STRUCTURAL_FIELDS = {
    "network_layer_index",
    "population_name",
    "target_polarity",
    "dendritic_depth",
    "depth_label",
    "pathway",
    "source_polarity",
    "n_soma",
    "branches_per_soma",
    "source_dim",
    "selection_policy",
    "mask_source",
    "soma_index",
}
_SCALAR_FIELDS = {
    "realized_contacts_per_branch_mean",
    "active_effective_conductance_mean",
    "source_coverage_fraction",
    "eligible_source_fraction",
    "source_coverage_of_eligible",
    "source_entropy_bits",
    "normalized_source_entropy",
    "effective_source_fraction",
    "mean_pairwise_branch_jaccard",
}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if "job_index" in fields:
        fields.remove("job_index")
        fields.insert(0, "job_index")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["job_index"])
        writer.writeheader()
        writer.writerows(rows)


def _finite_or_none(value: Any, *, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite or null")
    return number


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_support(
    result: Mapping[str, Any],
    *,
    result_path: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    support = result.get("source_support")
    if not isinstance(support, dict):
        raise ValueError(f"weight result has no source_support: {result_path}")
    contract = support.get("quantity_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("not_activity_or_information") is not True
    ):
        raise ValueError(f"source-support quantity contract is absent: {result_path}")
    records = support.get("records")
    metadata = support.get("profile_metadata")
    if not isinstance(records, list) or not records:
        raise ValueError(f"source-support records are absent: {result_path}")
    if not isinstance(metadata, list) or not metadata:
        raise ValueError(f"source-support profile metadata are absent: {result_path}")
    if int(support.get("n_scalar_records", -1)) != len(records):
        raise ValueError(f"source-support record count is inconsistent: {result_path}")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(f"source-support record {index} is not a mapping")
        missing = (_STRUCTURAL_FIELDS | _SCALAR_FIELDS) - set(record)
        if missing:
            raise ValueError(
                f"source-support record {index} lacks fields: {sorted(missing)}"
            )
        for field in _SCALAR_FIELDS:
            _finite_or_none(
                record[field],
                context=f"source_support.records[{index}].{field}",
            )
    return support, records, metadata


def _profile_rows(
    *,
    job_index: int,
    job: Mapping[str, Any],
    metadata: Sequence[Mapping[str, Any]],
    profiles_path: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with np.load(profiles_path, allow_pickle=False) as profiles:
        expected_keys: set[str] = set()
        for index, item in enumerate(metadata):
            if not isinstance(item, dict):
                raise TypeError(f"profile metadata {index} is not a mapping")
            missing = _STRUCTURAL_FIELDS - {"soma_index"} - set(item)
            if missing:
                raise ValueError(
                    f"profile metadata {index} lacks fields: {sorted(missing)}"
                )
            expected_shape = (int(item["n_soma"]), int(item["source_dim"]))
            for quantity, key_field in (
                ("contact_frequency", "contact_frequency_key"),
                (
                    "mean_effective_conductance",
                    "mean_effective_conductance_key",
                ),
            ):
                key = str(item.get(key_field, ""))
                if not key or key in expected_keys or key not in profiles:
                    raise ValueError(
                        f"invalid or duplicate profile key {key!r} at metadata {index}"
                    )
                expected_keys.add(key)
                array = np.asarray(profiles[key])
                if array.shape != expected_shape:
                    raise ValueError(
                        f"profile {key} has shape {array.shape}; expected {expected_shape}"
                    )
                if not np.isfinite(array).all():
                    raise ValueError(f"profile {key} contains non-finite values")
                rows.append(
                    {
                        **_job_fields(job_index, job),
                        **{
                            field: item[field]
                            for field in sorted(_STRUCTURAL_FIELDS - {"soma_index"})
                        },
                        "quantity": quantity,
                        "profile_key": key,
                        "profile_shape": json.dumps(list(array.shape)),
                        "profile_dtype": str(array.dtype),
                        "profile_sha256": _array_sha256(array),
                        "technical_profile_is_independent_replicate": False,
                    }
                )
        unexpected = sorted(set(profiles.files) - expected_keys)
        if unexpected:
            raise ValueError(f"NPZ contains undeclared profiles: {unexpected}")
    return rows


def collect_synaptic_support(
    *,
    jobs_paths: Sequence[Path],
    output_dir: Path,
    expected_count: int | None = None,
) -> dict[str, Any]:
    """Validate declared support artifacts and emit checkpoint-bound tables."""
    jobs = _read_jobs(jobs_paths)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(f"declared {len(jobs)} jobs; expected {expected_count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    support_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for job_index, job in enumerate(jobs):
        run_dir = Path(str(job["output_dir"])).resolve()
        provenance_path = run_dir / "checkpoint_analysis_provenance.json"
        result_path = run_dir / RESULT_SUBDIRECTORY / RESULT_FILENAME
        if not provenance_path.is_file():
            raise FileNotFoundError(provenance_path)
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        _validate_provenance(
            job,
            provenance,
            hash_cache,
            input_hash_validation="rehash",
        )
        if "weight" not in {
            str(name) for name in provenance.get("enabled_analyzers", [])
        }:
            raise ValueError("weight analyzer is absent from run provenance")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise TypeError(f"weight result is not a mapping: {result_path}")
        support, records, metadata = _validate_support(
            result,
            result_path=result_path,
        )
        profiles_name = str(support.get("profiles_file", ""))
        profiles_path = result_path.parent / profiles_name
        if not profiles_name or not profiles_path.is_file():
            raise FileNotFoundError(profiles_path)

        for record in records:
            support_rows.append(
                {
                    **_job_fields(job_index, job),
                    **record,
                    "technical_record_is_independent_replicate": False,
                    "trained_checkpoint_is_inference_unit": True,
                }
            )
        profile_rows.extend(
            _profile_rows(
                job_index=job_index,
                job=job,
                metadata=metadata,
                profiles_path=profiles_path,
            )
        )
        source_rows.append(
            {
                **_job_fields(job_index, job),
                "result_path": str(result_path),
                "result_sha256": _sha256(result_path),
                "profiles_path": str(profiles_path),
                "profiles_sha256": _sha256(profiles_path),
                "provenance_path": str(provenance_path),
                "provenance_sha256": _sha256(provenance_path),
                "support_record_count": len(records),
                "profile_count": len(metadata) * 2,
            }
        )

    outputs = {
        "checkpoint_support_records.csv": support_rows,
        "profile_inventory.csv": profile_rows,
        "source_artifacts.csv": source_rows,
    }
    for filename, rows in outputs.items():
        _write_csv(output_dir / filename, rows)
    release = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_paths": [str(path.resolve()) for path in jobs_paths],
        "jobs_sha256": [_sha256(path.resolve()) for path in jobs_paths],
        "collected_checkpoint_count": len(jobs),
        "support_record_count": len(support_rows),
        "profile_record_count": len(profile_rows),
        "aggregation_policy": "none",
        "trained_checkpoint_is_inference_unit": True,
        "soma_depth_pathway_records_are_technical": True,
        "collector_code_identity": discover_source_identity(
            Path(__file__).resolve().parents[4],
            source_files=(Path(__file__),),
        ),
        "outputs": {filename: _sha256(output_dir / filename) for filename in outputs},
    }
    (output_dir / "collection_provenance.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release


def _resolve_config_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_collection_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported synaptic-support collection config: {path}")
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
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--jobs", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-count", type=int)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    if args.config is not None:
        if args.jobs or args.output_dir is not None or args.expected_count is not None:
            raise ValueError("--config cannot be combined with explicit options")
        return collect_synaptic_support(**_load_collection_config(args.config))
    if not args.jobs or args.output_dir is None:
        raise ValueError("--jobs and --output-dir are required without --config")
    return collect_synaptic_support(
        jobs_paths=args.jobs,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
