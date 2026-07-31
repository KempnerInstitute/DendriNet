"""Collect exact sparse synaptic-support records from frozen checkpoints.

The registered weight analyzer emits one scalar record per soma, dendritic
depth, and pathway, plus NPZ arrays containing the corresponding source
profiles. Optional source-region alignment records connect those profiles to a
declared reference split without using model activity. This collector validates
all representations and preserves their structural coordinates. It performs no
across-checkpoint aggregation and never treats somas, depths, pathways, or
source coordinates as independent replicates.
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

SCHEMA_VERSION = "synaptic_source_support_collection_v2"
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
_REGION_COMMON_FIELDS = _STRUCTURAL_FIELDS | {
    "soma_class",
    "quantity",
    "source_profile_key",
    "reference_split",
    "reference_sample_count",
    "preferred_class_sample_count",
    "rest_class_sample_count",
}
_ACTIVITY_BASE_METRIC_FIELDS = {
    "profile_total_mass",
    "preferred_foreground_fraction",
    "preferred_background_fraction",
    "rest_foreground_fraction",
    "rest_background_fraction",
    "preferred_rest_foreground_contrast",
    "preferred_rest_background_contrast",
    "uniform_preferred_foreground_occupancy",
    "uniform_preferred_background_occupancy",
    "uniform_rest_foreground_occupancy",
    "uniform_rest_background_occupancy",
    "preferred_foreground_occupancy_excess",
    "preferred_background_occupancy_excess",
    "rest_foreground_occupancy_excess",
    "rest_background_occupancy_excess",
    "preferred_foreground_occupancy_enrichment",
    "preferred_background_occupancy_enrichment",
    "rest_foreground_occupancy_enrichment",
    "rest_background_occupancy_enrichment",
}
_ACTIVITY_METRIC_FIELDS = _ACTIVITY_BASE_METRIC_FIELDS | {
    f"{region}_{field}"
    for region in ("center", "surround")
    for field in _ACTIVITY_BASE_METRIC_FIELDS
}
_ACTIVITY_BOOL_FIELDS = {
    "profile_has_mass",
    "center_profile_has_mass",
    "surround_profile_has_mass",
}
_SPATIAL_METRIC_FIELDS = {
    "profile_total_mass",
    "center_support_fraction",
    "surround_support_fraction",
    "center_pixel_fraction",
    "surround_pixel_fraction",
    "center_support_excess",
    "surround_support_excess",
    "center_support_enrichment",
    "surround_support_enrichment",
    "center_surround_enrichment_contrast",
}
_SPATIAL_BOUND_FIELDS = {
    "center_row_start",
    "center_row_stop",
    "center_col_start",
    "center_col_stop",
}
_REGION_CHECKPOINT_STRUCTURAL_FIELDS = (_STRUCTURAL_FIELDS - {"soma_index"}) | {
    "quantity",
    "source_profile_key",
    "reference_split",
    "reference_sample_count",
}
_SOURCE_ACTIVITY_CHECKPOINT_IDENTITY_FIELDS = _REGION_CHECKPOINT_STRUCTURAL_FIELDS | {
    "foreground_threshold"
}
_SPATIAL_REGION_CHECKPOINT_IDENTITY_FIELDS = (
    _REGION_CHECKPOINT_STRUCTURAL_FIELDS | _SPATIAL_BOUND_FIELDS
)


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


def _validate_region_alignment(
    result: Mapping[str, Any],
    *,
    result_path: Path,
) -> tuple[
    Mapping[str, Any] | None,
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
]:
    """Validate optional structural source-region alignment records."""
    alignment = result.get("source_region_alignment")
    if alignment is None:
        return None, [], [], []
    if not isinstance(alignment, dict):
        raise TypeError(f"source_region_alignment is not a mapping: {result_path}")
    contract = alignment.get("quantity_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("structural_not_activity_or_information") is not True
    ):
        raise ValueError(f"source-region quantity contract is absent: {result_path}")
    reference = alignment.get("reference")
    if not isinstance(reference, dict) or reference.get("split") not in {
        "train",
        "validation",
        "test",
    }:
        raise ValueError(f"source-region reference split is invalid: {result_path}")
    activity = alignment.get("source_activity_records")
    spatial = alignment.get("spatial_region_records")
    skipped = alignment.get("skipped_profile_blocks")
    if not all(isinstance(records, list) for records in (activity, spatial, skipped)):
        raise TypeError(f"source-region record payloads must be lists: {result_path}")
    assert isinstance(activity, list)
    assert isinstance(spatial, list)
    assert isinstance(skipped, list)
    declared_counts = (
        ("n_source_activity_records", activity),
        ("n_spatial_region_records", spatial),
        ("n_skipped_profile_blocks", skipped),
    )
    for count_field, records in declared_counts:
        if int(alignment.get(count_field, -1)) != len(records):
            raise ValueError(
                f"source-region {count_field} is inconsistent: {result_path}"
            )

    reference_split = str(reference["split"])
    activity_required = (
        _REGION_COMMON_FIELDS
        | {"foreground_threshold"}
        | _ACTIVITY_METRIC_FIELDS
        | _ACTIVITY_BOOL_FIELDS
    )
    for index, record in enumerate(activity):
        if not isinstance(record, dict):
            raise TypeError(f"source-activity record {index} is not a mapping")
        missing = activity_required - set(record)
        if missing:
            raise ValueError(
                f"source-activity record {index} lacks fields: {sorted(missing)}"
            )
        if record["reference_split"] != reference_split:
            raise ValueError("source-activity record/reference split mismatch")
        if record["quantity"] not in {
            "contact_frequency",
            "mean_effective_conductance",
        }:
            raise ValueError(f"source-activity record {index} has invalid quantity")
        for field in _ACTIVITY_BOOL_FIELDS:
            if not isinstance(record[field], bool):
                raise TypeError(f"source-activity {field} must be boolean")
        for field in _ACTIVITY_METRIC_FIELDS | {"foreground_threshold"}:
            _finite_or_none(
                record[field],
                context=f"source_activity_records[{index}].{field}",
            )

    spatial_required = (
        _REGION_COMMON_FIELDS
        | _SPATIAL_METRIC_FIELDS
        | _SPATIAL_BOUND_FIELDS
        | {"profile_has_mass"}
    )
    for index, record in enumerate(spatial):
        if not isinstance(record, dict):
            raise TypeError(f"spatial-region record {index} is not a mapping")
        missing = spatial_required - set(record)
        if missing:
            raise ValueError(
                f"spatial-region record {index} lacks fields: {sorted(missing)}"
            )
        if record["reference_split"] != reference_split:
            raise ValueError("spatial-region record/reference split mismatch")
        if not isinstance(record["profile_has_mass"], bool):
            raise TypeError("spatial-region profile_has_mass must be boolean")
        for field in _SPATIAL_METRIC_FIELDS | _SPATIAL_BOUND_FIELDS:
            _finite_or_none(
                record[field],
                context=f"spatial_region_records[{index}].{field}",
            )

    skip_required = (_STRUCTURAL_FIELDS - {"soma_index"}) | {
        "declared_image_network_layers",
        "expected_image_source_dim",
        "skip_reason",
    }
    for index, record in enumerate(skipped):
        if not isinstance(record, dict):
            raise TypeError(f"source-region skip record {index} is not a mapping")
        missing = skip_required - set(record)
        if missing:
            raise ValueError(
                f"source-region skip record {index} lacks fields: {sorted(missing)}"
            )
        if record["skip_reason"] not in {
            "network_layer_not_declared_image_source",
            "source_dimension_not_image_space",
        }:
            raise ValueError(f"source-region skip record {index} has invalid reason")
        if not isinstance(record["declared_image_network_layers"], list):
            raise TypeError("declared_image_network_layers must be a list")
    return alignment, activity, spatial, skipped


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


def _checkpoint_summary_rows(
    support_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Average technical soma records within each fitted checkpoint."""
    excluded = {
        *_SCALAR_FIELDS,
        "soma_index",
        "technical_record_is_independent_replicate",
        "trained_checkpoint_is_inference_unit",
    }
    groups: dict[tuple[tuple[str, Any], ...], list[Mapping[str, Any]]] = {}
    for row in support_rows:
        identity = tuple(
            (key, value) for key, value in sorted(row.items()) if key not in excluded
        )
        groups.setdefault(identity, []).append(row)

    summaries: list[dict[str, Any]] = []
    for identity, rows in groups.items():
        n_soma = int(rows[0]["n_soma"])
        soma_indices = {int(row["soma_index"]) for row in rows}
        if len(rows) != n_soma or soma_indices != set(range(n_soma)):
            raise ValueError(
                "source-support checkpoint group does not contain exactly one "
                "record for every soma"
            )
        summary = dict(identity)
        summary.update(
            {
                "technical_soma_count": len(rows),
                "technical_aggregation": "mean_across_somas_within_checkpoint",
                "trained_checkpoint_is_inference_unit": True,
            }
        )
        for field in sorted(_SCALAR_FIELDS):
            values = [float(row[field]) for row in rows if row.get(field) is not None]
            summary[f"{field}_defined_count"] = len(values)
            summary[f"{field}_mean"] = float(np.mean(values)) if values else None
            summary[f"{field}_sd"] = float(np.std(values, ddof=0)) if values else None
        summaries.append(summary)
    return summaries


def _checkpoint_region_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    identity_fields: set[str],
    metric_fields: set[str],
    table_name: str,
) -> list[dict[str, Any]]:
    """Reduce one complete technical soma inventory per checkpoint group."""
    groups: dict[tuple[tuple[str, Any], ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        if "job_index" not in row:
            raise ValueError(f"{table_name} checkpoint summary lacks job_index")
        job_fields = {
            key for key in row if key == "job_index" or str(key).startswith("job_")
        }
        required = (
            job_fields
            | identity_fields
            | metric_fields
            | {
                "soma_index",
                "n_soma",
            }
        )
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"{table_name} checkpoint summary lacks identity fields: "
                f"{sorted(missing)}"
            )
        identity = tuple(
            (key, row[key]) for key in sorted(job_fields | identity_fields)
        )
        groups.setdefault(identity, []).append(row)

    summaries: list[dict[str, Any]] = []
    for identity, group_rows in groups.items():
        summary = dict(identity)
        n_soma = int(summary["n_soma"])
        soma_indices = [int(row["soma_index"]) for row in group_rows]
        if len(soma_indices) != len(set(soma_indices)):
            raise ValueError(f"{table_name} checkpoint group contains duplicate somas")
        if len(group_rows) != n_soma or set(soma_indices) != set(range(n_soma)):
            raise ValueError(
                f"{table_name} checkpoint group must contain exactly one record "
                "for every soma"
            )

        summary.update(
            {
                "technical_soma_count": n_soma,
                "technical_aggregation": "equal_mean_across_somas_within_checkpoint",
                "soma_weighting": "equal",
                "technical_somas_are_independent_replicates": False,
                "trained_checkpoint_is_inference_unit": True,
            }
        )
        for field in sorted(metric_fields):
            values = [
                value
                for row in group_rows
                if (
                    value := _finite_or_none(
                        row.get(field), context=f"{table_name}.{field}"
                    )
                )
                is not None
            ]
            summary[f"{field}_defined_count"] = len(values)
            if not values:
                summary[f"{field}_mean"] = None
                summary[f"{field}_sd"] = None
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            summary[f"{field}_mean"] = mean
            summary[f"{field}_sd"] = math.sqrt(variance)
        summaries.append(summary)
    return summaries


def _checkpoint_source_activity_summary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Equal-soma checkpoint summaries for source-activity alignment."""
    return _checkpoint_region_summary_rows(
        rows,
        identity_fields=_SOURCE_ACTIVITY_CHECKPOINT_IDENTITY_FIELDS,
        metric_fields=_ACTIVITY_METRIC_FIELDS,
        table_name="source-activity alignment",
    )


def _checkpoint_spatial_region_summary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Equal-soma checkpoint summaries for spatial center/surround alignment."""
    return _checkpoint_region_summary_rows(
        rows,
        identity_fields=_SPATIAL_REGION_CHECKPOINT_IDENTITY_FIELDS,
        metric_fields=_SPATIAL_METRIC_FIELDS,
        table_name="spatial-region alignment",
    )


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
    source_activity_rows: list[dict[str, Any]] = []
    spatial_region_rows: list[dict[str, Any]] = []
    skipped_region_rows: list[dict[str, Any]] = []
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
        alignment, activity_records, spatial_records, skipped_records = (
            _validate_region_alignment(
                result,
                result_path=result_path,
            )
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
        for record in activity_records:
            source_activity_rows.append(
                {
                    **_job_fields(job_index, job),
                    **record,
                    "technical_record_is_independent_replicate": False,
                    "trained_checkpoint_is_inference_unit": True,
                }
            )
        for record in spatial_records:
            spatial_region_rows.append(
                {
                    **_job_fields(job_index, job),
                    **record,
                    "technical_record_is_independent_replicate": False,
                    "trained_checkpoint_is_inference_unit": True,
                }
            )
        for record in skipped_records:
            skipped_region_rows.append(
                {
                    **_job_fields(job_index, job),
                    **record,
                    "technical_record_is_independent_replicate": False,
                }
            )

        alignment_artifacts: dict[str, Any] = {
            "source_region_alignment_present": alignment is not None,
            "source_activity_record_count": len(activity_records),
            "spatial_region_record_count": len(spatial_records),
            "skipped_region_block_count": len(skipped_records),
        }
        if alignment is not None:
            reference = alignment["reference"]
            alignment_artifacts["source_region_reference_split"] = reference["split"]
            alignment_artifacts["source_region_foreground_thresholds"] = json.dumps(
                sorted(
                    {
                        float(record["foreground_threshold"])
                        for record in activity_records
                    }
                )
            )
            for filename_field, declared_records in (
                ("source_activity_records_file", activity_records),
                ("spatial_region_records_file", spatial_records),
                ("skipped_profile_blocks_file", skipped_records),
            ):
                declared = alignment.get(filename_field)
                if declared is None:
                    if declared_records:
                        raise ValueError(
                            f"{filename_field} is absent despite emitted records"
                        )
                    alignment_artifacts[f"{filename_field}_path"] = None
                    alignment_artifacts[f"{filename_field}_sha256"] = None
                    continue
                artifact_path = result_path.parent / str(declared)
                if not artifact_path.is_file():
                    raise FileNotFoundError(artifact_path)
                alignment_artifacts[f"{filename_field}_path"] = str(artifact_path)
                alignment_artifacts[f"{filename_field}_sha256"] = _sha256(artifact_path)
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
                **alignment_artifacts,
            }
        )

    outputs = {
        "checkpoint_support_records.csv": support_rows,
        "checkpoint_support_summary.csv": _checkpoint_summary_rows(support_rows),
        "profile_inventory.csv": profile_rows,
        "checkpoint_source_activity_alignment_records.csv": source_activity_rows,
        "checkpoint_source_activity_alignment_summary.csv": (
            _checkpoint_source_activity_summary_rows(source_activity_rows)
        ),
        "checkpoint_spatial_region_alignment_records.csv": spatial_region_rows,
        "checkpoint_spatial_region_alignment_summary.csv": (
            _checkpoint_spatial_region_summary_rows(spatial_region_rows)
        ),
        "source_region_skipped_blocks.csv": skipped_region_rows,
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
        "source_activity_record_count": len(source_activity_rows),
        "source_activity_checkpoint_summary_count": len(
            outputs["checkpoint_source_activity_alignment_summary.csv"]
        ),
        "spatial_region_record_count": len(spatial_region_rows),
        "spatial_region_checkpoint_summary_count": len(
            outputs["checkpoint_spatial_region_alignment_summary.csv"]
        ),
        "skipped_region_block_count": len(skipped_region_rows),
        "source_region_reference_splits": sorted(
            {
                str(row["source_region_reference_split"])
                for row in source_rows
                if row.get("source_region_reference_split") is not None
            }
        ),
        "source_region_foreground_thresholds": sorted(
            {float(row["foreground_threshold"]) for row in source_activity_rows}
        ),
        "aggregation_policy": "equal_soma_mean_within_checkpoint_only",
        "within_checkpoint_soma_weighting": "equal",
        "technical_soma_aggregation_performed": True,
        "trained_seed_aggregation_performed": False,
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
