"""Collect split-safe source-tuning/support records from frozen checkpoints.

The ``source_tuning_support`` analyzer emits scalar records by fitted
checkpoint, network layer, target soma, dendritic depth, pathway, and native
source block.  This collector validates the split and coordinate contracts,
flattens scalar alignment and preference records into CSV, and preserves the
checkpoint-analysis hash chain.  It performs no aggregation across fitted
checkpoints or trained seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dendritic_modeling.scripts.analysis.collect_checkpoint_analysis import (
    INPUT_HASH_VALIDATION_MODES,
    _job_fields,
    _read_jobs,
    _sha256,
    _validate_provenance,
    require_completed_analyses,
)
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity
from dendritic_modeling.utils.logging_config import setup_logging

SCHEMA_VERSION = "source_tuning_support_collection_v1"
CONFIG_SCHEMA_VERSION = "source_tuning_support_collection_config_v1"
RESULT_SCHEMA_VERSION = 1
ANALYZER_NAME = "source_tuning_support"
RESULT_SUBDIRECTORY = "source_tuning_support_analysis"
RESULT_FILENAME = "final"

_PREFERENCE_FIELDS = {
    "target_key",
    "network_layer_index",
    "network_layer_name",
    "target_population",
    "target_polarity",
    "target_soma_index",
    "preferred_class",
    "preferred_reference_mean",
    "runner_up_class",
    "runner_up_reference_mean",
    "preference_margin",
}
_ALIGNMENT_IDENTITY_FIELDS = {
    "source_block_key",
    "target_key",
    "network_layer_index",
    "network_layer_name",
    "target_population",
    "target_polarity",
    "n_soma",
    "dendritic_depth",
    "pathway",
    "source_name",
    "source_polarity",
    "source_kind",
    "source_layer_name",
    "source_population_name",
    "coordinate_space",
    "image_shape",
    "source_start",
    "source_stop",
    "source_dim",
    "target_soma_index",
    "preferred_class",
    "activity_threshold_kind",
    "profile_key",
}
_ALIGNMENT_COUNT_FIELDS = {
    "n_branches",
    "n_preferred_evaluation_samples",
    "n_rest_evaluation_samples",
    "candidate_contact_count",
    "active_contact_count",
    "n_source_coordinates",
    "n_eligible_source_coordinates",
    "n_class_profile_classes",
    "n_degenerate_target_class_profiles",
    "n_degenerate_source_class_profiles",
    "n_eligible_degenerate_source_class_profiles",
}
_ALIGNMENT_BASE_NUMERIC_FIELDS = {
    "activity_threshold",
    "active_fraction_of_candidates",
    "total_effective_conductance",
}
_TUNING_METRICS = {
    "preferred_mean",
    "rest_mean",
    "mean_contrast",
    "preferred_activity_probability",
    "rest_activity_probability",
    "activity_probability_contrast",
    "standardized_mean_difference",
    "class_profile_alignment",
}
_WEIGHTINGS = {"source_uniform", "contact_weighted", "conductance_weighted"}
_ALIGNMENT_WEIGHTED_FIELDS = {
    f"{weighting}_{metric}" for weighting in _WEIGHTINGS for metric in _TUNING_METRICS
}
_ALIGNMENT_EXCESS_FIELDS = {
    f"{weighting}_{metric}_excess_over_uniform"
    for weighting in {"contact_weighted", "conductance_weighted"}
    for metric in {
        "mean_contrast",
        "activity_probability_contrast",
        "standardized_mean_difference",
        "class_profile_alignment",
    }
}
_ALIGNMENT_NUMERIC_FIELDS = (
    _ALIGNMENT_BASE_NUMERIC_FIELDS
    | _ALIGNMENT_WEIGHTED_FIELDS
    | _ALIGNMENT_EXCESS_FIELDS
)
_ALIGNMENT_REQUIRED_FIELDS = (
    _ALIGNMENT_IDENTITY_FIELDS | _ALIGNMENT_COUNT_FIELDS | _ALIGNMENT_NUMERIC_FIELDS
)
_CHECKPOINT_ALIGNMENT_STRUCTURAL_FIELDS = _ALIGNMENT_IDENTITY_FIELDS - {
    "target_soma_index",
    "preferred_class",
    "profile_key",
} | {
    "activity_threshold",
    "n_branches",
    "n_class_profile_classes",
    "n_source_coordinates",
}
_CHECKPOINT_ALIGNMENT_SUMMARY_FIELDS = _ALIGNMENT_COUNT_FIELDS - {
    "n_branches",
    "n_class_profile_classes",
    "n_source_coordinates",
} | _ALIGNMENT_NUMERIC_FIELDS - {"activity_threshold"}
_PROFILE_VECTOR_FIELDS = {
    "candidate_frequency",
    "exact_contact_frequency",
    "mean_effective_conductance",
    *_TUNING_METRICS,
}
_COORDINATE_SPACES = {
    "image_pixels",
    "external_features",
    "native_source_neurons",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


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


def _nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{context} must be a non-negative integer")
    return int(value)


def _validate_scalar_mapping(record: Mapping[str, Any], *, context: str) -> None:
    for field, value in record.items():
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"{context}.{field} must be scalar or null")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{context}.{field} must be finite")


def _csv_scalar_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Encode declared structured metadata while preserving scalar CSV rows."""
    return {
        key: (
            json.dumps(value, separators=(",", ":"))
            if key == "image_shape" and value is not None
            else value
        )
        for key, value in record.items()
    }


def _validate_image_shape(
    value: Any,
    *,
    coordinate_space: str,
    source_dim: int,
    context: str,
) -> None:
    """Validate optional [H,W]/[C,H,W] metadata against source dimensionality."""
    if coordinate_space != "image_pixels":
        if value is not None:
            raise ValueError(f"{context} must be null for non-image coordinates")
        return
    if not isinstance(value, list) or len(value) not in {2, 3}:
        raise ValueError(f"{context} must be [H,W] or [C,H,W]")
    dimensions = [
        _nonnegative_int(dimension, context=f"{context}[{index}]")
        for index, dimension in enumerate(value)
    ]
    if any(dimension < 1 for dimension in dimensions):
        raise ValueError(f"{context} dimensions must be positive")
    if math.prod(dimensions) != source_dim:
        raise ValueError(f"{context} product does not match source_dim")


def _class_counts(value: Any, *, context: str) -> dict[int, int]:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{context} must be a nonempty mapping")
    counts: dict[int, int] = {}
    for label, count in value.items():
        try:
            class_id = int(label)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{context} has a non-integer class label") from exc
        counts[class_id] = _nonnegative_int(count, context=f"{context}[{label!r}]")
    if any(count < 1 for count in counts.values()):
        raise ValueError(f"{context} class counts must be positive")
    return counts


def _validate_preference_records(
    records: Any,
    *,
    reference_counts: Mapping[int, int],
) -> tuple[list[Mapping[str, Any]], dict[tuple[str, int], int]]:
    if not isinstance(records, list) or not records:
        raise ValueError("reference preference_records must be a nonempty list")
    preferences: dict[tuple[str, int], int] = {}
    for index, record in enumerate(records):
        context = f"reference.preference_records[{index}]"
        if not isinstance(record, Mapping):
            raise TypeError(f"{context} must be a mapping")
        missing = _PREFERENCE_FIELDS - set(record)
        if missing:
            raise ValueError(f"{context} lacks fields: {sorted(missing)}")
        _validate_scalar_mapping(record, context=context)
        soma_index = _nonnegative_int(
            record["target_soma_index"], context=f"{context}.target_soma_index"
        )
        preferred_class = _nonnegative_int(
            record["preferred_class"], context=f"{context}.preferred_class"
        )
        runner_up_class = _nonnegative_int(
            record["runner_up_class"], context=f"{context}.runner_up_class"
        )
        if preferred_class == runner_up_class:
            raise ValueError(f"{context} preferred and runner-up classes are equal")
        if preferred_class not in reference_counts:
            raise ValueError(f"{context} preferred class is absent from reference")
        if runner_up_class not in reference_counts:
            raise ValueError(f"{context} runner-up class is absent from reference")
        for field in (
            "preferred_reference_mean",
            "runner_up_reference_mean",
            "preference_margin",
        ):
            _finite_or_none(record[field], context=f"{context}.{field}")
        expected_margin = float(record["preferred_reference_mean"]) - float(
            record["runner_up_reference_mean"]
        )
        if not math.isclose(
            float(record["preference_margin"]),
            expected_margin,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{context} preference margin is inconsistent")
        identity = (str(record["target_key"]), soma_index)
        if identity in preferences:
            raise ValueError(f"duplicate target-soma preference: {identity}")
        preferences[identity] = preferred_class
    return records, preferences


def _validate_alignment_records(
    records: Any,
    *,
    declared_count: Any,
    preferences: Mapping[tuple[str, int], int],
    evaluation_counts: Mapping[int, int],
) -> list[Mapping[str, Any]]:
    if not isinstance(records, list) or not records:
        raise ValueError("alignment_records must be a nonempty list")
    if _nonnegative_int(declared_count, context="n_alignment_records") != len(records):
        raise ValueError("n_alignment_records does not match alignment_records")
    evaluation_total = sum(evaluation_counts.values())
    seen_profiles: set[str] = set()
    for index, record in enumerate(records):
        context = f"alignment_records[{index}]"
        if not isinstance(record, Mapping):
            raise TypeError(f"{context} must be a mapping")
        missing = _ALIGNMENT_REQUIRED_FIELDS - set(record)
        if missing:
            raise ValueError(f"{context} lacks fields: {sorted(missing)}")
        _validate_scalar_mapping(
            {key: value for key, value in record.items() if key != "image_shape"},
            context=context,
        )
        for field in _ALIGNMENT_COUNT_FIELDS:
            _nonnegative_int(record[field], context=f"{context}.{field}")
        for field in _ALIGNMENT_NUMERIC_FIELDS:
            _finite_or_none(record[field], context=f"{context}.{field}")

        layer_index = _nonnegative_int(
            record["network_layer_index"],
            context=f"{context}.network_layer_index",
        )
        soma_index = _nonnegative_int(
            record["target_soma_index"], context=f"{context}.target_soma_index"
        )
        n_soma = _nonnegative_int(record["n_soma"], context=f"{context}.n_soma")
        if n_soma < 1 or soma_index >= n_soma:
            raise ValueError(f"{context} target soma index is outside n_soma")
        source_start = _nonnegative_int(
            record["source_start"], context=f"{context}.source_start"
        )
        source_stop = _nonnegative_int(
            record["source_stop"], context=f"{context}.source_stop"
        )
        source_dim = _nonnegative_int(
            record["source_dim"], context=f"{context}.source_dim"
        )
        if source_dim < 1 or source_stop - source_start != source_dim:
            raise ValueError(f"{context} source slice is inconsistent")
        if int(record["n_source_coordinates"]) != source_dim:
            raise ValueError(f"{context} source coordinate count is inconsistent")
        if int(record["n_eligible_source_coordinates"]) > source_dim:
            raise ValueError(f"{context} eligible coordinate count exceeds source_dim")
        n_classes = int(record["n_class_profile_classes"])
        if n_classes < 2 or n_classes != len(evaluation_counts):
            raise ValueError(f"{context} class-profile class count is inconsistent")
        if int(record["n_degenerate_target_class_profiles"]) > 1:
            raise ValueError(f"{context} has more than one target class profile")
        n_degenerate_sources = int(record["n_degenerate_source_class_profiles"])
        if n_degenerate_sources > source_dim:
            raise ValueError(f"{context} source degeneracy count exceeds source_dim")
        n_eligible_degenerate = int(
            record["n_eligible_degenerate_source_class_profiles"]
        )
        if n_eligible_degenerate > int(record["n_eligible_source_coordinates"]):
            raise ValueError(
                f"{context} eligible degeneracy count exceeds eligible coordinates"
            )
        if n_eligible_degenerate > n_degenerate_sources:
            raise ValueError(
                f"{context} eligible degeneracy count exceeds source degeneracies"
            )
        class_alignment_fields = [
            f"{weighting}_class_profile_alignment" for weighting in _WEIGHTINGS
        ]
        for field in class_alignment_fields:
            value = record[field]
            if value is not None and not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"{context}.{field} is outside [-1, 1]")
        if int(record["n_degenerate_target_class_profiles"]) == 1:
            if any(
                record[field] is not None
                and not math.isclose(float(record[field]), 0.0, abs_tol=1e-12)
                for field in class_alignment_fields
            ):
                raise ValueError(
                    f"{context} degenerate target profile must have zero alignment"
                )
        if int(record["active_contact_count"]) > int(record["candidate_contact_count"]):
            raise ValueError(f"{context} active contacts exceed candidates")

        coordinate_space = str(record["coordinate_space"])
        if coordinate_space not in _COORDINATE_SPACES:
            raise ValueError(f"{context} has unknown coordinate_space")
        if layer_index > 0 and coordinate_space == "image_pixels":
            raise ValueError(
                f"{context} labels a later-layer native source as image pixels"
            )
        _validate_image_shape(
            record["image_shape"],
            coordinate_space=coordinate_space,
            source_dim=source_dim,
            context=f"{context}.image_shape",
        )
        expected_threshold_kind = (
            "image_binarization"
            if coordinate_space == "image_pixels"
            else "continuous_activity"
        )
        if record["activity_threshold_kind"] != expected_threshold_kind:
            raise ValueError(f"{context} activity threshold kind is inconsistent")
        if record["pathway"] not in {"ff_excitatory", "ff_inhibitory"}:
            raise ValueError(f"{context} has unsupported pathway")

        target_identity = (str(record["target_key"]), soma_index)
        preferred_class = _nonnegative_int(
            record["preferred_class"], context=f"{context}.preferred_class"
        )
        if preferences.get(target_identity) != preferred_class:
            raise ValueError(f"{context} does not match reference preference")
        if preferred_class not in evaluation_counts:
            raise ValueError(f"{context} preferred class is absent from evaluation")
        if (
            int(record["n_preferred_evaluation_samples"])
            != evaluation_counts[preferred_class]
        ):
            raise ValueError(f"{context} preferred evaluation count is inconsistent")
        if int(record["n_rest_evaluation_samples"]) != (
            evaluation_total - evaluation_counts[preferred_class]
        ):
            raise ValueError(f"{context} rest evaluation count is inconsistent")

        profile_key = str(record["profile_key"])
        if not profile_key or profile_key in seen_profiles:
            raise ValueError(f"{context} profile_key is blank or duplicated")
        seen_profiles.add(profile_key)
    return records


def _validate_coordinate_profiles(
    profiles: Any,
    *,
    alignment_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(profiles, Mapping):
        raise TypeError("coordinate_profiles must be a mapping")
    if not profiles:
        return {}
    records_by_profile = {
        str(record["profile_key"]): record for record in alignment_records
    }
    if set(profiles) != set(records_by_profile):
        raise ValueError(
            "coordinate profile keys must exactly match scalar alignment records"
        )
    for profile_key, profile in profiles.items():
        context = f"coordinate_profiles[{profile_key!r}]"
        if not isinstance(profile, Mapping):
            raise TypeError(f"{context} must be a mapping")
        missing = {"coordinate_axis", "source_dim", *_PROFILE_VECTOR_FIELDS} - set(
            profile
        )
        if missing:
            raise ValueError(f"{context} lacks fields: {sorted(missing)}")
        source_dim = _nonnegative_int(
            profile["source_dim"], context=f"{context}.source_dim"
        )
        record = records_by_profile[str(profile_key)]
        if source_dim != int(record["source_dim"]):
            raise ValueError(f"{context} source_dim differs from scalar record")
        if profile["coordinate_axis"] != record["coordinate_space"]:
            raise ValueError(f"{context} coordinate axis differs from scalar record")
        for field in _PROFILE_VECTOR_FIELDS:
            vector = profile[field]
            if not isinstance(vector, list) or len(vector) != source_dim:
                raise ValueError(f"{context}.{field} must have length {source_dim}")
            for coordinate, value in enumerate(vector):
                _finite_or_none(
                    value,
                    context=f"{context}.{field}[{coordinate}]",
                )
    return profiles


def _validate_result(
    result: Any,
    *,
    result_path: Path,
) -> tuple[
    list[Mapping[str, Any]],
    list[Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
]:
    if not isinstance(result, Mapping):
        raise TypeError(f"source-tuning result is not a mapping: {result_path}")
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unexpected source-tuning result schema: {result_path}")
    if result.get("analysis_type") != ANALYZER_NAME:
        raise ValueError(f"unexpected source-tuning analysis type: {result_path}")
    contract = result.get("quantity_contract")
    required_contract = {
        "preferred_class",
        "source_tuning",
        "class_profile_alignment",
        "class_profile_degeneracy",
        "source_uniform",
        "contact_weighted",
        "conductance_weighted",
        "image_coordinates",
        "hidden_coordinates",
        "stored_state",
    }
    if not isinstance(contract, Mapping) or not required_contract.issubset(contract):
        raise ValueError(
            f"source-tuning quantity contract is incomplete: {result_path}"
        )
    reference = result.get("reference")
    evaluation = result.get("evaluation")
    if not isinstance(reference, Mapping) or not isinstance(evaluation, Mapping):
        raise TypeError("source-tuning split summaries must be mappings")
    reference_split = str(reference.get("split", ""))
    evaluation_split = str(evaluation.get("split", ""))
    require_independent_splits = result.get("require_independent_splits")
    if not isinstance(require_independent_splits, bool):
        raise TypeError("require_independent_splits must be Boolean")
    if not reference_split or not evaluation_split:
        raise ValueError("reference and evaluation splits must be named")
    if require_independent_splits and reference_split == evaluation_split:
        raise ValueError(
            "reference and evaluation splits must be distinct under the declared "
            "independence policy"
        )
    reference_counts = _class_counts(
        reference.get("class_counts"), context="reference.class_counts"
    )
    evaluation_counts = _class_counts(
        evaluation.get("class_counts"), context="evaluation.class_counts"
    )
    if set(reference_counts) != set(evaluation_counts):
        raise ValueError(
            "reference and evaluation class sets must match for class-profile alignment"
        )
    if _nonnegative_int(reference.get("n_samples"), context="reference.n_samples") != (
        sum(reference_counts.values())
    ):
        raise ValueError("reference sample count is inconsistent")
    if _nonnegative_int(
        evaluation.get("n_samples"), context="evaluation.n_samples"
    ) != sum(evaluation_counts.values()):
        raise ValueError("evaluation sample count is inconsistent")
    preferences, preference_index = _validate_preference_records(
        reference.get("preference_records"), reference_counts=reference_counts
    )
    alignments = _validate_alignment_records(
        result.get("alignment_records"),
        declared_count=result.get("n_alignment_records"),
        preferences=preference_index,
        evaluation_counts=evaluation_counts,
    )
    profiles = _validate_coordinate_profiles(
        result.get("coordinate_profiles"), alignment_records=alignments
    )
    return preferences, alignments, profiles


def _result_path(
    run_dir: Path,
    *,
    result_subdirectory: str,
    result_filename: str,
) -> Path:
    base = run_dir / result_subdirectory
    candidates = [
        base / result_filename,
        base / f"{result_filename}.json",
        base / "epochs" / f"{result_filename}.json",
    ]
    existing = [path for path in candidates if path.is_file()]
    if len(existing) != 1:
        if not existing:
            raise FileNotFoundError(candidates[0])
        raise ValueError(f"ambiguous source-tuning result files: {existing}")
    return existing[0]


def _checkpoint_alignment_summary_rows(
    alignment_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Reduce technical soma records within each fitted checkpoint.

    Every target soma contributes equally after any source-coordinate weighting
    encoded by the analyzer's metric prefix.  Somas are never treated as
    independent trained replicates, and no aggregation across jobs or training
    seeds occurs here.
    """
    groups: dict[tuple[tuple[str, Any], ...], list[Mapping[str, Any]]] = {}
    for row in alignment_rows:
        encoded_row = _csv_scalar_record(row)
        job_fields = {
            key
            for key in encoded_row
            if key == "job_index" or str(key).startswith("job_")
        }
        # Preserve operator semantics in new outputs while accepting historical
        # records that predate this optional metadata.
        operator_fields = {
            "selection_policy",
            "transmission_mode",
            "signal_rounding",
        } & set(encoded_row)
        identity_fields = (
            job_fields | _CHECKPOINT_ALIGNMENT_STRUCTURAL_FIELDS | operator_fields
        )
        missing = identity_fields - set(encoded_row)
        if missing:
            raise ValueError(
                "source-tuning checkpoint summary lacks identity fields: "
                f"{sorted(missing)}"
            )
        identity = tuple((key, encoded_row[key]) for key in sorted(identity_fields))
        groups.setdefault(identity, []).append(row)

    summaries: list[dict[str, Any]] = []
    for identity, rows in groups.items():
        summary = dict(identity)
        n_soma = int(summary["n_soma"])
        soma_indices = [int(row["target_soma_index"]) for row in rows]
        if len(soma_indices) != len(set(soma_indices)):
            raise ValueError(
                "source-tuning checkpoint group contains duplicate target somas"
            )
        if len(rows) != n_soma or set(soma_indices) != set(range(n_soma)):
            raise ValueError(
                "source-tuning checkpoint group must contain exactly one record "
                "for every target soma"
            )

        summary.update(
            {
                "technical_soma_count": n_soma,
                "technical_aggregation": (
                    "equal_mean_across_target_somas_within_checkpoint"
                ),
                "soma_weighting": "equal",
                "source_coordinate_weighting": "encoded_by_metric_prefix",
                "technical_somas_are_independent_replicates": False,
                "trained_checkpoint_is_inference_unit": True,
            }
        )
        for field in sorted(_CHECKPOINT_ALIGNMENT_SUMMARY_FIELDS):
            values = [float(row[field]) for row in rows if row.get(field) is not None]
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


def collect_source_tuning_support(
    *,
    jobs_paths: Sequence[Path],
    output_dir: Path,
    expected_count: int | None = None,
    result_subdirectory: str = RESULT_SUBDIRECTORY,
    result_filename: str = RESULT_FILENAME,
    input_hash_validation: str = "rehash",
) -> dict[str, Any]:
    """Validate and collect checkpoint-bound source-tuning scalar records."""
    hash_validation = str(input_hash_validation).strip().lower()
    if hash_validation not in INPUT_HASH_VALIDATION_MODES:
        raise ValueError(
            "input_hash_validation must be 'rehash' or 'recorded_hash_chain'"
        )
    result_subdirectory = str(result_subdirectory).strip()
    result_filename = str(result_filename).strip()
    if not result_subdirectory or not result_filename:
        raise ValueError("result subdirectory and filename must be nonempty")

    jobs = _read_jobs(jobs_paths)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(f"declared {len(jobs)} jobs; expected {expected_count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    preference_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for job_index, job in enumerate(jobs):
        run_dir = Path(str(job["output_dir"])).resolve()
        provenance_path = run_dir / "checkpoint_analysis_provenance.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(provenance_path)
        result_path = _result_path(
            run_dir,
            result_subdirectory=result_subdirectory,
            result_filename=result_filename,
        )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        _validate_provenance(
            job,
            provenance,
            hash_cache,
            input_hash_validation=hash_validation,
        )
        enabled = {str(name) for name in provenance.get("enabled_analyzers", [])}
        if ANALYZER_NAME not in enabled:
            raise ValueError(f"{ANALYZER_NAME} analyzer is absent from run provenance")
        require_completed_analyses(provenance, [ANALYZER_NAME], run_dir=run_dir)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        preferences, alignments, profiles = _validate_result(
            result, result_path=result_path
        )
        base = _job_fields(job_index, job)
        for record in preferences:
            preference_rows.append(
                {
                    **base,
                    **record,
                    "technical_soma_record_is_independent_replicate": False,
                    "trained_checkpoint_is_inference_unit": True,
                }
            )
        for record in alignments:
            alignment_rows.append(
                {
                    **base,
                    **_csv_scalar_record(record),
                    "technical_alignment_record_is_independent_replicate": False,
                    "trained_checkpoint_is_inference_unit": True,
                }
            )
        source_rows.append(
            {
                **base,
                "result_path": str(result_path),
                "result_sha256": _sha256(result_path),
                "provenance_path": str(provenance_path),
                "provenance_sha256": _sha256(provenance_path),
                "config_sha256": provenance["config_sha256"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "analysis_config_sha256": provenance["analysis_config_sha256"],
                "run_code_identity": json.dumps(
                    provenance.get("code_identity"),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "reference_split": result["reference"]["split"],
                "evaluation_split": result["evaluation"]["split"],
                "preference_record_count": len(preferences),
                "alignment_record_count": len(alignments),
                "embedded_coordinate_profile_count": len(profiles),
                "trained_checkpoint_is_inference_unit": True,
            }
        )

    outputs = {
        "checkpoint_source_tuning_preference_records.csv": preference_rows,
        "checkpoint_source_tuning_alignment_records.csv": alignment_rows,
        "checkpoint_source_tuning_alignment_summary.csv": (
            _checkpoint_alignment_summary_rows(alignment_rows)
        ),
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
        "preference_record_count": len(preference_rows),
        "alignment_record_count": len(alignment_rows),
        "checkpoint_alignment_summary_record_count": len(
            outputs["checkpoint_source_tuning_alignment_summary.csv"]
        ),
        "result_subdirectory": result_subdirectory,
        "result_filename": result_filename,
        "input_hash_validation": hash_validation,
        "aggregation_policy": "equal_soma_mean_within_checkpoint_only",
        "within_checkpoint_soma_weighting": "equal",
        "source_coordinate_weighting": "encoded_by_metric_prefix",
        "technical_soma_aggregation_performed": True,
        "trained_seed_aggregation_performed": False,
        "trained_checkpoint_is_inference_unit": True,
        "soma_depth_pathway_source_records_are_technical": True,
        "collector_code_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
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
    if not isinstance(payload, Mapping) or payload.get("schema_version") != (
        CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported source-tuning collection config: {path}")
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
        "result_subdirectory": str(
            payload.get("result_subdirectory", RESULT_SUBDIRECTORY)
        ),
        "result_filename": str(payload.get("result_filename", RESULT_FILENAME)),
        "input_hash_validation": str(payload.get("input_hash_validation", "rehash")),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        help="Collection configuration file; replaces the other options",
    )
    parser.add_argument(
        "--jobs",
        action="append",
        type=Path,
        help="Job file (JSON Lines) of the analyzed checkpoints; repeat for several files",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Directory to write the collected tables into"
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Number of declared jobs the collection must contain",
    )
    parser.add_argument(
        "--result-subdirectory",
        default=RESULT_SUBDIRECTORY,
        help="Analyzer directory, inside each job's output directory, to read",
    )
    parser.add_argument(
        "--result-filename",
        default=RESULT_FILENAME,
        help="Result name to read in that directory",
    )
    parser.add_argument(
        "--input-hash-validation",
        choices=sorted(INPUT_HASH_VALIDATION_MODES),
        default="rehash",
        help="rehash: hash the input files again; recorded_hash_chain: compare the job file's *_sha256 values with the provenance",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    if args.config is not None:
        explicit = any(
            (
                args.jobs,
                args.output_dir is not None,
                args.expected_count is not None,
                args.result_subdirectory != RESULT_SUBDIRECTORY,
                args.result_filename != RESULT_FILENAME,
                args.input_hash_validation != "rehash",
            )
        )
        if explicit:
            raise ValueError("--config cannot be combined with explicit options")
        return collect_source_tuning_support(**_load_collection_config(args.config))
    if not args.jobs or args.output_dir is None:
        raise ValueError("--jobs and --output-dir are required without --config")
    return collect_source_tuning_support(
        jobs_paths=args.jobs,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
        result_subdirectory=args.result_subdirectory,
        result_filename=args.result_filename,
        input_hash_validation=args.input_hash_validation,
    )


def cli_main(argv: list[str] | None = None) -> int:
    """Console entry point: run :func:`main` and exit 0 on success."""
    setup_logging()
    main(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())


__all__ = [
    "ANALYZER_NAME",
    "CONFIG_SCHEMA_VERSION",
    "RESULT_FILENAME",
    "RESULT_SCHEMA_VERSION",
    "RESULT_SUBDIRECTORY",
    "SCHEMA_VERSION",
    "collect_source_tuning_support",
    "main",
]
