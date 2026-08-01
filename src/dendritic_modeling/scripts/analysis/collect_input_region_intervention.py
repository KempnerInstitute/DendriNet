"""Collect frozen-checkpoint input-region intervention summaries.

This collector consumes ``input_region_intervention/final.json`` artifacts from
standard checkpoint-analysis jobs.  It emits separate checkpoint-bound tables
for classification, aggregate responses, per-class responses, assigned-soma
responses, and assigned-soma per-class responses.  It validates that results
contain aggregate sufficient statistics only and performs no aggregation over
fitted checkpoints or trained seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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
)
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity

SCHEMA_VERSION = "input_region_intervention_collection_v1"
CONFIG_SCHEMA_VERSION = "input_region_intervention_collection_config_v1"
RESULT_SCHEMA_VERSION = 1
ANALYZER_NAME = "input_region_intervention"
RESULT_SUBDIRECTORY = "input_region_intervention"
RESULT_FILENAME = "final.json"
_SUMMARY_ORDER_ABS_TOLERANCE = 1e-12

_TOP_LEVEL_FIELDS = {
    "schema_version",
    "analysis_type",
    "analysis_contract",
    "image_shape_channels_height_width",
    "evaluation_split",
    "evaluation_max_samples",
    "region_definitions",
    "operations",
    "replacement_methods",
    "branch_target_population_polarities",
    "replacement_semantics",
    "baseline",
    "interventions",
    "response_interpretation",
    "serialization_policy",
    "reference",
    "identity_soma_mapping",
}
_INTERVENTION_LIST_FIELDS = {
    "response_records",
    "response_class_records",
    "assigned_soma_response_records",
    "assigned_soma_class_response_records",
}
_REGION_FIELDS = {
    "region_name",
    "region_kind",
    "mask_scope",
    "sample_dependent",
    "center_fraction",
    "foreground_threshold",
    "foreground_rule",
    "background_rule",
}
_INTERVENTION_FIELDS = _REGION_FIELDS | {
    "operation",
    "operation_semantics",
    "replacement_method",
    "classification",
    "named_region_feature_fraction",
    "replaced_feature_fraction",
    *_INTERVENTION_LIST_FIELDS,
}
_CLASSIFICATION_FIELDS = {
    "accuracy",
    "accuracy_change",
    "accuracy_drop",
    "mean_negative_log_likelihood",
    "mean_negative_log_likelihood_change",
}
_BASELINE_FIELDS = {
    "sample_count",
    "accuracy",
    "mean_negative_log_likelihood",
}
_FRACTION_FIELDS = {
    "sample_count",
    "mean",
    "standard_deviation",
    "minimum",
    "maximum",
}
_RESPONSE_METRIC_FIELDS = {
    "sample_count",
    "value_count",
    "baseline_mean",
    "intervention_mean",
    "mean_signed_change",
    "mean_absolute_change",
    "root_mean_square_change",
    "relative_l2_change",
}
_MECHANISM_FIELDS = {
    "integration_mechanism",
    "use_shunting",
    "additive_mode",
    "reactivation_enabled",
    "reactivation_type",
    "inhibitory_drive_semantics",
    "pre_gate_voltage_semantics",
    "post_gate_output_semantics",
}
_BRANCH_RESPONSE_TYPES = {"branch_current", "branch_voltage", "branch_output"}
_ALL_RESPONSE_TYPES = {
    "soma_population_output",
    "true_class_assigned_soma_output",
    *_BRANCH_RESPONSE_TYPES,
}
_FORBIDDEN_SAMPLE_KEYS = {
    "activation_samples",
    "activations",
    "labels",
    "logits",
    "per_sample_records",
    "predictions",
    "raw_samples",
    "sample_activations",
    "sample_by_synapse",
    "sample_records",
    "sample_values",
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


def _finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


def _nonnegative_int(value: Any, *, context: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    minimum = 1 if positive else 0
    if value < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context} must be {qualifier}")
    return int(value)


def _validate_scalar_record(
    record: Mapping[str, Any],
    *,
    context: str,
    allowed_list_fields: Sequence[str] = (),
) -> None:
    allowed_lists = set(allowed_list_fields)
    for field, value in record.items():
        if field in allowed_lists:
            if not isinstance(value, list):
                raise TypeError(f"{context}.{field} must be a list")
            continue
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"{context}.{field} must be scalar or null")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{context}.{field} must be finite")


def _reject_sample_payloads(value: Any, *, path: str = "result") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            if key_text.lower() in _FORBIDDEN_SAMPLE_KEYS:
                raise ValueError(
                    f"sample-level payload is forbidden at {path}.{key_text}"
                )
            _reject_sample_payloads(child, path=f"{path}.{key_text}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sample_payloads(child, path=f"{path}[{index}]")


def _validate_classification_summary(
    summary: Any,
    *,
    required_fields: set[str],
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping) or set(summary) != required_fields:
        raise ValueError(f"{context} has an unexpected classification schema")
    _validate_scalar_record(summary, context=context)
    if "sample_count" in summary:
        _nonnegative_int(
            summary["sample_count"], context=f"{context}.sample_count", positive=True
        )
    accuracy = _finite(summary["accuracy"], context=f"{context}.accuracy")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError(f"{context}.accuracy must lie in [0, 1]")
    nll = _finite(
        summary["mean_negative_log_likelihood"],
        context=f"{context}.mean_negative_log_likelihood",
    )
    if nll < 0:
        raise ValueError(f"{context} negative log likelihood must be non-negative")
    return summary


def _validate_fraction_summary(
    summary: Any,
    *,
    expected_samples: int,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(summary, Mapping) or set(summary) != _FRACTION_FIELDS:
        raise ValueError(f"{context} has an unexpected fraction schema")
    count = _nonnegative_int(
        summary["sample_count"], context=f"{context}.sample_count", positive=True
    )
    if count != expected_samples:
        raise ValueError(f"{context} sample count differs from baseline")
    values = {
        field: _finite(summary[field], context=f"{context}.{field}")
        for field in _FRACTION_FIELDS - {"sample_count"}
    }
    if values["standard_deviation"] < 0:
        raise ValueError(f"{context} standard deviation must be non-negative")
    if not all(0.0 <= values[field] <= 1.0 for field in ("mean", "minimum", "maximum")):
        raise ValueError(f"{context} coverage statistics must lie in [0, 1]")
    minimum_above_mean = values["minimum"] > values["mean"] and not math.isclose(
        values["minimum"],
        values["mean"],
        rel_tol=0.0,
        abs_tol=_SUMMARY_ORDER_ABS_TOLERANCE,
    )
    mean_above_maximum = values["mean"] > values["maximum"] and not math.isclose(
        values["mean"],
        values["maximum"],
        rel_tol=0.0,
        abs_tol=_SUMMARY_ORDER_ABS_TOLERANCE,
    )
    if minimum_above_mean or mean_above_maximum:
        raise ValueError(f"{context} minimum/mean/maximum ordering is invalid")
    return summary


def _validate_region_metadata(record: Mapping[str, Any], *, context: str) -> None:
    required = {"region_name", "region_kind", "mask_scope", "sample_dependent"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"{context} lacks region fields: {sorted(missing)}")
    kind = str(record["region_kind"])
    if kind not in {"central_square", "surround", "foreground", "background"}:
        raise ValueError(f"{context} has unknown region_kind")
    sample_dependent = record["sample_dependent"]
    if not isinstance(sample_dependent, bool):
        raise TypeError(f"{context}.sample_dependent must be boolean")
    expected_dependent = kind in {"foreground", "background"}
    if sample_dependent is not expected_dependent:
        raise ValueError(f"{context} sample-dependent declaration is inconsistent")
    expected_scope = (
        "sample_dependent_intensity" if expected_dependent else "fixed_spatial"
    )
    if record["mask_scope"] != expected_scope:
        raise ValueError(f"{context} mask scope is inconsistent")
    if expected_dependent:
        required_dependent = {
            "foreground_threshold",
            "foreground_rule",
            "background_rule",
        }
        if not required_dependent.issubset(record):
            raise ValueError(f"{context} lacks intensity-mask metadata")
        _finite(
            record["foreground_threshold"],
            context=f"{context}.foreground_threshold",
        )
    else:
        if "center_fraction" not in record:
            raise ValueError(f"{context} lacks center_fraction")
        center_fraction = _finite(
            record["center_fraction"], context=f"{context}.center_fraction"
        )
        if not 0.0 < center_fraction <= 1.0:
            raise ValueError(f"{context}.center_fraction must lie in (0, 1]")


def _response_site_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(
        record.get(field)
        for field in (
            "response_type",
            "signal",
            "module_name",
            "network_layer_index",
            "population_name",
            "source_current_polarity",
            "soma_relative_dendritic_depth",
        )
    )


def _validate_response_records(
    records: Any,
    *,
    table_kind: str,
    evaluation_samples: int,
    context: str,
) -> list[Mapping[str, Any]]:
    if not isinstance(records, list):
        raise TypeError(f"{context} must be a list")
    assigned = table_kind in {"assigned_soma", "assigned_soma_per_class"}
    per_class = table_kind in {"per_class", "assigned_soma_per_class"}
    seen: set[tuple[Any, ...]] = set()
    validated: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        item_context = f"{context}[{index}]"
        if not isinstance(record, Mapping):
            raise TypeError(f"{item_context} must be a mapping")
        _validate_scalar_record(
            record,
            context=item_context,
            allowed_list_fields=("identity_soma_class_mapping",),
        )
        missing = (
            _RESPONSE_METRIC_FIELDS
            | _MECHANISM_FIELDS
            | {
                "response_type",
                "signal",
                "module_name",
                "network_layer_index",
                "network_layer_number",
                "population_name",
                "population_polarity",
            }
        ) - set(record)
        if missing:
            raise ValueError(f"{item_context} lacks fields: {sorted(missing)}")
        response_type = str(record["response_type"])
        if response_type not in _ALL_RESPONSE_TYPES:
            raise ValueError(f"{item_context} has unknown response_type")
        if assigned != (response_type == "true_class_assigned_soma_output"):
            raise ValueError(f"{item_context} assigned-soma response type mismatch")
        expected_scope = (
            "true_class_assigned_soma_within_class"
            if table_kind == "assigned_soma_per_class"
            else (
                "true_class_assigned_soma"
                if table_kind == "assigned_soma"
                else "within_true_class" if table_kind == "per_class" else None
            )
        )
        if expected_scope is None:
            if "aggregation_scope" in record or "class_label" in record:
                raise ValueError(f"{item_context} aggregate response has class fields")
        elif record.get("aggregation_scope") != expected_scope:
            raise ValueError(f"{item_context} aggregation_scope is inconsistent")
        if per_class:
            _nonnegative_int(
                record.get("class_label"), context=f"{item_context}.class_label"
            )
        elif "class_label" in record:
            raise ValueError(f"{item_context} unexpectedly contains class_label")
        if assigned:
            mapping = record.get("identity_soma_class_mapping")
            if not isinstance(mapping, list) or not mapping:
                raise ValueError(f"{item_context} lacks identity soma mapping")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in mapping
            ):
                raise ValueError(f"{item_context} identity soma mapping is invalid")

        sample_count = _nonnegative_int(
            record["sample_count"],
            context=f"{item_context}.sample_count",
            positive=True,
        )
        value_count = _nonnegative_int(
            record["value_count"], context=f"{item_context}.value_count", positive=True
        )
        if sample_count > evaluation_samples or value_count < sample_count:
            raise ValueError(f"{item_context} response counts are inconsistent")
        baseline_mean = _finite(
            record["baseline_mean"], context=f"{item_context}.baseline_mean"
        )
        intervention_mean = _finite(
            record["intervention_mean"],
            context=f"{item_context}.intervention_mean",
        )
        signed_change = _finite(
            record["mean_signed_change"],
            context=f"{item_context}.mean_signed_change",
        )
        if not math.isclose(
            signed_change,
            intervention_mean - baseline_mean,
            rel_tol=1e-8,
            abs_tol=1e-10,
        ):
            raise ValueError(f"{item_context} signed response change is inconsistent")
        for field in (
            "mean_absolute_change",
            "root_mean_square_change",
            "relative_l2_change",
        ):
            if _finite(record[field], context=f"{item_context}.{field}") < 0:
                raise ValueError(f"{item_context}.{field} must be non-negative")

        mechanism = str(record["integration_mechanism"])
        if mechanism not in {"shunting", "additive"}:
            raise ValueError(f"{item_context} has unknown integration mechanism")
        if not isinstance(record["use_shunting"], bool):
            raise TypeError(f"{item_context}.use_shunting must be boolean")
        if (mechanism == "shunting") != bool(record["use_shunting"]):
            raise ValueError(f"{item_context} mechanism metadata are inconsistent")
        layer_index = _nonnegative_int(
            record["network_layer_index"],
            context=f"{item_context}.network_layer_index",
        )
        if (
            _nonnegative_int(
                record["network_layer_number"],
                context=f"{item_context}.network_layer_number",
                positive=True,
            )
            != layer_index + 1
        ):
            raise ValueError(f"{item_context} layer number is inconsistent")
        if response_type in _BRANCH_RESPONSE_TYPES:
            branch_required = {
                "response_stage",
                "target_population_name",
                "target_population_polarity",
                "soma_relative_dendritic_depth",
                "dendritic_depth_reference",
            }
            missing_branch = branch_required - set(record)
            if missing_branch:
                raise ValueError(
                    f"{item_context} lacks branch fields: {sorted(missing_branch)}"
                )
            _nonnegative_int(
                record["soma_relative_dendritic_depth"],
                context=f"{item_context}.soma_relative_dendritic_depth",
            )
            if record["dendritic_depth_reference"] != "soma_relative":
                raise ValueError(f"{item_context} depth reference is not soma-relative")
            if response_type == "branch_current":
                if record.get("source_current_polarity") not in {
                    "excitatory",
                    "inhibitory",
                }:
                    raise ValueError(f"{item_context} lacks source-current polarity")
        identity = _response_site_key(record)
        if per_class:
            identity = (*identity, int(record["class_label"]))
        if identity in seen:
            raise ValueError(f"{item_context} duplicates a response site")
        seen.add(identity)
        validated.append(record)
    return validated


def _validate_class_partition(
    aggregate: Sequence[Mapping[str, Any]],
    per_class: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> None:
    aggregate_by_site = {_response_site_key(record): record for record in aggregate}
    classes_by_site: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for record in per_class:
        classes_by_site[_response_site_key(record)].append(record)
    if set(aggregate_by_site) != set(classes_by_site):
        raise ValueError(f"{context} class-response sites do not match aggregates")
    for site, aggregate_record in aggregate_by_site.items():
        class_records = classes_by_site[site]
        if sum(int(record["sample_count"]) for record in class_records) != int(
            aggregate_record["sample_count"]
        ):
            raise ValueError(f"{context} per-class sample counts do not partition site")
        if sum(int(record["value_count"]) for record in class_records) != int(
            aggregate_record["value_count"]
        ):
            raise ValueError(f"{context} per-class value counts do not partition site")


def _validate_result(result: Any, *, result_path: Path) -> Mapping[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError(f"input-region result is not a mapping: {result_path}")
    _reject_sample_payloads(result)
    unknown_top = set(result) - _TOP_LEVEL_FIELDS
    missing_top = (_TOP_LEVEL_FIELDS - {"reference", "identity_soma_mapping"}) - set(
        result
    )
    if unknown_top or missing_top:
        raise ValueError(
            f"input-region top-level schema mismatch: missing={sorted(missing_top)}, "
            f"unknown={sorted(unknown_top)}"
        )
    if result.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"unexpected input-region result schema: {result_path}")
    if result.get("analysis_type") != ANALYZER_NAME:
        raise ValueError(f"unexpected analysis type: {result_path}")
    serialization = str(result.get("serialization_policy", "")).lower()
    if "no per-sample" not in serialization or "aggregate paired metrics" not in (
        serialization
    ):
        raise ValueError("input-region serialization policy is not aggregate-only")

    image_shape = result["image_shape_channels_height_width"]
    if (
        not isinstance(image_shape, list)
        or len(image_shape) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in image_shape
        )
    ):
        raise ValueError("image_shape_channels_height_width must be positive C,H,W")
    evaluation_split = str(result["evaluation_split"])
    if not evaluation_split:
        raise ValueError("evaluation_split must be named")
    baseline = _validate_classification_summary(
        result["baseline"], required_fields=_BASELINE_FIELDS, context="baseline"
    )
    evaluation_samples = int(baseline["sample_count"])

    region_definitions = result["region_definitions"]
    if not isinstance(region_definitions, list) or not region_definitions:
        raise ValueError("region_definitions must be a nonempty list")
    regions_by_name: dict[str, Mapping[str, Any]] = {}
    for index, region in enumerate(region_definitions):
        if not isinstance(region, Mapping):
            raise TypeError(f"region_definitions[{index}] must be a mapping")
        _validate_scalar_record(region, context=f"region_definitions[{index}]")
        _validate_region_metadata(region, context=f"region_definitions[{index}]")
        name = str(region["region_name"])
        if name in regions_by_name:
            raise ValueError(f"duplicate region definition {name!r}")
        regions_by_name[name] = region

    operations = result["operations"]
    replacements = result["replacement_methods"]
    if not isinstance(operations, list) or not operations:
        raise ValueError("operations must be a nonempty list")
    if not isinstance(replacements, list) or not replacements:
        raise ValueError("replacement_methods must be a nonempty list")
    interventions = result["interventions"]
    if not isinstance(interventions, list) or not interventions:
        raise ValueError("interventions must be a nonempty list")
    seen_conditions: set[tuple[str, str, str]] = set()
    for index, intervention in enumerate(interventions):
        context = f"interventions[{index}]"
        if not isinstance(intervention, Mapping):
            raise TypeError(f"{context} must be a mapping")
        unknown = set(intervention) - _INTERVENTION_FIELDS
        if unknown:
            raise ValueError(f"{context} has unknown fields: {sorted(unknown)}")
        _validate_region_metadata(intervention, context=context)
        region_name = str(intervention["region_name"])
        definition = regions_by_name.get(region_name)
        if definition is None:
            raise ValueError(f"{context} references an undeclared region")
        if any(
            intervention.get(field) != definition.get(field) for field in _REGION_FIELDS
        ):
            raise ValueError(f"{context} region metadata differs from its definition")
        operation = str(intervention.get("operation"))
        replacement = str(intervention.get("replacement_method"))
        if operation not in operations or replacement not in replacements:
            raise ValueError(f"{context} uses an undeclared operation or replacement")
        condition = (region_name, operation, replacement)
        if condition in seen_conditions:
            raise ValueError(f"{context} duplicates an intervention condition")
        seen_conditions.add(condition)

        classification = _validate_classification_summary(
            intervention.get("classification"),
            required_fields=_CLASSIFICATION_FIELDS,
            context=f"{context}.classification",
        )
        expected_change = float(classification["accuracy"]) - float(
            baseline["accuracy"]
        )
        if not math.isclose(
            float(classification["accuracy_change"]),
            expected_change,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ) or not math.isclose(
            float(classification["accuracy_drop"]),
            -expected_change,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{context} accuracy contrasts are inconsistent")
        expected_nll_change = float(
            classification["mean_negative_log_likelihood"]
        ) - float(baseline["mean_negative_log_likelihood"])
        if not math.isclose(
            float(classification["mean_negative_log_likelihood_change"]),
            expected_nll_change,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{context} NLL contrast is inconsistent")
        _validate_fraction_summary(
            intervention.get("named_region_feature_fraction"),
            expected_samples=evaluation_samples,
            context=f"{context}.named_region_feature_fraction",
        )
        _validate_fraction_summary(
            intervention.get("replaced_feature_fraction"),
            expected_samples=evaluation_samples,
            context=f"{context}.replaced_feature_fraction",
        )
        aggregate = _validate_response_records(
            intervention.get("response_records"),
            table_kind="aggregate",
            evaluation_samples=evaluation_samples,
            context=f"{context}.response_records",
        )
        per_class = _validate_response_records(
            intervention.get("response_class_records"),
            table_kind="per_class",
            evaluation_samples=evaluation_samples,
            context=f"{context}.response_class_records",
        )
        assigned = _validate_response_records(
            intervention.get("assigned_soma_response_records"),
            table_kind="assigned_soma",
            evaluation_samples=evaluation_samples,
            context=f"{context}.assigned_soma_response_records",
        )
        assigned_per_class = _validate_response_records(
            intervention.get("assigned_soma_class_response_records"),
            table_kind="assigned_soma_per_class",
            evaluation_samples=evaluation_samples,
            context=f"{context}.assigned_soma_class_response_records",
        )
        _validate_class_partition(
            aggregate, per_class, context=f"{context}.response_records"
        )
        if bool(assigned) != bool(assigned_per_class):
            raise ValueError(f"{context} assigned-soma record levels are incomplete")
        if assigned:
            _validate_class_partition(
                assigned,
                assigned_per_class,
                context=f"{context}.assigned_soma_response_records",
            )

    expected_conditions = {
        (region_name, str(operation), str(replacement))
        for region_name in regions_by_name
        for operation in operations
        for replacement in replacements
    }
    if seen_conditions != expected_conditions:
        raise ValueError("interventions do not cover the declared condition factorial")
    if "reference_mean" in replacements:
        reference = result.get("reference")
        if not isinstance(reference, Mapping):
            raise ValueError("reference_mean replacement lacks reference provenance")
        if reference.get("split") == evaluation_split:
            raise ValueError("reference and evaluation splits must be distinct")
        _nonnegative_int(
            reference.get("sample_count"),
            context="reference.sample_count",
            positive=True,
        )
        digest = str(reference.get("featurewise_mean_sha256", ""))
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("reference featurewise mean hash is invalid")
    has_identity_mapping = isinstance(result.get("identity_soma_mapping"), Mapping)
    any_assigned = any(
        bool(intervention["assigned_soma_response_records"])
        for intervention in interventions
    )
    if has_identity_mapping != any_assigned:
        raise ValueError("identity soma mapping and assigned-soma records disagree")
    return result


def _csv_scalar_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            json.dumps(value, sort_keys=True, separators=(",", ":"))
            if isinstance(value, (list, dict))
            else value
        )
        for key, value in record.items()
    }


def _flatten_fraction(prefix: str, value: Mapping[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{field}": field_value for field, field_value in value.items()}


def _intervention_context(
    result: Mapping[str, Any],
    intervention: Mapping[str, Any],
    *,
    intervention_index: int,
) -> dict[str, Any]:
    baseline = result["baseline"]
    context = {
        "intervention_index": intervention_index,
        "image_shape_channels_height_width": json.dumps(
            result["image_shape_channels_height_width"], separators=(",", ":")
        ),
        "evaluation_split": result["evaluation_split"],
        "evaluation_max_samples": result["evaluation_max_samples"],
        "baseline_sample_count": baseline["sample_count"],
        "baseline_accuracy": baseline["accuracy"],
        "baseline_mean_negative_log_likelihood": baseline[
            "mean_negative_log_likelihood"
        ],
        **{field: intervention.get(field) for field in sorted(_REGION_FIELDS)},
        "operation": intervention["operation"],
        "operation_semantics": intervention["operation_semantics"],
        "replacement_method": intervention["replacement_method"],
        **_flatten_fraction(
            "named_region_feature_fraction",
            intervention["named_region_feature_fraction"],
        ),
        **_flatten_fraction(
            "replaced_feature_fraction",
            intervention["replaced_feature_fraction"],
        ),
    }
    reference = result.get("reference")
    if isinstance(reference, Mapping):
        context["reference_split"] = reference.get("split")
        context["reference_sample_count"] = reference.get("sample_count")
        context["reference_featurewise_mean_sha256"] = reference.get(
            "featurewise_mean_sha256"
        )
    return context


def _normalized_response_record(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _csv_scalar_record(record)
    normalized.setdefault("source_current_polarity", None)
    normalized.setdefault("target_population_name", record.get("population_name"))
    normalized.setdefault(
        "target_population_polarity", record.get("population_polarity")
    )
    normalized.setdefault("soma_relative_dendritic_depth", None)
    normalized.setdefault("dendritic_depth_reference", None)
    normalized.setdefault("network_layer_name", None)
    return normalized


def _flatten_tables(
    result: Mapping[str, Any],
    *,
    job_index: int,
    job: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {
        "classification": [],
        "aggregate_response": [],
        "per_class_response": [],
        "assigned_soma_response": [],
        "assigned_soma_per_class_response": [],
    }
    base = _job_fields(job_index, job)
    table_to_source = {
        "aggregate_response": "response_records",
        "per_class_response": "response_class_records",
        "assigned_soma_response": "assigned_soma_response_records",
        "assigned_soma_per_class_response": ("assigned_soma_class_response_records"),
    }
    for intervention_index, intervention in enumerate(result["interventions"]):
        context = _intervention_context(
            result,
            intervention,
            intervention_index=intervention_index,
        )
        tables["classification"].append(
            {
                **base,
                **context,
                **{
                    f"intervention_{field}": value
                    for field, value in intervention["classification"].items()
                },
                "trained_checkpoint_is_inference_unit": True,
            }
        )
        for table_name, source_name in table_to_source.items():
            for record in intervention[source_name]:
                tables[table_name].append(
                    {
                        **base,
                        **context,
                        **_normalized_response_record(record),
                        "technical_response_record_is_independent_replicate": False,
                        "trained_checkpoint_is_inference_unit": True,
                    }
                )
    return tables


def collect_input_region_intervention(
    *,
    jobs_paths: Sequence[Path],
    output_dir: Path,
    expected_count: int | None = None,
    input_hash_validation: str = "rehash",
) -> dict[str, Any]:
    """Validate and collect aggregate input-region intervention records."""
    hash_validation = str(input_hash_validation).strip().lower()
    if hash_validation not in INPUT_HASH_VALIDATION_MODES:
        raise ValueError(
            "input_hash_validation must be 'rehash' or 'recorded_hash_chain'"
        )
    jobs = _read_jobs(jobs_paths)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(f"declared {len(jobs)} jobs; expected {expected_count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    combined_tables: dict[str, list[dict[str, Any]]] = {
        "classification": [],
        "aggregate_response": [],
        "per_class_response": [],
        "assigned_soma_response": [],
        "assigned_soma_per_class_response": [],
    }
    source_rows: list[dict[str, Any]] = []

    for job_index, job in enumerate(jobs):
        run_dir = Path(str(job["output_dir"])).resolve()
        result_path = run_dir / RESULT_SUBDIRECTORY / RESULT_FILENAME
        provenance_path = run_dir / "checkpoint_analysis_provenance.json"
        if not result_path.is_file():
            raise FileNotFoundError(result_path)
        if not provenance_path.is_file():
            raise FileNotFoundError(provenance_path)
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
        result = json.loads(result_path.read_text(encoding="utf-8"))
        _validate_result(result, result_path=result_path)
        tables = _flatten_tables(result, job_index=job_index, job=job)
        for table_name, rows in tables.items():
            combined_tables[table_name].extend(rows)
        source_rows.append(
            {
                **_job_fields(job_index, job),
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
                "evaluation_split": result["evaluation_split"],
                "intervention_count": len(result["interventions"]),
                "classification_record_count": len(tables["classification"]),
                "aggregate_response_record_count": len(tables["aggregate_response"]),
                "per_class_response_record_count": len(tables["per_class_response"]),
                "assigned_soma_response_record_count": len(
                    tables["assigned_soma_response"]
                ),
                "assigned_soma_per_class_response_record_count": len(
                    tables["assigned_soma_per_class_response"]
                ),
                "sample_level_records_present": False,
                "trained_checkpoint_is_inference_unit": True,
            }
        )

    output_rows = {
        "checkpoint_classification_records.csv": combined_tables["classification"],
        "checkpoint_aggregate_response_records.csv": combined_tables[
            "aggregate_response"
        ],
        "checkpoint_per_class_response_records.csv": combined_tables[
            "per_class_response"
        ],
        "checkpoint_assigned_soma_response_records.csv": combined_tables[
            "assigned_soma_response"
        ],
        "checkpoint_assigned_soma_per_class_response_records.csv": combined_tables[
            "assigned_soma_per_class_response"
        ],
        "source_artifacts.csv": source_rows,
    }
    for filename, rows in output_rows.items():
        _write_csv(output_dir / filename, rows)
    release = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_paths": [str(path.resolve()) for path in jobs_paths],
        "jobs_sha256": [_sha256(path.resolve()) for path in jobs_paths],
        "collected_checkpoint_count": len(jobs),
        "record_counts": {
            filename: len(rows) for filename, rows in output_rows.items()
        },
        "input_hash_validation": hash_validation,
        "aggregation_policy": "none",
        "trained_seed_aggregation_performed": False,
        "trained_checkpoint_is_inference_unit": True,
        "response_records_are_within_checkpoint_technical_summaries": True,
        "sample_level_records_collected": False,
        "collector_code_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
        "outputs": {
            filename: _sha256(output_dir / filename) for filename in output_rows
        },
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
        raise ValueError(f"unsupported input-region collection config: {path}")
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
        "input_hash_validation": str(payload.get("input_hash_validation", "rehash")),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--jobs", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument(
        "--input-hash-validation",
        choices=sorted(INPUT_HASH_VALIDATION_MODES),
        default="rehash",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    if args.config is not None:
        if any(
            (
                args.jobs,
                args.output_dir is not None,
                args.expected_count is not None,
                args.input_hash_validation != "rehash",
            )
        ):
            raise ValueError("--config cannot be combined with explicit options")
        return collect_input_region_intervention(**_load_collection_config(args.config))
    if not args.jobs or args.output_dir is None:
        raise ValueError("--jobs and --output-dir are required without --config")
    return collect_input_region_intervention(
        jobs_paths=args.jobs,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
        input_hash_validation=args.input_hash_validation,
    )


if __name__ == "__main__":  # pragma: no cover
    main()


__all__ = [
    "ANALYZER_NAME",
    "CONFIG_SCHEMA_VERSION",
    "RESULT_FILENAME",
    "RESULT_SUBDIRECTORY",
    "SCHEMA_VERSION",
    "collect_input_region_intervention",
    "main",
]
