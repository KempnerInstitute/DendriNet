"""Collect generic registered-analyzer outputs from frozen checkpoints.

This collector is deliberately schema-agnostic. It validates the checkpoint-run
provenance, preserves each complete result dictionary in JSONL, and emits a long
table of finite numeric leaves. It performs no across-checkpoint inference and
never treats list elements, branches, somas, or technical draws as replicates.
Analyzers with dedicated technical-draw collectors should continue to use them.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fnmatch
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dendritic_modeling.analysis.managers.analysis_status import (
    describe_incomplete,
    incomplete_analyses,
    read_status_file,
)
from dendritic_modeling.analysis.registry import get_registered_analyzers
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity
from dendritic_modeling.utils.logging_config import setup_logging

SCHEMA_VERSION = "checkpoint_analysis_collection_v1"
CONFIG_SCHEMA_VERSION = "checkpoint_analysis_collection_config_v1"
RUN_SCHEMA_VERSION = "checkpoint_analysis_run_v1"
INPUT_HASH_VALIDATION_MODES = {"rehash", "recorded_hash_chain"}


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
    *,
    input_hash_validation: str,
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
        if input_hash_validation == "recorded_hash_chain":
            digest = job.get(f"{field}_sha256")
            if not isinstance(digest, str) or not digest:
                raise ValueError(
                    f"{field} hash is absent from the immutable job declaration"
                )
        else:
            digest = hash_cache.get(path)
            if digest is None:
                digest = _sha256(path)
                hash_cache[path] = digest
        if digest != provenance.get(f"{field}_sha256"):
            raise ValueError(f"{field} hash differs from provenance")


def incomplete_recorded_analyses(
    provenance: Mapping[str, Any],
    analyzers: Sequence[str],
    *,
    run_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """The analyzers among ``analyzers`` the run recorded as not completed.

    The record is the pass of ``analysis_status.json`` in ``run_dir`` named by
    the provenance's ``filename`` (default ``final``), which reflects the last
    pass that wrote the directory, or else the copy in the provenance's
    ``analysis_status``. Runs written before status records existed have
    neither and are not checked.
    """
    record: Any = None
    if run_dir is not None:
        pass_name = str(provenance.get("filename") or "final")
        record = read_status_file(run_dir).get(pass_name)
    if not isinstance(record, Mapping):
        record = provenance.get("analysis_status")
    if not isinstance(record, Mapping):
        return {}
    return incomplete_analyses(record, analyzers)


def require_completed_analyses(
    provenance: Mapping[str, Any],
    analyzers: Sequence[str],
    *,
    run_dir: Path | None = None,
) -> None:
    """Raise when the run recorded that an analyzer to collect did not complete.

    See :func:`incomplete_recorded_analyses`; runs without a status record are
    accepted.
    """
    incomplete = incomplete_recorded_analyses(provenance, analyzers, run_dir=run_dir)
    if incomplete:
        raise ValueError(
            "the run recorded incomplete analyses: " + describe_incomplete(incomplete)
        )


def _json_path(parent: str, key: str | int) -> str:
    if isinstance(key, int):
        return f"{parent}[{key}]"
    return f"{parent}.{key}" if parent else key


def _static_glob_prefix(pattern: str) -> str:
    positions = [position for token in "*?[" if (position := pattern.find(token)) >= 0]
    return pattern[: min(positions)] if positions else pattern


def _path_can_match(path: str, patterns: Sequence[str]) -> bool:
    if not path or not patterns:
        return True
    for pattern in patterns:
        prefix = _static_glob_prefix(pattern)
        if path.startswith(prefix) or prefix.startswith(f"{path}."):
            return True
    return False


def _numeric_leaves(
    value: Any,
    *,
    path: str = "",
    patterns: Sequence[str] = (),
) -> list[tuple[str, float]]:
    if not _path_can_match(path, patterns):
        return []
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        if patterns and not any(
            fnmatch.fnmatchcase(path, pattern) for pattern in patterns
        ):
            return []
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"non-finite numeric result at {path}")
        return [(path, number)]
    if isinstance(value, Mapping):
        leaves: list[tuple[str, float]] = []
        for key in sorted(value, key=str):
            leaves.extend(
                _numeric_leaves(
                    value[key],
                    path=_json_path(path, str(key)),
                    patterns=patterns,
                )
            )
        return leaves
    if isinstance(value, (list, tuple)):
        leaves = []
        for index, item in enumerate(value):
            leaves.extend(
                _numeric_leaves(
                    item,
                    path=_json_path(path, index),
                    patterns=patterns,
                )
            )
        return leaves
    return []


def _add_information_null_excess_in_place(value: Any) -> None:
    """Add signed observed-minus-permutation-floor information metrics.

    This backfills the package's ``*_excess`` convention when collecting
    information results written before analyzers emitted that field directly.
    It handles both per-signal keys (``I(...)`` / ``I(...)_null``) and
    branch-averaged layer-statistic keys
    (``I(...)_mean`` / ``I(...)_null_mean``).
    """

    if not isinstance(value, dict):
        return
    for child in list(value.values()):
        _add_information_null_excess_in_place(child)

    additions: dict[str, float] = {}
    for null_key, null_value in list(value.items()):
        if not isinstance(null_key, str) or not null_key.startswith("I("):
            continue
        if null_key.endswith("_null_mean"):
            observed_key = f"{null_key[:-10]}_mean"
            excess_key = f"{null_key[:-10]}_excess_mean"
        elif null_key.endswith("_null"):
            observed_key = null_key[:-5]
            excess_key = f"{observed_key}_excess"
        else:
            continue
        observed_value = value.get(observed_key)
        if excess_key in value:
            continue
        if not isinstance(observed_value, (int, float)) or isinstance(
            observed_value, bool
        ):
            continue
        if not isinstance(null_value, (int, float)) or isinstance(null_value, bool):
            continue
        observed_number = float(observed_value)
        null_number = float(null_value)
        if not (math.isfinite(observed_number) and math.isfinite(null_number)):
            raise ValueError(f"non-finite information floor pair at {null_key}")
        additions[excess_key] = observed_number - null_number
    value.update(additions)


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
    metric_patterns: Sequence[str] = (),
    include_full_results: bool = True,
    input_hash_validation: str = "rehash",
    derive_information_null_excess: bool = False,
) -> dict[str, Any]:
    """Validate and collect generic checkpoint-level analyzer results."""
    hash_validation = str(input_hash_validation).strip().lower()
    if hash_validation not in INPUT_HASH_VALIDATION_MODES:
        raise ValueError(
            "input_hash_validation must be 'rehash' or 'recorded_hash_chain'"
        )
    jobs = _read_jobs(jobs_paths)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(f"declared {len(jobs)} jobs; expected {expected_count}")

    registry = get_registered_analyzers()
    required = tuple(dict.fromkeys(str(name) for name in required_analyzers))
    patterns = tuple(dict.fromkeys(str(pattern) for pattern in metric_patterns))
    if any(not pattern for pattern in patterns):
        raise ValueError("metric_patterns cannot contain blank patterns")
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
        _validate_provenance(
            job,
            provenance,
            hash_cache,
            input_hash_validation=hash_validation,
        )

        enabled = tuple(str(name) for name in provenance.get("enabled_analyzers", []))
        missing_enabled = sorted(set(required) - set(enabled))
        if missing_enabled:
            raise ValueError(
                f"required analyzers absent from run provenance: {missing_enabled}"
            )
        selected = required or tuple(name for name in enabled if name in registry)
        if not selected:
            raise ValueError("run provenance contains no registered analyzers")
        require_completed_analyses(provenance, selected, run_dir=run_dir)

        for analyzer_name in selected:
            spec = registry[analyzer_name]
            result_path = _result_path(run_dir, spec.save_subdir)
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ValueError(f"analyzer result is not a mapping: {result_path}")
            collected_result = copy.deepcopy(result)
            if derive_information_null_excess:
                _add_information_null_excess_in_place(collected_result)
            digest = _sha256(result_path)
            base = _job_fields(job_index, job)
            numeric_leaves = _numeric_leaves(collected_result, patterns=patterns)
            if patterns and not numeric_leaves:
                raise ValueError(
                    f"metric patterns selected no values for {analyzer_name}: {result_path}"
                )
            file_rows.append(
                {
                    **base,
                    "analyzer": analyzer_name,
                    "result_path": str(result_path),
                    "result_sha256": digest,
                    "numeric_leaf_count": len(numeric_leaves),
                    "selected_numeric_leaf_count": len(numeric_leaves),
                    "provenance_path": str(provenance_path),
                    "provenance_sha256": _sha256(provenance_path),
                }
            )
            for metric_path, metric_value in numeric_leaves:
                scalar_rows.append(
                    {
                        **base,
                        "analyzer": analyzer_name,
                        "metric_path": metric_path,
                        "value": metric_value,
                        "value_role": (
                            "derived_observed_minus_permutation_floor"
                            if "_excess" in metric_path
                            else "raw_saved_numeric_leaf"
                        ),
                        "independent_replicate": False,
                    }
                )
            record = {
                "schema_version": SCHEMA_VERSION,
                "job_index": job_index,
                "job": job,
                "analyzer": analyzer_name,
                "result_path": str(result_path),
                "result_sha256": digest,
            }
            if include_full_results:
                record["result"] = collected_result
            else:
                record["selected_metrics"] = dict(numeric_leaves)
            jsonl_records.append(record)

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
        "metric_patterns": list(patterns),
        "include_full_results": bool(include_full_results),
        "derive_information_null_excess": bool(derive_information_null_excess),
        "input_hash_validation": hash_validation,
        "input_hash_validation_semantics": (
            "current input bytes were re-hashed during collection"
            if hash_validation == "rehash"
            else (
                "immutable job-declaration hashes were required to match "
                "run-time analyzer provenance; current input bytes were not re-hashed"
            )
        ),
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


def _resolve_config_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_collection_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported checkpoint-analysis collection config: {path}")
    required = {"jobs", "output_dir"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"collection config lacks required fields: {sorted(missing)}")
    jobs = payload["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("collection config jobs must be a nonempty list")
    base = path.resolve().parent
    return {
        "jobs_paths": [_resolve_config_path(value, base=base) for value in jobs],
        "output_dir": _resolve_config_path(payload["output_dir"], base=base),
        "required_analyzers": list(payload.get("required_analyzers", [])),
        "expected_count": payload.get("expected_count"),
        "metric_patterns": list(payload.get("metric_patterns", [])),
        "include_full_results": bool(payload.get("include_full_results", True)),
        "input_hash_validation": str(payload.get("input_hash_validation", "rehash")),
        "derive_information_null_excess": bool(
            payload.get("derive_information_null_excess", False)
        ),
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
        "--require-analyzer",
        action="append",
        default=[],
        help="Analyzer whose results to collect; repeat for several (default: "
        "every enabled analyzer)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Number of declared jobs the collection must contain",
    )
    parser.add_argument(
        "--metric-pattern",
        action="append",
        default=[],
        help="Shell-style pattern of the dotted metric paths to keep, such as "
        "'*accuracy*'; repeat for several (default: every numeric value)",
    )
    parser.add_argument(
        "--omit-full-results",
        action="store_true",
        help="Leave the complete result dictionaries out of the JSON Lines file",
    )
    parser.add_argument(
        "--derive-information-null-excess",
        action="store_true",
        help="Add the observed-minus-null value of every information term "
        "that has a *_null counterpart",
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
                args.output_dir,
                args.require_analyzer,
                args.expected_count is not None,
                args.metric_pattern,
                args.omit_full_results,
                args.derive_information_null_excess,
                args.input_hash_validation != "rehash",
            )
        )
        if explicit:
            raise ValueError(
                "--config cannot be combined with explicit collection options"
            )
        return collect_checkpoint_analysis(**_load_collection_config(args.config))
    if not args.jobs or args.output_dir is None:
        raise ValueError("--jobs and --output-dir are required without --config")
    return collect_checkpoint_analysis(
        jobs_paths=args.jobs,
        output_dir=args.output_dir,
        required_analyzers=args.require_analyzer,
        expected_count=args.expected_count,
        metric_patterns=args.metric_pattern,
        include_full_results=not args.omit_full_results,
        input_hash_validation=args.input_hash_validation,
        derive_information_null_excess=args.derive_information_null_excess,
    )


def cli_main(argv: list[str] | None = None) -> int:
    """Console entry point: run :func:`main` and exit 0 on success."""
    setup_logging()
    main(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
