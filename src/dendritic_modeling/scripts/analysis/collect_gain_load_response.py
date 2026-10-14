"""Collect frozen-checkpoint shared-gain/load factorial responses.

The registered analyzer emits one record per technical perturbation draw.  This
collector validates immutable checkpoint provenance, verifies the complete
factorial and common-random-number contract, averages draws within checkpoint,
and emits tidy draw- and checkpoint-level tables.  It performs no inference
across independently trained checkpoints.
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

from dendritic_modeling.scripts.analysis.collect_branch_local_sensitivity import (
    _validate_result as _validate_branch_local_sensitivity_result,
)
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

SCHEMA_VERSION = "gain_load_response_collection_v1"
CONFIG_SCHEMA_VERSION = "gain_load_response_collection_config_v1"
REQUIRED_ANALYZER = "gain_load_perturbation"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _integer(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or int(value) != value:
        raise TypeError(f"{context} must be an integer")
    return int(value)


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"result must be a mapping: {path}")
    return payload


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({str(key) for row in rows for key in row})
    if "job_index" in fields:
        fields.remove("job_index")
        fields.insert(0, "job_index")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or ["job_index"])
        writer.writeheader()
        writer.writerows(rows)


def _validate_protocol(payload: Mapping[str, Any], *, path: Path) -> None:
    if payload.get("gain_model") != "one mean-one lognormal scalar per trial":
        raise ValueError(f"unexpected shared-gain model: {path}")
    if payload.get("technical_draws_are_inference_units") is not False:
        raise ValueError(f"technical draws must not be inference units: {path}")
    if payload.get("common_random_numbers") is not True:
        raise ValueError(f"common-random-number contract is absent: {path}")


def _checkpoint_records(
    *,
    job_index: int,
    job: Mapping[str, Any],
    chance_accuracy: float,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[float, float, int], tuple[int, int]],
]:
    run_dir = Path(str(job["output_dir"])).resolve()
    result_path = run_dir / "gain_load_perturbation/final.json"
    payload = _read_mapping(result_path)
    _validate_protocol(payload, path=result_path)

    gain_grid = tuple(
        _finite(value, context="gain_log_sds")
        for value in payload.get("gain_log_sds", [])
    )
    load_grid = tuple(
        _finite(value, context="load_levels")
        for value in payload.get("load_levels", [])
    )
    draws = _integer(payload.get("draws_per_condition"), context="draws_per_condition")
    if not gain_grid or not load_grid or draws <= 0:
        raise ValueError(f"invalid gain/load grid declaration: {result_path}")
    expected = {
        (gain, load, draw)
        for gain in gain_grid
        for load in load_grid
        for draw in range(draws)
    }
    raw_records = payload.get("records")
    if not isinstance(raw_records, list):
        raise TypeError(f"gain/load records must be a list: {result_path}")

    draw_rows: list[dict[str, Any]] = []
    input_statistic_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    observed: set[tuple[float, float, int]] = set()
    seed_contract: dict[tuple[float, float, int], tuple[int, int]] = {}
    for record in raw_records:
        if not isinstance(record, Mapping):
            raise TypeError(f"gain/load record must be a mapping: {result_path}")
        gain = _finite(record.get("gain_log_sd"), context="gain_log_sd")
        load = _finite(record.get("load_level"), context="load_level")
        draw = _integer(record.get("draw"), context="draw")
        key = (gain, load, draw)
        if key in observed:
            raise ValueError(f"duplicate gain/load draw {key}: {result_path}")
        observed.add(key)
        gain_seed = _integer(record.get("gain_draw_seed"), context="gain_draw_seed")
        load_seed = _integer(record.get("load_draw_seed"), context="load_draw_seed")
        seed_contract[key] = (gain_seed, load_seed)
        base_record = {
            **_job_fields(job_index, job),
            "measurement_split": "test",
            "gain_log_sd": gain,
            "load_level": load,
            "load_mode": str(payload.get("load_mode")),
            "draw": draw,
            "draw_seed": _integer(record.get("draw_seed"), context="draw_seed"),
            "gain_draw_seed": gain_seed,
            "load_draw_seed": load_seed,
            "n_samples": _integer(record.get("n_samples"), context="n_samples"),
            "independent_replicate": False,
        }
        draw_rows.append(
            {
                **base_record,
                "accuracy": _finite(record.get("accuracy"), context="accuracy"),
            }
        )
        input_statistics = record.get("input_statistics")
        if input_statistics is not None:
            if not isinstance(input_statistics, Mapping):
                raise TypeError(f"input statistics must be a mapping: {result_path}")
            input_statistic_rows.append(
                {
                    **base_record,
                    **{
                        f"input_{key}": _finite(
                            value, context=f"input_statistics.{key}"
                        )
                        for key, value in input_statistics.items()
                    },
                }
            )
        sensitivity = record.get("branch_local_sensitivity")
        if sensitivity is not None:
            if not isinstance(sensitivity, Mapping):
                raise TypeError(
                    f"branch local sensitivity must be a mapping: {result_path}"
                )
            sensitivity_records, sensitivity_contract = (
                _validate_branch_local_sensitivity_result(sensitivity)
            )
            for sensitivity_record in sensitivity_records:
                sensitivity_rows.append(
                    {
                        **base_record,
                        **sensitivity_contract,
                        **sensitivity_record,
                        "inference_unit": "independently_trained_checkpoint",
                    }
                )
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise ValueError(
            f"incomplete gain/load factorial in {result_path}; "
            f"missing={missing}, extra={extra}"
        )

    by_condition: dict[tuple[float, float], list[float]] = {}
    for row in draw_rows:
        condition = (float(row["gain_log_sd"]), float(row["load_level"]))
        by_condition.setdefault(condition, []).append(float(row["accuracy"]))
    clean_values = by_condition.get((0.0, 0.0))
    if clean_values is None:
        raise ValueError(
            f"gain/load grid must include the clean (0,0) cell: {result_path}"
        )
    if max(clean_values) - min(clean_values) > 1e-12:
        raise ValueError(f"clean technical draws are not identical: {result_path}")
    clean_accuracy = sum(clean_values) / len(clean_values)
    denominator = clean_accuracy - chance_accuracy
    if denominator <= 0:
        raise ValueError(
            "chance-corrected retention requires clean accuracy above chance"
        )

    checkpoint_rows: list[dict[str, Any]] = []
    for (gain, load), values in sorted(by_condition.items()):
        if len(values) != draws:
            raise ValueError(
                f"condition {(gain, load)} has {len(values)} draws, expected {draws}"
            )
        accuracy = sum(values) / len(values)
        checkpoint_rows.append(
            {
                **_job_fields(job_index, job),
                "measurement_split": "test",
                "gain_log_sd": gain,
                "load_level": load,
                "load_mode": str(payload.get("load_mode")),
                "clean_accuracy": clean_accuracy,
                "perturbed_accuracy": accuracy,
                "chance_accuracy": chance_accuracy,
                "accuracy_change_from_clean": accuracy - clean_accuracy,
                "accuracy_loss_from_clean": clean_accuracy - accuracy,
                "retention_above_chance": ((accuracy - chance_accuracy) / denominator),
                "technical_draws": draws,
                "inference_unit": "independently_trained_checkpoint",
            }
        )
    return (
        draw_rows,
        checkpoint_rows,
        input_statistic_rows,
        sensitivity_rows,
        seed_contract,
    )


def collect_gain_load_response(
    *,
    jobs_paths: Sequence[Path],
    output_dir: Path,
    chance_accuracy: float,
    expected_count: int | None = None,
    input_hash_validation: str = "rehash",
) -> dict[str, Any]:
    """Validate and collect gain/load responses without across-seed inference."""

    hash_validation = str(input_hash_validation).strip().lower()
    if hash_validation not in INPUT_HASH_VALIDATION_MODES:
        raise ValueError(
            "input_hash_validation must be 'rehash' or 'recorded_hash_chain'"
        )
    chance = _finite(chance_accuracy, context="chance_accuracy")
    if not 0 <= chance < 1:
        raise ValueError("chance_accuracy must be in [0, 1)")
    jobs = _read_jobs(jobs_paths)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(f"declared {len(jobs)} jobs; expected {expected_count}")

    draw_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    input_statistic_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
    reference_seeds: dict[tuple[float, float, int], tuple[int, int]] | None = None
    hash_cache: dict[Path, str] = {}
    for job_index, job in enumerate(jobs):
        run_dir = Path(str(job["output_dir"])).resolve()
        provenance_path = run_dir / "checkpoint_analysis_provenance.json"
        provenance = _read_mapping(provenance_path)
        _validate_provenance(
            job,
            provenance,
            hash_cache,
            input_hash_validation=hash_validation,
        )
        enabled = {str(value) for value in provenance.get("enabled_analyzers", [])}
        if REQUIRED_ANALYZER not in enabled:
            raise ValueError(f"required analyzer absent from run: {REQUIRED_ANALYZER}")
        require_completed_analyses(provenance, [REQUIRED_ANALYZER], run_dir=run_dir)
        (
            current_draws,
            current_checkpoints,
            current_input_statistics,
            current_sensitivities,
            current_seeds,
        ) = _checkpoint_records(job_index=job_index, job=job, chance_accuracy=chance)
        if reference_seeds is None:
            reference_seeds = current_seeds
        elif current_seeds != reference_seeds:
            raise ValueError(
                "gain/load common-random-number seeds differ across checkpoints"
            )
        draw_rows.extend(current_draws)
        checkpoint_rows.extend(current_checkpoints)
        input_statistic_rows.extend(current_input_statistics)
        sensitivity_rows.extend(current_sensitivities)
        result_path = run_dir / "gain_load_perturbation/final.json"
        file_rows.append(
            {
                **_job_fields(job_index, job),
                "result_path": str(result_path),
                "result_sha256": _sha256(result_path),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    draws_path = output_dir / "gain_load_draws.csv"
    checkpoints_path = output_dir / "gain_load_checkpoint_response.csv"
    files_path = output_dir / "gain_load_files.csv"
    input_statistics_path = output_dir / "gain_load_input_statistics.csv"
    sensitivity_path = output_dir / "gain_load_branch_sensitivity.csv"
    _write_csv(draws_path, draw_rows)
    _write_csv(checkpoints_path, checkpoint_rows)
    _write_csv(files_path, file_rows)
    _write_csv(input_statistics_path, input_statistic_rows)
    _write_csv(sensitivity_path, sensitivity_rows)
    release = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_paths": [str(path.resolve()) for path in jobs_paths],
        "jobs_sha256": [_sha256(path.resolve()) for path in jobs_paths],
        "collected_checkpoint_count": len(jobs),
        "draw_record_count": len(draw_rows),
        "checkpoint_condition_count": len(checkpoint_rows),
        "input_statistic_record_count": len(input_statistic_rows),
        "branch_sensitivity_record_count": len(sensitivity_rows),
        "chance_accuracy": chance,
        "input_hash_validation": hash_validation,
        "aggregation_policy": (
            "technical perturbation draws are averaged within checkpoint; "
            "no across-checkpoint aggregation or inference is performed"
        ),
        "inference_unit": "independently_trained_checkpoint",
        "common_random_numbers_validated_across_checkpoints": True,
        "collector_code_identity": discover_source_identity(
            _repo_root(),
            source_files=(
                Path(__file__),
                Path(__file__).with_name("collect_checkpoint_analysis.py"),
            ),
        ),
        "outputs": {
            draws_path.name: _sha256(draws_path),
            checkpoints_path.name: _sha256(checkpoints_path),
            files_path.name: _sha256(files_path),
            input_statistics_path.name: _sha256(input_statistics_path),
            sensitivity_path.name: _sha256(sensitivity_path),
        },
    }
    (output_dir / "collection_provenance.json").write_text(
        json.dumps(release, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return release


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_collection_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported gain/load collection config: {path}")
    missing = {"jobs", "output_dir", "chance_accuracy"} - set(payload)
    if missing:
        raise ValueError(f"collection config lacks required fields: {sorted(missing)}")
    jobs = payload["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("collection config jobs must be a nonempty list")
    base = path.resolve().parent
    return {
        "jobs_paths": [_resolve_path(value, base=base) for value in jobs],
        "output_dir": _resolve_path(payload["output_dir"], base=base),
        "chance_accuracy": float(payload["chance_accuracy"]),
        "expected_count": payload.get("expected_count"),
        "input_hash_validation": str(payload.get("input_hash_validation", "rehash")),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="Collection configuration file"
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    return collect_gain_load_response(**_load_collection_config(args.config))


def cli_main(argv: list[str] | None = None) -> int:
    """Console entry point: run :func:`main` and exit 0 on success."""
    setup_logging()
    main(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
