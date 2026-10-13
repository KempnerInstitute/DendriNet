"""Collect tidy accuracy responses from frozen-checkpoint perturbation analyses.

The registered noise and multiplicative-gain analyzers deliberately save their
native result dictionaries.  This collector validates those outputs against an
immutable checkpoint-analysis ledger and aligns every perturbed accuracy with
the clean accuracy from the same checkpoint.  It emits one row per checkpoint
and perturbation level; it performs no across-checkpoint aggregation.
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

SCHEMA_VERSION = "perturbation_response_collection_v1"
CONFIG_SCHEMA_VERSION = "perturbation_response_collection_config_v1"
REQUIRED_ANALYZERS = {"performance", "noise_perturbation", "multiplicative_gain"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


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


def _analysis_noise_draws(path: Path) -> int:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        value = payload["analysis"]["noise_perturbation"]["params"]["n_samples"]
    except (KeyError, TypeError):
        raise ValueError(f"analysis config does not declare noise n_samples: {path}")
    draws = int(value)
    if draws < 1:
        raise ValueError(f"analysis noise n_samples must be positive: {path}")
    return draws


def _response_row(
    *,
    job_index: int,
    job: Mapping[str, Any],
    perturbation: str,
    level: float,
    perturbed_accuracy: float,
    clean_accuracy: float,
    chance_accuracy: float,
    technical_noise_draws: int,
) -> dict[str, Any]:
    denominator = clean_accuracy - chance_accuracy
    if denominator <= 0:
        raise ValueError(
            "chance-corrected retention requires clean accuracy above chance; "
            f"got clean={clean_accuracy} and chance={chance_accuracy}"
        )
    return {
        **_job_fields(job_index, job),
        "measurement_split": "test",
        "perturbation": perturbation,
        "perturbation_level": level,
        "clean_accuracy": clean_accuracy,
        "perturbed_accuracy": perturbed_accuracy,
        "chance_accuracy": chance_accuracy,
        "accuracy_change_from_clean": perturbed_accuracy - clean_accuracy,
        "accuracy_loss_from_clean": clean_accuracy - perturbed_accuracy,
        "clean_accuracy_above_chance": denominator,
        "perturbed_accuracy_above_chance": perturbed_accuracy - chance_accuracy,
        "retention_above_chance": (
            (perturbed_accuracy - chance_accuracy) / denominator
        ),
        "technical_noise_draws": technical_noise_draws,
        "independent_replicate": False,
    }


def _checkpoint_rows(
    *,
    job_index: int,
    job: Mapping[str, Any],
    chance_accuracy: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir = Path(str(job["output_dir"])).resolve()
    performance_path = run_dir / "performance/final.json"
    noise_path = run_dir / "noise_perturbation_analysis/final"
    gain_path = run_dir / "multiplicative_gain_analysis/final.json"
    performance = _read_mapping(performance_path)
    noise = _read_mapping(noise_path)
    gain = _read_mapping(gain_path)

    try:
        clean_accuracy = _finite_number(
            performance["accuracy"]["test"], context="performance.accuracy.test"
        )
    except (KeyError, TypeError):
        raise ValueError(f"performance result lacks test accuracy: {performance_path}")
    noise_draws = _analysis_noise_draws(Path(str(job["analysis_config"])))

    rows: list[dict[str, Any]] = []
    for noise_name in ("gaussian", "uniform"):
        levels = noise.get(noise_name)
        if not isinstance(levels, dict) or not levels:
            raise ValueError(f"noise result lacks {noise_name} levels: {noise_path}")
        ordered = sorted(levels.items(), key=lambda item: float(item[0]))
        for level_text, metrics in ordered:
            if not isinstance(metrics, dict) or "accuracy" not in metrics:
                raise ValueError(
                    f"noise result lacks {noise_name}.{level_text}.accuracy: {noise_path}"
                )
            rows.append(
                _response_row(
                    job_index=job_index,
                    job=job,
                    perturbation=noise_name,
                    level=_finite_number(
                        float(level_text), context=f"{noise_name} level"
                    ),
                    perturbed_accuracy=_finite_number(
                        metrics["accuracy"],
                        context=f"{noise_name}.{level_text}.accuracy",
                    ),
                    clean_accuracy=clean_accuracy,
                    chance_accuracy=chance_accuracy,
                    technical_noise_draws=noise_draws,
                )
            )

    factors = gain.get("gain_factors")
    scores = gain.get("raw_scores", {}).get("accuracy")
    if not isinstance(factors, list) or not isinstance(scores, list):
        raise ValueError(f"gain result lacks factors or raw accuracy: {gain_path}")
    if not factors or len(factors) != len(scores):
        raise ValueError(
            f"gain factors and accuracy have different lengths: {gain_path}"
        )
    for factor, score in zip(factors, scores):
        rows.append(
            _response_row(
                job_index=job_index,
                job=job,
                perturbation="multiplicative_gain",
                level=_finite_number(factor, context="gain factor"),
                perturbed_accuracy=_finite_number(score, context="gain accuracy"),
                clean_accuracy=clean_accuracy,
                chance_accuracy=chance_accuracy,
                technical_noise_draws=1,
            )
        )

    files = [
        {
            **_job_fields(job_index, job),
            "result_role": role,
            "result_path": str(path),
            "result_sha256": _sha256(path),
        }
        for role, path in (
            ("clean_performance", performance_path),
            ("additive_noise", noise_path),
            ("multiplicative_gain", gain_path),
        )
    ]
    return rows, files


def collect_perturbation_response(
    *,
    jobs_paths: Sequence[Path],
    output_dir: Path,
    chance_accuracy: float,
    expected_count: int | None = None,
    input_hash_validation: str = "rehash",
) -> dict[str, Any]:
    """Validate and collect clean-aligned perturbation response records."""
    hash_validation = str(input_hash_validation).strip().lower()
    if hash_validation not in INPUT_HASH_VALIDATION_MODES:
        raise ValueError(
            "input_hash_validation must be 'rehash' or 'recorded_hash_chain'"
        )
    chance = _finite_number(chance_accuracy, context="chance_accuracy")
    if not 0.0 <= chance < 1.0:
        raise ValueError("chance_accuracy must be in [0, 1)")
    jobs = _read_jobs(jobs_paths)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(f"declared {len(jobs)} jobs; expected {expected_count}")

    response_rows: list[dict[str, Any]] = []
    file_rows: list[dict[str, Any]] = []
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
        missing = sorted(REQUIRED_ANALYZERS - enabled)
        if missing:
            raise ValueError(f"required analyzers absent from run: {missing}")
        require_completed_analyses(
            provenance, sorted(REQUIRED_ANALYZERS), run_dir=run_dir
        )
        rows, files = _checkpoint_rows(
            job_index=job_index,
            job=job,
            chance_accuracy=chance,
        )
        response_rows.extend(rows)
        file_rows.extend(files)

    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "checkpoint_perturbation_response.csv"
    files_path = output_dir / "checkpoint_perturbation_files.csv"
    _write_csv(responses_path, response_rows)
    _write_csv(files_path, file_rows)
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "jobs_paths": [str(path.resolve()) for path in jobs_paths],
        "jobs_sha256": [_sha256(path.resolve()) for path in jobs_paths],
        "collected_checkpoint_count": len(jobs),
        "response_record_count": len(response_rows),
        "chance_accuracy": chance,
        "input_hash_validation": hash_validation,
        "aggregation_policy": (
            "noise draws are reduced by the registered analyzer within checkpoint; "
            "this collector performs no across-checkpoint aggregation"
        ),
        "inference_unit": "independently_trained_checkpoint",
        "collector_code_identity": discover_source_identity(
            _repo_root(),
            source_files=(
                Path(__file__),
                Path(__file__).with_name("collect_checkpoint_analysis.py"),
            ),
        ),
        "outputs": {
            responses_path.name: _sha256(responses_path),
            files_path.name: _sha256(files_path),
        },
    }
    (output_dir / "collection_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def _resolve_path(value: str | Path, *, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_collection_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported perturbation collection config: {path}")
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
    return collect_perturbation_response(**_load_collection_config(args.config))


def cli_main(argv: list[str] | None = None) -> int:
    """Console entry point: run :func:`main` and exit 0 on success."""
    setup_logging()
    main(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
