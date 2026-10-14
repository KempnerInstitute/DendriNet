"""Collect reactivation operating-state records from frozen checkpoints.

The generic checkpoint collector deliberately preserves analyzer dictionaries
without interpreting module names.  This dedicated collector adds the standard
population-network coordinates needed for analyses over network position,
population identity, and soma-relative dendritic depth.  It performs no
across-checkpoint aggregation; every emitted module and metric row remains
bound to one declared checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from dendritic_modeling.scripts.analysis.collect_checkpoint_analysis import (
    _job_fields,
    _read_jobs,
    _sha256,
    _validate_provenance,
    require_completed_analyses,
)
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity
from dendritic_modeling.utils.logging_config import setup_logging

SCHEMA_VERSION = "reactivation_dynamics_collection_v2"
CONFIG_SCHEMA_VERSION = "reactivation_dynamics_collection_config_v1"
RESULT_SUBDIRECTORY = Path("reactivation_dynamics/epochs")

_POPULATION_PATH = re.compile(
    r"^(?P<prefix>.*?)"
    r"(?:layers\.(?P<network_layer>\d+)\.)?"
    r"populations\.(?P<population>[A-Za-z][A-Za-z0-9_]*)\."
    r"branch_layers\.(?P<branch_layer_list_index>\d+)\.reactivation$"
)


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
        writer = csv.DictWriter(handle, fieldnames=fields or ["job_index"])
        writer.writeheader()
        writer.writerows(rows)


def _module_coordinates(module_name: str, entry: Mapping[str, Any]) -> dict[str, Any]:
    """Population coordinates of one gate.

    Depth and polarity come from the analyzer, which reads them from the
    model: the ``branch_layers`` list index runs from the distal level to the
    soma and is not the soma-relative depth.
    """
    match = _POPULATION_PATH.fullmatch(module_name)
    depth = entry.get("soma_relative_dendritic_depth")
    polarity = entry.get("population_polarity")
    if match is None:
        return {
            "structured_population_path": False,
            "network_layer_index_zero_based": None,
            "network_layer_position": None,
            "network_layer_coordinate_inferred_single_layer": False,
            "population_name": None,
            "population_type": polarity,
            "branch_layer_list_index": None,
            "soma_relative_dendritic_depth": depth,
        }
    layer_text = match.group("network_layer")
    layer_index = None if layer_text is None else int(layer_text)
    return {
        "structured_population_path": True,
        "network_layer_index_zero_based": layer_index,
        "network_layer_position": 1 if layer_index is None else layer_index + 1,
        "network_layer_coordinate_inferred_single_layer": layer_index is None,
        "population_name": match.group("population"),
        "population_type": polarity,
        "branch_layer_list_index": int(match.group("branch_layer_list_index")),
        "soma_relative_dendritic_depth": depth,
    }


def _finite_number(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{context} must be finite")
    return number


def _validate_result(
    result: Mapping[str, Any],
    *,
    result_path: Path,
    require_structured_coordinates: bool,
) -> None:
    if result.get("filename") != "final" or result.get("training") is not False:
        raise ValueError(
            f"Reactivation result is not a final evaluation: {result_path}"
        )
    layers = result.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"Reactivation result has no layer mapping: {result_path}")
    if int(result.get("n_layers", -1)) != len(layers):
        raise ValueError(f"Reactivation layer count is inconsistent: {result_path}")

    for module_name, entry in layers.items():
        if not isinstance(entry, dict) or entry.get("skipped"):
            raise ValueError(
                f"Incomplete reactivation module {module_name}: {result_path}"
            )
        if not isinstance(entry.get("soma_relative_dendritic_depth"), int) or (
            entry.get("population_polarity") not in {"excitatory", "inhibitory"}
        ):
            raise ValueError(
                f"Reactivation module {module_name} has no soma-relative depth or "
                "population polarity; the result predates these fields, so re-run "
                f"the reactivation analysis: {result_path}"
            )
        coordinates = _module_coordinates(str(module_name), entry)
        if (
            require_structured_coordinates
            and not coordinates["structured_population_path"]
        ):
            raise ValueError(
                f"Reactivation module lacks standard population coordinates: {module_name}"
            )
        for group_name in ("V", "params", "reactivation_output"):
            group = entry.get(group_name)
            if not isinstance(group, dict) or not group:
                raise ValueError(
                    f"Reactivation module {module_name} lacks {group_name}: {result_path}"
                )
            for metric_name, value in group.items():
                _finite_number(
                    value,
                    context=f"{module_name}.{group_name}.{metric_name}",
                )
        output = entry["reactivation_output"]
        if "sat_lo" not in output:
            continue  # rail occupancy is reported for unit-range gates only
        occupancy = sum(
            _finite_number(output[name], context=f"{module_name}.{name}")
            for name in ("sat_lo", "sat_hi", "active_frac")
        )
        if not math.isclose(occupancy, 1.0, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError(
                f"Reactivation occupancy fractions do not sum to one: {module_name}"
            )


def _module_rows(
    job_index: int,
    job: Mapping[str, Any],
    result: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    modules: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for module_name, entry in sorted(result["layers"].items()):
        coordinates = _module_coordinates(str(module_name), entry)
        base = {
            **_job_fields(job_index, job),
            "measurement_split": "test",
            "module_name": str(module_name),
            **coordinates,
        }
        voltage = entry["V"]
        params = entry["params"]
        output = entry["reactivation_output"]
        module_row = {
            **base,
            "sample_count": int(voltage["n"]),
            **{
                f"voltage_{name}": _csv_cell(value)
                for name, value in voltage.items()
                if name != "n"
            },
            **{f"parameter_{name}": _csv_cell(value) for name, value in params.items()},
            **{f"output_{name}": _csv_cell(value) for name, value in output.items()},
        }
        modules.append(module_row)
        for group_name, group in (
            ("pre_gate_voltage", voltage),
            ("reactivation_parameter", params),
            ("post_gate_output", output),
        ):
            for metric_name, value in sorted(group.items()):
                if group_name == "pre_gate_voltage" and metric_name == "n":
                    continue
                metrics.append(
                    {
                        **base,
                        "sample_count": int(voltage["n"]),
                        "metric_group": group_name,
                        "metric_name": str(metric_name),
                        "value": float(value),
                        "independent_replicate": False,
                    }
                )
    return modules, metrics


def collect_reactivation_dynamics(
    *,
    jobs_paths: Sequence[Path],
    output_dir: Path,
    expected_count: int | None = None,
    require_structured_coordinates: bool = False,
) -> dict[str, Any]:
    """Validate declared final snapshots and emit checkpoint-bound tables."""

    jobs = _read_jobs(jobs_paths)
    if expected_count is not None and len(jobs) != int(expected_count):
        raise ValueError(f"declared {len(jobs)} jobs; expected {expected_count}")
    output_dir.mkdir(parents=True, exist_ok=True)
    hash_cache: dict[Path, str] = {}
    module_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []

    for job_index, job in enumerate(jobs):
        run_dir = Path(str(job["output_dir"])).resolve()
        provenance_path = run_dir / "checkpoint_analysis_provenance.json"
        result_path = run_dir / RESULT_SUBDIRECTORY / "final.json"
        for path in (provenance_path, result_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        _validate_provenance(
            job,
            provenance,
            hash_cache,
            input_hash_validation="rehash",
        )
        if "reactivation_dynamics" not in {
            str(name) for name in provenance.get("enabled_analyzers", [])
        }:
            raise ValueError("reactivation_dynamics is absent from run provenance")
        require_completed_analyses(
            provenance, ["reactivation_dynamics"], run_dir=run_dir
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise TypeError(f"Reactivation result is not a mapping: {result_path}")
        _validate_result(
            result,
            result_path=result_path,
            require_structured_coordinates=require_structured_coordinates,
        )
        modules, metrics = _module_rows(job_index, job, result)
        module_rows.extend(modules)
        metric_rows.extend(metrics)
        source_rows.append(
            {
                **_job_fields(job_index, job),
                "result_path": str(result_path),
                "result_sha256": _sha256(result_path),
                "provenance_path": str(provenance_path),
                "provenance_sha256": _sha256(provenance_path),
                "module_count": len(modules),
                "metric_count": len(metrics),
            }
        )

    outputs = {
        "checkpoint_modules.csv": module_rows,
        "checkpoint_metric_records.csv": metric_rows,
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
        "module_record_count": len(module_rows),
        "metric_record_count": len(metric_rows),
        "measurement_split": "test",
        "technical_draw_aggregation": "none",
        "trained_checkpoint_is_inference_unit": True,
        "require_structured_coordinates": bool(require_structured_coordinates),
        "collector_code_identity": discover_source_identity(
            Path(__file__).resolve().parents[4], source_files=(Path(__file__),)
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
    if not isinstance(payload, dict) or payload.get("schema_version") != (
        CONFIG_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported reactivation collection config: {path}")
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
        "require_structured_coordinates": bool(
            payload.get("require_structured_coordinates", False)
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
        "--expected-count",
        type=int,
        help="Number of declared jobs the collection must contain",
    )
    parser.add_argument(
        "--require-structured-coordinates",
        action="store_true",
        help="Require gate module names of population_network layers",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    if args.config is not None:
        explicit = any(
            (
                args.jobs,
                args.output_dir,
                args.expected_count is not None,
                args.require_structured_coordinates,
            )
        )
        if explicit:
            raise ValueError("--config cannot be combined with explicit options")
        return collect_reactivation_dynamics(**_load_collection_config(args.config))
    if not args.jobs or args.output_dir is None:
        raise ValueError("--jobs and --output-dir are required without --config")
    return collect_reactivation_dynamics(
        jobs_paths=args.jobs,
        output_dir=args.output_dir,
        expected_count=args.expected_count,
        require_structured_coordinates=args.require_structured_coordinates,
    )


def cli_main(argv: list[str] | None = None) -> int:
    """Console entry point: run :func:`main` and exit 0 on success."""
    setup_logging()
    main(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
