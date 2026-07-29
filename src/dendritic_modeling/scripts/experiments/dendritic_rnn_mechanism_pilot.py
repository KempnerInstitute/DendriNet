"""Materialize and launch the fixed-soma dendritic-RNN mechanism pilot.

Every generated run is a standalone standard training YAML consumed by
``dendritic_modeling.scripts.training.train_experiments``.  This launcher owns
only the frozen 3-seed x 2-timescale x 2-recurrent-placement matrix, scheduler
routing, and lifecycle metadata.
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from dendritic_modeling.scripts.experiments.dendritic_rnn_extreme_cube import (
    REPO_ROOT,
    SEED_FIELDS,
    _atomic_json,
    _git_metadata,
    _materialize_source_snapshot,
    _package_tree_sha256,
    _plain_yaml,
    _runtime_gpu_metadata,
    _sha256,
    _tree_sha256,
)

MATRIX_SCHEMA = "dendritic-rnn-mechanism-identification-pilot-matrix-v1"
MANIFEST_SCHEMA = "dendritic-rnn-mechanism-identification-pilot-manifest-v1"
RECEIPT_SCHEMA = "dendritic-rnn-mechanism-identification-pilot-receipt-v1"
CONFIG_ROOT = REPO_ROOT / "configs/dendritic_rnn/mechanism_identification_pilot_v1"
DEFAULT_MATRIX = CONFIG_ROOT / "matrix.yaml"
DEFAULT_GENERATED = CONFIG_ROOT / "generated"
EXPECTED_CONDITIONS = {
    "tau_distal_slow__rec_proximal_concentrated": (
        (125.0, 25.0, 5.0),
        (1, 1, 55),
    ),
    "tau_distal_slow__rec_distal_concentrated": (
        (125.0, 25.0, 5.0),
        (7, 1, 1),
    ),
    "tau_distal_fast__rec_proximal_concentrated": (
        (5.0, 25.0, 5.0),
        (1, 1, 55),
    ),
    "tau_distal_fast__rec_distal_concentrated": (
        (5.0, 25.0, 5.0),
        (7, 1, 1),
    ),
}
LEVEL_SIZES = (576, 192, 64)
FIXED_FF = (1, 1, 50)
EXPECTED_RECURRENT_CONTACTS = 4288


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_matrix(path: Path) -> tuple[dict[str, Any], Path]:
    path = path.resolve(strict=True)
    matrix = _plain_yaml(path)
    if matrix.get("schema_version") != MATRIX_SCHEMA:
        raise ValueError("Unsupported mechanism-pilot matrix schema")
    conditions = matrix.get("conditions")
    replicates = matrix.get("replicates")
    if not isinstance(conditions, list) or len(conditions) != 4:
        raise ValueError("Mechanism pilot requires four conditions")
    if not isinstance(replicates, list) or len(replicates) != 3:
        raise ValueError("Mechanism pilot requires three paired seeds")
    if {row.get("name") for row in conditions} != set(EXPECTED_CONDITIONS):
        raise ValueError("Mechanism-pilot conditions changed")
    for row in conditions:
        tau, recurrent = EXPECTED_CONDITIONS[str(row["name"])]
        if tuple(row.get("level_taus", ())) != tau:
            raise ValueError(f"Time constants changed for {row['name']}")
        if tuple(row.get("rec_excitatory_synapses_by_level", ())) != recurrent:
            raise ValueError(f"Recurrent profile changed for {row['name']}")
        if tuple(row.get("ff_excitatory_synapses_by_level", ())) != FIXED_FF:
            raise ValueError(f"Feedforward profile changed for {row['name']}")
        contacts = sum(size * count for size, count in zip(LEVEL_SIZES, recurrent))
        if contacts != EXPECTED_RECURRENT_CONTACTS:
            raise ValueError(f"Active recurrent contacts changed for {row['name']}")
        if tau[-1] != 5.0 or tau[1] != 25.0:
            raise ValueError("Soma and middle time constants must remain fixed")
    if [row.get("replicate") for row in replicates] != ["r000", "r001", "r002"]:
        raise ValueError("Pilot replicates must be ordered r000 through r002")
    for row in replicates:
        missing = [field for field in SEED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"Replicate {row['replicate']} lacks seeds: {missing}")
    design = matrix.get("design", {})
    if (
        design.get("status") != "post_outcome_descriptive_pilot"
        or design.get("inference") != "no_null_tests_three_seed_go_no_go"
        or design.get("recurrent_active_contacts_per_profile")
        != EXPECTED_RECURRENT_CONTACTS
    ):
        raise ValueError("Pilot claim or matching contract changed")
    run_prefix = str(matrix.get("run_prefix", "mechanism-pilot"))
    if not run_prefix or "/" in run_prefix or "\\" in run_prefix:
        raise ValueError("run_prefix must be a non-empty path-safe name")
    isolation = matrix.get("runtime_isolation", {})
    if isolation and (
        isolation.get("immutable_source_snapshot") is not True
        or isolation.get("frozen_protocol_inputs") is not True
    ):
        raise ValueError("Runtime-isolated pilots require frozen source and protocol")
    return matrix, path


def _merge_base(matrix: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    base_path = (REPO_ROOT / str(matrix["base_config"])).resolve(strict=True)
    if _sha256(base_path) != matrix.get("base_config_sha256"):
        raise ValueError("Base training config changed after pilot design")
    base = OmegaConf.create(_plain_yaml(base_path))
    overrides = OmegaConf.create(matrix["base_overrides"])
    merged = OmegaConf.to_container(OmegaConf.merge(base, overrides), resolve=True)
    if not isinstance(merged, dict):
        raise TypeError("Merged standard config is not a mapping")
    return merged, base_path


def _materialized_config(
    base: dict[str, Any],
    *,
    replicate: dict[str, Any],
    condition: dict[str, Any],
    output_root: Path,
    run_prefix: str,
) -> tuple[str, dict[str, Any]]:
    config = copy.deepcopy(base)
    run_id = f"{run_prefix}-{replicate['replicate']}-{condition['name']}"
    run_dir = output_root / run_id
    for field in SEED_FIELDS:
        config["experiment"][field] = int(replicate[field])
    layer = config["model"]["core"]["population_network"]["layers"][0]
    layer["connection_seed"] = int(replicate["topology_seed"])
    layer["population_defaults"]["indexed_seed"] = int(replicate["topology_seed"])
    layer["population_defaults"]["initialization_seed"] = int(replicate["model_seed"])
    pyramidal = layer["populations"][0]["population"]
    pyramidal["indexed_seed"] = int(replicate["topology_seed"])
    pyramidal["initialization_seed"] = int(replicate["model_seed"])
    pyramidal["level_taus"] = list(condition["level_taus"])
    pyramidal["ff_excitatory_synapses_by_level"] = list(
        condition["ff_excitatory_synapses_by_level"]
    )
    pyramidal["rec_excitatory_synapses_by_level"] = list(
        condition["rec_excitatory_synapses_by_level"]
    )
    config["experiment"]["checkpointing"]["checkpoint_dir"] = str(
        run_dir / "main_network/checkpoints"
    )
    config["outputs"]["run_name"] = run_id
    config["outputs"]["results_dir"] = str(run_dir)
    config["outputs"]["exact_run_dir"] = True
    return run_id, config


def _resolve_output_root(matrix: dict[str, Any], results_root: Path | None) -> Path:
    raw = results_root or Path(str(matrix["output_root"]))
    output_root = raw.resolve() if raw.is_absolute() else (REPO_ROOT / raw).resolve()
    isolation = matrix.get("runtime_isolation", {})
    if isolation.get("require_external_output_root"):
        if output_root == REPO_ROOT or output_root.is_relative_to(REPO_ROOT):
            raise ValueError(
                "Protected pilot output root must be outside the repository"
            )
        output_text = str(output_root)
        forbidden = tuple(isolation.get("forbidden_output_path_tokens", ()))
        if any(token in output_text for token in forbidden):
            raise ValueError(
                f"Protected pilot output uses a forbidden path: {output_root}"
            )
    return output_root


def _freeze_protocol_input(source: Path, destination: Path) -> dict[str, str]:
    """Copy one protocol input atomically and make the copy read-only."""

    source = source.resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _sha256(destination) != _sha256(source):
            raise RuntimeError(f"Frozen protocol input changed: {destination}")
    else:
        temporary = destination.with_name(f".{destination.name}.tmp")
        shutil.copy2(source, temporary)
        temporary.replace(destination)
        destination.chmod(0o444)
    return {"path": str(destination), "sha256": _sha256(destination)}


def materialize(
    *,
    matrix_path: Path = DEFAULT_MATRIX,
    generated_dir: Path = DEFAULT_GENERATED,
    results_root: Path | None = None,
) -> Path:
    """Materialize twelve standalone standard training YAMLs."""

    matrix, matrix_path = _load_matrix(matrix_path)
    base, base_path = _merge_base(matrix)
    output_root = _resolve_output_root(matrix, results_root)
    generated_dir = generated_dir.resolve()
    generated_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot = _materialize_source_snapshot(output_root)
    source_metadata = _git_metadata()
    if (
        source_snapshot["training_runtime_source_sha256"]
        != source_metadata["training_source_sha256"]
    ):
        raise RuntimeError("Frozen training source differs from the live source")
    protocol_input_root = output_root / "_protocol" / "inputs"
    frozen_matrix = _freeze_protocol_input(
        matrix_path, protocol_input_root / "matrix.yaml"
    )
    frozen_base = _freeze_protocol_input(base_path, protocol_input_root / "base.yaml")
    frozen_protocol_inputs: dict[str, dict[str, str]] = {}
    for name, specification in matrix.get("protocol_inputs", {}).items():
        if not isinstance(specification, dict):
            raise ValueError(f"Invalid protocol input specification: {name}")
        protocol_source = Path(str(specification.get("source", "")))
        if not protocol_source.is_absolute():
            protocol_source = REPO_ROOT / protocol_source
        destination_name = str(specification.get("destination", ""))
        destination = Path(destination_name)
        if (
            not destination_name
            or destination.is_absolute()
            or len(destination.parts) != 1
        ):
            raise ValueError(f"Invalid protocol input destination: {name}")
        frozen_protocol_inputs[str(name)] = _freeze_protocol_input(
            protocol_source, output_root / "_protocol" / destination
        )
    run_prefix = str(matrix.get("run_prefix", "mechanism-pilot"))
    runs = []
    for replicate in matrix["replicates"]:
        for condition in matrix["conditions"]:
            run_id, config = _materialized_config(
                base,
                replicate=replicate,
                condition=condition,
                output_root=output_root,
                run_prefix=run_prefix,
            )
            config_path = generated_dir / f"{run_id}.yaml"
            if config_path.exists():
                raise RuntimeError(
                    f"Refusing to overwrite frozen config: {config_path}"
                )
            OmegaConf.save(OmegaConf.create(config), config_path)
            config_path.chmod(0o444)
            runs.append(
                {
                    "index": len(runs),
                    "run_id": run_id,
                    "pairing_group": replicate["replicate"],
                    "condition": condition["name"],
                    "config": str(config_path),
                    "config_sha256": _sha256(config_path),
                    "output_dir": config["outputs"]["results_dir"],
                    "seeds": {field: int(replicate[field]) for field in SEED_FIELDS},
                }
            )
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": matrix["experiment_id"],
        "created_utc": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "matrix_config": frozen_matrix,
        "base_config": frozen_base,
        "canonical_inputs": {
            "matrix": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
            "base": {"path": str(base_path), "sha256": _sha256(base_path)},
        },
        "protocol_inputs": frozen_protocol_inputs,
        "output_root": str(output_root),
        "design": matrix["design"],
        "runtime": matrix["runtime"],
        "scheduler": matrix["scheduler"],
        "source": source_metadata,
        "source_snapshot": source_snapshot,
        "training_receipt_path": matrix.get("training_receipt_path"),
        "runs": runs,
    }
    manifest_path = generated_dir / "manifest.json"
    if manifest_path.exists():
        raise RuntimeError(f"Refusing to overwrite frozen manifest: {manifest_path}")
    _atomic_json(manifest_path, manifest)
    manifest_path.chmod(0o444)
    return manifest_path


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate configs, matching, pairing, and standard-trainer compatibility."""

    from dendritic_modeling.config import load_config

    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("Unsupported mechanism-pilot manifest")
    output_root = Path(str(manifest["output_root"])).resolve()
    source_snapshot = manifest.get("source_snapshot")
    if not isinstance(source_snapshot, dict):
        raise ValueError("Manifest lacks an immutable source snapshot")
    snapshot_root = Path(str(source_snapshot.get("root", ""))).resolve()
    if not snapshot_root.is_dir():
        raise ValueError(f"Source snapshot is missing: {snapshot_root}")
    package_sha256 = _package_tree_sha256(snapshot_root)
    training_sha256 = _tree_sha256(snapshot_root)
    if package_sha256 != source_snapshot.get("package_sha256"):
        raise ValueError("Source snapshot package hash changed")
    if training_sha256 != source_snapshot.get("training_runtime_source_sha256"):
        raise ValueError("Source snapshot training hash changed")
    if training_sha256 != manifest.get("source", {}).get("training_source_sha256"):
        raise ValueError("Live and frozen training hashes differed at materialization")
    for label in ("matrix_config", "base_config"):
        frozen = manifest.get(label, {})
        frozen_path = Path(str(frozen.get("path", ""))).resolve()
        if not frozen_path.is_file() or _sha256(frozen_path) != frozen.get("sha256"):
            raise ValueError(f"Frozen {label} changed: {frozen_path}")
    for label, frozen in manifest.get("protocol_inputs", {}).items():
        frozen_path = Path(str(frozen.get("path", ""))).resolve()
        if not frozen_path.is_file() or _sha256(frozen_path) != frozen.get("sha256"):
            raise ValueError(f"Frozen protocol input changed: {label}")
    matrix, _ = _load_matrix(Path(manifest["matrix_config"]["path"]))
    isolation = matrix.get("runtime_isolation", {})
    if isolation.get("require_external_output_root"):
        if output_root == REPO_ROOT or output_root.is_relative_to(REPO_ROOT):
            raise ValueError("Protected pilot output root is inside the repository")
        if snapshot_root != Path(str(source_snapshot["root"])).resolve():
            raise ValueError("Invalid source snapshot root")
        if not snapshot_root.is_relative_to(output_root / "_runtime"):
            raise ValueError("Source snapshot is outside the protected output root")
    if len(manifest.get("runs", [])) != 12:
        raise ValueError("Mechanism-pilot manifest must contain twelve runs")
    paired: dict[str, list[dict[str, Any]]] = {}
    signatures: dict[str, dict[str, Any]] = {}
    for row in manifest["runs"]:
        path = Path(row["config"]).resolve()
        if not path.is_file() or _sha256(path) != row["config_sha256"]:
            raise ValueError(f"Generated config changed: {path}")
        if isolation.get("frozen_protocol_inputs") and path.stat().st_mode & 0o222:
            raise ValueError(f"Generated config is writable: {path}")
        row_output = Path(str(row["output_dir"])).resolve()
        if row_output.parent != output_root:
            raise ValueError(f"Run output escaped the protected root: {row_output}")
        raw = _plain_yaml(path)
        typed = load_config(str(path))
        if (
            typed.data.dataset_name != "romo_delay_comparison_positive"
            or typed.model.core.type != "population_network"
            or typed.training.main.strategy != "recurrent"
            or typed.training.main.common.epochs != 200
        ):
            raise ValueError(f"Generated config left standard pilot contract: {path}")
        if not typed.experiment.strict_deterministic:
            raise ValueError(f"Strict determinism is disabled: {path}")
        if not typed.experiment.sealed_test or typed.experiment.allow_test_data:
            raise ValueError(f"Test seal is open: {path}")
        if "/drafts/" in json.dumps(raw):
            raise ValueError(f"Draft-owned runtime path detected: {path}")
        population = raw["model"]["core"]["population_network"]["layers"][0][
            "populations"
        ][0]["population"]
        condition_name = row["condition"]
        expected_tau, expected_rec = EXPECTED_CONDITIONS[condition_name]
        if tuple(population["level_taus"]) != expected_tau:
            raise ValueError(f"Materialized time constants changed: {path}")
        if tuple(population["rec_excitatory_synapses_by_level"]) != expected_rec:
            raise ValueError(f"Materialized recurrent placement changed: {path}")
        if tuple(population["ff_excitatory_synapses_by_level"]) != FIXED_FF:
            raise ValueError(f"Materialized feedforward placement changed: {path}")
        signature = {
            "branch_factors": tuple(
                raw["model"]["core"]["population_network"]["layers"][0]["populations"][
                    0
                ]["branch_factors"]
            ),
            "ff": tuple(population["ff_excitatory_synapses_by_level"]),
            "rec_inhibitory": tuple(population["rec_inhibitory_synapses_by_level"]),
        }
        signatures[condition_name] = signature
        paired.setdefault(row["pairing_group"], []).append(row)
    if any(len(rows) != 4 for rows in paired.values()) or len(paired) != 3:
        raise ValueError("Paired-seed quartet is incomplete")
    for name, rows in paired.items():
        if any(row["seeds"] != rows[0]["seeds"] for row in rows[1:]):
            raise ValueError(f"Seed streams differ within {name}")
    if len({json.dumps(value, sort_keys=True) for value in signatures.values()}) != 1:
        raise ValueError("Nontargeted architecture or input placement changed")
    if matrix["design"]["recurrent_active_contacts_per_profile"] != 4288:
        raise ValueError("Matrix contact-matching declaration changed")
    return {
        "manifest": str(manifest_path),
        "run_count": 12,
        "pairing_groups": 3,
        "standard_configs_valid": True,
        "contact_matching": "PASSED_EXACT",
        "soma_tau_fixed": "PASSED",
        "source_snapshot": str(snapshot_root),
        "package_sha256": package_sha256,
        "training_source_sha256": training_sha256,
    }


def _route(manifest: MappingLike, name: str, *, allow_h200: bool) -> dict[str, Any]:
    if name not in {"primary", "h100_fallback", "h200_opt_in"}:
        raise ValueError(f"Unknown scheduler route: {name}")
    route = dict(manifest["scheduler"][name])
    if "h200" in route["gpu_type"].lower() and not allow_h200:
        raise ValueError("H200 route requires --allow-h200")
    return route


MappingLike = dict[str, Any]


def _sbatch_command(
    *,
    manifest: MappingLike,
    row: MappingLike,
    route_name: str,
    allow_h200: bool,
    validate_only: bool,
    output_dir: Path | None = None,
) -> list[str]:
    route = _route(manifest, route_name, allow_h200=allow_h200)
    resources = manifest["scheduler"]["resources"]
    logs = Path(manifest["output_root"]) / "_slurm"
    runtime_logs = Path(manifest["output_root"]) / "_runtime_logs" / "training"
    logs.mkdir(parents=True, exist_ok=True)
    runtime_logs.mkdir(parents=True, exist_ok=True)
    source_snapshot = manifest["source_snapshot"]
    snapshot_root = Path(str(source_snapshot["root"])).resolve()
    job_name = (
        f"dendrnn-mpilot-pre-{row['condition'][:18]}"
        if validate_only
        else f"dendrnn-mpilot-{row['pairing_group']}-{row['condition'][:14]}"
    )
    runtime = [
        str(manifest["runtime"]["python"]),
        "-m",
        "dendritic_modeling.scripts.experiments.dendritic_rnn_mechanism_pilot",
        "run-one",
        "--config",
        str(row["config"]),
        "--expected-partition",
        str(route["partition"]),
        "--expected-gpu-type",
        str(route["gpu_type"]),
        "--expected-source-sha256",
        str(source_snapshot["training_runtime_source_sha256"]),
        "--expected-package-sha256",
        str(source_snapshot["package_sha256"]),
        "--expected-source-root",
        str(snapshot_root),
        "--expected-config-sha256",
        str(row["config_sha256"]),
    ]
    if validate_only:
        if output_dir is None:
            raise ValueError("Preflight output directory is required")
        runtime.extend(["--validate-only", "--output-dir", str(output_dir)])
    wrapped = " && ".join(
        [
            f"cd {shlex.quote(str(snapshot_root))}",
            "export PYTHONPATH=" + shlex.quote(str(snapshot_root / "src")),
            "export PYTHONDONTWRITEBYTECODE=1",
            "export OMP_NUM_THREADS=1",
            "export DENDRITIC_LOG_DIR="
            + shlex.quote(str(runtime_logs))
            + "/${SLURM_JOB_ID}",
            shlex.join(runtime),
        ]
    )
    walltime = (
        resources["preflight_walltime"] if validate_only else resources["walltime"]
    )
    return [
        "sbatch",
        "--parsable",
        "--no-requeue",
        f"--job-name={job_name}",
        f"--partition={route['partition']}",
        f"--account={route['account']}",
        "--nodes=1",
        "--ntasks=1",
        f"--gres=gpu:{route['gpu_type']}:{resources['gpus']}",
        f"--cpus-per-task={resources['cpus']}",
        f"--mem={resources['memory']}",
        f"--time={walltime}",
        f"--output={logs}/%x_%j.out",
        f"--error={logs}/%x_%j.err",
        f"--chdir={snapshot_root}",
        f"--comment={manifest['experiment_id']}:{row['run_id']}",
        f"--wrap={wrapped}",
    ]


def submit(
    *,
    manifest_path: Path,
    route_name: str,
    allow_h200: bool,
    dry_run: bool,
    preflight: bool,
) -> dict[str, Any]:
    """Submit all runs or the first seed's four validate-only preflights."""

    validate_manifest(manifest_path)
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    selected = (
        [row for row in manifest["runs"] if row["pairing_group"] == "r000"]
        if preflight
        else list(manifest["runs"])
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submitted = []
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "created_utc": _utc_now(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "source_snapshot": manifest["source_snapshot"],
        "route": route_name,
        "preflight": preflight,
        "dry_run": dry_run,
        "state": "DRY_RUN" if dry_run else "SUBMITTING",
        "jobs": submitted,
    }
    receipt_path: Path | None = None
    if not dry_run:
        configured_receipt = manifest.get("training_receipt_path")
        if not preflight and configured_receipt:
            configured_path = Path(str(configured_receipt))
            if not configured_path.is_absolute():
                raise ValueError("Configured training receipt must be absolute")
            receipt_path = configured_path.resolve()
        else:
            kind = "preflight" if preflight else "training"
            receipt_path = (
                Path(manifest["output_root"])
                / "_launch"
                / f"{kind}_submission_{stamp}.json"
            )
        if receipt_path.exists():
            raise RuntimeError(f"Refusing to overwrite launch receipt: {receipt_path}")
        _atomic_json(receipt_path, receipt)
    for row in selected:
        output_dir = (
            Path(manifest["output_root"]) / "_preflight" / stamp / row["condition"]
            if preflight
            else None
        )
        if not preflight:
            run_dir = Path(row["output_dir"])
            if run_dir.exists() and any(run_dir.iterdir()):
                raise RuntimeError(f"Output directory is not empty: {run_dir}")
        command = _sbatch_command(
            manifest=manifest,
            row=row,
            route_name=route_name,
            allow_h200=allow_h200,
            validate_only=preflight,
            output_dir=output_dir,
        )
        job_id = None
        if not dry_run:
            try:
                result = subprocess.run(
                    command, check=True, capture_output=True, text=True
                )
                job_id = result.stdout.strip().split(";", 1)[0]
                if not job_id.isdigit():
                    raise RuntimeError(f"Invalid sbatch job ID: {job_id!r}")
            except Exception as exc:
                receipt["state"] = "SUBMISSION_FAILED"
                receipt["error"] = f"{type(exc).__name__}: {exc}"
                if receipt_path is not None:
                    _atomic_json(receipt_path, receipt)
                raise
        submitted.append(
            {
                "run_id": row["run_id"],
                "condition": row["condition"],
                "job_id": job_id,
                "output_dir": str(output_dir or row["output_dir"]),
                "command": command,
            }
        )
        if receipt_path is not None:
            _atomic_json(receipt_path, receipt)
    if receipt_path is not None:
        receipt["state"] = "SUBMITTED"
        receipt["finished_utc"] = _utc_now()
        _atomic_json(receipt_path, receipt)
        receipt_path.chmod(0o444)
        receipt["receipt"] = str(receipt_path)
    return receipt


def run_one(
    *,
    config_path: Path,
    expected_partition: str,
    expected_gpu_type: str,
    expected_source_sha256: str,
    expected_package_sha256: str,
    expected_source_root: Path,
    expected_config_sha256: str,
    validate_only: bool,
    output_dir: Path | None,
) -> None:
    """Delegate one generated YAML to the standard source training lifecycle."""

    from dendritic_modeling.scripts.experiments.dendritic_rnn_extreme_cube import (
        run_one as standard_run_one,
    )

    config_path = config_path.resolve(strict=True)
    expected_source_root = expected_source_root.resolve(strict=True)
    if REPO_ROOT.resolve() != expected_source_root:
        raise RuntimeError(
            f"Pilot imported from {REPO_ROOT}, expected {expected_source_root}"
        )
    if _package_tree_sha256() != expected_package_sha256:
        raise RuntimeError("Runtime source snapshot package changed")
    if _tree_sha256() != expected_source_sha256:
        raise RuntimeError("Training source changed after pilot materialization")
    if _sha256(config_path) != expected_config_sha256:
        raise RuntimeError("Training config changed after pilot materialization")
    _runtime_gpu_metadata(expected_gpu_type)
    standard_run_one(
        config_path=config_path,
        expected_partition=expected_partition,
        expected_gpu_type=expected_gpu_type,
        expected_source_sha256=expected_source_sha256,
        validate_only=validate_only,
        output_dir=output_dir,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    materialize_parser.add_argument(
        "--generated-dir", type=Path, default=DEFAULT_GENERATED
    )
    materialize_parser.add_argument("--results-root", type=Path)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_GENERATED / "manifest.json"
    )
    for command in ("preflight", "submit"):
        submit_parser = subparsers.add_parser(command)
        submit_parser.add_argument(
            "--manifest", type=Path, default=DEFAULT_GENERATED / "manifest.json"
        )
        submit_parser.add_argument(
            "--route",
            choices=("primary", "h100_fallback", "h200_opt_in"),
            default="primary",
        )
        submit_parser.add_argument("--allow-h200", action="store_true")
        submit_parser.add_argument("--dry-run", action="store_true")
    run_parser = subparsers.add_parser("run-one")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--expected-partition", required=True)
    run_parser.add_argument("--expected-gpu-type", required=True)
    run_parser.add_argument("--expected-source-sha256", required=True)
    run_parser.add_argument("--expected-package-sha256", required=True)
    run_parser.add_argument("--expected-source-root", type=Path, required=True)
    run_parser.add_argument("--expected-config-sha256", required=True)
    run_parser.add_argument("--validate-only", action="store_true")
    run_parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "materialize":
        result: Any = {
            "manifest": str(
                materialize(
                    matrix_path=args.matrix,
                    generated_dir=args.generated_dir,
                    results_root=args.results_root,
                )
            )
        }
    elif args.command == "validate":
        result = validate_manifest(args.manifest)
    elif args.command in {"preflight", "submit"}:
        result = submit(
            manifest_path=args.manifest,
            route_name=args.route,
            allow_h200=args.allow_h200,
            dry_run=args.dry_run,
            preflight=args.command == "preflight",
        )
    else:
        run_one(
            config_path=args.config,
            expected_partition=args.expected_partition,
            expected_gpu_type=args.expected_gpu_type,
            expected_source_sha256=args.expected_source_sha256,
            expected_package_sha256=args.expected_package_sha256,
            expected_source_root=args.expected_source_root,
            expected_config_sha256=args.expected_config_sha256,
            validate_only=args.validate_only,
            output_dir=args.output_dir,
        )
        result = {"state": "COMPLETE"}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["materialize", "run_one", "submit", "validate_manifest"]
