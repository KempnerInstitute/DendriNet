"""Materialize, validate, and launch the dendritic-RNN extreme cube.

The training subprocess is always the canonical
``dendritic_modeling.scripts.training.train_experiments`` module. This file
owns only matrix materialization, scheduler routing, and outcome-free runtime
metadata. Paper analysis and visualization are separate downstream steps.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[4]
CONFIG_ROOT = REPO_ROOT / "configs/dendritic_rnn/extreme_placement_cube_v1"
DEFAULT_BASE = CONFIG_ROOT / "base.yaml"
DEFAULT_MATRIX = CONFIG_ROOT / "matrix.yaml"
DEFAULT_GENERATED = CONFIG_ROOT / "generated"
DEFAULT_PRIMARY_MANIFEST = Path(
    "/n/holylabs/kempner_dev/Users/hsafaai/dendrinet_revision_2026/"
    "manifests/confirmatory_balanced_v5_mainrc7_harnessrc17_v1/manifest.json"
)
DEFAULT_EXTENSION_REFERENCE_MANIFEST = Path(
    "/n/holylabs/kempner_dev/Users/hsafaai/dendrinet_revision_2026/"
    "bundles/extreme_placement_cube_priority_20260720_v14/manifest.json"
)
DEFAULT_RUNTIME_ATTESTATION = CONFIG_ROOT / "evaluation_runtime_attestation.json"
SEED_FIELDS = (
    "seed",
    "dataset_seed",
    "split_seed",
    "model_seed",
    "topology_seed",
    "loader_seed",
    "evaluation_seed",
    "probe_seed",
)
TRAINING_SOURCE_ROOTS = (
    "config",
    "datasets",
    "models",
    "networks",
    "training",
    "utils",
    "scripts/script_utils",
    "scripts/training",
)
REQUIRED_TRAINING_ARTIFACTS = (
    "config.json",
    "dataset_fingerprints.json",
    "final_model.pt",
    "model_resources.json",
    "training_summary.json",
)
REQUIRED_PREFLIGHT_ARTIFACTS = (
    "config.json",
    "dataset_fingerprints.json",
    "model_resources.json",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(repo_root: Path = REPO_ROOT) -> str:
    """Hash source files that can affect the standard training path."""

    package_root = repo_root / "src/dendritic_modeling"
    paths: list[Path] = []
    for relative_root in TRAINING_SOURCE_ROOTS:
        root = package_root / relative_root
        if root.is_file() and root.suffix == ".py":
            paths.append(root)
        elif root.is_dir():
            paths.extend(root.rglob("*.py"))
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = path.relative_to(repo_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _package_tree_sha256(repo_root: Path = REPO_ROOT) -> str:
    """Hash every Python source file in the repository package."""

    package_root = repo_root / "src/dendritic_modeling"
    paths = sorted(package_root.rglob("*.py"))
    if not paths:
        raise RuntimeError(f"No package source found below {package_root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(repo_root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _materialize_source_snapshot(output_root: Path) -> dict[str, str]:
    """Create a read-only runtime snapshot of the standard source package."""

    package_sha256 = _package_tree_sha256()
    snapshot_root = output_root / "_runtime" / f"source_{package_sha256[:16]}"
    snapshot_package = snapshot_root / "src/dendritic_modeling"
    if snapshot_root.exists():
        if (
            not snapshot_package.is_dir()
            or _package_tree_sha256(snapshot_root) != package_sha256
        ):
            raise RuntimeError(
                f"Existing source snapshot is incomplete: {snapshot_root}"
            )
    else:
        snapshot_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = snapshot_root.with_name(f".{snapshot_root.name}.{os.getpid()}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary_package = temporary / "src/dendritic_modeling"
        shutil.copytree(
            REPO_ROOT / "src/dendritic_modeling",
            temporary_package,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        if _package_tree_sha256(temporary) != package_sha256:
            shutil.rmtree(temporary)
            raise RuntimeError("Source snapshot hash differs from the live package")
        for path in sorted(temporary.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        temporary.chmod(0o555)
        temporary.replace(snapshot_root)
    return {
        "root": str(snapshot_root),
        "package_sha256": package_sha256,
        "training_runtime_source_sha256": _tree_sha256(snapshot_root),
    }


def _materialize_overlay_source_snapshot(
    output_root: Path,
    *,
    base_snapshot_root: Path,
    overlay_paths: tuple[str, ...],
) -> dict[str, Any]:
    """Overlay repository-owned analysis scripts on an immutable runtime base."""

    base_snapshot_root = base_snapshot_root.resolve()
    base_package = base_snapshot_root / "src/dendritic_modeling"
    if not base_package.is_dir():
        raise RuntimeError(f"Missing base source snapshot: {base_snapshot_root}")
    base_package_sha256 = _package_tree_sha256(base_snapshot_root)
    base_runtime_sha256 = _tree_sha256(base_snapshot_root)
    overlays = []
    for relative in overlay_paths:
        source = REPO_ROOT / "src/dendritic_modeling" / relative
        if not source.is_file() or source.suffix != ".py":
            raise RuntimeError(f"Invalid source overlay: {source}")
        overlays.append(
            {
                "relative_path": relative,
                "source": str(source),
                "sha256": _sha256(source),
            }
        )

    identity = hashlib.sha256()
    identity.update(base_package_sha256.encode())
    for row in overlays:
        identity.update(row["relative_path"].encode() + b"\0")
        identity.update(row["sha256"].encode() + b"\0")
    snapshot_root = output_root / "_runtime" / f"source_{identity.hexdigest()[:16]}"
    snapshot_package = snapshot_root / "src/dendritic_modeling"
    if snapshot_root.exists():
        if not snapshot_package.is_dir():
            raise RuntimeError(
                f"Existing source snapshot is incomplete: {snapshot_root}"
            )
    else:
        snapshot_root.parent.mkdir(parents=True, exist_ok=True)
        temporary = snapshot_root.with_name(f".{snapshot_root.name}.{os.getpid()}.tmp")
        if temporary.exists():
            shutil.rmtree(temporary)
        shutil.copytree(
            base_package,
            temporary / "src/dendritic_modeling",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        for path in temporary.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        for row in overlays:
            destination = temporary / "src/dendritic_modeling" / row["relative_path"]
            shutil.copy2(row["source"], destination)
        if _tree_sha256(temporary) != base_runtime_sha256:
            shutil.rmtree(temporary)
            raise RuntimeError("An analysis overlay changed the model runtime source")
        for path in sorted(temporary.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        temporary.chmod(0o555)
        temporary.replace(snapshot_root)

    if _tree_sha256(snapshot_root) != base_runtime_sha256:
        raise RuntimeError("Overlay snapshot runtime hash differs from its base")
    for row in overlays:
        path = snapshot_package / row["relative_path"]
        if _sha256(path) != row["sha256"]:
            raise RuntimeError(f"Overlay snapshot changed: {path}")
    return {
        "root": str(snapshot_root),
        "package_sha256": _package_tree_sha256(snapshot_root),
        "training_runtime_source_sha256": base_runtime_sha256,
        "base_snapshot": {
            "root": str(base_snapshot_root),
            "package_sha256": base_package_sha256,
        },
        "overlays": overlays,
    }


def _plain_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_metadata(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "dirty": bool(status),
        "dirty_path_count": len(status.splitlines()) if status else 0,
        "training_source_sha256": _tree_sha256(repo_root),
    }


def _validate_matrix(matrix: dict[str, Any]) -> None:
    if matrix.get("schema_version") != "dendritic-rnn-extreme-placement-cube-matrix-v1":
        raise ValueError("Unsupported matrix schema")
    conditions = matrix.get("conditions", [])
    replicates = matrix.get("replicates", [])
    if len(conditions) != 2 or len(replicates) != 10:
        raise ValueError(
            "The confirmatory matrix must contain 2 conditions x 10 replicates"
        )
    condition_names = {row["name"] for row in conditions}
    expected_conditions = {
        "ff_proximal__rec_proximal__tau_reversed__standard",
        "ff_distal__rec_distal__tau_reversed__standard",
    }
    if condition_names != expected_conditions:
        raise ValueError(f"Unexpected conditions: {sorted(condition_names)}")
    replicate_names = [row["replicate"] for row in replicates]
    if replicate_names != [f"r{index:03d}" for index in range(10)]:
        raise ValueError("Replicates must be ordered r000 through r009")
    for replicate in replicates:
        missing = [field for field in SEED_FIELDS if field not in replicate]
        if missing:
            raise ValueError(
                f"Replicate {replicate['replicate']} is missing seeds: {missing}"
            )


def _materialized_config(
    base: dict[str, Any],
    *,
    replicate: dict[str, Any],
    condition: dict[str, Any],
    output_root: Path,
) -> tuple[str, dict[str, Any]]:
    config = copy.deepcopy(base)
    run_id = f"extreme-cube-{replicate['replicate']}-{condition['name']}"
    run_dir = output_root / run_id

    for field in SEED_FIELDS:
        config["experiment"][field] = int(replicate[field])

    layer = config["model"]["core"]["population_network"]["layers"][0]
    layer["connection_seed"] = int(replicate["topology_seed"])
    layer["population_defaults"]["indexed_seed"] = int(replicate["topology_seed"])
    layer["population_defaults"]["initialization_seed"] = int(replicate["model_seed"])
    pyramidal = layer["populations"][0]["population"]
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


def materialize(
    *,
    base_path: Path = DEFAULT_BASE,
    matrix_path: Path = DEFAULT_MATRIX,
    generated_dir: Path = DEFAULT_GENERATED,
    results_root: Path | None = None,
) -> Path:
    """Create 20 standalone standard YAMLs and their launch manifest."""

    base_path = base_path.resolve()
    matrix_path = matrix_path.resolve()
    generated_dir = generated_dir.resolve()
    base = _plain_yaml(base_path)
    matrix = _plain_yaml(matrix_path)
    _validate_matrix(matrix)
    output_root = (results_root or Path(matrix["output_root"])).resolve()
    generated_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for replicate in matrix["replicates"]:
        for condition in matrix["conditions"]:
            run_id, config = _materialized_config(
                base,
                replicate=replicate,
                condition=condition,
                output_root=output_root,
            )
            config_path = generated_dir / f"{run_id}.yaml"
            OmegaConf.save(OmegaConf.create(config), config_path)
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

    source = _git_metadata()
    manifest = {
        "schema_version": "dendritic-rnn-standard-training-manifest-v1",
        "experiment_id": matrix["experiment_id"],
        "created_utc": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "base_config": {"path": str(base_path), "sha256": _sha256(base_path)},
        "matrix_config": {
            "path": str(matrix_path),
            "sha256": _sha256(matrix_path),
        },
        "output_root": str(output_root),
        "runtime": matrix["runtime"],
        "scheduler": matrix["scheduler"],
        "source": source,
        "training_command": [
            matrix["runtime"]["python"],
            "-m",
            matrix["runtime"]["training_entrypoint"],
            "<CONFIG_YAML>",
        ],
        "runs": runs,
    }
    manifest_path = generated_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def _science_payload(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize only output-location fields for semantic parity checks."""

    payload = copy.deepcopy(config)
    payload["outputs"]["results_dir"] = "<RUN_DIR>"
    payload["experiment"]["checkpointing"][
        "checkpoint_dir"
    ] = "<RUN_DIR>/main_network/checkpoints"
    return payload


def validate_manifest(
    manifest_path: Path,
    *,
    compare_reference: bool = True,
) -> dict[str, Any]:
    """Validate standard configs without reading result or checkpoint payloads."""

    from dendritic_modeling.config import load_config

    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != "dendritic-rnn-standard-training-manifest-v1":
        raise ValueError("Unsupported manifest schema")
    runs = manifest.get("runs", [])
    if len(runs) != 20:
        raise ValueError(f"Expected exactly 20 runs, found {len(runs)}")
    if len({row["run_id"] for row in runs}) != 20:
        raise ValueError("Run IDs are not unique")
    if len({row["output_dir"] for row in runs}) != 20:
        raise ValueError("Output directories are not unique")

    matrix = _plain_yaml(Path(manifest["matrix_config"]["path"]))
    _validate_matrix(matrix)
    reference_root = Path(matrix["reference"]["immutable_bundle"]) / "configs"
    paired: dict[str, list[dict[str, Any]]] = {}

    for row in runs:
        config_path = Path(row["config"])
        if not config_path.is_file() or _sha256(config_path) != row["config_sha256"]:
            raise ValueError(f"Missing or changed config: {config_path}")
        raw = _plain_yaml(config_path)
        typed = load_config(str(config_path))
        if typed.data.dataset_name != "romo_delay_comparison_positive":
            raise ValueError(f"Unexpected task in {config_path}")
        if typed.model.core.type != "population_network":
            raise ValueError(f"Unexpected core in {config_path}")
        if typed.training.main.strategy != "recurrent":
            raise ValueError(f"Unexpected strategy in {config_path}")
        if typed.training.main.common.epochs != 400:
            raise ValueError(f"Unexpected epoch count in {config_path}")
        if not typed.experiment.strict_deterministic:
            raise ValueError(f"Strict determinism is disabled in {config_path}")
        if not typed.experiment.sealed_test or typed.experiment.allow_test_data:
            raise ValueError(f"The test seal is open in {config_path}")
        if raw["outputs"]["results_dir"] != row["output_dir"]:
            raise ValueError(f"Manifest output mismatch for {row['run_id']}")
        runtime_text = json.dumps(
            {
                "config": row["config"],
                "output": row["output_dir"],
                "command": manifest["training_command"],
            }
        )
        if "/drafts/" in runtime_text:
            raise ValueError(f"Draft-owned runtime path detected for {row['run_id']}")
        paired.setdefault(row["pairing_group"], []).append(row)

        if compare_reference:
            reference_path = reference_root / config_path.name
            if not reference_path.is_file():
                raise ValueError(
                    f"Missing immutable reference config: {reference_path}"
                )
            if _science_payload(raw) != _science_payload(_plain_yaml(reference_path)):
                raise ValueError(
                    f"Scientific config differs from immutable reference: {config_path}"
                )

    if sorted(paired) != [f"r{index:03d}" for index in range(10)]:
        raise ValueError("Pairing groups are incomplete")
    for pairing_group, rows in paired.items():
        if len(rows) != 2 or rows[0]["seeds"] != rows[1]["seeds"]:
            raise ValueError(f"Broken paired-seed contract for {pairing_group}")

    return {
        "manifest": str(manifest_path),
        "run_count": len(runs),
        "pair_count": len(paired),
        "reference_parity": bool(compare_reference),
        "training_source_sha256": manifest["source"]["training_source_sha256"],
    }


def _runtime_gpu_metadata(expected_gpu_type: str) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside a GPU training allocation")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Expected exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    name = torch.cuda.get_device_name(0)
    expected_token = "H200" if "h200" in expected_gpu_type.lower() else "H100"
    if expected_token not in name.upper():
        raise RuntimeError(f"Expected {expected_token}, allocated GPU is {name!r}")
    properties = torch.cuda.get_device_properties(0)
    return {
        "name": name,
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def run_one(
    *,
    config_path: Path,
    expected_partition: str,
    expected_gpu_type: str,
    expected_source_sha256: str,
    validate_only: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Run one standard config and write metadata-only lifecycle markers."""

    from dendritic_modeling.config import load_config

    config_path = config_path.resolve()
    config = load_config(str(config_path))
    run_dir = (output_dir or Path(config.outputs.results_dir)).resolve()
    actual_partition = os.environ.get("SLURM_JOB_PARTITION", "")
    if actual_partition and actual_partition != expected_partition:
        raise RuntimeError(
            f"Partition mismatch: expected {expected_partition}, got {actual_partition}"
        )
    if _tree_sha256() != expected_source_sha256:
        raise RuntimeError("Training source changed after manifest materialization")
    package_path = Path(__file__).resolve()
    if not package_path.is_relative_to(REPO_ROOT / "src"):
        raise RuntimeError(
            f"Launcher was not imported from repository src: {package_path}"
        )
    gpu = _runtime_gpu_metadata(expected_gpu_type)

    if run_dir.exists() and any(run_dir.iterdir()):
        if (run_dir / "COMPLETE.json").is_file():
            print(f"Already complete: {run_dir}")
            return
        raise RuntimeError(f"Refusing to overwrite non-empty run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = _utc_now()
    common = {
        "schema_version": "dendritic-rnn-standard-run-marker-v1",
        "run_id": config.outputs.run_name,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "repo_root": str(REPO_ROOT),
        "training_source_sha256": expected_source_sha256,
        "validate_only": validate_only,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "partition": actual_partition or expected_partition,
            "account": os.environ.get("SLURM_JOB_ACCOUNT"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
        },
        "gpu": gpu,
        "started_utc": started,
    }
    _atomic_json(run_dir / "RUNNING.json", common)

    command = [
        sys.executable,
        "-m",
        "dendritic_modeling.scripts.training.train_experiments",
        str(config_path),
    ]
    if validate_only:
        command.extend(["--validate-only", "--output_dir", str(run_dir)])

    environment = os.environ.copy()
    environment["PYTHONPATH"] = (
        str(REPO_ROOT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    )
    environment.setdefault("OMP_NUM_THREADS", "6")
    try:
        subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)
        required = (
            REQUIRED_PREFLIGHT_ARTIFACTS
            if validate_only
            else REQUIRED_TRAINING_ARTIFACTS
        )
        missing = [name for name in required if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(
                f"Standard trainer omitted required artifacts: {missing}"
            )
    except Exception as exc:
        failed = {
            **common,
            "finished_utc": _utc_now(),
            "state": "FAILED",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_json(run_dir / "FAILED.json", failed)
        raise

    complete = {
        **common,
        "finished_utc": _utc_now(),
        "state": "PREFLIGHT_COMPLETE" if validate_only else "COMPLETE",
        "required_artifacts": list(required),
    }
    _atomic_json(run_dir / "COMPLETE.json", complete)
    (run_dir / "RUNNING.json").unlink(missing_ok=True)


def _route(manifest: dict[str, Any], name: str, *, allow_h200: bool) -> dict[str, Any]:
    routes = manifest["scheduler"]
    if name not in {"primary", "h100_fallback", "h200_opt_in"}:
        raise ValueError(f"Unknown scheduler route: {name}")
    route = routes[name]
    if "h200" in route["gpu_type"].lower() and not allow_h200:
        raise ValueError("The H200 engineering route requires --allow-h200")
    return route


def _sbatch_command(
    *,
    manifest: dict[str, Any],
    row: dict[str, Any],
    route_name: str,
    allow_h200: bool,
    validate_only: bool = False,
    output_dir: Path | None = None,
) -> list[str]:
    route = _route(manifest, route_name, allow_h200=allow_h200)
    resources = manifest["scheduler"]["resources"]
    logs_dir = Path(manifest["output_root"]) / "_slurm"
    logs_dir.mkdir(parents=True, exist_ok=True)
    job_suffix = (
        "preflight" if validate_only else row["run_id"].replace("extreme-cube-", "")
    )
    job_name = f"dendrnn-xcube-{job_suffix}"
    runtime = [
        manifest["runtime"]["python"],
        "-m",
        "dendritic_modeling.scripts.experiments.dendritic_rnn_extreme_cube",
        "run-one",
        "--config",
        row["config"],
        "--expected-partition",
        route["partition"],
        "--expected-gpu-type",
        route["gpu_type"],
        "--expected-source-sha256",
        manifest["source"]["training_source_sha256"],
    ]
    if validate_only:
        if output_dir is None:
            raise ValueError("Preflight output directory is required")
        runtime.extend(["--validate-only", "--output-dir", str(output_dir)])
    wrapped = " && ".join(
        [
            f"cd {shlex.quote(str(REPO_ROOT))}",
            "export PYTHONPATH=" + shlex.quote(str(REPO_ROOT / "src")),
            shlex.join(runtime),
        ]
    )
    walltime = "00:30:00" if validate_only else str(resources["walltime"])
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
        f"--output={logs_dir}/%x_%j.out",
        f"--error={logs_dir}/%x_%j.err",
        f"--chdir={REPO_ROOT}",
        f"--wrap={wrapped}",
    ]


def submit(
    *,
    manifest_path: Path,
    route_name: str,
    allow_h200: bool,
    dry_run: bool,
    limit: int | None = None,
) -> dict[str, Any]:
    validate_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    selected = manifest["runs"][:limit] if limit is not None else manifest["runs"]
    submitted = []
    for row in selected:
        run_dir = Path(row["output_dir"])
        if run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError(f"Output directory is not empty: {run_dir}")
        command = _sbatch_command(
            manifest=manifest,
            row=row,
            route_name=route_name,
            allow_h200=allow_h200,
        )
        if dry_run:
            submitted.append({"run_id": row["run_id"], "command": command})
            continue
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(";", 1)[0]
        submitted.append({"run_id": row["run_id"], "job_id": job_id})

    receipt = {
        "schema_version": "dendritic-rnn-standard-submission-receipt-v1",
        "created_utc": _utc_now(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "route": route_name,
        "dry_run": dry_run,
        "jobs": submitted,
    }
    if not dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        receipt_path = (
            Path(manifest["output_root"]) / "_launch" / f"submission_{stamp}.json"
        )
        _atomic_json(receipt_path, receipt)
        receipt["receipt"] = str(receipt_path)
    return receipt


def submit_preflight(
    *,
    manifest_path: Path,
    route_name: str,
    allow_h200: bool,
    dry_run: bool,
) -> dict[str, Any]:
    validate_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    row = manifest["runs"][0]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(manifest["output_root"]) / "_preflight" / stamp
    command = _sbatch_command(
        manifest=manifest,
        row=row,
        route_name=route_name,
        allow_h200=allow_h200,
        validate_only=True,
        output_dir=output_dir,
    )
    payload: dict[str, Any] = {
        "schema_version": "dendritic-rnn-standard-preflight-submission-v1",
        "created_utc": _utc_now(),
        "manifest": str(manifest_path.resolve()),
        "route": route_name,
        "output_dir": str(output_dir),
        "command": command,
        "dry_run": dry_run,
    }
    if not dry_run:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload["job_id"] = result.stdout.strip().split(";", 1)[0]
        receipt_path = output_dir.parent / f"submission_{stamp}.json"
        _atomic_json(receipt_path, payload)
        payload["receipt"] = str(receipt_path)
    return payload


def submit_evaluations(
    *,
    training_receipt_path: Path,
    reference_manifest_path: Path,
    route_name: str,
    allow_h200: bool,
    dry_run: bool,
    base_source_snapshot_path: Path | None = None,
) -> dict[str, Any]:
    """Submit one frozen-bank evaluation after each successful training job."""

    training_receipt_path = training_receipt_path.resolve()
    receipt = json.loads(training_receipt_path.read_text())
    if receipt.get("schema_version") != "dendritic-rnn-standard-submission-receipt-v1":
        raise ValueError("Unsupported training submission receipt")
    if receipt.get("dry_run") is not False or len(receipt.get("jobs", [])) != 20:
        raise ValueError("Evaluation requires one real exact-20 training receipt")
    manifest_path = Path(receipt["manifest"]).resolve()
    if _sha256(manifest_path) != receipt["manifest_sha256"]:
        raise ValueError("Training manifest changed after submission")
    validate_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    reference_manifest_path = reference_manifest_path.resolve()
    if not reference_manifest_path.is_file():
        raise FileNotFoundError(reference_manifest_path)
    route = _route(manifest, route_name, allow_h200=allow_h200)
    job_ids = {row["run_id"]: row["job_id"] for row in receipt["jobs"]}
    if set(job_ids) != {row["run_id"] for row in manifest["runs"]}:
        raise ValueError("Training receipt run IDs differ from the exact-20 manifest")

    if base_source_snapshot_path is None:
        source_snapshot = _materialize_source_snapshot(Path(manifest["output_root"]))
    else:
        source_snapshot = _materialize_overlay_source_snapshot(
            Path(manifest["output_root"]),
            base_snapshot_root=base_source_snapshot_path,
            overlay_paths=("scripts/analysis/evaluate_dendritic_rnn_frozen_bank.py",),
        )
    snapshot_root = Path(source_snapshot["root"])
    evaluator = (
        snapshot_root
        / "src/dendritic_modeling/scripts/analysis/evaluate_dendritic_rnn_frozen_bank.py"
    )
    evaluator_sha256 = _sha256(evaluator)
    evaluation_runtime_source_sha256 = source_snapshot["training_runtime_source_sha256"]
    logs_dir = Path(manifest["output_root"]) / "_slurm"
    logs_dir.mkdir(parents=True, exist_ok=True)
    submitted = []
    for row in manifest["runs"]:
        run_id = row["run_id"]
        training_job_id = job_ids[run_id]
        accounting = subprocess.run(
            [
                "sacct",
                "-X",
                "-j",
                training_job_id,
                "-n",
                "-P",
                "-o",
                "State,ExitCode",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        records = [line.split("|") for line in accounting.splitlines() if line]
        if len(records) != 1 or len(records[0]) < 2:
            raise RuntimeError(
                f"Could not resolve training job state for {training_job_id}"
            )
        training_state = records[0][0].split("+", 1)[0]
        training_exit_code = records[0][1]
        dependency: str | None
        if training_state == "COMPLETED" and training_exit_code == "0:0":
            marker_path = Path(row["output_dir"]) / "COMPLETE.json"
            marker = (
                json.loads(marker_path.read_text()) if marker_path.is_file() else {}
            )
            if marker.get("state") != "COMPLETE" or marker.get("run_id") != run_id:
                raise RuntimeError(
                    f"Completed scheduler job lacks a valid run marker: {run_id}"
                )
            dependency = None
        elif training_state in {
            "PENDING",
            "RUNNING",
            "CONFIGURING",
            "COMPLETING",
        }:
            dependency = training_job_id
        else:
            raise RuntimeError(
                f"Training job {training_job_id} is not eligible: "
                f"{training_state}/{training_exit_code}"
            )
        evaluation_dir = Path(row["output_dir"]) / "evaluation"
        if evaluation_dir.exists() and any(evaluation_dir.iterdir()):
            raise RuntimeError(f"Evaluation directory is not empty: {evaluation_dir}")
        runtime = [
            manifest["runtime"]["python"],
            "-m",
            "dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_frozen_bank",
            "--training-manifest",
            str(manifest_path),
            "--reference-manifest",
            str(reference_manifest_path),
            "--run-id",
            run_id,
            "--device",
            "auto",
            "--batch-size",
            "128",
            "--expected-evaluator-sha256",
            evaluator_sha256,
            "--expected-runtime-source-sha256",
            evaluation_runtime_source_sha256,
        ]
        wrapped = " && ".join(
            [
                f"cd {shlex.quote(str(REPO_ROOT))}",
                "export PYTHONDONTWRITEBYTECODE=1",
                "export DENDRITIC_LOG_DIR="
                + shlex.quote(
                    str(
                        Path(manifest["output_root"])
                        / "_runtime/logs/evaluation"
                        / run_id
                    )
                ),
                "export PYTHONPATH=" + shlex.quote(str(snapshot_root / "src")),
                shlex.join(runtime),
            ]
        )
        job_name = "dendrnn-xeval-" + run_id.replace("extreme-cube-", "")
        command = [
            "sbatch",
            "--parsable",
            "--no-requeue",
            f"--job-name={job_name}",
            f"--partition={route['partition']}",
            f"--account={route['account']}",
            "--nodes=1",
            "--ntasks=1",
            f"--gres=gpu:{route['gpu_type']}:1",
            "--cpus-per-task=2",
            "--mem=6G",
            "--time=01:00:00",
            f"--output={logs_dir}/%x_%j.out",
            f"--error={logs_dir}/%x_%j.err",
            f"--chdir={REPO_ROOT}",
            f"--wrap={wrapped}",
        ]
        if dependency is not None:
            command.insert(3, f"--dependency=afterok:{dependency}")
        if dry_run:
            submitted.append(
                {
                    "run_id": run_id,
                    "training_job_id": training_job_id,
                    "training_state_at_submission": training_state,
                    "dependency": dependency,
                    "command": command,
                }
            )
            continue
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        submitted.append(
            {
                "run_id": run_id,
                "training_job_id": training_job_id,
                "training_state_at_submission": training_state,
                "dependency": dependency,
                "job_id": result.stdout.strip().split(";", 1)[0],
            }
        )

    payload = {
        "schema_version": "dendritic-rnn-standard-evaluation-submission-receipt-v1",
        "created_utc": _utc_now(),
        "training_receipt": str(training_receipt_path),
        "training_receipt_sha256": _sha256(training_receipt_path),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "reference_manifest": str(reference_manifest_path),
        "reference_manifest_sha256": _sha256(reference_manifest_path),
        "evaluator_sha256": evaluator_sha256,
        "source_snapshot": source_snapshot,
        "training_runtime_source_sha256": manifest["source"]["training_source_sha256"],
        "evaluation_runtime_source_sha256": evaluation_runtime_source_sha256,
        "runtime_source_relationship": (
            "Evaluation freezes the live standard source at evaluation submission. "
            "Any difference from the training-submission hash must be audited before "
            "scientific interpretation."
        ),
        "route": route_name,
        "resources": {"gpus": 1, "cpus": 2, "memory": "6G", "walltime": "01:00:00"},
        "dry_run": dry_run,
        "jobs": submitted,
    }
    if not dry_run:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        receipt_path = (
            Path(manifest["output_root"])
            / "_launch"
            / f"evaluation_submission_{stamp}.json"
        )
        _atomic_json(receipt_path, payload)
        payload["receipt"] = str(receipt_path)
    return payload


def submit_analysis(
    *,
    evaluation_receipt_path: Path,
    primary_manifest_path: Path,
    extension_reference_manifest_path: Path,
    runtime_attestation_path: Path,
    output_dir: Path | None,
    dry_run: bool,
) -> dict[str, Any]:
    """Submit one CPU-only collector/statistics/figure job after all evaluations."""

    evaluation_receipt_path = evaluation_receipt_path.resolve()
    receipt = json.loads(evaluation_receipt_path.read_text())
    expected_schema = "dendritic-rnn-standard-evaluation-submission-receipt-v1"
    if receipt.get("schema_version") != expected_schema:
        raise ValueError("Unsupported evaluation submission receipt")
    jobs = receipt.get("jobs", [])
    if receipt.get("dry_run") is not False or len(jobs) != 20:
        raise ValueError("Analysis requires one real exact-20 evaluation receipt")
    job_ids = [str(row["job_id"]) for row in jobs]
    if len(set(job_ids)) != 20 or any(not job_id.isdigit() for job_id in job_ids):
        raise ValueError("Evaluation receipt has invalid or duplicate job IDs")

    manifest_path = Path(receipt["manifest"]).resolve()
    if _sha256(manifest_path) != receipt["manifest_sha256"]:
        raise ValueError("Extension manifest changed after evaluation submission")
    validate_manifest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    manifest_rows = {row["run_id"]: row for row in manifest["runs"]}
    receipt_run_ids = {str(row["run_id"]) for row in jobs}
    if receipt_run_ids != set(manifest_rows):
        raise ValueError("Evaluation receipt run IDs differ from the exact-20 manifest")

    dependency_job_ids = []
    evaluation_states = []
    for job in jobs:
        run_id = str(job["run_id"])
        job_id = str(job["job_id"])
        accounting = subprocess.run(
            [
                "sacct",
                "-X",
                "-j",
                job_id,
                "-n",
                "-P",
                "-o",
                "State,ExitCode",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        records = [line.split("|") for line in accounting.splitlines() if line]
        if len(records) != 1 or len(records[0]) < 2:
            raise RuntimeError(f"Could not resolve evaluation job state for {job_id}")
        state = records[0][0].split("+", 1)[0]
        exit_code = records[0][1]
        dependency: str | None
        if state == "COMPLETED" and exit_code == "0:0":
            marker_path = (
                Path(manifest_rows[run_id]["output_dir"]) / "evaluation/COMPLETE.json"
            )
            marker = (
                json.loads(marker_path.read_text()) if marker_path.is_file() else {}
            )
            if marker.get("state") != "COMPLETE" or marker.get("run_id") != run_id:
                raise RuntimeError(
                    f"Completed evaluation job lacks a valid marker: {run_id}"
                )
            dependency = None
        elif state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
            dependency = job_id
            dependency_job_ids.append(job_id)
        else:
            raise RuntimeError(
                f"Evaluation job {job_id} is not eligible: {state}/{exit_code}"
            )
        evaluation_states.append(
            {
                "run_id": run_id,
                "job_id": job_id,
                "state_at_submission": state,
                "exit_code": exit_code,
                "dependency": dependency,
            }
        )
    primary_manifest_path = primary_manifest_path.resolve()
    extension_reference_manifest_path = extension_reference_manifest_path.resolve()
    for label, path in (
        ("primary manifest", primary_manifest_path),
        ("extension reference manifest", extension_reference_manifest_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
        if "/drafts/" in str(path):
            raise ValueError(f"Draft-owned {label} is forbidden: {path}")
    if (
        _sha256(extension_reference_manifest_path)
        != receipt["reference_manifest_sha256"]
    ):
        raise ValueError("Extension reference manifest differs from evaluation receipt")

    evaluation_source_snapshot = receipt.get("source_snapshot")
    if not isinstance(evaluation_source_snapshot, dict):
        raise ValueError("Evaluation receipt lacks an immutable source snapshot")
    evaluation_snapshot_root = Path(
        str(evaluation_source_snapshot.get("root"))
    ).resolve()
    if (
        not evaluation_snapshot_root.is_dir()
        or _package_tree_sha256(evaluation_snapshot_root)
        != evaluation_source_snapshot.get("package_sha256")
        or _tree_sha256(evaluation_snapshot_root)
        != receipt.get("evaluation_runtime_source_sha256")
    ):
        raise ValueError("Evaluation source snapshot is missing or changed")
    runtime_attestation_path = runtime_attestation_path.resolve()
    if not runtime_attestation_path.is_file():
        raise FileNotFoundError(runtime_attestation_path)
    runtime_attestation = json.loads(runtime_attestation_path.read_text())
    if runtime_attestation.get("training_runtime_source_sha256") != receipt.get(
        "training_runtime_source_sha256"
    ) or runtime_attestation.get("evaluation_runtime_source_sha256") != receipt.get(
        "evaluation_runtime_source_sha256"
    ):
        raise ValueError("Runtime attestation does not match the evaluation receipt")
    analysis_source_snapshot = _materialize_overlay_source_snapshot(
        Path(manifest["output_root"]),
        base_snapshot_root=evaluation_snapshot_root,
        overlay_paths=("scripts/analysis/analyze_dendritic_rnn_extreme_cube.py",),
    )
    analysis_snapshot_root = Path(analysis_source_snapshot["root"])
    analysis_script = (
        analysis_snapshot_root / "src/dendritic_modeling/scripts/analysis/"
        "analyze_dendritic_rnn_extreme_cube.py"
    )
    analysis_source_sha256 = _sha256(analysis_script)
    analysis_dir = (
        output_dir.resolve()
        if output_dir is not None
        else Path(manifest["output_root"]).resolve() / "analysis"
    )
    if "/drafts/" in str(analysis_dir):
        raise ValueError("Draft-owned analysis output is forbidden")
    if analysis_dir.exists() and any(analysis_dir.iterdir()):
        raise RuntimeError(f"Analysis directory is not empty: {analysis_dir}")

    runtime = [
        manifest["runtime"]["python"],
        "-m",
        "dendritic_modeling.scripts.analysis.analyze_dendritic_rnn_extreme_cube",
        "--primary-manifest",
        str(primary_manifest_path),
        "--extension-manifest",
        str(manifest_path),
        "--extension-reference-manifest",
        str(extension_reference_manifest_path),
        "--evaluation-receipt",
        str(evaluation_receipt_path),
        "--runtime-attestation",
        str(runtime_attestation_path),
        "--output-dir",
        str(analysis_dir),
        "--expected-source-sha256",
        analysis_source_sha256,
    ]
    wrapped = " && ".join(
        [
            f"cd {shlex.quote(str(REPO_ROOT))}",
            "export PYTHONDONTWRITEBYTECODE=1",
            "export DENDRITIC_LOG_DIR="
            + shlex.quote(
                str(Path(manifest["output_root"]) / "_runtime/logs/analysis")
            ),
            "export PYTHONPATH=" + shlex.quote(str(analysis_snapshot_root / "src")),
            "export OMP_NUM_THREADS=2",
            shlex.join(runtime),
        ]
    )
    logs_dir = Path(manifest["output_root"]) / "_slurm"
    logs_dir.mkdir(parents=True, exist_ok=True)
    resources = {
        "partition": "shared",
        "account": "bsabatini_lab",
        "gpus": 0,
        "cpus": 2,
        "memory": "4G",
        "walltime": "01:00:00",
        "rationale": (
            "Serial JSON validation, 10,000-draw seed bootstrap, exact 2^10 "
            "sign flips, and matplotlib rendering require no accelerator."
        ),
    }
    command = [
        "sbatch",
        "--parsable",
        "--no-requeue",
        "--job-name=dendrnn-xcube-analysis",
        f"--partition={resources['partition']}",
        f"--account={resources['account']}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={resources['cpus']}",
        f"--mem={resources['memory']}",
        f"--time={resources['walltime']}",
        f"--output={logs_dir}/%x_%j.out",
        f"--error={logs_dir}/%x_%j.err",
        f"--chdir={REPO_ROOT}",
        f"--wrap={wrapped}",
    ]
    if dependency_job_ids:
        command.insert(3, "--dependency=afterok:" + ":".join(dependency_job_ids))
    payload: dict[str, Any] = {
        "schema_version": "dendritic-rnn-extreme-cube-analysis-submission-v1",
        "created_utc": _utc_now(),
        "evaluation_receipt": str(evaluation_receipt_path),
        "evaluation_receipt_sha256": _sha256(evaluation_receipt_path),
        "prerequisite_job_ids": job_ids,
        "dependency_job_ids": dependency_job_ids,
        "evaluation_states": evaluation_states,
        "primary_manifest": str(primary_manifest_path),
        "primary_manifest_sha256": _sha256(primary_manifest_path),
        "extension_manifest": str(manifest_path),
        "extension_manifest_sha256": _sha256(manifest_path),
        "extension_reference_manifest": str(extension_reference_manifest_path),
        "extension_reference_manifest_sha256": _sha256(
            extension_reference_manifest_path
        ),
        "evaluation_source_snapshot": evaluation_source_snapshot,
        "analysis_source_snapshot": analysis_source_snapshot,
        "runtime_attestation": str(runtime_attestation_path),
        "runtime_attestation_sha256": _sha256(runtime_attestation_path),
        "analysis_source_sha256": analysis_source_sha256,
        "output_dir": str(analysis_dir),
        "resources": resources,
        "command": command,
        "dry_run": dry_run,
    }
    if not dry_run:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        payload["job_id"] = result.stdout.strip().split(";", 1)[0]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        receipt_path = (
            Path(manifest["output_root"])
            / "_launch"
            / f"analysis_submission_{stamp}.json"
        )
        _atomic_json(receipt_path, payload)
        payload["receipt"] = str(receipt_path)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    materialize_parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    materialize_parser.add_argument(
        "--generated-dir", type=Path, default=DEFAULT_GENERATED
    )
    materialize_parser.add_argument("--results-root", type=Path)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_GENERATED / "manifest.json"
    )
    validate_parser.add_argument("--skip-reference-parity", action="store_true")

    for name in ("submit", "preflight"):
        launch_parser = subparsers.add_parser(name)
        launch_parser.add_argument(
            "--manifest", type=Path, default=DEFAULT_GENERATED / "manifest.json"
        )
        launch_parser.add_argument(
            "--route",
            choices=("primary", "h100_fallback", "h200_opt_in"),
            default="primary",
        )
        launch_parser.add_argument("--allow-h200", action="store_true")
        launch_parser.add_argument("--dry-run", action="store_true")
        if name == "submit":
            launch_parser.add_argument("--limit", type=int)

    evaluation_parser = subparsers.add_parser("submit-evaluations")
    evaluation_parser.add_argument("--training-receipt", type=Path, required=True)
    evaluation_parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=Path(
            "/n/holylabs/kempner_dev/Users/hsafaai/dendrinet_revision_2026/"
            "bundles/extreme_placement_cube_priority_20260720_v14/manifest.json"
        ),
    )
    evaluation_parser.add_argument(
        "--route",
        choices=("primary", "h100_fallback", "h200_opt_in"),
        default="primary",
    )
    evaluation_parser.add_argument("--allow-h200", action="store_true")
    evaluation_parser.add_argument("--dry-run", action="store_true")
    evaluation_parser.add_argument("--base-source-snapshot", type=Path)

    analysis_parser = subparsers.add_parser("submit-analysis")
    analysis_parser.add_argument("--evaluation-receipt", type=Path, required=True)
    analysis_parser.add_argument(
        "--primary-manifest", type=Path, default=DEFAULT_PRIMARY_MANIFEST
    )
    analysis_parser.add_argument(
        "--extension-reference-manifest",
        type=Path,
        default=DEFAULT_EXTENSION_REFERENCE_MANIFEST,
    )
    analysis_parser.add_argument(
        "--runtime-attestation", type=Path, default=DEFAULT_RUNTIME_ATTESTATION
    )
    analysis_parser.add_argument("--output-dir", type=Path)
    analysis_parser.add_argument("--dry-run", action="store_true")

    run_parser = subparsers.add_parser("run-one")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--expected-partition", required=True)
    run_parser.add_argument("--expected-gpu-type", required=True)
    run_parser.add_argument("--expected-source-sha256", required=True)
    run_parser.add_argument("--validate-only", action="store_true")
    run_parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "materialize":
        result: Any = {
            "manifest": str(
                materialize(
                    base_path=args.base,
                    matrix_path=args.matrix,
                    generated_dir=args.generated_dir,
                    results_root=args.results_root,
                )
            )
        }
    elif args.command == "validate":
        result = validate_manifest(
            args.manifest,
            compare_reference=not args.skip_reference_parity,
        )
    elif args.command == "submit":
        result = submit(
            manifest_path=args.manifest,
            route_name=args.route,
            allow_h200=args.allow_h200,
            dry_run=args.dry_run,
            limit=args.limit,
        )
    elif args.command == "preflight":
        result = submit_preflight(
            manifest_path=args.manifest,
            route_name=args.route,
            allow_h200=args.allow_h200,
            dry_run=args.dry_run,
        )
    elif args.command == "submit-evaluations":
        result = submit_evaluations(
            training_receipt_path=args.training_receipt,
            reference_manifest_path=args.reference_manifest,
            route_name=args.route,
            allow_h200=args.allow_h200,
            dry_run=args.dry_run,
            base_source_snapshot_path=args.base_source_snapshot,
        )
    elif args.command == "submit-analysis":
        result = submit_analysis(
            evaluation_receipt_path=args.evaluation_receipt,
            primary_manifest_path=args.primary_manifest,
            extension_reference_manifest_path=args.extension_reference_manifest,
            runtime_attestation_path=args.runtime_attestation,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
        )
    elif args.command == "run-one":
        run_one(
            config_path=args.config,
            expected_partition=args.expected_partition,
            expected_gpu_type=args.expected_gpu_type,
            expected_source_sha256=args.expected_source_sha256,
            validate_only=args.validate_only,
            output_dir=args.output_dir,
        )
        return
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
