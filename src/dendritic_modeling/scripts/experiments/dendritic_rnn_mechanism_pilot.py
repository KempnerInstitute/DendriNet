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
) -> tuple[str, dict[str, Any]]:
    config = copy.deepcopy(base)
    run_id = f"mechanism-pilot-{replicate['replicate']}-{condition['name']}"
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


def materialize(
    *, matrix_path: Path = DEFAULT_MATRIX, generated_dir: Path = DEFAULT_GENERATED
) -> Path:
    """Materialize twelve standalone standard training YAMLs."""

    matrix, matrix_path = _load_matrix(matrix_path)
    base, base_path = _merge_base(matrix)
    output_root = (REPO_ROOT / str(matrix["output_root"])).resolve()
    generated_dir = generated_dir.resolve()
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
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": matrix["experiment_id"],
        "created_utc": _utc_now(),
        "repo_root": str(REPO_ROOT),
        "matrix_config": {"path": str(matrix_path), "sha256": _sha256(matrix_path)},
        "base_config": {"path": str(base_path), "sha256": _sha256(base_path)},
        "output_root": str(output_root),
        "design": matrix["design"],
        "runtime": matrix["runtime"],
        "scheduler": matrix["scheduler"],
        "source": _git_metadata(),
        "runs": runs,
    }
    manifest_path = generated_dir / "manifest.json"
    _atomic_json(manifest_path, manifest)
    return manifest_path


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate configs, matching, pairing, and standard-trainer compatibility."""

    from dendritic_modeling.config import load_config

    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("Unsupported mechanism-pilot manifest")
    matrix, _ = _load_matrix(Path(manifest["matrix_config"]["path"]))
    if len(manifest.get("runs", [])) != 12:
        raise ValueError("Mechanism-pilot manifest must contain twelve runs")
    paired: dict[str, list[dict[str, Any]]] = {}
    signatures: dict[str, dict[str, Any]] = {}
    for row in manifest["runs"]:
        path = Path(row["config"])
        if not path.is_file() or _sha256(path) != row["config_sha256"]:
            raise ValueError(f"Generated config changed: {path}")
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
        "training_source_sha256": manifest["source"]["training_source_sha256"],
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
    logs.mkdir(parents=True, exist_ok=True)
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
        str(manifest["source"]["training_source_sha256"]),
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
        f"--chdir={REPO_ROOT}",
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
    manifest = json.loads(manifest_path.read_text())
    selected = (
        [row for row in manifest["runs"] if row["pairing_group"] == "r000"]
        if preflight
        else list(manifest["runs"])
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    submitted = []
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
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            job_id = result.stdout.strip().split(";", 1)[0]
        submitted.append(
            {
                "run_id": row["run_id"],
                "condition": row["condition"],
                "job_id": job_id,
                "output_dir": str(output_dir or row["output_dir"]),
                "command": command,
            }
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "created_utc": _utc_now(),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "route": route_name,
        "preflight": preflight,
        "dry_run": dry_run,
        "jobs": submitted,
    }
    if not dry_run:
        kind = "preflight" if preflight else "training"
        receipt_path = (
            Path(manifest["output_root"])
            / "_launch"
            / f"{kind}_submission_{stamp}.json"
        )
        _atomic_json(receipt_path, receipt)
        receipt["receipt"] = str(receipt_path)
    return receipt


def run_one(
    *,
    config_path: Path,
    expected_partition: str,
    expected_gpu_type: str,
    expected_source_sha256: str,
    validate_only: bool,
    output_dir: Path | None,
) -> None:
    """Delegate one generated YAML to the standard source training lifecycle."""

    from dendritic_modeling.scripts.experiments.dendritic_rnn_extreme_cube import (
        run_one as standard_run_one,
    )

    if _tree_sha256() != expected_source_sha256:
        raise RuntimeError("Training source changed after pilot materialization")
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
    run_parser.add_argument("--validate-only", action="store_true")
    run_parser.add_argument("--output-dir", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "materialize":
        result: Any = {
            "manifest": str(
                materialize(matrix_path=args.matrix, generated_dir=args.generated_dir)
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
            validate_only=args.validate_only,
            output_dir=args.output_dir,
        )
        result = {"state": "COMPLETE"}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["materialize", "run_one", "submit", "validate_manifest"]
