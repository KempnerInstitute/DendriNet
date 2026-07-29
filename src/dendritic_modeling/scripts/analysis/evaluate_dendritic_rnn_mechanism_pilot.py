"""Build and evaluate the fixed-soma mechanism pilot's held-out paired banks."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from dendritic_modeling.datasets.sequence.neuro_tasks import RomoDelayComparisonDataset
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_counterfactual_bank import (
    _atomic_json,
    _atomic_npz,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_frozen_bank import (
    _deterministic_evaluation,
    _evaluate_arrays,
    _load_exact_model,
    _semantic_state_sha256,
    _sha256,
    _sha256_json,
    evaluation_runtime_source_sha256,
)
from dendritic_modeling.scripts.experiments.dendritic_rnn_extreme_cube import (
    _package_tree_sha256,
    _tree_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs/dendritic_rnn/mechanism_identification_pilot_v1/evaluation.yaml"
)
CONFIG_SCHEMA = "dendritic-rnn-mechanism-pilot-evaluation-v1"
BANK_SCHEMA = "dendritic-rnn-mechanism-pilot-bank-index-v1"
RESULT_SCHEMA = "dendritic-rnn-mechanism-pilot-evaluation-result-v1"
RECEIPT_SCHEMA = "dendritic-rnn-mechanism-pilot-evaluation-receipt-v1"
EXPECTED_CONDITIONS = (
    "tau_distal_slow__rec_proximal_concentrated",
    "tau_distal_slow__rec_distal_concentrated",
    "tau_distal_fast__rec_proximal_concentrated",
    "tau_distal_fast__rec_distal_concentrated",
)


class MechanismPilotEvaluationError(RuntimeError):
    """Raised when the held-out pilot evaluation contract is violated."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def _require_external_path(path: Path, *, repo_root: Path, label: str) -> Path:
    """Require a v2 mutable artifact to live outside the source checkout."""

    resolved = path.resolve()
    source_root = repo_root.resolve()
    if resolved == source_root or resolved.is_relative_to(source_root):
        raise MechanismPilotEvaluationError(
            f"{label} must be outside the source repository: {resolved}"
        )
    return resolved


def _expected_sha256(path: Path, expected: str, *, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise MechanismPilotEvaluationError(f"Invalid expected {label} SHA-256")
    try:
        actual = _sha256(path.resolve(strict=True))
    except FileNotFoundError as exc:
        raise MechanismPilotEvaluationError(f"Missing {label}: {path}") from exc
    if actual != expected:
        raise MechanismPilotEvaluationError(f"{label} changed after submission")


def _validated_source_snapshot(
    config: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any] | None:
    """Validate the immutable package and training-source hashes for v2."""

    if config.get("study", {}).get("require_source_snapshot") is not True:
        return None
    snapshot = manifest.get("source_snapshot")
    if not isinstance(snapshot, dict):
        raise MechanismPilotEvaluationError(
            "Training manifest lacks an immutable source snapshot"
        )
    raw_root = snapshot.get("root")
    if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
        raise MechanismPilotEvaluationError("Source snapshot root must be absolute")
    try:
        root = Path(raw_root).resolve(strict=True)
    except FileNotFoundError as exc:
        raise MechanismPilotEvaluationError(
            f"Source snapshot is missing: {raw_root}"
        ) from exc
    if not root.is_dir() or not (root / "src/dendritic_modeling").is_dir():
        raise MechanismPilotEvaluationError(
            f"Source snapshot package is missing: {root}"
        )
    package_sha256 = _package_tree_sha256(root)
    training_sha256 = _tree_sha256(root)
    if package_sha256 != snapshot.get("package_sha256"):
        raise MechanismPilotEvaluationError("Source snapshot package hash mismatch")
    if training_sha256 != snapshot.get("training_runtime_source_sha256"):
        raise MechanismPilotEvaluationError("Source snapshot training hash mismatch")
    declared_training_sha256 = manifest.get("source", {}).get("training_source_sha256")
    if declared_training_sha256 not in (None, training_sha256):
        raise MechanismPilotEvaluationError(
            "Manifest training-source hash differs from its source snapshot"
        )
    return {
        **snapshot,
        "root": str(root),
        "package_sha256": package_sha256,
        "training_runtime_source_sha256": training_sha256,
    }


def _require_running_from_snapshot(snapshot: dict[str, Any] | None) -> None:
    if snapshot is None:
        return
    root = Path(str(snapshot["root"]))
    expected = (
        root / "src/dendritic_modeling/scripts/analysis/"
        "evaluate_dendritic_rnn_mechanism_pilot.py"
    ).resolve(strict=True)
    if Path(__file__).resolve() != expected:
        raise MechanismPilotEvaluationError(
            "Snapshot-required evaluation was not imported from the frozen source"
        )


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise MechanismPilotEvaluationError(f"Invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise MechanismPilotEvaluationError(f"JSON root is not an object: {path}")
    return payload


def load_config(path: Path = DEFAULT_CONFIG) -> tuple[dict[str, Any], Path]:
    path = path.resolve(strict=True)
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise MechanismPilotEvaluationError("Invalid pilot evaluation YAML") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CONFIG_SCHEMA:
        raise MechanismPilotEvaluationError("Unsupported pilot evaluation schema")
    study = payload.get("study", {})
    bank = payload.get("bank", {})
    evaluation = payload.get("evaluation", {})
    resources = payload.get("resources", {})
    if study.get("status") != "post_outcome_descriptive_pilot":
        raise MechanismPilotEvaluationError("Pilot status changed")
    if study.get("inference") != "no_null_tests_three_seed_go_no_go":
        raise MechanismPilotEvaluationError("Pilot inferential scope changed")
    if tuple(evaluation.get("conditions", ())) != EXPECTED_CONDITIONS:
        raise MechanismPilotEvaluationError("Condition order changed")
    expected_bank = {
        "task": "romo_delay_comparison_positive",
        "split": "pilot_ood",
        "n_trials": 4096,
        "stimulus_duration": 20,
        "response_duration": 8,
        "noise_std": 0.15,
        "positive_input_encoding": True,
        "tuning_dim": 8,
        "pair_sampling": "balanced_second_stimulus_v1",
        "min_delay": 240,
        "max_delay": 320,
    }
    changed = {
        key: (bank.get(key), expected)
        for key, expected in expected_bank.items()
        if bank.get(key) != expected
    }
    if changed:
        raise MechanismPilotEvaluationError(
            f"Held-out bank contract changed: {changed}"
        )
    if resources.get("checkpoints_per_job") != 4:
        raise MechanismPilotEvaluationError(
            "Each GPU job must evaluate four checkpoints"
        )
    return payload, path


def _manifest_groups(
    config: dict[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, list[dict[str, Any]]]]:
    manifest_path = _resolve(config["study"]["training_manifest"]).resolve(strict=True)
    manifest = _strict_json(manifest_path)
    snapshot = _validated_source_snapshot(config, manifest)
    if snapshot is not None:
        raw_repo_root = manifest.get("repo_root")
        if not isinstance(raw_repo_root, str) or not Path(raw_repo_root).is_absolute():
            raise MechanismPilotEvaluationError(
                "Snapshot-required manifest lacks an absolute source repository root"
            )
        source_repo = Path(raw_repo_root)
        _require_external_path(
            manifest_path,
            repo_root=source_repo,
            label="training manifest",
        )
        output_root = Path(str(manifest.get("output_root", "")))
        if not output_root.is_absolute():
            raise MechanismPilotEvaluationError(
                "Snapshot-required manifest output root must be absolute"
            )
        _require_external_path(
            output_root,
            repo_root=source_repo,
            label="training output root",
        )
        for key, label in (
            ("bank_root", "evaluation bank root"),
            ("output_root", "evaluation output root"),
        ):
            configured_root = Path(str(config["study"].get(key, "")))
            if not configured_root.is_absolute():
                raise MechanismPilotEvaluationError(
                    f"Snapshot-required {label} must be absolute"
                )
            _require_external_path(
                configured_root,
                repo_root=source_repo,
                label=label,
            )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest.get("runs", []):
        groups[str(row["pairing_group"])].append(dict(row))
    if sorted(groups) != ["r000", "r001", "r002"]:
        raise MechanismPilotEvaluationError(
            "Training manifest lacks three paired groups"
        )
    for name, rows in groups.items():
        if (
            len(rows) != 4
            or tuple(row["condition"] for row in rows) != EXPECTED_CONDITIONS
        ):
            raise MechanismPilotEvaluationError(
                f"Incomplete or reordered quartet: {name}"
            )
        if any(row["seeds"] != rows[0]["seeds"] for row in rows[1:]):
            raise MechanismPilotEvaluationError(f"Seed streams differ within {name}")
    return manifest, manifest_path, dict(groups)


def _bank_paths(config: dict[str, Any], pairing_group: str) -> tuple[Path, Path]:
    root = _resolve(config["study"]["bank_root"])
    return root / f"{pairing_group}.npz", root / f"{pairing_group}.index.json"


def _pair_audit(dataset: RomoDelayComparisonDataset) -> None:
    groups: dict[int, list[int]] = defaultdict(list)
    for index, pair_id in enumerate(dataset.pair_ids.tolist()):
        groups[int(pair_id)].append(index)
    if len(groups) * 2 != len(dataset) or any(
        len(rows) != 2 for rows in groups.values()
    ):
        raise MechanismPilotEvaluationError("Balanced bank pair cardinality failed")
    start = int(dataset.stimulus_duration)
    for rows in groups.values():
        if sorted(int(dataset.targets[index]) for index in rows) != [0, 1]:
            raise MechanismPilotEvaluationError("Pair labels are not opposite")
        left, right = rows
        if not torch.equal(dataset.inputs[left, start:], dataset.inputs[right, start:]):
            raise MechanismPilotEvaluationError("Pair inputs differ after delay onset")
        if int(dataset.seq_lengths[left]) != int(dataset.seq_lengths[right]):
            raise MechanismPilotEvaluationError("Pair sequence lengths differ")


def build_banks(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Generate three immutable, seed-paired OOD banks before outcomes are read."""

    config, config_path = load_config(config_path)
    manifest, manifest_path, groups = _manifest_groups(config)
    source_snapshot = _validated_source_snapshot(config, manifest)
    _require_running_from_snapshot(source_snapshot)
    bank_contract = config["bank"]
    records = []
    for pairing_group, rows in groups.items():
        bank_path, index_path = _bank_paths(config, pairing_group)
        if bank_path.exists() or index_path.exists():
            raise MechanismPilotEvaluationError(
                f"Bank output already exists: {pairing_group}"
            )
        evaluation_seed = int(rows[0]["seeds"]["evaluation_seed"])
        kwargs = {
            key: bank_contract[key]
            for key in (
                "n_trials",
                "frequency_values",
                "stimulus_duration",
                "response_duration",
                "noise_std",
                "positive_input_encoding",
                "tuning_dim",
                "pair_sampling",
                "min_delay",
                "max_delay",
            )
        }
        kwargs["n_samples"] = kwargs.pop("n_trials")
        kwargs["frequency_values"] = tuple(
            float(value) for value in kwargs["frequency_values"]
        )
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(evaluation_seed)
            dataset = RomoDelayComparisonDataset(**kwargs)
        _pair_audit(dataset)
        bank_id = f"mechanism-pilot-ood-{pairing_group}"
        trial_ids = [f"{bank_id}:trial-{index:06d}" for index in range(len(dataset))]
        trials = []
        for index, trial_id in enumerate(trial_ids):
            target = int(dataset.targets[index])
            pair_id = int(dataset.pair_ids[index])
            trials.append(
                {
                    "trial_id": trial_id,
                    "pair_id": f"{bank_id}:pair-{pair_id:06d}",
                    "pair_role": (
                        "lower_first_positive"
                        if target == 1
                        else "higher_first_negative"
                    ),
                    "target": target,
                    "delay": int(dataset.delays[index]),
                    "frequency1_index": int(dataset.frequency1_indices[index]),
                    "frequency2_index": int(dataset.frequency2_indices[index]),
                    "comparison_distance_index": int(
                        dataset.comparison_distance_indices[index]
                    ),
                }
            )
        arrays = {
            "inputs": dataset.inputs.numpy().astype(np.float32, copy=False),
            "targets": dataset.targets.numpy().astype(np.int64, copy=False),
            "seq_lengths": dataset.seq_lengths.numpy().astype(np.int64, copy=False),
            "trial_ids": np.asarray(trial_ids),
        }
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_npz(bank_path, arrays)
        index = {
            "schema_version": BANK_SCHEMA,
            "bank_id": bank_id,
            "split": bank_contract["split"],
            "task": bank_contract["task"],
            "pairing_group": pairing_group,
            "evaluation_seed": evaluation_seed,
            "n_trials": len(dataset),
            "trial_ids": trial_ids,
            "trials": trials,
            "trial_ids_sha256": _sha256_json(trial_ids),
            "trial_metadata_sha256": _sha256_json(trials),
            "bank_sha256": _sha256(bank_path),
            "config": str(config_path),
            "config_sha256": _sha256(config_path),
            "training_manifest": str(manifest_path),
            "training_manifest_sha256": _sha256(manifest_path),
            "pair_audit": "PASSED_EXACT",
        }
        if source_snapshot is not None:
            index["source_snapshot"] = source_snapshot
        _atomic_json(index_path, index, readonly=True)
        records.append(
            {
                "pairing_group": pairing_group,
                "bank": str(bank_path),
                "bank_sha256": index["bank_sha256"],
                "index": str(index_path),
                "index_sha256": _sha256(index_path),
                "evaluation_seed": evaluation_seed,
            }
        )
    return {"state": "COMPLETE", "banks": records}


def _load_bank(
    config: dict[str, Any], pairing_group: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    bank_path, index_path = _bank_paths(config, pairing_group)
    index = _strict_json(index_path.resolve(strict=True))
    if (
        index.get("schema_version") != BANK_SCHEMA
        or index.get("pairing_group") != pairing_group
    ):
        raise MechanismPilotEvaluationError("Bank index identity mismatch")
    if _sha256(bank_path) != index.get("bank_sha256"):
        raise MechanismPilotEvaluationError("Bank hash mismatch")
    if _sha256_json(index.get("trial_ids")) != index.get("trial_ids_sha256"):
        raise MechanismPilotEvaluationError("Trial ID hash mismatch")
    if _sha256_json(index.get("trials")) != index.get("trial_metadata_sha256"):
        raise MechanismPilotEvaluationError("Trial metadata hash mismatch")
    with np.load(bank_path, allow_pickle=False) as archive:
        arrays = {key: np.array(archive[key], copy=True) for key in archive.files}
    if arrays["inputs"].dtype != np.float32 or arrays["inputs"].shape[0] != 4096:
        raise MechanismPilotEvaluationError("Unexpected bank tensor")
    arrays["trial_ids"] = arrays["trial_ids"].tolist()
    if arrays["trial_ids"] != index["trial_ids"]:
        raise MechanismPilotEvaluationError("NPZ/index trial order mismatch")
    return index, arrays


def _metric_summary(
    predictions: list[dict[str, Any]], trials: list[dict[str, Any]]
) -> dict[str, Any]:
    targets = np.asarray([row["target"] for row in predictions], dtype=np.int64)
    logits = np.asarray([row["logits"] for row in predictions], dtype=np.float64)
    predicted = np.asarray([row["prediction"] for row in predictions], dtype=np.int64)
    target_logits = logits[np.arange(len(logits)), targets]
    other_logits = logits[np.arange(len(logits)), 1 - targets]
    margins = target_logits - other_logits
    log_scores = target_logits - np.logaddexp(logits[:, 0], logits[:, 1])
    delays = np.asarray([row["delay"] for row in trials], dtype=np.int64)
    summary = {
        "n_trials": len(predictions),
        "accuracy": float(np.mean(predicted == targets)),
        "target_log_score": float(np.mean(log_scores, dtype=np.float64)),
        "target_margin": float(np.mean(margins, dtype=np.float64)),
        "cross_entropy": float(-np.mean(log_scores, dtype=np.float64)),
        "prediction_count_class1": int(np.sum(predicted == 1)),
    }
    strata = []
    for low, high in ((240, 260), (261, 280), (281, 300), (301, 320)):
        selected = (delays >= low) & (delays <= high)
        strata.append(
            {
                "delay_low": low,
                "delay_high": high,
                "n_trials": int(selected.sum()),
                "accuracy": float(np.mean(predicted[selected] == targets[selected])),
                "target_log_score": float(
                    np.mean(log_scores[selected], dtype=np.float64)
                ),
                "target_margin": float(np.mean(margins[selected], dtype=np.float64)),
            }
        )
    summary["delay_strata"] = strata
    return summary


def run_one(
    *,
    config_path: Path,
    pairing_group: str,
    expected_evaluator_sha256: str,
    expected_runtime_source_sha256: str,
    expected_config_sha256: str,
    expected_manifest_sha256: str,
    expected_bank_index_sha256: str,
) -> dict[str, Any]:
    """Evaluate one paired quartet sequentially on a single GPU."""

    if _sha256(Path(__file__).resolve()) != expected_evaluator_sha256:
        raise MechanismPilotEvaluationError("Evaluator changed after submission")
    if evaluation_runtime_source_sha256() != expected_runtime_source_sha256:
        raise MechanismPilotEvaluationError("Model runtime changed after submission")
    config_path = config_path.resolve(strict=True)
    _expected_sha256(config_path, expected_config_sha256, label="evaluation config")
    config, config_path = load_config(config_path)
    manifest_path = _resolve(config["study"]["training_manifest"])
    _expected_sha256(manifest_path, expected_manifest_sha256, label="training manifest")
    manifest, manifest_path, groups = _manifest_groups(config)
    source_snapshot = _validated_source_snapshot(config, manifest)
    _require_running_from_snapshot(source_snapshot)
    if pairing_group not in groups:
        raise MechanismPilotEvaluationError(f"Unknown pairing group: {pairing_group}")
    expected_training_source_sha256 = (
        source_snapshot["training_runtime_source_sha256"]
        if source_snapshot is not None
        else manifest.get("source", {}).get("training_source_sha256")
    )
    if not isinstance(expected_training_source_sha256, str):
        raise MechanismPilotEvaluationError(
            "Training manifest lacks its training-source hash"
        )
    for row in groups[pairing_group]:
        run_dir = Path(row["output_dir"]).resolve()
        marker = _strict_json((run_dir / "COMPLETE.json").resolve(strict=True))
        if marker.get("state") != "COMPLETE" or marker.get("run_id") != row["run_id"]:
            raise MechanismPilotEvaluationError(
                f"Training marker mismatch: {row['run_id']}"
            )
        if marker.get("config_sha256") != row.get("config_sha256"):
            raise MechanismPilotEvaluationError(
                f"Training marker config hash mismatch: {row['run_id']}"
            )
        if marker.get("training_source_sha256") != expected_training_source_sha256:
            raise MechanismPilotEvaluationError(
                f"Training marker source hash mismatch: {row['run_id']}"
            )
    expected_partition = config["resources"]["partition"]
    if os.environ.get("SLURM_JOB_PARTITION") != expected_partition:
        raise MechanismPilotEvaluationError("Evaluation ran on the wrong partition")
    if not torch.cuda.is_available():
        raise MechanismPilotEvaluationError("Pilot evaluation requires CUDA")
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name:
        raise MechanismPilotEvaluationError(f"Unexpected accelerator: {gpu_name}")
    bank_index_path = _bank_paths(config, pairing_group)[1]
    _expected_sha256(
        bank_index_path,
        expected_bank_index_sha256,
        label=f"{pairing_group} bank index",
    )
    index, bank = _load_bank(config, pairing_group)
    if index.get("config_sha256") != expected_config_sha256:
        raise MechanismPilotEvaluationError(
            f"Bank index is bound to a different config: {pairing_group}"
        )
    if index.get("training_manifest_sha256") != expected_manifest_sha256:
        raise MechanismPilotEvaluationError(
            f"Bank index is bound to a different manifest: {pairing_group}"
        )
    output_dir = _resolve(config["study"]["output_root"]) / pairing_group
    if output_dir.exists() and any(output_dir.iterdir()):
        raise MechanismPilotEvaluationError(
            f"Evaluation output is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    condition_results = []
    prediction_arrays: dict[str, np.ndarray] = {
        "targets": np.asarray(bank["targets"], dtype=np.int64),
        "seq_lengths": np.asarray(bank["seq_lengths"], dtype=np.int64),
    }
    for row in groups[pairing_group]:
        run_dir = Path(row["output_dir"]).resolve()
        checkpoint = (run_dir / "final_model.pt").resolve(strict=True)
        best_checkpoint = (run_dir / "main_network/standard_best_model.pt").resolve(
            strict=True
        )
        resolved_config = (run_dir / "config.json").resolve(strict=True)
        with _deterministic_evaluation(int(index["evaluation_seed"])):
            model, typed_config, state_sha256 = _load_exact_model(
                config_path=resolved_config,
                checkpoint_path=checkpoint,
                best_checkpoint_path=best_checkpoint,
                device=device,
            )
            if (
                not typed_config.experiment.sealed_test
                or typed_config.experiment.allow_test_data
            ):
                raise MechanismPilotEvaluationError("Training test seal is not intact")
            initial_state_keys = set(model.state_dict())
            predictions = _evaluate_arrays(
                model,
                bank,
                device=device,
                batch_size=int(config["evaluation"]["batch_size"]),
            )
            final_state = model.state_dict()
            if (
                not initial_state_keys.issubset(final_state)
                or _semantic_state_sha256(
                    {key: final_state[key] for key in initial_state_keys}
                )
                != state_sha256
            ):
                raise MechanismPilotEvaluationError(
                    "Model state changed during evaluation"
                )
        logits = np.asarray(
            [record["logits"] for record in predictions], dtype=np.float32
        )
        prediction_arrays[f"logits__{row['condition']}"] = logits
        condition_results.append(
            {
                "run_id": row["run_id"],
                "condition": row["condition"],
                "checkpoint_sha256": _sha256(checkpoint),
                "checkpoint_semantic_state_sha256": state_sha256,
                "resolved_config_sha256": _sha256(resolved_config),
                "metrics": _metric_summary(predictions, index["trials"]),
            }
        )
        del model
        torch.cuda.empty_cache()
    prediction_path = output_dir / "predictions.npz"
    _atomic_npz(prediction_path, prediction_arrays)
    result = {
        "schema_version": RESULT_SCHEMA,
        "state": "COMPLETE",
        "created_utc": _utc_now(),
        "pairing_group": pairing_group,
        "replication_unit": "paired_trained_model_seed",
        "inferential_status": "descriptive_three_seed_pilot_no_null_tests",
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": expected_manifest_sha256,
        "bank_index": str(_bank_paths(config, pairing_group)[1]),
        "bank_index_sha256": expected_bank_index_sha256,
        "bank_sha256": index["bank_sha256"],
        "predictions": str(prediction_path),
        "predictions_sha256": _sha256(prediction_path),
        "evaluation_runtime_source_sha256": expected_runtime_source_sha256,
        "source_snapshot": source_snapshot,
        "device": {
            "name": gpu_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "conditions": condition_results,
    }
    result_path = output_dir / "result.json"
    _atomic_json(result_path, result, readonly=True)
    return {"state": "COMPLETE", "result": str(result_path)}


def submit(
    config_path: Path = DEFAULT_CONFIG, *, dry_run: bool = False
) -> dict[str, Any]:
    """Submit three quartet evaluations with afterok dependencies."""

    config, config_path = load_config(config_path)
    manifest, manifest_path, groups = _manifest_groups(config)
    source_snapshot = _validated_source_snapshot(config, manifest)
    require_snapshot = source_snapshot is not None
    source_repo = Path(str(manifest.get("repo_root", REPO_ROOT))).resolve()
    if require_snapshot:
        _require_external_path(
            config_path, repo_root=source_repo, label="evaluation config"
        )
    training_receipt_path = _resolve(config["study"]["training_receipt"]).resolve(
        strict=True
    )
    if require_snapshot:
        _require_external_path(
            training_receipt_path,
            repo_root=source_repo,
            label="training receipt",
        )
    receipt = _strict_json(training_receipt_path)
    manifest_sha256 = _sha256(manifest_path)
    if require_snapshot:
        if (
            receipt.get("state") != "SUBMITTED"
            or receipt.get("dry_run") is not False
            or receipt.get("preflight") is not False
        ):
            raise MechanismPilotEvaluationError(
                "Snapshot evaluation requires a finalized real training receipt"
            )
        if Path(str(receipt.get("manifest", ""))).resolve() != manifest_path:
            raise MechanismPilotEvaluationError(
                "Training receipt points to a different manifest"
            )
        if receipt.get("manifest_sha256") != manifest_sha256:
            raise MechanismPilotEvaluationError(
                "Training receipt manifest hash mismatch"
            )
    receipt_jobs = receipt.get("jobs", [])
    if not isinstance(receipt_jobs, list):
        raise MechanismPilotEvaluationError("Training receipt jobs are invalid")
    training_jobs = {
        str(row.get("run_id")): str(row.get("job_id"))
        for row in receipt_jobs
        if isinstance(row, dict)
    }
    if set(training_jobs) != {
        row["run_id"] for rows in groups.values() for row in rows
    } or len(receipt_jobs) != len(training_jobs):
        raise MechanismPilotEvaluationError("Training receipt does not match manifest")
    if any(not job_id.isdigit() for job_id in training_jobs.values()):
        raise MechanismPilotEvaluationError(
            "Training receipt contains a non-numeric Slurm job ID"
        )
    resources = config["resources"]
    runtime = config["runtime"]
    if require_snapshot:
        runtime_root = Path(str(source_snapshot["root"]))
        evaluator_path = (
            runtime_root / "src/dendritic_modeling/scripts/analysis/"
            "evaluate_dendritic_rnn_mechanism_pilot.py"
        ).resolve(strict=True)
        runtime_sha256 = evaluation_runtime_source_sha256(runtime_root)
    else:
        runtime_root = REPO_ROOT
        evaluator_path = Path(__file__).resolve()
        runtime_sha256 = evaluation_runtime_source_sha256()
    evaluator_sha256 = _sha256(evaluator_path)
    config_sha256 = _sha256(config_path)
    output_root = _resolve(config["study"]["output_root"])
    raw_log_root = config["study"].get("log_root")
    log_root = (
        _resolve(str(raw_log_root))
        if raw_log_root is not None
        else output_root / "_slurm"
    )
    if require_snapshot:
        if raw_log_root is not None and not Path(str(raw_log_root)).is_absolute():
            raise MechanismPilotEvaluationError(
                "Snapshot-required log root must be absolute"
            )
        _require_external_path(log_root, repo_root=source_repo, label="log root")
    log_root.mkdir(parents=True, exist_ok=True)
    if require_snapshot and not os.access(log_root, os.W_OK):
        raise MechanismPilotEvaluationError(f"Log root is not writable: {log_root}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_evaluation_receipt = config["study"].get("evaluation_receipt")
    if raw_evaluation_receipt is not None:
        if require_snapshot and not Path(str(raw_evaluation_receipt)).is_absolute():
            raise MechanismPilotEvaluationError(
                "Snapshot-required evaluation receipt must be absolute"
            )
        evaluation_receipt_path = _resolve(str(raw_evaluation_receipt))
    else:
        evaluation_receipt_path = output_root / "_launch" / f"submission_{stamp}.json"
    if require_snapshot:
        _require_external_path(
            evaluation_receipt_path,
            repo_root=source_repo,
            label="evaluation receipt",
        )
    if not dry_run and evaluation_receipt_path.exists():
        raise MechanismPilotEvaluationError(
            f"Refusing to overwrite evaluation receipt: {evaluation_receipt_path}"
        )

    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "created_utc": _utc_now(),
        "state": "DRY_RUN" if dry_run else "SUBMITTING",
        "config": str(config_path),
        "config_sha256": config_sha256,
        "training_manifest": str(manifest_path),
        "training_manifest_sha256": manifest_sha256,
        "training_receipt": str(training_receipt_path),
        "training_receipt_sha256": _sha256(training_receipt_path),
        "evaluator_sha256": evaluator_sha256,
        "runtime_source_sha256": runtime_sha256,
        "source_snapshot": source_snapshot,
        "runtime_root": str(runtime_root),
        "dry_run": dry_run,
        "jobs": [],
    }
    if not dry_run:
        result["receipt"] = str(evaluation_receipt_path)
        _atomic_json(evaluation_receipt_path, result, readonly=False)

    try:
        for pairing_group, rows in groups.items():
            dependencies = ":".join(training_jobs[row["run_id"]] for row in rows)
            bank_index_path = _bank_paths(config, pairing_group)[1].resolve(strict=True)
            bank_index_sha256 = _sha256(bank_index_path)
            runtime_argv = [
                str(runtime["python"]),
                "-m",
                "dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_mechanism_pilot",
                "run-one",
                "--config",
                str(config_path),
                "--pairing-group",
                pairing_group,
                "--expected-evaluator-sha256",
                evaluator_sha256,
                "--expected-runtime-source-sha256",
                runtime_sha256,
                "--expected-config-sha256",
                config_sha256,
                "--expected-manifest-sha256",
                manifest_sha256,
                "--expected-bank-index-sha256",
                bank_index_sha256,
            ]
            environment_commands = []
            if require_snapshot:
                runtime_log_dir = log_root / "runtime" / pairing_group
                if not dry_run:
                    runtime_log_dir.mkdir(parents=True, exist_ok=True)
                environment_commands.extend(
                    [
                        "export PYTHONDONTWRITEBYTECODE=1",
                        "export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
                        "export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}",
                        "export DENDRITIC_LOG_DIR=" + shlex.quote(str(runtime_log_dir)),
                    ]
                )
            environment_commands.append(
                f"export PYTHONPATH={shlex.quote(str(runtime_root / 'src'))}"
            )
            wrapped = " && ".join(
                [
                    f"cd {shlex.quote(str(runtime_root))}",
                    *environment_commands,
                    shlex.join(runtime_argv),
                ]
            )
            command = [
                "sbatch",
                "--parsable",
                "--no-requeue",
                f"--job-name=dendrnn-mpilot-eval-{pairing_group}",
                f"--dependency=afterok:{dependencies}",
                f"--partition={resources['partition']}",
                f"--account={resources['account']}",
                "--nodes=1",
                "--ntasks=1",
                f"--gres=gpu:{resources['gpu_type']}:{resources['gpus']}",
                f"--cpus-per-task={resources['cpus_per_task']}",
                f"--mem={resources['memory']}",
                f"--time={resources['walltime']}",
                f"--output={log_root}/%x_%j.out",
                f"--error={log_root}/%x_%j.err",
                f"--chdir={runtime_root}",
                f"--wrap={wrapped}",
            ]
            job_id = None
            if not dry_run:
                completed = subprocess.run(
                    command, check=True, capture_output=True, text=True
                )
                job_id = completed.stdout.strip().split(";", 1)[0]
                if not job_id.isdigit():
                    raise MechanismPilotEvaluationError(
                        f"Unexpected sbatch job ID: {job_id!r}"
                    )
            result["jobs"].append(
                {
                    "pairing_group": pairing_group,
                    "job_id": job_id,
                    "dependencies": dependencies.split(":"),
                    "bank_index": str(bank_index_path),
                    "bank_index_sha256": bank_index_sha256,
                    "command": command,
                }
            )
            if not dry_run:
                result["updated_utc"] = _utc_now()
                _atomic_json(evaluation_receipt_path, result, readonly=False)
    except Exception as exc:
        if not dry_run:
            result.update(
                {
                    "state": "SUBMISSION_FAILED",
                    "finished_utc": _utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            _atomic_json(evaluation_receipt_path, result, readonly=True)
        raise

    if not dry_run:
        result["state"] = "SUBMITTED"
        result["finished_utc"] = _utc_now()
        _atomic_json(evaluation_receipt_path, result, readonly=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-banks")
    build.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    submit_parser = subparsers.add_parser("submit")
    submit_parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    submit_parser.add_argument("--dry-run", action="store_true")
    run = subparsers.add_parser("run-one")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--pairing-group", choices=("r000", "r001", "r002"), required=True)
    run.add_argument("--expected-evaluator-sha256", required=True)
    run.add_argument("--expected-runtime-source-sha256", required=True)
    run.add_argument("--expected-config-sha256", required=True)
    run.add_argument("--expected-manifest-sha256", required=True)
    run.add_argument("--expected-bank-index-sha256", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "build-banks":
        result = build_banks(args.config)
    elif args.command == "submit":
        result = submit(args.config, dry_run=args.dry_run)
    else:
        result = run_one(
            config_path=args.config,
            pairing_group=args.pairing_group,
            expected_evaluator_sha256=args.expected_evaluator_sha256,
            expected_runtime_source_sha256=args.expected_runtime_source_sha256,
            expected_config_sha256=args.expected_config_sha256,
            expected_manifest_sha256=args.expected_manifest_sha256,
            expected_bank_index_sha256=args.expected_bank_index_sha256,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["build_banks", "load_config", "run_one", "submit"]
