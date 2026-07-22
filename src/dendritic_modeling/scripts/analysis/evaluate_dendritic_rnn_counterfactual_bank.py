"""Evaluate the four RNN crossover checkpoints on one frozen counterfactual bank.

One invocation handles one paired trained-model seed and evaluates all four
primary crossover cells sequentially on a single GPU.  The evaluator consumes
only repository-owned source code plus a YAML study config.  It never imports
or writes analysis code under ``drafts``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from dendritic_modeling.scripts.analysis.analyze_dendritic_rnn_extreme_cube import (
    COLLECTION_SCHEMA,
    N_REPLICATES,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_frozen_bank import (
    _deterministic_evaluation,
    _load_exact_model,
    _semantic_state_sha256,
    _terminal_logits,
    evaluation_runtime_source_sha256,
)

STUDY_SCHEMA = "dendritic-rnn-counterfactual-psychometric-study-v1"
LEDGER_SCHEMA = "dendritic-rnn-counterfactual-psychometric-bank-ledger-v1"
INDEX_SCHEMA = "dendritic-rnn-counterfactual-psychometric-bank-index-v1"
RESULT_SCHEMA = "dendritic-rnn-counterfactual-psychometric-result-v1"
MARKER_SCHEMA = "dendritic-rnn-counterfactual-psychometric-marker-v1"
REPLICATE_MARKER_SCHEMA = (
    "dendritic-rnn-counterfactual-psychometric-replicate-marker-v1"
)
REPO_ROOT = Path(__file__).resolve().parents[4]
ROLE_CELLS = {
    "hierarchical_aligned": ("hierarchical", "proximal", "distal"),
    "hierarchical_reverse": ("hierarchical", "distal", "proximal"),
    "reversed_aligned": ("reversed", "proximal", "distal"),
    "reversed_reverse": ("reversed", "distal", "proximal"),
}
REQUIRED_ARRAYS = {
    "inputs",
    "targets",
    "seq_lengths",
    "trial_ids",
    "receiver_indices",
    "receiver_trial_ids",
    "receiver_first_ranks",
    "target_second_ranks",
    "comparison_defined",
    "identity_inputs",
    "identity_targets",
    "identity_seq_lengths",
    "identity_trial_ids",
    "identity_source_row_indices",
    "identity_first_ranks",
    "identity_second_ranks",
}


class CounterfactualEvaluationError(RuntimeError):
    """Raised when evaluation or provenance violates the frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise CounterfactualEvaluationError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise CounterfactualEvaluationError(f"JSON root must be an object: {path}")
    return payload


def _regular_file(path: Path, label: str, *, readonly: bool = False) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise CounterfactualEvaluationError(f"Missing {label}: {path}") from exc
    if path.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise CounterfactualEvaluationError(f"{label} must be a regular file: {path}")
    if readonly and stat.S_IMODE(resolved.stat().st_mode) != 0o444:
        raise CounterfactualEvaluationError(f"{label} must have mode 0444: {path}")
    return resolved


def _atomic_json(
    path: Path, payload: Mapping[str, Any], *, readonly: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)
    if readonly:
        path.chmod(0o444)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    path.chmod(0o444)


def _resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def load_study_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = _regular_file(path, "study config")
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise CounterfactualEvaluationError("Invalid study YAML") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != STUDY_SCHEMA:
        raise CounterfactualEvaluationError("Unsupported study config schema")
    study = payload.get("study")
    evaluation = payload.get("evaluation")
    if not isinstance(study, Mapping) or not isinstance(evaluation, Mapping):
        raise CounterfactualEvaluationError("Study config is incomplete")
    configured_roles = {
        role: tuple(cell) for role, cell in study.get("roles", {}).items()
    }
    if configured_roles != ROLE_CELLS or study.get("replicates") != N_REPLICATES:
        raise CounterfactualEvaluationError("Study pairing design changed")
    if evaluation.get("identity_bridge") != "bitwise_equal_to_frozen_ordinary_logits":
        raise CounterfactualEvaluationError("Identity bridge contract changed")
    if evaluation.get("rotated_partition_audit") != "bitwise_equal":
        raise CounterfactualEvaluationError("Partition audit contract changed")
    return payload, path


def _ledger_bank(
    ledger_path: Path,
    replicate: int,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, Path]:
    ledger_path = _regular_file(ledger_path, "counterfactual ledger", readonly=True)
    ledger = _strict_json(ledger_path)
    if (
        ledger.get("schema_version") != LEDGER_SCHEMA
        or ledger.get("status") != "FROZEN_BEFORE_CONFIRMATORY_CHECKPOINT_ACCESS"
    ):
        raise CounterfactualEvaluationError("Counterfactual ledger is not frozen")
    rows = ledger.get("banks")
    if not isinstance(rows, list) or len(rows) != N_REPLICATES:
        raise CounterfactualEvaluationError("Counterfactual ledger must bind ten banks")
    matches = [row for row in rows if row.get("replicate_index") == replicate]
    if len(matches) != 1:
        raise CounterfactualEvaluationError("Ledger replicate binding is ambiguous")
    row = dict(matches[0])
    root = ledger_path.parent
    bank_path = _regular_file(
        root / row["bank_file"], "counterfactual bank", readonly=True
    )
    index_path = _regular_file(
        root / row["index_file"], "counterfactual index", readonly=True
    )
    if _sha256(bank_path) != row.get("bank_sha256"):
        raise CounterfactualEvaluationError("Counterfactual bank hash mismatch")
    if _sha256(index_path) != row.get("index_sha256"):
        raise CounterfactualEvaluationError("Counterfactual index hash mismatch")
    index = _strict_json(index_path)
    if index.get("schema_version") != INDEX_SCHEMA:
        raise CounterfactualEvaluationError("Unsupported counterfactual index schema")
    expected = {
        "bank_id": row["bank_id"],
        "bank_sha256": row["bank_sha256"],
        "base_seed": row["base_seed"],
        "generation_seed": row["generation_seed"],
        "n_counterfactual_rows": 4096,
        "n_identity_rows": 1024,
        "n_receivers": 1024,
    }
    mismatches = {
        key: (index.get(key), value)
        for key, value in expected.items()
        if index.get(key) != value
    }
    if mismatches:
        raise CounterfactualEvaluationError(
            f"Counterfactual index mismatch: {mismatches}"
        )
    return row, index, bank_path, index_path, ledger_path


def load_counterfactual_arrays(bank_path: Path) -> dict[str, np.ndarray]:
    """Load and validate the arrays needed by the executor and downstream analysis."""

    try:
        with np.load(bank_path, allow_pickle=False) as archive:
            missing = REQUIRED_ARRAYS - set(archive.files)
            if missing:
                raise CounterfactualEvaluationError(
                    f"Counterfactual bank lacks arrays: {sorted(missing)}"
                )
            arrays = {
                name: np.array(archive[name], copy=True) for name in REQUIRED_ARRAYS
            }
    except (OSError, ValueError) as exc:
        raise CounterfactualEvaluationError(
            "Could not load counterfactual bank"
        ) from exc
    if (
        arrays["inputs"].shape != (4096, 148, 11)
        or arrays["inputs"].dtype != np.float32
    ):
        raise CounterfactualEvaluationError(
            "Counterfactual inputs have invalid metadata"
        )
    if (
        arrays["identity_inputs"].shape != (1024, 148, 11)
        or arrays["identity_inputs"].dtype != np.float32
    ):
        raise CounterfactualEvaluationError("Identity inputs have invalid metadata")
    for prefix, count in (("", 4096), ("identity_", 1024)):
        if (
            arrays[f"{prefix}targets"].shape != (count,)
            or arrays[f"{prefix}targets"].dtype != np.int64
        ):
            raise CounterfactualEvaluationError(f"{prefix}targets metadata changed")
        if (
            arrays[f"{prefix}seq_lengths"].shape != (count,)
            or arrays[f"{prefix}seq_lengths"].dtype != np.int64
        ):
            raise CounterfactualEvaluationError(f"{prefix}length metadata changed")
    if (
        not np.isfinite(arrays["inputs"]).all()
        or not np.isfinite(arrays["identity_inputs"]).all()
    ):
        raise CounterfactualEvaluationError(
            "Counterfactual bank contains non-finite input"
        )
    if not np.array_equal(arrays["receiver_indices"], np.repeat(np.arange(1024), 4)):
        raise CounterfactualEvaluationError("Receiver-major order changed")
    if not np.array_equal(
        arrays["target_second_ranks"], np.tile(np.arange(1, 5), 1024)
    ):
        raise CounterfactualEvaluationError("Target-rank order changed")
    expected_targets = (
        arrays["target_second_ranks"] > arrays["receiver_first_ranks"]
    ).astype(np.int64)
    if not np.array_equal(arrays["targets"], expected_targets):
        raise CounterfactualEvaluationError("Counterfactual labels changed")
    expected_defined = arrays["target_second_ranks"] != arrays["receiver_first_ranks"]
    if not np.array_equal(arrays["comparison_defined"], expected_defined):
        raise CounterfactualEvaluationError("Comparison-defined mask changed")
    if not np.array_equal(
        arrays["identity_trial_ids"], arrays["receiver_trial_ids"][::4]
    ):
        raise CounterfactualEvaluationError("Identity/receiver order changed")
    return arrays


def _collection_rows(
    collection: Mapping[str, Any],
    replicate: int,
) -> dict[str, dict[str, Any]]:
    if collection.get("schema_version") != COLLECTION_SCHEMA:
        raise CounterfactualEvaluationError("Unsupported cube collection schema")
    rows = collection.get("runs")
    if not isinstance(rows, list) or len(rows) != 80:
        raise CounterfactualEvaluationError("Cube collection must contain 80 runs")
    selected = {}
    for role, cell in ROLE_CELLS.items():
        matches = [
            row
            for row in rows
            if row.get("replicate") == f"r{replicate:03d}"
            and tuple(row.get(name) for name in ("tau", "ff", "rec")) == cell
        ]
        if len(matches) != 1 or matches[0].get("source") != "primary":
            raise CounterfactualEvaluationError(
                f"Missing primary collection row: {role}"
            )
        selected[role] = dict(matches[0])
    return selected


def _evaluate_logits(
    model: torch.nn.Module,
    inputs: np.ndarray,
    lengths: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    order: np.ndarray | None = None,
) -> np.ndarray:
    count = inputs.shape[0]
    if order is None:
        order = np.arange(count)
    if not np.array_equal(np.sort(order), np.arange(count)):
        raise CounterfactualEvaluationError("Evaluation order is not a permutation")
    output = np.empty((count, 2), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            indices = order[start : start + batch_size]
            x = torch.from_numpy(np.ascontiguousarray(inputs[indices])).to(device)
            seq_lengths = torch.from_numpy(np.ascontiguousarray(lengths[indices])).to(
                device
            )
            logits = _terminal_logits(
                model(x, seq_lengths=seq_lengths),
                seq_lengths=seq_lengths,
                batch_size=len(indices),
                device=device,
            )
            if logits.shape[1] != 2 or not torch.isfinite(logits).all():
                raise CounterfactualEvaluationError("Model returned invalid logits")
            output[indices] = (
                logits.detach().to(device="cpu", dtype=torch.float32).numpy()
            )
    return output


def _expected_identity_logits(
    prediction_payload: Mapping[str, Any], identity_trial_ids: Sequence[str]
) -> np.ndarray:
    predictions = prediction_payload.get("predictions")
    if not isinstance(predictions, list):
        raise CounterfactualEvaluationError(
            "Ordinary prediction artifact is incomplete"
        )
    by_id = {row.get("trial_id"): row.get("logits") for row in predictions}
    if len(by_id) != len(predictions):
        raise CounterfactualEvaluationError("Ordinary trial IDs are not unique")
    try:
        expected = np.asarray([by_id[str(trial_id)] for trial_id in identity_trial_ids])
    except KeyError as exc:
        raise CounterfactualEvaluationError(
            "Identity trial is absent from ordinary output"
        ) from exc
    if (
        expected.shape != (len(identity_trial_ids), 2)
        or not np.isfinite(expected).all()
    ):
        raise CounterfactualEvaluationError("Ordinary identity logits are invalid")
    return expected.astype(np.float32)


def _evaluate_role(
    *,
    role: str,
    row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    index: Mapping[str, Any],
    provenance: Mapping[str, Any],
    output_dir: Path,
    public_output_dir: Path,
    device: torch.device,
    batch_size: int,
    rotation: int,
) -> dict[str, Any]:
    prediction_path = _regular_file(
        Path(str(row["prediction_artifact"])), "ordinary predictions"
    )
    if _sha256(prediction_path) != row.get("prediction_sha256"):
        raise CounterfactualEvaluationError("Ordinary prediction hash mismatch")
    prediction_payload = _strict_json(prediction_path)
    source = index.get("source")
    if not isinstance(source, Mapping):
        raise CounterfactualEvaluationError("Counterfactual index lacks source binding")
    if (
        row.get("bank_id") != source.get("bank_id")
        or row.get("bank_sha256") != source.get("bank_sha256")
        or row.get("index_sha256") != source.get("index_sha256")
    ):
        raise CounterfactualEvaluationError(
            "Ordinary/counterfactual bank join mismatch"
        )
    if prediction_payload.get("run_id") != row.get("run_id"):
        raise CounterfactualEvaluationError("Ordinary run binding mismatch")

    run_dir = Path(str(row["training_complete_marker"])).resolve().parent
    config_path = _regular_file(run_dir / "config.json", "resolved config")
    checkpoint_path = _regular_file(run_dir / "final_model.pt", "checkpoint")
    best_path = _regular_file(
        run_dir / "main_network/standard_best_model.pt", "best checkpoint"
    )
    if _sha256(config_path) != prediction_payload.get(
        "resolved_config_sha256"
    ) or _sha256(checkpoint_path) != prediction_payload.get("checkpoint_sha256"):
        raise CounterfactualEvaluationError(
            "Model artifact changed after ordinary evaluation"
        )

    checkpoint_sha256 = _sha256(checkpoint_path)
    with _deterministic_evaluation(int(source["evaluation_seed"])):
        model, config, state_sha256 = _load_exact_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=best_path,
            device=device,
        )
        if config.experiment.evaluation_seed != source["evaluation_seed"]:
            raise CounterfactualEvaluationError("Evaluation seed mismatch")
        if (
            config.experiment.sealed_test is not True
            or config.experiment.allow_test_data
        ):
            raise CounterfactualEvaluationError(
                "Resolved config violates the test seal"
            )
        if int(config.model.encoder.params.input_dim) != 11:
            raise CounterfactualEvaluationError(
                "Resolved model input dimension changed"
            )
        initial_keys = set(model.state_dict())
        counterfactual_logits = _evaluate_logits(
            model,
            arrays["inputs"],
            arrays["seq_lengths"],
            device=device,
            batch_size=batch_size,
        )
        rotated_order = np.roll(np.arange(4096), -rotation)
        rotated_logits = _evaluate_logits(
            model,
            arrays["inputs"],
            arrays["seq_lengths"],
            device=device,
            batch_size=batch_size,
            order=rotated_order,
        )
        if not np.array_equal(counterfactual_logits, rotated_logits):
            maximum = float(np.max(np.abs(counterfactual_logits - rotated_logits)))
            raise CounterfactualEvaluationError(
                f"Rotated partition audit failed; max_abs_difference={maximum}"
            )
        identity_logits = _evaluate_logits(
            model,
            arrays["identity_inputs"],
            arrays["identity_seq_lengths"],
            device=device,
            batch_size=batch_size,
        )
        expected_identity = _expected_identity_logits(
            prediction_payload, arrays["identity_trial_ids"].tolist()
        )
        if not np.array_equal(identity_logits, expected_identity):
            maximum = float(np.max(np.abs(identity_logits - expected_identity)))
            raise CounterfactualEvaluationError(
                f"Identity bridge failed; max_abs_difference={maximum}"
            )
        final_state = model.state_dict()
        if (
            _semantic_state_sha256({key: final_state[key] for key in initial_keys})
            != state_sha256
        ):
            raise CounterfactualEvaluationError("Model state changed during evaluation")
        transient = sorted(set(final_state) - initial_keys)
        allowed = ("._last_forward_param_tensor", "._last_forward_weight_mask")
        if any(not key.endswith(allowed) for key in transient):
            raise CounterfactualEvaluationError(
                f"Unexpected transient model state: {transient}"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if _sha256(checkpoint_path) != checkpoint_sha256:
        raise CounterfactualEvaluationError("Checkpoint changed during evaluation")
    output_dir.mkdir(parents=True, exist_ok=False)
    logits_path = output_dir / "logits.npz"
    _atomic_npz(
        logits_path,
        {
            "counterfactual_logits": counterfactual_logits,
            "identity_logits": identity_logits,
        },
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "role": role,
        "cell": list(ROLE_CELLS[role]),
        "replicate": row["replicate"],
        "run_id": row["run_id"],
        "collection_run_binding": {
            "prediction_artifact": str(prediction_path),
            "prediction_sha256": row["prediction_sha256"],
        },
        "counterfactual_bank": dict(provenance),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha256},
        "checkpoint_semantic_state_sha256": state_sha256,
        "logits": {
            "path": str(public_output_dir / "logits.npz"),
            "sha256": _sha256(logits_path),
            "counterfactual_shape": list(counterfactual_logits.shape),
            "identity_shape": list(identity_logits.shape),
            "dtype": str(counterfactual_logits.dtype),
        },
        "audits": {
            "identity_bridge": "PASSED_BITWISE",
            "rotated_partition": "PASSED_BITWISE",
            "audit_rotation": rotation,
            "batch_size": batch_size,
            "state_immutability": "PASSED",
        },
        "device": {
            "resolved": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if device.type == "cuda" else None,
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
    }
    result_path = output_dir / "result.json"
    _atomic_json(result_path, result)
    marker = {
        "schema_version": MARKER_SCHEMA,
        "state": "COMPLETE",
        "run_id": row["run_id"],
        "result": {
            "path": str(public_output_dir / "result.json"),
            "sha256": _sha256(result_path),
        },
        "logits": {
            "path": str(public_output_dir / "logits.npz"),
            "sha256": _sha256(logits_path),
        },
    }
    _atomic_json(output_dir / "COMPLETE.json", marker)
    return marker


def evaluate_replicate(
    *,
    study_config_path: Path,
    replicate: int,
    expected_evaluator_sha256: str | None = None,
    expected_runtime_source_sha256: str | None = None,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate all four primary cells for one paired trained-model seed."""

    if replicate not in range(N_REPLICATES):
        raise ValueError("replicate must be in 0,...,9")
    source_path = Path(__file__).resolve()
    if (
        expected_evaluator_sha256 is not None
        and _sha256(source_path) != expected_evaluator_sha256
    ):
        raise CounterfactualEvaluationError("Evaluator changed after job submission")
    runtime_hash = evaluation_runtime_source_sha256()
    if (
        expected_runtime_source_sha256 is not None
        and runtime_hash != expected_runtime_source_sha256
    ):
        raise CounterfactualEvaluationError(
            "Model runtime changed after job submission"
        )
    config, study_config_path = load_study_config(study_config_path)
    if (
        expected_config_sha256 is not None
        and _sha256(study_config_path) != expected_config_sha256
    ):
        raise CounterfactualEvaluationError("Study config changed after job submission")
    study = config["study"]
    evaluation = config["evaluation"]
    runtime = config.get("runtime")
    if not isinstance(runtime, Mapping):
        raise CounterfactualEvaluationError("Study config lacks a pinned runtime")
    if torch.__version__ != runtime.get("torch") or torch.version.cuda != runtime.get(
        "cuda"
    ):
        raise CounterfactualEvaluationError(
            "Active Torch/CUDA build differs from the pinned study runtime"
        )
    collection_path = _regular_file(
        _resolve_repo_path(str(study["collection"])), "cube collection"
    )
    if "/drafts/" in str(collection_path):
        raise CounterfactualEvaluationError("Draft-owned collection is forbidden")
    collection = _strict_json(collection_path)
    rows = _collection_rows(collection, replicate)
    ledger_row, index, bank_path, index_path, ledger_path = _ledger_bank(
        _resolve_repo_path(str(study["counterfactual_ledger"])), replicate
    )
    if any(row["seeds"]["seed"] != ledger_row["base_seed"] for row in rows.values()):
        raise CounterfactualEvaluationError(
            "Training/counterfactual base-seed mismatch"
        )
    arrays = load_counterfactual_arrays(bank_path)
    source = index["source"]
    provenance = {
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "bank_id": index["bank_id"],
        "bank_path": str(bank_path),
        "bank_sha256": _sha256(bank_path),
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
        "source_bank_id": source["bank_id"],
        "source_bank_sha256": source["bank_sha256"],
    }

    device = torch.device(str(evaluation["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CounterfactualEvaluationError("CUDA was requested but is unavailable")
    batch_size = int(evaluation["batch_size"])
    rotation = int(evaluation["audit_rotation"])
    if batch_size <= 0 or 4096 % batch_size or 1024 % batch_size:
        raise CounterfactualEvaluationError("Batch size must divide both bank sizes")
    if not 0 < rotation < batch_size:
        raise CounterfactualEvaluationError("Audit rotation must be within one batch")

    output_root = _resolve_repo_path(str(study["output_root"])).resolve()
    if "/drafts/" in str(output_root):
        raise CounterfactualEvaluationError("Draft-owned output is forbidden")
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / f"r{replicate:03d}"
    if final_dir.exists():
        raise CounterfactualEvaluationError(
            f"Replicate output already exists: {final_dir}"
        )
    staging = output_root / f"._r{replicate:03d}.{os.getpid()}.staging"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        markers = {}
        for role in ROLE_CELLS:
            markers[role] = _evaluate_role(
                role=role,
                row=rows[role],
                arrays=arrays,
                index=index,
                provenance=provenance,
                output_dir=staging / role,
                public_output_dir=final_dir / role,
                device=device,
                batch_size=batch_size,
                rotation=rotation,
            )
        marker = {
            "schema_version": REPLICATE_MARKER_SCHEMA,
            "state": "COMPLETE",
            "replicate": f"r{replicate:03d}",
            "study_config": {
                "path": str(study_config_path),
                "sha256": _sha256(study_config_path),
            },
            "collection": {
                "path": str(collection_path),
                "sha256": _sha256(collection_path),
            },
            "evaluator": {"path": str(source_path), "sha256": _sha256(source_path)},
            "runtime_source_sha256": runtime_hash,
            "roles": markers,
        }
        _atomic_json(staging / "COMPLETE.json", marker)
        staging.replace(final_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-config", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--expected-evaluator-sha256")
    parser.add_argument("--expected-runtime-source-sha256")
    parser.add_argument("--expected-config-sha256")
    args = parser.parse_args()
    marker = evaluate_replicate(
        study_config_path=args.study_config,
        replicate=args.replicate,
        expected_evaluator_sha256=args.expected_evaluator_sha256,
        expected_runtime_source_sha256=args.expected_runtime_source_sha256,
        expected_config_sha256=args.expected_config_sha256,
    )
    print(
        json.dumps(
            {
                "state": marker["state"],
                "replicate": marker["replicate"],
                "roles": sorted(marker["roles"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "CounterfactualEvaluationError",
    "evaluate_replicate",
    "load_counterfactual_arrays",
    "load_study_config",
]
