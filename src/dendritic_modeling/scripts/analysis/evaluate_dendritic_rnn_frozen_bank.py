"""Evaluate one completed dendritic-RNN run on its paired frozen OOD bank.

This is the repository-owned successor to the paper-draft evaluation wrapper.
It consumes a standard training manifest plus the immutable reference manifest,
reconstructs the model through ``src/dendritic_modeling``, strictly loads the
checkpoint, evaluates the ordered bank, and writes a provenance-bound
``evaluation/predictions.json`` artifact. It never imports code from ``drafts``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dendritic_modeling.config import load_config
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model

PREDICTION_SCHEMA = "dendritic-rnn-evaluation-predictions-v1"
EVALUATION_MARKER_SCHEMA = "dendritic-rnn-standard-evaluation-marker-v1"
POPULATION_NOOP_SCHEMA = "dendritic-rnn-population-noop-audit-v1"
POPULATION_NOOP_METHOD = (
    "ordinary-classifier-forward-vs-manual-population-no-target-unroll"
)
REPO_ROOT = Path(__file__).resolve().parents[4]
EVALUATION_RUNTIME_SOURCE_ROOTS = (
    "config",
    "datasets",
    "models",
    "networks",
    "training",
    "utils",
    "scripts/script_utils",
    "scripts/training",
)


class FrozenBankEvaluationError(RuntimeError):
    """Raised when evaluation inputs or outputs violate the frozen contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def evaluation_runtime_source_sha256(repo_root: Path = REPO_ROOT) -> str:
    """Hash the live standard source that can affect model reconstruction."""

    package_root = repo_root / "src/dendritic_modeling"
    paths: list[Path] = []
    for relative_root in EVALUATION_RUNTIME_SOURCE_ROOTS:
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


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite token {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise FrozenBankEvaluationError(f"Invalid JSON artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise FrozenBankEvaluationError(f"JSON root must be an object: {path}")
    return payload


def _atomic_json(
    path: Path, payload: dict[str, Any], *, readonly: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    if readonly:
        path.chmod(0o444)


def _find_run(manifest: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    matches = [row for row in manifest.get("runs", []) if row.get("run_id") == run_id]
    if len(matches) != 1:
        raise FrozenBankEvaluationError(
            f"Expected one manifest row for {run_id!r}, found {len(matches)}"
        )
    return dict(matches[0])


def _regular_file(path: Path, label: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FrozenBankEvaluationError(f"Missing {label}: {path}") from exc
    if path.is_symlink() or resolved.is_symlink() or not resolved.is_file():
        raise FrozenBankEvaluationError(f"{label} must be a regular file: {path}")
    return resolved


def _resolve_reference_path(reference_manifest: Path, raw: str) -> Path:
    candidate = Path(raw)
    return (
        candidate if candidate.is_absolute() else reference_manifest.parent / candidate
    )


def _validate_seed_parity(
    training_row: Mapping[str, Any], reference_row: Mapping[str, Any]
) -> None:
    training = training_row["seeds"]
    reference = reference_row["seeds"]
    mapping = {
        "seed": "base_seed",
        "dataset_seed": "dataset_seed",
        "split_seed": "split_seed",
        "model_seed": "model_init_seed",
        "topology_seed": "topology_seed",
        "loader_seed": "loader_seed",
        "evaluation_seed": "evaluation_seed",
        "probe_seed": "probe_seed",
    }
    mismatches = {
        field: (training.get(field), reference.get(reference_field))
        for field, reference_field in mapping.items()
        if training.get(field) != reference.get(reference_field)
    }
    if mismatches:
        raise FrozenBankEvaluationError(
            f"Training/reference seed mismatch: {mismatches}"
        )


def _load_bank(
    *, reference_manifest: Path, contract: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray | list[str]], dict[str, Any], Path, Path]:
    bank_path = _regular_file(
        _resolve_reference_path(reference_manifest, contract["bank"]["path"]),
        "frozen bank",
    )
    index_path = _regular_file(
        _resolve_reference_path(reference_manifest, contract["index"]["path"]),
        "frozen bank index",
    )
    if _sha256(bank_path) != contract["bank"]["sha256"]:
        raise FrozenBankEvaluationError("Frozen bank SHA256 mismatch")
    if _sha256(index_path) != contract["index"]["sha256"]:
        raise FrozenBankEvaluationError("Frozen bank index SHA256 mismatch")
    index = _strict_json(index_path)
    expected_header = {
        "bank_id": contract["bank_id"],
        "split": contract["split"],
        "evaluation_seed": contract["evaluation_seed"],
        "n_trials": contract["n_trials"],
        "trial_ids_sha256": contract["trial_ids_sha256"],
        "trial_metadata_sha256": contract["trial_metadata_sha256"],
        "bank_sha256": contract["bank"]["sha256"],
    }
    for key, expected in expected_header.items():
        if index.get(key) != expected:
            raise FrozenBankEvaluationError(f"Frozen bank index {key} mismatch")
    if index.get("split") == "test_ood":
        raise FrozenBankEvaluationError("Sealed analysis may not access test_ood")
    trial_ids = index.get("trial_ids")
    trials = index.get("trials")
    if not isinstance(trial_ids, list) or not isinstance(trials, list):
        raise FrozenBankEvaluationError("Frozen bank index lacks ordered trials")
    if _sha256_json(trial_ids) != contract["trial_ids_sha256"]:
        raise FrozenBankEvaluationError("Frozen trial ID hash mismatch")
    if _sha256_json(trials) != contract["trial_metadata_sha256"]:
        raise FrozenBankEvaluationError("Frozen trial metadata hash mismatch")

    try:
        with np.load(bank_path, allow_pickle=False) as archive:
            required = {"inputs", "targets", "seq_lengths", "trial_ids"}
            if not required.issubset(archive.files):
                raise FrozenBankEvaluationError(
                    f"Frozen bank lacks arrays: {sorted(required - set(archive.files))}"
                )
            arrays: dict[str, np.ndarray | list[str]] = {
                "inputs": np.array(archive["inputs"], copy=True),
                "targets": np.array(archive["targets"], copy=True),
                "seq_lengths": np.array(archive["seq_lengths"], copy=True),
                "trial_ids": np.array(archive["trial_ids"], copy=True).tolist(),
            }
    except (OSError, ValueError) as exc:
        raise FrozenBankEvaluationError(
            f"Could not load frozen bank: {bank_path}"
        ) from exc
    inputs = arrays["inputs"]
    targets = arrays["targets"]
    lengths = arrays["seq_lengths"]
    if (
        not isinstance(inputs, np.ndarray)
        or inputs.dtype != np.float32
        or inputs.ndim != 3
    ):
        raise FrozenBankEvaluationError(
            "Bank inputs must be float32 [trial,time,feature]"
        )
    n_trials = int(contract["n_trials"])
    if inputs.shape[0] != n_trials:
        raise FrozenBankEvaluationError("Bank input count mismatch")
    if not isinstance(targets, np.ndarray) or targets.dtype != np.int64:
        raise FrozenBankEvaluationError("Bank targets must be int64")
    if not isinstance(lengths, np.ndarray) or lengths.dtype != np.int64:
        raise FrozenBankEvaluationError("Bank seq_lengths must be int64")
    if targets.shape != (n_trials,) or lengths.shape != (n_trials,):
        raise FrozenBankEvaluationError("Bank target/length shape mismatch")
    if arrays["trial_ids"] != trial_ids:
        raise FrozenBankEvaluationError("Bank trial order differs from its index")
    if not np.isfinite(inputs).all():
        raise FrozenBankEvaluationError("Bank inputs contain non-finite values")
    if np.any(lengths <= 0) or np.any(lengths > inputs.shape[1]):
        raise FrozenBankEvaluationError("Bank sequence lengths are invalid")
    return arrays, index, bank_path, index_path


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise FrozenBankEvaluationError(f"Could not load checkpoint: {path}") from exc
    if not isinstance(payload, dict) or not payload:
        raise FrozenBankEvaluationError(
            "Checkpoint must be a direct non-empty state dict"
        )
    if not all(isinstance(key, str) for key in payload):
        raise FrozenBankEvaluationError("Checkpoint keys must be strings")
    if not all(isinstance(value, torch.Tensor) for value in payload.values()):
        raise FrozenBankEvaluationError("Checkpoint values must be tensors")
    return payload


def _semantic_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256(b"dendritic-rnn-evaluation-state-v1\0")
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode() + b"\0")
        digest.update(str(tensor.dtype).encode() + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        raw_bytes = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(b"\0" + raw_bytes + b"\0")
    return digest.hexdigest()


def _load_exact_model(
    *,
    config_path: Path,
    checkpoint_path: Path,
    best_checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, str]:
    config = load_config(str(config_path))
    model_seed = config.experiment.model_seed
    if model_seed is None:
        raise FrozenBankEvaluationError("Resolved config lacks model_seed")
    with torch.random.fork_rng(devices=[], enabled=True):
        torch.manual_seed(int(model_seed))
        model, _ = initialize_model(config.model)
    state = _load_state(checkpoint_path)
    best_state = _load_state(best_checkpoint_path)
    if set(state) != set(best_state) or any(
        not torch.equal(state[key], best_state[key]) for key in state
    ):
        raise FrozenBankEvaluationError("Final and best checkpoints are not identical")
    expected = model.state_dict()
    if set(expected) != set(state):
        raise FrozenBankEvaluationError(
            "Checkpoint keys do not exactly match the reconstructed model"
        )
    for key, tensor in state.items():
        if tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
            raise FrozenBankEvaluationError(f"Checkpoint metadata mismatch for {key}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise FrozenBankEvaluationError(f"Checkpoint contains non-finite {key}")
    model.load_state_dict(state, strict=True)
    state_sha256 = _semantic_state_sha256(model.state_dict())
    model.to(device)
    model.eval()
    return model, config, state_sha256


def _terminal_logits(
    logits: Any, *, seq_lengths: torch.Tensor, batch_size: int, device: torch.device
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise FrozenBankEvaluationError("Model did not return a logits tensor")
    if logits.ndim == 3:
        positions = (seq_lengths - 1).long().clamp(0, logits.shape[1] - 1)
        logits = logits[torch.arange(batch_size, device=device), positions]
    if logits.ndim != 2 or logits.shape[0] != batch_size:
        raise FrozenBankEvaluationError(
            f"Unexpected logits shape: {tuple(logits.shape)}"
        )
    return logits


@contextmanager
def _deterministic_evaluation(seed: int) -> Iterator[None]:
    previous_enabled = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_precision = torch.get_float32_matmul_precision()
    previous_cudnn_deterministic = torch.backends.cudnn.deterministic
    previous_cudnn_benchmark = torch.backends.cudnn.benchmark
    previous_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    previous_cuda_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        with torch.random.fork_rng(devices=[], enabled=True):
            torch.manual_seed(seed)
            torch.use_deterministic_algorithms(True)
            torch.set_float32_matmul_precision("highest")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.allow_tf32 = False
            yield
    finally:
        torch.use_deterministic_algorithms(
            previous_enabled, warn_only=previous_warn_only
        )
        torch.set_float32_matmul_precision(previous_precision)
        torch.backends.cudnn.deterministic = previous_cudnn_deterministic
        torch.backends.cudnn.benchmark = previous_cudnn_benchmark
        torch.backends.cudnn.allow_tf32 = previous_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = previous_cuda_tf32
        if previous_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace


def _evaluate_arrays(
    model: torch.nn.Module,
    bank: Mapping[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> list[dict[str, Any]]:
    inputs = np.asarray(bank["inputs"])
    targets = np.asarray(bank["targets"])
    lengths_array = np.asarray(bank["seq_lengths"])
    trial_ids = list(bank["trial_ids"])
    rows = []
    with torch.inference_mode():
        for start in range(0, len(trial_ids), batch_size):
            stop = min(start + batch_size, len(trial_ids))
            x = torch.from_numpy(inputs[start:stop]).to(device)
            lengths = torch.from_numpy(lengths_array[start:stop]).to(device)
            logits = _terminal_logits(
                model(x, seq_lengths=lengths),
                seq_lengths=lengths,
                batch_size=stop - start,
                device=device,
            )
            if not torch.isfinite(logits).all():
                raise FrozenBankEvaluationError("Evaluation produced non-finite logits")
            predictions = logits.argmax(dim=-1).cpu()
            logits_cpu = logits.detach().to(device="cpu", dtype=torch.float64)
            for offset in range(stop - start):
                target = int(targets[start + offset])
                prediction = int(predictions[offset])
                rows.append(
                    {
                        "trial_id": trial_ids[start + offset],
                        "target": target,
                        "logits": [
                            float(value) for value in logits_cpu[offset].tolist()
                        ],
                        "prediction": prediction,
                        "correct": bool(prediction == target),
                    }
                )
    return rows


def _bitwise_logits(logits: torch.Tensor) -> dict[str, Any]:
    tensor = logits.detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "bytes_hex": tensor.reshape(-1).view(torch.uint8).numpy().tobytes().hex(),
    }


def _population_noop_audit(
    model: torch.nn.Module,
    bank: Mapping[str, Any],
    *,
    device: torch.device,
    contract: Mapping[str, Any],
    provenance: Mapping[str, str],
) -> dict[str, Any]:
    from dendritic_modeling.analysis.utils.causal_interventions import (
        manual_population_classifier_forward,
    )

    subset_size = int(contract["subset_size"])
    trial_ids = list(bank["trial_ids"][:subset_size])
    x = torch.from_numpy(np.asarray(bank["inputs"][:subset_size])).to(device)
    lengths = torch.from_numpy(np.asarray(bank["seq_lengths"][:subset_size])).to(device)
    with torch.inference_mode():
        ordinary = _terminal_logits(
            model(x, seq_lengths=lengths),
            seq_lengths=lengths,
            batch_size=subset_size,
            device=device,
        )
        manual_result = manual_population_classifier_forward(
            model, x, seq_lengths=lengths
        )
        manual = _terminal_logits(
            manual_result.logits,
            seq_lengths=lengths,
            batch_size=subset_size,
            device=device,
        )
    ordinary_record = _bitwise_logits(ordinary)
    manual_record = _bitwise_logits(manual)
    difference = float((ordinary.double() - manual.double()).abs().max().item())
    if ordinary_record != manual_record or difference != 0.0:
        raise FrozenBankEvaluationError(
            f"Population no-op forward mismatch; max_abs_difference={difference}"
        )
    return {
        "schema_version": POPULATION_NOOP_SCHEMA,
        "status": "PASSED",
        "method": POPULATION_NOOP_METHOD,
        "subset_policy": contract["subset_policy"],
        "subset_size": subset_size,
        "trial_ids": trial_ids,
        "trial_ids_sha256": _sha256_json(trial_ids),
        **provenance,
        "ordinary_logits": ordinary_record,
        "manual_logits": manual_record,
        "ordinary_logits_sha256": _sha256_json(ordinary_record),
        "manual_logits_sha256": _sha256_json(manual_record),
        "bitwise_equal": True,
        "max_abs_difference": difference,
    }


def evaluate_run(
    *,
    training_manifest_path: Path,
    reference_manifest_path: Path,
    run_id: str,
    device_name: str = "auto",
    batch_size: int = 128,
    expected_evaluator_sha256: str | None = None,
    expected_runtime_source_sha256: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if (
        expected_evaluator_sha256 is not None
        and _sha256(Path(__file__).resolve()) != expected_evaluator_sha256
    ):
        raise FrozenBankEvaluationError(
            "Evaluator source changed after the dependent job was submitted"
        )
    runtime_source_sha256 = evaluation_runtime_source_sha256()
    if (
        expected_runtime_source_sha256 is not None
        and runtime_source_sha256 != expected_runtime_source_sha256
    ):
        raise FrozenBankEvaluationError(
            "Standard model runtime source changed after the evaluation job was submitted"
        )
    training_manifest_path = _regular_file(training_manifest_path, "training manifest")
    reference_manifest_path = _regular_file(
        reference_manifest_path, "reference manifest"
    )
    training_manifest = _strict_json(training_manifest_path)
    reference_manifest = _strict_json(reference_manifest_path)
    training_row = _find_run(training_manifest, run_id)
    reference_row = _find_run(reference_manifest, run_id)
    _validate_seed_parity(training_row, reference_row)
    if reference_row.get("uses_ood_evaluation") is not True:
        raise FrozenBankEvaluationError("Reference row does not require OOD evaluation")
    contract = reference_row["evaluation_bank"]

    run_dir = Path(training_row["output_dir"]).resolve()
    complete = _strict_json(_regular_file(run_dir / "COMPLETE.json", "training marker"))
    if complete.get("state") != "COMPLETE" or complete.get("run_id") != run_id:
        raise FrozenBankEvaluationError("Training COMPLETE marker mismatch")
    config_path = _regular_file(run_dir / "config.json", "resolved config")
    checkpoint_path = _regular_file(run_dir / "final_model.pt", "final checkpoint")
    best_checkpoint_path = _regular_file(
        run_dir / "main_network/standard_best_model.pt", "best checkpoint"
    )
    evaluation_dir = (
        output_dir.resolve() if output_dir is not None else run_dir / "evaluation"
    )
    prediction_path = evaluation_dir / "predictions.json"
    marker_path = evaluation_dir / "COMPLETE.json"
    if prediction_path.exists() or marker_path.exists():
        raise FrozenBankEvaluationError(
            f"Evaluation output already exists for {run_id}"
        )
    if "/drafts/" in str(config_path) or "/drafts/" in str(prediction_path):
        raise FrozenBankEvaluationError("Draft-owned analysis path is forbidden")

    bank, index, bank_path, _index_path = _load_bank(
        reference_manifest=reference_manifest_path,
        contract=contract,
    )
    if device_name == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise FrozenBankEvaluationError("CUDA was requested but is unavailable")
    checkpoint_sha256 = _sha256(checkpoint_path)
    config_sha256 = _sha256(config_path)
    with _deterministic_evaluation(int(contract["evaluation_seed"])):
        model, config, state_sha256 = _load_exact_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=best_checkpoint_path,
            device=device,
        )
        if (
            config.experiment.sealed_test is not True
            or config.experiment.allow_test_data
        ):
            raise FrozenBankEvaluationError(
                "Resolved config does not preserve the test seal"
            )
        if config.experiment.evaluation_seed != contract["evaluation_seed"]:
            raise FrozenBankEvaluationError("Resolved evaluation seed mismatch")
        if config.data.dataset_name != index["task"]:
            raise FrozenBankEvaluationError("Resolved task differs from frozen bank")
        if int(config.model.encoder.params.input_dim) != int(
            np.asarray(bank["inputs"]).shape[-1]
        ):
            raise FrozenBankEvaluationError(
                "Resolved input width differs from frozen bank"
            )
        initial_keys = set(model.state_dict())
        predictions = _evaluate_arrays(
            model, bank, device=device, batch_size=batch_size
        )
        noop_contract = contract.get("population_noop_audit")
        noop = None
        if noop_contract is not None:
            noop = _population_noop_audit(
                model,
                bank,
                device=device,
                contract=noop_contract,
                provenance={
                    "frozen_bank_trial_ids_sha256": contract["trial_ids_sha256"],
                    "bank_sha256": contract["bank"]["sha256"],
                    "index_sha256": contract["index"]["sha256"],
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_semantic_state_sha256": state_sha256,
                },
            )
        final_state = model.state_dict()
        if (
            not initial_keys.issubset(final_state)
            or _semantic_state_sha256({key: final_state[key] for key in initial_keys})
            != state_sha256
        ):
            raise FrozenBankEvaluationError("Model state changed during evaluation")
        transient = sorted(set(final_state) - initial_keys)
        allowed = ("._last_forward_param_tensor", "._last_forward_weight_mask")
        if any(not key.endswith(allowed) for key in transient):
            raise FrozenBankEvaluationError(
                f"Unexpected transient model state: {transient}"
            )

    if (
        _sha256(checkpoint_path) != checkpoint_sha256
        or _sha256(config_path) != config_sha256
    ):
        raise FrozenBankEvaluationError("Training artifact changed during evaluation")
    if _sha256(bank_path) != contract["bank"]["sha256"]:
        raise FrozenBankEvaluationError("Frozen bank changed during evaluation")

    payload = {
        "schema_version": PREDICTION_SCHEMA,
        "run_id": run_id,
        "evaluation_runtime_source_sha256": runtime_source_sha256,
        "manifest_sha256": _sha256(training_manifest_path),
        "record_sha256": _sha256_json(training_row),
        "manifest_config_sha256": training_row["config_sha256"],
        "resolved_config_artifact": "config.json",
        "resolved_config_sha256": config_sha256,
        "checkpoint_artifact": "final_model.pt",
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_semantic_state_sha256": state_sha256,
        "evaluation_transient_state_keys": transient,
        "bank_id": contract["bank_id"],
        "bank_sha256": contract["bank"]["sha256"],
        "index_sha256": contract["index"]["sha256"],
        "evaluation_seed": contract["evaluation_seed"],
        "trial_ids_sha256": contract["trial_ids_sha256"],
        "trial_metadata_sha256": contract["trial_metadata_sha256"],
        "n_trials": contract["n_trials"],
        "model_core_type": str(config.model.core.type),
        "state_reset_policy": "fresh-zero-hidden-every-batch",
        "batch_size": batch_size,
        "device": {
            "requested": device_name,
            "resolved": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if device.type == "cuda" else None,
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "predictions": predictions,
    }
    if noop is not None:
        payload["population_noop_audit"] = noop
    _atomic_json(prediction_path, payload, readonly=True)
    marker = {
        "schema_version": EVALUATION_MARKER_SCHEMA,
        "state": "COMPLETE",
        "run_id": run_id,
        "training_manifest": str(training_manifest_path),
        "training_manifest_sha256": _sha256(training_manifest_path),
        "reference_manifest": str(reference_manifest_path),
        "reference_manifest_sha256": _sha256(reference_manifest_path),
        "evaluation_runtime_source_sha256": runtime_source_sha256,
        "prediction_artifact": "predictions.json",
        "prediction_sha256": _sha256(prediction_path),
        "n_trials": len(predictions),
        "bank_id": contract["bank_id"],
    }
    _atomic_json(marker_path, marker, readonly=True)
    return marker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--expected-evaluator-sha256")
    parser.add_argument("--expected-runtime-source-sha256")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    marker = evaluate_run(
        training_manifest_path=args.training_manifest,
        reference_manifest_path=args.reference_manifest,
        run_id=args.run_id,
        device_name=args.device,
        batch_size=args.batch_size,
        expected_evaluator_sha256=args.expected_evaluator_sha256,
        expected_runtime_source_sha256=args.expected_runtime_source_sha256,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "state": marker["state"],
                "run_id": marker["run_id"],
                "prediction_sha256": marker["prediction_sha256"],
                "n_trials": marker["n_trials"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FrozenBankEvaluationError",
    "evaluate_run",
    "evaluation_runtime_source_sha256",
]
