"""Evaluate first- and second-stimulus signal blanks for one four-cell seed block."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from dendritic_modeling.scripts.analysis.analyze_dendritic_rnn_extreme_cube import (
    COLLECTION_SCHEMA,
    N_REPLICATES,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_counterfactual_bank import (
    ROLE_CELLS,
    _collection_rows,
    _deterministic_evaluation,
    _evaluate_logits,
    _ledger_bank,
    _load_exact_model,
    _regular_file,
    _resolve_repo_path,
    _semantic_state_sha256,
    _sha256,
    _strict_json,
    evaluation_runtime_source_sha256,
)

STUDY_SCHEMA = "dendritic-rnn-stimulus-blank-study-v1"
RESULT_SCHEMA = "dendritic-rnn-stimulus-blank-result-v1"
MARKER_SCHEMA = "dendritic-rnn-stimulus-blank-marker-v1"
REPLICATE_MARKER_SCHEMA = "dendritic-rnn-stimulus-blank-replicate-marker-v1"
POLICIES = ("first_stimulus_signal_zero", "second_stimulus_signal_zero")


class StimulusBlankEvaluationError(RuntimeError):
    """Raised when a blank evaluation violates its source or causal contract."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)
    path.chmod(0o444)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    path.chmod(0o444)


def load_study_config(path: Path) -> tuple[dict[str, Any], Path]:
    path = _regular_file(path, "stimulus-blank study config")
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise StimulusBlankEvaluationError("Invalid stimulus-blank YAML") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != STUDY_SCHEMA:
        raise StimulusBlankEvaluationError("Unsupported stimulus-blank study schema")
    study = payload.get("study")
    evaluation = payload.get("evaluation")
    if not isinstance(study, Mapping) or not isinstance(evaluation, Mapping):
        raise StimulusBlankEvaluationError("Stimulus-blank study config is incomplete")
    roles = {role: tuple(cell) for role, cell in study.get("roles", {}).items()}
    if roles != ROLE_CELLS or study.get("replicates") != N_REPLICATES:
        raise StimulusBlankEvaluationError("Stimulus-blank paired design changed")
    if tuple(evaluation.get("policies", [])) != POLICIES:
        raise StimulusBlankEvaluationError("Blank policy order changed")
    if evaluation.get("signal_channels") != [0, 8]:
        raise StimulusBlankEvaluationError("Blank signal channels changed")
    if evaluation.get("stimulus_duration") != 20:
        raise StimulusBlankEvaluationError("Stimulus duration changed")
    return payload, path


def load_ordinary_bank(
    index_source: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], Path, Path]:
    bank_path = _regular_file(Path(str(index_source["bank_path"])), "ordinary bank")
    index_path = _regular_file(Path(str(index_source["index_path"])), "ordinary index")
    if _sha256(bank_path) != index_source.get("bank_sha256"):
        raise StimulusBlankEvaluationError("Ordinary bank hash mismatch")
    if _sha256(index_path) != index_source.get("index_sha256"):
        raise StimulusBlankEvaluationError("Ordinary index hash mismatch")
    index = _strict_json(index_path)
    expected = {
        "bank_id": index_source["bank_id"],
        "bank_sha256": index_source["bank_sha256"],
        "evaluation_seed": index_source["evaluation_seed"],
        "n_trials": 8192,
        "split": "validation_ood",
    }
    if any(index.get(key) != value for key, value in expected.items()):
        raise StimulusBlankEvaluationError("Ordinary index header changed")
    trials = index.get("trials")
    trial_ids = index.get("trial_ids")
    if not isinstance(trials, list) or not isinstance(trial_ids, list):
        raise StimulusBlankEvaluationError("Ordinary index lacks ordered trials")
    try:
        with np.load(bank_path, allow_pickle=False) as archive:
            required = {"inputs", "targets", "seq_lengths", "trial_ids", "delays"}
            if not required.issubset(archive.files):
                raise StimulusBlankEvaluationError(
                    "Ordinary bank arrays are incomplete"
                )
            arrays = {name: np.array(archive[name], copy=True) for name in required}
    except (OSError, ValueError) as exc:
        raise StimulusBlankEvaluationError("Could not load ordinary bank") from exc
    if (
        arrays["inputs"].shape != (8192, 148, 11)
        or arrays["inputs"].dtype != np.float32
    ):
        raise StimulusBlankEvaluationError("Ordinary input metadata changed")
    for name in ("targets", "seq_lengths", "delays"):
        if arrays[name].shape != (8192,) or arrays[name].dtype != np.int64:
            raise StimulusBlankEvaluationError(f"Ordinary {name} metadata changed")
    if arrays["trial_ids"].tolist() != trial_ids:
        raise StimulusBlankEvaluationError("Ordinary trial order changed")
    if not np.array_equal(
        arrays["targets"], np.asarray([row["target"] for row in trials], dtype=np.int64)
    ):
        raise StimulusBlankEvaluationError("Ordinary targets differ from index")
    if not np.array_equal(
        arrays["delays"], np.asarray([row["delay"] for row in trials], dtype=np.int64)
    ):
        raise StimulusBlankEvaluationError("Ordinary delays differ from index")
    return arrays, index, bank_path, index_path


def blank_stimulus_signal(
    inputs: np.ndarray,
    delays: np.ndarray,
    policy: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Copy a bank and zero only frequency-code channels in one stimulus."""

    if policy not in POLICIES:
        raise StimulusBlankEvaluationError(f"Unknown blank policy: {policy}")
    if inputs.shape != (8192, 148, 11) or inputs.dtype != np.float32:
        raise StimulusBlankEvaluationError("Blank input metadata changed")
    if delays.shape != (8192,) or delays.dtype != np.int64:
        raise StimulusBlankEvaluationError("Blank delay metadata changed")
    corrupted = np.array(inputs, copy=True, order="C")
    mask = np.zeros(inputs.shape, dtype=bool)
    for index, delay in enumerate(delays.tolist()):
        start = 0 if policy == POLICIES[0] else 20 + int(delay)
        stop = start + 20
        mask[index, start:stop, :8] = True
        corrupted[index, start:stop, :8] = 0.0
    if not np.array_equal(corrupted[:, :, 8:], inputs[:, :, 8:]):
        raise StimulusBlankEvaluationError("Blank changed phase markers")
    if not np.array_equal(corrupted[~mask], inputs[~mask]):
        raise StimulusBlankEvaluationError("Blank changed undeclared input")
    if np.any(corrupted[mask] != 0.0):
        raise StimulusBlankEvaluationError("Blank did not write exact zero")
    return corrupted, {
        "policy": policy,
        "operation": "set frequency-code channels 0:8 to exact zero",
        "declared_mask_element_count": int(mask.sum()),
        "nonzero_values_replaced": int(np.count_nonzero(inputs[mask])),
        "phase_markers_bitwise_unchanged": True,
        "outside_mask_bitwise_unchanged": True,
        "bank_file_mutated": False,
    }


def _pair_groups(index: Mapping[str, Any]) -> list[tuple[int, int]]:
    positions: dict[str, list[int]] = defaultdict(list)
    trials = index["trials"]
    for position, row in enumerate(trials):
        positions[str(row["pair_id"])].append(position)
    if len(positions) != 4096:
        raise StimulusBlankEvaluationError("Ordinary bank must contain 4,096 pairs")
    pairs = []
    for indices in positions.values():
        if len(indices) != 2 or {trials[index]["target"] for index in indices} != {
            0,
            1,
        }:
            raise StimulusBlankEvaluationError("Malformed ordinary pair")
        pairs.append((indices[0], indices[1]))
    return pairs


def _first_blank_pair_audit(
    inputs: np.ndarray,
    logits: np.ndarray,
    targets: np.ndarray,
    pairs: list[tuple[int, int]],
) -> dict[str, Any]:
    for left, right in pairs:
        if not np.array_equal(inputs[left], inputs[right]):
            raise StimulusBlankEvaluationError(
                "First-blank pair inputs are not identical"
            )
        if not np.array_equal(logits[left], logits[right]):
            raise StimulusBlankEvaluationError(
                "Identical first-blank inputs changed logits"
            )
    predictions = logits.argmax(axis=1)
    accuracy = float(np.mean(predictions == targets))
    if accuracy != 0.5:
        raise StimulusBlankEvaluationError("First-blank accuracy is not exactly 0.5")
    return {
        "n_pairs": len(pairs),
        "pair_inputs_bitwise_equal": True,
        "pair_logits_bitwise_equal": True,
        "accuracy": accuracy,
        "status": "PASSED",
    }


def _ordinary_accuracy(prediction_payload: Mapping[str, Any]) -> float:
    predictions = prediction_payload.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 8192:
        raise StimulusBlankEvaluationError("Ordinary predictions are incomplete")
    return sum(int(row["correct"]) for row in predictions) / len(predictions)


def _evaluate_role(
    *,
    role: str,
    row: Mapping[str, Any],
    bank: Mapping[str, np.ndarray],
    index: Mapping[str, Any],
    bank_provenance: Mapping[str, Any],
    output_dir: Path,
    public_output_dir: Path,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    prediction_path = _regular_file(
        Path(str(row["prediction_artifact"])), "ordinary prediction artifact"
    )
    if _sha256(prediction_path) != row.get("prediction_sha256"):
        raise StimulusBlankEvaluationError("Ordinary prediction hash mismatch")
    prediction_payload = _strict_json(prediction_path)
    if prediction_payload.get("run_id") != row.get("run_id"):
        raise StimulusBlankEvaluationError("Ordinary run binding mismatch")
    ordinary_accuracy = _ordinary_accuracy(prediction_payload)
    if ordinary_accuracy != row.get("accuracy"):
        raise StimulusBlankEvaluationError("Ordinary accuracy binding mismatch")

    run_dir = Path(str(row["training_complete_marker"])).resolve().parent
    config_path = _regular_file(run_dir / "config.json", "resolved config")
    checkpoint_path = _regular_file(run_dir / "final_model.pt", "checkpoint")
    best_path = _regular_file(
        run_dir / "main_network/standard_best_model.pt", "best checkpoint"
    )
    if _sha256(config_path) != prediction_payload.get(
        "resolved_config_sha256"
    ) or _sha256(checkpoint_path) != prediction_payload.get("checkpoint_sha256"):
        raise StimulusBlankEvaluationError(
            "Model artifact changed after ordinary evaluation"
        )

    pairs = _pair_groups(index)
    first_inputs, first_certificate = blank_stimulus_signal(
        bank["inputs"], bank["delays"], POLICIES[0]
    )
    second_inputs, second_certificate = blank_stimulus_signal(
        bank["inputs"], bank["delays"], POLICIES[1]
    )
    with _deterministic_evaluation(int(index["evaluation_seed"])):
        model, config, state_sha256 = _load_exact_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=best_path,
            device=device,
        )
        if config.experiment.evaluation_seed != index["evaluation_seed"]:
            raise StimulusBlankEvaluationError("Evaluation seed mismatch")
        if (
            config.experiment.sealed_test is not True
            or config.experiment.allow_test_data
        ):
            raise StimulusBlankEvaluationError("Resolved config violates the test seal")
        initial_keys = set(model.state_dict())
        first_logits = _evaluate_logits(
            model,
            first_inputs,
            bank["seq_lengths"],
            device=device,
            batch_size=batch_size,
        )
        second_logits = _evaluate_logits(
            model,
            second_inputs,
            bank["seq_lengths"],
            device=device,
            batch_size=batch_size,
        )
        final_state = model.state_dict()
        if (
            _semantic_state_sha256({key: final_state[key] for key in initial_keys})
            != state_sha256
        ):
            raise StimulusBlankEvaluationError(
                "Model state changed during blank evaluation"
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    first_audit = _first_blank_pair_audit(
        first_inputs, first_logits, bank["targets"], pairs
    )
    first_accuracy = float(np.mean(first_logits.argmax(axis=1) == bank["targets"]))
    second_accuracy = float(np.mean(second_logits.argmax(axis=1) == bank["targets"]))

    output_dir.mkdir(parents=True, exist_ok=False)
    logits_path = output_dir / "blank_logits.npz"
    _atomic_npz(
        logits_path,
        {"first_blank_logits": first_logits, "second_blank_logits": second_logits},
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "role": role,
        "cell": list(ROLE_CELLS[role]),
        "replicate": row["replicate"],
        "run_id": row["run_id"],
        "ordinary": {
            "prediction_artifact": str(prediction_path),
            "prediction_sha256": row["prediction_sha256"],
            "accuracy": ordinary_accuracy,
        },
        "bank": dict(bank_provenance),
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        },
        "checkpoint_semantic_state_sha256": state_sha256,
        "policies": {
            POLICIES[0]: {
                "corruption": first_certificate,
                "accuracy": first_accuracy,
                "exact_pair_audit": first_audit,
            },
            POLICIES[1]: {
                "corruption": second_certificate,
                "accuracy": second_accuracy,
                "role": "out_of_distribution_deletion_diagnostic",
            },
        },
        "logits": {
            "path": str(public_output_dir / logits_path.name),
            "sha256": _sha256(logits_path),
            "dtype": "float32",
            "shape_per_policy": [8192, 2],
        },
        "audits": {
            "first_blank_pair_identity": "PASSED_BITWISE",
            "first_blank_exact_half_accuracy": "PASSED",
            "state_immutability": "PASSED",
            "bank_file_mutated": False,
        },
    }
    result_path = output_dir / "result.json"
    _atomic_json(result_path, result)
    marker = {
        "schema_version": MARKER_SCHEMA,
        "state": "COMPLETE",
        "run_id": row["run_id"],
        "result": {
            "path": str(public_output_dir / result_path.name),
            "sha256": _sha256(result_path),
        },
        "logits": {
            "path": str(public_output_dir / logits_path.name),
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
    if replicate not in range(N_REPLICATES):
        raise ValueError("replicate must be in 0,...,9")
    source_path = Path(__file__).resolve()
    if (
        expected_evaluator_sha256 is not None
        and _sha256(source_path) != expected_evaluator_sha256
    ):
        raise StimulusBlankEvaluationError("Evaluator changed after submission")
    runtime_hash = evaluation_runtime_source_sha256()
    if (
        expected_runtime_source_sha256 is not None
        and runtime_hash != expected_runtime_source_sha256
    ):
        raise StimulusBlankEvaluationError("Model runtime changed after submission")
    config, study_config_path = load_study_config(study_config_path)
    if (
        expected_config_sha256 is not None
        and _sha256(study_config_path) != expected_config_sha256
    ):
        raise StimulusBlankEvaluationError("Study config changed after submission")
    runtime = config.get("runtime")
    if (
        not isinstance(runtime, Mapping)
        or torch.__version__ != runtime.get("torch")
        or torch.version.cuda != runtime.get("cuda")
    ):
        raise StimulusBlankEvaluationError("Active Torch/CUDA build differs from study")
    study = config["study"]
    evaluation = config["evaluation"]
    collection_path = _regular_file(
        _resolve_repo_path(str(study["collection"])), "cube collection"
    )
    collection = _strict_json(collection_path)
    if collection.get("schema_version") != COLLECTION_SCHEMA:
        raise StimulusBlankEvaluationError("Unsupported cube collection")
    rows = _collection_rows(collection, replicate)
    ledger_row, counterfactual_index, _, _, ledger_path = _ledger_bank(
        _resolve_repo_path(str(study["counterfactual_ledger"])), replicate
    )
    source = counterfactual_index["source"]
    bank, index, bank_path, index_path = load_ordinary_bank(source)
    if any(row["seeds"]["seed"] != ledger_row["base_seed"] for row in rows.values()):
        raise StimulusBlankEvaluationError("Training/bank base-seed mismatch")
    if any(row["bank_id"] != index["bank_id"] for row in rows.values()):
        raise StimulusBlankEvaluationError("Collection/blank bank join mismatch")
    bank_provenance = {
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "bank_id": index["bank_id"],
        "bank_path": str(bank_path),
        "bank_sha256": _sha256(bank_path),
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
    }
    device = torch.device(str(evaluation["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise StimulusBlankEvaluationError("CUDA was requested but unavailable")
    batch_size = int(evaluation["batch_size"])
    output_root = _resolve_repo_path(str(study["output_root"])).resolve()
    if "/drafts/" in str(output_root):
        raise StimulusBlankEvaluationError("Draft-owned output is forbidden")
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / f"r{replicate:03d}"
    if final_dir.exists():
        raise StimulusBlankEvaluationError("Replicate output already exists")
    staging = output_root / f"._r{replicate:03d}.{os.getpid()}.staging"
    staging.mkdir(exist_ok=False)
    try:
        markers = {}
        for role in ROLE_CELLS:
            markers[role] = _evaluate_role(
                role=role,
                row=rows[role],
                bank=bank,
                index=index,
                bank_provenance=bank_provenance,
                output_dir=staging / role,
                public_output_dir=final_dir / role,
                device=device,
                batch_size=batch_size,
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
            {"state": marker["state"], "replicate": marker["replicate"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "StimulusBlankEvaluationError",
    "blank_stimulus_signal",
    "evaluate_replicate",
    "load_ordinary_bank",
    "load_study_config",
]
