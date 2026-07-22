"""Run phase- and depth-specific current interventions on frozen RNN checkpoints.

The evaluator uses the repository's standard population-network causal-current
utility.  It never edits weights or time constants.  For each trained-model
seed, all four primary placement-timescale checkpoints are loaded sequentially
and evaluated on one common frozen counterfactual bank.  Trial-specific
pathway current is replaced by a feature-wise mean calibrated on the bank's
1,024 unique identity receivers, and effects are measured on the 4,096-row
counterfactual expansion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from dendritic_modeling.analysis.utils.causal_interventions import (
    CausalInterventionBatch,
    PathwayInterventionTarget,
    calibrate_fixed_mean_pathways,
    manual_population_classifier_forward,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_counterfactual_bank import (
    N_REPLICATES,
    ROLE_CELLS,
    _atomic_json,
    _atomic_npz,
    _collection_rows,
    _ledger_bank,
    _regular_file,
    _resolve_repo_path,
    _sha256,
    _strict_json,
    load_counterfactual_arrays,
)
from dendritic_modeling.scripts.analysis.evaluate_dendritic_rnn_frozen_bank import (
    _deterministic_evaluation,
    _load_exact_model,
    _semantic_state_sha256,
    evaluation_runtime_source_sha256,
)

STUDY_SCHEMA = "dendritic-rnn-causal-current-pilot-v1"
RESULT_SCHEMA = "dendritic-rnn-causal-current-result-v1"
MARKER_SCHEMA = "dendritic-rnn-causal-current-marker-v1"
REPLICATE_MARKER_SCHEMA = "dendritic-rnn-causal-current-replicate-marker-v1"
EXPECTED_ENDPOINTS = {
    "ff_write": {"pathway": "ff_excitatory", "phase": "first_stimulus"},
    "rec_late_delay": {
        "pathway": "rec_excitatory",
        "phase": "late_delay",
    },
}


class CausalCurrentEvaluationError(RuntimeError):
    """Raised when the intervention contract or an execution audit fails."""


def causal_runtime_source_sha256() -> str:
    """Bind the model runtime and the exact causal-current implementation."""

    causal_path = Path(
        manual_population_classifier_forward.__code__.co_filename
    ).resolve(strict=True)
    payload = {
        "causal_interventions": _sha256(causal_path),
        "evaluation_runtime": evaluation_runtime_source_sha256(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_study_config(path: Path) -> tuple[dict[str, Any], Path]:
    """Load and validate the small, source-owned intervention contract."""

    path = _regular_file(path, "causal-current study config")
    try:
        payload = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise CausalCurrentEvaluationError("Invalid causal-current YAML") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != STUDY_SCHEMA:
        raise CausalCurrentEvaluationError("Unsupported causal-current study schema")
    study = payload.get("study")
    intervention = payload.get("interventions")
    runtime = payload.get("runtime")
    resources = payload.get("resources")
    if not all(
        isinstance(section, Mapping)
        for section in (study, intervention, runtime, resources)
    ):
        raise CausalCurrentEvaluationError("Causal-current config is incomplete")
    configured_roles = {
        role: tuple(cell) for role, cell in study.get("roles", {}).items()
    }
    if configured_roles != ROLE_CELLS or study.get("replicates") != N_REPLICATES:
        raise CausalCurrentEvaluationError("Primary quartet pairing changed")
    endpoints = intervention.get("endpoints")
    if endpoints != EXPECTED_ENDPOINTS:
        raise CausalCurrentEvaluationError("Causal endpoint definitions changed")
    expected = {
        "mode": "fixed_mean",
        "calibration_bank": "identity_receivers",
        "evaluation_bank": "counterfactual_psychometric",
        "layer": "layer0",
        "population": "pyr",
        "soma_relative_depths": [0, 1, 2],
        "stimulus_duration": 20,
        "response_duration": 8,
        "late_delay_steps": 20,
        "no_op_bridge": {
            "direct_manual_same_batch": "bitwise_equal",
            "prior_logits_max_abs_tolerance": 2.0e-5,
            "prior_predictions": "bitwise_equal",
        },
    }
    changed = {
        key: (intervention.get(key), value)
        for key, value in expected.items()
        if intervention.get(key) != value
    }
    if changed:
        raise CausalCurrentEvaluationError(
            f"Causal intervention contract changed: {changed}"
        )
    if resources.get("checkpoints_per_job") != 4:
        raise CausalCurrentEvaluationError("One job must evaluate the four checkpoints")
    return payload, path


def build_phase_masks(
    seq_lengths: np.ndarray,
    *,
    timesteps: int,
    stimulus_duration: int,
    response_duration: int,
    late_delay_steps: int,
) -> dict[str, np.ndarray]:
    """Return exact first-stimulus and last-delay-window masks."""

    lengths = np.asarray(seq_lengths)
    if lengths.ndim != 1 or not np.issubdtype(lengths.dtype, np.integer):
        raise CausalCurrentEvaluationError("Sequence lengths must be integer vector")
    delays = lengths.astype(np.int64) - 2 * stimulus_duration - response_duration
    if np.any(delays < late_delay_steps) or np.any(lengths > timesteps):
        raise CausalCurrentEvaluationError("Sequence geometry cannot support masks")
    time = np.arange(timesteps, dtype=np.int64)[None, :]
    second_start = stimulus_duration + delays[:, None]
    first = time < stimulus_duration
    first = np.broadcast_to(first, (len(lengths), timesteps)).copy()
    late = (time >= second_start - late_delay_steps) & (time < second_start)
    valid = time < lengths[:, None]
    if np.any(first & ~valid) or np.any(late & ~valid):
        raise CausalCurrentEvaluationError("A phase mask selected padded timesteps")
    if not np.all(first.sum(axis=1) == stimulus_duration):
        raise CausalCurrentEvaluationError("First-stimulus mask width changed")
    if not np.all(late.sum(axis=1) == late_delay_steps):
        raise CausalCurrentEvaluationError("Late-delay mask width changed")
    return {"first_stimulus": first, "late_delay": late}


def target_margin(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Binary target logit margin for each trial."""

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.ndim != 2 or logits.shape[1] != 2 or labels.shape != (len(logits),):
        raise CausalCurrentEvaluationError("Invalid logits or labels for margin")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(logits).all():
        raise CausalCurrentEvaluationError("Non-binary labels or non-finite logits")
    signed = 2.0 * labels.astype(np.float64) - 1.0
    return signed * (logits[:, 1] - logits[:, 0])


def _prior_counterfactual_logits(
    root: Path, replicate: int, role: str
) -> tuple[np.ndarray, Path, str]:
    result_path = _regular_file(
        root / f"r{replicate:03d}" / role / "result.json",
        "prior counterfactual result",
    )
    result = _strict_json(result_path)
    logits_record = result.get("logits")
    if not isinstance(logits_record, Mapping):
        raise CausalCurrentEvaluationError("Prior result lacks logits binding")
    logits_path = _regular_file(Path(str(logits_record["path"])), "prior logits")
    logits_hash = _sha256(logits_path)
    if logits_hash != logits_record.get("sha256"):
        raise CausalCurrentEvaluationError("Prior counterfactual logits changed")
    with np.load(logits_path, allow_pickle=False) as archive:
        logits = np.array(archive["counterfactual_logits"], copy=True)
    if logits.shape != (4096, 2) or logits.dtype != np.float32:
        raise CausalCurrentEvaluationError("Prior logits metadata changed")
    return logits, logits_path, logits_hash


def _targets(
    *, layer: str, population: str, pathway: str, depths: list[int]
) -> list[PathwayInterventionTarget]:
    return [
        PathwayInterventionTarget(
            layer=layer,
            target_population=population,
            soma_relative_depth=depth,
            pathway=pathway,
        )
        for depth in depths
    ]


def _audit_intervention(
    *, audit: Any, expected_mask: torch.Tensor, expected_key_fragment: str
) -> dict[str, Any]:
    if len(audit.targets) != 1:
        raise CausalCurrentEvaluationError("Intervention did not resolve one target")
    key = str(audit.targets[0]["key"])
    if expected_key_fragment not in key:
        raise CausalCurrentEvaluationError("Resolved target differs from the contract")
    touched = audit.touched_masks.get(key)
    changed = audit.directly_changed_masks.get(key)
    if (
        touched is None
        or changed is None
        or not torch.equal(touched, expected_mask.cpu())
    ):
        raise CausalCurrentEvaluationError(
            "Direct-touch audit does not equal phase mask"
        )
    if bool((changed & ~touched).any()):
        raise CausalCurrentEvaluationError("Current changed outside the target phase")
    selected = int(touched.sum().item())
    changed_count = int(changed.sum().item())
    if selected <= 0 or changed_count <= 0:
        raise CausalCurrentEvaluationError(
            "Intervention selected or changed no current"
        )
    return {
        "target": dict(audit.targets[0]),
        "selected_trial_timesteps": selected,
        "directly_changed_trial_timesteps": changed_count,
        "direct_change_fraction": changed_count / selected,
        "calls_per_timestep": list(audit.calls_per_timestep[key]),
    }


def _evaluate_role(
    *,
    role: str,
    row: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    intervention: Mapping[str, Any],
    prior_root: Path,
    replicate: int,
    device: torch.device,
    output_dir: Path,
    public_output_dir: Path,
) -> dict[str, Any]:
    run_dir = Path(str(row["training_complete_marker"])).resolve().parent
    config_path = _regular_file(run_dir / "config.json", "resolved config")
    checkpoint_path = _regular_file(run_dir / "final_model.pt", "checkpoint")
    best_path = _regular_file(
        run_dir / "main_network/standard_best_model.pt", "best checkpoint"
    )
    prior_logits, prior_logits_path, prior_logits_hash = _prior_counterfactual_logits(
        prior_root, replicate, role
    )
    checkpoint_hash = _sha256(checkpoint_path)
    evaluation_seed = int(row["seeds"]["evaluation_seed"])
    with _deterministic_evaluation(evaluation_seed):
        model, config, state_hash = _load_exact_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            best_checkpoint_path=best_path,
            device=device,
        )
        if config.experiment.evaluation_seed != evaluation_seed:
            raise CausalCurrentEvaluationError("Evaluation seed binding changed")
        initial_keys = set(model.state_dict())
        x_eval = torch.from_numpy(np.ascontiguousarray(arrays["inputs"])).to(device)
        lengths_eval = torch.from_numpy(np.ascontiguousarray(arrays["seq_lengths"])).to(
            device
        )
        x_cal = torch.from_numpy(np.ascontiguousarray(arrays["identity_inputs"])).to(
            device
        )
        lengths_cal = torch.from_numpy(
            np.ascontiguousarray(arrays["identity_seq_lengths"])
        ).to(device)
        timesteps = int(x_eval.shape[1])
        mask_kwargs = {
            "timesteps": timesteps,
            "stimulus_duration": int(intervention["stimulus_duration"]),
            "response_duration": int(intervention["response_duration"]),
            "late_delay_steps": int(intervention["late_delay_steps"]),
        }
        eval_masks_np = build_phase_masks(arrays["seq_lengths"], **mask_kwargs)
        cal_masks_np = build_phase_masks(arrays["identity_seq_lengths"], **mask_kwargs)
        eval_masks = {
            name: torch.from_numpy(mask).to(device)
            for name, mask in eval_masks_np.items()
        }
        cal_masks = {
            name: torch.from_numpy(mask).to(device)
            for name, mask in cal_masks_np.items()
        }

        with torch.no_grad():
            direct_baseline = model(x_eval, seq_lengths=lengths_eval)
        manual_baseline = manual_population_classifier_forward(
            model, x_eval, seq_lengths=lengths_eval
        ).logits
        direct = direct_baseline.detach().cpu().to(torch.float32).numpy()
        manual = manual_baseline.detach().cpu().to(torch.float32).numpy()
        if not np.array_equal(direct, manual):
            difference = float(np.max(np.abs(direct - manual)))
            raise CausalCurrentEvaluationError(
                "Same-batch direct/manual no-op bridge failed; "
                f"max_abs_difference={difference}"
            )
        prior_difference = float(np.max(np.abs(direct - prior_logits)))
        tolerance = float(
            intervention["no_op_bridge"]["prior_logits_max_abs_tolerance"]
        )
        if prior_difference > tolerance:
            raise CausalCurrentEvaluationError(
                "Prior partition bridge exceeded its frozen bound; "
                f"max_abs_difference={prior_difference}"
            )
        if not np.array_equal(direct.argmax(axis=1), prior_logits.argmax(axis=1)):
            raise CausalCurrentEvaluationError(
                "Prior partition bridge changed at least one prediction"
            )
        baseline = direct

        depths = [int(value) for value in intervention["soma_relative_depths"]]
        output_arrays: dict[str, np.ndarray] = {"baseline_logits": baseline}
        summaries: dict[str, Any] = {}
        baseline_margin = target_margin(baseline, arrays["targets"])
        baseline_accuracy = float((baseline.argmax(axis=1) == arrays["targets"]).mean())
        for endpoint, endpoint_spec in intervention["endpoints"].items():
            pathway = str(endpoint_spec["pathway"])
            phase = str(endpoint_spec["phase"])
            endpoint_targets = _targets(
                layer=str(intervention["layer"]),
                population=str(intervention["population"]),
                pathway=pathway,
                depths=depths,
            )
            calibration = calibrate_fixed_mean_pathways(
                model,
                [
                    CausalInterventionBatch(
                        inputs=x_cal,
                        phase_mask=cal_masks[phase],
                        seq_lengths=lengths_cal,
                    )
                ],
                targets=endpoint_targets,
                source_id=f"r{replicate:03d}:{role}:identity_receivers:{phase}",
            )
            endpoint_rows = {}
            for target in endpoint_targets:
                result = manual_population_classifier_forward(
                    model,
                    x_eval,
                    seq_lengths=lengths_eval,
                    targets=[target],
                    phase_mask=eval_masks[phase],
                    mode="fixed_mean",
                    fixed_mean_calibration=calibration,
                )
                logits = result.logits.detach().cpu().to(torch.float32).numpy()
                if logits.shape != (4096, 2) or not np.isfinite(logits).all():
                    raise CausalCurrentEvaluationError(
                        "Intervention logits are invalid"
                    )
                depth = int(target.soma_relative_depth)
                output_arrays[f"{endpoint}_depth{depth}_logits"] = logits
                intervened_margin = target_margin(logits, arrays["targets"])
                accuracy = float((logits.argmax(axis=1) == arrays["targets"]).mean())
                key_fragment = f"depth={depth}|pathway={pathway}"
                endpoint_rows[str(depth)] = {
                    "baseline_accuracy": baseline_accuracy,
                    "intervened_accuracy": accuracy,
                    "accuracy_drop": baseline_accuracy - accuracy,
                    "mean_target_margin_drop": float(
                        np.mean(baseline_margin - intervened_margin, dtype=np.float64)
                    ),
                    "calibration": calibration.metadata(),
                    "direct_intervention_audit": _audit_intervention(
                        audit=result.audit,
                        expected_mask=eval_masks[phase],
                        expected_key_fragment=key_fragment,
                    ),
                }
            summaries[endpoint] = {
                "pathway": pathway,
                "phase": phase,
                "depths": endpoint_rows,
            }

        final_state = model.state_dict()
        if (
            _semantic_state_sha256({key: final_state[key] for key in initial_keys})
            != state_hash
        ):
            raise CausalCurrentEvaluationError("Model state changed during evaluation")
        transient = sorted(set(final_state) - initial_keys)
        allowed = ("._last_forward_param_tensor", "._last_forward_weight_mask")
        if any(not key.endswith(allowed) for key in transient):
            raise CausalCurrentEvaluationError(
                f"Unexpected transient model state: {transient}"
            )
        del model, x_eval, lengths_eval, x_cal, lengths_cal
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if _sha256(checkpoint_path) != checkpoint_hash:
        raise CausalCurrentEvaluationError("Checkpoint changed during evaluation")
    output_dir.mkdir(parents=True, exist_ok=False)
    logits_path = output_dir / "intervention_logits.npz"
    _atomic_npz(logits_path, output_arrays)
    result_payload = {
        "schema_version": RESULT_SCHEMA,
        "role": role,
        "cell": list(ROLE_CELLS[role]),
        "replicate": f"r{replicate:03d}",
        "run_id": row["run_id"],
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_hash},
        "checkpoint_semantic_state_sha256": state_hash,
        "prior_counterfactual_logits": {
            "path": str(prior_logits_path),
            "sha256": prior_logits_hash,
            "partition_bridge": "PASSED_BOUNDED_WITH_PREDICTION_IDENTITY",
            "max_abs_difference": prior_difference,
            "max_abs_tolerance": tolerance,
            "prediction_identity": "PASSED_BITWISE",
        },
        "intervention": {
            "mode": "fixed_mean",
            "calibration_bank": "identity_receivers",
            "evaluation_bank": "counterfactual_psychometric",
            "summaries": summaries,
        },
        "logits": {
            "path": str(public_output_dir / logits_path.name),
            "sha256": _sha256(logits_path),
            "arrays": {key: list(value.shape) for key, value in output_arrays.items()},
        },
        "audits": {
            "same_batch_direct_manual_noop": "PASSED_BITWISE",
            "prior_partition_bridge": "PASSED_BOUNDED_WITH_PREDICTION_IDENTITY",
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
    _atomic_json(result_path, result_payload)
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
    """Evaluate the four frozen quartet checkpoints for one trained seed."""

    if replicate not in range(N_REPLICATES):
        raise ValueError("replicate must be in 0,...,9")
    evaluator_path = Path(__file__).resolve()
    if (
        expected_evaluator_sha256
        and _sha256(evaluator_path) != expected_evaluator_sha256
    ):
        raise CausalCurrentEvaluationError("Evaluator changed after submission")
    runtime_hash = causal_runtime_source_sha256()
    if (
        expected_runtime_source_sha256
        and runtime_hash != expected_runtime_source_sha256
    ):
        raise CausalCurrentEvaluationError("Causal runtime changed after submission")
    config, config_path = load_study_config(study_config_path)
    if expected_config_sha256 and _sha256(config_path) != expected_config_sha256:
        raise CausalCurrentEvaluationError("Study config changed after submission")
    study = config["study"]
    runtime = config["runtime"]
    if torch.__version__ != runtime["torch"] or torch.version.cuda != runtime["cuda"]:
        raise CausalCurrentEvaluationError(
            "Active Torch/CUDA build is not pinned runtime"
        )
    collection_path = _regular_file(
        _resolve_repo_path(str(study["collection"])), "cube collection"
    )
    collection = _strict_json(collection_path)
    rows = _collection_rows(collection, replicate)
    _, _, bank_path, _, _ = _ledger_bank(
        _resolve_repo_path(str(study["counterfactual_ledger"])), replicate
    )
    arrays = load_counterfactual_arrays(bank_path)
    prior_root = _resolve_repo_path(str(study["counterfactual_results"])).resolve()
    output_root = _resolve_repo_path(str(study["output_root"])).resolve()
    if any(
        "/drafts/" in str(path) for path in (collection_path, prior_root, output_root)
    ):
        raise CausalCurrentEvaluationError(
            "Draft-owned inputs or outputs are forbidden"
        )
    device = torch.device(str(runtime["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise CausalCurrentEvaluationError("CUDA was requested but is unavailable")
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / f"r{replicate:03d}"
    if final_dir.exists():
        raise CausalCurrentEvaluationError(f"Output already exists: {final_dir}")
    staging = output_root / f"._r{replicate:03d}.{os.getpid()}.staging"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        markers = {}
        for role in ROLE_CELLS:
            markers[role] = _evaluate_role(
                role=role,
                row=rows[role],
                arrays=arrays,
                intervention=config["interventions"],
                prior_root=prior_root,
                replicate=replicate,
                device=device,
                output_dir=staging / role,
                public_output_dir=final_dir / role,
            )
        marker = {
            "schema_version": REPLICATE_MARKER_SCHEMA,
            "state": "COMPLETE",
            "replicate": f"r{replicate:03d}",
            "study_config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "evaluator": {
                "path": str(evaluator_path),
                "sha256": _sha256(evaluator_path),
            },
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
    "CausalCurrentEvaluationError",
    "build_phase_masks",
    "causal_runtime_source_sha256",
    "evaluate_replicate",
    "load_study_config",
    "target_margin",
]
