"""Run config-declared analyses on one frozen model checkpoint.

This entry point reconstructs the model and canonical dataset splits from the
original training config, loads a frozen state dictionary, and executes a
standalone analysis YAML through :class:`~dendritic_modeling.analysis.AnalysisManager`.
It never trains or modifies the checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.analysis.registry import get_registered_analyzers
from dendritic_modeling.config import load_analysis_config, load_config
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity
from dendritic_modeling.utils import resolve_experiment_seeds, save_dict

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "checkpoint_analysis_run_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256_file(path: Path, *, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_state_dict(payload: Any) -> Mapping[str, torch.Tensor]:
    """Extract a model state dictionary from common checkpoint layouts."""

    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                return candidate
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return payload
    raise ValueError(
        "Checkpoint must be a state dictionary or contain model_state_dict, "
        "state_dict, or model"
    )


def _normalize_state_dict_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Remove a uniform DDP ``module.`` prefix without altering other keys."""

    keys = list(state_dict)
    if keys and all(key.startswith("module.") for key in keys):
        return {key[7:]: value for key, value in state_dict.items()}
    return dict(state_dict)


def _enabled_analyzer_names(analysis_config: Any) -> list[str]:
    """Return enabled analyzer registry names in stable order."""

    return sorted(
        name
        for name, spec in get_registered_analyzers().items()
        if bool(
            getattr(
                getattr(analysis_config, spec.config_field, None),
                "enabled",
                False,
            )
        )
    )


def run_checkpoint_analysis(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    analysis_config_path: str | Path,
    output_dir: str | Path,
    filename: str = "final",
    device: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Execute one reproducible, post-training checkpoint analysis."""

    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    analysis_config_path = Path(analysis_config_path).resolve()
    output_dir = Path(output_dir).resolve()
    for path in (config_path, checkpoint_path, analysis_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = load_config(str(config_path))
    config.analysis = load_analysis_config(
        str(analysis_config_path),
        distributed_mode=str(getattr(config.distributed, "mode", "none")),
    )
    config.wandb.use_wandb = False
    config.outputs.results_dir = str(output_dir)
    config.outputs.run_name = "checkpoint_analysis"
    config.outputs.exact_run_dir = True
    # Training provenance may request a complete dataset inventory. Re-emitting
    # that inventory for every frozen-checkpoint analysis is redundant and can
    # be multi-gigabyte for ImageFolder datasets. The analysis provenance below
    # already binds the training config, checkpoint, analysis config, and code.
    config.experiment.record_dataset_fingerprints = False

    run_save_path, train_ds, valid_ds, test_ds, model, _ = setup_environment(
        model_config=config.model,
        training_config=config.training,
        data_config=config.data,
        wandb_config=config.wandb,
        outputs_config=config.outputs,
        experiment_config=config.experiment,
        is_main=True,
    )

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _normalize_state_dict_keys(_extract_state_dict(payload))
    incompatibilities = model.load_state_dict(state_dict, strict=strict)
    model.to(selected_device)
    model.eval()

    seeds = resolve_experiment_seeds(config.experiment)
    manager = AnalysisManager(
        model=model,
        data={"train": train_ds, "valid": valid_ds, "test": test_ds},
        analysis_config=config.analysis,
        save_root=run_save_path,
        device=selected_device,
        evaluation_seed=seeds.evaluation_seed,
        probe_seed=seeds.probe_seed,
    )
    results = manager.run_analysis(filename=filename, training=False)

    provenance = {
        "schema_version": SCHEMA_VERSION,
        "training_performed": False,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "analysis_config_path": str(analysis_config_path),
        "analysis_config_sha256": _sha256_file(analysis_config_path),
        "output_dir": str(output_dir),
        "filename": filename,
        "device": selected_device,
        "strict_checkpoint_loading": bool(strict),
        "state_dict_key_count": len(state_dict),
        "missing_keys": list(incompatibilities.missing_keys),
        "unexpected_keys": list(incompatibilities.unexpected_keys),
        "resolved_seeds": seeds.asdict(),
        "dataset_sizes": {
            "train": len(train_ds),
            "valid": len(valid_ds),
            "test": len(test_ds),
        },
        "enabled_analyzers": _enabled_analyzer_names(config.analysis),
        "code_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
    }
    save_dict(provenance, run_save_path, "checkpoint_analysis_provenance.json")
    logger.info(
        "Checkpoint analysis complete: %s",
        json.dumps(provenance, sort_keys=True),
    )
    return {"results": results, "provenance": provenance}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Resolved training YAML")
    parser.add_argument("--checkpoint", required=True, help="Frozen model checkpoint")
    parser.add_argument(
        "--analysis-config",
        required=True,
        help="Standalone YAML containing the analysis profile",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--filename", default="final")
    parser.add_argument("--device", default=None, help="For example cuda, cuda:0, cpu")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Allow missing or unexpected checkpoint keys and record them",
    )
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    return run_checkpoint_analysis(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        analysis_config_path=args.analysis_config,
        output_dir=args.output_dir,
        filename=args.filename,
        device=args.device,
        strict=not args.no_strict,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
