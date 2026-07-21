"""Evaluate one frozen Poisson-trained checkpoint across exposure durations.

The command reconstructs the model from its resolved training YAML, loads the
checkpoint strictly, rebuilds the canonical test split at each declared
exposure, and delegates all scientific evaluation to the package
``PoissonExposureTransferAnalyzer``. It never retrains the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from dendritic_modeling.analysis.tools.poisson_exposure_transfer import (
    PoissonExposureTransferAnalyzer,
)
from dendritic_modeling.config import PoissonExposureTransferParams, load_config
from dendritic_modeling.datasets.standard_datasets import (
    load_poisson_generator_datasets,
)
from dendritic_modeling.scripts.analysis.run_checkpoint_analysis import (
    _extract_state_dict,
    _normalize_state_dict_keys,
)
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity

SCHEMA_VERSION = "poisson_exposure_transfer_run_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


def _parameter(container: Any, name: str, default: Any = None) -> Any:
    if isinstance(container, Mapping):
        return container.get(name, default)
    return getattr(container, name, default)


def _poisson_parameters(config: Any) -> Any:
    params = config.data.dataset_params
    poisson = (
        params.get("poisson_generator")
        if isinstance(params, Mapping)
        else getattr(params, "poisson_generator", None)
    )
    if poisson is None:
        raise ValueError("data.dataset_params.poisson_generator is required")
    return poisson


def _validate_training_contract(config: Any) -> tuple[Any, float, str]:
    if str(config.data.dataset_name).lower() != "poisson_generator":
        raise ValueError("Poisson transfer requires dataset_name=poisson_generator")
    params = _poisson_parameters(config)
    base_dataset = str(_parameter(params, "base_dataset", "")).lower()
    if not base_dataset:
        raise ValueError("poisson_generator.base_dataset is required")
    if bool(_parameter(params, "multiplicative_gain", False)):
        raise ValueError("Poisson exposure transfer requires multiplicative_gain=false")
    if not bool(_parameter(params, "poisson_sampling", False)):
        raise ValueError("Poisson exposure transfer requires poisson_sampling=true")
    duration = float(_parameter(params, "stimulus_duration", math.nan))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("training stimulus_duration must be finite and positive")
    return params, duration, base_dataset


def _test_dataset(config: Any, params: Any, duration: float):
    _, _, test_dataset = load_poisson_generator_datasets(
        base_dataset=str(_parameter(params, "base_dataset")),
        multiplicative_gain=False,
        fixed_gain_factor=None,
        max_gain_factor=None,
        uniform_gain=True,
        gain_sampling="log_uniform",
        poisson_sampling=True,
        stimulus_duration=float(duration),
        max_gain_tau_ratio=None,
        train_valid_split=float(config.experiment.train_valid_split),
        flatten=bool(config.data.processing.flatten),
        task_data_path=(str(config.data.base_dir) if config.data.base_dir else None),
        split_seed=int(config.experiment.seed),
    )
    return test_dataset


def run_poisson_exposure_transfer(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    transfer_config_path: str | Path,
    output_dir: str | Path,
    device: str | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Run one hash-bound frozen-checkpoint exposure-transfer analysis."""

    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    transfer_config_path = Path(transfer_config_path).resolve()
    output_dir = Path(output_dir).resolve()
    for path in (config_path, checkpoint_path, transfer_config_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = load_config(str(config_path))
    params, train_duration, base_dataset = _validate_training_contract(config)
    transfer_params = PoissonExposureTransferParams.load(str(transfer_config_path))
    config.wandb.use_wandb = False
    config.outputs.results_dir = str(output_dir)
    config.outputs.run_name = "poisson_exposure_transfer"
    config.outputs.exact_run_dir = True
    run_save_path, _train, _valid, _test, model, _ = setup_environment(
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
    if strict and (incompatibilities.missing_keys or incompatibilities.unexpected_keys):
        raise RuntimeError("strict checkpoint loading recorded incompatibilities")
    model.to(selected_device)
    model.eval()

    analyzer = PoissonExposureTransferAnalyzer(transfer_params)
    result = analyzer.analyze(
        model,
        dataset_factory=lambda duration: _test_dataset(config, params, duration),
        train_duration=train_duration,
        device=selected_device,
    )
    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "base_dataset": base_dataset,
            "experiment_seed": int(config.experiment.seed),
            "training_performed": False,
        }
    )

    run_path = Path(run_save_path)
    run_path.mkdir(parents=True, exist_ok=True)
    result_path = run_path / "poisson_exposure_transfer.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "training_performed": False,
        "strict_checkpoint_loading": bool(strict),
        "missing_keys": list(incompatibilities.missing_keys),
        "unexpected_keys": list(incompatibilities.unexpected_keys),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "transfer_config_path": str(transfer_config_path),
        "transfer_config_sha256": _sha256(transfer_config_path),
        "output_dir": str(run_path),
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "device": selected_device,
        "code_identity": discover_source_identity(
            _repo_root(), source_files=(Path(__file__),)
        ),
    }
    (run_path / "poisson_exposure_transfer_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"result": result, "provenance": provenance}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--transfer-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-strict", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = _build_parser().parse_args(argv)
    return run_poisson_exposure_transfer(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        transfer_config_path=args.transfer_config,
        output_dir=args.output_dir,
        device=args.device,
        strict=not args.no_strict,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
