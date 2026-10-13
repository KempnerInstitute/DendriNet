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
from dendritic_modeling.config.conversion import dataset_specific_params
from dendritic_modeling.datasets.parameters import effective_processing
from dendritic_modeling.datasets.poisson import (
    POISSON_GENERATOR_DEFAULTS,
    PoissonGeneratorDataset,
)
from dendritic_modeling.datasets.standard_datasets import (
    load_poisson_generator_datasets,
)
from dendritic_modeling.networks.checkpoints import load_state_dict_with_record
from dendritic_modeling.scripts.analysis.run_checkpoint_analysis import (
    _extract_state_dict,
    _normalize_state_dict_keys,
)
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment
from dendritic_modeling.scripts.sweeps.control_plane import discover_source_identity
from dendritic_modeling.utils.logging_config import setup_logging

SCHEMA_VERSION = "poisson_exposure_transfer_run_v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(2**20):
            digest.update(chunk)
    return digest.hexdigest()


_GAIN_KEYS = (
    "multiplicative_gain",
    "fixed_gain_factor",
    "max_gain_factor",
    "max_gain_tau_ratio",
    "uniform_gain",
    "gain_sampling",
)


def _poisson_parameters(config: Any) -> dict[str, Any]:
    """Return the training run's Poisson options as its loader read them.

    Keys left out of ``data.dataset_params.poisson_generator`` (or a missing
    block) take the loader defaults, ``POISSON_GENERATOR_DEFAULTS``.
    """
    written = dataset_specific_params(config.data)
    return {
        key: written.get(key, default)
        for key, default in POISSON_GENERATOR_DEFAULTS.items()
    }


def _gain_description(dataset: PoissonGeneratorDataset) -> str:
    """Describe the gains a Poisson dataset draws, for error messages."""
    if dataset.fixed_gain_factor is not None:
        return f"a fixed gain of {dataset.fixed_gain_factor:g}"
    return f"gains drawn from [1, {dataset.max_gain_factor:g}]"


def _validate_training_contract(config: Any) -> tuple[dict[str, Any], float, str]:
    """Check that the checkpoint was trained on Poisson data with gain 1.

    Omitted options take the loader defaults, so the check sees the data the
    run trained on: Poisson sampling, a finite positive stimulus duration and
    a multiplicative gain of exactly 1 for every sample.
    """
    if str(config.data.dataset_name).lower() != "poisson_generator":
        raise ValueError("Poisson transfer requires dataset_name=poisson_generator")
    params = _poisson_parameters(config)
    base_dataset = str(params["base_dataset"] or "").lower()
    if not base_dataset:
        raise ValueError("poisson_generator.base_dataset must name a dataset")
    if not bool(params["poisson_sampling"]):
        raise ValueError(
            "Poisson exposure transfer requires poisson_sampling=true (the "
            "default); the checkpoint was trained without Poisson noise"
        )
    try:
        duration = float(params["stimulus_duration"])
    except (TypeError, ValueError):
        duration = math.nan
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("training stimulus_duration must be finite and positive")
    # The training loader's own resolution of the gain options (and its errors).
    training_data = PoissonGeneratorDataset(
        torch.empty(0),
        torch.empty(0),
        poisson_sampling=True,
        stimulus_duration=duration,
        **{key: params[key] for key in _GAIN_KEYS},
    )
    if not training_data.gain_is_unity:
        raise ValueError(
            "Poisson exposure transfer requires training data whose "
            "multiplicative gain is exactly 1, but the training configuration "
            f"gives {_gain_description(training_data)}. Gain 1 means "
            "multiplicative_gain: false, fixed_gain_factor: 1, or no "
            "max_gain_factor and max_gain_tau_ratio (or ones giving a bound of 1)."
        )
    return params, duration, base_dataset


def _test_dataset(config: Any, params: Mapping[str, Any], duration: float):
    _, _, test_dataset = load_poisson_generator_datasets(
        base_dataset=str(params["base_dataset"]),
        multiplicative_gain=False,
        fixed_gain_factor=None,
        max_gain_factor=None,
        uniform_gain=True,
        gain_sampling="log_uniform",
        poisson_sampling=True,
        stimulus_duration=float(duration),
        max_gain_tau_ratio=None,
        train_valid_split=float(config.experiment.train_valid_split),
        flatten=effective_processing(
            "poisson_generator",
            config.data.processing,
            dataset_specific_params(config.data),
        )["flatten"],
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
    config.outputs.results_dir = str(output_dir)
    config.outputs.run_name = "poisson_exposure_transfer"
    config.outputs.exact_run_dir = True
    run_save_path, _train, _valid, _test, model, _ = setup_environment(
        model_config=config.model,
        training_config=config.training,
        data_config=config.data,
        outputs_config=config.outputs,
        experiment_config=config.experiment,
        is_main=True,
    )

    selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = _normalize_state_dict_keys(_extract_state_dict(payload))
    load_record = load_state_dict_with_record(model, state_dict, strict=strict)
    if strict and (load_record["missing_keys"] or load_record["unexpected_keys"]):
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
        **load_record,
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
    parser.add_argument(
        "--config",
        required=True,
        help="Resolved training configuration of the run (its config.json)",
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Model checkpoint to evaluate"
    )
    parser.add_argument(
        "--transfer-config",
        required=True,
        help="Transfer configuration file (durations, draws, seeds)",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory to write the results into"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="For example cuda, cuda:0, cpu (default: cuda when available)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Allow missing or unexpected checkpoint keys and record them",
    )
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


def cli_main(argv: list[str] | None = None) -> int:
    """Console entry point: run :func:`main` and exit 0 on success."""
    setup_logging()
    main(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli_main())
