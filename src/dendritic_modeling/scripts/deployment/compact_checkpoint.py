"""Export a trained model with losslessly frozen sparse topology."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from dendritic_modeling.analysis.utils.model_resources import model_resource_summary
from dendritic_modeling.config import ModelCompressionConfig, load_config
from dendritic_modeling.config.keys import field_default
from dendritic_modeling.deployment import (
    compact_model_checkpoint,
    load_compact_model_checkpoint,
)
from dendritic_modeling.networks.checkpoints import (
    commit_temporary_file,
    create_temporary_file,
    load_model_state_checkpoint,
    sha256_file,
)
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.utils.logging_config import setup_logging


def _module_prefix(model: torch.nn.Module, target: torch.nn.Module) -> str:
    for name, module in model.named_modules():
        if module is target:
            return f"{name}." if name else ""
    raise ValueError("target module is not registered in the model")


def _runtime_storage_ledger(model: torch.nn.Module) -> dict[str, int]:
    """Count exact in-memory parameter and persistent-buffer tensor bytes."""

    core = getattr(model, "core_network", model)
    core_parameter_ids = {id(value) for value in core.parameters()}
    core_buffer_ids = {id(value) for value in core.buffers()}
    parameters = list(model.parameters())
    buffers = list(model.buffers())

    def tensor_bytes(values: list[torch.Tensor]) -> int:
        return int(sum(value.numel() * value.element_size() for value in values))

    return {
        "total_parameter_bytes": tensor_bytes(parameters),
        "total_persistent_buffer_bytes": tensor_bytes(buffers),
        "total_tensor_bytes": tensor_bytes(parameters) + tensor_bytes(buffers),
        "core_parameter_bytes": tensor_bytes(
            [value for value in parameters if id(value) in core_parameter_ids]
        ),
        "core_persistent_buffer_bytes": tensor_bytes(
            [value for value in buffers if id(value) in core_buffer_ids]
        ),
    }


def _serialized_storage_ledger(
    model: torch.nn.Module,
    checkpoint: Path,
) -> dict[str, int]:
    """Count tensor bytes after checkpoint-specific unsigned-index packing."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = payload.get("state_dict")
    if not isinstance(state, dict):
        raise TypeError("compact checkpoint is missing its tensor state dictionary")
    core = getattr(model, "core_network", model)
    core_prefix = _module_prefix(model, core)
    parameter_names = set(dict(model.named_parameters()))

    def nbytes(value: torch.Tensor) -> int:
        return int(value.numel() * value.element_size())

    tensor_items = [
        (str(name), value)
        for name, value in state.items()
        if isinstance(value, torch.Tensor)
    ]
    core_items = [
        (name, value) for name, value in tensor_items if name.startswith(core_prefix)
    ]
    core_parameter_bytes = sum(
        nbytes(value) for name, value in core_items if name in parameter_names
    )
    core_topology_bytes = sum(
        nbytes(value) for name, value in core_items if "connection_indices" in name
    )
    return {
        "total_tensor_bytes": sum(nbytes(value) for _, value in tensor_items),
        "core_tensor_bytes": sum(nbytes(value) for _, value in core_items),
        "core_parameter_bytes": int(core_parameter_bytes),
        "core_topology_bytes": int(core_topology_bytes),
        "core_other_buffer_bytes": int(
            sum(nbytes(value) for _, value in core_items)
            - core_parameter_bytes
            - core_topology_bytes
        ),
    }


def _verification_input_shape(config) -> tuple[int, ...]:
    for name in ("vision_replacement", "pretrained_replacement"):
        replacement = getattr(config.model, name, None)
        if replacement is not None and getattr(replacement, "enabled", False):
            shape = tuple(
                int(value)
                for value in (getattr(replacement, "input_shape", None) or ())
            )
            if shape:
                return shape
    params = config.model.encoder.params
    shape = tuple(int(value) for value in (getattr(params, "input_shape", None) or ()))
    default_shape = tuple(field_default(type(params), "input_shape") or ())
    input_dim = getattr(params, "input_dim", None)
    # A run's recorded config.json holds the input shape training saw. A
    # hand-written config may leave input_shape at its (image) default.
    if shape and shape != default_shape:
        return shape
    if input_dim is not None:
        flatten = getattr(getattr(config.data, "processing", None), "flatten", None)
        if shape and math.prod(shape) == int(input_dim) and flatten is False:
            return shape
        return (int(input_dim),)
    raise ValueError(
        "cannot infer the full-model verification input shape: pass the run's "
        "recorded config.json, which stores model.encoder.params.input_dim and "
        "input_shape"
    )


def _random_verification_input(config, *, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(20260809)
    return torch.randn(
        (1, *_verification_input_shape(config)),
        generator=generator,
        device=device,
    )


def _check_report_destination(report: Path, *checkpoints: Path) -> None:
    """Refuse an unusable ``--report`` path before the export writes anything."""
    resolved = report.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(resolved)
    if any(resolved == path.expanduser().resolve() for path in checkpoints):
        raise ValueError("--report must differ from the source and output checkpoints")


def _write_report_atomically(report: dict[str, object], destination: Path) -> None:
    resolved = destination.expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(resolved)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = create_temporary_file(resolved.parent, resolved.name)
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        commit_temporary_file(temporary, resolved)
    finally:
        temporary.unlink(missing_ok=True)


def export_compact_checkpoint(
    experiment_config: Path,
    source_checkpoint: Path,
    output_checkpoint: Path,
    compression_config: Path | None = None,
    *,
    verify_full_model: bool = False,
    device: str = "cpu",
) -> dict[str, object]:
    """Build, strictly load, freeze, and export one standard trained model."""

    config = load_config(str(experiment_config.resolve()))
    policy = (
        ModelCompressionConfig.load(compression_config)
        if compression_config is not None
        else ModelCompressionConfig()
    )
    model, _ = initialize_model(config.model)
    resolved_device = torch.device(device)
    model.to(resolved_device)
    state = load_model_state_checkpoint(source_checkpoint)
    model.load_state_dict(state, strict=True)
    del state
    gc.collect()
    model.eval()
    training_resources = model_resource_summary(model)
    training_storage = _runtime_storage_ledger(model)
    sample = None
    expected = None
    if verify_full_model:
        sample = _random_verification_input(config, device=resolved_device)
        with torch.no_grad():
            expected = model(sample).detach().cpu()
        # Dense-candidate Top-K forwards can leave large allocator caches.  Use
        # a fresh reconstruction for conversion so full-model verification does
        # not increase the peak memory of topology freezing.
        del model
        gc.collect()
        model, _ = initialize_model(config.model)
        model.to(resolved_device)
        state = load_model_state_checkpoint(source_checkpoint)
        model.load_state_dict(state, strict=True)
        del state
        gc.collect()
        model.eval()
    report = compact_model_checkpoint(
        model,
        source_checkpoint,
        output_checkpoint,
        policy,
        provenance={
            "experiment_config": {
                "path": str(experiment_config.resolve()),
                "sha256": sha256_file(experiment_config),
            },
            "compression_config": (
                None
                if compression_config is None
                else {
                    "path": str(compression_config.resolve()),
                    "sha256": sha256_file(compression_config),
                }
            ),
        },
    )
    deployment_resources = model_resource_summary(model)
    deployment_runtime_storage = _runtime_storage_ledger(model)
    serialized_storage = _serialized_storage_ledger(model, output_checkpoint)
    report["schema_version"] = "compact_checkpoint_export_v2"
    report["training_resources"] = training_resources
    report["training_runtime_storage"] = training_storage
    report["deployment_resources"] = deployment_resources
    report["deployment_runtime_storage"] = deployment_runtime_storage
    report["deployment_serialized_storage"] = serialized_storage
    report["core_parameter_reduction_factor"] = float(
        training_resources["core_total_parameters"]
        / max(deployment_resources["core_total_parameters"], 1)
    )
    if verify_full_model:
        del model
        gc.collect()
        restored, _ = initialize_model(config.model)
        restored.to(resolved_device)
        load_compact_model_checkpoint(restored, output_checkpoint, strict=True)
        restored.eval()
        assert sample is not None and expected is not None
        with torch.no_grad():
            observed = restored(sample).detach().cpu()
        difference = (expected - observed).abs()
        close = torch.allclose(
            expected,
            observed,
            atol=policy.verification.atol,
            rtol=policy.verification.rtol,
        )
        if not close:
            raise RuntimeError(
                "Compacted full-model output failed numerical verification; "
                f"maximum absolute error={float(difference.max().item()):.6g}"
            )
        report["full_model_verification"] = {
            "enabled": True,
            "input_shape": list(sample.shape),
            "maximum_absolute_error": float(difference.max().item()),
            "mean_absolute_error": float(difference.mean().item()),
            "prediction_equal": bool(
                expected.argmax(dim=-1).eq(observed.argmax(dim=-1)).all()
            ),
            "atol": policy.verification.atol,
            "rtol": policy.verification.rtol,
        }
    else:
        report["full_model_verification"] = {"enabled": False}
    report["experiment_config"] = str(experiment_config.resolve())
    report["compression_config"] = (
        None if compression_config is None else str(compression_config.resolve())
    )
    report["dense_runtime_backend"] = policy.dense_runtime.backend
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment_config",
        type=Path,
        help="Training YAML used to construct the original model.",
    )
    parser.add_argument(
        "source_checkpoint",
        type=Path,
        help="File holding the model state dictionary, such as final_model.pt.",
    )
    parser.add_argument(
        "output_checkpoint",
        type=Path,
        help="Compact checkpoint to write; it must not exist.",
    )
    parser.add_argument(
        "--compression-config",
        type=Path,
        default=None,
        help="Standalone deployment-compression YAML.",
    )
    parser.add_argument(
        "--verify-full-model",
        action="store_true",
        help=(
            "Compare deterministic full-model outputs before export and after a "
            "fresh strict compact-checkpoint load."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used for conversion and optional full-model verification.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Atomically write the complete JSON resource and verification "
            "ledger to this new file; an existing file is refused before "
            "anything is written."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    setup_logging()
    args = build_parser().parse_args(argv)
    if args.report is not None:
        _check_report_destination(
            args.report, args.source_checkpoint, args.output_checkpoint
        )
    report = export_compact_checkpoint(
        args.experiment_config,
        args.source_checkpoint,
        args.output_checkpoint,
        args.compression_config,
        verify_full_model=args.verify_full_model,
        device=args.device,
    )
    if args.report is not None:
        _write_report_atomically(report, args.report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
