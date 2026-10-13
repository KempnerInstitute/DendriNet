"""Measure classification updates on a resident synthetic batch from a YAML config.

This measures model/optimizer computation, not dataset loading, validation,
checkpointing, scheduler evolution or scientific accuracy. It preserves the
configured loss, clipping, parameter groups, sparse maintenance and impact
rewiring. Only identity-encoder, standard/presentation-stream classification
without extra regularization is supported. Use the normal trainer to measure
end-to-end throughput; use separate configs for backend comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from dendritic_modeling.config import load_config
from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.training import default_grad_clip_mode
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.impact_rewiring import (
    build_population_impact_rewiring,
)
from dendritic_modeling.scripts.script_utils.config_utils import (
    normalize_param_groups_config,
)
from dendritic_modeling.scripts.script_utils.setup_utils import (
    _apply_initialization_seed_defaults,
    _apply_topology_seed_defaults,
    initialize_model,
    resolve_runtime_performance_settings,
)
from dendritic_modeling.training.clipping import clip_gradients
from dendritic_modeling.training.loss.functions import get_loss_function
from dendritic_modeling.training.optimizers import CustomWeightDecayOptimizer
from dendritic_modeling.training.optimizers.factory import (
    create_optimizer,
    validate_param_group_learning_rates,
)
from dendritic_modeling.utils import resolve_experiment_seeds, set_seed
from dendritic_modeling.utils.logging_config import setup_logging
from dendritic_modeling.utils.runtime_environment import (
    dependency_version,
    source_revision,
)


def build_step(config, *, input_dim, classes, device, steps_per_epoch):
    """Build a fresh training model and one complete update callable."""
    main = config.training.main
    if main.strategy not in {"standard", "presentation_stream"}:
        raise ValueError("benchmark supports standard/presentation_stream only")
    if config.model.encoder.type.lower() != "identity":
        raise ValueError("benchmark requires an identity encoder")
    if config.model.task != "classification":
        raise ValueError("benchmark supports classification only")
    regularization = to_plain_dict(main.regularization)
    if (
        regularization.get("l1_weight", 0)
        or regularization.get("l2_weight", 0)
        or regularization.get("enforce_ei_weight_ratio", False)
        or any(
            v
            for group in regularization.get("param_group_weights", {}).values()
            for v in group.values()
        )
    ):
        raise ValueError("benchmark does not implement additional regularization")
    if (
        main.common.use_amp
        or main.common.enable_nan_checking
        or main.common.reactivation_update_mode != "backprop"
    ):
        raise ValueError("benchmark requires ordinary full-precision backpropagation")
    if config.model.initial_checkpoint:
        raise ValueError(
            "benchmark starts from initialization, not an initial_checkpoint"
        )
    config.model.encoder.params.input_dim = input_dim
    config.model.encoder.params.input_shape = [input_dim]
    seeds = resolve_experiment_seeds(config.experiment, write_back=True)
    flags = resolve_runtime_performance_settings(config.experiment, write_back=True)
    set_seed(seeds.model_seed, strict_deterministic=seeds.strict_deterministic, **flags)
    _apply_topology_seed_defaults(config.model, seeds.topology_seed)
    _apply_initialization_seed_defaults(config.model, seeds.model_seed)
    model, _ = initialize_model(config.model)
    model = model.to(device).train()
    common = main.common
    groups = normalize_param_groups_config(common.param_groups)
    validate_param_group_learning_rates(main.optimizer, groups)
    base_optimizer = create_optimizer(model.get_param_groups(groups), main.optimizer)
    controllers = []
    if common.impact_rewiring:
        controllers.append(
            build_population_impact_rewiring(
                model.core_network,
                common.impact_rewiring,
                steps_per_epoch=steps_per_epoch,
                epochs=common.epochs,
                seed=seeds.topology_seed,
            )
        )
    optimizer = CustomWeightDecayOptimizer(
        model,
        base_optimizer,
        weight_decay=common.synapse_decay_rate,
        weight_boosting=common.weight_boosting,
        topology_controllers=controllers,
    )
    # Keep input generation off the model/release random stream.
    generator = torch.Generator(device=device).manual_seed(seeds.probe_seed)
    x = torch.rand(common.batch_size, input_dim, device=device, generator=generator)
    y = torch.randint(classes, (common.batch_size,), device=device, generator=generator)
    criterion = get_loss_function(common.loss_function)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.profiler.record_function("configured_step.forward_loss"):
            loss = criterion(model, x, y)
        with torch.profiler.record_function("configured_step.backward"):
            loss.backward()
        with torch.profiler.record_function(
            "configured_step.clip_optimizer_maintenance"
        ):
            clip_gradients(
                model.parameters(),
                common.grad_clip_value,
                common.grad_clip_mode
                or default_grad_clip_mode(main.strategy, config.distributed.mode),
            )
            optimizer.step()
        # Match the trainer's loss reporting synchronization.
        return loss.detach().item()

    return model, optimizer, step, flags


def measure_step(step, *, device, warmup, steps):
    """Return synchronized wall times; event intervals are not GPU utilization."""
    if steps < 1 or warmup < 0:
        raise ValueError("steps must be positive and warmup nonnegative")
    sync = (
        (lambda: torch.cuda.synchronize(device))
        if device.type == "cuda"
        else lambda: None
    )
    for _ in range(warmup):
        step()
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    durations, losses = [], []
    for _ in range(steps):
        start = time.perf_counter()
        loss = step()
        sync()
        durations.append(1000 * (time.perf_counter() - start))
        losses.append(loss)
    if not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError("non-finite benchmark loss")
    return {
        "step_ms_median": statistics.median(durations),
        "step_ms_mean": statistics.mean(durations),
        "step_ms_max": max(durations),
        "step_ms": durations,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "peak_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / 1e6
            if device.type == "cuda"
            else None
        ),
    }


def profile_step(step, *, device, steps, trace=None):
    """Profile separately so profiler overhead never enters benchmark timing."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profiler:
        for _ in range(steps):
            step()
            profiler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    if trace:
        profiler.export_chrome_trace(str(trace))
    events = sorted(
        profiler.key_averages(), key=lambda e: e.self_device_time_total, reverse=True
    )
    return [
        {
            "name": e.key,
            "count": e.count,
            "self_device_ms": e.self_device_time_total / 1000,
            "self_cpu_ms": e.self_cpu_time_total / 1000,
        }
        for e in events[:30]
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        required=True,
        help="Real training epoch length for rewiring freeze scheduling",
    )
    parser.add_argument("--profile-steps", type=int, default=0)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.input_dim, args.classes, args.steps_per_epoch) < 1:
        parser.error("input-dim, classes and steps-per-epoch must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.trace and args.profile_steps < 1:
        parser.error("--trace requires positive --profile-steps")
    config = load_config(str(args.config))
    if args.batch_size:
        config.training.main.common.batch_size = args.batch_size
    device = torch.device(args.device)
    model, optimizer, step, flags = build_step(
        config,
        input_dim=args.input_dim,
        classes=args.classes,
        device=device,
        steps_per_epoch=args.steps_per_epoch,
    )
    result = measure_step(step, device=device, warmup=args.warmup, steps=args.steps)
    result["topology_replacements_including_warmup"] = optimizer.topology_replacements
    module_counts = Counter(type(m).__name__ for m in model.modules())
    result.update(
        scope="resident synthetic batch; fixed learning rate; no dataset/validation/I/O",
        config_path=str(args.config.resolve()),
        config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        resolved_config=to_plain_dict(config),
        runtime_flags=flags,
        parameters=sum(p.numel() for p in model.parameters()),
        module_counts=dict(module_counts),
        resolved_projection_backends=dict(
            Counter(
                str(m._last_resolved_projection_backend)
                for m in model.modules()
                if getattr(m, "_last_resolved_projection_backend", None) is not None
            )
        ),
        projection_backends=dict(
            Counter(
                str(m.projection_backend)
                for m in model.modules()
                if hasattr(m, "projection_backend")
            )
        ),
        batch_size=config.training.main.common.batch_size,
        steps=args.steps,
        warmup=args.warmup,
        steps_per_epoch=args.steps_per_epoch,
        device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        source_file=__file__,
    )
    result.update(
        library_commit=source_revision(),
        numpy=dependency_version("numpy"),
        triton=dependency_version("triton"),
        cpu_threads=torch.get_num_threads(),
        parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}),
        input_dtype=str(torch.get_default_dtype()),
    )
    if args.profile_steps:
        result["profile_steps"] = args.profile_steps
        result["profile"] = profile_step(
            step, device=device, steps=args.profile_steps, trace=args.trace
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "step_ms_median",
                    "step_ms_mean",
                    "peak_allocated_mb",
                    "parameters",
                ]
            }
        )
    )


if __name__ == "__main__":
    setup_logging()
    main()
