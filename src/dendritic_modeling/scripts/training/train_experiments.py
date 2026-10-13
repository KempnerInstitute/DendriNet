#!/usr/bin/env python
"""
train_experiments.py
====================
Main training script for dendritic modeling experiments.

Supports both single-GPU and multi-GPU (DDP) training.  For DDP, launch with
``torchrun``::

    torchrun --nproc_per_node=4 train_experiments.py <config_path>

Single-GPU runs work exactly as before with plain ``python``.

For FSDP (very large models that don't fit on one GPU), use
``train_experiments_fsdp.py`` instead.
"""

from __future__ import annotations

import argparse
import copy
import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

# External imports
import torchinfo
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.analysis.utils.model_resources import model_resource_summary
from dendritic_modeling.config import load_config
from dendritic_modeling.config.analysis import fill_analysis_seeds
from dendritic_modeling.config.keys import apply_renamed_fields, field_default
from dendritic_modeling.config.training import (
    LOCAL_ERROR_BROADCAST_MODES,
    CommonTrainerConfig,
    OptimizerConfig,
    ParamGroupsConfig,
    default_grad_clip_mode,
)
from dendritic_modeling.models.builders import resolve_model_builder
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.classical.autoencoder import (
    BaseAutoencoder,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.impact_rewiring import (
    build_population_impact_rewiring,
)
from dendritic_modeling.networks.architectures.factory import (
    builds_autoencoder,
    builtin_architecture_class,
)
from dendritic_modeling.networks.checkpoints import save_state_dict
from dendritic_modeling.networks.utils.sparse_maintenance import (
    mark_rank_reduced_statistics,
)
from dendritic_modeling.plotting.visualizations.performance_plots import (
    PerformancePlotter,
)
from dendritic_modeling.scripts.script_utils import config_utils as _config_utils
from dendritic_modeling.scripts.script_utils.init_diagnostics import (
    dump_init_gate_stats,
)
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment
from dendritic_modeling.training import CustomWeightDecayOptimizer, get_trainer
from dendritic_modeling.training.dataloader_utils import (
    dataloader_kwargs_from_config,
    seeded_dataloader_kwargs,
)
from dendritic_modeling.training.encoder_pretraining import pretrain_autoencoder
from dendritic_modeling.training.optimizers import (
    create_optimizer,
    validate_param_group_learning_rates,
)
from dendritic_modeling.training.reactivation_sync import broadcast_reactivation_state
from dendritic_modeling.utils import resolve_experiment_seeds, save_dict
from dendritic_modeling.utils.logging_config import setup_logging
from dendritic_modeling.utils.training_stability_validator import (
    validate_training_stability,
)

logger = logging.getLogger(__name__)

prepare_trainer_config = _config_utils.prepare_trainer_config
normalize_param_groups_config = getattr(
    _config_utils, "normalize_param_groups_config", None
)
canonicalize_model_core_flags = getattr(
    _config_utils, "canonicalize_model_core_flags", None
)
canonicalize_rnn_core_flags = getattr(
    _config_utils, "canonicalize_rnn_core_flags", None
)


@dataclass(frozen=True)
class _MainTrainingSettings:
    common_config: Any
    strategy: str
    optimizer_config: OptimizerConfig
    param_groups: Any
    synapse_decay_rate: float
    weight_boosting: bool
    common_epochs: int


@dataclass(frozen=True)
class _MainTrainerSetup:
    trainer: Any
    main_train_config: Any
    trainer_config_dict: dict[str, Any]
    optimizer: Any
    strategy: str
    common_epochs: int
    original_pruning_config: Any
    # The trainer config before get_trainer consumed its keys, for the
    # trainer that applies post-training pruning.
    pruning_trainer_config: dict[str, Any]


@dataclass(frozen=True)
class _MainTrainingRun:
    main_train_config: Any
    trainer_config_dict: dict[str, Any]
    optimizer: Any
    strategy: str
    common_epochs: int
    original_pruning_config: Any
    training_results: dict[str, Any]
    pruning_trainer_config: dict[str, Any]


@dataclass(frozen=True)
class _TrainingEnvironment:
    run_save_path: str
    train_ds: Any
    valid_ds: Any
    test_ds: Any
    model: Any
    unwrapped_model: Any
    encoder_network: Any


@dataclass(frozen=True)
class _EncodedTrainingState:
    encoder_network: Any
    encoded_train_ds: Any
    encoded_valid_ds: Any
    encoded_test_ds: Any
    analysis_manager: Any


@dataclass(frozen=True)
class _PreparedTrainingConfig:
    config: Any
    experiment_config: Any
    data_config: Any
    model_config: Any
    training_config: Any
    analysis_config: Any
    outputs_config: Any


if normalize_param_groups_config is None:
    # Backward-compatible fallback for environments where config_utils lacks this helper.
    def normalize_param_groups_config(param_groups):
        if param_groups is None:
            return ParamGroupsConfig()

        if isinstance(param_groups, ParamGroupsConfig):
            return param_groups

        if isinstance(param_groups, DictConfig):
            param_groups = OmegaConf.to_container(param_groups, resolve=True)

        if hasattr(param_groups, "asdict") and callable(param_groups.asdict):
            param_groups = param_groups.asdict()
        elif not isinstance(param_groups, dict) and hasattr(param_groups, "__dict__"):
            param_groups = {
                key: value
                for key, value in vars(param_groups).items()
                if not key.startswith("_")
            }

        if not isinstance(param_groups, dict):
            return ParamGroupsConfig()

        allowed_fields = set(ParamGroupsConfig.__dataclass_fields__.keys())
        sanitized = {k: v for k, v in param_groups.items() if k in allowed_fields}
        return ParamGroupsConfig(**sanitized)


def _configured_main_strategy(config) -> str:
    """Return ``training.main.strategy`` from dict or object configs (default standard)."""
    main = getattr(getattr(config, "training", None), "main", None)
    if isinstance(main, dict):
        return str(main.get("strategy") or "standard")
    return str(getattr(main, "strategy", None) or "standard")


def _no_calibration_batches_message(train_dataset, batch_size: int) -> str:
    """Explain why data-driven gate calibration found no training batch."""
    message = (
        "Data-driven reactivation initialization was requested, but no "
        "training batches were available for calibration"
    )
    try:
        split_size = len(train_dataset)
    except TypeError:
        return message + "."
    if split_size < batch_size:
        return (
            f"{message}: the training split has {split_size} samples, fewer than "
            f"one calibration batch of {batch_size} "
            "(training.main.common.reactivation_initialization_batch_size), and "
            "calibration drops incomplete batches. Set "
            "reactivation_initialization_batch_size to at most "
            f"{split_size}, or use a larger training split."
        )
    return (
        f"{message} (training split of {split_size} samples, calibration batch "
        f"size {batch_size})."
    )


def _maybe_calibrate_data_driven_reactivation(
    config,
    model,
    train_dataset,
    is_main: bool = True,
    n_batches: int | None = None,
    batch_size: int | None = None,
    n_iterations: int | None = None,
):
    """Run and report data-driven reactivation calibration when requested."""
    from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
        calibrate_reactivation_from_data,
        collect_model_data_driven_reactivation_policies,
        is_data_driven_reactivation_policy,
        normalize_reactivation_init_policy,
        reactivation_policy_to_calibration_mode,
    )

    config_requested_policies = set()

    try:
        reactivation_cfg = config.model.core.reactivation
        init_policy = normalize_reactivation_init_policy(
            getattr(reactivation_cfg, "init_policy", "analytical")
        )
        if is_data_driven_reactivation_policy(init_policy):
            config_requested_policies.add(init_policy)
    except AttributeError:
        # The reactivation config path is optional; fall back to collecting
        # any per-layer policies directly from the built modules below.
        pass

    layer_requested_policies, saw_layer_policy = (
        collect_model_data_driven_reactivation_policies(model)
    )

    # Once the model is built, module-level policies are authoritative. This is
    # important for recurrent aliases where the factory may override stale YAML
    # policies for one population while preserving data-driven policies for the
    # other population.
    requested_policies = (
        layer_requested_policies if saw_layer_policy else config_requested_policies
    )
    if not requested_policies:
        return None
    initial_checkpoint = getattr(
        getattr(config, "model", None), "initial_checkpoint", None
    )
    if initial_checkpoint:
        # The warm-start checkpoint already holds trained gates; recalibrating
        # would overwrite them.
        policies = sorted(requested_policies)
        logger.warning(
            "[reactivation-calibration] skipped: model.initial_checkpoint (%s) "
            "provides the reactivation gates, so the data-driven init_policy (%s) "
            "does not recalibrate them.",
            initial_checkpoint,
            ", ".join(policies),
        )
        return {
            "calibration": "skipped_initial_checkpoint",
            "policies": policies,
            "initial_checkpoint": str(initial_checkpoint),
        }
    if _configured_main_strategy(config) == "presentation_stream":
        # Calibration batches come from the training set's items, which for a
        # presentation stream are not the sampled training presentations (for
        # contextual_remapping: clean images at gain 1 with context 0).
        raise ValueError(
            "Data-driven reactivation initialization "
            f"({', '.join(sorted(requested_policies))}) is not supported with the "
            "presentation_stream strategy: calibration would fit voltages of the "
            "training set's items, not of the sampled training presentations. Use "
            "reactivation init_policy 'fixed' or 'analytical'."
        )

    if saw_layer_policy:
        mode = None
        policy_msg = ", ".join(sorted(requested_policies))
    elif len(requested_policies) == 1:
        init_policy = next(iter(requested_policies))
        mode = reactivation_policy_to_calibration_mode(init_policy)
        policy_msg = init_policy
    else:
        mode = None
        policy_msg = ", ".join(sorted(requested_policies))

    logger.info("[reactivation-calibration] init_policy=%s", policy_msg)

    device = next(model.parameters()).device
    # Don't pass a global k — let each layer use its own
    # reactivation_sigma_aware_k (threaded by the factory from config).
    # This respects per-population overrides in unified_ei / recurrent configs.

    common_config = getattr(
        getattr(getattr(config, "training", None), "main", None), "common", None
    )
    if n_batches is None:
        n_batches = int(
            getattr(common_config, "reactivation_initialization_num_batches", 3)
        )
    if batch_size is None:
        batch_size = int(
            getattr(common_config, "reactivation_initialization_batch_size", 256)
        )
    if n_iterations is None:
        configured_iterations = getattr(
            common_config,
            "reactivation_initialization_max_iterations",
            None,
        )
        if configured_iterations is not None:
            n_iterations = int(configured_iterations)
    n_batches = max(1, int(n_batches))
    batch_size = max(1, int(batch_size))
    if n_iterations is not None:
        n_iterations = max(1, int(n_iterations))
    experiment_config = getattr(config, "experiment", None)
    calibration_loader_seed = resolve_experiment_seeds(experiment_config).loader_seed
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **seeded_dataloader_kwargs(
            calibration_loader_seed,
            stream=2,
        ),
        **dataloader_kwargs_from_config(train_dataset, common_config, device=device),
    )
    is_recurrent = getattr(model, "_is_recurrent_core", False)
    batches = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            sample = next(it)
        except StopIteration:
            break
        if isinstance(sample, (list, tuple)):
            x = sample[0]
            x = x.to(device) if torch.is_tensor(x) else x
            if is_recurrent and len(sample) > 2:
                seq_lengths = sample[2]
                seq_lengths = (
                    seq_lengths.to(device)
                    if torch.is_tensor(seq_lengths)
                    else seq_lengths
                )
                batches.append((x, seq_lengths))
            else:
                batches.append(x)
        else:
            batches.append(sample.to(device))

    if not batches:
        raise RuntimeError(_no_calibration_batches_message(train_dataset, batch_size))

    # Recurrent models expect [B, T, D]. Preserve full sequence inputs when the
    # dataset already provides them, and only synthesize a length-1 time axis
    # for flat [B, D] batches.
    if is_recurrent:
        normalized_batches = []
        added_time_axis = False
        for batch in batches:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
                rest = list(batch[1:])
            else:
                x = batch
                rest = []

            if torch.is_tensor(x) and x.dim() == 2:
                x = x.unsqueeze(1)
                added_time_axis = True

            normalized_batches.append((x, *rest) if rest else x)

        batches = normalized_batches
        if is_main:
            shape_msg = "[B, 1, D]" if added_time_axis else "native [B, T, D]"
            logger.info(
                "[reactivation-calibration] recurrent model detected; using %s batches",
                shape_msg,
            )

    # Iterate: writing new (m, b) at layer i reshapes the V distribution at
    # every downstream layer, so a single-pass calibration is only
    # self-consistent at layer 0. Repeating 2-3 times converges the chain.
    # Median/MAD mode is fairly stable; quantile mode is very sensitive and
    # diverges without this iteration.
    diag = None
    quantile_aggregation = "chunk_median" if is_recurrent else "global"
    # The correction propagates ~one layer per iteration, so a fixed small
    # iteration count leaves the deepest layers of a many-layer stack calibrated
    # on the still-uncalibrated upstream distribution (the reactivation then
    # saturates and the stack collapses to chance).  Iterate until the (m, b)
    # chain converges, with a generous cap; shallow models still break after
    # ~2-3 iterations, so converged runs are unchanged.
    # Production calls iterate to convergence with a generous cap. An explicit
    # value remains a deterministic test/debug override.
    convergence_max_iterations = (
        50 if n_iterations is None else max(1, int(n_iterations))
    )
    convergence_tol = 1e-3
    previous_mb: dict[str, tuple[float, float]] | None = None
    converged = False
    iterations_completed = 0
    for iteration in range(convergence_max_iterations):
        diag = calibrate_reactivation_from_data(
            model,
            batches,
            k=None,
            device=device,
            mode=mode,
            quantile_aggregation=quantile_aggregation,
        )
        if not diag:
            raise RuntimeError(
                "Data-driven reactivation initialization was requested, but "
                "calibration found no eligible DendriticBranchLayers."
            )
        iterations_completed = iteration + 1
        current_mb = {name: (d["m"], d["b"]) for name, d in diag.items()}
        converged = False
        if previous_mb is not None and set(previous_mb) == set(current_mb):
            max_mb_change = max(
                max(
                    abs(current_mb[name][0] - previous_mb[name][0]),
                    abs(current_mb[name][1] - previous_mb[name][1]),
                )
                for name in current_mb
            )
            converged = max_mb_change < convergence_tol
        previous_mb = current_mb
        if is_main:
            m_values = [d["m"] for d in diag.values()]
            b_values = [d["b"] for d in diag.values()]
            sig_values = [d["V_std"] for d in diag.values()]
            logger.info(
                "[reactivation-calibration] iter %s/%s | mode=%s | %s layers "
                "calibrated | m range [%.4f, %.4f] | b range [%.4f, %.4f] "
                "| sigma range [%.4f, %.4f] | aggregation=%s | k=per-layer "
                "n_batches=%s",
                iteration + 1,
                convergence_max_iterations,
                mode if mode is not None else "per-layer",
                len(diag),
                min(m_values),
                max(m_values),
                min(b_values),
                max(b_values),
                min(sig_values),
                max(sig_values),
                quantile_aggregation,
                n_batches,
            )
        if converged:
            if is_main:
                logger.info(
                    "[reactivation-calibration] converged after %s iterations",
                    iteration + 1,
                )
            break
    if not converged and is_main:
        logger.warning(
            "[reactivation-calibration] reached %s iterations without meeting "
            "the %.1e (m, b) convergence tolerance",
            convergence_max_iterations,
            convergence_tol,
        )
    if is_main and diag:
        for name, d in diag.items():
            extra = ""
            if d.get("occupancy_m_was_clamped"):
                extra += (
                    f" occ[dq_raw={d.get('occupancy_delta_q_raw', float('nan')):.3g}"
                    f" dq_used={d.get('occupancy_delta_q_used', float('nan')):.3g}"
                    f" m_raw={d.get('occupancy_m_raw', float('nan')):.3g} CLAMPED]"
                )
            if d.get("calibration_reverted"):
                extra += " REVERTED"
            logger.info(
                "[reactivation-calibration]   %s: m=%.4f b=%.4f "
                "V[mean=%.4f std=%.4f mad_std=%.4f q10=%.3g q90=%.3g "
                "min=%.3g max=%.3g] n=%s%s",
                name,
                d["m"],
                d["b"],
                d["V_mean"],
                d["V_std"],
                d.get("V_mad_std", float("nan")),
                d.get("V_q10", float("nan")),
                d.get("V_q90", float("nan")),
                d.get("V_min", float("nan")),
                d.get("V_max", float("nan")),
                d["n"],
                extra,
            )
    return {
        "requested": True,
        "policies": sorted(requested_policies),
        "mode": mode if mode is not None else "per-layer",
        "n_batches": len(batches),
        "batch_size": int(batch_size),
        "iterations_completed": int(iterations_completed),
        "converged": bool(converged),
        "convergence_tolerance": float(convergence_tol),
        "layers": diag,
    }


def launch_ranks() -> tuple[int, int, int]:
    """``(rank, local_rank, world_size)`` of a ``torchrun`` launch, else ``(0, 0, 1)``.

    Read from the environment ``torchrun`` sets, before any process group
    exists, and the local GPU is selected for a multi-process launch.
    """
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def setup_distributed(timeout_seconds: Optional[int] = None) -> tuple[int, int, int]:
    """Initialize distributed training if launched via ``torchrun``.

    ``torchrun`` sets ``RANK``, ``LOCAL_RANK``, and ``WORLD_SIZE``
    automatically.  When these are absent the function returns
    ``(0, 0, 1)`` and does nothing, preserving single-GPU behaviour.
    ``timeout_seconds`` (``distributed.timeout_seconds``) bounds every
    collective; None keeps PyTorch's NCCL default of 600 seconds.

    Returns:
        (rank, local_rank, world_size)
    """
    rank, local_rank, world_size = launch_ranks()
    if world_size > 1 and not dist.is_initialized():
        options = {}
        if timeout_seconds is not None:
            options["timeout"] = timedelta(seconds=int(timeout_seconds))
        dist.init_process_group(backend="nccl", **options)
    return rank, local_rank, world_size


def _resolve_distributed_mode(config, *, launcher: str, world_size: int) -> None:
    """Check ``distributed.mode`` against the launch and record the runtime.

    ``train_experiments`` trains in one process (``none``) or with DDP under
    ``torchrun`` (``ddp``); ``train_experiments_fsdp`` trains with FSDP
    (``fsdp``). A mode the launch cannot run raises; a mode that differs
    from the actual runtime otherwise is replaced, with a warning, so
    ``config.json`` records how the run trained.
    """
    distributed = getattr(config, "distributed", None)
    if distributed is None:
        return
    mode = str(getattr(distributed, "mode", "none") or "none").strip().lower()
    if launcher == "fsdp":
        if mode == "ddp":
            raise ValueError(
                "distributed.mode: ddp trains with train_experiments under "
                "torchrun; train_experiments_fsdp trains with FSDP"
            )
        actual = "fsdp"
    else:
        if mode == "fsdp":
            raise ValueError(
                "distributed.mode: fsdp trains with train_experiments_fsdp: "
                "torchrun --nproc_per_node=<gpus> -m "
                "dendritic_modeling.scripts.training.train_experiments_fsdp <config>"
            )
        if mode == "ddp" and "RANK" not in os.environ:
            raise ValueError(
                "distributed.mode: ddp needs torchrun: torchrun "
                "--nproc_per_node=<gpus> -m "
                "dendritic_modeling.scripts.training.train_experiments <config>"
            )
        actual = "ddp" if world_size > 1 else "none"
    if mode != actual:
        logger.warning(
            "distributed.mode is %r but this run trains as %r; config.json "
            "records %r.",
            mode,
            actual,
            actual,
        )
        distributed.mode = actual


def normalize_optimizer_config(optimizer_cfg) -> OptimizerConfig:
    """Normalize optimizer config to a typed OptimizerConfig."""
    if optimizer_cfg is None:
        return OptimizerConfig()

    if isinstance(optimizer_cfg, OptimizerConfig):
        return optimizer_cfg

    if isinstance(optimizer_cfg, DictConfig):
        optimizer_cfg = OmegaConf.to_container(optimizer_cfg, resolve=True)

    if hasattr(optimizer_cfg, "asdict") and callable(optimizer_cfg.asdict):
        optimizer_cfg = optimizer_cfg.asdict()
    elif not isinstance(optimizer_cfg, dict) and hasattr(optimizer_cfg, "__dict__"):
        optimizer_cfg = {
            key: value
            for key, value in vars(optimizer_cfg).items()
            if not key.startswith("_")
        }

    if not isinstance(optimizer_cfg, dict):
        return OptimizerConfig()

    allowed_fields = set(OptimizerConfig.__dataclass_fields__.keys())
    sanitized = {k: v for k, v in optimizer_cfg.items() if k in allowed_fields}
    return OptimizerConfig(**sanitized)


def _maybe_attach_pruning_results(
    analysis_manager: AnalysisManager,
    training_results: dict | None,
) -> None:
    """Attach pruning metrics to analysis when the training result includes them."""
    if training_results and (
        "pre_pruning_performance" in training_results
        or "post_pruning_performance" in training_results
    ):
        analysis_manager.set_pruning_results(training_results)


def _resolve_final_epoch_for_performance_plot(training_config) -> int | None:
    """Return final epoch metadata from the training config when available."""
    if (
        training_config is not None
        and hasattr(training_config, "main")
        and hasattr(training_config.main, "common")
        and hasattr(training_config.main.common, "epochs")
    ):
        return training_config.main.common.epochs
    return None


def _maybe_plot_performance_evolution(
    *,
    run_save_path: str,
    analysis_manager: AnalysisManager,
    training_config=None,
) -> None:
    """Generate final performance-evolution plots when epoch tracking exists."""
    if not analysis_manager.config.performance.training:
        return

    performance_dir = os.path.join(run_save_path, "performance", "epochs")
    logger.info(f"Checking for performance evolution data in: {performance_dir}")
    if not os.path.exists(performance_dir):
        logger.warning(f"Performance directory does not exist: {performance_dir}")
        return

    logger.info(
        "Performance directory exists. Generating performance evolution plots..."
    )
    try:
        final_epoch = _resolve_final_epoch_for_performance_plot(training_config)
        pruning_performance = getattr(analysis_manager, "pruning_results", None)
        logger.info(f"Pruning results available: {pruning_performance is not None}")

        plotter = PerformancePlotter()
        plotter.plot_performance_evolution(
            performance_dir=performance_dir,
            save_path=os.path.join(run_save_path, "performance"),
            pruning_performance=pruning_performance,
            final_epoch=final_epoch,
        )
        logger.info("Performance evolution plots generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate performance evolution plots: {e}")
        import traceback

        logger.error(traceback.format_exc())


def run_analysis(
    train_ds: torch.utils.data.Dataset,
    valid_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    run_save_path: str,
    analysis_manager: AnalysisManager | None = None,
    training: bool = False,
    training_results: dict | None = None,
    training_config=None,
):
    """Run analysis using the AnalysisManager."""
    logger.info("Running analyses...")
    if analysis_manager is not None:
        _maybe_attach_pruning_results(analysis_manager, training_results)

        analysis_manager.run_analysis(filename="final", training=training)

        if not training:
            _maybe_plot_performance_evolution(
                run_save_path=run_save_path,
                analysis_manager=analysis_manager,
                training_config=training_config,
            )

        logger.info("Analysis completed")


_LEARNING_RATE_ARGUMENTS = (
    "lr",
    "topk_lr",
    "blocklinear_lr",
    "reactivation_lr",
    "decoder_lr",
)
_IMAGENET_SAMPLE_ARGUMENTS = ("train_samples_per_class", "val_samples_per_class")


def _cli_overrides(args) -> dict[str, Any]:
    """Return the configuration overrides requested on the command line.

    The overrides are applied to the YAML before validation, so they are
    checked exactly like values written in the file.
    """
    if not args:
        return {}
    overrides: dict[str, Any] = {}

    def put(path: str, value: Any) -> None:
        if value is not None:
            overrides[path] = value

    put("training.main.strategy", getattr(args, "learning_strategy", None))
    put(
        "training.main.learning_strategy_config.error_broadcast_mode",
        getattr(args, "error_broadcast_mode", None),
    )
    put("outputs.results_dir", getattr(args, "output_dir", None))
    put("data.base_dir", getattr(args, "data_dir", None))
    put("outputs.run_name", getattr(args, "run_name", None))
    put("experiment.seed", getattr(args, "seed", None))
    epochs = getattr(args, "epochs", None)
    if epochs is not None and epochs < 1:
        raise ValueError("--epochs must be positive")
    put("training.main.common.epochs", epochs)
    for name in _LEARNING_RATE_ARGUMENTS:
        value = getattr(args, name, None)
        if value is not None and value < 0:
            raise ValueError(f"--{name} must be non-negative")
        put(f"training.main.common.param_groups.{name}", value)
    for name in _IMAGENET_SAMPLE_ARGUMENTS:
        value = getattr(args, name, None)
        if value is not None and value < 1:
            raise ValueError(f"--{name} must be positive")
        put(f"data.dataset_params.imagenet.{name}", value)
    return overrides


def _reject_unread_cli_overrides(config, overrides: dict[str, Any]) -> None:
    """Fail when an override sets dataset options the selected dataset ignores."""
    imagenet = [
        f"--{name}"
        for name in _IMAGENET_SAMPLE_ARGUMENTS
        if f"data.dataset_params.imagenet.{name}" in overrides
    ]
    if not imagenet:
        return
    dataset_name = str(getattr(config.data, "dataset_name", ""))
    if dataset_name != "imagenet":
        raise ValueError(
            f"{', '.join(imagenet)} applies only to data.dataset_name: imagenet "
            f"(this config uses {dataset_name!r})"
        )


def _load_and_prepare_training_config(
    config_path: str,
    args,
    *,
    is_main: bool,
    world_size: int,
) -> _PreparedTrainingConfig:
    """Load, validate, safeguard, and split the training configuration."""
    overrides = _cli_overrides(args)
    config = load_config(config_path, overrides=overrides)
    _reject_unread_cli_overrides(config, overrides)
    if is_main:
        logger.info(f"Loaded configuration from {config_path}")
        for path, value in overrides.items():
            logger.info("Command-line override: %s = %r", path, value)
        if world_size > 1:
            logger.info(f"Distributed run: world_size={world_size}")

    if is_main:
        logger.info("Validating configuration...")
    validation_passed = validate_training_stability(config)
    if not validation_passed:
        logger.error(
            "Configuration validation failed. Please address the errors above."
        )
        raise ValueError("Invalid configuration - critical errors detected")

    return _PreparedTrainingConfig(
        config=config,
        experiment_config=config.experiment,
        data_config=config.data,
        model_config=config.model,
        training_config=config.training,
        analysis_config=config.analysis,
        outputs_config=config.outputs,
    )


def _resolve_main_training_settings(main_train_config) -> _MainTrainingSettings:
    """Resolve main training settings from dict or object config shapes."""
    if isinstance(main_train_config, dict):
        common_config = main_train_config.get("common", {})
        strategy = main_train_config.get("strategy", "standard")
        optimizer_config = normalize_optimizer_config(
            main_train_config.get("optimizer")
        )
    else:
        common_config = main_train_config.common
        strategy = main_train_config.strategy
        optimizer_config = normalize_optimizer_config(
            getattr(main_train_config, "optimizer", None)
        )

    if isinstance(common_config, dict):
        # Unset keys take the typed defaults, as on the typed path.
        param_groups = common_config.get("param_groups", {})
        common_config = apply_renamed_fields(
            CommonTrainerConfig, common_config, section="training.main.common"
        )
        synapse_decay_rate = common_config.get(
            "synapse_decay_rate",
            field_default(CommonTrainerConfig, "synapse_decay_rate"),
        )
        weight_boosting = common_config.get(
            "weight_boosting", field_default(CommonTrainerConfig, "weight_boosting")
        )
        common_epochs = common_config.get(
            "epochs", field_default(CommonTrainerConfig, "epochs")
        )
    else:
        param_groups = common_config.param_groups
        synapse_decay_rate = common_config.synapse_decay_rate
        weight_boosting = getattr(common_config, "weight_boosting", False)
        common_epochs = common_config.epochs

    return _MainTrainingSettings(
        common_config=common_config,
        strategy=strategy,
        optimizer_config=optimizer_config,
        param_groups=param_groups,
        synapse_decay_rate=synapse_decay_rate,
        weight_boosting=weight_boosting,
        common_epochs=common_epochs,
    )


def _install_encoder_network(unwrapped_model, encoder_network) -> None:
    """Install the encoder the main phase uses on the unwrapped model.

    An autoencoder contributes its (pretrained) encoder.
    """
    if isinstance(encoder_network, Identity):
        unwrapped_model.encoder_network = encoder_network
    elif hasattr(encoder_network, "encoder"):
        unwrapped_model.encoder_network = encoder_network.encoder
    else:
        unwrapped_model.encoder_network = encoder_network


def _record_model_size_summary(unwrapped_model, config, run_save_path: str) -> int:
    """Persist model size metadata and log size-based configuration suggestions."""
    nparams = sum(p.numel() for p in unwrapped_model.parameters())
    save_dict({"nparams": nparams}, run_save_path, "nparams")
    resources = model_resource_summary(unwrapped_model)
    resources["runtime_device"] = "cuda" if torch.cuda.is_available() else "cpu"
    resources["cuda_peak_memory_allocated_bytes"] = None
    resources["cuda_peak_memory_reserved_bytes"] = None
    if torch.cuda.is_available():
        try:
            device = torch.cuda.current_device()
            resources["cuda_device_name"] = torch.cuda.get_device_name(device)
            resources["cuda_peak_memory_allocated_bytes"] = int(
                torch.cuda.max_memory_allocated(device)
            )
            resources["cuda_peak_memory_reserved_bytes"] = int(
                torch.cuda.max_memory_reserved(device)
            )
        except (AssertionError, RuntimeError) as error:
            # Some login/test environments expose a CUDA-enabled build without
            # assigning a usable device. Preserve a valid resource artifact.
            resources["cuda_memory_measurement_error"] = str(error)
    save_dict(resources, run_save_path, "model_resources.json")

    # Companion artifact: the operator axes of the constructed core, read from
    # its dendritic layers. Best-effort by design: the card must never
    # interfere with training.
    try:
        from dendritic_modeling.config.operator_card import build_operator_card

        save_dict(
            build_operator_card(
                getattr(config.model, "core", None), model=unwrapped_model
            ),
            run_save_path,
            "operator_card.json",
        )
    except Exception as card_error:  # pragma: no cover - defensive guard
        logger.warning("Skipping operator card: %s", card_error)

    return nparams


def _compact_training_results(value: Any) -> Any:
    """Remove checkpoint-sized state dictionaries from a result tree.

    Trainer return values mix small scalar/curve metadata with complete model
    states.  Checkpoints already preserve those states in PyTorch format; JSON
    artifacts should remain compact enough to hash, inspect, and collect.
    """

    if isinstance(value, dict):
        return {
            key: _compact_training_results(item)
            for key, item in value.items()
            if not str(key).endswith("state_dict")
        }
    if isinstance(value, tuple):
        return [_compact_training_results(item) for item in value]
    if isinstance(value, list):
        return [_compact_training_results(item) for item in value]
    return value


def _save_training_summary(
    training_results: dict[str, Any], run_save_path: str
) -> None:
    """Persist state-free training curves and best-epoch metadata."""

    save_dict(
        _compact_training_results(training_results),
        run_save_path,
        "training_summary.json",
    )


def _save_run_config(config, run_save_path: str) -> None:
    """Persist the resolved run config with legacy flag canonicalization."""
    config_payload = asdict(config)
    if canonicalize_model_core_flags is not None:
        config_payload = canonicalize_model_core_flags(config_payload)
    if canonicalize_rnn_core_flags is not None:
        config_payload = canonicalize_rnn_core_flags(config_payload)
    save_dict(config_payload, run_save_path, "config.json")


def _check_encoder_request(
    *,
    autoencoder: bool,
    name: str,
    pretraining,
    initial_checkpoint,
    world_size: int,
    fsdp: bool,
) -> None:
    """Raise for an encoder request the main phase cannot run.

    ``autoencoder`` says whether ``model.encoder`` builds an autoencoder and
    ``name`` names what it builds. The same rules run before the run directory
    is created (:func:`_validate_encoder_request`) and on the built encoder.
    """
    epochs = 0 if pretraining is None else int(pretraining.epochs)
    if not autoencoder:
        if epochs > 0:
            raise ValueError(
                f"training.encoder_pretraining.epochs={epochs} pretrains autoencoder "
                f"encoders, but model.encoder builds {name}, "
                "which the main phase trains with the core; set it to 0"
            )
        return
    if fsdp:
        raise ValueError(
            f"model.encoder builds an autoencoder ({name}); FSDP training does not "
            "support autoencoder encoders, so train in a single process or with DDP"
        )
    if epochs > 0 and initial_checkpoint:
        raise ValueError(
            "training.encoder_pretraining.epochs is set, but model.initial_checkpoint "
            "already holds the trained encoder; set the epochs to 0"
        )
    if epochs == 0 and not initial_checkpoint:
        raise ValueError(
            f"model.encoder builds an autoencoder ({name}), which must be trained "
            "before the main phase: set training.encoder_pretraining.epochs, or load "
            "a trained one with model.initial_checkpoint"
        )
    if epochs > 0 and world_size > 1:
        raise ValueError(
            "training.encoder_pretraining runs in single-process training; "
            "pretrain in a single-process run and start the distributed run "
            "from its checkpoint (model.initial_checkpoint)"
        )


def _validate_encoder_request(
    model_config, training_config, *, world_size: int, fsdp: bool = False
) -> None:
    """Reject an unsupported encoder request before the run directory exists.

    Built-in encoder types tell from their name whether they build an
    autoencoder. An encoder that a registered model builder or an extension
    architecture builds is checked once it is built, by
    :func:`_encoder_datasets`.
    """
    encoder_config = getattr(model_config, "encoder", None)
    if encoder_config is None or resolve_model_builder(model_config) is not None:
        return
    encoder_type = str(getattr(encoder_config, "type", None) or "").lower()
    autoencoder = builds_autoencoder(encoder_type)
    if autoencoder is None:
        return
    cls = builtin_architecture_class(encoder_type)
    _check_encoder_request(
        autoencoder=autoencoder,
        name=cls.__name__ if cls is not None else f"the {encoder_type!r} architecture",
        pretraining=getattr(training_config, "encoder_pretraining", None),
        initial_checkpoint=getattr(model_config, "initial_checkpoint", None),
        world_size=world_size,
        fsdp=fsdp,
    )


def _encoder_datasets(
    *,
    encoder_network,
    train_ds,
    valid_ds,
    test_ds,
    pretraining=None,
    initial_checkpoint=None,
    run_save_path: str | None = None,
    seed: int = 0,
    loader_seed: int | None = None,
    world_size: int = 1,
    fsdp: bool = False,
):
    """Return the encoder and the datasets the main phase trains on.

    The main phase trains a regular encoder jointly with the core. An
    autoencoder encoder must be trained first: by reconstruction pretraining
    (``training.encoder_pretraining``, in single-process training) or as part
    of ``model.initial_checkpoint``. Its encoder is then the model's encoder,
    frozen unless ``training.encoder_pretraining.freeze_encoder`` is false.
    """
    autoencoder = isinstance(encoder_network, BaseAutoencoder)
    _check_encoder_request(
        autoencoder=autoencoder,
        name=type(encoder_network).__name__,
        pretraining=pretraining,
        initial_checkpoint=initial_checkpoint,
        world_size=world_size,
        fsdp=fsdp,
    )
    if not autoencoder:
        return encoder_network, train_ds, valid_ds, test_ds
    epochs = 0 if pretraining is None else int(pretraining.epochs)
    if epochs > 0:
        save_path = (
            None
            if run_save_path is None
            else os.path.join(run_save_path, "encoder_pretraining")
        )
        summary = pretrain_autoencoder(
            encoder_network,
            train_ds,
            valid_ds,
            pretraining,
            save_path=save_path,
            seed=seed,
            loader_seed=loader_seed,
        )
        if save_path is not None:
            save_dict(summary, save_path, "summary.json")
    if pretraining is None or pretraining.freeze_encoder:
        encoder_network.encoder.freeze()
    return encoder_network, train_ds, valid_ds, test_ds


def _create_analysis_manager(
    *,
    unwrapped_model,
    encoded_train_ds,
    encoded_valid_ds,
    encoded_test_ds,
    analysis_config,
    run_save_path: str,
    evaluation_seed: int | None = None,
    probe_seed: int | None = None,
):
    """Create the analysis manager from encoded train/valid/test datasets."""
    encoded_data = {
        "train": encoded_train_ds,
        "valid": encoded_valid_ds,
        "test": encoded_test_ds,
    }
    manager_kwargs = {
        "model": unwrapped_model,
        "data": encoded_data,
        "analysis_config": analysis_config,
        "save_root": run_save_path,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if evaluation_seed is not None:
        manager_kwargs["evaluation_seed"] = int(evaluation_seed)
    if probe_seed is not None:
        manager_kwargs["probe_seed"] = int(probe_seed)
    return AnalysisManager(**manager_kwargs)


def _validate_sealed_test_analysis(experiment_config, analysis_config) -> None:
    """Reject analysis pipelines that could inspect a sealed test audit set."""
    if experiment_config is None or not bool(
        getattr(experiment_config, "sealed_test", False)
    ):
        return
    if isinstance(analysis_config, dict) or OmegaConf.is_config(analysis_config):
        entries = analysis_config.items()
    else:
        entries = vars(analysis_config).items()
    enabled = sorted(
        name
        for name, value in entries
        if bool(
            value.get("enabled", False)
            if isinstance(value, dict) or OmegaConf.is_config(value)
            else getattr(value, "enabled", False)
        )
    )
    if enabled:
        raise ValueError(
            "experiment.sealed_test=true forbids enabled analyses; disable: "
            + ", ".join(enabled)
        )


def _prepare_encoded_training_state(
    *,
    config,
    unwrapped_model,
    encoder_network,
    training_config,
    model_config,
    data_config,
    train_ds,
    valid_ds,
    test_ds,
    analysis_config,
    run_save_path: str,
    world_size: int,
    is_main: bool,
) -> _EncodedTrainingState:
    """Prepare encoded datasets, analysis manager, and post-encoder calibration."""
    _validate_sealed_test_analysis(
        getattr(config, "experiment", None),
        analysis_config,
    )
    experiment_config = getattr(config, "experiment", None)
    resolved_seeds = (
        None
        if experiment_config is None
        else resolve_experiment_seeds(experiment_config)
    )
    encoder_network, encoded_train_ds, encoded_valid_ds, encoded_test_ds = (
        _encoder_datasets(
            encoder_network=encoder_network,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            pretraining=getattr(training_config, "encoder_pretraining", None),
            initial_checkpoint=getattr(model_config, "initial_checkpoint", None),
            run_save_path=run_save_path if is_main else None,
            seed=0 if resolved_seeds is None else resolved_seeds.seed,
            loader_seed=None if resolved_seeds is None else resolved_seeds.loader_seed,
            world_size=world_size,
        )
    )

    analysis_seed_kwargs = {}
    if resolved_seeds is not None:
        analysis_seed_kwargs = {
            "evaluation_seed": resolved_seeds.evaluation_seed,
            "probe_seed": resolved_seeds.probe_seed,
        }

    analysis_manager = _create_analysis_manager(
        unwrapped_model=unwrapped_model,
        encoded_train_ds=encoded_train_ds,
        encoded_valid_ds=encoded_valid_ds,
        encoded_test_ds=encoded_test_ds,
        analysis_config=analysis_config,
        run_save_path=run_save_path,
        **analysis_seed_kwargs,
    )

    _install_encoder_network(unwrapped_model, encoder_network)
    calibration_summary = _maybe_calibrate_data_driven_reactivation(
        config=config,
        model=unwrapped_model,
        train_dataset=encoded_train_ds,
        is_main=is_main,
    )
    # DDP replicas calibrated after wrapping; rank 0's gates become everyone's.
    broadcast_reactivation_state(unwrapped_model)
    if is_main and calibration_summary is not None:
        save_dict(
            calibration_summary,
            run_save_path,
            "reactivation_calibration.json",
        )
        if calibration_summary.get("calibration") != "skipped_initial_checkpoint":
            _dump_post_calibration_gate_stats_diagnostic(
                unwrapped_model,
                run_save_path,
            )

    return _EncodedTrainingState(
        encoder_network=encoder_network,
        encoded_train_ds=encoded_train_ds,
        encoded_valid_ds=encoded_valid_ds,
        encoded_test_ds=encoded_test_ds,
        analysis_manager=analysis_manager,
    )


def _maybe_plot_multistage_training_performance(
    *,
    strategy: str,
    analysis_config,
    run_save_path: str,
    main_train_config,
    common_epochs: int,
    analysis_manager,
) -> None:
    """Plot multi-stage training performance when epoch traces are available."""
    if strategy != "multi_stage":
        return
    if not analysis_config.performance.training:
        return

    performance_dir = os.path.join(run_save_path, "performance", "epochs")
    if not os.path.exists(performance_dir):
        return

    try:
        multi_stage = main_train_config.strategies.multi_stage
        stages = multi_stage.stages
        total_epochs = sum(
            getattr(stage, "epochs", None) or common_epochs for stage in stages
        )
        plotter = PerformancePlotter()
        plotter.plot_performance_evolution(
            performance_dir=performance_dir,
            save_path=os.path.join(run_save_path, "performance"),
            pruning_performance=getattr(analysis_manager, "pruning_results", None),
            final_epoch=total_epochs,
        )
    except Exception as e:
        logger.error(f"Failed to generate performance plots: {e}")


def _main_steps_per_epoch(strategy, trainer_config_dict, train_dataset) -> int:
    batch_size = int(trainer_config_dict["batch_size"])
    if strategy == "presentation_stream":
        stream_config = trainer_config_dict.get("presentation_stream_config") or {}
        presentations = int(stream_config.get("presentations_per_epoch", 65536))
        return math.ceil(presentations / batch_size)
    return math.ceil(len(train_dataset) / batch_size)


def _build_topology_controllers(
    *,
    unwrapped_model,
    strategy: str,
    trainer_config_dict: dict,
    experiment_config,
    train_dataset,
) -> list:
    """Optimizer-step topology controllers configured under training.main.common."""
    impact_options = trainer_config_dict.pop("impact_rewiring", None) or {}
    if not impact_options:
        return []
    if strategy not in {"standard", "presentation_stream"}:
        raise ValueError(
            "training.main.common.impact_rewiring supports the standard and "
            f"presentation_stream strategies, not {strategy!r}"
        )
    if dist.is_available() and dist.is_initialized():
        raise ValueError("impact_rewiring does not support distributed training")
    seed = resolve_experiment_seeds(experiment_config).topology_seed
    return [
        build_population_impact_rewiring(
            unwrapped_model.core_network,
            impact_options,
            steps_per_epoch=_main_steps_per_epoch(
                strategy, trainer_config_dict, train_dataset
            ),
            epochs=int(trainer_config_dict["epochs"]),
            seed=int(seed),
        )
    ]


def _validate_pruning_request(pruning_config, unwrapped_model) -> None:
    """Fail before training when the post-training pruning step cannot work."""
    if not pruning_config or not pruning_config.get("enabled", False):
        return
    if not isinstance(unwrapped_model, torch.nn.Module):
        return  # the weight selection needs module introspection
    from dendritic_modeling.config.regularization import PruningConfig, SelectiveConfig
    from dendritic_modeling.training.regularization import PruningManager

    options = dict(pruning_config)
    if isinstance(options.get("selective"), dict):
        options["selective"] = SelectiveConfig(**options["selective"])
    PruningManager(PruningConfig(**options)).validate_model(unwrapped_model)


def _run_main_training_phase(
    *,
    unwrapped_model,
    model,
    training_config,
    model_config,
    experiment_config,
    run_save_path: str,
    encoded_train_ds,
    encoded_valid_ds,
    analysis_manager,
) -> _MainTrainingRun:
    """Build the main trainer, run training, and return downstream state."""
    setup = _build_main_trainer(
        unwrapped_model=unwrapped_model,
        training_config=training_config,
        model_config=model_config,
        experiment_config=experiment_config,
        run_save_path=run_save_path,
        encoded_train_ds=encoded_train_ds,
        analysis_manager=analysis_manager,
    )
    training_results = setup.trainer.train(
        model=model, train_data=encoded_train_ds, valid_data=encoded_valid_ds
    )
    return _MainTrainingRun(
        main_train_config=setup.main_train_config,
        trainer_config_dict=setup.trainer_config_dict,
        optimizer=setup.optimizer,
        strategy=setup.strategy,
        common_epochs=setup.common_epochs,
        original_pruning_config=setup.original_pruning_config,
        training_results=training_results,
        pruning_trainer_config=setup.pruning_trainer_config,
    )


def _build_main_trainer(
    *,
    unwrapped_model,
    training_config,
    model_config,
    experiment_config,
    run_save_path: str,
    encoded_train_ds,
    analysis_manager,
) -> _MainTrainerSetup:
    """Build the optimizer and main trainer without training."""
    main_net_save_path = os.path.join(run_save_path, "main_network")
    os.makedirs(main_net_save_path, exist_ok=True)

    main_train_config = training_config.main
    main_trainer_config_dict = prepare_trainer_config(
        main_train_config=main_train_config,
        experiment_config=experiment_config,
        save_path=main_net_save_path,
    )
    resolved_seeds = resolve_experiment_seeds(experiment_config)
    main_trainer_config_dict["seed"] = resolved_seeds.seed
    main_trainer_config_dict["loader_seed"] = resolved_seeds.loader_seed

    settings = _resolve_main_training_settings(main_train_config)
    strategy = settings.strategy
    common_epochs = settings.common_epochs
    param_groups = normalize_param_groups_config(settings.param_groups)
    model_param_groups = unwrapped_model.get_param_groups(param_groups)
    # optimizer.lr is only the optimizer default; param_groups.lr is the base rate.
    validate_param_group_learning_rates(settings.optimizer_config, param_groups)
    base_optimizer = create_optimizer(model_param_groups, settings.optimizer_config)

    topology_controllers = _build_topology_controllers(
        unwrapped_model=unwrapped_model,
        strategy=strategy,
        trainer_config_dict=main_trainer_config_dict,
        experiment_config=experiment_config,
        train_dataset=encoded_train_ds,
    )
    optimizer = CustomWeightDecayOptimizer(
        model=unwrapped_model,
        optimizer=base_optimizer,
        weight_decay=settings.synapse_decay_rate,
        weight_boosting=settings.weight_boosting,
        topology_controllers=topology_controllers,
    )

    original_pruning_config = main_trainer_config_dict.get("pruning", {})
    _validate_pruning_request(original_pruning_config, unwrapped_model)
    main_trainer_config_dict["pruning"] = {}

    if strategy == "multi_stage":
        if "multi_stage_config" not in main_trainer_config_dict:
            raise ValueError(
                "Learning strategy is 'multi_stage' but no 'multi_stage_config' found"
            )
        # Stages that rebuild their optimizer use training.main.optimizer.
        main_trainer_config_dict["optimizer_config"] = settings.optimizer_config
        logger.info("Using multi-stage training strategy")

    main_trainer_config_dict["task"] = model_config.task
    # get_trainer pops keys from the dict it receives.
    pruning_trainer_config = copy.deepcopy(main_trainer_config_dict)

    main_trainer = get_trainer(
        strategy=strategy,
        optimizer=optimizer,
        trainer_config_dict=main_trainer_config_dict,
        analysis_manager=analysis_manager,
    )
    return _MainTrainerSetup(
        trainer=main_trainer,
        main_train_config=main_train_config,
        trainer_config_dict=main_trainer_config_dict,
        optimizer=optimizer,
        strategy=strategy,
        common_epochs=common_epochs,
        original_pruning_config=original_pruning_config,
        pruning_trainer_config=pruning_trainer_config,
    )


def _maybe_apply_pruning_step(
    *,
    original_pruning_config,
    is_main: bool,
    main_trainer_config_dict,
    strategy: str,
    optimizer,
    analysis_manager,
    model,
    common_epochs: int,
    encoded_train_ds,
    encoded_valid_ds,
    training_results: dict,
) -> dict:
    """Apply the separate pruning step when enabled and merge its results."""
    if not original_pruning_config or not original_pruning_config.get("enabled", False):
        return training_results

    if is_main:
        logger.info("Applying pruning as separate step...")
    main_trainer_config_dict["pruning"] = original_pruning_config

    pruning_trainer = get_trainer(
        strategy=strategy,
        optimizer=optimizer,
        trainer_config_dict=main_trainer_config_dict,
        analysis_manager=analysis_manager,
    )

    # Device setup only: the pruning step does not track a best state.
    pruning_trainer._initialize_attributes(model, common_epochs, snapshot_best=False)

    if is_main and analysis_manager is not None:
        analysis_manager.record_pruning_start()
    pruning_results = pruning_trainer._apply_pruning_and_evaluate(
        model=model, train_data=encoded_train_ds, valid_data=encoded_valid_ds
    )
    if is_main and analysis_manager is not None:
        analysis_manager.record_pruning_end(pruning_results)
    training_results.update(pruning_results)
    if is_main:
        logger.info("Pruning completed")
    return training_results


def _save_final_model_and_analysis(
    *,
    unwrapped_model,
    run_save_path: str,
    any_analysis_enabled: bool,
    strategy: str,
    train_ds,
    valid_ds,
    test_ds,
    analysis_manager,
    training_config,
    training_results: dict,
) -> None:
    """Save the final model and run non-multi-stage final analysis.

    When the final weights are those of the run's best-validation checkpoint
    (training restored them), ``final_model.pt`` is a hard link to it rather
    than a second copy. The main trainer writes that checkpoint under
    ``main_network/``; with several best checkpoints nothing is linked.
    """
    best_checkpoints = sorted(
        path
        for folder in (Path(run_save_path), Path(run_save_path) / "main_network")
        for path in folder.glob("*best_model.pt")
    )
    outcome = save_state_dict(
        unwrapped_model.state_dict(),
        os.path.join(run_save_path, "final_model.pt"),
        identical_to=best_checkpoints[0] if len(best_checkpoints) == 1 else None,
    )
    if outcome == "linked":
        logger.info("final_model.pt links %s (same weights)", best_checkpoints[0].name)

    if any_analysis_enabled and strategy != "multi_stage":
        run_analysis(
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            run_save_path=run_save_path,
            analysis_manager=analysis_manager,
            training_config=training_config,
            training=False,
            training_results=training_results,
        )


def _finalize_main_training_phase(
    *,
    is_main: bool,
    unwrapped_model,
    config,
    run_save_path: str,
    analysis_config,
    main_training: _MainTrainingRun,
    analysis_manager,
    model,
    encoded_train_ds,
    encoded_valid_ds,
    train_ds,
    valid_ds,
    test_ds,
    training_config,
    training_results: dict[str, Any],
) -> dict[str, Any]:
    """Run post-training bookkeeping, pruning, and final rank-0 outputs."""
    if is_main:
        _record_model_size_summary(unwrapped_model, config, run_save_path)
        any_analysis_enabled = analysis_config.any_enabled()
        _maybe_plot_multistage_training_performance(
            strategy=main_training.strategy,
            analysis_config=analysis_config,
            run_save_path=run_save_path,
            main_train_config=main_training.main_train_config,
            common_epochs=main_training.common_epochs,
            analysis_manager=analysis_manager,
        )

    training_results = _maybe_apply_pruning_step(
        original_pruning_config=main_training.original_pruning_config,
        is_main=is_main,
        main_trainer_config_dict=main_training.pruning_trainer_config,
        strategy=main_training.strategy,
        optimizer=main_training.optimizer,
        analysis_manager=analysis_manager,
        model=model,
        common_epochs=main_training.common_epochs,
        encoded_train_ds=encoded_train_ds,
        encoded_valid_ds=encoded_valid_ds,
        training_results=training_results,
    )

    if is_main:
        _save_training_summary(training_results, run_save_path)
        _save_final_model_and_analysis(
            unwrapped_model=unwrapped_model,
            run_save_path=run_save_path,
            any_analysis_enabled=any_analysis_enabled,
            strategy=main_training.strategy,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            analysis_manager=analysis_manager,
            training_config=training_config,
            training_results=training_results,
        )

    return training_results


def _broadcast_run_save_path(
    run_save_path: str,
    *,
    world_size: int,
    is_main: bool,
    device: str | torch.device = "cuda",
) -> str:
    """Broadcast the rank-0 run directory to all DDP ranks."""
    if world_size <= 1:
        return run_save_path

    if is_main:
        path_bytes = run_save_path.encode("utf-8")
        path_len = torch.tensor([len(path_bytes)], dtype=torch.long, device=device)
    else:
        path_len = torch.tensor([0], dtype=torch.long, device=device)
    dist.broadcast(path_len, src=0)

    if is_main:
        path_tensor = torch.tensor(list(path_bytes), dtype=torch.uint8, device=device)
    else:
        path_tensor = torch.empty(path_len.item(), dtype=torch.uint8, device=device)
    dist.broadcast(path_tensor, src=0)
    return bytes(path_tensor.cpu().tolist()).decode("utf-8")


def _reject_rank_dependent_rewiring(model) -> None:
    """Fail when DeepST rewiring would differ between data-parallel replicas.

    DDP synchronizes gradients, not the parameter writes rewiring makes, so
    every replica must take the same rewiring decisions. Seeded DeepST layers
    do; unseeded ones draw from each rank's RNG, and credit-gated DeepST
    decides from credit recorded before gradients are averaged.
    """
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.credit_deepst import (
        CreditGatedDeepstLinear,
    )
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
        DeepstLinear,
    )

    for name, module in model.named_modules():
        if isinstance(module, CreditGatedDeepstLinear):
            raise ValueError(
                f"{name}: credit_deepst does not support multi-process training; "
                "its rewiring uses credit recorded before gradients are averaged"
            )
        if isinstance(module, DeepstLinear) and module.seed is None:
            raise ValueError(
                f"{name}: an unseeded DeepST layer would rewire differently on "
                "each rank; set the topology seed (sparsity.indexed.seed or the "
                "population indexed_seed)"
            )


def _wrap_ddp_model(model, *, world_size: int, local_rank: int, is_main: bool):
    """Wrap the model in DDP when running with multiple processes."""
    if world_size <= 1:
        return model

    _reject_rank_dependent_rewiring(model)
    # Synapses that keep batch statistics reduce them over the ranks.
    mark_rank_reduced_statistics(model)
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    if is_main:
        logger.info("Model wrapped in DistributedDataParallel")
    return model


def _dump_init_gate_stats_diagnostic(unwrapped_model, run_save_path: str) -> None:
    """Write init-gate diagnostics without letting failures stop training."""
    try:
        dump_init_gate_stats(unwrapped_model, run_save_path)
    except Exception as exc:  # diagnostic must not break training
        logger.warning("init_gate_stats diagnostic failed: %s", exc)


def _dump_post_calibration_gate_stats_diagnostic(
    unwrapped_model, run_save_path: str
) -> None:
    """Persist the executed data-driven gate state; failure is fatal."""
    try:
        dump_init_gate_stats(
            unwrapped_model,
            run_save_path,
            name="post_calibration_gate_stats.json",
        )
    except Exception as exc:
        raise RuntimeError(
            "Data-driven reactivation calibration completed, but its post-"
            "calibration gate state could not be persisted."
        ) from exc


def _cleanup_training_runtime(*, world_size: int) -> None:
    """Tear down distributed runtime state."""
    if world_size > 1 and dist.is_initialized():
        dist.destroy_process_group()


def _log_model_summary_before_training(unwrapped_model, train_ds, config) -> None:
    """Log the pre-training architecture summary and parameter counts."""
    try:
        is_recurrent_core = getattr(unwrapped_model.core_network, "is_recurrent", False)
        if callable(is_recurrent_core):
            is_recurrent_core = is_recurrent_core()

        if is_recurrent_core:
            strategy = getattr(
                getattr(config.training, "main", None), "strategy", "standard"
            )
            if strategy not in ("recurrent", "multi_stage"):
                logger.warning(
                    "Recurrent model detected but training.main.strategy='%s'. "
                    "Use strategy='recurrent' for correct sequence handling.",
                    strategy,
                )
            total_params = sum(p.numel() for p in unwrapped_model.parameters())
            trainable_params = sum(
                p.numel() for p in unwrapped_model.parameters() if p.requires_grad
            )
            logger.info(
                "Skipping torchinfo summary for recurrent model. "
                f"total_params={total_params}, trainable_params={trainable_params}"
            )
        else:
            sample_input: torch.Tensor = train_ds[0][0]
            torchinfo.summary(
                unwrapped_model,
                input_size=(100, *sample_input.shape),
                col_names=["input_size", "output_size", "num_params"],
                col_width=25,
                row_settings=["depth"],
                depth=7,
            )
            logger.info("Model summary generated (see console output)")
    except Exception as e:
        logger.error(f"Failed to generate model summary before training: {e}")


def _setup_training_environment(
    *,
    config,
    model_config,
    training_config,
    data_config,
    outputs_config,
    experiment_config,
    world_size: int,
    local_rank: int,
    is_main: bool,
) -> _TrainingEnvironment:
    """Build datasets/model, apply DDP wrapping, and run rank-0 setup outputs."""
    run_save_path, train_ds, valid_ds, test_ds, model, encoder_network = (
        setup_environment(
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            outputs_config=outputs_config,
            experiment_config=experiment_config,
            is_main=is_main,
        )
    )

    run_save_path = _broadcast_run_save_path(
        run_save_path,
        world_size=world_size,
        is_main=is_main,
    )
    # Record the clipping operator the strategy applies when none is configured.
    main_training = getattr(getattr(config, "training", None), "main", None)
    common = getattr(main_training, "common", None)
    if common is not None and getattr(common, "grad_clip_mode", None) is None:
        common.grad_clip_mode = default_grad_clip_mode(
            getattr(main_training, "strategy", None),
            getattr(getattr(config, "distributed", None), "mode", None),
        )
    # Fill unset analysis seeds before config.json records the run.
    analysis_config = getattr(config, "analysis", None)
    if analysis_config is not None:
        resolved_seeds = resolve_experiment_seeds(experiment_config)
        fill_analysis_seeds(
            analysis_config,
            evaluation_seed=resolved_seeds.evaluation_seed,
            probe_seed=resolved_seeds.probe_seed,
        )

    model = _wrap_ddp_model(
        model,
        world_size=world_size,
        local_rank=local_rank,
        is_main=is_main,
    )

    if is_main:
        logger.info("Network architecture summary before training:")

    unwrapped_model = model.module if hasattr(model, "module") else model

    if is_main:
        _dump_init_gate_stats_diagnostic(unwrapped_model, run_save_path)
        _log_model_summary_before_training(unwrapped_model, train_ds, config)
        _save_run_config(config, run_save_path)

    return _TrainingEnvironment(
        run_save_path=run_save_path,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        model=model,
        unwrapped_model=unwrapped_model,
        encoder_network=encoder_network,
    )


def main(config_path: str, args=None):
    rank, local_rank, world_size = launch_ranks()
    is_main = rank == 0

    # Measure the complete per-process model build and training footprint.
    # Reset before dataset/model setup so architecture comparisons use the same
    # scope and the final resource artifact can be collected without rerunning.
    if torch.cuda.is_available():
        try:
            # ``launch_ranks`` already selected the local device for DDP.
            # Let PyTorch resolve the current device here; passing an integer
            # before the first CUDA context exists fails on some Slurm nodes.
            torch.cuda.reset_peak_memory_stats()
        except (AssertionError, RuntimeError) as error:
            logger.warning("CUDA peak-memory reset was unavailable: %s", error)

    prepared_config = _load_and_prepare_training_config(
        config_path,
        args,
        is_main=is_main,
        world_size=world_size,
    )
    config = prepared_config.config
    setup_distributed(timeout_seconds=config.distributed.timeout_seconds)
    _resolve_distributed_mode(config, launcher="standard", world_size=world_size)
    experiment_config = prepared_config.experiment_config
    data_config = prepared_config.data_config
    model_config = prepared_config.model_config
    training_config = prepared_config.training_config
    analysis_config = prepared_config.analysis_config
    outputs_config = prepared_config.outputs_config

    _validate_sealed_test_analysis(experiment_config, analysis_config)
    _validate_encoder_request(model_config, training_config, world_size=world_size)
    environment = _setup_training_environment(
        config=config,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        outputs_config=outputs_config,
        experiment_config=experiment_config,
        world_size=world_size,
        local_rank=local_rank,
        is_main=is_main,
    )
    run_save_path = environment.run_save_path
    train_ds = environment.train_ds
    valid_ds = environment.valid_ds
    test_ds = environment.test_ds
    model = environment.model
    unwrapped_model = environment.unwrapped_model
    encoder_network = environment.encoder_network

    validate_only = bool(getattr(args, "validate_only", False))
    encoded_state = _prepare_encoded_training_state(
        config=config,
        unwrapped_model=unwrapped_model,
        encoder_network=encoder_network,
        training_config=training_config,
        model_config=model_config,
        data_config=data_config,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        analysis_config=analysis_config,
        run_save_path=run_save_path,
        world_size=world_size,
        is_main=is_main,
    )
    encoded_train_ds = encoded_state.encoded_train_ds
    encoded_valid_ds = encoded_state.encoded_valid_ds
    analysis_manager = encoded_state.analysis_manager

    if validate_only:
        # Build everything a run needs (analysis manager, calibrated gates,
        # optimizer, trainer) so every configuration check runs; skip training.
        _build_main_trainer(
            unwrapped_model=unwrapped_model,
            training_config=training_config,
            model_config=model_config,
            experiment_config=experiment_config,
            run_save_path=run_save_path,
            encoded_train_ds=encoded_train_ds,
            analysis_manager=analysis_manager,
        )
        if is_main:
            _record_model_size_summary(unwrapped_model, config, run_save_path)
            logger.info(
                "Validation-only preflight completed successfully; training was not run"
            )
        _cleanup_training_runtime(world_size=world_size)
        return

    main_training = _run_main_training_phase(
        unwrapped_model=unwrapped_model,
        model=model,
        training_config=training_config,
        model_config=model_config,
        experiment_config=experiment_config,
        run_save_path=run_save_path,
        encoded_train_ds=encoded_train_ds,
        encoded_valid_ds=encoded_valid_ds,
        analysis_manager=analysis_manager,
    )
    _finalize_main_training_phase(
        is_main=is_main,
        unwrapped_model=unwrapped_model,
        config=config,
        run_save_path=run_save_path,
        analysis_config=analysis_config,
        main_training=main_training,
        analysis_manager=analysis_manager,
        model=model,
        encoded_train_ds=encoded_train_ds,
        encoded_valid_ds=encoded_valid_ds,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        training_config=training_config,
        training_results=main_training.training_results,
    )

    _cleanup_training_runtime(world_size=world_size)

    if is_main:
        logger.info("Training experiment completed successfully")


def _public_training_strategies() -> list[str]:
    """Return the training strategies this installation can run, with plugins."""
    from dendritic_modeling.training.factory import get_available_trainer_strategies

    return sorted(set(get_available_trainer_strategies()))


def _build_cli_parser(description: str) -> argparse.ArgumentParser:
    """Return the command-line parser of the training entry points."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument(
        "--output_dir", help="Override output directory specified in config."
    )
    parser.add_argument("--run_name", help="Override run name specified in config.")
    parser.add_argument(
        "--data_dir", help="Override data.base_dir (dataset root) specified in config."
    )
    parser.add_argument(
        "--learning_strategy",
        choices=_public_training_strategies(),
        help="Override learning strategy specified in config.",
    )
    parser.add_argument(
        "--error_broadcast_mode",
        choices=LOCAL_ERROR_BROADCAST_MODES,
        help="Override the LocalCA error-routing mode specified in config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override experiment seed (for running multiple seeds per config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of main-training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the main trainer's fallback learning rate.",
    )
    parser.add_argument(
        "--topk_lr",
        type=float,
        default=None,
        help="Override the main trainer's sparse-synapse learning rate.",
    )
    parser.add_argument(
        "--blocklinear_lr",
        type=float,
        default=None,
        help="Override the main trainer's dendritic-coupling learning rate.",
    )
    parser.add_argument(
        "--reactivation_lr",
        type=float,
        default=None,
        help="Override the main trainer's reactivation learning rate.",
    )
    parser.add_argument(
        "--decoder_lr",
        type=float,
        default=None,
        help="Override the main trainer's decoder learning rate.",
    )
    parser.add_argument(
        "--train_samples_per_class",
        type=int,
        default=None,
        help="Bound ImageNet training samples per class (for a short check run).",
    )
    parser.add_argument(
        "--val_samples_per_class",
        type=int,
        default=None,
        help="Bound ImageNet evaluation samples per class (for a short check run).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Load and validate the config; build the datasets, model, analysis "
            "manager, calibrated gates, optimizer and trainer; write construction "
            "diagnostics, then exit without training."
        ),
    )
    return parser


def _parse_cli_args():
    return _build_cli_parser(
        "Run dendritic modeling training and evaluation."
    ).parse_args()


def cli_main():
    """Console-script entry point for ``dendritic-train``."""
    setup_logging()
    args = _parse_cli_args()
    main(config_path=args.config, args=args)


if __name__ == "__main__":
    cli_main()
