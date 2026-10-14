#!/usr/bin/env python
"""
Large Model FSDP Training Script
================================

This script enables FSDP (Fully Sharded Data Parallel) training for large dendritic models
that don't fit on a single GPU.

The run is set up exactly as by ``train_experiments``: the same command-line
overrides, configuration checks, run directory, seeds, datasets, model (and
warm start) and run records; only the trainer differs.

Usage:
    torchrun --nproc_per_node=<num_gpus> -m \
        dendritic_modeling.scripts.training.train_experiments_fsdp <config_path>
"""

import gc
import json
import logging
import os
import time
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.config.analysis import fill_analysis_seeds
from dendritic_modeling.config.reactivation import normalize_reactivation_update_mode
from dendritic_modeling.config.regularization import (
    RegularizationConfig,
    SelectiveConfig,
)
from dendritic_modeling.config.training import OptimizerConfig, default_grad_clip_mode
from dendritic_modeling.datasets.dimensions import flat_dataset_input_dim
from dendritic_modeling.networks import (
    DeepstLinear,
    DenseToSparseLinear,
    IndexedDynamicTopKLinear,
    IndexedSparseLinear,
    TopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    collect_model_data_driven_reactivation_policies,
    is_data_driven_reactivation_policy,
    normalize_reactivation_init_policy,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_rewire import (
    IndexedRewireLinear,
)
from dendritic_modeling.networks.checkpoints import atomic_torch_save
from dendritic_modeling.scripts.script_utils import config_utils as _config_utils
from dendritic_modeling.scripts.script_utils.setup_utils import (
    _config_to_plain_dict,
    initialize_model,
    setup_environment,
)

# The run steps shared with single-process and DDP training.
from dendritic_modeling.scripts.training.train_experiments import (
    _broadcast_run_save_path,
    _build_cli_parser,
    _dump_init_gate_stats_diagnostic,
    _encoder_datasets,
    _load_and_prepare_training_config,
    _maybe_calibrate_data_driven_reactivation,
    _record_model_size_summary,
    _resolve_distributed_mode,
    _save_run_config,
    _save_training_summary,
    _validate_encoder_request,
    _validate_sealed_test_analysis,
    launch_ranks,
)
from dendritic_modeling.training.dataloader_utils import (
    DataLoaderTuning,
    data_parallel_layout,
    resolve_dataloader_tuning_from_config,
    seeded_dataloader_kwargs,
    visible_cpu_count,
)
from dendritic_modeling.training.fsdp_utils import (
    frozen_decay_buffers,
    gather_full_state_dict,
    get_fsdp_config,
    mixed_precision_record,
)
from dendritic_modeling.training.loss.functions import (
    get_loss_function,
    validate_task_loss,
)
from dendritic_modeling.training.optimizers import validate_param_group_learning_rates
from dendritic_modeling.training.optimizers.muonh import MUONH_NAMES
from dendritic_modeling.training.regularization import RegularizationManager
from dendritic_modeling.training.strategies.fsdp_standard import (
    FSDPTrainer,
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.utils import resolve_experiment_seeds
from dendritic_modeling.utils.general import save_dict
from dendritic_modeling.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)
# Collective timeout when distributed.timeout_seconds is unset.
FSDP_TIMEOUT_SECONDS = 1800
# Synapse modules whose weights CustomWeightDecayOptimizer maintains.
_MAINTAINED_SYNAPSES = (
    TopKLinear,
    IndexedSparseLinear,
    IndexedDynamicTopKLinear,
    DeepstLinear,
)
# torchinfo import (optional)
try:
    import torchinfo

    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    logger.warning("torchinfo not available. Model summary will be skipped.")


def setup_distributed(timeout_seconds: Optional[int] = None) -> tuple[int, int, int]:
    """Initialize the distributed environment of a ``torchrun`` launch.

    FSDP requires a process group, so a launch without ``torchrun`` raises.
    ``timeout_seconds`` (``distributed.timeout_seconds``) bounds every
    collective; None uses 1800 seconds.

    Returns:
        (rank, local_rank, world_size)
    """
    if "RANK" not in os.environ:
        raise RuntimeError(
            "FSDP requires torchrun.  Launch with:\n"
            "  torchrun --nproc_per_node=<N> train_experiments_fsdp.py <config>"
        )
    rank, local_rank, world_size = launch_ranks()
    timeout = FSDP_TIMEOUT_SECONDS if timeout_seconds is None else int(timeout_seconds)
    if timeout <= 0:
        raise ValueError(
            f"distributed.timeout_seconds must be positive, got {timeout_seconds!r}"
        )
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=timeout))
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def print_model_parameter_summary(model, sample_input=None):
    """Print detailed model parameter summary using torchinfo if available."""
    logger.info("Network architecture summary before training:")
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        is_sequence_input = (
            isinstance(sample_input, torch.Tensor) and sample_input.dim() >= 2
        )
        summary_is_safe = total_params <= 10_000_000 and not is_sequence_input

        if TORCHINFO_AVAILABLE and sample_input is not None and summary_is_safe:
            torchinfo.summary(
                model,
                input_size=(1, *sample_input.shape),
                col_names=["input_size", "output_size", "num_params"],
                col_width=25,
                row_settings=["depth"],
                depth=7,
            )
            logger.info("Model summary generated (see console output)")
        else:
            if not TORCHINFO_AVAILABLE:
                logger.warning(
                    "torchinfo is not available. Model summary not generated."
                )
            elif not summary_is_safe:
                logger.info(
                    "Skipping torchinfo forward for a recurrent or large model; "
                    "reporting parameter counts without an artificial pre-FSDP batch."
                )
            else:
                logger.warning("No sample input provided. Using basic parameter count.")

            # Basic parameter breakdown
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

    except Exception as e:
        logger.error(f"Failed to generate model summary before training: {e}")
        # Fallback to basic count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")


def _print_rank0_model_summary(model, train_ds) -> None:
    """Generate the rank-0 model summary, falling back when sampling fails."""
    try:
        sample_input = train_ds[0][0]
        print_model_parameter_summary(model, sample_input)
    except Exception as e:
        logger.warning(f"Could not generate model summary with sample input: {e}")
        print_model_parameter_summary(model)


def _typed_optimizer_config(optimizer) -> OptimizerConfig:
    """Return ``training.main.optimizer`` as an ``OptimizerConfig``."""
    if isinstance(optimizer, OptimizerConfig):
        return optimizer
    if optimizer is None:
        return OptimizerConfig()
    values = optimizer if isinstance(optimizer, dict) else vars(optimizer)
    allowed = set(OptimizerConfig.__dataclass_fields__)
    return OptimizerConfig(
        **{key: value for key, value in values.items() if key in allowed}
    )


def _typed_regularization_config(regularization) -> RegularizationConfig:
    """Return ``training.main.regularization`` as a ``RegularizationConfig``."""
    if regularization is None:
        return RegularizationConfig()
    if not isinstance(regularization, dict):
        return regularization
    values = dict(regularization)
    if isinstance(values.get("selective"), dict):
        values["selective"] = SelectiveConfig(**values["selective"])
    return RegularizationConfig(**values)


def _unapplied_recurrent_overrides(main) -> list[str]:
    """strategies.recurrent settings that differ from what FSDP applies.

    FSDP trains with ``common.use_amp``, clips at ``common.grad_clip_value``
    and has no LR schedule; an unset (null) override inherits those.
    """
    recurrent = getattr(getattr(main, "strategies", None), "recurrent", None)
    if recurrent is None:
        return []
    common = main.common
    section = "training.main.strategies.recurrent"
    problems = []
    if recurrent.lr_schedule is not None and str(recurrent.lr_schedule) != "none":
        problems.append(f"{section}.lr_schedule (no LR scheduler)")
    if recurrent.lr_warmup_epochs:
        problems.append(f"{section}.lr_warmup_epochs (no LR scheduler)")
    if recurrent.use_amp is not None and bool(recurrent.use_amp) != bool(
        common.use_amp
    ):
        problems.append(f"{section}.use_amp (FSDP applies common.use_amp)")
    if recurrent.grad_clip_value is not None and recurrent.grad_clip_value != (
        common.grad_clip_value
    ):
        problems.append(
            f"{section}.grad_clip_value (FSDP clips at common.grad_clip_value)"
        )
    return problems


def _validate_fsdp_training_options(config, model) -> None:
    """Reject configured options that the FSDP trainer would not apply.

    The FSDP loop trains with backpropagation and a plain optimizer step: no
    LR schedule, auxiliary regularization, reactivation refresh, pruning,
    sparse maintenance, rewiring or topology controllers. FSDP cannot shard
    scalar parameters, MuonH needs unsharded 2-D parameters, and
    per-component learning rates need FSDP to keep the original parameters
    (``use_orig_params``).
    """
    main = config.training.main
    common = main.common
    problems = []
    strategy = str(getattr(main, "strategy", "standard") or "standard").lower()
    if strategy == "standard":
        validate_task_loss(
            str(getattr(getattr(config, "model", None), "task", "classification")),
            get_loss_function(str(getattr(common, "loss_function", "ce"))),
            owner="FSDPTrainer",
        )
    if strategy not in {"standard", "recurrent"}:
        problems.append(f"training.main.strategy={strategy!r} (FSDP uses backprop)")
    if strategy == "recurrent":
        problems.extend(_unapplied_recurrent_overrides(main))
    schedule = str(getattr(common, "lr_schedule", "none") or "none").lower()
    if schedule != "none" or (getattr(common, "lr_warmup_epochs", 0) or 0) > 0:
        problems.append("common.lr_schedule/lr_warmup_epochs (no LR scheduler)")
    regularization = _typed_regularization_config(getattr(main, "regularization", None))
    if RegularizationManager(regularization).has_regularization():
        problems.append("training.main.regularization (no auxiliary losses)")
    pruning = getattr(main, "pruning", None)
    pruning_enabled = (
        pruning.get("enabled", False)
        if isinstance(pruning, dict)
        else getattr(pruning, "enabled", False)
    )
    if bool(pruning_enabled):
        problems.append("training.main.pruning (no pruning step)")
    mode = normalize_reactivation_update_mode(
        getattr(common, "reactivation_update_mode", None)
    )
    if mode != "backprop":
        problems.append(f"common.reactivation_update_mode={mode!r}")
    if int(getattr(common, "recalibrate_reactivation_every", 0) or 0) > 0:
        problems.append("common.recalibrate_reactivation_every (no recalibration)")
    if (common.best_state_device, common.save_best_checkpoint) != (
        "model",
        "every_improvement",
    ):
        problems.append(
            "common.best_state_device/save_best_checkpoint (FSDP keeps its final "
            "weights and tracks no best state)"
        )
    if not bool(getattr(common, "shuffle", True)):
        problems.append("common.shuffle=false (the FSDP sampler always shuffles)")
    if getattr(common, "impact_rewiring", None):
        problems.append("common.impact_rewiring (no topology controllers)")
    maintained = sorted(
        {
            type(m).__name__
            for m in model.modules()
            if isinstance(m, _MAINTAINED_SYNAPSES)
        }
    )
    if maintained and (
        float(getattr(common, "synapse_decay_rate", 0.0) or 0.0) > 0
        or bool(getattr(common, "weight_boosting", False))
    ):
        problems.append(
            "common.synapse_decay_rate/weight_boosting (the sparse-synapse "
            "maintenance decay, default 0.01, not optimizer weight decay): FSDP "
            f"runs no sparse maintenance for {maintained}; set "
            "synapse_decay_rate: 0 and weight_boosting: false"
        )
    rewiring = sorted(
        {
            type(m).__name__
            for m in model.modules()
            if isinstance(m, (IndexedRewireLinear, DeepstLinear))
            or (
                isinstance(m, DenseToSparseLinear)
                and not bool(getattr(m, "advance_on_forward", True))
            )
        }
    )
    if rewiring:
        problems.append(f"rewiring synapses {rewiring} (no apply_rewiring step)")
    for name in sorted(FSDPTrainer.unapplied_options):
        if bool(getattr(common, name, False)):
            problems.append(f"training.main.common.{name}")
    fsdp = getattr(getattr(config, "distributed", None), "fsdp", None)
    sharding = str(getattr(fsdp, "sharding_strategy", "FULL_SHARD")).upper()
    use_orig_params = bool(getattr(fsdp, "use_orig_params", True))
    optimizer = _typed_optimizer_config(getattr(main, "optimizer", None))
    if str(optimizer.name).lower() in MUONH_NAMES and (
        sharding != "NO_SHARD" or not use_orig_params
    ):
        problems.append(
            f"optimizer {optimizer.name!r} with sharding_strategy={sharding} and "
            f"use_orig_params={use_orig_params} (MuonH needs full 2-D matrices, "
            "which FSDP shards into 1-D views; use NO_SHARD with use_orig_params: "
            "true)"
        )
    scalars = [name for name, p in model.named_parameters() if p.ndim == 0]
    if scalars:
        problems.append(
            f"scalar parameters {scalars} (FSDP shards only parameters with at "
            "least one dimension; give each of them a shape of at least (1,), "
            "as the global output scale of model.output_scale_mode: global has)"
        )
    if bool(getattr(fsdp, "mixed_precision", True)):
        knob = getattr(fsdp, "mixed_precision_dtype", None) or "bf16"
        param_dtype = getattr(fsdp, "param_dtype", None) or knob
        reduce_dtype = getattr(fsdp, "reduce_dtype", None) or param_dtype
        half = {"fp16", "float16", "half"}
        if (
            str(param_dtype).lower() in half or str(reduce_dtype).lower() in half
        ) and not bool(getattr(common, "use_amp", False)):
            problems.append(
                f"fp16 FSDP mixed precision (param_dtype={param_dtype}, "
                f"reduce_dtype={reduce_dtype}) without loss scaling: fp16 "
                "gradients underflow; set training.main.common.use_amp: true "
                "for the sharded gradient scaler, or use bf16"
            )
    param_groups = _config_utils.normalize_param_groups_config(common.param_groups)
    if not use_orig_params and param_groups.split_params:
        problems.append(
            "distributed.fsdp.use_orig_params: false with param_groups."
            "split_params (the groups hold module parameters, which FSDP "
            "replaces with flat parameters)"
        )
    if problems:
        raise ValueError(
            "The FSDP trainer does not apply: "
            + "; ".join(problems)
            + ". Remove these options or train without FSDP (train_experiments.py)."
        )
    validate_param_group_learning_rates(optimizer, param_groups)


def _record_fsdp_final_weights(results: dict, train_common, *, rank: int) -> None:
    """Record that FSDP keeps the final weights, even with load_best_state_dict.

    The FSDP trainer does not reload a best-validation state (its optional
    ``best_model.pt`` is written only with early stopping and checkpointing,
    on rank 0), so the final weights are the last completed epoch's.
    """
    requested = bool(getattr(train_common, "load_best_state_dict", True))
    results["final_weights"] = "last_completed_epoch"
    results["load_best_state_dict_requested"] = requested
    results["load_best_state_dict_applied"] = False
    if requested and rank == 0:
        logger.warning(
            "training.main.common.load_best_state_dict is true, but the FSDP "
            "trainer keeps the final weights (last completed epoch) and does not "
            "reload the best-validation state; recorded in training_summary.json."
        )


def _build_fsdp_runtime_configs(config):
    """Build trainer-common, FSDP, and optimizer configs from typed config."""
    train_common = config.training.main.common

    fsdp_cfg = config.distributed.fsdp
    logger.info("Using FSDP configuration from config file")
    fsdp_config = get_fsdp_config(
        mixed_precision=fsdp_cfg.mixed_precision,
        cpu_offload=fsdp_cfg.cpu_offload,
        sharding_strategy=fsdp_cfg.sharding_strategy,
        fsdp_config_dict=vars(fsdp_cfg) if hasattr(fsdp_cfg, "__dict__") else None,
    )

    optimizer_config = {
        # training.main.optimizer (name, betas, eps, weight_decay, ...).
        # synapse_decay_rate is sparse maintenance, which FSDP does not apply;
        # it is never turned into optimizer L2 decay.
        "optimizer": _typed_optimizer_config(
            getattr(getattr(config.training, "main", None), "optimizer", None)
        ),
        # The groups and learning rates of single-process training.
        "param_groups": _config_utils.normalize_param_groups_config(
            train_common.param_groups
        ),
    }
    return train_common, fsdp_config, optimizer_config


def _create_fsdp_trainer(
    *,
    rank: int,
    world_size: int,
    fsdp_config,
    run_save_path,
    train_common,
    optimizer_config,
    analysis_manager,
    config,
):
    """Create the FSDP trainer from resolved runtime config."""
    checkpointing = _config_to_plain_dict(
        getattr(config.experiment, "checkpointing", None)
    )
    checkpointing_enabled = bool(checkpointing.get("enabled", False))
    save_every_n_epochs = (
        int(checkpointing.get("save_every_n_epochs", 10))
        if checkpointing_enabled
        else None
    )
    return FSDPTrainer(
        rank=rank,
        world_size=world_size,
        fsdp_config=fsdp_config,
        checkpoint_dir=run_save_path if rank == 0 else None,
        save_every_n_epochs=save_every_n_epochs,
        optimizer_config=optimizer_config,
        analysis_manager=analysis_manager,
        epochs=train_common.epochs,
        loss_function=train_common.loss_function,
        task=config.model.task,
        grad_clip_value=train_common.grad_clip_value,
        use_amp=train_common.use_amp,
        early_stopping=train_common.early_stopping,
        patience=train_common.patience,
        print_every=train_common.print_every,
        log_first_step_diagnostics=train_common.log_first_step_diagnostics,
        cache_frozen_core_outputs=getattr(
            config.distributed.fsdp,
            "cache_frozen_core_outputs",
            False,
        ),
        cached_core_batch_size=getattr(
            config.distributed.fsdp,
            "cached_core_batch_size",
            None,
        ),
        checkpointing=checkpointing_enabled,
        plot_losses=train_common.plot_losses,
        save_path=run_save_path if rank == 0 else None,
        seed=int(config.experiment.seed),
    )


def _resolve_large_model_input_dim(data_config, model_config=None) -> int:
    """Resolve input size for manually assembled FSDP models.

    Sequence datasets have configuration-dependent feature widths and are not
    covered by the fixed vision-dataset table.  Match the standard model
    builder by honoring an explicit encoder ``params.input_dim`` first.
    """
    encoder_config = getattr(model_config, "encoder", None)
    encoder_params = getattr(encoder_config, "params", None)
    if encoder_params:
        explicit_input_dim = getattr(encoder_params, "input_dim", None)
        if explicit_input_dim is None:
            try:
                explicit_input_dim = dict(encoder_params).get("input_dim")
            except (TypeError, ValueError):
                explicit_input_dim = None
        if explicit_input_dim is not None:
            explicit_input_dim = int(explicit_input_dim)
            if explicit_input_dim <= 0:
                raise ValueError(
                    "model.encoder.params.input_dim must be positive, got "
                    f"{explicit_input_dim}"
                )
            return explicit_input_dim

    return flat_dataset_input_dim(
        getattr(data_config, "dataset_name", None),
        default_input_dim=784,
    )


def create_large_model(config):
    """Build the run's model as single-process training does (``initialize_model``).

    ``model.encoder.params.input_dim`` is the input width; when it is unset
    (training writes it from a sample), it comes from the dataset's flat size.
    """
    params = config.model.encoder.params
    if params.input_dim is None:
        params.input_dim = _resolve_large_model_input_dim(config.data, config.model)
    model, _encoder_network = initialize_model(config.model)
    return model


def _apply_fsdp_trainability_policy(model, fsdp_config) -> int:
    """Apply opt-in FSDP component freezing before wrapping the model."""
    freeze_core = bool(getattr(fsdp_config, "freeze_core", False))
    cache_core = bool(getattr(fsdp_config, "cache_frozen_core_outputs", False))
    if cache_core and not freeze_core:
        raise ValueError(
            "distributed.fsdp.cache_frozen_core_outputs requires freeze_core=true"
        )
    if not freeze_core:
        return 0
    core = getattr(model, "core_network", None)
    if core is None:
        raise ValueError("distributed.fsdp.freeze_core requires model.core_network")
    encoder = getattr(model, "encoder_network", None)
    if (
        cache_core
        and encoder is not None
        and any(parameter.requires_grad for parameter in encoder.parameters())
    ):
        raise ValueError(
            "cache_frozen_core_outputs requires a frozen or parameter-free encoder"
        )
    model._freeze_core_forward = True
    for module in core.modules():
        module._fsdp_frozen_core = True
    frozen_parameters = 0
    for parameter in core.parameters():
        if parameter.requires_grad:
            frozen_parameters += parameter.numel()
            parameter.requires_grad_(False)
    logger.info(
        "FSDP readout-only mode: froze %s core parameters before wrapping",
        f"{frozen_parameters:,}",
    )
    return frozen_parameters


def _requested_data_driven_reactivation_policies(config, model) -> set[str]:
    """Return data-driven reactivation policies requested by config or layers."""
    requested_policies = set()
    try:
        reactivation_cfg = config.model.core.reactivation
        init_policy = normalize_reactivation_init_policy(
            getattr(reactivation_cfg, "init_policy", "analytical")
        )
        if is_data_driven_reactivation_policy(init_policy):
            requested_policies.add(init_policy)
    except AttributeError:
        logger.debug(
            "Reactivation config path missing (config.model.core.reactivation); "
            "skipping config-level data-driven init policy detection."
        )

    layer_requested_policies, _saw_layer_policy = (
        collect_model_data_driven_reactivation_policies(model)
    )
    requested_policies.update(layer_requested_policies)

    return requested_policies


def _calibrate_fsdp_reactivation(config, model, train_dataset, *, rank: int) -> None:
    """Run data-driven reactivation calibration before FSDP wrapping.

    Calibration runs on rank 0 and reaches the other ranks through
    ``distributed.fsdp.sync_module_states``, which therefore must be true.
    """
    requested_policies = _requested_data_driven_reactivation_policies(config, model)
    if not requested_policies:
        return
    if not bool(getattr(config.distributed.fsdp, "sync_module_states", True)):
        raise ValueError(
            "reactivation init_policy "
            f"{', '.join(sorted(requested_policies))} is calibrated on rank 0 and "
            "reaches the other ranks only through "
            "distributed.fsdp.sync_module_states; set it to true"
        )
    if rank != 0:
        return
    _maybe_calibrate_data_driven_reactivation(
        config=config,
        model=model,
        train_dataset=train_dataset,
        is_main=True,
    )


def _setup_rank0_analysis_manager(
    *,
    config,
    model,
    encoded_train_ds,
    encoded_valid_ds,
    encoded_test_ds,
    run_save_path,
    local_rank: int,
):
    """Create the rank-0 analysis manager."""
    logger.info("Setting up AnalysisManager...")
    experiment_config = getattr(config, "experiment", None)
    encoded_data = {
        "train": encoded_train_ds,
        "valid": encoded_valid_ds,
        "test": encoded_test_ds,
    }
    manager_kwargs = {
        "model": model,
        "data": encoded_data,
        "analysis_config": config.analysis,
        "save_root": run_save_path,
        "device": f"cuda:{local_rank}",
    }
    if experiment_config is not None:
        analysis_seeds = resolve_experiment_seeds(experiment_config)
        manager_kwargs.update(
            evaluation_seed=analysis_seeds.evaluation_seed,
            probe_seed=analysis_seeds.probe_seed,
        )
    analysis_manager = AnalysisManager(**manager_kwargs)
    logger.info(f"AnalysisManager created: {type(analysis_manager)}")
    logger.info(f"Performance analysis enabled: {config.analysis.performance.enabled}")
    logger.info(
        f"Performance analysis training: {config.analysis.performance.training}"
    )
    return analysis_manager


def _run_final_analysis_on_plain_model(
    *,
    config,
    analysis_manager: AnalysisManager,
    state_dict: dict,
    local_rank: int,
) -> None:
    """Run the final analysis on an unsharded copy of the trained model."""
    logger.info("Running final analysis on non-FSDP model...")
    plain_model, _encoder_network = initialize_model(config.model)
    plain_model.load_state_dict(state_dict)
    plain_model = plain_model.to(f"cuda:{local_rank}")
    plain_model.eval()
    analysis_manager.model = plain_model
    analysis_manager.run_analysis(filename="final", training=False)
    logger.info("Final analysis completed")


def _save_final_model_and_run_analysis(
    *,
    rank: int,
    trainer,
    run_save_path,
    config,
    analysis_manager,
    local_rank: int,
) -> str:
    """Save the trained weights as ``final_model.pt`` and run the final analysis.

    Every rank takes part in gathering the full model state. Rank 0 saves it
    as a plain state_dict, as single-process training does, and runs the
    requested analysis on an unsharded copy of the model after the FSDP model
    and its optimizer are released.
    """
    state_dict = gather_full_state_dict(trainer.fsdp_model)
    trainer.release_model()
    if analysis_manager is not None:
        analysis_manager.model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if rank != 0:
        return ""
    final_model_path = os.path.join(run_save_path, "final_model.pt")
    atomic_torch_save(state_dict, final_model_path)
    logger.info("Saved final model weights to %s", final_model_path)
    if analysis_manager is not None:
        _run_final_analysis_on_plain_model(
            config=config,
            analysis_manager=analysis_manager,
            state_dict=state_dict,
            local_rank=local_rank,
        )
    return final_model_path


def _record_fsdp_peak_memory(results: dict, run_save_path: str) -> None:
    """Record the training peak memory (maximum over ranks) in model_resources.json.

    The resources are counted before wrapping, when every parameter is whole.
    """
    path = os.path.join(run_save_path, "model_resources.json")
    if "peak_cuda_allocated_bytes" not in results or not os.path.exists(path):
        return
    with open(path) as handle:
        resources = json.load(handle)
    resources["cuda_peak_memory_allocated_bytes"] = results["peak_cuda_allocated_bytes"]
    resources["cuda_peak_memory_reserved_bytes"] = results["peak_cuda_reserved_bytes"]
    resources["cuda_peak_memory_scope"] = "maximum over FSDP ranks"
    save_dict(resources, run_save_path, "model_resources.json")


class _FSDPLoaderTuningDataset(Dataset):
    """Placeholder used only for backwards-compatible helper calls."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        del index
        return torch.empty(0), torch.empty(0, dtype=torch.long)


def _dataloader_kwargs_from_common(
    common_config,
    dataset: Dataset | None = None,
    *,
    device: torch.device | str | None = None,
) -> dict:
    """Build validated DataLoader kwargs from training.main.common."""
    return _dataloader_tuning_from_common(
        common_config,
        dataset,
        device=device,
    ).as_kwargs()


def _dataloader_tuning_from_common(
    common_config,
    dataset: Dataset | None = None,
    *,
    device: torch.device | str | None = None,
) -> DataLoaderTuning:
    """Resolve DataLoader tuning from training.main.common."""
    default_num_workers = None
    if dataset is None and getattr(common_config, "num_workers", None) is None:
        # Preserve the historical standalone helper default.  Real call sites
        # pass a dataset so shared tuning can choose tensor-vs-file loading.
        default_num_workers = min(4, visible_cpu_count())

    return resolve_dataloader_tuning_from_config(
        dataset if dataset is not None else _FSDPLoaderTuningDataset(),
        common_config,
        device=device,
        default_num_workers=default_num_workers,
    )


def setup_data_loaders(config, train_ds, valid_ds, *, rank: int, world_size: int):
    """Return the distributed training and validation loaders of the run's datasets."""
    resolved_seeds = resolve_experiment_seeds(config.experiment)
    # Create distributed sampler for training only.
    # Validation uses a non-padded sampler: each rank gets a disjoint
    # shard *without* padding, and the trainer reduces exact counts
    # (loss_sum, correct, total_examples) via all-reduce SUM.
    train_sampler = DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=resolved_seeds.loader_seed,
    )

    # PaddedDistributedEvalSampler gives each rank an equal number of
    # indices (required by FSDP all-gather) but marks the extras as
    # padding (-1 sentinels).  The trainer's _epoch_valid_fsdp excludes
    # padding from the reduction, so validation is numerically identical
    # to a non-distributed run over the full dataset.
    valid_sampler = PaddedDistributedEvalSampler(
        valid_ds,
        num_replicas=world_size,
        rank=rank,
    )

    # Create data loaders
    batch_size = config.training.main.common.batch_size
    loader_tuning = _dataloader_tuning_from_common(
        config.training.main.common,
        train_ds,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    if rank == 0:
        logger.info("FSDP DataLoader tuning: %s", loader_tuning.summary(train_ds))
    loader_kwargs = loader_tuning.as_kwargs()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        **seeded_dataloader_kwargs(
            resolved_seeds.loader_seed,
            stream=0,
            rank=rank,
        ),
        **loader_kwargs,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        sampler=valid_sampler,
        **seeded_dataloader_kwargs(
            resolved_seeds.loader_seed,
            stream=1,
            rank=rank,
        ),
        **loader_kwargs,
    )

    return train_loader, valid_loader


def main(config_path, args=None):
    """Main FSDP training function."""
    script_started = time.perf_counter()
    if "RANK" not in os.environ:
        setup_distributed()  # raises: FSDP needs torchrun
    rank, local_rank, world_size = launch_ranks()
    is_main = rank == 0
    if is_main:
        logger.info(f"Starting FSDP training on {world_size} GPUs")

    # Command-line overrides, validation and stability checks of
    # train_experiments.
    config = _load_and_prepare_training_config(
        config_path, args, is_main=is_main, world_size=world_size
    ).config
    setup_distributed(timeout_seconds=config.distributed.timeout_seconds)
    _resolve_distributed_mode(config, launcher="fsdp", world_size=world_size)
    _validate_sealed_test_analysis(config.experiment, config.analysis)
    _validate_encoder_request(
        config.model, config.training, world_size=world_size, fsdp=True
    )

    # Run directory, seeds, datasets, model and warm start of single-process
    # training; the model stays on the CPU until FSDP wraps it.
    setup_started = time.perf_counter()
    run_save_path, train_ds, valid_ds, test_ds, model, encoder_network = (
        setup_environment(
            model_config=config.model,
            training_config=config.training,
            data_config=config.data,
            outputs_config=config.outputs,
            experiment_config=config.experiment,
            is_main=is_main,
        )
    )
    run_save_path = _broadcast_run_save_path(
        run_save_path, world_size=world_size, is_main=is_main
    )
    # Rejects autoencoder encoders and encoder pretraining (built-in encoder
    # types were already rejected before the run directory was created).
    _encoder_datasets(
        encoder_network=encoder_network,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        pretraining=config.training.encoder_pretraining,
        fsdp=True,
    )
    if config.training.main.common.grad_clip_mode is None:
        config.training.main.common.grad_clip_mode = default_grad_clip_mode(
            config.training.main.strategy, "fsdp"
        )
    resolved_seeds = resolve_experiment_seeds(config.experiment)
    # Fill unset analysis seeds before config.json records the run.
    fill_analysis_seeds(
        config.analysis,
        evaluation_seed=resolved_seeds.evaluation_seed,
        probe_seed=resolved_seeds.probe_seed,
    )
    _apply_fsdp_trainability_policy(model, config.distributed.fsdp)
    _validate_fsdp_training_options(config, model)
    train_loader, valid_loader = setup_data_loaders(
        config, train_ds, valid_ds, rank=rank, world_size=world_size
    )
    if is_main:
        _dump_init_gate_stats_diagnostic(model, run_save_path)
        _print_rank0_model_summary(model, train_ds)
        _save_run_config(config, run_save_path)
        # Counted before wrapping, while every parameter is whole.
        _record_model_size_summary(model, config, run_save_path)
    _calibrate_fsdp_reactivation(config, model, train_ds, rank=rank)
    setup_seconds = time.perf_counter() - setup_started

    train_common, fsdp_config, optimizer_config = _build_fsdp_runtime_configs(config)
    precision = mixed_precision_record(fsdp_config)
    frozen_decays = (
        frozen_decay_buffers(model, fsdp_config["mixed_precision"].param_dtype)
        if precision is not None
        else []
    )
    if frozen_decays and is_main:
        logger.warning(
            "FSDP mixed precision computes recurrent decays in %s, which rounds "
            "these to exactly 1, so their levels stop responding to input: %s. "
            "Use distributed.fsdp.mixed_precision_dtype: fp32 for these time "
            "constants.",
            precision["param_dtype"],
            "; ".join(frozen_decays),
        )

    analysis_manager = None
    if is_main and config.analysis.any_enabled():
        # The encoder trains jointly with the core, so analyses read the
        # datasets as they are.
        analysis_manager = _setup_rank0_analysis_manager(
            config=config,
            model=model,
            encoded_train_ds=train_ds,
            encoded_valid_ds=valid_ds,
            encoded_test_ds=test_ds,
            run_save_path=run_save_path,
            local_rank=local_rank,
        )
    elif is_main:
        logger.info("No final analysis modules enabled; skipping AnalysisManager.")

    trainer = _create_fsdp_trainer(
        rank=rank,
        world_size=world_size,
        fsdp_config=fsdp_config,
        run_save_path=run_save_path,
        train_common=train_common,
        optimizer_config=optimizer_config,
        analysis_manager=analysis_manager,
        config=config,
    )

    if bool(getattr(args, "validate_only", False)):
        # Wrap the model and build the optimizer so every check runs.
        trainer.prepare_model(model)
        if is_main:
            logger.info(
                "Validation-only preflight completed successfully; training was not run"
            )
        dist.destroy_process_group()
        return

    if is_main:
        logger.info("Starting FSDP training...")
    results = trainer.train(model, train_loader, valid_loader)
    del model
    results.update(
        {
            "setup_seconds": setup_seconds,
            "runtime_through_training_seconds": time.perf_counter() - script_started,
            "world_size": world_size,
            "data_parallel": data_parallel_layout(train_loader, world_size),
            "fsdp_mixed_precision": precision,
            "frozen_decay_buffers": frozen_decays,
        }
    )
    _record_fsdp_final_weights(results, train_common, rank=rank)
    if is_main:
        logger.info("Training completed successfully!")
        _save_training_summary(results, run_save_path)
        _record_fsdp_peak_memory(results, run_save_path)

    # FSDP shards the parameters, so the weights are gathered once into a
    # plain state_dict (final_model.pt) and analysed on an unsharded copy.
    _save_final_model_and_run_analysis(
        rank=rank,
        trainer=trainer,
        run_save_path=run_save_path,
        config=config,
        analysis_manager=analysis_manager,
        local_rank=local_rank,
    )

    dist.destroy_process_group()
    if is_main:
        logger.info("FSDP training script completed successfully")


def cli_main() -> None:
    """Command-line entry point; takes the options of ``train_experiments``."""
    args = _build_cli_parser("FSDP training for large dendritic models").parse_args()
    main(args.config, args)


if __name__ == "__main__":
    setup_logging()
    cli_main()
