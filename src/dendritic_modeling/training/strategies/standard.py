"""
Base training strategy classes.

This module contains the base trainer class that provides common functionality
for all training strategies in the dendritic modeling framework.
"""

import logging
import math
import os
import time
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import Optional

import torch
import torch.distributed as dist
import torch.utils.data
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler

from dendritic_modeling.config.choices import normalize_choice
from dendritic_modeling.config.keys import UnknownConfigKeyError
from dendritic_modeling.config.reactivation import (
    normalize_reactivation_recalibration_mode,
    normalize_reactivation_update_mode,
    validate_reactivation_updates,
)
from dendritic_modeling.config.training import (
    BEST_CHECKPOINT_SAVES,
    BEST_STATE_DEVICES,
    GRAD_CLIP_MODES,
    CommonTrainerConfig,
    PruningConfig,
    RegularizationConfig,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer, ParametricActivation
from dendritic_modeling.networks.checkpoints import (
    atomic_torch_save,
    copy_state_metadata,
)
from dendritic_modeling.plotting import plot_loss_curves
from dendritic_modeling.training.clipping import (
    clip_gradients,
    zero_nonfinite_gradients,
)
from dendritic_modeling.training.dataloader_utils import (
    data_parallel_layout,
    resolve_dataloader_tuning_from_config,
    seeded_dataloader_kwargs,
)
from dendritic_modeling.training.distributed_samplers import (
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.training.loss.functions import (
    LossFunction,
    get_loss_function,
    validate_task_loss,
)
from dendritic_modeling.training.reactivation_sync import broadcast_reactivation_state
from dendritic_modeling.training.regularization import (
    PruningManager,
    RegularizationManager,
)
from dendritic_modeling.utils.hooks import iter_modules_of_type

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ValidationLossStats:
    """Explicit validation aggregates for the non-distributed path.

    ``_epoch_valid`` historically returns the first two fields as a tuple.
    Keeping both batch and sample aggregates here lets checkpoint selection use
    an exact sample-weighted mean without silently changing that tuple's
    second value from a batch count to a sample count.
    """

    batch_loss_sum: float
    batch_count: int
    sample_weighted_loss_sum: float
    sample_count: int

    # Without validation samples the loss is undefined, so the means are NaN
    # rather than a misleading zero.
    @property
    def batch_mean(self) -> float:
        if self.batch_count == 0:
            return math.nan
        return self.batch_loss_sum / self.batch_count

    @property
    def sample_mean(self) -> float:
        if self.sample_count == 0:
            return math.nan
        return self.sample_weighted_loss_sum / self.sample_count


class _ValidationSums:
    """Per-batch validation losses summed on the device in float64.

    The batch and sample-weighted sums are read once at the end; float64
    additions in batch order (with a separate multiply by the batch size)
    reproduce the host sums of per-batch ``loss.item()`` values.
    """

    def __init__(self, device) -> None:
        self.batch_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.weighted_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
        self.batch_count = 0
        self.sample_count = 0

    def add(self, loss: torch.Tensor, batch_size: int) -> None:
        value = loss.detach().double()
        self.batch_loss_sum += value
        self.weighted_loss_sum += value * batch_size
        self.batch_count += 1
        self.sample_count += int(batch_size)

    def stats(self) -> _ValidationLossStats:
        batch_loss_sum, weighted_loss_sum = torch.stack(
            [self.batch_loss_sum, self.weighted_loss_sum]
        ).tolist()
        return _ValidationLossStats(
            batch_loss_sum=batch_loss_sum,
            batch_count=self.batch_count,
            sample_weighted_loss_sum=weighted_loss_sum,
            sample_count=self.sample_count,
        )

    def distributed_sample_mean(self) -> float:
        """Sample-weighted mean over all ranks (NaN without samples)."""
        stats = torch.stack(
            [
                self.weighted_loss_sum,
                torch.full(
                    (),
                    float(self.sample_count),
                    dtype=torch.float64,
                    device=self.weighted_loss_sum.device,
                ),
            ]
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        weighted_loss_sum, sample_count = stats.tolist()
        if sample_count == 0:
            return math.nan
        return weighted_loss_sum / sample_count


LR_SCHEDULES = ("none", "cosine")

# DendriticBranchLayer attributes set from model.core.{sparsity,reactivation,
# blocklinear}.gradient_scaling. Their backward hooks act only on gradients
# produced by autograd.
_GRADIENT_SCALING_ATTRIBUTES = (
    "topk_strategy",
    "reactivation_strategy",
    "blocklinear_strategy",
)


def configured_gradient_scaling(model) -> list[str]:
    """Return ``name=value`` for every gradient-scaling strategy in ``model``."""
    model = model.module if hasattr(model, "module") else model
    return sorted(
        {
            f"{name}={value}"
            for module in iter_modules_of_type(model, DendriticBranchLayer)
            for name in _GRADIENT_SCALING_ATTRIBUTES
            for value in [str(getattr(module, name, "none") or "none")]
            if value != "none"
        }
    )


# Common options whose effect lives in Trainer._epoch_train, plus
# log_first_step_diagnostics, which only the FSDP trainer reads. A trainer whose
# loop does not apply some of them lists those in ``unapplied_options``.
STEP_OPTIONS = (
    "use_amp",
    "enable_nan_checking",
    "profile_dataloader",
    "log_first_step_diagnostics",
)


# Options every trainer accepts as keywords: the training.<phase>.common
# settings (train_experiments consumes some of them before building the
# trainer) and the values the training scripts add.
COMMON_OPTIONS = frozenset(item.name for item in fields(CommonTrainerConfig))
PIPELINE_OPTIONS = frozenset(
    {
        "seed",
        "loader_seed",
        "task",
        "regularization_config",
        "pruning_config",
        "param_groups_config",
        "optimizer_config",
        "save_every_n_epochs",
        # Device to train on; by default CUDA when available, else CPU.
        "device",
    }
)
# Options that __init__ normalizes; the raw keyword must not overwrite them.
_NORMALIZED_OPTIONS = frozenset(
    {
        "seed",
        "loader_seed",
        "regularization_config",
        "pruning_config",
        "grad_clip_mode",
        "reactivation_update_mode",
        "reactivation_recalibration_mode",
        "best_state_device",
        "save_best_checkpoint",
    }
)


def reject_unknown_options(trainer_cls: type, options) -> None:
    """Fail on keyword options that ``trainer_cls`` does not read."""
    accepted = (
        COMMON_OPTIONS
        | PIPELINE_OPTIONS
        | getattr(trainer_cls, "strategy_options", frozenset())
    )
    unknown = sorted(set(options) - accepted)
    if unknown:
        raise UnknownConfigKeyError(f"{trainer_cls.__name__}", unknown, accepted)


def reject_unapplied_options(trainer_cls: type, options) -> None:
    """Fail when an option is enabled that ``trainer_cls`` does not apply."""
    unapplied = getattr(trainer_cls, "unapplied_options", frozenset())
    requested = [
        name for name in STEP_OPTIONS if name in unapplied and options.get(name)
    ]
    if requested:
        raise ValueError(
            f"{trainer_cls.__name__} does not apply "
            + ", ".join(f"{name}=true" for name in requested)
            + ": its training loop does not implement them. Set them to false "
            "(training.main.common) for this strategy."
        )


def validate_lr_schedule(
    lr_schedule, lr_warmup_epochs, *, supported: bool, owner: str
) -> tuple[str, float]:
    """Validate ``lr_schedule``/``lr_warmup_epochs`` for one training loop.

    Returns the normalized ``(schedule, warmup)`` pair. Raises ``ValueError``
    for unknown schedules, for a warmup without the cosine schedule (the
    warmup is part of that schedule), and for strategies whose training loops
    do not apply the per-epoch schedule (``supports_lr_schedule = False``).
    """
    schedule = str(lr_schedule or "none").strip().lower()
    warmup = lr_warmup_epochs or 0
    if schedule not in LR_SCHEDULES:
        raise ValueError(
            f"Unknown lr_schedule: {schedule!r} (expected one of {LR_SCHEDULES})"
        )
    if warmup < 0:
        raise ValueError(f"lr_warmup_epochs must be non-negative, got {warmup}")
    if schedule == "none" and warmup > 0:
        raise ValueError(
            f"lr_warmup_epochs={warmup} requires lr_schedule='cosine'; the "
            "warmup is part of the cosine schedule and is not applied with "
            "lr_schedule='none'."
        )
    if schedule != "none" and not supported:
        raise ValueError(
            f"{owner} does not apply "
            f"lr_schedule={schedule!r}: its training loop runs separate phases "
            "or its own optimizer steps. Set training.main.common.lr_schedule "
            "to 'none', or use a strategy that applies the per-epoch schedule "
            "(standard, recurrent, fa, dfa, soma_dfa)."
        )
    return schedule, warmup


def _step_seconds(step_start: float, device_tensor: torch.Tensor) -> float:
    """Seconds since ``step_start``, including the step's queued GPU work.

    Training loops read their losses once per epoch, so profiled step times
    wait for the device explicitly.
    """
    if device_tensor.is_cuda:
        torch.cuda.synchronize(device_tensor.device)
    return time.perf_counter() - step_start


class Trainer:

    # Clipping operator applied when training.<phase>.common.grad_clip_mode is
    # unset (see config.training.default_grad_clip_mode).
    default_grad_clip_mode = "value"
    # Whether ``train()`` of this strategy applies ``lr_schedule`` /
    # ``lr_warmup_epochs`` through ``_build_scheduler`` and ``_run_epoch``.
    # Strategies with their own phase loops set this to False so a configured
    # schedule fails explicitly instead of being skipped or restarted.
    supports_lr_schedule = True
    # Whether the trainer applies and checks its own learning-rate schedule
    # (presentation_stream schedules per step) instead of the per-epoch one.
    applies_own_lr_schedule = False
    # STEP_OPTIONS this trainer's loop does not apply; enabling one raises.
    unapplied_options: frozenset[str] = frozenset({"log_first_step_diagnostics"})
    # Keyword options of the strategy's own config that this trainer reads,
    # beyond COMMON_OPTIONS and PIPELINE_OPTIONS.
    strategy_options: frozenset[str] = frozenset()
    # Whether construction checks loss_function against model.task (the
    # ``task`` keyword); trainers that substitute their own task loss do not.
    checks_task_loss = True
    # Whether a DistributedDataParallel wrapper reduces this trainer's
    # gradients, through ``backward()`` on its output. Trainers that compute
    # gradients with ``torch.autograd.grad`` and average them across ranks
    # themselves (FA/DFA, LocalCA) set this to False.
    ddp_reduces_gradients = True

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        loss_function: str = "mse",
        epochs: int = 100,
        early_stopping: bool = False,
        patience: int = 10,
        batch_size: int = 256,
        shuffle: bool = True,
        grad_clip_value: Optional[float] = 2.0,
        checkpointing: bool = False,
        checkpoint_interval: int = 10,
        load_best_state_dict: bool = True,
        plot_losses: bool = False,
        save_path: Optional[str] = None,
        suppress_prints=False,
        print_every=10,
        analysis_manager: Optional[object] = None,
        **kwargs,
    ):
        reject_unknown_options(type(self), kwargs)
        reject_unapplied_options(type(self), kwargs)
        self.optimizer = optimizer
        self.loss_function: LossFunction = get_loss_function(loss_function)
        if self.checks_task_loss:
            validate_task_loss(
                kwargs.get("task"), self.loss_function, owner=type(self).__name__
            )
        self.epochs = epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.grad_clip_value = grad_clip_value
        self.checkpointing = checkpointing
        self.checkpoint_interval = checkpoint_interval
        self.load_best_state_dict = load_best_state_dict
        self.plot_losses = plot_losses
        self.save_path = save_path
        self.suppress_prints = suppress_prints
        self.print_every = print_every
        self.analysis_manager = analysis_manager
        self.seed = int(kwargs.get("seed", 0) or 0)
        self.best_state_device = normalize_choice(
            kwargs.get("best_state_device", "model"),
            BEST_STATE_DEVICES,
            field="best_state_device",
        )
        self.save_best_checkpoint = normalize_choice(
            kwargs.get("save_best_checkpoint", "every_improvement"),
            BEST_CHECKPOINT_SAVES,
            field="save_best_checkpoint",
        )
        # Training-time gate updates, in canonical spelling, checked here.
        (
            self.reactivation_update_mode,
            self.reactivation_recalibration_mode,
        ) = validate_reactivation_updates(
            update_mode=kwargs.get("reactivation_update_mode"),
            recalibration_mode=kwargs.get("reactivation_recalibration_mode"),
            recalibrate_every=kwargs.get("recalibrate_reactivation_every"),
            start_epoch=kwargs.get("reactivation_recalibration_start_epoch"),
            num_batches=kwargs.get("reactivation_recalibration_num_batches"),
            ema_alpha=kwargs.get("reactivation_recalibration_ema_alpha"),
        )
        configured_loader_seed = kwargs.get("loader_seed", None)
        self.loader_seed = int(
            self.seed if configured_loader_seed is None else configured_loader_seed
        )

        regularization_config = kwargs.get("regularization_config")
        pruning_config = kwargs.get("pruning_config")

        # Initialize enhanced feature configs with safe defaults
        self.regularization_config = regularization_config or RegularizationConfig()
        self.pruning_config = pruning_config or PruningConfig()
        if (
            self.pruning_config.enabled
            and self.pruning_config.save_pruning_stats
            and not self.save_path
        ):
            raise ValueError("pruning.save_pruning_stats requires a save_path")

        # Initialize RegularizationManager
        self.regularization_manager = RegularizationManager(self.regularization_config)
        self.pruning_manager = PruningManager(self.pruning_config)

        # Log regularization settings
        if (
            self.regularization_config.l1_weight > 0
            or self.regularization_config.l2_weight > 0
        ):
            logger.info(
                f"L1/L2 regularization enabled: l1_weight={self.regularization_config.l1_weight}, l2_weight={self.regularization_config.l2_weight}"
            )
        if self.regularization_config.split_params and any(
            group_cfg.get("l1_weight", 0) > 0 or group_cfg.get("l2_weight", 0) > 0
            for group_cfg in self.regularization_config.param_group_weights.values()
            if isinstance(group_cfg, dict)
        ):
            logger.info(
                "Split-parameter regularization enabled for groups: %s",
                {
                    name: cfg
                    for name, cfg in self.regularization_config.param_group_weights.items()
                    if isinstance(cfg, dict)
                    and (cfg.get("l1_weight", 0) > 0 or cfg.get("l2_weight", 0) > 0)
                },
            )
        if self.regularization_config.enforce_ei_weight_ratio:
            logger.info(
                f"E/I weight ratio regularization enabled: target_ratio={self.regularization_config.target_ei_weight_ratio}, loss_weight={self.regularization_config.ei_ratio_loss_weight}"
            )

        # Performance optimization flags (disabled by default for speed)
        self.enable_nan_checking = kwargs.get("enable_nan_checking", False)
        configured_mode = kwargs.get("grad_clip_mode")
        self.grad_clip_mode = (
            type(self).default_grad_clip_mode
            if configured_mode is None
            else normalize_choice(
                configured_mode, GRAD_CLIP_MODES, field="grad_clip_mode"
            )
        )
        self.profile_dataloader = kwargs.get("profile_dataloader", False)

        self.enable_amp = kwargs.get("use_amp", False)
        if self.enable_amp and torch.cuda.is_available():
            try:
                self.scaler = GradScaler("cuda")
                logger.info("Mixed precision training enabled")
            except ImportError:
                logger.warning("AMP not available, falling back to standard precision")
                self.enable_amp = False
        else:
            if self.enable_amp:
                logger.warning(
                    "%s: use_amp=true requests CUDA mixed precision, but CUDA is "
                    "not available, so this run trains in full precision.",
                    type(self).__name__,
                )
            self.enable_amp = False

        self.filename_prefix = "standard_"
        self._train_sampler: DistributedSampler | None = None
        self._valid_sampler: Sampler | None = None
        for k, v in kwargs.items():
            if k not in _NORMALIZED_OPTIONS:
                setattr(self, k, v)

    # ------------------------------------------------------------------
    # DDP helpers
    # ------------------------------------------------------------------

    @property
    def _is_distributed(self) -> bool:
        return dist.is_initialized()

    @property
    def _is_main_process(self) -> bool:
        return not self._is_distributed or dist.get_rank() == 0

    @staticmethod
    def _unwrap_model(model):
        """Return the underlying model, stripping DDP wrapper if present."""
        return model.module if hasattr(model, "module") else model

    def _require_single_process(self, reason: str) -> None:
        """Raise under distributed training, which ``reason`` rules out."""
        if self._is_distributed:
            raise NotImplementedError(
                f"{type(self).__name__} does not support distributed training: "
                f"{reason}. Run it in a single process."
            )

    def _reject_trainability_change_under_ddp(self, model, change: str) -> None:
        """Raise when ``change`` would starve a DistributedDataParallel reducer.

        DDP registers the parameters that require gradients when it wraps the
        model and waits for a gradient of each from every ``backward()``.
        Freezing a registered parameter afterwards, or a loss that reaches
        only some of them, stops training with a reduction error at the next
        step.
        """
        if self.ddp_reduces_gradients and isinstance(model, DistributedDataParallel):
            raise NotImplementedError(
                f"{type(self).__name__}: {change}, which DistributedDataParallel "
                "does not support: it waits for a gradient of every parameter "
                "that required one when it wrapped the model. Run it in a single "
                "process."
            )

    # ------------------------------------------------------------------
    # Learning-rate schedule (per epoch)
    # ------------------------------------------------------------------

    def _get_torch_optimizer(self):
        """Unwrap CustomWeightDecayOptimizer to get the real torch.optim.Optimizer."""
        opt = self.optimizer
        # CustomWeightDecayOptimizer stores the real optimizer as .optimizer
        if hasattr(opt, "optimizer") and isinstance(
            opt.optimizer, torch.optim.Optimizer
        ):
            return opt.optimizer
        return opt

    def _resolve_lr_schedule(self) -> tuple[str, float]:
        """Return the validated ``(lr_schedule, lr_warmup_epochs)`` pair."""
        return validate_lr_schedule(
            getattr(self, "lr_schedule", "none"),
            getattr(self, "lr_warmup_epochs", 0),
            supported=bool(type(self).supports_lr_schedule),
            owner=type(self).__name__,
        )

    def _build_scheduler(self, n_epochs: int) -> None:
        """Create the per-epoch LR scheduler once the epoch count is known.

        ``lr_schedule='cosine'`` scales every parameter group's learning rate
        by ``(e + 1) / warmup`` for epochs ``e < warmup`` and by
        ``0.5 * (1 + cos(pi * (e - warmup) / (n_epochs - warmup)))`` after it.
        ``_run_epoch`` steps the scheduler once after each training epoch.
        ``lr_schedule='none'`` leaves the optimizer untouched.
        """
        schedule, warmup = self._resolve_lr_schedule()
        if schedule == "none":
            self.scheduler = None
            return

        torch_opt = self._get_torch_optimizer()
        base_lrs = [pg["lr"] for pg in torch_opt.param_groups]

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return (epoch + 1) / max(warmup, 1)
            progress = (epoch - warmup) / max(n_epochs - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(torch_opt, lr_lambda)
        logger.info(
            f"Cosine LR schedule: warmup={warmup}, total={n_epochs}, "
            f"base_lr={base_lrs}"
        )

    def _dataloaders(
        self, train_data: torch.utils.data.Dataset, valid_data: torch.utils.data.Dataset
    ) -> tuple[DataLoader, DataLoader]:
        # The device is resolved first so pin_memory: null pins host batches
        # for a GPU run (it had resolved against "cpu" before the device was
        # set). Pinning changes neither values nor batch order.
        tuning = resolve_dataloader_tuning_from_config(
            train_data,
            device=self._resolve_device(),
            config=self,
        )
        # The resolved loader settings are part of the run record.
        self.dataloader_tuning = asdict(tuning)
        if not self.suppress_prints and tuning.auto_num_workers:
            total_cpus = os.cpu_count() or tuning.available_cpus
            logger.info(
                "Auto-detected num_workers=%s (visible CPUs: %s, total system CPUs: %s)",
                tuning.num_workers,
                tuning.available_cpus,
                total_cpus,
            )
        if not self.suppress_prints:
            logger.info("DataLoader tuning: %s", tuning.summary(train_data))

        # Use distributed samplers for training and validation under DDP.
        # Validation uses an explicit padding sentinel sampler so every rank
        # executes the same number of forward passes and metrics can be
        # reduced exactly without silent sample duplication.
        if self._is_distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            self._train_sampler = DistributedSampler(
                train_data,
                num_replicas=world_size,
                rank=rank,
                shuffle=self.shuffle,
                seed=self.loader_seed,
            )
            self._valid_sampler = PaddedDistributedEvalSampler(
                valid_data,
                num_replicas=world_size,
                rank=rank,
            )
            train_shuffle = False  # sampler handles shuffling
        else:
            self._train_sampler = None
            self._valid_sampler = None
            train_shuffle = self.shuffle

        loader_rank = dist.get_rank() if self._is_distributed else 0
        train_seed_kwargs = seeded_dataloader_kwargs(
            self.loader_seed,
            stream=0,
            rank=loader_rank,
        )
        valid_seed_kwargs = seeded_dataloader_kwargs(
            self.loader_seed,
            stream=1,
            rank=loader_rank,
        )

        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=train_shuffle,
            sampler=self._train_sampler,
            **train_seed_kwargs,
            **tuning.as_kwargs(),
        )
        valid_loader = DataLoader(
            valid_data,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self._valid_sampler,
            **valid_seed_kwargs,
            **tuning.as_kwargs(),
        )
        self.data_parallel = data_parallel_layout(
            train_loader, dist.get_world_size() if self._is_distributed else 1
        )
        return train_loader, valid_loader

    def _resolve_device(self):
        """The training device: as set, else this rank's GPU, the GPU or the CPU."""
        if not hasattr(self, "device"):
            if self._is_distributed:
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                self.device = f"cuda:{local_rank}"
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _initialize_attributes(
        self, model: BaseModel, total_epochs: int, *, snapshot_best: bool = True
    ) -> None:
        """Prepare a training phase; ``snapshot_best=False`` skips the best-state copy."""
        model.to(self._resolve_device())
        if snapshot_best and not hasattr(self, "best_state_dict"):
            # Unwrapped, so a DDP run restores it into the unwrapped model.
            self.best_state_dict = self._snapshot_state(model)
            # Until a validation loss improves on inf, the best state is the
            # pre-training snapshot.
            self.best_state_is_initial = True
        if not hasattr(self, "best_loss"):
            self.best_loss = float("inf")
        if not hasattr(self, "best_epoch"):
            self.best_epoch = 1
        if not hasattr(self, "start_time"):
            self.start_time = time.time()
        if not hasattr(self, "logger_info_prefix"):
            self.logger_info_prefix = "[Standard] "
        if not getattr(self, "_counting_run_epochs", False):
            self.epoch_counter = 0
            self.total_epochs = total_epochs
        self.patience_counter = 0
        self._capture_reactivation_param_states(model)
        self._configure_reactivation_update_mode(model)
        self._warn_if_recalibration_updates_no_gate(model)

    def _reject_unapplied_gradient_scaling(self, model: BaseModel) -> None:
        """Fail when gradient scaling cannot act on this trainer's gradients.

        ``gradient_scaling`` strategies are backward hooks on dendritic
        modules. Trainers that assign ``param.grad`` directly (FA/DFA and the
        local rules) never run those hooks, so the configured scaling would
        be silently skipped.
        """
        configured = configured_gradient_scaling(model)
        if configured:
            raise ValueError(
                f"{type(self).__name__} assigns gradients directly, so the "
                f"configured gradient scaling ({', '.join(configured)}) would "
                "never be applied. Set model.core.sparsity/reactivation/"
                "blocklinear.gradient_scaling to 'none', or use a "
                "backpropagation strategy (standard, recurrent, soma_dfa)."
            )

    def _iter_reactivation_params(
        self, model: BaseModel
    ) -> list[tuple[torch.nn.Parameter, bool]]:
        """Return reactivation params with their original trainability."""
        from dendritic_modeling.networks.activations.endpoint_tanh import (
            EndpointNormalizedTanh,
        )

        params: list[tuple[torch.nn.Parameter, bool]] = []
        for module in iter_modules_of_type(
            self._unwrap_model(model), (ParametricActivation, EndpointNormalizedTanh)
        ):
            for param in module.parameters(recurse=False):
                params.append((param, bool(param.requires_grad)))
        return params

    def _capture_reactivation_param_states(self, model: BaseModel) -> None:
        """Cache the original trainability of reactivation parameters."""
        model_id = id(self._unwrap_model(model))
        if getattr(self, "_reactivation_state_model_id", None) == model_id:
            return
        self._reactivation_state_model_id = model_id
        self._reactivation_param_states = self._iter_reactivation_params(model)

    def _normalize_reactivation_update_mode(self) -> str:
        return normalize_reactivation_update_mode(
            getattr(self, "reactivation_update_mode", None)
        )

    def _set_reactivation_trainability(self, requires_grad: bool) -> None:
        """Apply trainability while preserving originally fixed parameters."""
        for param, original_requires_grad in getattr(
            self, "_reactivation_param_states", []
        ):
            target_requires_grad = original_requires_grad if requires_grad else False
            param.requires_grad_(target_requires_grad)
            if not target_requires_grad:
                param.grad = None

    def _configure_reactivation_update_mode(self, model: BaseModel) -> None:
        """Freeze/unfreeze reactivation params according to the training mode."""
        self._capture_reactivation_param_states(model)
        mode = self._normalize_reactivation_update_mode()
        self.reactivation_update_mode = mode
        if mode == "backprop":
            self._set_reactivation_trainability(True)
            return

        if any(trainable for _, trainable in self._reactivation_param_states):
            self._reject_trainability_change_under_ddp(
                model,
                f"reactivation_update_mode={mode!r} freezes the reactivation "
                "parameters",
            )
        self._set_reactivation_trainability(False)

    def _normalize_reactivation_recalibration_mode(self) -> str:
        return normalize_reactivation_recalibration_mode(
            getattr(self, "reactivation_recalibration_mode", None)
        )

    def _resolve_reactivation_recalibration_mode(
        self, model: BaseModel
    ) -> Optional[str]:
        """Resolve the calibration rule used for periodic (m, b) refreshes."""
        explicit_mode = self._normalize_reactivation_recalibration_mode()
        if explicit_mode != "from_init_policy":
            return explicit_mode

        if self._normalize_reactivation_update_mode() == "quantile":
            return "occupancy_quantile"

        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
            collect_model_data_driven_reactivation_policies,
        )

        requested_policies, saw_layer_policy = (
            collect_model_data_driven_reactivation_policies(self._unwrap_model(model))
        )
        if requested_policies:
            # Defer to per-layer policies so fixed/analytical layers are
            # not recalibrated just because a sibling layer is data-driven.
            return None

        return None if saw_layer_policy else "occupancy_quantile"

    def _warn_if_recalibration_updates_no_gate(self, model: BaseModel) -> None:
        """Warn once when periodic reactivation recalibration calibrates no gate.

        With ``reactivation_recalibration_mode: from_init_policy`` only gates
        with a data-driven ``init_policy`` are recalibrated, so a model whose
        gates are all initialized analytically or fixed is left unchanged;
        a model without calibratable gates is left unchanged by every mode.
        Each scheduled recalibration still draws training batches, which
        advances the training loader's random stream, so removing the setting
        can change the data order of later epochs.
        """
        every = int(getattr(self, "recalibrate_reactivation_every", 0) or 0)
        if every <= 0 or getattr(self, "_warned_recalibration_updates_no_gate", False):
            return
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
            reactivation_calibration_targets,
        )

        mode = self._resolve_reactivation_recalibration_mode(model)
        if reactivation_calibration_targets(self._unwrap_model(model), mode):
            return
        self._warned_recalibration_updates_no_gate = True
        if not self._is_main_process:
            return
        if mode is None:
            reason = (
                "reactivation_recalibration_mode 'from_init_policy' recalibrates "
                "only gates with a data-driven init_policy, and the gates of this "
                "model are initialized analytically or fixed (a calibration rule "
                "such as occupancy_quantile would recalibrate them)"
            )
        else:
            reason = "the model has no reactivation gate that calibration can set"
        logger.warning(
            "%srecalibrate_reactivation_every=%d requests periodic reactivation "
            "recalibration, but no gate will be recalibrated: %s. Each scheduled "
            "recalibration still draws training batches, which advances the "
            "training data loader's random stream, so setting "
            "recalibrate_reactivation_every to 0 can change the data order of "
            "later epochs; keep the setting to reproduce an earlier run.",
            getattr(self, "logger_info_prefix", ""),
            every,
            reason,
        )

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict:
        self._recurrent_model = bool(
            getattr(self._unwrap_model(model), "_is_recurrent_core", False)
        )
        self._build_scheduler(self.epochs)
        train_loader, valid_loader = self._dataloaders(train_data, valid_data)

        train_losses = []
        valid_losses = []
        train_losses_base = []
        train_losses_reg = []

        self._initialize_attributes(model, self.epochs)
        self._save_checkpoint(model)

        train_losses, valid_losses, train_losses_base, train_losses_reg = (
            self._run_training_loop(
                n_epochs=self.epochs,
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_losses=train_losses,
                valid_losses=valid_losses,
                train_losses_base=train_losses_base,
                train_losses_reg=train_losses_reg,
                logger_info=self.logger_info_prefix,
            )
        )

        if self._is_main_process:
            logger.info(
                f"Best epoch: {self.best_epoch}, Best loss: {self.best_loss:.4f}"
            )
        self._restore_best_state(model, valid_losses)

        results = {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_state_dict": self.best_state_dict,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_losses_base": train_losses_base,
            "train_losses_reg": train_losses_reg,
            "dataloader_tuning": getattr(self, "dataloader_tuning", None),
            "data_parallel": getattr(self, "data_parallel", None),
        }

        self._loss_plotting(
            train_losses, valid_losses, train_losses_base, train_losses_reg
        )

        return results

    def _run_training_loop(
        self,
        n_epochs: int,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: list[float],
        valid_losses: list[float],
        train_losses_base: list[float],
        train_losses_reg: list[float],
        logger_info: str = "",
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        for _ in range(1, n_epochs + 1):
            train_losses, valid_losses, train_losses_base, train_losses_reg = (
                self._run_epoch(
                    model=model,
                    train_loader=train_loader,
                    valid_loader=valid_loader,
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    train_losses_base=train_losses_base,
                    train_losses_reg=train_losses_reg,
                    logger_info=logger_info,
                )
            )

            if self._check_early_stopping():
                break

        return train_losses, valid_losses, train_losses_base, train_losses_reg

    def _run_epoch(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        train_losses: list[float],
        valid_losses: list[float],
        train_losses_base: list[float],
        train_losses_reg: list[float],
        logger_info: str = "",
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        self.epoch_counter += 1

        # Tell DistributedSampler which epoch we're in so each rank
        # sees a different shuffle order per epoch.
        if self._train_sampler is not None:
            self._train_sampler.set_epoch(self.epoch_counter)

        # Periodic reactivation recalibration: recompute (m, b) from the
        # current voltage distribution every N epochs using the same
        # quantile formula as the init calibration. This keeps the gate
        # matched to the evolving V distribution and eliminates drift.
        recalib_every = int(getattr(self, "recalibrate_reactivation_every", 0) or 0)
        recalib_start = int(getattr(self, "reactivation_recalibration_start_epoch", 1))
        if (
            recalib_every > 0
            and self.epoch_counter >= recalib_start
            and (self.epoch_counter - recalib_start) % recalib_every == 0
        ):
            self._recalibrate_reactivation(model, train_loader)

        train_loss_total, train_loss_base, train_loss_reg = self._epoch_train(
            model, train_loader
        )
        # Step the per-epoch LR schedule once, after the training epoch.
        scheduler = getattr(self, "scheduler", None)
        if scheduler is not None:
            scheduler.step()
        train_losses.append(train_loss_total)
        train_losses_base.append(train_loss_base)
        train_losses_reg.append(train_loss_reg)
        # Under DDP, all ranks validate on disjoint shards with equal-length
        # padded samplers and then all-reduce exact sample-weighted loss.
        if self._is_distributed:
            unwrapped = self._unwrap_model(model)
            valid_loss = self._epoch_valid_distributed(unwrapped, valid_loader)
        else:
            valid_loss = self._non_distributed_validation_loss(model, valid_loader)

        valid_losses.append(valid_loss)

        self._update_best(model, valid_loss)
        self._save_checkpoint(model)
        self._analyze()
        self._tracking(
            train_loss_total,
            valid_loss,
            logger_info,
            train_loss_base=train_loss_base,
            train_loss_reg=train_loss_reg,
        )
        return train_losses, valid_losses, train_losses_base, train_losses_reg

    def _recalibrate_reactivation(
        self, model: BaseModel, train_loader: DataLoader
    ) -> None:
        """Recompute (m, b) from the current voltage distribution.

        Uses the same calibration function as the init-time calibration
        but runs it on live training data with the current weights. This
        keeps the gate matched to the evolving V distribution.
        """
        from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
            calibrate_reactivation_from_data,
        )

        mode = self._resolve_reactivation_recalibration_mode(model)
        n_batches = int(getattr(self, "reactivation_recalibration_num_batches", 3))
        ema_alpha = float(getattr(self, "reactivation_recalibration_ema_alpha", 1.0))

        # Collect a few batches from the current train loader
        batches = []
        for i, batch in enumerate(train_loader):
            if i >= n_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            batches.append(x.to(self.device))

        if not batches:
            return

        diag = calibrate_reactivation_from_data(
            model,
            batches,
            k=None,
            device=self.device,
            mode=mode,
            ema_alpha=ema_alpha,
        )
        # Each DDP rank calibrated on its own shard; adopt rank 0's result.
        self._broadcast_reactivation_state_from_rank0(model)

        if self._is_main_process and diag:
            m_values = [d["m"] for d in diag.values()]
            applied_m_values = [d.get("m_applied", d["m"]) for d in diag.values()]
            logger.info(
                f"{self.logger_info_prefix}Recalibrated reactivation at epoch "
                f"{self.epoch_counter}: {len(diag)} layers, "
                f"m target range [{min(m_values):.3f}, {max(m_values):.3f}], "
                f"applied m range [{min(applied_m_values):.3f}, {max(applied_m_values):.3f}], "
                f"batches={n_batches}, ema_alpha={ema_alpha:.3f}"
            )

    def _broadcast_reactivation_state_from_rank0(self, model: BaseModel) -> None:
        """Make every DDP rank use the reactivation state calibrated on rank 0.

        Parameters and buffers of each reactivation gate are broadcast in
        module order and loaded through ``load_state_dict`` so Python-side
        calibrated values (``fixed_b``) follow the broadcast buffers.
        """
        if self._is_distributed:
            broadcast_reactivation_state(self._unwrap_model(model))

    def _epoch_train(
        self, model: BaseModel, train_loader: DataLoader
    ) -> tuple[float, float, float]:
        model.train()
        # (total, base, regularization) sums stay on the device and are read
        # once per epoch; float64 additions in step order reproduce host sums.
        loss_sums = torch.zeros(3, dtype=torch.float64, device=self.device)
        no_regularization = torch.zeros((), device=self.device)
        n_batches = 0

        # Check once if regularization is enabled to avoid repeated checks
        has_regularization = self.regularization_manager.has_regularization()
        profile_loader = bool(getattr(self, "profile_dataloader", False))
        data_wait_total = 0.0
        step_time_total = 0.0
        max_data_wait = 0.0

        train_iter = iter(train_loader)
        while True:
            fetch_start = time.perf_counter()
            try:
                batch = next(train_iter)
            except StopIteration:
                break
            if profile_loader:
                data_wait = time.perf_counter() - fetch_start
                data_wait_total += data_wait
                max_data_wait = max(max_data_wait, data_wait)
                step_start = time.perf_counter()

            x_batch, y_batch = self._move_training_batch_to_device(
                batch,
                train_loader,
            )

            if self.enable_amp:
                # Mixed precision forward pass
                with autocast("cuda"):
                    base_loss = self.loss_function(model, x_batch, y_batch)

                    # Add all regularization losses only if regularization is enabled
                    if has_regularization:
                        reg_loss = (
                            self.regularization_manager.compute_regularization_loss(
                                model,
                                x_batch,
                                y_batch,
                                param_groups=self.optimizer.param_groups,
                            )
                        )
                        total_loss = base_loss + reg_loss
                    else:
                        total_loss = base_loss
                        reg_loss = no_regularization

                self.optimizer.zero_grad()
                self.scaler.scale(total_loss).backward()
                # Unscale before inspecting or clipping gradients; the scaler has
                # then recorded any overflow and still skips that step.
                self.scaler.unscale_(self.optimizer)
                self._finalize_gradients(model)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                base_loss: torch.Tensor = self.loss_function(model, x_batch, y_batch)

                # Add all regularization losses only if regularization is enabled
                if has_regularization:
                    reg_loss = self.regularization_manager.compute_regularization_loss(
                        model,
                        x_batch,
                        y_batch,
                        param_groups=self.optimizer.param_groups,
                    )
                    total_loss = base_loss + reg_loss
                else:
                    total_loss = base_loss
                    reg_loss = no_regularization

                self.optimizer.zero_grad()
                total_loss.backward()
                self._finalize_gradients(model)
                self.optimizer.step()

            loss_sums += torch.stack(
                [total_loss.detach(), base_loss.detach(), reg_loss.detach()]
            ).double()
            n_batches += 1
            if profile_loader:
                step_time_total += _step_seconds(step_start, loss_sums)

        if profile_loader and self._is_main_process:
            self._log_dataloader_profile(
                n_batches=n_batches,
                data_wait_total=data_wait_total,
                step_time_total=step_time_total,
                max_data_wait=max_data_wait,
            )
        return self._mean_epoch_losses(loss_sums, n_batches)

    def _mean_epoch_losses(
        self, loss_sums: torch.Tensor, n_batches: int
    ) -> tuple[float, float, float]:
        """Mean (total, base, regularization) step losses of an epoch.

        ``loss_sums`` holds this rank's float64 step sums; under DDP the sums
        and batch counts of all ranks are added first. One host read.
        """
        if self._is_distributed:
            stats = torch.cat(
                [
                    loss_sums,
                    torch.full(
                        (1,),
                        float(n_batches),
                        dtype=torch.float64,
                        device=loss_sums.device,
                    ),
                ]
            )
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            *sums, count = stats.tolist()
            n_batches = int(count)
        else:
            sums = loss_sums.tolist()
        denom = max(n_batches, 1)
        total, base, regularization = sums
        return total / denom, base / denom, regularization / denom

    def _loader_uses_non_blocking_transfer(self, loader: DataLoader) -> bool:
        """Return whether batch copies can be issued asynchronously."""
        try:
            device_type = torch.device(self.device).type
        except (RuntimeError, TypeError):
            return False
        return device_type == "cuda" and bool(getattr(loader, "pin_memory", False))

    def _move_training_batch_to_device(
        self,
        batch,
        loader: DataLoader,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Move the input/target tensors using the loader's transfer policy.

        A recurrent model reads per-sample sequence lengths; a batch that
        carries them would be run at its padded length, so it raises.
        """
        if len(batch) > 2 and getattr(self, "_recurrent_model", False):
            raise ValueError(
                f"{type(self).__name__} trains on (inputs, targets) and would "
                "drop the per-sample sequence lengths this dataset provides; "
                "train recurrent models on variable-length data with "
                "training.main.strategy: recurrent"
            )
        non_blocking = self._loader_uses_non_blocking_transfer(loader)
        return (
            batch[0].to(self.device, non_blocking=non_blocking),
            batch[1].to(self.device, non_blocking=non_blocking),
        )

    def _log_dataloader_profile(
        self,
        *,
        n_batches: int,
        data_wait_total: float,
        step_time_total: float,
        max_data_wait: float,
    ) -> None:
        """Log epoch-level dataloader wait time for input-pipeline diagnosis."""
        if n_batches <= 0:
            return
        total_time = data_wait_total + step_time_total
        wait_fraction = data_wait_total / max(total_time, 1e-12)
        logger.info(
            "%sDataLoader profile epoch=%s batches=%s "
            "data_wait=%.3fs step_time=%.3fs wait_fraction=%.1f%% "
            "avg_wait=%.1fms max_wait=%.1fms",
            getattr(self, "logger_info_prefix", "[Standard] "),
            getattr(self, "epoch_counter", 0),
            n_batches,
            data_wait_total,
            step_time_total,
            100.0 * wait_fraction,
            1000.0 * data_wait_total / n_batches,
            1000.0 * max_data_wait,
        )

    def _epoch_valid_stats(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> _ValidationLossStats:
        """Collect explicit batch- and sample-level validation aggregates."""
        model.eval()
        sums = _ValidationSums(self.device)
        with torch.no_grad():
            for batch in valid_loader:
                x_batch, y_batch = self._move_training_batch_to_device(
                    batch,
                    valid_loader,
                )
                loss: torch.Tensor = self.loss_function(model, x_batch, y_batch)
                sums.add(loss, y_batch.size(0))
        return sums.stats()

    def _epoch_valid(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> tuple[float, int]:
        """Return the legacy ``(batch_loss_sum, batch_count)`` tuple."""
        stats = self._epoch_valid_stats(model, valid_loader)
        return stats.batch_loss_sum, stats.batch_count

    def _validation_loss_reduction(self) -> str | None:
        """Return the configured scalar-loss reduction, when declared."""
        reduction = getattr(self.loss_function, "reduction", None)
        return str(reduction) if reduction is not None else None

    def _non_distributed_validation_loss(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> float:
        """Return the checkpoint metric for non-distributed validation.

        Mean-reduced losses are weighted by the number of examples in each
        batch, matching evaluation of the complete dataset when the final
        batch is smaller. Other or undeclared reductions retain the historical
        mean-over-batches behavior.
        """
        stats = self._epoch_valid_stats(model, valid_loader)
        if self._validation_loss_reduction() == "mean":
            return stats.sample_mean
        return stats.batch_mean

    def _epoch_valid_distributed(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> float:
        """Return exact global validation loss under DDP.

        All ranks process equal-length validation shards via
        ``PaddedDistributedEvalSampler``. Padding entries are excluded from the
        reduced sample-weighted loss.
        """
        model.eval()
        sums = _ValidationSums(self.device)
        sampler = valid_loader.sampler
        if isinstance(sampler, PaddedDistributedEvalSampler):
            real_count = len(sampler.real_indices)
        else:
            real_count = None

        seen = 0
        with torch.no_grad():
            for batch in valid_loader:
                x_batch, y_batch = self._move_training_batch_to_device(
                    batch,
                    valid_loader,
                )
                batch_size = y_batch.size(0)
                if real_count is not None:
                    real_in_batch = min(batch_size, real_count - seen)
                    if real_in_batch <= 0:
                        seen += batch_size
                        continue
                    if real_in_batch < batch_size:
                        x_batch = x_batch[:real_in_batch]
                        y_batch = y_batch[:real_in_batch]
                        batch_size = real_in_batch
                    seen += batch_size
                loss: torch.Tensor = self.loss_function(model, x_batch, y_batch)
                sums.add(loss, batch_size)
        return sums.distributed_sample_mean()

    def _evaluate_loader_loss(self, model: BaseModel, loader: DataLoader) -> float:
        if self._is_distributed:
            return self._epoch_valid_distributed(self._unwrap_model(model), loader)
        return self._non_distributed_validation_loss(model, loader)

    def _apply_pruning_and_evaluate(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict:
        """Apply configured pruning after training and measure pre/post loss."""
        if not hasattr(self, "device"):
            self._initialize_attributes(model, self.epochs)

        unwrapped = self._unwrap_model(model)
        self.pruning_manager.validate_model(unwrapped)
        evaluate = self.pruning_config.re_evaluate_after
        if evaluate:
            train_loader, valid_loader = self._dataloaders(train_data, valid_data)
            pre_train_loss = self._evaluate_loader_loss(model, train_loader)
            pre_valid_loss = self._evaluate_loader_loss(model, valid_loader)
        pruning_stats = self.pruning_manager.prune_model(unwrapped)
        if self._is_main_process:
            if self.pruning_config.report_non_pruned:
                self.pruning_manager.print_pruning_report()
            if self.pruning_config.save_pruning_stats:
                self.pruning_manager.save_pruning_stats(
                    os.path.join(self.save_path, "pruning_stats.json")
                )
        result = {"pruning_stats": pruning_stats}
        if evaluate:
            post_train_loss = self._evaluate_loader_loss(model, train_loader)
            post_valid_loss = self._evaluate_loader_loss(model, valid_loader)
            result["pre_pruning_performance"] = {
                "train_loss": pre_train_loss,
                "valid_loss": pre_valid_loss,
            }
            result["post_pruning_performance"] = {
                "train_loss": post_train_loss,
                "valid_loss": post_valid_loss,
            }
        return result

    def _restore_best_state(self, model: BaseModel, valid_losses: list) -> None:
        """Load the best validation epoch's weights when ``load_best_state_dict``.

        Raises when no epoch had a finite validation loss: the best state is
        then the untrained snapshot, and restoring it would report an untrained
        network as the trained result.
        """
        if self.save_best_checkpoint == "end" and not getattr(
            self, "best_state_is_initial", False
        ):
            self._write_best_checkpoint()
        if not self.load_best_state_dict:
            return
        if getattr(self, "best_state_is_initial", False):
            raise RuntimeError(
                f"{self.logger_info_prefix}no epoch had a finite validation loss "
                f"(validation losses: {valid_losses}), so load_best_state_dict "
                "would restore the untrained weights. The run failed; set "
                "load_best_state_dict: false to keep the final weights."
            )
        self._unwrap_model(model).load_state_dict(self.best_state_dict)
        if self._is_main_process:
            logger.info("Final model loaded from best checkpoint")

    @contextmanager
    def _run_epoch_count(self, total_epochs: int):
        """Number the epochs of all phases inside this block consecutively.

        Multi-phase trainers call ``Trainer.train`` once per phase. Inside the
        block those calls continue the epoch count instead of restarting it,
        so epoch files, logs and ``best_epoch`` index the run's concatenated
        loss histories, and ``total_epochs`` is the run's total.
        """
        self.epoch_counter = 0
        self.total_epochs = int(total_epochs)
        self._counting_run_epochs = True
        try:
            yield
        finally:
            self._counting_run_epochs = False

    def _snapshot_state(self, model: BaseModel) -> dict:
        """Copy the (unwrapped) model state where ``best_state_device`` says."""
        state = self._unwrap_model(model).state_dict()
        if self.best_state_device == "model":
            return deepcopy(state)
        copied = OrderedDict(
            (
                name,
                (
                    value.detach().to("cpu", copy=True)
                    if torch.is_tensor(value)
                    else deepcopy(value)
                ),
            )
            for name, value in state.items()
        )
        return copy_state_metadata(state, copied)

    def _write_best_checkpoint(self) -> None:
        if self._is_main_process and self.save_path and os.path.exists(self.save_path):
            atomic_torch_save(
                self.best_state_dict,
                os.path.join(self.save_path, f"{self.filename_prefix}best_model.pt"),
            )

    def _restart_best_state_selection(self, model: BaseModel) -> None:
        """Select the best state among the epochs that follow.

        Multi-phase trainers call this when a phase with another objective or
        estimator ends, so the result is never a state chosen by that phase.
        """
        self.best_loss = float("inf")
        self.best_epoch = self.epoch_counter
        self.best_state_dict = self._snapshot_state(model)
        self.best_state_is_initial = True

    def _update_best(self, model: BaseModel, valid_loss: float) -> None:
        if valid_loss < self.best_loss:
            self.best_loss = valid_loss
            self.best_epoch = self.epoch_counter
            self.best_state_is_initial = False
            self.best_state_dict = self._snapshot_state(model)
            self.patience_counter = 0
            if self.save_best_checkpoint == "every_improvement":
                self._write_best_checkpoint()
        else:
            self.patience_counter += 1

    def _save_checkpoint(self, model: BaseModel):
        if not self._is_main_process:
            return
        if self.checkpointing and (self.epoch_counter % self.checkpoint_interval == 0):
            checkpoint_path = os.path.join(
                self.save_path, f"{self.filename_prefix}checkpoints"
            )
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path, exist_ok=True)
            atomic_torch_save(
                self._unwrap_model(model).state_dict(),
                os.path.join(checkpoint_path, f"epoch_{self.epoch_counter}.pt"),
            )

    def _analyze(self):
        if not self._is_main_process:
            return
        if self.analysis_manager is not None:
            # Lazy import to avoid circular dependency
            from dendritic_modeling.analysis import AnalysisManager

            if isinstance(self.analysis_manager, AnalysisManager):
                self.analysis_manager.run_analysis(
                    filename=f"epoch{self.epoch_counter}", training=True
                )

    def _tracking(
        self,
        train_loss: float,
        valid_loss: float,
        logger_info: str = "",
        train_loss_base: Optional[float] = None,
        train_loss_reg: Optional[float] = None,
    ) -> None:
        elapsed = time.time() - self.start_time
        if not self.suppress_prints and (self.epoch_counter % self.print_every == 0):
            if train_loss_base is not None and train_loss_reg is not None:
                logger_info = (
                    logger_info
                    + f"epoch={self.epoch_counter}/{self.total_epochs}, "
                    + f"train_loss={train_loss:.4f} (base={train_loss_base:.4f}, reg={train_loss_reg:.4f}), "
                    + f"valid_loss={valid_loss:.4f}, "
                    + f"elapsed={elapsed:.1f}s"
                )
            else:
                logger_info = (
                    logger_info
                    + f"epoch={self.epoch_counter}/{self.total_epochs}, "
                    + f"train_loss={train_loss:.4f}, "
                    + f"valid_loss={valid_loss:.4f}, "
                    + f"elapsed={elapsed:.1f}s"
                )
            logger.info(logger_info)

    def _check_early_stopping(self) -> bool:
        if self.early_stopping and self.patience_counter >= self.patience:
            if not self.suppress_prints:
                logger.info(
                    f"Early stopping triggered after {self.epoch_counter} epochs."
                )
            return True
        else:
            return False

    def _loss_plotting(
        self,
        train_losses: list[float],
        valid_losses: list[float],
        train_losses_base: Optional[list[float]] = None,
        train_losses_reg: Optional[list[float]] = None,
    ) -> None:
        if not self._is_main_process:
            return
        if self.plot_losses and len(train_losses) > 1:
            if self.save_path and os.path.exists(self.save_path):
                plot_loss_curves(
                    train_losses=train_losses,
                    valid_losses=valid_losses,
                    loss_name=self.loss_function._loss_name,
                    save_dir=self.save_path,
                    filename_prefix=self.filename_prefix,
                    train_losses_base=train_losses_base,
                    train_losses_reg=train_losses_reg,
                )

    def _finalize_gradients(self, model: BaseModel) -> None:
        """Zero non-finite gradients when requested, then clip."""
        if self.enable_nan_checking and zero_nonfinite_gradients(model.parameters()):
            logger.warning("Non-finite gradient entries were set to zero")
        clip_gradients(model.parameters(), self.grad_clip_value, self.grad_clip_mode)


__all__ = ["Trainer"]
