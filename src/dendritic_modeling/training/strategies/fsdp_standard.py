"""
FSDP (Fully Sharded Data Parallel) training strategy.

This module provides FSDP-based distributed training functionality
for scaling to very large models across multiple GPUs and nodes.
"""

import logging
import os
import time
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
from torch.utils.data import DataLoader, Dataset, TensorDataset

try:  # PyTorch exposes the FSDP-aware scaler from this submodule.
    from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
except Exception:  # pragma: no cover - depends on installed PyTorch version.
    ShardedGradScaler = None

from dendritic_modeling.config.keys import field_default
from dendritic_modeling.config.training import (
    CommonTrainerConfig,
    OptimizerConfig,
    ParamGroupsConfig,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.models.base import model_param_groups
from dendritic_modeling.training.clipping import clip_gradients
from dendritic_modeling.training.distributed_samplers import (
    PaddedDistributedEvalSampler,
)
from dendritic_modeling.training.fsdp_utils import (
    get_fsdp_config,
    save_fsdp_checkpoint,
    wrap_model_with_fsdp,
)
from dendritic_modeling.training.loss.functions import LossFunction
from dendritic_modeling.training.optimizers.factory import (
    create_optimizer as build_optimizer,
)
from dendritic_modeling.training.optimizers.muonh import MuonH
from dendritic_modeling.training.reactivation_sync import refresh_fixed_b_from_buffers
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)

# Keys of FSDPTrainer(optimizer_config=...).
_OPTIMIZER_CONFIG_KEYS = frozenset({"optimizer", "param_groups"})


def _check_optimizer_config_keys(optimizer_config: Optional[dict]) -> None:
    unknown = sorted(set(optimizer_config or {}) - _OPTIMIZER_CONFIG_KEYS)
    if unknown:
        raise ValueError(
            f"FSDPTrainer optimizer_config has unknown keys {unknown}; it takes "
            "'optimizer' (training.main.optimizer) and 'param_groups' "
            "(training.main.common.param_groups)"
        )


def _optimizer_settings(optimizer_config: Optional[dict]) -> OptimizerConfig:
    """Return ``training.main.optimizer`` (``optimizer_config["optimizer"]``).

    ``synapse_decay_rate`` is sparse maintenance, never optimizer weight decay.
    """
    settings = (optimizer_config or {}).get("optimizer")
    if settings is None:
        return OptimizerConfig()
    if isinstance(settings, OptimizerConfig):
        return settings
    if isinstance(settings, dict):
        return OptimizerConfig(**settings)
    raise TypeError(
        "optimizer_config['optimizer'] must be an OptimizerConfig or a mapping, "
        f"got {type(settings).__name__}"
    )


def _param_groups_settings(optimizer_config: Optional[dict]) -> ParamGroupsConfig:
    """Return ``training.main.common.param_groups`` (``optimizer_config["param_groups"]``)."""
    settings = (optimizer_config or {}).get("param_groups")
    if settings is None:
        return ParamGroupsConfig()
    if isinstance(settings, ParamGroupsConfig):
        return settings
    if isinstance(settings, dict):
        return ParamGroupsConfig(**settings)
    raise TypeError(
        "optimizer_config['param_groups'] must be a ParamGroupsConfig or a "
        f"mapping, got {type(settings).__name__}"
    )


def _fsdp_units(model: torch.nn.Module) -> list[FullyShardedDataParallel]:
    return [m for m in model.modules() if isinstance(m, FullyShardedDataParallel)]


def fsdp_param_groups(
    model: torch.nn.Module, param_groups_config: ParamGroupsConfig
) -> list[dict[str, Any]]:
    """Return the trainable optimizer groups of an FSDP-wrapped model.

    The groups are those of single-process training, ``BaseModel.get_param_groups``
    (:func:`~dendritic_modeling.models.base.model_param_groups` for other
    modules) of the wrapped module, restricted to trainable parameters; empty
    groups are dropped. FSDP with ``use_orig_params`` keeps the original
    parameter objects, which the groups therefore hold. Without
    ``split_params`` the single group holds the wrapped model's trainable
    parameters (flat parameters under ``use_orig_params: false``).

    Raises ``ValueError`` when the groups do not cover each trainable
    parameter of the wrapped model exactly once, as when FSDP replaced the
    module parameters with flat parameters (``use_orig_params: false``).
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError("FSDP training requires at least one trainable parameter")
    if not param_groups_config.split_params:
        return [{"name": "all", "params": trainable, "lr": param_groups_config.lr}]
    module = model.module if isinstance(model, FullyShardedDataParallel) else model
    if isinstance(module, BaseModel):
        groups = module.get_param_groups(param_groups_config)
    else:
        groups = model_param_groups(module, param_groups_config)
    trainable_ids = {id(p) for p in trainable}
    grouped_ids: set[int] = set()
    param_groups = []
    for group in groups:
        params = [p for p in group["params"] if p.requires_grad]
        for param in params:
            if id(param) in grouped_ids:
                raise ValueError(
                    f"parameter group {group.get('name')!r} repeats a parameter"
                )
            grouped_ids.add(id(param))
        if params:
            param_groups.append({**group, "params": params})
    if grouped_ids != trainable_ids:
        raise ValueError(
            "split_params groups parameters by module, but "
            f"{len(trainable_ids - grouped_ids)} trainable parameters of the "
            "FSDP model are not module parameters (FSDP flat parameters, "
            "use_orig_params: false). Set distributed.fsdp.use_orig_params: true "
            "or param_groups.split_params: false."
        )
    return param_groups


def _reject_sharded_muonh(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> None:
    """Reject MuonH on FSDP units that flatten or shard parameters.

    MuonH orthogonalizes 2-D matrices. FSDP exposes each parameter of a
    sharded unit (and every flat parameter) as a 1-D view, which MuonH would
    silently update with its AdamW fallback.
    """
    if not isinstance(optimizer, MuonH):
        return
    for unit in _fsdp_units(model):
        if getattr(
            unit, "sharding_strategy", None
        ) != ShardingStrategy.NO_SHARD or not getattr(unit, "_use_orig_params", False):
            raise ValueError(
                "optimizer muonh needs full 2-D parameter matrices, but FSDP "
                f"{getattr(unit, 'sharding_strategy', None)} with use_orig_params="
                f"{getattr(unit, '_use_orig_params', None)} exposes parameters as "
                "1-D shards. Use distributed.fsdp.sharding_strategy: NO_SHARD "
                "with use_orig_params: true, or another optimizer."
            )


def _sequence_lengths_from_batch(
    batch,
    *,
    x_batch: torch.Tensor,
    device: torch.device,
) -> torch.Tensor | None:
    """Return optional per-sample sequence lengths from a recurrent batch.

    Sequence datasets use ``(inputs, labels, lengths)``.  Image and tabular
    datasets may also expose extra metadata, so only accept a one-value-per-row
    tensor when the input has an explicit time dimension.
    """
    if x_batch.dim() != 3 or not isinstance(batch, (tuple, list)) or len(batch) < 3:
        return None
    lengths = batch[2]
    if not torch.is_tensor(lengths):
        try:
            lengths = torch.as_tensor(lengths)
        except (TypeError, ValueError):
            return None
    lengths = lengths.reshape(-1)
    if lengths.numel() != x_batch.shape[0]:
        return None
    return lengths.to(device=device, dtype=torch.long, non_blocking=True)


def _correct_predictions(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Correct class predictions of a batch, or NaN when it has no class logits.

    Accuracy is defined for ``[batch, classes]`` logits with integer class
    targets; other outputs (regression) report NaN.
    """
    if outputs.dim() == 2 and targets.dim() == 1 and not targets.is_floating_point():
        return (outputs.argmax(dim=1) == targets).sum().double()
    return torch.full((), float("nan"), dtype=torch.float64, device=outputs.device)


def _mean_over_ranks(
    loss_sum: torch.Tensor, correct: torch.Tensor, examples: int
) -> tuple[float, float]:
    """Sample-weighted mean loss and accuracy over all ranks.

    The sums are reduced in float64 (float32 had rounded the loss sum and,
    beyond 2**24 examples, the counts) and read with one host transfer.
    """
    stats = torch.stack(
        [
            loss_sum,
            correct.double(),
            torch.full(
                (), float(examples), dtype=torch.float64, device=loss_sum.device
            ),
        ]
    )
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    loss_total, correct_total, global_examples = stats.tolist()
    denom = max(global_examples, 1)
    return loss_total / denom, correct_total / denom


def _tensor_diagnostics(tensor: torch.Tensor) -> dict[str, float | int]:
    """Return compact, JSON-safe finite-value diagnostics for a tensor."""
    values = tensor.detach().float()
    finite = torch.isfinite(values)
    finite_count = int(finite.sum().item())
    diagnostics: dict[str, float | int] = {
        "numel": values.numel(),
        "finite_count": finite_count,
    }
    if finite_count == 0:
        return diagnostics
    finite_values = values[finite]
    diagnostics.update(
        {
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "rms": float(finite_values.square().mean().sqrt().item()),
        }
    )
    if finite_count == values.numel() and values.ndim >= 2:
        rows = values.reshape(-1, values.shape[-1])
        diagnostics.update(
            {
                "within_row_std_mean": float(
                    rows.std(dim=-1, unbiased=False).mean().item()
                ),
                "row_mean_std": float(rows.mean(dim=-1).std(unbiased=False).item()),
                "across_row_feature_std_mean": float(
                    rows.std(dim=0, unbiased=False).mean().item()
                ),
            }
        )
    return diagnostics


class FSDPTrainer(Trainer):
    """
    FSDP-enabled trainer for large dendritic models.

    This trainer extends the standard trainer with FSDP support,
    allowing training of models that don't fit on a single GPU.
    """

    default_grad_clip_mode = "norm"

    # The FSDP loop does not use Trainer.train() or its LR scheduler.
    supports_lr_schedule = False
    # Options its own step loop does not apply (see STEP_OPTIONS); it reads
    # log_first_step_diagnostics itself.
    unapplied_options = frozenset({"enable_nan_checking", "profile_dataloader"})

    def __init__(
        self,
        rank: int,
        world_size: int,
        fsdp_config: Optional[dict] = None,
        checkpoint_dir: Optional[str] = None,
        save_every_n_epochs: Optional[int] = None,
        optimizer_config: Optional[dict] = None,
        analysis_manager: Optional[object] = None,
        log_first_step_diagnostics: bool = False,
        cache_frozen_core_outputs: bool = False,
        cached_core_batch_size: int | None = None,
        **kwargs,
    ):
        """
        Initialize FSDP trainer.

        Args:
            rank: Global rank of current process
            world_size: Total number of processes
            fsdp_config: FSDP configuration
            checkpoint_dir: Directory to save checkpoints
            save_every_n_epochs: Save checkpoint every N epochs (None disables periodic checkpointing)
            optimizer_config: ``{"optimizer": training.main.optimizer,
                "param_groups": training.main.common.param_groups}`` (typed
                configs or mappings; missing entries take their defaults)
            **kwargs: Arguments passed to base Trainer
        """
        # Store configuration
        self.rank = rank
        self.world_size = world_size
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.fsdp_config = fsdp_config or get_fsdp_config()
        self.save_every_n_epochs = save_every_n_epochs

        # Set up checkpoint directory structure (consistent with standard trainer)
        if checkpoint_dir is not None:
            self.checkpoint_dir = os.path.join(checkpoint_dir, "checkpoints")
        else:
            self.checkpoint_dir = f"results/fsdp_run_{time.time()}/checkpoints"
        _check_optimizer_config_keys(optimizer_config)
        self.optimizer_config = optimizer_config or {}
        self.analysis_manager = analysis_manager
        self.log_first_step_diagnostics = bool(log_first_step_diagnostics)
        self.cache_frozen_core_outputs = bool(cache_frozen_core_outputs)
        if cached_core_batch_size is not None and int(cached_core_batch_size) <= 0:
            raise ValueError("cached_core_batch_size must be positive when provided")
        self.cached_core_batch_size = (
            int(cached_core_batch_size) if cached_core_batch_size is not None else None
        )
        self.first_step_diagnostics: dict[str, Any] | None = None
        self._decoder_input_diagnostics: dict[str, float | int] | None = None
        self._use_precomputed_core = False

        # Set device using local rank
        self.device = torch.device(f"cuda:{self.local_rank}")
        torch.cuda.set_device(self.local_rank)

        # Initialize parent with dummy optimizer (we'll create the real one after FSDP wrapping)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.01)
        if self.rank == 0:
            logger.info(f"Initializing FSDPTrainer with kwargs: {kwargs}")

        # Store checkpoint_dir before parent call (parent might override it)
        checkpoint_dir_backup = self.checkpoint_dir

        fsdp_kwargs = kwargs.copy()
        # Without a loss_function the run trains the typed configuration's
        # default loss.
        fsdp_kwargs.setdefault(
            "loss_function", field_default(CommonTrainerConfig, "loss_function")
        )
        super().__init__(optimizer=dummy_optimizer, **fsdp_kwargs)
        # Restore checkpoint_dir after parent call
        self.checkpoint_dir = checkpoint_dir_backup

        if self.rank == 0 and self.checkpointing:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # The registry loss (``loss_function``), evaluated on the outputs the
        # FSDP forward has already computed, as in single-process training.
        loss_type = type(self.loss_function)
        if loss_type.from_predictions is LossFunction.from_predictions:
            raise ValueError(
                f"FSDPTrainer evaluates the loss on precomputed outputs; "
                f"{self.loss_function._loss_name!r} needs the model itself"
            )
        if loss_type.temporal_event_loss or loss_type.sequence_prediction_loss:
            raise ValueError(
                f"{self.loss_function._loss_name!r} is a sequence loss of the "
                "recurrent trainer; FSDP trains per-sample outputs"
            )
        self.loss_fn = self.loss_function.from_predictions
        self._class_logit_loss = loss_type.supported_tasks == frozenset(
            {"classification"}
        )

        # Mixed Precision Training Support
        self.use_amp = kwargs.get("use_amp", False)
        if self.use_amp:
            try:
                # autocast and GradScaler already imported at top
                scaler_cls = ShardedGradScaler or GradScaler
                self.scaler = scaler_cls()
                if self.rank == 0:
                    logger.info("Mixed precision training enabled for FSDP")
            except ImportError:
                if self.rank == 0:
                    logger.warning(
                        "AMP not available, falling back to standard precision"
                    )
                self.use_amp = False

    def prepare_model(self, model: BaseModel) -> FullyShardedDataParallel:
        """Wrap ``model`` with FSDP and build its optimizer (``self.optimizer``).

        The wrapped model is kept as ``self.fsdp_model`` for checkpoint saving
        and full-state gathering after training.
        """
        model = wrap_model_with_fsdp(model, self.fsdp_config)
        # sync_module_states broadcast rank 0's buffers; the calibrated fixed_b
        # floats the forward reads must follow them.
        refresh_fixed_b_from_buffers(model)
        # With CPU offloading FSDP manages device placement itself.
        if not self.fsdp_config.get("cpu_offload"):
            model = model.to(self.device)
        self.fsdp_model = model
        self.optimizer = self.create_optimizer(model)
        return model

    def release_model(self) -> None:
        """Drop the trainer's references to the FSDP model and its optimizer."""
        self.fsdp_model = None
        self.optimizer = None

    def create_optimizer(self, model: BaseModel) -> torch.optim.Optimizer:
        """Create the optimizer of the FSDP-wrapped model.

        ``optimizer_config["optimizer"]`` is ``training.main.optimizer`` and
        ``optimizer_config["param_groups"]`` is
        ``training.main.common.param_groups``; the groups are those of
        single-process training (:func:`fsdp_param_groups`).
        """
        optimizer_config = self.optimizer_config or {}
        _check_optimizer_config_keys(optimizer_config)
        settings = _optimizer_settings(optimizer_config)
        param_groups = fsdp_param_groups(
            model, _param_groups_settings(optimizer_config)
        )
        optimizer = build_optimizer(param_groups, settings)
        _reject_sharded_muonh(optimizer, model)
        param_counts = [
            sum(param.numel() for param in group["params"])
            for group in optimizer.param_groups
        ]
        if dist.is_available() and dist.is_initialized():
            # NCCL reduces only CUDA tensors; under CPU offload the parameters
            # (and so any device read from them) stay on the CPU.
            count_device = (
                self.device if dist.get_backend() == "nccl" else torch.device("cpu")
            )
            global_counts = torch.tensor(
                param_counts,
                dtype=torch.int64,
                device=count_device,
            )
            dist.all_reduce(global_counts, op=dist.ReduceOp.SUM)
            param_counts = [int(value) for value in global_counts.tolist()]
        if getattr(self, "rank", 0) == 0:
            group_summary = ", ".join(
                (
                    f"{group['name']}: lr={group['lr']:.3g}, "
                    f"global_params={param_count:,}"
                )
                for group, param_count in zip(
                    optimizer.param_groups,
                    param_counts,
                    strict=True,
                )
            )
            logger.info("FSDP optimizer groups: %s", group_summary)
        return optimizer

    def _cache_core_output_loader(
        self,
        model: BaseModel,
        loader: DataLoader,
        *,
        shuffle: bool,
        seed_offset: int,
    ) -> DataLoader:
        """Materialize one frozen-core output for each sample on this rank."""
        decoder = getattr(model, "decoder_network", None)
        if decoder is None:
            raise ValueError("cache_frozen_core_outputs requires model.decoder_network")

        cached_features: list[torch.Tensor] = []
        cached_targets: list[torch.Tensor] = []

        def capture_core_output(_module, args) -> None:
            if not args or not torch.is_tensor(args[0]):
                raise RuntimeError("Decoder did not receive a tensor core output")
            cached_features.append(args[0].detach().to(device="cpu", copy=True))

        hook = decoder.register_forward_pre_hook(capture_core_output)
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for batch in loader:
                    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
                        raise ValueError(
                            "Cached frozen-core training requires (input, target) batches"
                        )
                    x_batch = batch[0].to(
                        self.device,
                        non_blocking=self._loader_uses_non_blocking_transfer(loader),
                    )
                    seq_lengths = _sequence_lengths_from_batch(
                        batch,
                        x_batch=x_batch,
                        device=self.device,
                    )
                    captures_before = len(cached_features)
                    model(x_batch, seq_lengths=seq_lengths)
                    if len(cached_features) != captures_before + 1:
                        raise RuntimeError(
                            "Expected exactly one decoder input per cached batch"
                        )
                    cached_targets.append(batch[1].detach().to(device="cpu", copy=True))
        finally:
            hook.remove()
            model.train(was_training)

        if not cached_features:
            raise ValueError("Cannot cache frozen-core outputs from an empty loader")
        features = torch.cat(cached_features, dim=0)
        targets = torch.cat(cached_targets, dim=0)
        if features.shape[0] != targets.shape[0]:
            raise RuntimeError(
                "Cached core-output and target counts differ: "
                f"{features.shape[0]} != {targets.shape[0]}"
            )

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.rank + seed_offset)
        cached_batch_size = self.cached_core_batch_size or loader.batch_size
        if cached_batch_size is None:
            raise ValueError(
                "cached_core_batch_size is required when the source loader "
                "does not define batch_size"
            )
        cached_loader = DataLoader(
            TensorDataset(features, targets),
            batch_size=cached_batch_size,
            shuffle=shuffle,
            drop_last=loader.drop_last,
            num_workers=0,
            pin_memory=loader.pin_memory,
            generator=generator,
        )
        if isinstance(loader.sampler, PaddedDistributedEvalSampler):
            cached_loader._dendritic_real_count = len(loader.sampler.real_indices)
        return cached_loader

    def _cache_frozen_core_loaders(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: DataLoader | None,
    ) -> tuple[DataLoader, DataLoader | None]:
        """Replace raw loaders with rank-local deterministic core features."""
        cached_train_loader = self._cache_core_output_loader(
            model,
            train_loader,
            shuffle=True,
            seed_offset=17,
        )
        cached_valid_loader = (
            self._cache_core_output_loader(
                model,
                valid_loader,
                shuffle=False,
                seed_offset=29,
            )
            if valid_loader is not None
            else None
        )
        local_counts = torch.tensor(
            [
                len(cached_train_loader.dataset),
                (
                    len(cached_valid_loader.dataset)
                    if cached_valid_loader is not None
                    else 0
                ),
            ],
            dtype=torch.int64,
            device=self.device,
        )
        dist.all_reduce(local_counts, op=dist.ReduceOp.SUM)
        if self.rank == 0:
            feature_shape = cached_train_loader.dataset.tensors[0].shape[1:]
            logger.info(
                "Cached frozen-core outputs: train=%s, valid=%s, feature_shape=%s",
                f"{int(local_counts[0].item()):,}",
                f"{int(local_counts[1].item()):,}",
                tuple(feature_shape),
            )
        self._use_precomputed_core = True
        return cached_train_loader, cached_valid_loader

    def train(
        self,
        model: BaseModel,
        train_loader: DataLoader | Dataset,
        valid_loader: Optional[DataLoader | Dataset] = None,
    ) -> dict[str, Any]:
        """
        Train model using FSDP with pre-created DataLoaders.

        Args:
            model: Model to train (will be wrapped with FSDP)
            train_loader: Training DataLoader (or training Dataset for auto-wrapping)
            valid_loader: Validation DataLoader (or validation Dataset for auto-wrapping)

        Returns:
            Training results dictionary
        """
        trainer_started = time.perf_counter()
        if isinstance(train_loader, DataLoader):
            if valid_loader is not None and not isinstance(valid_loader, DataLoader):
                raise TypeError(
                    "If train_loader is a DataLoader, valid_loader must also be a DataLoader or None."
                )
        else:
            train_loader, valid_loader = self._dataloaders(train_loader, valid_loader)

        # Include FSDP wrapping, state synchronization, training, validation,
        # and checkpoint work in one per-rank peak-memory measurement.
        track_cuda_memory = torch.cuda.is_available()
        if track_cuda_memory:
            torch.cuda.reset_peak_memory_stats(self.device)

        model = self.prepare_model(model)

        diagnostic_hook = None
        if self.log_first_step_diagnostics and hasattr(model, "decoder_network"):

            def capture_decoder_input(_module, args) -> None:
                if self._decoder_input_diagnostics is not None or not args:
                    return
                decoder_input = args[0]
                if torch.is_tensor(decoder_input):
                    self._decoder_input_diagnostics = _tensor_diagnostics(decoder_input)

            diagnostic_hook = model.decoder_network.register_forward_pre_hook(
                capture_decoder_input
            )

        fsdp_setup_peak_memory = None
        if track_cuda_memory:
            torch.cuda.synchronize(self.device)
            fsdp_setup_peak_memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(self.device),
                    torch.cuda.max_memory_reserved(self.device),
                ],
                dtype=torch.int64,
                device=self.device,
            )
            dist.all_reduce(fsdp_setup_peak_memory, op=dist.ReduceOp.MAX)
            # Measure the recurrent train/validation phase independently while
            # retaining the setup peak above for an overall maximum.
            torch.cuda.reset_peak_memory_stats(self.device)
        fsdp_setup_seconds = time.perf_counter() - trainer_started

        frozen_core_cache_seconds = None
        if self.cache_frozen_core_outputs:
            if not bool(getattr(model, "_freeze_core_forward", False)):
                raise ValueError(
                    "cache_frozen_core_outputs requires the frozen-core "
                    "trainability policy"
                )
            if track_cuda_memory:
                torch.cuda.synchronize(self.device)
            cache_started = time.perf_counter()
            train_loader, valid_loader = self._cache_frozen_core_loaders(
                model,
                train_loader,
                valid_loader,
            )
            if track_cuda_memory:
                torch.cuda.synchronize(self.device)
            frozen_core_cache_seconds = time.perf_counter() - cache_started

        # Training loop
        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []
        epoch_times_seconds = []
        best_valid_loss = float("inf")
        patience_counter = 0
        self.last_completed_epoch = -1

        if self.rank == 0:
            logger.info(f"Starting training for {self.epochs} epochs")

        training_loop_started = time.perf_counter()
        for epoch in range(self.epochs):
            self.last_completed_epoch = epoch
            if self.rank == 0:
                logger.info(f"Starting epoch {epoch}")
            epoch_start = time.perf_counter()

            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            # Training
            train_loss, train_acc = self._epoch_train_fsdp(model, train_loader)
            train_losses.append(train_loss)
            train_accs.append(train_acc)

            # Validation
            valid_loss = None
            valid_acc = None
            if valid_loader is not None:
                if hasattr(valid_loader.sampler, "set_epoch"):
                    valid_loader.sampler.set_epoch(epoch)
                valid_loss, valid_acc = self._epoch_valid_fsdp(model, valid_loader)
                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)

                improved = valid_loss < best_valid_loss
                if improved:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    if self.early_stopping and self.checkpointing:
                        save_fsdp_checkpoint(
                            model,
                            self.optimizer,
                            epoch,
                            os.path.join(self.checkpoint_dir, "best_model.pt"),
                            rank=self.rank,
                        )
                elif self.early_stopping:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.rank == 0:
                            logger.info(f"Early stopping at epoch {epoch}")
                        break

            # Epoch-level analysis is intentionally skipped for FSDP.
            # FSDP shards model parameters across ranks; running analysis
            # on the sharded model produces incorrect results.  Final
            # analysis runs on rank 0 after gathering the full state dict
            # (see train_experiments_fsdp.py).

            # Logging
            epoch_time = time.perf_counter() - epoch_start
            epoch_times_seconds.append(epoch_time)
            if self.rank == 0 and epoch % self.print_every == 0:
                msg = f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}"
                if valid_loss is not None and valid_acc is not None:
                    msg += (
                        f" - Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_acc:.4f}"
                    )
                msg += f" - Time: {epoch_time:.2f}s"
                logger.info(msg)

            # Checkpointing
            if (
                self.checkpointing
                and self.save_every_n_epochs is not None
                and self.save_every_n_epochs > 0
                and (epoch + 1) % self.save_every_n_epochs == 0
                and (epoch + 1) < self.epochs
            ):
                save_fsdp_checkpoint(
                    model,
                    self.optimizer,
                    epoch,
                    os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt"),
                    rank=self.rank,
                )
        training_loop_seconds = time.perf_counter() - training_loop_started

        # Final resumable checkpoint (model, optimizer and epoch). The entry
        # point saves the plain final weights separately as final_model.pt.
        self.final_checkpoint_path = None
        if self.checkpointing:
            self.final_checkpoint_path = os.path.join(
                self.checkpoint_dir, "final_checkpoint.pt"
            )
            save_fsdp_checkpoint(
                model,
                self.optimizer,
                self.last_completed_epoch,
                self.final_checkpoint_path,
                rank=self.rank,
            )

        # Generate plots (only on rank 0)
        if self.rank == 0:
            self._loss_plotting(train_losses, valid_losses)
            # Removed redundant accuracy plotting - handled by performance option

        results = {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "best_valid_loss": best_valid_loss,
            "train_accs": train_accs,
            "valid_accs": valid_accs,
            "final_train_loss": train_losses[-1] if train_losses else None,
            "final_valid_loss": valid_losses[-1] if valid_losses else None,
            "fsdp_setup_seconds": fsdp_setup_seconds,
            "frozen_core_cache_seconds": frozen_core_cache_seconds,
            "epoch_times_seconds": epoch_times_seconds,
            "training_loop_seconds": training_loop_seconds,
            "trainer_total_seconds": time.perf_counter() - trainer_started,
        }
        if diagnostic_hook is not None:
            diagnostic_hook.remove()
        if self.first_step_diagnostics is not None:
            results["first_step_diagnostics"] = self.first_step_diagnostics
        if track_cuda_memory:
            torch.cuda.synchronize(self.device)
            training_peak_memory = torch.tensor(
                [
                    torch.cuda.max_memory_allocated(self.device),
                    torch.cuda.max_memory_reserved(self.device),
                ],
                dtype=torch.int64,
                device=self.device,
            )
            dist.all_reduce(training_peak_memory, op=dist.ReduceOp.MAX)
            assert fsdp_setup_peak_memory is not None
            peak_memory = torch.maximum(
                fsdp_setup_peak_memory,
                training_peak_memory,
            )
            results["fsdp_setup_peak_cuda_allocated_bytes"] = int(
                fsdp_setup_peak_memory[0].item()
            )
            results["fsdp_setup_peak_cuda_reserved_bytes"] = int(
                fsdp_setup_peak_memory[1].item()
            )
            results["training_peak_cuda_allocated_bytes"] = int(
                training_peak_memory[0].item()
            )
            results["training_peak_cuda_reserved_bytes"] = int(
                training_peak_memory[1].item()
            )
            results["peak_cuda_allocated_bytes"] = int(peak_memory[0].item())
            results["peak_cuda_reserved_bytes"] = int(peak_memory[1].item())
            if self.rank == 0:
                gib = 1024**3
                logger.info(
                    "Peak CUDA memory across ranks: FSDP setup "
                    "allocated=%.3f/reserved=%.3f GiB; train+validation "
                    "allocated=%.3f/reserved=%.3f GiB; overall "
                    "allocated=%.3f/reserved=%.3f GiB",
                    results["fsdp_setup_peak_cuda_allocated_bytes"] / gib,
                    results["fsdp_setup_peak_cuda_reserved_bytes"] / gib,
                    results["training_peak_cuda_allocated_bytes"] / gib,
                    results["training_peak_cuda_reserved_bytes"] / gib,
                    results["peak_cuda_allocated_bytes"] / gib,
                    results["peak_cuda_reserved_bytes"] / gib,
                )

        return results

    def _epoch_train_fsdp(
        self, model: BaseModel, train_loader: DataLoader
    ) -> tuple[float, float]:
        """Training epoch with FSDP."""
        model.train()
        # Sample-weighted loss and correct counts stay on the device (float64)
        # and are read once per epoch.
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        total_correct = torch.zeros((), dtype=torch.float64, device=self.device)
        total_examples = 0

        non_blocking = self._loader_uses_non_blocking_transfer(train_loader)
        for _batch_idx, batch in enumerate(train_loader):
            x_batch = batch[0].to(self.device, non_blocking=non_blocking)
            y_batch = batch[1].to(self.device, non_blocking=non_blocking)
            seq_lengths = _sequence_lengths_from_batch(
                batch,
                x_batch=x_batch,
                device=self.device,
            )

            if self.use_amp:
                # Mixed precision forward pass
                with autocast("cuda"):
                    outputs = model(
                        x_batch,
                        seq_lengths=seq_lengths,
                        precomputed_core=self._use_precomputed_core,
                    )
                    loss = self._batch_loss(outputs, y_batch)

                # Backward pass with scaling
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping (0/None disables it; a zero max-norm would
                # zero every gradient)
                if self.grad_clip_value:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = self._clip_grad_norm_fsdp(model)
                else:
                    grad_norm = None

                # Optimizer step with scaling
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                outputs = model(
                    x_batch,
                    seq_lengths=seq_lengths,
                    precomputed_core=self._use_precomputed_core,
                )
                loss = self._batch_loss(outputs, y_batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping (0/None disables it)
                if self.grad_clip_value:
                    grad_norm = self._clip_grad_norm_fsdp(model)
                else:
                    grad_norm = None

                # Optimizer step
                self.optimizer.step()

            if self.log_first_step_diagnostics and self.first_step_diagnostics is None:
                self.first_step_diagnostics = {
                    "decoder_input": self._decoder_input_diagnostics,
                    "logits": _tensor_diagnostics(outputs),
                    "loss": float(loss.detach().item()),
                    "grad_norm_before_clip": (
                        float(grad_norm.detach().item())
                        if torch.is_tensor(grad_norm)
                        else (float(grad_norm) if grad_norm is not None else None)
                    ),
                }
                if self.rank == 0:
                    logger.info(
                        "First-step diagnostics: %s",
                        self.first_step_diagnostics,
                    )

            bs = y_batch.size(0)
            loss_sum += loss.detach().double() * bs
            total_correct += _correct_predictions(outputs.detach(), y_batch)
            total_examples += bs

        return _mean_over_ranks(loss_sum, total_correct, total_examples)

    def _batch_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self._class_logit_loss and outputs.dim() != 2:
            raise ValueError(
                f"{self.loss_function._loss_name!r} under FSDP needs [batch, "
                f"classes] logits, got outputs of shape {tuple(outputs.shape)}; "
                "sequence outputs need the recurrent trainer"
            )
        return self.loss_fn(outputs, targets)

    def _clip_grad_norm_fsdp(self, model: BaseModel):
        """Clip gradients; return the global norm before clipping (norm mode)."""
        if self.grad_clip_mode == "value":
            # Element-wise clamping is shard-local, so it is exact under FSDP.
            clip_gradients(model.parameters(), self.grad_clip_value, "value")
            return None
        # Norm clipping needs the global norm, which FSDP reduces across shards.
        if hasattr(model, "clip_grad_norm_"):
            return model.clip_grad_norm_(self.grad_clip_value)
        return torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_value)

    def _epoch_valid_fsdp(
        self, model: BaseModel, valid_loader: DataLoader
    ) -> tuple[float, float]:
        """Validation epoch with FSDP.

        All ranks must participate (FSDP needs all-gather during forward).
        We accumulate sample-weighted loss, correct count, and total
        examples, then all-reduce with SUM for an exact global average.

        When used with ``PaddedDistributedEvalSampler``, padding indices
        (-1) may appear in the last batch.  These are sentinel values that
        index into real data (Python wraps -1 to the last element), so the
        forward pass succeeds on every rank, but we exclude them from the
        stats by tracking the known number of real samples.
        """
        model.eval()
        weighted_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        total_correct = torch.zeros((), dtype=torch.float64, device=self.device)
        total_examples = 0

        # Determine how many real (non-padding) samples this rank has.
        sampler = valid_loader.sampler
        cached_real_count = getattr(
            valid_loader,
            "_dendritic_real_count",
            None,
        )
        if cached_real_count is not None:
            real_count = int(cached_real_count)
        elif isinstance(sampler, PaddedDistributedEvalSampler):
            real_count = len(sampler.real_indices)
        else:
            real_count = None  # unknown — count all

        seen = 0
        with torch.no_grad():
            non_blocking = self._loader_uses_non_blocking_transfer(valid_loader)
            for batch in valid_loader:
                x_batch = batch[0].to(self.device, non_blocking=non_blocking)
                y_batch = batch[1].to(self.device, non_blocking=non_blocking)
                seq_lengths = _sequence_lengths_from_batch(
                    batch,
                    x_batch=x_batch,
                    device=self.device,
                )
                bs = y_batch.size(0)

                outputs = model(
                    x_batch,
                    seq_lengths=seq_lengths,
                    precomputed_core=self._use_precomputed_core,
                )
                loss = self._batch_loss(outputs, y_batch)

                if real_count is not None:
                    # Only count the real (non-padding) samples.
                    real_in_batch = min(bs, real_count - seen)
                    if real_in_batch <= 0:
                        # Entire batch is padding — still ran forward for
                        # FSDP all-gather, but skip stats.
                        seen += bs
                        continue
                    if real_in_batch < bs:
                        # Partial padding in this batch.
                        outputs = outputs[:real_in_batch]
                        y_batch = y_batch[:real_in_batch]
                        loss = self._batch_loss(outputs, y_batch)
                        bs = real_in_batch
                    seen += bs

                weighted_loss_sum += loss.detach().double() * bs
                total_correct += _correct_predictions(outputs, y_batch)
                total_examples += bs

        return _mean_over_ranks(weighted_loss_sum, total_correct, total_examples)
