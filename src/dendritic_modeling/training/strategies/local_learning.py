"""
Local learning training strategies.

This module implements multi-factor local learning rules for dendritic networks.

It provides biologically-inspired 3-/4-/5-factor Hebbian-like rules that use
locally available signals (pre-synaptic input, compartment voltages) modulated
by a broadcast soma error and optional morphology/information factors.
"""

import logging
import time
from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

from dendritic_modeling.config.local_learning import (
    build_local_rule_config,
    coerce_legacy_local_rule_config,
    section_to_dict,
)
from dendritic_modeling.config.training import CommonTrainerConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.dense_to_sparse import (
    DenseToSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_release import (
    IndexedReleaseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import (
    VarianceTopKLinear,
)
from dendritic_modeling.networks.structure import inhibitory_branch_layer_ids
from dendritic_modeling.networks.utils.sparse_maintenance import frozen_sparse_state
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_mixin import (
    LocalLearningBroadcastMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_epoch import (
    _accumulate_backprop_gradients as _accumulate_backprop_gradients,
    _accumulate_backprop_update_gradients,
    _accumulate_local_logging_loss,
    _apply_decoder_local_gradients as _apply_decoder_local_gradients,
    _apply_local_decoder_update,
    _apply_path_propagation_factor as _apply_path_propagation_factor,
    _average_distributed_gradients,
    _average_local_epoch_losses,
    _clear_topk_mask_cache as _clear_topk_mask_cache,
    _clip_model_gradients,
    _collect_backprop_param_groups as _collect_backprop_param_groups,
    _collect_topk_modules as _collect_topk_modules,
    _enable_topk_mask_cache as _enable_topk_mask_cache,
    _forward_with_local_recorders,
    _has_dendritic_branch_layers,
    _prepare_hsic_target,
    _resolve_epoch_update_modes as _resolve_epoch_update_modes,
    _resolve_local_epoch_update_modes,
    _resolve_reactivation_update_enabled,
    _resolve_stdp_error_signals as _resolve_stdp_error_signals,
    _zero_optimizer_gradients,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_gradients_mixin import (
    LocalLearningGradientMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_hsic_mixin import (
    LocalLearningHSICMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_modulators import (
    LocalLearningModulatorMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_morphology_mixin import (
    LocalLearningMorphologyMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_recorders import (
    LocalLearningRecorderMixin,
    reject_mismatched_dynamics_mode,
    reject_unsupported_forward_paths,
    reject_unsupported_synapses,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_rule_mixin import (
    LocalLearningRuleMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_signals import (
    LocalLearningSignalMixin,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_state import (
    _LayerStats,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_stdp_mixin import (
    LocalLearningSTDPMixin,
)
from dendritic_modeling.training.strategies.standard import Trainer, _step_seconds
from dendritic_modeling.training.strategies.trainability import (
    TrainabilitySnapshot,
    restore_requires_grad,
    snapshot_requires_grad,
)
from dendritic_modeling.utils.hooks import iter_named_modules_of_type

logger = logging.getLogger(__name__)

_COMMON_GRAD_CLIP_DEFAULT = CommonTrainerConfig.__dataclass_fields__[
    "grad_clip_value"
].default

# Sparse layers whose training forwards change their selection state.
_FORWARD_STATE_SPARSE_LAYERS = (
    DenseToSparseLinear,
    IndexedReleaseLinear,
    VarianceTopKLinear,
)


def _same_clip_value(first: Any, second: Any) -> bool:
    """Compare clip settings, treating ``None``/non-positive as disabled."""
    first_disabled = not first or float(first) <= 0
    second_disabled = not second or float(second) <= 0
    if first_disabled or second_disabled:
        return first_disabled and second_disabled
    return float(first) == float(second)


def inactive_auxiliary_terms(local_cfg: Any) -> list[str]:
    """The enabled auxiliary terms of ``local_cfg`` whose weights make them no-ops.

    HSIC scales its gradients by ``hsic.weight`` (and its self and target parts
    by ``self_weight`` and ``target_weight``), so a zero weight adds nothing.
    Inhibitory and voltage homeostasis apply only a positive ``weight``, and
    gate homeostasis only a positive ``center_weight`` or ``saturation_weight``.
    """
    inactive = []
    hsic = getattr(local_cfg, "hsic", None)
    if hsic is not None and getattr(hsic, "enabled", False):
        if float(hsic.weight) == 0.0:
            inactive.append("hsic (weight=0)")
        elif float(hsic.self_weight) == 0.0 and float(hsic.target_weight) == 0.0:
            inactive.append("hsic (self_weight=0 and target_weight=0)")
    for name in ("inhibitory_homeostasis", "voltage_homeostasis"):
        cfg = getattr(local_cfg, name, None)
        if cfg is not None and getattr(cfg, "enabled", False):
            if float(cfg.weight) <= 0.0:
                inactive.append(f"{name} (weight={float(cfg.weight):g})")
    gate = getattr(local_cfg, "gate_homeostasis", None)
    if gate is not None and getattr(gate, "enabled", False):
        center, saturation = float(gate.center_weight), float(gate.saturation_weight)
        if center <= 0.0 and saturation <= 0.0:
            inactive.append(
                f"gate_homeostasis (center_weight={center:g}, "
                f"saturation_weight={saturation:g})"
            )
    return inactive


class LocalCreditAssignment(
    LocalLearningSignalMixin,
    LocalLearningRecorderMixin,
    LocalLearningSTDPMixin,
    LocalLearningBroadcastMixin,
    LocalLearningHSICMixin,
    LocalLearningRuleMixin,
    LocalLearningGradientMixin,
    LocalLearningMorphologyMixin,
    LocalLearningModulatorMixin,
    Trainer,
):
    """
    Local multi-factor learning strategy for dendritic networks.

    Implements 3-/4-/5-factor Hebbian-like local rules for both synaptic
    conductances (TopKLinear) and dendritic branch conductances (BlockLinear)
    using locally available signals plus a broadcast error from the soma.
    """

    # The local-rule loop bypasses Trainer.train() and its LR scheduler.
    supports_lr_schedule = False
    # Options its own step loop does not apply (see STEP_OPTIONS).
    unapplied_options = Trainer.unapplied_options | frozenset(
        {"use_amp", "enable_nan_checking"}
    )
    # Local gradients are assigned directly and averaged explicitly.
    ddp_reduces_gradients = False

    def __init__(self, *args, **kwargs):
        config_dict = kwargs.pop("local_rule_config", {}) or {}
        # Keep the common clip value only when it was passed, so the local-rule
        # path can reject a value it would not apply.
        self._configured_grad_clip_value = kwargs.get("grad_clip_value")
        self._configured_grad_clip_mode = kwargs.get("grad_clip_mode")
        super().__init__(*args, **kwargs)
        # Build LocalRuleConfig with proper nested config handling
        self.local_cfg = self._build_local_rule_config(config_dict)
        inactive = inactive_auxiliary_terms(self.local_cfg)
        if inactive and self._is_main_process:
            logger.warning(
                "local_ca: the auxiliary terms %s are enabled but their weights "
                "make them change no update; set a positive weight or "
                "enabled: false (training.main.learning_strategy_config).",
                "; ".join(inactive),
            )
        self._layer_stats: dict[int, _LayerStats] = {}
        self._decoder_cache: dict[str, Any] = {}
        self._additive_running_var: dict[int, torch.Tensor] = {}
        self._broadcast_cache: dict[tuple[Any, ...], Any] = {}
        self._stdp_traces: dict[int, dict[str, torch.Tensor]] = {}
        self._warned_decoder_soma_fallback = False
        # Trainable parameters that receive no gradient; set at the first step.
        self.uncovered_parameters: list[str] | None = None
        self.filename_prefix = "local_learning_"
        self.logger_info_prefix = "[Local] "

    @staticmethod
    def _section_to_dict(section: Any) -> dict[str, Any]:
        """Convert a config section object to a plain dict."""
        return section_to_dict(section)

    def _coerce_legacy_local_rule_config(
        self, config_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Normalize legacy flat local learning config to nested LocalRuleConfig layout."""
        return coerce_legacy_local_rule_config(config_dict)

    def _build_local_rule_config(self, config_dict: dict):
        """Build LocalRuleConfig from dict with proper nested config handling."""
        return build_local_rule_config(config_dict)

    def _should_update_local_record(self, rec: dict[str, Any]) -> bool:
        """Return whether LocalCA should write gradients for a recorded branch.

        This only changes explicit inhibitory-cell DendriNet populations. I-to-E
        synapses that live on excitatory dendrites remain ordinary excitatory
        population records and are still updated by LocalCA.
        """
        if rec.get("population") != "explicit_inhibitory":
            return True

        return self._explicit_inhibitory_mode() == "local_ca"

    def _report_uncovered_parameters(self, model: BaseModel) -> None:
        """Record and warn once about trainable parameters left without gradient."""
        self.uncovered_parameters = [
            name
            for name, param in self._unwrap_model(model).named_parameters()
            if param.requires_grad and param.grad is None
        ]
        if self.uncovered_parameters:
            names = self.uncovered_parameters
            logger.warning(
                "%s%d trainable parameters receive no gradient from the local "
                "rules and the configured update modes and stay fixed: %s",
                self.logger_info_prefix,
                len(names),
                ", ".join(names[:10]) + (" ..." if len(names) > 10 else ""),
            )

    def _explicit_inhibitory_mode(self) -> str:
        mode = getattr(self.local_cfg, "explicit_inhibitory_update_mode", None)
        if mode is None:
            # Compatibility for older LocalRuleConfig-like objects in tests.
            update = getattr(self.local_cfg, "update_explicit_inhibitory_cells", True)
            return "local_ca" if update else "freeze"
        return str(mode).lower()

    # Override: do not run extra pretrain/final phases here
    def _pretrain(self, *args, **kwargs):  # type: ignore[override]
        return [], []

    def _check_local_model(self, model: BaseModel) -> None:
        """Reject a model without branch layers or with unapplied options."""
        if not _has_dendritic_branch_layers(model):
            raise ValueError(
                "strategy 'local_ca' updates dendritic branch layers with local "
                "rules, but this model has none; use 'standard' (backpropagation) "
                "or 'dfa' for this core"
            )
        self._validate_local_rule_options(model)

    def _validate_local_rule_options(self, model: BaseModel) -> None:
        """Reject common options that the local-rule loop does not apply.

        Local rules assign ``param.grad`` directly: backward-hook gradient
        scaling and the auxiliary regularization loss never act on them, and
        clipping uses ``learning_strategy_config.clip_grad_value``.
        """
        self._reject_unapplied_gradient_scaling(model)
        reject_unsupported_forward_paths(
            self._unwrap_model(model),
            allow=self.local_cfg.allow_unmodeled_forward_features,
        )
        three_factor = self.local_cfg.three_factor
        reject_mismatched_dynamics_mode(
            self._unwrap_model(model),
            three_factor.dynamics_mode,
            allow_mismatch=three_factor.allow_mismatched_dynamics_mode,
        )
        normalized = [
            name
            for name, layer in iter_named_modules_of_type(
                self._unwrap_model(model), DendriticBranchLayer
            )
            if getattr(layer, "use_additive_normalization", False)
        ]
        if normalized and three_factor.additive_gain_mode != "normalization_jacobian":
            logger.warning(
                "%slocal_ca omits the derivative of use_additive_normalization at "
                "%s; three_factor.additive_gain_mode: normalization_jacobian "
                "supplies it (exact gradients also need exact delivered errors)",
                self.logger_info_prefix,
                ", ".join(normalized[:4]) + (" ..." if len(normalized) > 4 else ""),
            )
        reject_unsupported_synapses(
            self._unwrap_model(model),
            stdp_pathways={
                pathway
                for pathway in ("exc", "inh", "rec_exc", "rec_inh")
                if self._stdp_enabled() and self._stdp_pathway_enabled(pathway)
            },
        )
        if self._explicit_inhibitory_mode() == "freeze" and not (
            inhibitory_branch_layer_ids(self._unwrap_model(model))
        ):
            raise ValueError(
                "explicit_inhibitory_update_mode 'freeze' has no inhibitory "
                "population to freeze in this model"
            )
        if self.regularization_manager.has_regularization():
            raise ValueError(
                "local_ca assigns local-rule gradients and does not apply "
                "training.main.regularization; set its l1/l2/E-I weights to 0 "
                "or use a strategy that applies it (standard, dfa)."
            )
        if self._configured_grad_clip_mode not in (None, "value"):
            raise ValueError(
                "local_ca clips local-rule gradients element-wise with "
                "training.main.learning_strategy_config.clip_grad_value, so "
                "training.main.common.grad_clip_mode="
                f"{self._configured_grad_clip_mode!r} would not be applied"
            )
        configured = self._configured_grad_clip_value
        local_clip = getattr(self.local_cfg, "clip_grad_value", None)
        if configured is None or _same_clip_value(configured, local_clip):
            return
        if _same_clip_value(configured, _COMMON_GRAD_CLIP_DEFAULT):
            # The typed common default: local_ca documents that it clips with
            # learning_strategy_config.clip_grad_value instead.
            return
        raise ValueError(
            "local_ca clips local-rule gradients with "
            f"training.main.learning_strategy_config.clip_grad_value={local_clip}; "
            f"training.main.common.grad_clip_value={configured} would not be "
            "applied. Set learning_strategy_config.clip_grad_value instead, or "
            f"leave common.grad_clip_value at its default "
            f"{_COMMON_GRAD_CLIP_DEFAULT}."
        )

    def _freeze_core(self, model: BaseModel) -> TrainabilitySnapshot | None:
        """Hold the core fixed under ``core_update_mode: none``.

        The training step already skips the local-rule gradients. Here the
        core's parameters stop requiring gradients, which also keeps sparse
        maintenance (decay, boosting, rewiring, topology controllers) off
        them, and ``train`` runs the epochs under ``frozen_sparse_state`` so
        that Top-K statistics, release tracking and dense-to-sparse schedules
        do not change. Settings that would still change the core raise.
        Returns the trainability to restore afterwards, or ``None``.
        """
        if getattr(self.local_cfg, "core_update_mode", "local") != "none":
            return None
        component = self._unwrap_model(model)
        core = getattr(component, "core_network", None)
        if core is None:
            raise ValueError(
                "core_update_mode 'none' holds the core network fixed, but this "
                "model has no core_network"
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                raise ValueError(
                    "core_update_mode 'none' supports single-process training only"
                )
        every = int(getattr(self, "recalibrate_reactivation_every", 0) or 0)
        if every > 0:
            raise ValueError(
                "core_update_mode 'none' keeps the core's gates fixed, but "
                f"recalibrate_reactivation_every={every} recalibrates them during "
                "training; set it to 0"
            )
        batch_norms = [
            name
            for name, module in core.named_modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        ]
        if batch_norms:
            raise ValueError(
                "core_update_mode 'none' keeps the core fixed, but these batch "
                "normalization layers update running statistics in training: "
                + ", ".join(batch_norms)
            )
        core_modules = {id(module) for module in core.modules()}
        outside = [
            name
            for name, module in component.named_modules()
            if isinstance(module, _FORWARD_STATE_SPARSE_LAYERS)
            and id(module) not in core_modules
        ]
        if outside:
            raise ValueError(
                "core_update_mode 'none' freezes the sparse state that training "
                "forwards update (Top-K statistics, release tracking, dense-to-"
                "sparse schedules), which would also freeze these layers outside "
                "the core: " + ", ".join(outside)
            )
        snapshot = snapshot_requires_grad(component)
        for parameter in core.parameters():
            parameter.requires_grad = False
        return snapshot

    def _resolve_local_reactivation_update_flag(
        self, update_reactivation: bool
    ) -> bool:
        """Disable local reactivation updates when the shared trainer mode does not use learned b,m."""
        return _resolve_reactivation_update_enabled(
            update_reactivation,
            self._normalize_reactivation_update_mode(),
        )

    # ----------------------- Core training overrides -----------------------
    def assign_gradients(
        self,
        model: BaseModel,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        *,
        epoch: int | None = None,
    ) -> torch.Tensor:
        """Assign one batch's update to the parameters' ``.grad``; no step.

        Runs the forward, local signals and gradient assignment of a training
        step and stops where the step calls ``optimizer.step()``: the local
        rule's gradients, the encoder and decoder updates, the distributed
        average and clipping. ``epoch`` (counted from 1, as in training)
        selects the update modes of that epoch (freeze windows and
        ``training_schedule``); by default the trainer's current epoch, the
        first before :meth:`train` has run. The optimizer's gradients are
        reset first, as in training. The model keeps its mode: in ``train()``
        mode this is a training step's update, including the forward's
        updates of sparse synapse statistics. Under ``core_update_mode:
        none`` the core is held fixed as in training: its parameters do not
        require gradients during the call and the forward leaves its sparse
        state unchanged. Before ``train`` has run, the trainer first prepares
        ``model`` as ``train`` does, without copying a best state. Returns the
        batch loss.
        """
        if not hasattr(self, "epoch_counter"):
            self._check_local_model(model)
            self._initialize_attributes(model, self.epochs, snapshot_best=False)
        if epoch is None:
            epoch = max(int(self.epoch_counter), 1)
        elif int(epoch) < 1:
            raise ValueError(f"epoch counts from 1, as in training; got {epoch}")
        frozen_core = self._freeze_core(model)
        try:
            with frozen_sparse_state() if frozen_core is not None else nullcontext():
                return self._assign_batch_gradients(
                    model, inputs.to(self.device), targets.to(self.device), int(epoch)
                )
        finally:
            if frozen_core is not None:
                restore_requires_grad(frozen_core)

    def _assign_batch_gradients(
        self,
        model: BaseModel,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """The gradient assignment of one training step at ``epoch``."""
        # Resolve per-epoch schedule for reactivation/decoder behavior
        update_reactivation, encoder_update_mode, decoder_update_mode = (
            _resolve_local_epoch_update_modes(
                self.local_cfg,
                epoch,
                self._normalize_reactivation_update_mode(),
            )
        )

        # Forward once to obtain outputs and local signals.
        y_hat, layer_records = _forward_with_local_recorders(self, model, inputs)

        # Reuse the recorded forward for pointwise losses. Model-dependent
        # losses retain the compatibility fallback to a separate evaluation.
        loss = self._compute_logging_loss(
            model,
            inputs,
            targets,
            predictions=y_hat,
        )

        # Compute output-space error (dL/dy_hat) then resolve local soma-space
        # signals for local dendritic updates.
        delta_out: torch.Tensor = self._compute_soma_error(y_hat, targets)
        v0_local, delta_local = self._resolve_local_soma_signals(
            model=model,
            y_hat=y_hat,
            delta_out=delta_out.detach(),
        )

        # Prepare HSIC targets if needed
        y_target = _prepare_hsic_target(self.local_cfg, targets, y_hat)

        # Zero gradients (we will set param.grads directly)
        _zero_optimizer_gradients(self.optimizer)

        # Apply local learning updates as gradients on parameters; with
        # core_update_mode "none" the core keeps its parameters.
        if getattr(self.local_cfg, "core_update_mode", "local") == "local":
            self._apply_local_rule_gradients(
                model,
                layer_records,
                v0=v0_local,
                delta=delta_local,
                y_target=y_target,
                update_reactivation=update_reactivation,
            )

        _accumulate_backprop_update_gradients(
            loss=loss,
            model=model,
            encoder_update_mode=encoder_update_mode,
            decoder_update_mode=decoder_update_mode,
        )

        _apply_local_decoder_update(
            model=model,
            decoder_update_mode=decoder_update_mode,
            decoder_cache=self._decoder_cache,
            delta_out=delta_out,
            normalize_by_batch=self.local_cfg.normalize_by_batch,
        )

        if self.uncovered_parameters is None:
            self._report_uncovered_parameters(model)

        # LocalCA assigns gradients directly rather than calling
        # ``loss.backward()``. DDP therefore cannot reduce them through
        # autograd hooks; average them explicitly before clipping.
        _average_distributed_gradients(model)

        # Optional gradient clipping (value-wise)
        _clip_model_gradients(model, self.local_cfg.clip_grad_value)
        return loss

    def _epoch_train(
        self, model: BaseModel, train_loader
    ) -> tuple[float, float, float]:  # type: ignore[override]
        model.train()
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)

        profile_loader = bool(getattr(self, "profile_dataloader", False))
        data_wait_total = 0.0
        step_time_total = 0.0
        max_data_wait = 0.0
        n_batches = 0

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
            loss = self._assign_batch_gradients(
                model, x_batch, y_batch, self.epoch_counter
            )
            _accumulate_local_logging_loss(loss_sum, loss)

            # Step optimizer
            self.optimizer.step()
            n_batches += 1
            if profile_loader:
                step_time_total += _step_seconds(step_start, loss_sum)

        if profile_loader and self._is_main_process:
            self._log_dataloader_profile(
                n_batches=n_batches,
                data_wait_total=data_wait_total,
                step_time_total=step_time_total,
                max_data_wait=max_data_wait,
            )

        return _average_local_epoch_losses(loss_sum, n_batches)

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        # The local-rule loop never builds the per-epoch LR scheduler.
        self._resolve_lr_schedule()
        # Standard training loop but with local rule gradient computation
        self._check_local_model(model)

        train_loader, valid_loader = self._dataloaders(train_data, valid_data)
        self._initialize_attributes(model, self.epochs)
        self._save_checkpoint(model)

        # Use the standard loop to ensure epoch counters and early stopping work
        train_losses: list[float] = []
        valid_losses: list[float] = []
        train_losses_base: list[float] = []
        train_losses_reg: list[float] = []
        frozen_core = self._freeze_core(model)
        try:
            with frozen_sparse_state() if frozen_core is not None else nullcontext():
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
        finally:
            if frozen_core is not None:
                restore_requires_grad(frozen_core)

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
        self._loss_plotting(train_losses, valid_losses)
        return results


__all__ = ["LocalCreditAssignment"]
