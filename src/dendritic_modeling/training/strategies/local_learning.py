"""
Local learning training strategies.

This module implements multi-factor local learning rules for dendritic networks.

It provides biologically-inspired 3-/4-/5-factor Hebbian-like rules that use
locally available signals (pre-synaptic input, compartment voltages) modulated
by a broadcast soma error and optional morphology/information factors.
"""

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from dendritic_modeling.config.local_learning import (
    build_local_rule_config,
    coerce_legacy_local_rule_config,
    section_to_dict,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.strategies.local_learning_broadcast_mixin import (
    LocalLearningBroadcastMixin,
)
from dendritic_modeling.training.strategies.local_learning_gradients_mixin import (
    LocalLearningGradientMixin,
)
from dendritic_modeling.training.strategies.local_learning_hsic_mixin import (
    LocalLearningHSICMixin,
)
from dendritic_modeling.training.strategies.local_learning_modulators import (
    LocalLearningModulatorMixin,
)
from dendritic_modeling.training.strategies.local_learning_morphology_mixin import (
    LocalLearningMorphologyMixin,
)
from dendritic_modeling.training.strategies.local_learning_recorders import (
    LocalLearningRecorderMixin,
)
from dendritic_modeling.training.strategies.local_learning_signals import (
    LocalLearningSignalMixin,
)
from dendritic_modeling.training.strategies.local_learning_state import _LayerStats
from dendritic_modeling.training.strategies.local_learning_stdp_mixin import (
    LocalLearningSTDPMixin,
)
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


class LocalCreditAssignment(
    LocalLearningSignalMixin,
    LocalLearningRecorderMixin,
    LocalLearningSTDPMixin,
    LocalLearningBroadcastMixin,
    LocalLearningHSICMixin,
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

    def __init__(self, *args, **kwargs):
        config_dict = kwargs.pop("local_rule_config", {}) or {}
        super().__init__(*args, **kwargs)
        # Build LocalRuleConfig with proper nested config handling
        self.local_cfg = self._build_local_rule_config(config_dict)
        self._layer_stats: dict[int, _LayerStats] = {}
        self._decoder_cache: dict[str, Any] = {}
        self._additive_gain_cache: dict[str, Any] = {}
        self._additive_running_var: dict[int, torch.Tensor] = {}
        self._broadcast_cache: dict[tuple[Any, ...], Any] = {}
        self._stdp_traces: dict[int, dict[str, torch.Tensor]] = {}
        self._warned_decoder_soma_fallback = False
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

        mode = getattr(self.local_cfg, "explicit_inhibitory_update_mode", None)
        if mode is None:
            # Compatibility for older LocalRuleConfig-like objects in tests.
            return bool(
                getattr(self.local_cfg, "update_explicit_inhibitory_cells", True)
            )
        return str(mode).lower() == "local_ca"

    # Override: do not run extra pretrain/final phases here
    def _pretrain(self, *args, **kwargs):  # type: ignore[override]
        return [], []

    def _resolve_local_reactivation_update_flag(
        self, update_reactivation: bool
    ) -> bool:
        """Disable local reactivation updates when the shared trainer mode does not use learned b,m."""
        if self._normalize_reactivation_update_mode() == "backprop":
            return bool(update_reactivation)
        return False

    # ----------------------- Core training overrides -----------------------
    def _epoch_train(
        self, model: BaseModel, train_loader
    ) -> tuple[float, float, float]:  # type: ignore[override]
        model.train()
        train_loss = 0.0
        train_loss_base = 0.0
        train_loss_reg = 0.0

        # Fallback: if model has no dendritic branch layers, use standard epoch_train
        try:
            has_branch_layers = any(
                isinstance(m, DendriticBranchLayer) for m in model.modules()
            )
        except Exception:
            has_branch_layers = False
        if not has_branch_layers:
            # Defer to base implementation
            return super()._epoch_train(model, train_loader)

        for batch in train_loader:
            x_batch, y_batch = batch[0], batch[1]
            # Resolve per-epoch schedule for reactivation/decoder behavior
            update_reactivation = self.local_cfg.update_reactivation
            encoder_update_mode = self.local_cfg.encoder_update_mode
            decoder_update_mode = self.local_cfg.decoder_update_mode
            # Freeze phases by epoch index
            if self.epoch_counter <= max(0, int(self.local_cfg.freeze_encoder_epochs)):
                encoder_update_mode = "none"
            if self.epoch_counter <= max(
                0, int(self.local_cfg.freeze_reactivation_epochs)
            ):
                update_reactivation = False
            if self.epoch_counter <= max(0, int(self.local_cfg.freeze_decoder_epochs)):
                decoder_update_mode = "none"
            # Optional schedule entries override
            if self.local_cfg.training_schedule:
                for sched in self.local_cfg.training_schedule:
                    try:
                        s = int(sched.get("start_epoch", 1))
                        e = int(sched.get("end_epoch", 10**9))
                    except Exception:
                        s, e = 1, 10**9
                    if s <= self.epoch_counter <= e:
                        if "update_reactivation" in sched:
                            update_reactivation = bool(sched["update_reactivation"])
                        if "encoder_update_mode" in sched:
                            encoder_update_mode = str(
                                sched["encoder_update_mode"]
                            ).lower()
                        if "decoder_update_mode" in sched:
                            decoder_update_mode = str(
                                sched["decoder_update_mode"]
                            ).lower()
                        break
            update_reactivation = self._resolve_local_reactivation_update_flag(
                update_reactivation
            )
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # Capture layer-wise local signals via forward hooks
            layer_records, handles = self._attach_local_recorders(model)
            topk_modules = [m for m in model.modules() if isinstance(m, TopKLinear)]
            for m in topk_modules:
                m.cache_mask = True
            try:
                # Forward once to obtain outputs and local signals
                y_hat: torch.Tensor = model(x_batch)
            finally:
                for h in handles:
                    h.remove()
                for m in topk_modules:
                    m.cache_mask = False
                    m._last_forward_weight_mask = None

            # Compute scalar loss for logging (no backward)
            loss = self._compute_logging_loss(model, x_batch, y_batch)
            loss_item = loss.item()
            train_loss += loss_item
            train_loss_base += loss_item  # Local learning doesn't separate base/reg
            # train_loss_reg remains 0

            # Compute output-space error (dL/dy_hat) then resolve local soma-space
            # signals for local dendritic updates.
            delta_out: torch.Tensor = self._compute_soma_error(y_hat, y_batch)
            v0_local, delta_local = self._resolve_local_soma_signals(
                model=model,
                y_hat=y_hat,
                delta_out=delta_out.detach(),
            )

            # Prepare HSIC targets if needed
            y_target: Optional[torch.Tensor] = None
            if self.local_cfg.hsic.enabled:
                if self.local_cfg.hsic.target_source == "labels":
                    if y_batch.dim() == 1 or (
                        y_batch.dim() == 2 and y_batch.size(1) == 1
                    ):
                        num_classes = y_hat.size(-1)
                        y_target = torch.nn.functional.one_hot(
                            y_batch.view(-1), num_classes=num_classes
                        ).to(y_hat.dtype)
                    else:
                        y_target = y_batch.to(y_hat.dtype)
                else:
                    y_target = y_hat.detach()

            # Zero gradients (we will set param.grads directly)
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except TypeError:
                self.optimizer.zero_grad()

            # Apply local learning updates as gradients on parameters
            self._apply_local_rule_gradients(
                model,
                layer_records,
                v0=v0_local,
                delta=delta_local,
                y_target=y_target,
                update_reactivation=update_reactivation,
            )

            backprop_param_groups: list[list[torch.nn.Parameter]] = []
            if (
                encoder_update_mode == "backprop"
                and hasattr(model, "encoder_network")
                and model.encoder_network is not None
            ):
                enc_params = [
                    p for p in model.encoder_network.parameters() if p.requires_grad
                ]
                if enc_params:
                    backprop_param_groups.append(enc_params)

            if hasattr(model, "decoder_network") and decoder_update_mode == "backprop":
                dec_params = [
                    p for p in model.decoder_network.parameters() if p.requires_grad
                ]
                if dec_params:
                    backprop_param_groups.append(dec_params)

            for group_idx, params in enumerate(backprop_param_groups):
                grads = torch.autograd.grad(
                    loss,
                    params,
                    retain_graph=group_idx < len(backprop_param_groups) - 1,
                    allow_unused=True,
                )
                for p, g in zip(params, grads):
                    if g is None:
                        continue
                    grad = g.detach()
                    if p.grad is None:
                        p.grad = grad.clone()
                    else:
                        p.grad = p.grad + grad

            # Decoder update mode
            if hasattr(model, "decoder_network") and decoder_update_mode == "local":
                # Simple local 3-factor on last Linear: grad_W = e_out^T @ h_in; grad_b = sum(e_out)
                lin_mod = self._decoder_cache.get("module")
                h_in = self._decoder_cache.get("input")
                if isinstance(lin_mod, nn.Linear) and isinstance(h_in, torch.Tensor):
                    e_out = delta_out.detach()  # [B, out]
                    if self.local_cfg.normalize_by_batch and e_out.size(0) > 0:
                        scale = 1.0 / float(e_out.size(0))
                    else:
                        scale = 1.0
                    grad_w = scale * (e_out.t() @ h_in.detach())  # [out, in]
                    if lin_mod.weight.grad is None:
                        lin_mod.weight.grad = grad_w.to(lin_mod.weight.dtype)
                    else:
                        lin_mod.weight.grad = lin_mod.weight.grad + grad_w.to(
                            lin_mod.weight.dtype
                        )
                    if lin_mod.bias is not None:
                        grad_b = scale * e_out.sum(dim=0)
                        if lin_mod.bias.grad is None:
                            lin_mod.bias.grad = grad_b.to(lin_mod.bias.dtype)
                        else:
                            lin_mod.bias.grad = lin_mod.bias.grad + grad_b.to(
                                lin_mod.bias.dtype
                            )

            # Optional gradient clipping (value-wise)
            if self.local_cfg.clip_grad_value and self.local_cfg.clip_grad_value > 0:
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), self.local_cfg.clip_grad_value
                )

            # Step optimizer
            self.optimizer.step()

        train_loss /= len(train_loader)
        train_loss_base /= len(train_loader)
        train_loss_reg /= len(train_loader)
        return train_loss, train_loss_base, train_loss_reg

    # ------------------------ Helper: local rule grads -----------------------
    def _apply_local_rule_gradients(
        self,
        model: BaseModel,
        layer_records: list[dict[str, Any]],
        v0: torch.Tensor,
        delta: torch.Tensor,
        y_target: Optional[torch.Tensor] = None,
        update_reactivation: bool = True,
    ) -> None:
        batch_size = v0.size(0)
        delta_scalar: torch.Tensor = self._reduce_error_to_scalar(delta)  # [B, 1]
        broadcast_state = self._prepare_local_rule_broadcast_state(
            model=model,
            layer_records=layer_records,
            delta=delta,
            delta_scalar=delta_scalar,
        )

        # Iterate in the order records were captured
        for layer_idx, rec in enumerate(layer_records):
            if not self._should_update_local_record(rec):
                # In freeze mode, explicit inhibitory-cell dendrites are held
                # fixed. The E-cell I-to-E synapses still learn locally because
                # they are part of excitatory dendrite records.
                continue

            v_n: Optional[torch.Tensor] = rec.get("v_n")  # [B, out]
            if v_n is None:
                continue

            # Layer depth scaling (1-indexed)
            layer_depth = layer_idx + 1
            out_features = v_n.size(1)

            e_n = self._compute_local_broadcast_error(
                rec=rec,
                layer_idx=layer_idx,
                out_features=out_features,
                v_n=v_n,
                delta=delta,
                delta_scalar=delta_scalar,
                broadcast_state=broadcast_state,
            )
            stdp_error_signal = e_n
            if str(getattr(self.local_cfg, "rule_variant", "")).lower() == "stdp":
                e_n = torch.zeros_like(e_n)

            layer_dynamics_mode = self._resolve_layer_dynamics_mode(rec)
            R_tot = self._compute_local_input_resistance(
                rec=rec,
                v_n=v_n,
                layer_dynamics_mode=layer_dynamics_mode,
            )

            # === PATH PROPAGATION FACTOR ===
            # Apply path-integrated attenuation if enabled
            if (
                self.local_cfg.morphology_aware.use_path_propagation
                and broadcast_state.mode != "path_transport"
            ):
                path_factor = rec.get("path_factor", 1.0)
                # Modulate error by path attenuation
                if isinstance(path_factor, torch.Tensor):
                    e_n = e_n * path_factor

            modulators = self._compute_local_layer_modulators(
                rec=rec,
                v0=v0,
                layer_depth=layer_depth,
            )
            post_factors = self._compute_local_post_factors(
                rec=rec,
                e_n=e_n,
                v_n=v_n,
                r_tot=R_tot,
                layer_dynamics_mode=layer_dynamics_mode,
                layer_idx=layer_idx,
                modulators=modulators,
            )
            self._apply_layer_local_gradients(
                rec=rec,
                layer_idx=layer_idx,
                num_layers=len(layer_records),
                y_target=y_target,
                batch_size=batch_size,
                e_n=e_n,
                stdp_error_signal=stdp_error_signal,
                v_n=v_n,
                r_tot=R_tot,
                layer_dynamics_mode=layer_dynamics_mode,
                modulators=modulators,
                post_factors=post_factors,
                update_reactivation=update_reactivation,
            )

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        # Standard training loop but with local rule gradient computation
        has_ei = hasattr(model, "core_network") and isinstance(
            model.core_network, ExcitationInhibitionNetwork
        )
        has_branches = any(isinstance(m, DendriticBranchLayer) for m in model.modules())
        if not (has_ei or has_branches):
            logger.info(
                "No EI/dendritic branch layers detected; falling back to standard training."
            )
            return super().train(model, train_data, valid_data)

        train_loader, valid_loader = self._dataloaders(train_data, valid_data)
        self._initialize_attributes(model, self.epochs)
        self._save_checkpoint(model)

        # Use the standard loop to ensure epoch counters and early stopping work
        train_losses: list[float] = []
        valid_losses: list[float] = []
        train_losses_base: list[float] = []
        train_losses_reg: list[float] = []
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
                wandb_info={},
            )
        )

        if self.load_best_state_dict:
            model.load_state_dict(self.best_state_dict)

        # Log best model information like standard training
        logger.info(f"Best epoch: {self.best_epoch}, Best loss: {self.best_loss:.4f}")
        logger.info("Final model loaded from best checkpoint")

        results = {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "best_state_dict": self.best_state_dict,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
        }
        self._loss_plotting(train_losses, valid_losses)
        return results


__all__ = ["LocalCreditAssignment"]
