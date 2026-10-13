"""
Two-Step Training Strategies
============================

This module contains training strategies that implement two-step learning approaches,
where models are first pre-trained with local supervision and then fine-tuned end-to-end.
"""

import logging
from dataclasses import fields

import torch
import torch.nn as nn
import torch.nn.functional as functional

from dendritic_modeling.config.training import (
    FreezeBranchesStrategyConfig,
    FreezeBranchKLStrategyConfig,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.utils.sparse_maintenance import frozen_sparse_state
from dendritic_modeling.training.clipping import clip_gradients
from dendritic_modeling.training.loss.functions import LossFunction
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer
from dendritic_modeling.training.strategies.phases import (
    evaluation_mode,
    merge_phase_results,
)
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.training.strategies.trainability import (
    restore_requires_grad,
    snapshot_requires_grad,
    train_only,
)

logger = logging.getLogger(__name__)


def _model_output_dim(model) -> int:
    """Number of model outputs, the width of every local supervision head."""
    output_dim = getattr(model, "output_dim", None)
    if output_dim is not None:
        return int(output_dim)
    decoder = getattr(model, "decoder_network", None)
    if decoder is not None:
        linears = [
            module for module in decoder.modules() if isinstance(module, nn.Linear)
        ]
        if linears:
            return int(linears[-1].out_features)
        if getattr(decoder, "output_dim", None) is not None:
            return int(decoder.output_dim)
    raise ValueError(
        "Two-step local pretraining could not infer the number of model "
        "outputs for its local heads; give the model an output_dim attribute "
        "or a decoder that ends in nn.Linear."
    )


def _initialize_local_heads(net_obj, model, sample):
    """One linear head per layer, from its output width to the model's outputs.

    The widths are those of each layer's output in the model's forward on
    ``sample`` (a batch of model inputs), measured in eval mode without
    advancing sparse-synapse state.
    """
    n_outputs = _model_output_dim(model)
    with torch.no_grad(), frozen_sparse_state(), evaluation_mode(model):
        widths = [
            _local_features(model, net_obj, sample, layer_idx).shape[-1]
            for layer_idx in range(len(net_obj.layers))
        ]
    return [nn.Linear(width, n_outputs).to(sample.device) for width in widths]


class _LocalLayerMaintenance:
    """``decay_weights``/``apply_rewiring`` restricted to one local layer.

    Stage-1 optimizers wrap only the layer being pretrained, so sparse
    maintenance and rewiring reach that layer's synapses and never the frozen
    layers. Population layers forward the calls to their populations, as
    ``PopulationNetwork`` does.
    """

    def __init__(self, layer):
        self.layer = layer
        if hasattr(layer, "decay_weights"):
            self._targets = [layer]
        else:
            populations = getattr(layer, "populations", None)
            values = populations.values() if populations is not None else ()
            self._targets = [item for item in values if hasattr(item, "decay_weights")]
        maintained = [
            module
            for module in layer.modules()
            if module is not layer and hasattr(module, "decay_weights")
        ]
        if maintained and not self._targets:
            raise ValueError(
                f"Two-step local pretraining cannot run sparse maintenance for "
                f"{type(layer).__name__} layers (no decay_weights path)."
            )

    def modules(self):
        return self.layer.modules()

    def decay_weights(self, weight_decay, weight_boosting=False):
        for target in self._targets:
            target.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self):
        for target in self._targets:
            if hasattr(target, "apply_rewiring"):
                target.apply_rewiring()


def _local_param_groups(layer, head, torch_optimizer, head_lr: float) -> list[dict]:
    """Param groups of one local optimizer, with the main groups' settings.

    Each layer parameter keeps the learning rate (and other per-group
    options) of its group in the configured optimizer, so
    ``common.param_groups`` also governs stage 1. The local head is a readout
    and uses ``head_lr``. Layer parameters that do not require gradients (fixed
    gate slopes, reactivation parameters under ``reactivation_update_mode``)
    are left out.
    """
    layer_params = [param for param in layer.parameters() if param.requires_grad]
    layer_ids = {id(param) for param in layer_params}
    groups = []
    covered: set[int] = set()
    for group in torch_optimizer.param_groups:
        params = [param for param in group["params"] if id(param) in layer_ids]
        if params:
            options = {key: value for key, value in group.items() if key != "params"}
            groups.append({**options, "params": params})
            covered.update(id(param) for param in params)
    remaining = [param for param in layer_params if id(param) not in covered]
    groups.append({"params": [*remaining, *head.parameters()], "lr": head_lr})
    return groups


def _local_head_lr(model, torch_optimizer) -> float:
    """Learning rate of the model readout (decoder) group, else the default."""
    decoder = getattr(model, "decoder_network", None) if model is not None else None
    decoder_ids = (
        {id(param) for param in decoder.parameters()} if decoder is not None else set()
    )
    for group in torch_optimizer.param_groups:
        if any(id(param) in decoder_ids for param in group["params"]):
            return float(group["lr"])
    return float(torch_optimizer.defaults["lr"])


def _make_local_optimizers(net_obj, local_heads, base_optimizer, model=None):
    """Create per-layer optimizers that also own local-head parameters.

    They use the configured optimizer's class, hyperparameters and per-group
    learning rates. When the main optimizer is a ``CustomWeightDecayOptimizer``
    each local optimizer is wrapped the same way (``synapse_decay_rate``,
    ``weight_boosting``), restricted to its layer.
    """
    base_wrapper = (
        base_optimizer
        if isinstance(base_optimizer, CustomWeightDecayOptimizer)
        else None
    )
    torch_optimizer = (
        base_optimizer.optimizer
        if hasattr(base_optimizer, "optimizer")
        and isinstance(base_optimizer.optimizer, torch.optim.Optimizer)
        else base_optimizer
    )
    optimizer_cls = type(torch_optimizer)
    defaults = dict(getattr(torch_optimizer, "defaults", {}))
    head_lr = _local_head_lr(model, torch_optimizer)

    local_optimizers = []
    for layer, head in zip(net_obj.layers, local_heads):
        local = optimizer_cls(
            _local_param_groups(layer, head, torch_optimizer, head_lr), **defaults
        )
        if base_wrapper is not None:
            local = CustomWeightDecayOptimizer(
                model=_LocalLayerMaintenance(layer),
                optimizer=local,
                weight_decay=base_wrapper.weight_decay,
                weight_boosting=base_wrapper.weight_boosting,
            )
        local_optimizers.append(local)
    return local_optimizers


def _current_local_params(layer, local_head):
    params = [*local_head.parameters(), *layer.parameters()]
    return [param for param in params if param.requires_grad]


def _local_features(model, net_obj, x_batch, layer_idx):
    """Output of core layer ``layer_idx`` in the model's own forward pass.

    The encoder and the core up to that layer run as in ``model.forward``
    (transfer layer, inhibitory stream, projections and routing).
    """
    encoder = getattr(model, "encoder_network", None)
    encoded = encoder(x_batch) if encoder is not None else x_batch
    return net_obj.forward_until(encoded, layer_idx)


def _ensure_single_process_two_step(trainer_name: str, is_distributed: bool) -> None:
    if is_distributed:
        raise NotImplementedError(
            f"{trainer_name} staged local pretraining is not supported under DDP. "
            "Run the two-step pretraining phase on a single process or use "
            "strategy='standard' for distributed end-to-end training."
        )


def _resolve_layer_container(model):
    """Return the module that owns locally supervised layers."""
    for attr_name in ("core_network", "net", "network"):
        if hasattr(model, attr_name):
            return getattr(model, attr_name)
    return model


# Typed freeze_branches fields that no two-step trainer implements.
_UNIMPLEMENTED_TWO_STEP_FIELDS = (
    "reverse_training",
    "epochs_per_branch",
    "final_tune_epochs",
)


def _validate_two_step_request(trainer, model, net_obj) -> None:
    """Reject options and model families the two-step trainers would ignore."""
    name = type(trainer).__name__
    for field_name in _UNIMPLEMENTED_TWO_STEP_FIELDS:
        default = FreezeBranchesStrategyConfig.__dataclass_fields__[field_name].default
        value = getattr(trainer, field_name, default)
        if value != default:
            raise ValueError(
                f"{name} does not implement {field_name} (set to {value!r}): "
                "stage 1 pretrains every layer for pretrain_epochs and stage 2 "
                "trains end to end for epochs. Remove it from "
                "training.main.strategies."
            )
    if trainer.regularization_manager.has_regularization():
        raise ValueError(
            f"{name} stage 1 (local pretraining) does not apply auxiliary "
            "regularization while stage 2 would, so it would act on one phase "
            "only. Remove training.main.regularization for this strategy."
        )
    if type(trainer.loss_function).from_predictions is LossFunction.from_predictions:
        raise ValueError(
            f"{name} scores each local head's output with the configured loss, "
            f"but {type(trainer.loss_function).__name__} needs the whole model."
        )
    if not hasattr(net_obj, "layers"):
        raise ValueError(
            f"{name} pretrains core_network.layers with local heads, but "
            f"{type(net_obj).__name__} has no layers (it previously fell back "
            "to standard training silently). Use strategy 'standard'."
        )
    if bool(getattr(model, "_is_recurrent_core", False)):
        raise ValueError(
            f"{name} local pretraining needs a feed-forward core; recurrent "
            f"{type(net_obj).__name__} cores are not supported."
        )
    parameterless = sorted(
        {
            type(layer).__name__
            for layer in net_obj.layers
            if not any(True for _ in layer.parameters())
        }
    )
    if parameterless:
        raise ValueError(
            f"{name} pretrains each entry of core_network.layers with a local "
            f"head, but {parameterless} entries have no parameters, so "
            f"{type(net_obj).__name__} cores are not supported."
        )
    if not callable(getattr(net_obj, "forward_until", None)):
        raise ValueError(
            f"{name} trains each layer on its input in the model's forward, which "
            f"needs the core's forward_until (ExcitationInhibitionNetwork and "
            f"feed-forward PopulationNetwork have it); {type(net_obj).__name__} "
            "does not."
        )


def _prepare_local_pretraining(trainer, model, net_obj, train_data, valid_data):
    train_loader, valid_loader = trainer._dataloaders(train_data, valid_data)
    trainer._initialize_attributes(model, trainer.pretrain_epochs + trainer.epochs)

    if not hasattr(trainer, "local_heads"):
        # One example sizes the heads; indexing the dataset leaves the
        # loaders' random streams untouched.
        sample = torch.as_tensor(train_data[0][0]).unsqueeze(0).to(trainer.device)
        trainer.local_heads = _initialize_local_heads(net_obj, model, sample)
    if not hasattr(trainer, "local_optimizers"):
        trainer.local_optimizers = _make_local_optimizers(
            net_obj, trainer.local_heads, trainer.optimizer, model=model
        )

    return train_loader, valid_loader


class _LocalPretrainingTrainer(Trainer):
    """Stage 1 pretrains each core layer with a local head; stage 2 trains end to end.

    In stage 1 every layer in turn trains alone (the others and the model's
    fixed parameters stay fixed) on its input in the model's own forward,
    scored by a linear head on its output. Each stage-1 epoch is validated
    like a standard epoch (eval mode, the configured loss on the model). The
    result is selected among stage-2 epochs.
    """

    # Stage 1 uses per-layer local optimizers outside Trainer.train().
    supports_lr_schedule = False
    # Stage 1 (local pretraining) does not apply these; stage 2 would, so a
    # configured value would act on one phase only (see STEP_OPTIONS).
    unapplied_options = Trainer.unapplied_options | frozenset(
        {
            "use_amp",
            "enable_nan_checking",
            "profile_dataloader",
        }
    )
    pretrain_epochs: int = 5
    # Filename prefixes of stage 1 and stage 2, and the stage-1 log prefix.
    stage_prefixes = ("twostep_stage1_", "twostep_stage2_")
    stage1_label = "[TwoStep-Pretrain] "

    def _local_loss(self, model, head, features, x_batch, y_batch):
        """Training loss of one local head on one batch."""
        return self.loss_function.from_predictions(head(features), y_batch)

    def _local_loss_name(self) -> str:
        name = getattr(self.loss_function, "_loss_name", type(self.loss_function))
        return f"local head {name}"

    def _pretrain_layers(self, model, net_obj, train_loader, valid_loader):
        """Stage 1; returns its training and validation loss histories."""
        train_losses, valid_losses = [], []
        # Fixed parameters stay fixed while their layer pretrains, and every
        # flag returns to its value before stage 1, also after an error.
        trainability = snapshot_requires_grad(model)
        try:
            for _ in range(self.pretrain_epochs):
                self.epoch_counter += 1
                epoch_loss, n_batches = 0.0, 0
                for layer_idx, layer in enumerate(net_obj.layers):
                    head = self.local_heads[layer_idx]
                    optimizer = self.local_optimizers[layer_idx]
                    train_only(model, (layer, head), trainability)
                    params = _current_local_params(layer, head)
                    model.train()
                    for batch in train_loader:
                        x_batch, y_batch = self._move_training_batch_to_device(
                            batch, train_loader
                        )
                        features = _local_features(model, net_obj, x_batch, layer_idx)
                        loss = self._local_loss(model, head, features, x_batch, y_batch)
                        optimizer.zero_grad()
                        loss.backward()
                        clip_gradients(
                            params, self.grad_clip_value, self.grad_clip_mode
                        )
                        optimizer.step()
                        epoch_loss += loss.item()
                        n_batches += 1
                train_loss = epoch_loss / n_batches if n_batches else 0.0
                train_losses.append(train_loss)
                valid_loss = self._non_distributed_validation_loss(model, valid_loader)
                valid_losses.append(valid_loss)
                self._update_best(model, valid_loss)
                self._analyze()
                self._tracking(train_loss, valid_loss, self.stage1_label)
                if self._check_early_stopping():
                    break
        finally:
            restore_requires_grad(trainability)
        return train_losses, valid_losses

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """Run local pretraining, then end-to-end training.

        Returns the stage-2 results with the stage-1 histories prepended and
        ``phases`` naming both stages.
        """
        name = type(self).__name__
        _ensure_single_process_two_step(name, self._is_distributed)
        net_obj = _resolve_layer_container(model)
        _validate_two_step_request(self, model, net_obj)
        with self._run_epoch_count(self.pretrain_epochs + self.epochs):
            train_loader, valid_loader = _prepare_local_pretraining(
                self, model, net_obj, train_data, valid_data
            )

            logger.info(
                "=== %s: stage 1 (local pretraining) for %d epochs ===",
                name,
                self.pretrain_epochs,
            )
            self.filename_prefix = self.stage_prefixes[0]
            try:
                train_losses, valid_losses = self._pretrain_layers(
                    model, net_obj, train_loader, valid_loader
                )
            except Exception:
                self.filename_prefix = ""
                raise
            self._loss_plotting(train_losses, valid_losses)

            logger.info(
                "=== %s: stage 2 (end-to-end training) for %d epochs ===",
                name,
                self.epochs,
            )
            # Stage 1 selected its best state among locally pretrained weights;
            # the result is selected among stage-2 epochs.
            self._restart_best_state_selection(model)
            self.filename_prefix = self.stage_prefixes[1]
            try:
                final_results = super().train(model, train_data, valid_data)
            finally:
                self.filename_prefix = ""

            return merge_phase_results(
                {"train_losses": train_losses, "valid_losses": valid_losses},
                final_results,
                first_phase="local_pretraining",
                final_phase="end_to_end",
                first_loss=self._local_loss_name(),
            )


class TwoStepTrainer(_LocalPretrainingTrainer):
    """
    Two-Step Training Strategy for dendritic networks (``freeze_branches``).

    1. Pre-training phase: each layer is trained with a local supervision
       head on its input in the model's forward.
    2. Main training phase: end-to-end training of the full network.

    This approach can help with gradient flow and learning in deep dendritic networks.
    """

    strategy_options = frozenset(
        item.name for item in fields(FreezeBranchesStrategyConfig)
    )


class TwoStepTrainerWithKL(_LocalPretrainingTrainer):
    """
    Two-Step Training Strategy with a KL term (``freeze_branch_kl``).

    Like ``TwoStepTrainer``, but each local head is also pulled toward the full
    model's predictions: the stage-1 loss is the configured loss of the head
    plus ``kl_weight`` times KL(model || head), with the model's predictions
    computed in eval mode as a fixed target.
    """

    strategy_options = frozenset(
        item.name for item in fields(FreezeBranchKLStrategyConfig)
    )
    kl_weight: float = 0.1
    stage_prefixes = ("twostep_kl_stage1_", "twostep_kl_stage2_")
    stage1_label = "[TwoStepKL-Pretrain] "

    def _local_loss(self, model, head, features, x_batch, y_batch):
        local_logits = head(features)
        # The KL target is an auxiliary pass: it runs in eval mode and must not
        # advance schedules or statistics a second time.
        with torch.no_grad(), frozen_sparse_state(), evaluation_mode(model):
            full_logits = model(x_batch)
        task_loss = self.loss_function.from_predictions(local_logits, y_batch)
        kl_loss = functional.kl_div(
            functional.log_softmax(local_logits, dim=1),
            functional.softmax(full_logits, dim=1),
            reduction="batchmean",
        )
        return task_loss + self.kl_weight * kl_loss

    def _local_loss_name(self) -> str:
        return f"{super()._local_loss_name()} + {self.kl_weight} x KL"
