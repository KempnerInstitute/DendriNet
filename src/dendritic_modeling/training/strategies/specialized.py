"""
Specialized training strategies.

This module implements specialized training approaches including voltage stabilization
and other advanced training techniques for dendritic networks.
"""

import logging
from dataclasses import fields
from functools import partial

import torch

from dendritic_modeling.config.training import (
    EIEquilibriumStrategyConfig,
    HomeostaticControlStrategyConfig,
    VoltageStabilizationStrategyConfig,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.activations import ParametricActivation
from dendritic_modeling.networks.activations.endpoint_tanh import EndpointNormalizedTanh
from dendritic_modeling.training.loss.functions import (
    ExcitationInhibitionBalanceLoss,
    HomeostaticControlLoss,
    VoltageStabilizationLoss,
)
from dendritic_modeling.training.strategies.phases import merge_phase_results
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.training.strategies.trainability import (
    restore_requires_grad,
    snapshot_requires_grad,
)
from dendritic_modeling.utils.hooks import iter_modules_of_type, run_with_forward_hooks

logger = logging.getLogger(__name__)
_TRAINABLE_REACTIVATIONS = (ParametricActivation, EndpointNormalizedTanh)


def _set_parametric_activation_trainability(model, requires_grad: bool) -> None:
    for module in iter_modules_of_type(model, _TRAINABLE_REACTIVATIONS):
        for param in module.parameters():
            param.requires_grad = requires_grad


def _freeze_parametric_activations(model) -> list[tuple[torch.nn.Parameter, bool]]:
    """Freeze reactivation parameters and return their prior trainability."""
    states = [
        (param, bool(param.requires_grad))
        for module in iter_modules_of_type(model, _TRAINABLE_REACTIVATIONS)
        for param in module.parameters()
    ]
    _set_parametric_activation_trainability(model, False)
    return states


def _restore_parametric_activations(trainer, states) -> None:
    """Restore pre-warm-up trainability for the main phase.

    The warm-up phase's ``Trainer.train()`` caches the reactivation
    trainability it observes (frozen) as the original state, and every later
    ``train()`` call restores that cache. Dropping the cache after restoring
    the parameters makes the main phase recapture the pre-warm-up state.
    """
    for param, requires_grad in states:
        param.requires_grad = requires_grad
    if hasattr(trainer, "_reactivation_state_model_id"):
        trainer._reactivation_state_model_id = None


def _check_warmup_epochs(trainer) -> None:
    if int(trainer.pretrain_epochs) < 1:
        raise ValueError(
            f"pretrain_epochs={trainer.pretrain_epochs}: the warm-up runs at least "
            "one epoch; use strategy 'standard' to train without it"
        )


def _start_warmup_phase(trainer, warmup_loss, filename_prefix, logger_info_prefix):
    _check_warmup_epochs(trainer)
    default_loss_fn = trainer.loss_function
    default_epochs = trainer.epochs

    trainer.loss_function = warmup_loss
    trainer.epochs = int(trainer.pretrain_epochs)
    trainer.filename_prefix = filename_prefix
    trainer.logger_info_prefix = logger_info_prefix

    return default_loss_fn, default_epochs


def _start_standard_phase(trainer, default_loss_fn, default_epochs):
    trainer.loss_function = default_loss_fn
    trainer.epochs = default_epochs
    trainer.filename_prefix = "standard_"
    trainer.logger_info_prefix = "[Standard] "
    trainer.best_loss = float("inf")


def _merge_warmup_results(pretrain_results, final_results, *, warmup_loss=None):
    """Prepend the warm-up histories and record each phase's epochs.

    Warm-up entries are values of the warm-up loss; ``phases`` says how many
    leading epochs they cover.
    """
    return merge_phase_results(
        pretrain_results,
        final_results,
        first_phase="warmup",
        final_phase="main",
        first_loss=getattr(warmup_loss, "_loss_name", None),
    )


def _run_warmup_then_standard_training(
    trainer,
    train_fn,
    model,
    train_data,
    valid_data,
    warmup_loss,
    *,
    filename_prefix,
    logger_info_prefix,
    warmup_label,
    standard_label,
):
    trainer._reject_trainability_change_under_ddp(
        model,
        f"the {warmup_label} warm-up loss reaches only some parameters (the "
        "decoder gets no gradient)",
    )
    _check_warmup_epochs(trainer)
    with trainer._run_epoch_count(int(trainer.pretrain_epochs) + int(trainer.epochs)):
        handles = warmup_loss.attach_forward_hooks(model)

        default_loss_fn, default_epochs = _start_warmup_phase(
            trainer,
            warmup_loss,
            filename_prefix,
            logger_info_prefix,
        )

        frozen_states = (
            _freeze_parametric_activations(model) if trainer.freeze_reactivation else []
        )

        logger.info(
            f"=== {warmup_label}: "
            f"Starting pretraining phase for {trainer.pretrain_epochs} epochs ==="
        )
        try:
            pretrain_results = run_with_forward_hooks(
                attach=lambda: handles,
                remove=warmup_loss.remove_forward_hooks,
                body=partial(train_fn, model, train_data, valid_data),
            )
        finally:
            if trainer.freeze_reactivation:
                # freeze_reactivation applies to the warm-up phase only.
                _restore_parametric_activations(trainer, frozen_states)

            _start_standard_phase(trainer, default_loss_fn, default_epochs)

        logger.info(
            f"=== {standard_label}: "
            f"Starting normal training for {trainer.epochs} epochs ==="
        )
        # The warm-up chose its best state by the warm-up loss; the result is
        # selected among main-phase epochs only.
        trainer._restart_best_state_selection(model)
        final_results = train_fn(model, train_data, valid_data)
        return _merge_warmup_results(
            pretrain_results, final_results, warmup_loss=warmup_loss
        )


class VoltageStabilizationTrainer(Trainer):
    """
    Voltage Stabilization Training Strategy.

    This trainer includes a warm-up phase that stabilizes dendritic voltages
    to target values before proceeding with normal training. This can help
    with training stability in dendritic networks.
    """

    # Two phases (warm-up, then standard) each call Trainer.train().
    supports_lr_schedule = False
    strategy_options = frozenset(
        item.name for item in fields(VoltageStabilizationStrategyConfig)
    )
    pretrain_epochs: int = 1
    stabilize_mode: str = "vinf"
    freeze_reactivation: bool = False
    target_voltage: float = 0.5
    target_saturation: float = 0.2
    loss_metric: str = "mse"

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train the model with voltage stabilization warm-up.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """
        voltage_stabilization_loss = VoltageStabilizationLoss(
            stabilize_mode=self.stabilize_mode,
            target_voltage=self.target_voltage,
            target_saturation=self.target_saturation,
            loss_metric=self.loss_metric,
            reduction=self.loss_function.reduction,
        )

        return _run_warmup_then_standard_training(
            self,
            super().train,
            model,
            train_data,
            valid_data,
            voltage_stabilization_loss,
            filename_prefix="voltage_stabilization_",
            logger_info_prefix="[Voltage Stabilization] ",
            warmup_label="VoltageStabilization",
            standard_label="Voltage Stabilization",
        )


class ExcitationInhibitionEquilibriumTrainer(Trainer):
    """
    Excitation-Inhibition Equilibrium Training Strategy (``ei_equilibrium``).

    A warm-up phase drives each dendritic branch's excitatory conductance
    output toward its inhibitory one (``ExcitationInhibitionBalanceLoss``),
    then training proceeds with the task loss. The model needs branches with
    inhibitory synapses.
    """

    supports_lr_schedule = False
    strategy_options = frozenset(
        item.name for item in fields(EIEquilibriumStrategyConfig)
    )
    pretrain_epochs: int = 1
    freeze_reactivation: bool = False
    loss_metric: str = "mse"

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train the model with excitation-inhibition equilibrium warm-up.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """
        ei_equilibrium_loss = ExcitationInhibitionBalanceLoss(
            loss_metric=self.loss_metric
        )

        return _run_warmup_then_standard_training(
            self,
            super().train,
            model,
            train_data,
            valid_data,
            ei_equilibrium_loss,
            filename_prefix="ei_equilibrium_",
            logger_info_prefix="[Excitation-Inhibition Equilibrium] ",
            warmup_label="Excitation-Inhibition Equilibrium",
            standard_label="Excitation-Inhibition Equilibrium",
        )


class HomeostaticControlTrainer(Trainer):
    """
    Homeostatic Control Training Strategy.

    This trainer includes a warm-up phase that stabilizes dendritic voltages
    and excitatory-inhibitory balance to target values before proceeding with normal training.
    This can help with training stability in dendritic networks.
    """

    supports_lr_schedule = False
    strategy_options = frozenset(
        item.name for item in fields(HomeostaticControlStrategyConfig)
    )
    pretrain_epochs: int = 1
    stabilize_mode: str = "vinf"
    freeze_reactivation: bool = False
    target_voltage: float = 0.5
    target_saturation: float = 0.2
    loss_metric: str = "mse"

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train the model with homeostatic control warm-up.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """
        homeostatic_control_loss = HomeostaticControlLoss(
            stabilize_mode=self.stabilize_mode,
            target_voltage=self.target_voltage,
            target_saturation=self.target_saturation,
            loss_metric=self.loss_metric,
            reduction=self.loss_function.reduction,
        )

        return _run_warmup_then_standard_training(
            self,
            super().train,
            model,
            train_data,
            valid_data,
            homeostatic_control_loss,
            filename_prefix="homeostatic_control_",
            logger_info_prefix="[Homeostatic Control] ",
            warmup_label="Homeostatic Control",
            standard_label="HomeostaticControl",
        )


def _reactivation_midpoints(model) -> list[torch.nn.Parameter]:
    """Trainable midpoints ``b`` of the model's reactivation gates."""
    return [
        module.b
        for module in iter_modules_of_type(model, _TRAINABLE_REACTIVATIONS)
        if isinstance(getattr(module, "b", None), torch.nn.Parameter)
        and module.b.requires_grad
    ]


class TrainOnlyMReactivation(Trainer):
    """
    Standard training with the reactivation gate midpoints held fixed.

    The midpoint ``b`` of every trainable reactivation gate
    (``ParametricTanh``, ``ParametricReLU``, ``EndpointNormalizedTanh``) is
    frozen for the run, so the slopes ``m`` and all other parameters train.
    Every parameter's trainability is restored afterwards.
    """

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train with the reactivation midpoints held fixed.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """

        midpoints = _reactivation_midpoints(model)
        if not midpoints:
            raise ValueError(
                "TrainOnlyMReactivation holds reactivation midpoints b fixed, but "
                "the model has no reactivation gate with a trainable midpoint."
            )
        self._reject_trainability_change_under_ddp(
            model, "TrainOnlyMReactivation freezes the reactivation midpoints"
        )
        logger.info(
            "TrainOnlyMReactivation: holding %d reactivation midpoints fixed.",
            len(midpoints),
        )

        trainability = snapshot_requires_grad(model)
        for param in midpoints:
            param.requires_grad = False
        # Trainer.train() records the reactivation flags it finds as their
        # original state; record the frozen midpoints, not an earlier run's.
        self._reactivation_state_model_id = None
        self.filename_prefix = "train_only_m_reactivation_"
        try:
            return super().train(model, train_data, valid_data)
        finally:
            restore_requires_grad(trainability)
            self._reactivation_state_model_id = None
