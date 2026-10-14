"""
Layer-wise training strategies for dendritic models.

This module contains training strategies that train network layers sequentially,
either from soma to input or from input to soma.
"""

import logging
from dataclasses import fields

import torch

from dendritic_modeling.config.training import FreezeLayersStrategyConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.training.strategies.trainability import (
    restore_requires_grad,
    snapshot_requires_grad,
    train_only,
)

logger = logging.getLogger(__name__)


LossHistories = tuple[list[float], list[float], list[float], list[float]]


def _empty_loss_histories() -> LossHistories:
    return [], [], [], []


def _loss_histories_from_results(results: dict) -> LossHistories:
    return (
        results.get("train_losses", []),
        results.get("valid_losses", []),
        results.get("train_losses_base", []),
        results.get("train_losses_reg", []),
    )


def _ordered_layer_training_plan(
    layers,
    reverse_training: bool,
) -> tuple[str, list[tuple[int, torch.nn.Module]]]:
    layer_list = list(layers)
    n_layers = len(layer_list)

    if reverse_training:
        learning_strategy = "Freeze-Layers-Reversed"
        indexed_layers = [
            (n_layers - i - 1, layer) for i, layer in enumerate(reversed(layer_list))
        ]
    else:
        learning_strategy = "Freeze-Layers"
        indexed_layers = list(enumerate(layer_list))

    return learning_strategy, indexed_layers


def _layerwise_total_epochs(
    pretrain_epochs: int,
    n_layers: int,
    epochs_per_layer: int,
    final_tune_epochs: int,
) -> int:
    return pretrain_epochs + n_layers * epochs_per_layer + final_tune_epochs


def _strategy_filename_prefix(learning_strategy: str) -> str:
    return f"{learning_strategy.lower().replace('-', '_')}_"


def _strategy_logger_prefix(learning_strategy: str) -> str:
    return f"[{learning_strategy}] "


def _layer_logger_info(learning_strategy: str, layer_idx: int) -> str:
    return f"[{learning_strategy}] layer={layer_idx}, "


def _prepend_loss_histories(
    prefix: LossHistories,
    suffix: LossHistories,
) -> LossHistories:
    return tuple(
        prefix_part + suffix_part for prefix_part, suffix_part in zip(prefix, suffix)
    )


def _prepend_loss_histories_to_results(
    results: dict,
    prefix: LossHistories,
) -> dict:
    (
        train_losses,
        valid_losses,
        train_losses_base,
        train_losses_reg,
    ) = prefix
    results["train_losses"] = train_losses + results.get("train_losses", [])
    results["valid_losses"] = valid_losses + results.get("valid_losses", [])
    results["train_losses_base"] = train_losses_base + results.get(
        "train_losses_base", []
    )
    results["train_losses_reg"] = train_losses_reg + results.get("train_losses_reg", [])
    return results


class LayerWiseTrainer(Trainer):
    """
    Training strategy that trains network layers sequentially.

    This trainer provides methods to train layers one at a time, either from
    soma to input or from input to soma, followed by full-network fine-tuning.

    """

    # Pretrain, per-layer and final-tune phases use separate loops.
    supports_lr_schedule = False
    strategy_options = frozenset(
        item.name for item in fields(FreezeLayersStrategyConfig)
    )
    pretrain_epochs: int = 10
    reverse_training: bool = False
    epochs_per_layer: int = 5
    final_tune_epochs: int = 10

    def _pretrain(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> tuple[list[float], list[float], list[float], list[float]]:
        """
        Pretrain the model.
        """
        pretrain_histories = _empty_loss_histories()

        if self.pretrain_epochs > 0:
            logger.info(
                f"Starting full network pretraining "
                f"for {self.pretrain_epochs} epochs..."
            )
            default_epochs = self.epochs
            self.epochs = self.pretrain_epochs
            self.filename_prefix = "pretrain_"
            self.logger_info_prefix = "[Pretrain] "
            try:
                pretrain_results = super().train(model, train_data, valid_data)
            finally:
                self.epochs = default_epochs
                self.filename_prefix = ""

            pretrain_histories = _loss_histories_from_results(pretrain_results)

        return pretrain_histories

    def _train_layer_plan(
        self,
        model: BaseModel,
        train_loader: torch.utils.data.DataLoader,
        valid_loader: torch.utils.data.DataLoader,
        layer_plan: list[tuple[int, torch.nn.Module]],
        learning_strategy: str,
    ) -> LossHistories:
        train_losses, valid_losses, train_losses_base, train_losses_reg = (
            _empty_loss_histories()
        )

        self.filename_prefix = _strategy_filename_prefix(learning_strategy)
        self.logger_info_prefix = _strategy_logger_prefix(learning_strategy)

        # Parameters fixed before this phase (fixed gate slopes, frozen
        # encoders, reactivation_update_mode) stay fixed while their layer
        # trains, and every flag returns to its prior value afterwards.
        trainability = snapshot_requires_grad(model)
        try:
            for layer_idx, layer_module in layer_plan:
                train_only(model, [layer_module], trainability)
                logger.info(
                    f"Training only layer {layer_idx} for "
                    f"{self.epochs_per_layer} epochs."
                )

                train_losses, valid_losses, train_losses_base, train_losses_reg = (
                    self._run_training_loop(
                        n_epochs=self.epochs_per_layer,
                        model=model,
                        train_loader=train_loader,
                        valid_loader=valid_loader,
                        train_losses=train_losses,
                        valid_losses=valid_losses,
                        train_losses_base=train_losses_base,
                        train_losses_reg=train_losses_reg,
                        logger_info=_layer_logger_info(learning_strategy, layer_idx),
                    )
                )
        finally:
            restore_requires_grad(trainability)

        self._loss_plotting(
            train_losses, valid_losses, train_losses_base, train_losses_reg
        )
        return train_losses, valid_losses, train_losses_base, train_losses_reg

    def _resume_from_best_state(self, model: BaseModel) -> None:
        """Load the best validation state of the pretraining and layer phases."""
        if getattr(self, "best_state_is_initial", False):
            raise RuntimeError(
                f"{self.logger_info_prefix}no pretraining or layer epoch had a "
                "finite validation loss, so final tuning would restart from the "
                "untrained weights."
            )
        self._unwrap_model(model).load_state_dict(self.best_state_dict)

    def _run_final_tuning(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict:
        logger.info(
            f"Starting final full-network tuning "
            f"for {self.final_tune_epochs} epochs."
        )
        default_epochs = self.epochs
        self.epochs = self.final_tune_epochs
        self.filename_prefix = "final_tune_"
        self.logger_info_prefix = "[Final Tune] "
        try:
            final_results = super().train(model, train_data, valid_data)
        finally:
            self.epochs = default_epochs
            self.filename_prefix = ""
        return final_results

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train layers sequentially from soma to input.

        This method trains each layer of the network separately, starting from the layer
        closest to the soma and moving outward toward the input.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)

        Returns:
            Training results dictionary
        """

        self._require_single_process(
            "it freezes all but one layer after DistributedDataParallel has "
            "registered every trainable parameter"
        )
        core = getattr(self._unwrap_model(model), "core_network", None)
        if not isinstance(core, ExcitationInhibitionNetwork):
            # Previously this silently ran standard training instead.
            raise ValueError(
                "freeze_layers trains the layers of a DendriNet "
                "(ExcitationInhibitionNetwork) core one at a time; "
                f"{type(core).__name__} cores are not supported. Use strategy "
                "'standard'."
            )

        net_obj: ExcitationInhibitionNetwork = core
        n_layers = len(net_obj.layers)

        total_epochs = _layerwise_total_epochs(
            self.pretrain_epochs,
            n_layers,
            self.epochs_per_layer,
            self.final_tune_epochs,
        )
        with self._run_epoch_count(total_epochs):
            self._initialize_attributes(model, total_epochs)

            (
                pretrain_train_losses,
                pretrain_valid_losses,
                pretrain_train_losses_base,
                pretrain_train_losses_reg,
            ) = self._pretrain(model, train_data, valid_data)

            learning_strategy, layer_plan = _ordered_layer_training_plan(
                net_obj.layers,
                self.reverse_training,
            )

            logger.info(
                f"Training with {learning_strategy} strategy: "
                f"found {n_layers} layers. "
            )

            train_loader, valid_loader = self._dataloaders(train_data, valid_data)

            train_losses, valid_losses, train_losses_base, train_losses_reg = (
                self._train_layer_plan(
                    model,
                    train_loader,
                    valid_loader,
                    layer_plan,
                    learning_strategy,
                )
            )

            logger.info(
                f"Finished {learning_strategy} training. Final tuning starts from "
                "the best state so far."
            )
            self._resume_from_best_state(model)

            train_losses, valid_losses, train_losses_base, train_losses_reg = (
                _prepend_loss_histories(
                    (
                        pretrain_train_losses,
                        pretrain_valid_losses,
                        pretrain_train_losses_base,
                        pretrain_train_losses_reg,
                    ),
                    (
                        train_losses,
                        valid_losses,
                        train_losses_base,
                        train_losses_reg,
                    ),
                )
            )

            final_results = self._run_final_tuning(model, train_data, valid_data)

            final_epochs = len(final_results.get("train_losses", []))
            _prepend_loss_histories_to_results(
                final_results,
                (
                    train_losses,
                    valid_losses,
                    train_losses_base,
                    train_losses_reg,
                ),
            )
            layer_epochs = len(train_losses) - len(pretrain_train_losses)
            final_results["phases"] = [
                {"name": "pretraining", "epochs": len(pretrain_train_losses)},
                {"name": "layers", "epochs": layer_epochs},
                {"name": "final_tuning", "epochs": final_epochs},
            ]
            return final_results


__all__ = ["LayerWiseTrainer"]
