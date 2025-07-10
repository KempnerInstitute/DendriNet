"""
Specialized training strategies.

This module implements specialized training approaches including voltage stabilization
and other advanced training techniques for dendritic networks.
"""

import torch

from dendritic_modeling import logger
from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.loss.functions import VoltageStabilizationLoss
from dendritic_modeling.training.strategies import Trainer


class VoltageStabilizationTrainer(Trainer):
    """
    Voltage Stabilization Training Strategy.

    This trainer includes a warm-up phase that stabilizes dendritic voltages
    to target values before proceeding with normal training. This can help
    with training stability in dendritic networks.
    """
    pretrain_epochs: int = 1
    target_voltage: float = 0.5

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset
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
            target_voltage=self.target_voltage,
            reduction=self.loss_function.reduction
        )

        handles = voltage_stabilization_loss.attach_forward_hooks(model)

        default_loss_fn = self.loss_function
        self.loss_function = voltage_stabilization_loss
        default_epochs = self.epochs
        self.epochs = max(self.pretrain_epochs, 1)
        self.filename_prefix = "voltage_stabilization_"

        logger.info(
            "=== VoltageStabilization: "
            f"Starting pretraining phase for {self.pretrain_epochs} epochs ==="
        )
        pretrain_results = super().train(model, train_data, valid_data)
        voltage_stabilization_loss.remove_forward_hooks(handles)

        self.loss_function = default_loss_fn
        self.epochs = default_epochs
        self.filename_prefix = ""
        self.best_loss = float("inf")

        logger.info(
            "=== VoltageStabilization: "
            f"Starting normal training for {self.epochs} epochs ==="
        )
        final_results = super().train(model, train_data, valid_data)
        final_results["train_losses"] = (
            pretrain_results["train_losses"] + final_results["train_losses"]
        )
        final_results["valid_losses"] = (
            pretrain_results["valid_losses"] + final_results["valid_losses"]
        )
        return final_results

class TrainOnlyMReactivation(Trainer):
    """
    Training strategy that focuses only on M-reactivation parameters.

    This trainer freezes log_b parameters and trains all other parameters,
    which can be useful for specific dendritic network configurations.
    """

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset
    ):
        """
        Train the model with M-reactivation focus.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """

        logger.info(
            "TrainOnlyMReactivation: "
            "Freezing log_b and training all other parameters."
        )

        # Freeze log_b parameters, unfreeze all others
        for name, param in model.named_parameters():
            if "log_b" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.filename_prefix = "train_only_m_reactivation_"
        results = super().train(model, train_data, valid_data)
        return results
