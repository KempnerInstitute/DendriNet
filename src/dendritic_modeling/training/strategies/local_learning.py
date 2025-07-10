"""
Local learning training strategies.

This module implements local learning approaches for training dendritic networks.
"""

# typing imports removed - not needed
import torch

from dendritic_modeling import logger
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import (
    DendriticBranchLayer,
    ExcitationInhibitionNetwork,
)
from dendritic_modeling.training.strategies.standard import Trainer


class LocalCreditAssignment(Trainer):
    """
    Local Credit Assignment training strategy for dendritic networks.

    This trainer implements local learning rules that assign credit to different
    parts of the network based on local feedback, particularly useful for
    dendritic network architectures.
    """
    pretrain_epochs: int = 10
    reverse_training: bool = False
    epochs_per_branch: int = 5
    final_tune_epochs: int = 10

    def _pretrain(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset
    ) -> tuple[list[float], list[float]]:
        """
        Pretrain the model.
        """
        pretrain_train_losses = []
        pretrain_valid_losses = []

        if self.pretrain_epochs > 0:
            logger.info(
                f"Starting full network pretraining "
                f"for {self.pretrain_epochs} epochs..."
            )
            default_epochs = self.epochs
            self.epochs = self.pretrain_epochs
            self.filename_prefix = "pretrain_"
            pretrain_results = super().train(model, train_data, valid_data)
            self.epochs = default_epochs
            self.filename_prefix = ""

            pretrain_train_losses = pretrain_results.get("train_losses", [])
            pretrain_valid_losses = pretrain_results.get("valid_losses", [])

        return pretrain_train_losses, pretrain_valid_losses

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset
    ):
        """
        Train branch layers sequentially from soma to input.

        This method trains each layer of the network separately, starting from the layer
        closest to the soma and moving outward toward the input.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset (optional)

        Returns:
            Training results dictionary
        """

        if not isinstance(model.net, ExcitationInhibitionNetwork):
            logger.info(
                f"model.net has no branch layers (type: {type(model.net)}). "
                "Falling back to normal train."
            )
            return super().train(model, train_data, valid_data)

        net_obj: ExcitationInhibitionNetwork = model.net
        n_branch_layers = 0
        for module in net_obj.modules():
            if isinstance(module, DendriticBranchLayer):
                n_branch_layers += 1

        total_epochs = (
            self.pretrain_epochs +
            n_branch_layers * self.epochs_per_branch +
            self.final_tune_epochs
        )
        self._initialize_attributes(model, total_epochs)

        pretrain_train_losses, pretrain_valid_losses = self._pretrain(
            model, train_data, valid_data
        )

        branch_layer_modules = []
        branch_layer_indices = []
        if self.reverse_training:
            learning_strategy = "Freeze-Branches-Reversed"
            branch_layer_idx = n_branch_layers - 1
            for module in reversed(list(net_obj.modules())):
                if isinstance(module, DendriticBranchLayer):
                    branch_layer_modules.append(module)
                    branch_layer_indices.append(branch_layer_idx)
                    branch_layer_idx -= 1
        else:
            learning_strategy = "Freeze-Branches"
            branch_layer_idx = 0
            for module in net_obj.modules():
                if isinstance(module, DendriticBranchLayer):
                    branch_layer_modules.append(module)
                    branch_layer_indices.append(branch_layer_idx)
                    branch_layer_idx += 1

        logger.info(
            f"Training with {learning_strategy} strategy: "
            f"found {n_branch_layers} branch layers. "
        )

        train_losses = []
        valid_losses = []

        train_loader, valid_loader = self._dataloaders(train_data, valid_data)

        self.filename_prefix = f"{learning_strategy.lower().replace('-', '_')}_"

        for branch_layer_idx, branch_layer_module in zip(
            branch_layer_indices, branch_layer_modules
        ):
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            branch_layer_module: DendriticBranchLayer
            for param in branch_layer_module.parameters():
                param.requires_grad = True

            logger.info(
                f"Training only branch layer {branch_layer_idx} for "
                f"{self.epochs_per_branch} epochs."
            )

            train_losses, valid_losses = self._run_training_loop(
                n_epochs=self.epochs_per_branch,
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_losses=train_losses,
                valid_losses=valid_losses,
                logger_info=f"[{learning_strategy}] branch={branch_layer_idx}, ",
                wandb_info={"branch": branch_layer_idx}
            )

        self._loss_plotting(train_losses, valid_losses)

        logger.info(
            f"Finished {learning_strategy} training. "
            "Unfreezing all layers and loading best state."
        )
        model.load_state_dict(self.best_state_dict)
        for param in model.parameters():
            param.requires_grad = True

        train_losses = pretrain_train_losses + train_losses
        valid_losses = pretrain_valid_losses + valid_losses

        # Final full-network tuning
        logger.info(
            f"Starting final full-network tuning "
            f"for {self.final_tune_epochs} epochs."
        )
        default_epochs = self.epochs
        self.epochs = self.final_tune_epochs
        self.filename_prefix = "final_tune_"
        final_results = super().train(model, train_data, valid_data)
        self.epochs = default_epochs
        self.filename_prefix = ""

        # Combine all results
        final_results["train_losses"] = train_losses + final_results.get(
            "train_losses", []
        )
        final_results["valid_losses"] = valid_losses + final_results.get(
            "valid_losses", []
        )
        return final_results

__all__ = ["LocalCreditAssignment"]
