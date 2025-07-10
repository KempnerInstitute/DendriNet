"""
Layer-wise training strategies for dendritic models.

This module contains training strategies that train network layers sequentially,
either from soma to input or from input to soma.
"""

# typing imports removed - not needed  
import torch

from dendritic_modeling import logger
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import ExcitationInhibitionNetwork
from dendritic_modeling.training.strategies import Trainer

class LayerWiseTrainer(Trainer):
    """
    Training strategy that trains network layers sequentially.

    This trainer provides methods to train layers one at a time, either from
    soma to input or from input to soma, followed by full-network fine-tuning.

    """
    pretrain_epochs: int = 10
    reverse_training: bool = False
    epochs_per_layer: int = 5
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

        if not isinstance(model.net, ExcitationInhibitionNetwork):
            logger.info(
                "Model has no 'net.layers'. "
                "Falling back to normal train."
            )
            return super().train(model, train_data, valid_data)

        net_obj: ExcitationInhibitionNetwork = model.net
        n_layers = len(net_obj.layers)

        total_epochs = (
            self.pretrain_epochs +
            n_layers * self.epochs_per_layer +
            self.final_tune_epochs
        )
        self._initialize_attributes(model, total_epochs)

        pretrain_train_losses, pretrain_valid_losses = self._pretrain(
            model, train_data, valid_data
        )

        layer_modules = []
        layer_indices = []
        if self.reverse_training:
            learning_strategy = "Freeze-Layers-Reversed"
            for i, layer in enumerate(
                reversed(list(net_obj.layers))
            ):
                layer_modules.append(layer)
                layer_indices.append(n_layers - i - 1)
        else:
            learning_strategy = "Freeze-Layers"
            for i, layer in enumerate(net_obj.layers):
                layer_modules.append(layer)
                layer_indices.append(i)

        logger.info(
            f"Training with {learning_strategy} strategy: "
            f"found {n_layers} layers. "
        )

        train_losses = []
        valid_losses = []

        train_loader, valid_loader = self._dataloaders(train_data, valid_data)

        self.filename_prefix = f"{learning_strategy.lower().replace('-', '_')}_"

        for layer_idx, layer_module in zip(layer_indices, layer_modules):
            # Freeze all parameters
            for param in model.parameters():
                param.requires_grad = False
            layer_module: torch.nn.Module
            for param in layer_module.parameters():
                param.requires_grad = True

            logger.info(
                f"Training only layer {layer_idx} for "
                f"{self.epochs_per_layer} epochs."
            )

            train_losses, valid_losses = self._run_training_loop(
                n_epochs=self.epochs_per_layer,
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                train_losses=train_losses,
                valid_losses=valid_losses,
                logger_info=f"[{learning_strategy}] layer={layer_idx}, ",
                wandb_info={"layer": layer_idx}
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

__all__ = ["LayerWiseTrainer"]
