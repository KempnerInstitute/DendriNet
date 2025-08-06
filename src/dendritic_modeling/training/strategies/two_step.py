"""
Two-Step Training Strategies
============================

This module contains training strategies that implement two-step learning approaches,
where models are first pre-trained with local supervision and then fine-tuned end-to-end.
"""

import logging

import torch
import torch.nn.functional as functional
from torch.nn.utils import clip_grad_value_

logger = logging.getLogger(__name__)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.strategies.standard import Trainer


def _initialize_local_heads(net_obj, model, device):
    """Initialize local supervision heads for each layer."""
    local_heads = []

    for i, layer in enumerate(net_obj.layers):
        # Determine output dimension of current layer
        out_dim = getattr(layer, "out_features", None)
        if out_dim is None:
            excit_sizes = getattr(net_obj, "excitatory_layer_sizes", None)
            if isinstance(excit_sizes, list) and i < len(excit_sizes):
                out_dim = excit_sizes[i]
            else:
                # Infer from dummy forward pass
                try:
                    dummy = torch.randn(1, getattr(net_obj, "input_dim", 784)).to(
                        device
                    )
                except Exception:
                    dummy = torch.randn(1, 784).to(device)
                dummy_out = layer(dummy)
                if isinstance(dummy_out, tuple):
                    dummy_out = dummy_out[0]
                out_dim = dummy_out.size(1)

        # Create local head
        if hasattr(layer, "weight"):
            device_for_head = layer.weight.device
        else:
            device_for_head = next(model.parameters()).device

        head = torch.nn.Linear(out_dim, getattr(model, "output_dim", 10)).to(
            device_for_head
        )
        local_heads.append(head)

    return local_heads


class TwoStepTrainer(Trainer):
    """
    Two-Step Training Strategy for dendritic networks.

    This trainer implements a two-phase training approach:
    1. Pre-training phase: Each layer is trained locally with local supervision heads
    2. Main training phase: End-to-end training of the full network

    This approach can help with gradient flow and learning in deep dendritic networks.
    """

    pretrain_epochs: int = 5

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train the model using two-step approach.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """
        # Get network object
        if hasattr(model, "net"):
            net_obj = model.net
        elif hasattr(model, "network"):
            net_obj = model.network
        else:
            net_obj = model

        # Check if model has layers for local training
        if not hasattr(net_obj, "layers"):
            logger.info("No net.layers found => falling back to standard training.")
            return super().train(model, train_data, valid_data)

        # Training setup
        train_losses = []
        valid_losses = []

        train_loader, valid_loader = self._dataloaders(train_data, valid_data)

        self._initialize_attributes(model, self.pretrain_epochs + self.epochs)

        if not hasattr(self, "local_heads"):
            self.local_heads = _initialize_local_heads(net_obj, model, self.device)

        # Phase 1: Local pretraining
        logger.info(
            "=== TwoStepTrainer: Stage 1 "
            f"(local pretraining) for {self.pretrain_epochs} epochs ==="
        )
        n_layers = len(net_obj.layers)

        self.filename_prefix = "twostep_stage1_"
        for _ in range(1, self.pretrain_epochs + 1):
            self.epoch_counter += 1
            epoch_loss_accum = 0.0
            batch_counter = 0

            # Train each layer with its local head
            for layer_idx in range(n_layers):
                # Freeze all parameters
                for param in model.parameters():
                    param.requires_grad = False

                # Unfreeze current layer
                if hasattr(net_obj.layers[layer_idx], "weight"):
                    net_obj.layers[layer_idx].weight.requires_grad = True
                if (
                    hasattr(net_obj.layers[layer_idx], "bias")
                    and net_obj.layers[layer_idx].bias is not None
                ):
                    net_obj.layers[layer_idx].bias.requires_grad = True

                # Unfreeze local head
                local_head: torch.nn.Module = self.local_heads[layer_idx]
                for param in local_head.parameters():
                    param.requires_grad = True

                # Train on batches
                for x_batch, y_batch in train_loader:
                    x_batch: torch.Tensor
                    y_batch: torch.Tensor
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass through layers up to current layer
                    h_batch = x_batch
                    for j in range(layer_idx + 1):
                        h_batch = net_obj.layers[j](h_batch)
                        if isinstance(h_batch, tuple):
                            h_batch = h_batch[0]
                        if hasattr(net_obj.layers[j], "activation"):
                            h_batch = net_obj.layers[j].activation(h_batch)

                    # Local supervision
                    loss: torch.Tensor = self.loss_function(
                        local_head, h_batch, y_batch
                    )
                    self.optimizer.zero_grad()
                    loss.backward()
                    clip_grad_value_(model.parameters(), self.grad_clip_value)
                    self.optimizer.step()

                    epoch_loss_accum += loss.item()
                    batch_counter += 1

            avg_loss = epoch_loss_accum / batch_counter if batch_counter > 0 else 0.0
            train_losses.append(avg_loss)

            # Validation
            with torch.no_grad():
                valid_loss = 0
                for x_batch, y_batch in valid_loader:
                    x_batch: torch.Tensor
                    y_batch: torch.Tensor
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    loss = self.loss_function(model, x_batch, y_batch)
                    valid_loss += loss.item()

            valid_loss /= len(valid_loader)
            valid_losses.append(valid_loss)

            self._update_best(model, valid_loss)

            self._analyze()

            self._tracking(
                avg_loss, valid_loss, "[TwoStep-Pretrain] ", {"phase": "pretrain"}
            )

            if self._check_early_stopping():
                break

        self._loss_plotting(train_losses, valid_losses)

        # Phase 2: End-to-end training
        logger.info(
            "=== TwoStepTrainer: "
            "Stage 2 (end-to-end training) "
            f"for {self.epochs} epochs ==="
        )

        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

        self.filename_prefix = "twostep_stage2_"
        final_results = super().train(model, train_data, valid_data)
        self.filename_prefix = ""

        final_results["train_losses"] = train_losses + final_results["train_losses"]
        final_results["valid_losses"] = valid_losses + final_results["valid_losses"]

        return final_results


class TwoStepTrainerWithKL(Trainer):
    """
    Two-Step Training Strategy with KL Divergence Loss.

    Similar to TwoStepTrainer but uses a combined loss function during pre-training
    that includes both cross-entropy and KL divergence terms for better regularization.
    """

    pretrain_epochs: int = 5
    kl_weight: float = 0.1

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ):
        """
        Train the model using two-step approach with KL divergence.

        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing training results and metrics
        """
        # Get network object
        if hasattr(model, "net"):
            net_obj = model.net
        elif hasattr(model, "network"):
            net_obj = model.network
        else:
            net_obj = model

        # Check if model has layers for local training
        if not hasattr(net_obj, "layers"):
            logger.info("No net.layers found => falling back to standard training.")
            return super().train(model, train_data, valid_data)

        # Training setup
        train_losses = []
        valid_losses = []

        train_loader, valid_loader = self._dataloaders(train_data, valid_data)

        self._initialize_attributes(model, self.pretrain_epochs + self.epochs)

        if not hasattr(self, "local_heads"):
            self.local_heads = _initialize_local_heads(net_obj, model, self.device)

        # Phase 1: Local pretraining with KL divergence
        logger.info(
            "=== TwoStepTrainerWithKL: "
            "Stage 1 (local pretraining with KL) "
            f"for {self.pretrain_epochs} epochs ==="
        )
        n_layers = len(net_obj.layers)

        self.filename_prefix = "twostep_kl_stage1_"
        for _ in range(1, self.pretrain_epochs + 1):
            self.epoch_counter += 1
            epoch_loss_accum = 0.0
            batch_counter = 0

            # Train each layer with its local head
            for layer_idx in range(n_layers):
                # Freeze all parameters
                for param in model.parameters():
                    param.requires_grad = False

                # Unfreeze current layer
                if hasattr(net_obj.layers[layer_idx], "weight"):
                    net_obj.layers[layer_idx].weight.requires_grad = True
                if (
                    hasattr(net_obj.layers[layer_idx], "bias")
                    and net_obj.layers[layer_idx].bias is not None
                ):
                    net_obj.layers[layer_idx].bias.requires_grad = True

                # Unfreeze local head
                local_head: torch.nn.Module = self.local_heads[layer_idx]
                for param in local_head.parameters():
                    param.requires_grad = True

                # Train on batches
                for x_batch, y_batch in train_loader:
                    x_batch: torch.Tensor
                    y_batch: torch.Tensor
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass through layers up to current layer
                    h_batch = x_batch
                    for j in range(layer_idx + 1):
                        h_batch = net_obj.layers[j](h_batch)
                        if isinstance(h_batch, tuple):
                            h_batch = h_batch[0]
                        if hasattr(net_obj.layers[j], "activation"):
                            h_batch = net_obj.layers[j].activation(h_batch)

                    # Local head predictions
                    local_logits = local_head(h_batch)

                    # Get full model predictions for KL divergence
                    with torch.no_grad():
                        full_logits = model(x_batch)

                    # Combined loss: cross-entropy + KL divergence
                    ce_loss = functional.cross_entropy(local_logits, y_batch)
                    kl_loss = functional.kl_div(
                        functional.log_softmax(local_logits, dim=1),
                        functional.softmax(full_logits, dim=1),
                        reduction="batchmean",
                    )
                    loss = ce_loss + self.kl_weight * kl_loss

                    self.optimizer.zero_grad()
                    loss.backward()
                    clip_grad_value_(model.parameters(), self.grad_clip_value)
                    self.optimizer.step()

                    epoch_loss_accum += loss.item()
                    batch_counter += 1

            avg_loss = epoch_loss_accum / batch_counter if batch_counter > 0 else 0.0
            train_losses.append(avg_loss)

            # Validation
            with torch.no_grad():
                valid_loss = 0
                for x_batch, y_batch in valid_loader:
                    x_batch: torch.Tensor
                    y_batch: torch.Tensor
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    loss: torch.Tensor = self.loss_function(model, x_batch, y_batch)
                    valid_loss += loss.item()

            valid_loss /= len(valid_loader)
            valid_losses.append(valid_loss)

            self._update_best(model, valid_loss)

            self._analyze()

            self._tracking(
                avg_loss, valid_loss, "[TwoStepKL-Pretrain] ", {"phase": "pretrain_kl"}
            )

            if self._check_early_stopping():
                break

        self._loss_plotting(train_losses, valid_losses)

        # Phase 2: End-to-end training (same as TwoStepTrainer)
        logger.info(
            "=== TwoStepTrainerWithKL: "
            "Stage 2 (end-to-end training) "
            f"for {self.epochs} epochs ==="
        )

        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True

        self.filename_prefix = "twostep_kl_stage2_"
        final_results = super().train(model, train_data, valid_data)
        self.filename_prefix = ""

        final_results["train_losses"] = train_losses + final_results["train_losses"]
        final_results["valid_losses"] = valid_losses + final_results["valid_losses"]

        return final_results
