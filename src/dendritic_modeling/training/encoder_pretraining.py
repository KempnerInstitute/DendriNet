"""Reconstruction pretraining of autoencoder encoders.

``training.encoder_pretraining`` (:class:`EncoderPretrainingConfig`) trains a
whole autoencoder to reconstruct the training inputs before the main phase;
its encoder then serves as the model's encoder. The pretraining runs on the
standard trainer, so early stopping keeps the state with the lowest
validation reconstruction loss.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from dendritic_modeling.config.training import EncoderPretrainingConfig, OptimizerConfig
from dendritic_modeling.networks.architectures.classical.autoencoder import (
    BaseAutoencoder,
    BaseVariationalAutoencoder,
)
from dendritic_modeling.training.optimizers.factory import create_optimizer
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


class ReconstructionDataset(Dataset):
    """The inputs of ``dataset`` as both input and target; labels are dropped."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = self.dataset[index][0]
        return inputs, inputs


def pretrain_autoencoder(
    autoencoder: BaseAutoencoder,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    settings: EncoderPretrainingConfig,
    *,
    save_path: Optional[str] = None,
    seed: int = 0,
    loader_seed: Optional[int] = None,
    device: Optional[str | torch.device] = None,
) -> dict[str, Any]:
    """Train ``autoencoder`` to reconstruct the inputs of ``train_dataset``.

    The datasets yield ``(input, label)`` pairs; the labels are not used.
    ``valid_dataset`` selects the kept state when ``settings.early_stopping``
    is on. The autoencoder is trained in place, on ``device`` (default: the
    GPU when one is available). Returns the best epoch, its validation loss
    and the loss of every epoch.
    """
    if not isinstance(autoencoder, BaseAutoencoder):
        raise TypeError(
            "pretrain_autoencoder trains an autoencoder, got "
            f"{type(autoencoder).__name__}"
        )
    if settings.loss_function == "vae_elbo" and not isinstance(
        autoencoder, BaseVariationalAutoencoder
    ):
        raise ValueError(
            "training.encoder_pretraining.loss_function 'vae_elbo' needs a "
            f"variational autoencoder, got {type(autoencoder).__name__}"
        )
    optimizer = create_optimizer(
        [{"params": list(autoencoder.parameters()), "lr": settings.learning_rate}],
        OptimizerConfig(
            name=settings.optimizer,
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
        ),
    )
    trainer = Trainer(
        optimizer,
        loss_function=settings.loss_function,
        epochs=settings.epochs,
        early_stopping=settings.early_stopping,
        patience=settings.patience,
        batch_size=settings.batch_size,
        grad_clip_value=settings.grad_clip_value,
        load_best_state_dict=settings.early_stopping,
        save_path=save_path,
        save_best_checkpoint="never",
        plot_losses=False,
        suppress_prints=True,
        task="regression",
        seed=seed,
        loader_seed=loader_seed,
    )
    if device is not None:
        trainer.device = device
    results = trainer.train(
        autoencoder,
        ReconstructionDataset(train_dataset),
        ReconstructionDataset(valid_dataset),
    )
    summary = {
        "epochs": settings.epochs,
        "loss_function": settings.loss_function,
        "best_epoch": results.get("best_epoch"),
        "best_valid_loss": float(results.get("best_loss")),
        "train_losses": [float(value) for value in results.get("train_losses", [])],
        "valid_losses": [float(value) for value in results.get("valid_losses", [])],
    }
    logger.info(
        "Encoder pretraining: %d epochs, best validation %s %.6g at epoch %s",
        len(summary["train_losses"]),
        settings.loss_function,
        summary["best_valid_loss"],
        summary["best_epoch"],
    )
    return summary


__all__ = ["ReconstructionDataset", "pretrain_autoencoder"]
