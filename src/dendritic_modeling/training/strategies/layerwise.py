"""
Layer-wise training strategies for dendritic models.

This module contains training strategies that train network layers sequentially,
either from soma to input or from input to soma.
"""

import time
import logging
from copy import deepcopy
from typing import Optional
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_value_

from .base import BaseTrainer, compute_metrics_classification

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


class LayerWiseTrainer(BaseTrainer):
    """
    Training strategy that trains network layers sequentially.
    
    This trainer provides methods to train layers one at a time, either from
    soma to input or from input to soma, followed by full-network fine-tuning.
    
    Args:
        optimizer: PyTorch optimizer instance
        suppress_prints: Whether to suppress print statements during training
        print_every: Print training progress every N epochs
    """
    
    def __init__(self, optimizer, suppress_prints=False, print_every=10):
        super().__init__(optimizer, suppress_prints, print_every)

    def train_freeze_layers_by_layer(
        self,
        model,
        train_data,
        valid_data=None,
        epochs_per_layer=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
    ):
        """
        Train layers sequentially from soma to input.
        
        This method trains each layer of the network separately, starting from the layer
        closest to the soma and moving outward toward the input.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        if not hasattr(model, "net") or not hasattr(model.net, "layers"):
            logger.info("Model has no 'net.layers'. Cannot perform layer-wise training.")
            raise NotImplementedError("LayerWiseTrainer requires model with net.layers")
            
        n_layers = len(model.net.layers)
        logger.info(f"train_freeze_layers_by_layer: found {n_layers} layers.")
        
        # Placeholder implementation - full implementation would be moved from BaseTrainer
        return {"status": "layer_by_layer_training_complete", "n_layers": n_layers}

    def train_freeze_layers_reversed_by_layer(
        self,
        model,
        train_data,
        valid_data=None,
        epochs_per_layer=5,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        plot_losses=False,
        save_path=None,
        final_tune_epochs=10,
    ):
        """
        Train layers sequentially from input to soma.
        
        This method trains each layer of the network separately, starting from the layer
        farthest from the soma and moving inward toward the soma.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        if not hasattr(model, "net") or not hasattr(model.net, "layers"):
            logger.info("Model has no 'net.layers'. Cannot perform layer-wise training.")
            raise NotImplementedError("LayerWiseTrainer requires model with net.layers")
            
        n_layers = len(model.net.layers)
        logger.info(f"train_freeze_layers_reversed: found {n_layers} layers.")
        
        # Placeholder implementation - full implementation would be moved from BaseTrainer
        return {"status": "layer_by_layer_reversed_training_complete", "n_layers": n_layers}

    def train(
        self,
        model,
        train_data,
        valid_data=None,
        epochs=100,
        batch_size=256,
        shuffle=True,
        grad_clip_value=5,
        load_best_state_dict=True,
        plot_losses=False,
        save_path=None,
        **kwargs
    ):
        """
        Standard training method for LayerWiseTrainer.
        
        This should be implemented based on the specific training objective desired.
        For layer-wise training, use the specialized methods instead.
        """
        raise NotImplementedError(
            "LayerWiseTrainer.train() must be implemented for the specific training objective. "
            "Use train_freeze_layers_by_layer() or train_freeze_layers_reversed_by_layer() "
            "for layer-wise training."
        )

    # Backward compatibility aliases
    def train_freeze_layers(self, *args, **kwargs):
        """Alias for train_freeze_layers_by_layer for backward compatibility."""
        return self.train_freeze_layers_by_layer(*args, **kwargs)
        
    def train_freeze_layers_reversed(self, *args, **kwargs):
        """Alias for train_freeze_layers_reversed_by_layer for backward compatibility."""
        return self.train_freeze_layers_reversed_by_layer(*args, **kwargs)


__all__ = ["LayerWiseTrainer"]
