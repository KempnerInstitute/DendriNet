"""
FSDP-Enabled Training Strategy for Large Dendritic Models
=========================================================

This module extends the standard trainer to support FSDP for models
that are too large to fit on a single GPU.
"""

import os
import time
from typing import Optional, Dict, Any
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from dendritic_modeling import logger
from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.training.fsdp_utils import (
    wrap_model_with_fsdp,
    save_fsdp_checkpoint,
    load_fsdp_checkpoint,
    get_fsdp_config,
)


class FSDPTrainer(Trainer):
    """
    FSDP-enabled trainer for large dendritic models.
    
    This trainer extends the standard trainer with FSDP support,
    allowing training of models that don't fit on a single GPU.
    """
    
    def __init__(
        self,
        rank: int,
        world_size: int,
        fsdp_config: Optional[Dict[str, Any]] = None,
        checkpoint_dir: Optional[str] = None,
        save_every_n_epochs: int = 5,
        **kwargs
    ):
        """
        Initialize FSDP trainer.
        
        Args:
            rank: Process rank
            world_size: Total number of processes
            fsdp_config: FSDP configuration
            checkpoint_dir: Directory for checkpoints
            save_every_n_epochs: Checkpoint frequency
            **kwargs: Arguments passed to base Trainer
        """
        super().__init__(**kwargs)
        
        self.rank = rank
        self.world_size = world_size
        self.fsdp_config = fsdp_config or get_fsdp_config()
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_epochs = save_every_n_epochs
        
        # Override device
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
        
    def _wrap_model(self, model: BaseModel) -> BaseModel:
        """Wrap model with FSDP."""
        # Move base model to device first
        model = model.to(self.device)
        
        # Wrap with FSDP
        if hasattr(model, 'net'):
            # For models with separate encoder/decoder
            model.net = wrap_model_with_fsdp(model.net, self.fsdp_config)
            
            # Optionally wrap encoder/decoder if they're large
            if hasattr(model, 'encoder_network'):
                param_count = sum(p.numel() for p in model.encoder_network.parameters())
                if param_count > 10_000_000:  # > 10M parameters
                    model.encoder_network = wrap_model_with_fsdp(
                        model.encoder_network, self.fsdp_config
                    )
                    
            if hasattr(model, 'decoder_network'):
                param_count = sum(p.numel() for p in model.decoder_network.parameters())
                if param_count > 10_000_000:  # > 10M parameters
                    model.decoder_network = wrap_model_with_fsdp(
                        model.decoder_network, self.fsdp_config
                    )
        else:
            # Wrap entire model
            model = wrap_model_with_fsdp(model, self.fsdp_config)
            
        return model
    
    def _create_distributed_data_loaders(
        self,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset
    ) -> tuple[DataLoader, DataLoader]:
        """Create distributed data loaders with DistributedSampler."""
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        valid_sampler = DistributedSampler(
            valid_data,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        # Create data loaders
        # Note: batch_size is per-GPU batch size
        train_loader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True  # Important for FSDP
        )
        
        valid_loader = DataLoader(
            valid_data,
            batch_size=self.batch_size,
            sampler=valid_sampler,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, valid_loader
    
    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict:
        """
        Train model with FSDP.
        
        Args:
            model: Model to train (will be wrapped with FSDP)
            train_data: Training dataset
            valid_data: Validation dataset
            
        Returns:
            Training results dictionary
        """
        # Wrap model with FSDP
        model = self._wrap_model(model)
        
        # Create distributed data loaders
        train_loader, valid_loader = self._create_distributed_data_loaders(
            train_data, valid_data
        )
        
        # Initialize training state
        train_losses = []
        valid_losses = []
        self._initialize_attributes(model, self.epochs)
        
        # Training loop
        for epoch in range(1, self.epochs + 1):
            # Set epoch for distributed sampler
            train_loader.sampler.set_epoch(epoch)
            
            # Run epoch
            train_loss = self._epoch_train_fsdp(model, train_loader)
            valid_loss = self._epoch_valid_fsdp(model, valid_loader)
            
            # Aggregate losses across all processes
            train_loss = self._reduce_loss(train_loss)
            valid_loss = self._reduce_loss(valid_loss)
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            # Update best model (on all ranks to stay synchronized)
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_epoch = epoch
                
            # Save checkpoint periodically (rank 0 only)
            if epoch % self.save_every_n_epochs == 0 and self.checkpoint_dir:
                self._save_checkpoint(model, epoch)
            
            # Logging (rank 0 only)
            if self.rank == 0:
                elapsed = time.time() - self.start_time
                logger.info(
                    f"[FSDP] epoch={epoch}/{self.epochs}, "
                    f"train_loss={train_loss:.4f}, "
                    f"valid_loss={valid_loss:.4f}, "
                    f"elapsed={elapsed:.1f}s"
                )
            
            # Early stopping check
            if self._check_early_stopping():
                break
        
        # Save final checkpoint
        if self.checkpoint_dir:
            self._save_checkpoint(model, epoch, is_final=True)
        
        results = {
            "best_epoch": self.best_epoch,
            "best_loss": self.best_loss,
            "train_losses": train_losses,
            "valid_losses": valid_losses,
        }
        
        # Plot losses (rank 0 only)
        if self.rank == 0:
            self._loss_plotting(train_losses, valid_losses)
        
        return results
    
    def _epoch_train_fsdp(self, model: BaseModel, train_loader: DataLoader) -> float:
        """Training epoch with FSDP."""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            loss = self.loss_function(model, x_batch, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (FSDP-aware)
            if self.grad_clip_value:
                model.clip_grad_norm_(self.grad_clip_value)
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _epoch_valid_fsdp(self, model: BaseModel, valid_loader: DataLoader) -> float:
        """Validation epoch with FSDP."""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                loss = self.loss_function(model, x_batch, y_batch)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def _reduce_loss(self, loss: float) -> float:
        """Reduce loss across all processes."""
        loss_tensor = torch.tensor(loss, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
        return loss_tensor.item()
    
    def _save_checkpoint(self, model: BaseModel, epoch: int, is_final: bool = False):
        """Save FSDP checkpoint."""
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            
        checkpoint_name = "final_checkpoint.pt" if is_final else f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        save_fsdp_checkpoint(
            model=model,
            optimizer=self.optimizer,
            epoch=epoch,
            save_path=checkpoint_path,
            rank=self.rank
        ) 