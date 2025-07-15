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
        optimizer_config: Optional[Dict[str, Any]] = None,
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
            optimizer_config: Optimizer configuration
            **kwargs: Arguments passed to base Trainer
        """
        # Store configuration
        self.rank = rank
        self.world_size = world_size
        self.fsdp_config = fsdp_config or get_fsdp_config()
        self.checkpoint_dir = checkpoint_dir or f"checkpoints/fsdp_run_{time.time()}"
        self.save_every_n_epochs = save_every_n_epochs
        self.optimizer_config = optimizer_config or {}
        
        # Set device
        self.device = torch.device(f'cuda:{rank}')
        torch.cuda.set_device(rank)
        
        # Initialize parent with dummy optimizer (we'll create the real one after FSDP wrapping)
        dummy_optimizer = torch.optim.SGD([torch.zeros(1)], lr=0.01)
        super().__init__(optimizer=dummy_optimizer, **kwargs)
        
        # Create checkpoint directory
        if self.rank == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize loss function based on loss_function parameter
        loss_fn_name = kwargs.get('loss_function', 'cat_nll')
        if loss_fn_name == 'cat_nll':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss_fn_name == 'mse':
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(f"Unknown loss function: {loss_fn_name}")
    
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
    
    def create_optimizer(self, model: BaseModel) -> torch.optim.Optimizer:
        """Create optimizer with FSDP-aware parameter groups."""
        # Get all parameters first
        all_params = list(model.parameters())
        
        # If not using split params, use all parameters with single LR
        if not self.optimizer_config.get('split_params', False):
            return torch.optim.Adam(
                all_params,
                lr=self.optimizer_config.get('lr', 0.001),
                weight_decay=self.optimizer_config.get('weight_decay', 0.0)
            )
        
        # Otherwise, create parameter groups
        param_groups = []
        used_params = set()
        
        # TopK parameters
        topk_params = []
        for module in model.modules():
            if module.__class__.__name__ == 'TopKLinear':
                for param in module.parameters():
                    topk_params.append(param)
                    used_params.add(id(param))
        if topk_params:
            param_groups.append({
                'params': topk_params,
                'lr': self.optimizer_config.get('topk_lr', 0.001)
            })
        
        # BlockLinear parameters
        blocklinear_params = []
        for module in model.modules():
            if module.__class__.__name__ == 'BlockLinear':
                blocklinear_params.extend(module.parameters())
        if blocklinear_params:
            param_groups.append({
                'params': blocklinear_params,
                'lr': self.optimizer_config.get('blocklinear_lr', 0.001)
            })
        
        # Reactivation parameters
        reactivation_params = []
        for module in model.modules():
            if hasattr(module, 'm') and hasattr(module, 'b'):  # ParametricTanh
                reactivation_params.extend([module.m, module.b])
        if reactivation_params:
            param_groups.append({
                'params': reactivation_params,
                'lr': self.optimizer_config.get('reactivation_lr', 0.001)
            })
        
        # Decoder parameters
        decoder_params = []
        if hasattr(model, 'decoder_network'):
            decoder_params = list(model.decoder_network.parameters())
        if decoder_params:
            param_groups.append({
                'params': decoder_params,
                'lr': self.optimizer_config.get('decoder_lr', 0.001)
            })
        
        # All other parameters
        other_params = [p for p in all_params if id(p) not in used_params]
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': self.optimizer_config.get('lr', 0.001)
            })
        
        return torch.optim.Adam(
            param_groups,
            weight_decay=self.optimizer_config.get('weight_decay', 0.0)
        )
    
    def train(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Train model using FSDP with pre-created DataLoaders.
        
        Args:
            model: Model to train (will be wrapped with FSDP)
            train_loader: Training DataLoader with DistributedSampler
            valid_loader: Validation DataLoader with DistributedSampler
            
        Returns:
            Training results dictionary
        """
        # Wrap model with FSDP
        model = wrap_model_with_fsdp(model, self.fsdp_config)
        model = model.to(self.device)
        
        # Create optimizer after FSDP wrapping
        self.optimizer = self.create_optimizer(model)
        
        # Training loop
        train_losses = []
        valid_losses = []
        best_valid_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Set epoch for distributed sampler
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(epoch)
            
            # Training
            train_loss = self._epoch_train_fsdp(model, train_loader)
            train_losses.append(train_loss)
            
            # Validation
            valid_loss = None
            if valid_loader is not None:
                if hasattr(valid_loader.sampler, 'set_epoch'):
                    valid_loader.sampler.set_epoch(epoch)
                valid_loss = self._epoch_valid_fsdp(model, valid_loader)
                valid_losses.append(valid_loss)
                
                # Early stopping
                if self.early_stopping and valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    patience_counter = 0
                    if self.rank == 0:
                        save_fsdp_checkpoint(
                            model, self.optimizer, epoch, 
                            os.path.join(self.checkpoint_dir, 'best_model.pt')
                        )
                elif self.early_stopping:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        if self.rank == 0:
                            logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            # Logging
            epoch_time = time.time() - epoch_start
            if self.rank == 0 and epoch % self.print_every == 0:
                msg = f"Epoch {epoch}/{self.epochs} - Train Loss: {train_loss:.4f}"
                if valid_loss is not None:
                    msg += f" - Valid Loss: {valid_loss:.4f}"
                msg += f" - Time: {epoch_time:.2f}s"
                logger.info(msg)
            
            # Checkpointing
            if self.rank == 0 and (epoch + 1) % self.save_every_n_epochs == 0:
                save_fsdp_checkpoint(
                    model, self.optimizer, epoch,
                    os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
                )
        
        # Final save
        if self.rank == 0:
            save_fsdp_checkpoint(
                model, self.optimizer, self.epochs - 1,
                os.path.join(self.checkpoint_dir, 'final_model.pt')
            )
        
        return {
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'best_valid_loss': best_valid_loss,
            'final_train_loss': train_losses[-1] if train_losses else None,
            'final_valid_loss': valid_losses[-1] if valid_losses else None
        }
    
    def _epoch_train_fsdp(self, model: BaseModel, train_loader: DataLoader) -> float:
        """Training epoch with FSDP."""
        model.train()
        total_loss = 0
        num_batches = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            outputs = model(x_batch)
            loss = self.loss_fn(outputs, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip_value is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    self.grad_clip_value
                )
            
            # Optimizer step
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Average loss across batches
        avg_loss = total_loss / num_batches
        
        # Reduce loss across all processes
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / self.world_size
        
        return avg_loss
    
    def _epoch_valid_fsdp(self, model: BaseModel, valid_loader: DataLoader) -> float:
        """Validation epoch with FSDP."""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                outputs = model(x_batch)
                loss = self.loss_fn(outputs, y_batch)
                
                total_loss += loss.item()
                num_batches += 1
        
        # Average loss across batches
        avg_loss = total_loss / num_batches
        
        # Reduce loss across all processes
        avg_loss_tensor = torch.tensor(avg_loss, device=self.device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / self.world_size
        
        return avg_loss
    
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