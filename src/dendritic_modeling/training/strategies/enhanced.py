"""
Enhanced Training Strategy with Regularization and Pruning
=========================================================

This module provides an enhanced trainer that supports:
- Flexible optimizer selection
- L1/L2 regularization with selective application
- Weight pruning with detailed reporting
- Re-evaluation after pruning
"""

import os
import time
from copy import deepcopy
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from dendritic_modeling import logger
from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.training.regularization import RegularizationManager, PruningManager
from dendritic_modeling.training.optimizers.factory import create_optimizer, validate_optimizer_config
from dendritic_modeling.config.training import (
    OptimizerConfig, RegularizationConfig, PruningConfig, ReportingConfig
)


class EnhancedTrainer(Trainer):
    """
    Enhanced trainer with regularization, pruning, and flexible optimization.
    
    This trainer extends the standard trainer to support:
    - Customizable optimizers (Adam, SGD, RMSprop, AdamW)
    - L1/L2 regularization with selective application
    - Weight pruning with threshold-based or magnitude-based strategies
    - Detailed reporting of network sparsity and pruning statistics
    - Re-evaluation after pruning
    """
    
    def __init__(
        self,
        optimizer_config: OptimizerConfig,
        regularization_config: Optional[RegularizationConfig] = None,
        pruning_config: Optional[PruningConfig] = None,
        reporting_config: Optional[ReportingConfig] = None,
        # Standard trainer arguments
        loss_function: str = "mse",
        epochs: int = 100,
        early_stopping: bool = False,
        patience: int = 10,
        batch_size: int = 256,
        shuffle: bool = True,
        grad_clip_value: float = 5.0,
        load_best_state_dict: bool = True,
        plot_losses: bool = False,
        save_path: Optional[str] = None,
        suppress_prints: bool = False,
        print_every: int = 10,
        analysis_manager: Optional[object] = None,
        save_every_n_epochs: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        enable_adaptive_clipping: bool = False,
        disable_hooks_with_amp: bool = False,
        **kwargs,
    ):
        # Validate optimizer config
        validate_optimizer_config(optimizer_config)
        
        # Store enhanced configs
        self.optimizer_config = optimizer_config
        self.regularization_config = regularization_config or RegularizationConfig()
        self.pruning_config = pruning_config or PruningConfig()
        self.reporting_config = reporting_config or ReportingConfig()
        
        # Initialize managers
        self.regularization_manager = RegularizationManager(self.regularization_config)
        self.pruning_manager = PruningManager(self.pruning_config)
        
        # Track original performance for comparison
        self.pre_pruning_performance = {}
        self.post_pruning_performance = {}
        
        # Initialize with dummy optimizer (will be replaced in train method)
        dummy_optimizer = torch.optim.Adam([torch.tensor(0.0)], lr=0.001)
        
        # Initialize parent class
        super().__init__(
            optimizer=dummy_optimizer,
            loss_function=loss_function,
            epochs=epochs,
            early_stopping=early_stopping,
            patience=patience,
            batch_size=batch_size,
            shuffle=shuffle,
            grad_clip_value=grad_clip_value,
            load_best_state_dict=load_best_state_dict,
            plot_losses=plot_losses,
            save_path=save_path,
            suppress_prints=suppress_prints,
            print_every=print_every,
            analysis_manager=analysis_manager,
            save_every_n_epochs=save_every_n_epochs,
            checkpoint_dir=checkpoint_dir,
            enable_adaptive_clipping=enable_adaptive_clipping,
            disable_hooks_with_amp=disable_hooks_with_amp,
            **kwargs,
        )
    
    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> Dict[str, Any]:
        """
        Enhanced training with regularization and pruning support.
        
        Args:
            model: The model to train
            train_data: Training dataset
            valid_data: Validation dataset
            
        Returns:
            Dictionary containing training results and pruning statistics
        """
        # Create the actual optimizer with model parameters
        self.optimizer = create_optimizer(model.parameters(), self.optimizer_config)
        
        # Log configuration
        self._log_enhanced_config()
        
        # Run standard training with regularization
        training_results = self._train_with_regularization(model, train_data, valid_data)
        
        # Apply pruning if enabled
        if self.pruning_config.enabled:
            pruning_results = self._apply_pruning_and_evaluate(model, train_data, valid_data)
            training_results.update(pruning_results)
        
        return training_results
    
    def _train_with_regularization(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> Dict[str, Any]:
        """Train the model with regularization."""
        # Store original loss function
        original_loss_function = self.loss_function
        
        # Wrap loss function to include regularization
        self.loss_function = self._create_regularized_loss_function(original_loss_function, model)
        
        # Run standard training
        results = super().train(model, train_data, valid_data)
        
        # Restore original loss function
        self.loss_function = original_loss_function
        
        return results
    
    def _create_regularized_loss_function(self, original_loss_fn, model):
        """Create a loss function that includes regularization."""
        def regularized_loss(model_input, x_batch, y_batch):
            # Compute base loss
            base_loss = original_loss_fn(model_input, x_batch, y_batch)
            
            # Add regularization
            reg_loss = self.regularization_manager.compute_regularization_loss(model)
            
            return base_loss + reg_loss
        
        return regularized_loss
    
    def _apply_pruning_and_evaluate(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> Dict[str, Any]:
        """Apply pruning and evaluate performance."""
        logger.info("Starting pruning phase...")
        
        # Store pre-pruning performance
        train_loader, valid_loader = self._dataloaders(train_data, valid_data)
        pre_pruning_loss, pre_pruning_acc = self._epoch_validate(model, valid_loader)
        self.pre_pruning_performance = {
            'loss': pre_pruning_loss,
            'accuracy': pre_pruning_acc
        }
        
        # Apply pruning
        pruning_stats = self.pruning_manager.prune_model(model)
        
        # Report pruning statistics
        if self.reporting_config.report_non_pruned:
            self.pruning_manager.print_pruning_report()
        
        # Save pruning statistics if requested
        if self.reporting_config.save_pruning_stats and self.save_path:
            stats_path = Path(self.save_path).parent / "pruning_stats.json"
            self.pruning_manager.save_pruning_stats(stats_path)
        
        # Re-evaluate after pruning
        post_pruning_results = {}
        if self.pruning_config.re_evaluate_after:
            logger.info("Re-evaluating pruned model...")
            post_pruning_loss, post_pruning_acc = self._epoch_validate(model, valid_loader)
            self.post_pruning_performance = {
                'loss': post_pruning_loss,
                'accuracy': post_pruning_acc
            }
            
            # Log performance comparison
            self._log_pruning_performance_comparison()
            
            post_pruning_results = {
                'post_pruning_loss': post_pruning_loss,
                'post_pruning_accuracy': post_pruning_acc,
                'pruning_performance_delta': {
                    'loss_change': post_pruning_loss - pre_pruning_loss,
                    'accuracy_change': post_pruning_acc - pre_pruning_acc
                }
            }
        
        return {
            'pruning_stats': pruning_stats,
            'pre_pruning_performance': self.pre_pruning_performance,
            'post_pruning_performance': self.post_pruning_performance,
            **post_pruning_results
        }
    
    def _log_enhanced_config(self):
        """Log the enhanced training configuration."""
        if not self.suppress_prints:
            print("\n" + "="*60)
            print("ENHANCED TRAINING CONFIGURATION")
            print("="*60)
            print(f"Optimizer: {self.optimizer_config.name}")
            print(f"Learning rate: {self.optimizer_config.lr}")
            print(f"Weight decay: {self.optimizer_config.weight_decay}")
            
            if self.regularization_config.l1_weight > 0 or self.regularization_config.l2_weight > 0:
                print(f"L1 regularization: {self.regularization_config.l1_weight}")
                print(f"L2 regularization: {self.regularization_config.l2_weight}")
                if self.regularization_config.selective.layers:
                    print(f"Regularization layers: {self.regularization_config.selective.layers}")
                if self.regularization_config.selective.weight_types != ["all"]:
                    print(f"Regularization weight types: {self.regularization_config.selective.weight_types}")
            
            if self.pruning_config.enabled:
                print(f"Pruning enabled: threshold = {self.pruning_config.threshold}")
                if self.pruning_config.selective.layers:
                    print(f"Pruning layers: {self.pruning_config.selective.layers}")
                if self.pruning_config.selective.weight_types != ["all"]:
                    print(f"Pruning weight types: {self.pruning_config.selective.weight_types}")
            
            print("="*60)
    
    def _log_pruning_performance_comparison(self):
        """Log performance comparison before and after pruning."""
        if not self.suppress_prints and self.pre_pruning_performance and self.post_pruning_performance:
            print("\n" + "="*50)
            print("PRUNING PERFORMANCE COMPARISON")
            print("="*50)
            print(f"Pre-pruning  - Loss: {self.pre_pruning_performance['loss']:.6f}, "
                  f"Accuracy: {self.pre_pruning_performance['accuracy']:.2f}%")
            print(f"Post-pruning - Loss: {self.post_pruning_performance['loss']:.6f}, "
                  f"Accuracy: {self.post_pruning_performance['accuracy']:.2f}%")
            
            loss_change = self.post_pruning_performance['loss'] - self.pre_pruning_performance['loss']
            acc_change = self.post_pruning_performance['accuracy'] - self.pre_pruning_performance['accuracy']
            
            print(f"Change       - Loss: {loss_change:+.6f}, Accuracy: {acc_change:+.2f}%")
            print("="*50)
    
    def dynamic_pruning_step(self, model: BaseModel, epoch: int) -> bool:
        """
        Apply dynamic pruning during training.
        
        Args:
            model: The model to prune
            epoch: Current epoch number
            
        Returns:
            True if pruning was applied, False otherwise
        """
        if (not self.pruning_config.dynamic_pruning or 
            epoch % self.pruning_config.pruning_frequency != 0):
            return False
        
        logger.info(f"Applying dynamic pruning at epoch {epoch}")
        pruning_stats = self.pruning_manager.prune_model(model)
        
        if self.reporting_config.report_non_pruned and not self.suppress_prints:
            print(f"\nDynamic Pruning at Epoch {epoch}:")
            print(f"Pruned {pruning_stats.get('pruned_params', 0):,} parameters "
                  f"({pruning_stats.get('pruning_ratio', 0):.4f} ratio)")
        
        return True
    
    def get_model_sparsity(self, model: BaseModel) -> Dict[str, float]:
        """
        Calculate current model sparsity.
        
        Args:
            model: The model to analyze
            
        Returns:
            Dictionary containing sparsity statistics
        """
        total_params = 0
        zero_params = 0
        layer_sparsity = {}
        
        for name, param in model.named_parameters():
            if 'weight' in name:
                param_count = param.numel()
                zero_count = (param.data == 0).sum().item()
                
                total_params += param_count
                zero_params += zero_count
                
                layer_sparsity[name] = zero_count / param_count if param_count > 0 else 0.0
        
        overall_sparsity = zero_params / total_params if total_params > 0 else 0.0
        
        return {
            'overall_sparsity': overall_sparsity,
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'layer_sparsity': layer_sparsity
        }


def create_enhanced_trainer_from_config(
    config: Dict[str, Any],
    **trainer_kwargs
) -> EnhancedTrainer:
    """
    Create an enhanced trainer from configuration dictionary.
    
    Args:
        config: Configuration dictionary containing optimizer, regularization, 
                pruning, and reporting configs
        **trainer_kwargs: Additional trainer arguments
        
    Returns:
        Configured enhanced trainer
    """
    # Extract configurations
    optimizer_config = OptimizerConfig(**config.get('optimizer', {}))
    regularization_config = RegularizationConfig(**config.get('regularization', {}))
    pruning_config = PruningConfig(**config.get('pruning', {}))
    reporting_config = ReportingConfig(**config.get('reporting', {}))
    
    return EnhancedTrainer(
        optimizer_config=optimizer_config,
        regularization_config=regularization_config,
        pruning_config=pruning_config,
        reporting_config=reporting_config,
        **trainer_kwargs
    )