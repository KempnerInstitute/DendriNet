"""
Regularization and Pruning Utilities
====================================

This module provides utilities for L1/L2 regularization and weight pruning
with selective application based on layer, branch, and weight type.
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path

from dendritic_modeling.config.training import RegularizationConfig, PruningConfig


class RegularizationManager:
    """Manages L1/L2 regularization with selective application."""
    
    def __init__(self, config: RegularizationConfig):
        self.config = config
        self.l1_weight = config.l1_weight
        self.l2_weight = config.l2_weight
        self.selective = config.selective
    
    def compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Compute regularization loss based on configuration.
        
        Args:
            model: The model to regularize
            
        Returns:
            Regularization loss tensor
        """
        l1_reg = torch.tensor(0.0, device=next(model.parameters()).device)
        l2_reg = torch.tensor(0.0, device=next(model.parameters()).device)
        
        if self.l1_weight <= 0 and self.l2_weight <= 0:
            return l1_reg
        
        for name, param in model.named_parameters():
            if not param.requires_grad or 'weight' not in name:
                continue
            
            if self._should_regularize(name, model):
                if self.l1_weight > 0:
                    l1_reg += torch.sum(torch.abs(param))
                if self.l2_weight > 0:
                    l2_reg += torch.sum(param ** 2)
        
        return self.l1_weight * l1_reg + self.l2_weight * l2_reg
    
    def _should_regularize(self, param_name: str, model: nn.Module) -> bool:
        """Check if parameter should be regularized based on selective criteria."""
        # Extract layer index from parameter name
        layer_idx = self._extract_layer_index(param_name)
        
        # Check layer selection
        if self.selective.layers and layer_idx is not None:
            if layer_idx not in self.selective.layers:
                return False
        
        # Check branch selection
        if self.selective.branches:
            branch_found = any(branch in param_name for branch in self.selective.branches)
            if not branch_found:
                return False
        
        # Check weight type selection
        if self.selective.weight_types and "all" not in self.selective.weight_types:
            weight_type_found = any(wt in param_name for wt in self.selective.weight_types)
            if not weight_type_found:
                return False
        
        return True
    
    def _extract_layer_index(self, param_name: str) -> Optional[int]:
        """Extract layer index from parameter name."""
        parts = param_name.split('.')
        for part in parts:
            if part.isdigit():
                return int(part)
        return None


class PruningManager:
    """Manages weight pruning with selective application and reporting."""
    
    def __init__(self, config: PruningConfig):
        self.config = config
        self.threshold = config.threshold
        self.selective = config.selective
        self.pruning_stats = {}
    
    def prune_model(self, model: nn.Module) -> Dict[str, Any]:
        """
        Prune model weights based on configuration.
        
        Args:
            model: The model to prune
            
        Returns:
            Dictionary containing pruning statistics
        """
        if not self.config.enabled:
            return {}
        
        pruning_stats = {
            'total_params_before': 0,
            'total_params_after': 0,
            'pruned_params': 0,
            'layer_stats': {},
            'branch_stats': {},
            'weight_type_stats': {}
        }
        
        for name, param in model.named_parameters():
            if 'weight' not in name:
                continue
            
            original_count = param.numel()
            pruning_stats['total_params_before'] += original_count
            
            if self._should_prune(name, model):
                # Apply pruning mask
                mask = torch.abs(param.data) >= self.threshold
                param.data[~mask] = 0.0
                
                remaining_count = mask.sum().item()
                pruned_count = original_count - remaining_count
                
                # Update statistics
                layer_idx = self._extract_layer_index(name)
                layer_key = f"layer_{layer_idx}" if layer_idx is not None else "unknown"
                
                if layer_key not in pruning_stats['layer_stats']:
                    pruning_stats['layer_stats'][layer_key] = {
                        'total_before': 0, 'remaining': 0, 'pruned': 0
                    }
                
                pruning_stats['layer_stats'][layer_key]['total_before'] += original_count
                pruning_stats['layer_stats'][layer_key]['remaining'] += remaining_count
                pruning_stats['layer_stats'][layer_key]['pruned'] += pruned_count
                
                # Branch statistics
                branch_name = self._extract_branch_name(name)
                if branch_name:
                    if branch_name not in pruning_stats['branch_stats']:
                        pruning_stats['branch_stats'][branch_name] = {
                            'total_before': 0, 'remaining': 0, 'pruned': 0
                        }
                    
                    pruning_stats['branch_stats'][branch_name]['total_before'] += original_count
                    pruning_stats['branch_stats'][branch_name]['remaining'] += remaining_count
                    pruning_stats['branch_stats'][branch_name]['pruned'] += pruned_count
                
                # Weight type statistics
                weight_type = self._extract_weight_type(name)
                if weight_type not in pruning_stats['weight_type_stats']:
                    pruning_stats['weight_type_stats'][weight_type] = {
                        'total_before': 0, 'remaining': 0, 'pruned': 0
                    }
                
                pruning_stats['weight_type_stats'][weight_type]['total_before'] += original_count
                pruning_stats['weight_type_stats'][weight_type]['remaining'] += remaining_count
                pruning_stats['weight_type_stats'][weight_type]['pruned'] += pruned_count
                
                pruning_stats['total_params_after'] += remaining_count
                pruning_stats['pruned_params'] += pruned_count
            else:
                pruning_stats['total_params_after'] += original_count
        
        # Calculate pruning ratio
        if pruning_stats['total_params_before'] > 0:
            pruning_stats['pruning_ratio'] = (
                pruning_stats['pruned_params'] / pruning_stats['total_params_before']
            )
        else:
            pruning_stats['pruning_ratio'] = 0.0
        
        self.pruning_stats = pruning_stats
        return pruning_stats
    
    def _should_prune(self, param_name: str, model: nn.Module) -> bool:
        """Check if parameter should be pruned based on selective criteria."""
        # Extract layer index from parameter name
        layer_idx = self._extract_layer_index(param_name)
        
        # Check layer selection
        if self.selective.layers and layer_idx is not None:
            if layer_idx not in self.selective.layers:
                return False
        
        # Check branch selection
        if self.selective.branches:
            branch_found = any(branch in param_name for branch in self.selective.branches)
            if not branch_found:
                return False
        
        # Check weight type selection
        if self.selective.weight_types and "all" not in self.selective.weight_types:
            weight_type_found = any(wt in param_name for wt in self.selective.weight_types)
            if not weight_type_found:
                return False
        
        return True
    
    def _extract_layer_index(self, param_name: str) -> Optional[int]:
        """Extract layer index from parameter name."""
        parts = param_name.split('.')
        for part in parts:
            if part.isdigit():
                return int(part)
        return None
    
    def _extract_branch_name(self, param_name: str) -> Optional[str]:
        """Extract branch name from parameter name."""
        if 'branch' in param_name:
            parts = param_name.split('.')
            for i, part in enumerate(parts):
                if 'branch' in part:
                    return part
        return None
    
    def _extract_weight_type(self, param_name: str) -> str:
        """Extract weight type from parameter name."""
        if 'recurrent' in param_name:
            return 'recurrent'
        elif 'input' in param_name:
            return 'input'
        elif 'output' in param_name:
            return 'output'
        elif 'linear' in param_name:
            return 'linear'
        else:
            return 'other'
    
    def print_pruning_report(self) -> None:
        """Print detailed pruning report."""
        if not self.pruning_stats:
            print("No pruning statistics available.")
            return
        
        stats = self.pruning_stats
        
        print("\n" + "="*50)
        print("WEIGHT PRUNING REPORT")
        print("="*50)
        
        print(f"Total parameters before pruning: {stats['total_params_before']:,}")
        print(f"Total parameters after pruning:  {stats['total_params_after']:,}")
        print(f"Pruned parameters:               {stats['pruned_params']:,}")
        print(f"Pruning ratio:                   {stats['pruning_ratio']:.4f}")
        
        # Layer-wise report
        if stats['layer_stats']:
            print(f"\n{'Layer':<15} {'Before':<10} {'After':<10} {'Pruned':<10} {'Ratio':<10}")
            print("-" * 60)
            for layer, layer_stats in stats['layer_stats'].items():
                ratio = layer_stats['pruned'] / layer_stats['total_before'] if layer_stats['total_before'] > 0 else 0
                print(f"{layer:<15} {layer_stats['total_before']:<10,} {layer_stats['remaining']:<10,} "
                      f"{layer_stats['pruned']:<10,} {ratio:<10.4f}")
        
        # Branch-wise report
        if stats['branch_stats']:
            print(f"\n{'Branch':<20} {'Before':<10} {'After':<10} {'Pruned':<10} {'Ratio':<10}")
            print("-" * 65)
            for branch, branch_stats in stats['branch_stats'].items():
                ratio = branch_stats['pruned'] / branch_stats['total_before'] if branch_stats['total_before'] > 0 else 0
                print(f"{branch:<20} {branch_stats['total_before']:<10,} {branch_stats['remaining']:<10,} "
                      f"{branch_stats['pruned']:<10,} {ratio:<10.4f}")
        
        # Weight type report
        if stats['weight_type_stats']:
            print(f"\n{'Weight Type':<15} {'Before':<10} {'After':<10} {'Pruned':<10} {'Ratio':<10}")
            print("-" * 60)
            for weight_type, type_stats in stats['weight_type_stats'].items():
                ratio = type_stats['pruned'] / type_stats['total_before'] if type_stats['total_before'] > 0 else 0
                print(f"{weight_type:<15} {type_stats['total_before']:<10,} {type_stats['remaining']:<10,} "
                      f"{type_stats['pruned']:<10,} {ratio:<10.4f}")
        
        print("="*50)
    
    def save_pruning_stats(self, save_path: str) -> None:
        """Save pruning statistics to file."""
        if not self.pruning_stats:
            return
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.pruning_stats, f, indent=2)
        
        print(f"Pruning statistics saved to: {save_path}")


def create_sparsity_mask(model: nn.Module, sparsity_ratio: float) -> Dict[str, torch.Tensor]:
    """
    Create sparsity masks for magnitude-based pruning.
    
    Args:
        model: The model to create masks for
        sparsity_ratio: Fraction of weights to prune (0.0 to 1.0)
        
    Returns:
        Dictionary mapping parameter names to binary masks
    """
    masks = {}
    
    # Collect all weights and their magnitudes
    all_weights = []
    weight_names = []
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            all_weights.append(param.data.abs().flatten())
            weight_names.append(name)
    
    if not all_weights:
        return masks
    
    # Calculate global threshold
    all_weights_concat = torch.cat(all_weights)
    threshold = torch.quantile(all_weights_concat, sparsity_ratio)
    
    # Create masks
    for name, param in model.named_parameters():
        if 'weight' in name and param.requires_grad:
            masks[name] = (param.data.abs() >= threshold)
    
    return masks


def apply_sparsity_masks(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """
    Apply sparsity masks to model parameters.
    
    Args:
        model: The model to apply masks to
        masks: Dictionary mapping parameter names to binary masks
    """
    for name, param in model.named_parameters():
        if name in masks:
            param.data *= masks[name].float()