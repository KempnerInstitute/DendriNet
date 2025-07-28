"""
Configuration Validation and Safeguards
=======================================

This module provides validation functions and safeguards for configuration combinations
that may cause numerical instability or performance issues.
"""

import logging

logger = logging.getLogger(__name__)


def validate_optimization_combinations(config) -> bool:
    """
    Validate optimization combinations and warn about potential issues.
    
    Args:
        config: Configuration object or dictionary
        
    Returns:
        bool: True if validation passes, False if critical issues found
    """
    warnings = []
    errors = []
    
    # Extract configuration values safely
    enable_amp = getattr(config.train, 'enable_amp', False) if hasattr(config, 'train') else False
    enable_hooks = getattr(config.train, 'enable_hooks', True) if hasattr(config, 'train') else True
    
    # Model configuration
    model_params = {}
    if hasattr(config, 'model') and hasattr(config.model, 'core_network') and hasattr(config.model.core_network, 'parameters'):
        model_params = config.model.core_network.parameters
    
    topk_ultrafast = getattr(model_params, 'topk_ultrafast', False)
    efficient_blocklinear = getattr(model_params, 'efficient_blocklinear', False)
    print_hooks = getattr(model_params, 'print_hooks', False)
    
    # Gradient scaling strategies
    reactivation_strategy = getattr(model_params, 'reactivation_strategy', 'none')
    topk_strategy = getattr(model_params, 'topk_strategy', 'none')
    blocklinear_strategy = getattr(model_params, 'blocklinear_strategy', 'none')
    
    # Training parameters
    trainer_params = {}
    if hasattr(config, 'train') and hasattr(config.train, 'main') and hasattr(config.train.main, 'trainer'):
        trainer_params = config.train.main.trainer
    
    learning_rate = getattr(config.train.main.param_groups, 'lr', 0.001) if hasattr(config, 'train') and hasattr(config.train, 'main') and hasattr(config.train.main, 'param_groups') else 0.001
    
    # 1. AMP + Gradient Hooks Compatibility
    if enable_amp and enable_hooks:
        if any([reactivation_strategy != 'none', topk_strategy != 'none', blocklinear_strategy != 'none']):
            warnings.append(
                "AMP + Gradient Hooks: Mixed precision with gradient scaling hooks may cause dtype conflicts. "
                "Consider setting 'disable_hooks_with_amp: true' in trainer config if you encounter errors."
            )
    
    # 2. Aggressive Optimization Combinations (NaN Risk)
    risky_combinations = 0
    if enable_amp:
        risky_combinations += 1
    if topk_ultrafast:
        risky_combinations += 1
    if any([reactivation_strategy != 'none', topk_strategy != 'none', blocklinear_strategy != 'none']):
        risky_combinations += 1
    if learning_rate > 0.01:
        risky_combinations += 1
        
    if risky_combinations >= 3:
        warnings.append(
            f"High NaN Risk: {risky_combinations} aggressive optimizations detected (AMP, ultrafast TopK, gradient scaling, high LR). "
            "Consider: (1) Lower learning rate, (2) Enable adaptive gradient clipping, (3) Add NaN detection."
        )
    
    # 3. FSDP Configuration Recommendations
    use_fsdp = getattr(config.train, 'use_fsdp', False) if hasattr(config, 'train') else False
    if use_fsdp:
        fsdp_config = getattr(config.train, 'fsdp_config', {}) if hasattr(config, 'train') else {}
        sharding_strategy = fsdp_config.get('sharding_strategy', 'full')
        
        if sharding_strategy == 'full' and not fsdp_config.get('reduce_communication_overhead', True):
            warnings.append(
                "FSDP Performance: Using FULL_SHARD without communication optimizations. "
                "Consider enabling 'reduce_communication_overhead: true' or using 'sharding_strategy: HYBRID_SHARD'."
            )
    
    # 5. Memory vs Speed Trade-offs
    if efficient_blocklinear and not enable_amp:
        warnings.append(
            "Memory Optimization: 'efficient_blocklinear' provides memory savings but consider enabling 'enable_amp' for additional 50% memory reduction."
        )
    
    # 6. Gradient Clipping Recommendations
    grad_clip_value = getattr(trainer_params, 'grad_clip_value', None)
    if grad_clip_value and grad_clip_value > 10.0:
        warnings.append(
            f"High Gradient Clipping: grad_clip_value={grad_clip_value} is quite high. "
            "Consider lower values (1.0-5.0) for better training stability."
        )
    
    # Log warnings and errors
    if warnings:
        logger.warning("=== CONFIGURATION SAFEGUARDS ===")
        for i, warning in enumerate(warnings, 1):
            logger.warning(f"{i}. {warning}")
        logger.warning("=" * 35)
    
    if errors:
        logger.error("=== CRITICAL CONFIGURATION ERRORS ===")
        for i, error in enumerate(errors, 1):
            logger.error(f"{i}. {error}")
        logger.error("=" * 40)
        return False
    
    return True


def suggest_optimal_config_for_model_size(total_params: int) -> dict:
    """
    Suggest optimal configuration based on model size.
    
    Args:
        total_params: Total number of model parameters
        
    Returns:
        dict: Suggested configuration adjustments
    """
    suggestions = {}
    
    if total_params < 10_000_000:  # < 10M parameters
        suggestions.update({
            "enable_amp": False,  # Not needed for small models
            "use_fsdp": False,
            "enable_hooks": True,  # Good for development and analysis
            "enable_profiling": True,  # Good for development
        })
    elif total_params < 100_000_000:  # 10M - 100M parameters
        suggestions.update({
            "enable_amp": True,
            "use_fsdp": False,  # Single GPU should handle this
            "enable_hooks": True,  # Keep analysis capability
            "enable_profiling": False,  # Better performance
        })
    else:  # > 100M parameters
        suggestions.update({
            "enable_amp": True,
            "use_fsdp": True,
            "enable_hooks": False,  # Disable for maximum performance
            "enable_profiling": False,
            "fsdp_config": {
                "mixed_precision": True,
                "sharding_strategy": "HYBRID_SHARD",
                "reduce_communication_overhead": True,
            }
        })
    
    return suggestions


def apply_automatic_safeguards(config) -> None:
    """
    Automatically apply safeguards based on configuration analysis.
    
    Args:
        config: Configuration object to modify in-place
    """
    # Auto-enable adaptive clipping for risky combinations
    if hasattr(config, 'train') and hasattr(config.train, 'main') and hasattr(config.train.main, 'trainer'):
        enable_amp = getattr(config.train, 'enable_amp', False)
        
        # Check if gradient scaling is enabled
        gradient_scaling_enabled = False
        if hasattr(config, 'model') and hasattr(config.model, 'core_network') and hasattr(config.model.core_network, 'parameters'):
            params = config.model.core_network.parameters
            if any([
                getattr(params, 'reactivation_strategy', 'none') != 'none',
                getattr(params, 'topk_strategy', 'none') != 'none', 
                getattr(params, 'blocklinear_strategy', 'none') != 'none'
            ]):
                gradient_scaling_enabled = True
        
        # Auto-enable adaptive clipping for AMP + gradient scaling
        if enable_amp and gradient_scaling_enabled:
            if not hasattr(config.train.main.trainer, 'enable_adaptive_clipping'):
                config.train.main.trainer.enable_adaptive_clipping = True
                logger.info("Auto-enabled adaptive gradient clipping for AMP + gradient scaling combination")
            
            if not hasattr(config.train.main.trainer, 'disable_hooks_with_amp'):
                config.train.main.trainer.disable_hooks_with_amp = True
                logger.info("Auto-enabled hook disabling with AMP for compatibility") 