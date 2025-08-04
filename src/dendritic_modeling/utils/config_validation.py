"""
Unified Configuration Validation for Dendritic Networks
======================================================

This module provides a unified entry point for all configuration validation,
combining both model architecture and training stability validation.

Usage:
    from dendritic_modeling.utils.config_validation import validate_full_config
    
    # Validate everything
    is_valid = validate_full_config(config)
    
    # Or validate specific aspects
    arch_valid = validate_model_architecture(config)
    training_valid = validate_training_stability(config)
"""

import logging
from typing import Dict

from .model_architecture_validator import (
    ModelArchitectureValidator,
    validate_model_architecture,
    validate_config  # Legacy compatibility
)
from .training_stability_validator import (
    validate_training_stability,
    validate_optimization_combinations  # Legacy compatibility
)

logger = logging.getLogger(__name__)


def validate_full_config(config: Dict, strict: bool = True) -> bool:
    """
    Perform complete configuration validation including both architecture and training stability.
    
    Args:
        config: Configuration dictionary
        strict: If True, both validations must pass. If False, only architecture validation is required.
        
    Returns:
        bool: True if configuration is valid according to the strict setting
    """
    logger.info("Starting full configuration validation")
    
    # 1. Model Architecture Validation (required)
    logger.info("Validating model architecture...")
    arch_valid = validate_model_architecture(config)
    
    if not arch_valid:
        logger.error("Model architecture validation failed - cannot continue")
        return False
    
    logger.info("Model architecture validation passed")
    
    # 2. Training Stability Validation (warnings only unless strict=True)
    logger.info("Validating training stability...")
    training_valid = validate_training_stability(config)
    
    if training_valid:
        logger.info("Training stability validation passed")
    else:
        if strict:
            logger.error("Training stability validation failed in strict mode")
            return False
        else:
            logger.warning("Training stability validation failed - proceeding with warnings")
    
    logger.info("Full configuration validation completed successfully")
    return True


def validate_architecture_only(config: Dict) -> bool:
    """Validate only model architecture (faster, for sweeps/tests)."""
    return validate_model_architecture(config)


def validate_training_only(config: Dict) -> bool:
    """Validate only training stability (for training scripts)."""
    return validate_training_stability(config)


# Summary function for logging
def get_validation_summary(config: Dict) -> Dict[str, bool]:
    """
    Get validation results for both architecture and training stability.
    
    Returns:
        dict: Summary of validation results
    """
    return {
        'architecture_valid': validate_model_architecture(config),
        'training_stable': validate_training_stability(config),
    }