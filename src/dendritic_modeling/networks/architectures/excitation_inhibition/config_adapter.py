"""Config adapter for ExcitationInhibitionNetwork initialization."""

import logging
from typing import Any, Union

from omegaconf import DictConfig, OmegaConf

from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork,
)

logger = logging.getLogger(__name__)

# Cache for converted configs to avoid repeated conversions
_config_cache = {}


def prepare_ei_network_params(
    config: Union[dict, DictConfig], input_dim: int
) -> dict[str, Any]:
    """
    Prepare parameters for ExcitationInhibitionNetwork from structured config.

    Args:
        config: Either a dict or DictConfig with structured parameters
        input_dim: Input dimension from encoder

    Returns:
        Flattened parameter dictionary ready for EINet initialization
    """
    # Create a cache key (excluding input_dim since it changes the params)
    cache_key = id(config) if isinstance(config, DictConfig) else None

    # Check cache first
    if cache_key and cache_key in _config_cache:
        logger.debug("Using cached config conversion for EINet initialization")
        cached_params = _config_cache[cache_key].copy()
        cached_params["input_dim"] = input_dim
        return cached_params

    logger.info(
        "Converting DictConfig to dict for EINet initialization (this should happen only once)"
    )

    # Convert to regular dict if needed - do this ONCE
    if isinstance(config, DictConfig):
        # Single conversion of entire config
        config_dict = OmegaConf.to_container(config, resolve=True)
        architecture_dict = config_dict.get("architecture", {})
        connectivity_dict = config_dict.get("connectivity", {})
        transfer_dict = config_dict.get("transfer", {})
        morphology_dict = config_dict.get("morphology", {})
        implementation_dict = config_dict.get("implementation", {})
        sparsity_dict = config_dict.get("sparsity", {})
        reactivation_dict = config_dict.get("reactivation", {})
        blocklinear_dict = config_dict.get("blocklinear", {})
    else:
        architecture_dict = config.get("architecture", {})
        connectivity_dict = config.get("connectivity", {})
        transfer_dict = config.get("transfer", {})
        morphology_dict = config.get("morphology", {})
        implementation_dict = config.get("implementation", {})
        sparsity_dict = config.get("sparsity", {})
        reactivation_dict = config.get("reactivation", {})
        blocklinear_dict = config.get("blocklinear", {})

    # Build base parameters
    params = {
        "input_dim": input_dim,
        **architecture_dict,
        **connectivity_dict,
        "transfer_params": transfer_dict,
        **morphology_dict,
        **implementation_dict,
    }

    # Add sparsity parameters with proper naming
    # CRITICAL: When gradient_scaling is explicitly set in config, it MUST be respected
    # Only use EINet's defaults if the parameter is not defined in config at all
    if "gradient_scaling" in sparsity_dict:
        params["topk_strategy"] = sparsity_dict["gradient_scaling"]
        logger.debug(
            f"Setting topk_strategy={sparsity_dict['gradient_scaling']} from config"
        )
    # If not in config, we DON'T set it - let EINet use its own default

    for key, value in sparsity_dict.items():
        if key == "deepst":
            for deepst_key, deepst_value in value.items():
                params[deepst_key] = deepst_value
        elif key == "gradient_scaling":
            # Already handled above
            pass
        else:
            params[f"topk_{key}"] = value

    # Add reactivation parameters with proper naming
    params["reactivate"] = reactivation_dict.pop("enabled", True)
    # CRITICAL: When gradient_scaling is explicitly set in config, it MUST be respected
    if "gradient_scaling" in reactivation_dict:
        params["reactivation_strategy"] = reactivation_dict["gradient_scaling"]
        logger.debug(
            f"Setting reactivation_strategy={reactivation_dict['gradient_scaling']} from config"
        )
    # If not in config, we DON'T set it - let EINet use its own default

    for key, value in reactivation_dict.items():
        if key == "type":
            params["reactivation_type"] = value
        elif key == "gradient_scaling":
            # Already handled above
            pass
        elif key.startswith("init_"):
            params[f"reactivation_{key}"] = value
        else:
            params[f"reactivation_{key}"] = value

    # Add blocklinear parameters
    # CRITICAL: When gradient_scaling is explicitly set in config, it MUST be respected
    if "gradient_scaling" in blocklinear_dict:
        params["blocklinear_strategy"] = blocklinear_dict["gradient_scaling"]
        logger.debug(
            f"Setting blocklinear_strategy={blocklinear_dict['gradient_scaling']} from config"
        )

    for key, value in blocklinear_dict.items():
        if key == "efficient":
            params["efficient_blocklinear"] = value
        elif key == "gradient_scaling":
            # Already handled above
            pass
        else:
            params[f"blocklinear_{key}"] = value

    # Cache the params (without input_dim) if using DictConfig
    if cache_key:
        cache_params = params.copy()
        cache_params.pop("input_dim", None)
        _config_cache[cache_key] = cache_params

    # Log a summary of gradient scaling strategies for transparency
    strategies = {
        "topk_strategy": params.get("topk_strategy", "not set"),
        "reactivation_strategy": params.get("reactivation_strategy", "not set"),
        "blocklinear_strategy": params.get("blocklinear_strategy", "not set"),
    }
    logger.info(f"EINet gradient scaling strategies: {strategies}")

    return params


class ConfigurableEINetwork(ExcitationInhibitionNetwork):
    """
    ExcitationInhibitionNetwork that is initialized from structured config.

    This is now the standard way to create EINet instances, providing a clean
    separation between configuration structure and implementation details.
    """

    def __init__(self, config: Union[dict, DictConfig], input_dim: int):
        """
        Initialize from structured config.

        Args:
            config: Structured configuration (e.g., core_config from YAML)
                   Must contain 'architecture', 'connectivity', 'transfer', etc. sections
            input_dim: Input dimension from encoder
        """
        params = prepare_ei_network_params(config, input_dim)
        super().__init__(**params)
