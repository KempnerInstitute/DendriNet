"""Utilities for configuration handling and transformation."""

import logging
from typing import Any, Union

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Cache for converted configs to avoid repeated conversions
_config_cache = {}


def get_config_value(config: Union[dict, object], key: str, default: Any = None) -> Any:
    """
    Safely get a value from a config object, handling both dict and attribute access.

    Args:
        config: Configuration object (dict, DictConfig, or object with attributes)
        key: Key/attribute name to retrieve
        default: Default value if key is not found

    Returns:
        The value associated with the key, or the default value
    """
    if isinstance(config, dict):
        return config.get(key, default)
    else:
        return getattr(config, key, default)


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
    for key, value in sparsity_dict.items():
        if key == "deepst":
            for deepst_key, deepst_value in value.items():
                params[deepst_key] = deepst_value
        elif key == "gradient_scaling":
            params["topk_strategy"] = value
        else:
            params[f"topk_{key}"] = value

    # Add reactivation parameters with proper naming
    params["reactivate"] = (
        reactivation_dict.pop("enabled", True) if reactivation_dict else True
    )
    for key, value in reactivation_dict.items():
        if key == "type":
            params["reactivation_type"] = value
        elif key == "gradient_scaling":
            params["reactivation_strategy"] = value
        elif key.startswith("init_"):
            params[f"reactivation_{key}"] = value
        else:
            params[f"reactivation_{key}"] = value

    # Add blocklinear parameters
    for key, value in blocklinear_dict.items():
        if key == "efficient":
            params["efficient_blocklinear"] = value
        elif key == "gradient_scaling":
            params["blocklinear_strategy"] = value
        else:
            params[f"blocklinear_{key}"] = value

    # Cache the result (without input_dim)
    if cache_key:
        cached_params = params.copy()
        cached_params.pop("input_dim", None)
        _config_cache[cache_key] = cached_params

    return params


def prepare_trainer_config(
    main_train_config: DictConfig,
    experiment_config: DictConfig,
    save_path: str,
) -> dict[str, Any]:
    """
    Prepare trainer configuration from structured configs.

    Args:
        main_train_config: Main training configuration
        experiment_config: Experiment configuration
        save_path: Path for saving outputs

    Returns:
        Flattened trainer configuration dictionary
    """
    # Start with common training parameters
    trainer_config = OmegaConf.to_container(main_train_config.common, resolve=True)
    trainer_config["save_path"] = save_path

    # Add parameter groups
    trainer_config["param_groups"] = OmegaConf.to_container(
        main_train_config.common.param_groups, resolve=True
    )

    # Add regularization and pruning
    trainer_config["regularization"] = OmegaConf.to_container(
        main_train_config.regularization, resolve=True
    )
    trainer_config["pruning"] = OmegaConf.to_container(
        main_train_config.pruning, resolve=True
    )

    # Extract reporting config from pruning
    pruning_config = main_train_config.pruning
    trainer_config["reporting"] = {
        "report_non_pruned": pruning_config.report_non_pruned,
        "save_pruning_stats": pruning_config.save_pruning_stats,
        "detailed_branch_report": pruning_config.detailed_branch_report,
        "redo_analysis_after_pruning": pruning_config.redo_analysis_after_pruning,
    }

    # Add checkpointing if enabled
    if experiment_config.checkpointing.get("enabled", False):
        trainer_config["save_every_n_epochs"] = experiment_config.checkpointing.get(
            "save_every_n_epochs", 10
        )
        trainer_config["checkpoint_dir"] = experiment_config.checkpointing.get(
            "checkpoint_dir", save_path
        )

    # Add experimental features
    if experiment_config.enable_profiling:
        trainer_config["enable_profiling"] = True
        trainer_config["profiling_output_dir"] = f"{save_path}/profiling"

    trainer_config["enable_amp"] = experiment_config.enable_amp
    trainer_config["enable_hooks"] = experiment_config.enable_hooks

    # Add strategy-specific configurations
    if hasattr(main_train_config, "strategies"):
        strategies = OmegaConf.to_container(main_train_config.strategies, resolve=True)

        # Add local learning config if present
        if "local_ca" in strategies:
            trainer_config["local_rule_config"] = strategies["local_ca"]

        # Add other strategy configs
        for strategy_name in [
            "voltage_stabilization",
            "homeostatic_control",
            "freeze_layers",
            "freeze_branches",
            "freeze_branch_kl",
        ]:
            if strategy_name in strategies:
                trainer_config[f"{strategy_name}_config"] = strategies[strategy_name]

    return trainer_config


def prepare_local_learning_config(local_config: DictConfig) -> dict[str, Any]:
    """
    Prepare local learning configuration.

    Args:
        local_config: Local learning configuration

    Returns:
        Flattened local learning parameters
    """
    local_params = OmegaConf.to_container(local_config, resolve=True)

    # Extract specific local learning parameters
    local_learning_params = {
        "rule_variant": local_params.get("rule_variant", "standard"),
        "rho_mode": local_params.get("rho_mode", "fixed"),
        "update_inactive_weights": local_params.get("update_inactive_weights", False),
        "error_mode": local_params.get("error_mode", "mse"),
        "rho_value": local_params.get("rho_value", 0.1),
        "weight_lr_multiplier": local_params.get("weight_lr_multiplier", 1.0),
        "compute_gradient_stats": local_params.get("compute_gradient_stats", False),
        "use_conductance_scaling": local_params.get("use_conductance_scaling", True),
    }

    return local_learning_params


def extract_strategy_params(
    strategy_config: DictConfig, strategy_name: str
) -> dict[str, Any]:
    """
    Extract parameters for a specific training strategy.

    Args:
        strategy_config: Strategy configuration section
        strategy_name: Name of the strategy to extract

    Returns:
        Strategy-specific parameters
    """
    if not hasattr(strategy_config, strategy_name):
        return {}

    return OmegaConf.to_container(getattr(strategy_config, strategy_name), resolve=True)
