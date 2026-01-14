"""Utilities for configuration handling and transformation."""

import logging
from typing import Any, Union, Optional

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
    config: Union[dict, DictConfig], 
    input_dim: int, 
    use_shunting: Optional[bool] = None
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
    
    if use_shunting is not None:
        params["use_shunting"] = use_shunting

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

    # Helper function to convert config to dict
    def to_dict(config):
        if isinstance(config, dict):
            return config
        return (
            OmegaConf.to_container(config, resolve=True)
            if hasattr(config, "_metadata")
            else dict(config)
        )

    # Start with common training parameters
    common_config = (
        main_train_config.get("common")
        if isinstance(main_train_config, dict)
        else main_train_config.common
    )
    trainer_config = to_dict(common_config)
    trainer_config["save_path"] = save_path

    # Add parameter groups
    param_groups = (
        common_config.get("param_groups")
        if isinstance(common_config, dict)
        else common_config.param_groups
    )
    trainer_config["param_groups"] = to_dict(param_groups)

    # Add regularization and pruning
    regularization = (
        main_train_config.get("regularization")
        if isinstance(main_train_config, dict)
        else main_train_config.regularization
    )
    pruning = (
        main_train_config.get("pruning")
        if isinstance(main_train_config, dict)
        else main_train_config.pruning
    )

    trainer_config["regularization"] = to_dict(regularization)
    trainer_config["pruning"] = to_dict(pruning)

    # Extract reporting config from pruning
    pruning_dict = to_dict(pruning)
    trainer_config["reporting"] = {
        "report_non_pruned": pruning_dict.get("report_non_pruned", False),
        "save_pruning_stats": pruning_dict.get("save_pruning_stats", False),
        "detailed_branch_report": pruning_dict.get("detailed_branch_report", False),
        "redo_analysis_after_pruning": pruning_dict.get(
            "redo_analysis_after_pruning", False
        ),
    }

    # Add checkpointing if enabled
    checkpointing = (
        experiment_config.get("checkpointing")
        if isinstance(experiment_config, dict)
        else experiment_config.checkpointing
    )
    enable_profiling = (
        experiment_config.get("enable_profiling", False)
        if isinstance(experiment_config, dict)
        else experiment_config.enable_profiling
    )
    enable_amp = (
        experiment_config.get("enable_amp", False)
        if isinstance(experiment_config, dict)
        else experiment_config.enable_amp
    )
    enable_hooks = (
        experiment_config.get("enable_hooks", False)
        if isinstance(experiment_config, dict)
        else experiment_config.enable_hooks
    )

    checkpointing_dict = to_dict(checkpointing)
    if checkpointing_dict.get("enabled", False):
        trainer_config["save_every_n_epochs"] = checkpointing_dict.get(
            "save_every_n_epochs", 10
        )
        trainer_config["checkpoint_dir"] = checkpointing_dict.get(
            "checkpoint_dir", save_path
        )

    # Add experimental features
    if enable_profiling:
        trainer_config["enable_profiling"] = True
        trainer_config["profiling_output_dir"] = f"{save_path}/profiling"

    trainer_config["enable_amp"] = enable_amp
    trainer_config["enable_hooks"] = enable_hooks

    # Add strategy-specific configurations
    # First, check for learning_strategy_config (new structure)
    learning_strategy_config = (
        main_train_config.get("learning_strategy_config")
        if isinstance(main_train_config, dict)
        else getattr(main_train_config, "learning_strategy_config", None)
    )

    if learning_strategy_config is not None:
        trainer_config["local_rule_config"] = to_dict(learning_strategy_config)
    else:
        # Fallback to strategies.local_ca (old structure)
        strategies = (
            main_train_config.get("strategies")
            if isinstance(main_train_config, dict)
            else getattr(main_train_config, "strategies", None)
        )

        if strategies is not None:
            strategies_dict = to_dict(strategies)

            # Add local learning config if present (old structure)
            if "local_ca" in strategies_dict:
                trainer_config["local_rule_config"] = strategies_dict["local_ca"]

            # Add other strategy configs
            for strategy_name in [
                "voltage_stabilization",
                "homeostatic_control",
                "freeze_layers",
                "freeze_branches",
                "freeze_branch_kl",
            ]:
                if strategy_name in strategies_dict:
                    trainer_config[f"{strategy_name}_config"] = strategies_dict[
                        strategy_name
                    ]

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
