"""Utilities for configuration handling and transformation."""

from typing import Any, Union

from omegaconf import DictConfig, OmegaConf


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
