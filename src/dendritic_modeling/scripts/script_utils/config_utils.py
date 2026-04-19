"""Utilities for configuration handling and transformation."""

import copy
import logging
from dataclasses import (
    asdict as dataclass_asdict,
    fields as dataclass_fields,
    is_dataclass,
)
from typing import Any, Union

from omegaconf import DictConfig, OmegaConf

from dendritic_modeling.config.training import (
    ParamGroupsConfig,
    TrainingStrategiesConfig,
)

logger = logging.getLogger(__name__)

# Cache for converted configs to avoid repeated conversions
_config_cache = {}


def canonicalize_model_core_flags(
    config: Union[dict, DictConfig, object]
) -> dict[str, Any]:
    """Return a plain config dict with type-implied core flags normalized.

    The architecture factory treats ``model.core.type`` as authoritative for
    shunting/additive variants. Historically, some saved sweep configs retained
    ``morphology.use_shunting=True`` even when ``type='dendritic_additive'``.
    That mismatch is harmless during model construction because the factory
    overrides the flag, but it makes raw ``config.json`` files look mislabeled.

    This helper mirrors the factory's effective behavior in the serialized
    config so saved runs are easier to audit and downstream tools see a
    self-consistent payload.
    """
    if isinstance(config, DictConfig):
        result = OmegaConf.to_container(config, resolve=True)
    elif is_dataclass(config):
        result = dataclass_asdict(config)
    else:
        result = copy.deepcopy(config)

    if not isinstance(result, dict):
        return {}

    model_cfg = result.get("model")
    if not isinstance(model_cfg, dict):
        return result

    core_cfg = model_cfg.get("core")
    if not isinstance(core_cfg, dict):
        return result

    core_type = str(core_cfg.get("type", "")).lower()
    morphology_cfg = core_cfg.setdefault("morphology", {})
    if not isinstance(morphology_cfg, dict):
        morphology_cfg = {}
        core_cfg["morphology"] = morphology_cfg

    type_overrides = {
        "dendritic_shunting": True,
        "flat_shunting": True,
        "dendritic_additive": False,
        "flat_additive": False,
        "dendritic_mlp": False,
        "flat_mlp": False,
        "point_mlp": False,
        "mlp": False,
    }
    if core_type in type_overrides:
        morphology_cfg["use_shunting"] = type_overrides[core_type]

    return result


# Type-alias implied overrides for the recurrent / unified-EI path.
# Mirrors the `_UNIFIED_EI_ALIAS_OVERRIDES` in the factory; this helper
# rewrites a raw config dict so saved configs are self-consistent with
# the alias they declare.
_RNN_CORE_ALIAS_OVERRIDES: dict[str, dict[str, object]] = {
    "rnn_dendritic_shunting": {
        "use_shunting": True,
    },
    "rnn_dendritic_additive": {
        "use_shunting": False,
        "use_additive_normalization": True,
    },
    "rnn_flat_shunting": {
        "use_shunting": True,
        "branch_factors": [1],
    },
    "rnn_flat_additive": {
        "use_shunting": False,
        "use_additive_normalization": True,
        "branch_factors": [1],
    },
}


def canonicalize_rnn_core_flags(
    config: Union[dict, DictConfig, object]
) -> dict[str, Any]:
    """Return a plain config dict with recurrent unified-EI core aliases
    resolved to explicit excitatory-population flags.

    When ``model.core.type`` is one of the recurrent aliases
    (``rnn_dendritic_shunting``, ``rnn_dendritic_additive``,
    ``rnn_flat_shunting``, ``rnn_flat_additive``), this helper rewrites
    ``use_shunting`` / ``use_additive_normalization`` (and in the flat
    case, ``branch_factors``) on every excitatory population so the saved
    payload is self-describing. ``ei_unified`` / ``unified_ei`` / ``ei_net``
    are treated as no-ops: the explicit population-level flags are the
    source of truth.
    """
    if isinstance(config, DictConfig):
        result = OmegaConf.to_container(config, resolve=True)
    elif is_dataclass(config):
        result = dataclass_asdict(config)
    else:
        result = copy.deepcopy(config)

    if not isinstance(result, dict):
        return {}

    model_cfg = result.get("model")
    if not isinstance(model_cfg, dict):
        return result

    core_cfg = model_cfg.get("core")
    if not isinstance(core_cfg, dict):
        return result

    core_type = str(core_cfg.get("type", "")).lower()
    overrides = _RNN_CORE_ALIAS_OVERRIDES.get(core_type)
    if overrides is None:
        return result

    unified_cfg = core_cfg.get("unified_ei") or core_cfg.get("ei_unified")
    if not isinstance(unified_cfg, dict):
        return result
    layers = unified_cfg.get("layers")
    if not isinstance(layers, list):
        return result

    for layer in layers:
        if not isinstance(layer, dict):
            continue
        exc = layer.get("excitatory")
        if not isinstance(exc, dict):
            continue
        for k, v in overrides.items():
            if k == "branch_factors" and exc.get("branch_factors"):
                # Preserve explicit dendritic trees even under flat aliases.
                continue
            exc[k] = v

    return result


def normalize_sparsity_type(sparsity_type: Any) -> str:
    """Normalize public sparsity aliases to the internal TopK strategy names."""
    normalized = "standard" if sparsity_type is None else str(sparsity_type).lower()
    if normalized == "topk":
        return "standard"
    return normalized


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


def normalize_param_groups_config(
    param_groups: Union[dict, DictConfig, ParamGroupsConfig, object, None],
) -> ParamGroupsConfig:
    """
    Normalize parameter-group configuration to ParamGroupsConfig.

    The training script can receive param_groups as either a typed config object
    or a plain dict (e.g., sweep-generated configs). BaseModel.get_param_groups
    expects attribute access, so dict inputs must be converted.

    Args:
        param_groups: Param-groups configuration in any supported representation

    Returns:
        ParamGroupsConfig with defaults filled for unspecified fields
    """
    if param_groups is None:
        return ParamGroupsConfig()

    if isinstance(param_groups, ParamGroupsConfig):
        return param_groups

    if isinstance(param_groups, DictConfig):
        param_groups = OmegaConf.to_container(param_groups, resolve=True)

    if is_dataclass(param_groups):
        param_groups = dataclass_asdict(param_groups)

    if hasattr(param_groups, "asdict") and callable(param_groups.asdict):
        param_groups = param_groups.asdict()

    if not isinstance(param_groups, dict):
        if hasattr(param_groups, "__dict__"):
            param_groups = {
                key: value
                for key, value in vars(param_groups).items()
                if not key.startswith("_")
            }
        else:
            return ParamGroupsConfig()

    allowed_fields = set(ParamGroupsConfig.__dataclass_fields__.keys())
    sanitized = {k: v for k, v in param_groups.items() if k in allowed_fields}
    return ParamGroupsConfig(**sanitized)


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
        elif key == "type":
            params["topk_type"] = normalize_sparsity_type(value)
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


def _strip_strategy_defaults(
    strategy_name: str, strategy_dict: dict[str, Any]
) -> dict[str, Any]:
    """Fallback: keep only fields that differ from dataclass defaults.

    This is only used when the raw user-provided strategy block is unavailable.
    It cannot distinguish "explicitly set to the default value" from "omitted",
    so callers should prefer raw override tracking when possible.

    If no matching dataclass is found (e.g. raw dict from YAML), the full dict
    is returned unchanged so we don't accidentally drop fields.
    """
    strategies_cfg = TrainingStrategiesConfig()
    default_instance = getattr(strategies_cfg, strategy_name, None)
    if default_instance is None or not is_dataclass(default_instance):
        return strategy_dict

    defaults = {
        f.name: getattr(default_instance, f.name)
        for f in dataclass_fields(default_instance)
    }
    return {
        k: v for k, v in strategy_dict.items() if k not in defaults or v != defaults[k]
    }


def _get_explicit_strategy_overrides(
    main_train_config, strategy_name: str
) -> dict[str, Any]:
    """Return raw user-provided overrides for the selected strategy when available."""

    def _to_dict_local(config):
        if isinstance(config, DictConfig):
            return OmegaConf.to_container(config, resolve=True)
        if is_dataclass(config):
            return dataclass_asdict(config)
        if hasattr(config, "asdict") and callable(config.asdict):
            return config.asdict()
        if hasattr(config, "__dict__") and not isinstance(config, dict):
            return {
                key: value
                for key, value in vars(config).items()
                if not key.startswith("_")
            }
        return config if isinstance(config, dict) else {}

    explicit_overrides = (
        main_train_config.get("_explicit_strategy_overrides")
        if isinstance(main_train_config, dict)
        else getattr(main_train_config, "_explicit_strategy_overrides", None)
    )
    if not explicit_overrides:
        return {}

    explicit_dict = _to_dict_local(explicit_overrides)
    selected = explicit_dict.get(strategy_name, {})
    return _to_dict_local(selected) if selected is not None else {}


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
        if config is None:
            return {}
        if isinstance(config, dict):
            return config
        if isinstance(config, DictConfig) or hasattr(config, "_metadata"):
            return OmegaConf.to_container(config, resolve=True)
        # Many config objects are dataclasses/BaseConfig
        if hasattr(config, "asdict") and callable(config.asdict):
            return config.asdict()
        if is_dataclass(config):
            return dataclass_asdict(config)
        return dict(config)

    # Start with common training parameters
    common_config = (
        main_train_config.get("common")
        if isinstance(main_train_config, dict)
        else main_train_config.common
    )
    trainer_config = to_dict(common_config)
    trainer_config.pop("save_path", None)
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

    strategies = (
        main_train_config.get("strategies")
        if isinstance(main_train_config, dict)
        else getattr(main_train_config, "strategies", None)
    )

    if strategies is not None:
        strategies_dict = to_dict(strategies)
        selected_strategy = (
            main_train_config.get("strategy")
            if isinstance(main_train_config, dict)
            else getattr(main_train_config, "strategy", None)
        )

        # Only pass the config for the active strategy. Typed configs carry
        # default strategy blocks for every trainer; forwarding all of them can
        # accidentally overwrite explicit common settings with default values.
        if (
            selected_strategy
            in {
                "voltage_stabilization",
                "homeostatic_control",
                "freeze_layers",
                "freeze_branches",
                "freeze_branch_kl",
                "multi_stage",
                "recurrent",
            }
            and selected_strategy in strategies_dict
        ):
            strategy_dict = _get_explicit_strategy_overrides(
                main_train_config, selected_strategy
            )
            if not strategy_dict:
                strategy_dict = _strip_strategy_defaults(
                    selected_strategy, strategies_dict[selected_strategy]
                )
            if strategy_dict:
                trainer_config[f"{selected_strategy}_config"] = strategy_dict

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
