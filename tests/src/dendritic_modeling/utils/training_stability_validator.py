"""
Training stability checks and safeguard helpers.

This module validates risky optimization/configuration combinations before
training starts and provides lightweight automatic safeguard toggles.
"""

import logging

from dendritic_modeling.config.model_aliases import (
    get_core_morphology_alias_overrides,
    get_unified_ei_alias_overrides,
)

logger = logging.getLogger(__name__)

POSITIVE_WEIGHT_TRANSFORMS = {"softplus", "exp", "relu"}
NONNEGATIVE_TRANSFER_ACTIVATIONS = {"relu", "sigmoid", "softplus"}


def _cfg_get(node, key: str, default=None):
    """Get value from dict-like or attribute-based config nodes."""
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _cfg_path(node, path: list[str], default=None):
    """Get nested config value using a path list."""
    current = node
    for key in path:
        if current is None:
            return default
        current = _cfg_get(current, key, None)
    return default if current is None else current


def _first_present(node, paths: list[list[str]], default=None):
    """Return first non-None value from candidate nested paths."""
    for path in paths:
        value = _cfg_path(node, path, None)
        if value is not None:
            return value
    return default


def _effective_use_shunting(core_type: str, morphology_use_shunting: bool) -> bool:
    """Resolve the effective shunting flag from type aliases plus morphology."""
    core_type = (core_type or "").strip().lower()
    morphology_overrides = get_core_morphology_alias_overrides(core_type)
    if "use_shunting" in morphology_overrides:
        return bool(morphology_overrides["use_shunting"])
    unified_overrides = get_unified_ei_alias_overrides(core_type)
    if "use_shunting" in unified_overrides:
        return bool(unified_overrides["use_shunting"])
    if core_type == "point_mlp":
        return False
    if core_type in {"einet", "ei_unified", "unified_ei", "ei_net", "unified_einet"}:
        return bool(morphology_use_shunting)
    return False


def validate_training_stability(config) -> bool:
    """
    Validate optimization combinations and warn about potential issues.

    Args:
        config: Configuration object or dictionary

    Returns:
        bool: True if validation passes, False if critical issues found
    """
    # Sweep submission files wrap the actual train-time config under
    # ``base_config``. Validate that inner config so audit scripts do not
    # accidentally approve a sweep wrapper whose resolved jobs would be unsafe.
    base_config = _cfg_get(config, "base_config", None)
    if base_config is not None:
        return validate_training_stability(base_config)

    warnings = []
    errors = []

    enable_amp = bool(
        _first_present(
            config,
            [
                ["training", "main", "common", "use_amp"],
            ],
            default=False,
        )
    )
    enable_hooks = bool(
        _first_present(
            config,
            [["experiment", "enable_hooks"], ["training", "enable_hooks"]],
            default=True,
        )
    )

    # Model configuration (support both current and legacy paths).
    architecture_cfg = _cfg_path(config, ["model", "core", "architecture"], {})
    core_cfg = _cfg_path(config, ["model", "core"], {})
    sparsity_cfg = _cfg_path(config, ["model", "core", "sparsity"], {})
    blocklinear_cfg = _cfg_path(config, ["model", "core", "blocklinear"], {})
    reactivation_cfg = _cfg_path(config, ["model", "core", "reactivation"], {})
    morphology_cfg = _cfg_path(config, ["model", "core", "morphology"], {})
    transfer_cfg = _cfg_path(config, ["model", "core", "transfer"], {})
    data_processing_cfg = _cfg_path(config, ["data", "processing"], {})
    data_cfg = _cfg_path(config, ["data"], {})
    legacy_params = _cfg_path(config, ["model", "core_network", "parameters"], {})

    topk_ultrafast = bool(
        _cfg_get(architecture_cfg, "topk_ultrafast", False)
        or _cfg_get(sparsity_cfg, "ultrafast", False)
        or _cfg_get(legacy_params, "topk_ultrafast", False)
    )
    efficient_blocklinear = bool(
        _cfg_get(architecture_cfg, "efficient_blocklinear", False)
        or _cfg_get(blocklinear_cfg, "efficient", False)
        or _cfg_get(legacy_params, "efficient_blocklinear", False)
    )

    reactivation_strategy = (
        _cfg_get(architecture_cfg, "reactivation_strategy", None)
        or _cfg_get(legacy_params, "reactivation_strategy", None)
        or _cfg_get(reactivation_cfg, "gradient_scaling", "none")
    )
    topk_strategy = (
        _cfg_get(architecture_cfg, "topk_strategy", None)
        or _cfg_get(legacy_params, "topk_strategy", None)
        or _cfg_get(sparsity_cfg, "gradient_scaling", "none")
    )
    blocklinear_strategy = (
        _cfg_get(architecture_cfg, "blocklinear_strategy", None)
        or _cfg_get(legacy_params, "blocklinear_strategy", None)
        or _cfg_get(blocklinear_cfg, "gradient_scaling", "none")
    )
    core_type = str(_cfg_get(core_cfg, "type", "einet"))
    morphology_use_shunting = bool(_cfg_get(morphology_cfg, "use_shunting", True))
    effective_use_shunting = _effective_use_shunting(core_type, morphology_use_shunting)
    weight_transform = str(
        _cfg_get(
            morphology_cfg,
            "weight_transform",
            _cfg_get(legacy_params, "weight_transform", "softplus"),
        )
    ).lower()
    transfer_output_activation = _cfg_get(transfer_cfg, "output_activation", None)
    transfer_output_activation = (
        None
        if transfer_output_activation is None
        else str(transfer_output_activation).strip().lower()
    )
    dataset_name = str(_cfg_get(data_cfg, "dataset_name", "")).strip().lower()
    normalize_inputs = bool(_cfg_get(data_processing_cfg, "normalize", False))
    dataset_params_cfg = _cfg_path(config, ["data", "dataset_params"], {})
    if dataset_name == "stringer_v1":
        stringer_cfg = _cfg_get(dataset_params_cfg, "stringer_v1", {})
        normalize_inputs = normalize_inputs or bool(
            _cfg_get(stringer_cfg, "normalize", False)
        )

    trainer_params = _cfg_path(config, ["training", "main", "common"], {})
    learning_rate = float(
        _first_present(
            config,
            [
                ["training", "main", "common", "param_groups", "lr"],
                ["training", "main", "param_groups", "lr"],
            ],
            default=0.001,
        )
    )

    # 1. AMP + Gradient Hooks Compatibility
    if (
        enable_amp
        and enable_hooks
        and any(
            [
                reactivation_strategy != "none",
                topk_strategy != "none",
                blocklinear_strategy != "none",
            ]
        )
    ):
        warnings.append(
            "AMP + Gradient Hooks: Mixed precision with gradient scaling hooks may cause dtype conflicts. "
            "Consider setting 'disable_hooks_with_amp: true' in trainer config if you encounter errors."
        )

    # 1b. Signed-input shunting guard
    if (
        effective_use_shunting
        and weight_transform in POSITIVE_WEIGHT_TRANSFORMS
        and normalize_inputs
        and transfer_output_activation not in NONNEGATIVE_TRANSFER_ACTIVATIONS
    ):
        errors.append(
            "Signed shunting input: the configured preprocessing can produce "
            "signed inputs, but this model uses shunting with a positive "
            f"weight transform ({weight_transform!r}) and no nonnegative "
            "transfer.output_activation. Use all normalize flags=false, set "
            "transfer.output_activation to a nonnegative activation such as "
            "'relu', or switch to an additive morphology."
        )

    # 2. Aggressive Optimization Combinations (NaN Risk)
    risky_combinations = 0
    if enable_amp:
        risky_combinations += 1
    if topk_ultrafast:
        risky_combinations += 1
    if any(
        [
            reactivation_strategy != "none",
            topk_strategy != "none",
            blocklinear_strategy != "none",
        ]
    ):
        risky_combinations += 1
    if learning_rate > 0.01:
        risky_combinations += 1

    if risky_combinations >= 3:
        warnings.append(
            f"High NaN Risk: {risky_combinations} aggressive optimizations detected (AMP, ultrafast TopK, gradient scaling, high LR). "
            "Consider: (1) Lower learning rate, (2) Enable adaptive gradient clipping, (3) Add NaN detection."
        )

    # 3. FSDP Configuration Recommendations
    dist_mode = str(_cfg_path(config, ["distributed", "mode"], "none")).lower()
    if dist_mode == "fsdp":
        fsdp_config = _cfg_path(config, ["distributed", "fsdp"], {})
        sharding_strategy = str(
            _cfg_get(fsdp_config, "sharding_strategy", "FULL_SHARD")
        ).upper()

        if sharding_strategy in {"FULL", "FULL_SHARD"} and not bool(
            _cfg_get(fsdp_config, "reduce_communication_overhead", True)
        ):
            warnings.append(
                "FSDP Performance: Using FULL_SHARD without communication optimizations. "
                "Consider enabling 'distributed.fsdp.reduce_communication_overhead: true' "
                "or using 'sharding_strategy: HYBRID_SHARD'."
            )

    # 5. Memory vs Speed Trade-offs
    if efficient_blocklinear and not enable_amp:
        warnings.append(
            "Memory Optimization: 'efficient_blocklinear' provides memory savings but consider enabling 'training.main.common.use_amp: true' for additional 50% memory reduction."
        )

    # 6. Gradient Clipping Recommendations
    grad_clip_value = _cfg_get(trainer_params, "grad_clip_value", None)
    if grad_clip_value and grad_clip_value > 10.0:
        warnings.append(
            f"High Gradient Clipping: grad_clip_value={grad_clip_value} is quite high. "
            "Consider lower values (1.0-5.0) for better training stability."
        )

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
        suggestions.update(
            {
                "training.main.common.use_amp": False,
                "distributed.mode": "none",
                "experiment.enable_hooks": True,
                "experiment.enable_profiling": True,
            }
        )
    elif total_params < 100_000_000:  # 10M - 100M parameters
        suggestions.update(
            {
                "training.main.common.use_amp": True,
                "distributed.mode": "ddp",  # data-parallel across GPUs
                "experiment.enable_hooks": True,
                "experiment.enable_profiling": False,
            }
        )
    else:  # > 100M parameters
        suggestions.update(
            {
                "training.main.common.use_amp": True,
                "distributed.mode": "fsdp",
                "experiment.enable_hooks": False,
                "experiment.enable_profiling": False,
                "distributed.fsdp.mixed_precision": True,
                "distributed.fsdp.sharding_strategy": "HYBRID_SHARD",
                "distributed.fsdp.reduce_communication_overhead": True,
            }
        )

    return suggestions


def apply_automatic_safeguards(config) -> None:
    """
    Automatically apply safeguards based on configuration analysis.

    Args:
        config: Configuration object to modify in-place
    """
    common_cfg = _cfg_path(config, ["training", "main", "common"], None)
    if common_cfg is None:
        return

    enable_amp = bool(
        _first_present(
            config,
            [
                ["training", "main", "common", "use_amp"],
            ],
            default=False,
        )
    )

    # Check if gradient scaling is enabled (support current and legacy config paths)
    model_core_arch = _cfg_path(config, ["model", "core", "architecture"], {})
    model_core_reactivation = _cfg_path(config, ["model", "core", "reactivation"], {})
    model_core_sparsity = _cfg_path(config, ["model", "core", "sparsity"], {})
    model_core_blocklinear = _cfg_path(config, ["model", "core", "blocklinear"], {})
    legacy_params = _cfg_path(config, ["model", "core_network", "parameters"], {})

    gradient_scaling_enabled = any(
        [
            _cfg_get(model_core_reactivation, "gradient_scaling", "none") != "none",
            _cfg_get(model_core_sparsity, "gradient_scaling", "none") != "none",
            _cfg_get(model_core_blocklinear, "gradient_scaling", "none") != "none",
            _cfg_get(model_core_arch, "reactivation_strategy", "none") != "none",
            _cfg_get(model_core_arch, "topk_strategy", "none") != "none",
            _cfg_get(model_core_arch, "blocklinear_strategy", "none") != "none",
            _cfg_get(legacy_params, "reactivation_strategy", "none") != "none",
            _cfg_get(legacy_params, "topk_strategy", "none") != "none",
            _cfg_get(legacy_params, "blocklinear_strategy", "none") != "none",
        ]
    )

    # Keep existing behavior: only set fields that are not explicitly present.
    if enable_amp and gradient_scaling_enabled:
        if not hasattr(common_cfg, "enable_adaptive_clipping"):
            common_cfg.enable_adaptive_clipping = True
            logger.info(
                "Auto-enabled adaptive gradient clipping for AMP + gradient scaling combination"
            )

        if not hasattr(common_cfg, "disable_hooks_with_amp"):
            common_cfg.disable_hooks_with_amp = True
            logger.info("Auto-enabled hook disabling with AMP for compatibility")
