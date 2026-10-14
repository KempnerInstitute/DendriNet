"""
Training Factory and Utilities

Factory functions for creating training strategies and utility functions
for MLP training and weight transfer.
"""

import importlib
import logging
from typing import Any, Optional

import torch

from dendritic_modeling.config.conversion import to_plain_dict
from dendritic_modeling.config.keys import build_section, check_section_keys
from dendritic_modeling.config.multi_stage_training import MultiStageTrainingConfig
from dendritic_modeling.config.training import (
    OptimizerConfig,
    ParamGroupsConfig,
    PruningConfig,
    RegularizationConfig,
)
from dendritic_modeling.plugins import load_plugins
from dendritic_modeling.training.strategies import (
    ExcitationInhibitionEquilibriumTrainer,
    FeedbackAlignmentTrainer,
    HomeostaticControlTrainer,
    LayerWiseTrainer,
    LocalCreditAssignment,
    MultiStageTrainer,
    PresentationStreamTrainer,
    RecurrentTrainer,
    SomaDFATrainer,
    Trainer,
    TwoStepTrainer,
    TwoStepTrainerWithKL,
    VoltageStabilizationTrainer,
)
from dendritic_modeling.training.strategies.standard import validate_lr_schedule

logger = logging.getLogger(__name__)

# These classes are resolved by name through ``globals()`` below, and tests
# patch them through this module path. Keep them materialized in module globals.
_TRAINER_CLASS_GLOBALS = (
    ExcitationInhibitionEquilibriumTrainer,
    FeedbackAlignmentTrainer,
    HomeostaticControlTrainer,
    LayerWiseTrainer,
    LocalCreditAssignment,
    MultiStageTrainer,
    PresentationStreamTrainer,
    RecurrentTrainer,
    SomaDFATrainer,
    Trainer,
    TwoStepTrainer,
    TwoStepTrainerWithKL,
    VoltageStabilizationTrainer,
)

TRAINER_STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "standard": {"class": "Trainer"},
    "freeze_layers": {"class": "LayerWiseTrainer"},
    "freeze_branches": {"class": "TwoStepTrainer"},
    "fa": {"class": "FeedbackAlignmentTrainer", "fixed_kwargs": {"mode": "fa"}},
    "dfa": {"class": "FeedbackAlignmentTrainer", "fixed_kwargs": {"mode": "dfa"}},
    "soma_dfa": {"class": "SomaDFATrainer"},
    "local_ca": {"class": "LocalCreditAssignment"},
    "freeze_branch_kl": {"class": "TwoStepTrainerWithKL"},
    "voltage_stabilization": {"class": "VoltageStabilizationTrainer"},
    "homeostatic_control": {"class": "HomeostaticControlTrainer"},
    "ei_equilibrium": {"class": "ExcitationInhibitionEquilibriumTrainer"},
    "presentation_stream": {"class": "PresentationStreamTrainer"},
}

# Former strategies that now run as the strategy they name. FA and DFA compute
# each block's exact local gradient, which includes the conductance and
# driving-force terms the shunting variants approximated.
DEPRECATED_STRATEGY_ALIASES = {
    "shunting_fa": "fa",
    "shunting_dfa": "dfa",
}
SPECIAL_TRAINER_STRATEGIES = [
    "recurrent",
    "multi_stage",
]
_WARNED_STRATEGY_ALIASES: set[str] = set()


def register_trainer_strategy(
    name: str,
    trainer_class: type[Trainer] | str,
    *,
    fixed_kwargs: dict[str, Any] | None = None,
    allow_override: bool = False,
) -> None:
    """Register a trainer strategy without editing ``get_trainer``.

    ``trainer_class`` is a :class:`Trainer` subclass or a ``"module:Class"``
    reference, which is imported when the strategy is first used (plugins
    register strategies this way, so that loading them stays light). Replacing
    an existing strategy (built-in or registered) requires
    ``allow_override=True``.
    """
    if not name:
        raise ValueError("trainer strategy name must be non-empty")
    if isinstance(trainer_class, str) and trainer_class.count(":") != 1:
        raise ValueError(
            "a trainer class reference is written 'module:Class', got "
            f"{trainer_class!r}"
        )
    key = str(name).lower()
    if (
        key in TRAINER_STRATEGY_REGISTRY or key in SPECIAL_TRAINER_STRATEGIES
    ) and not allow_override:
        raise ValueError(
            f"Trainer strategy '{key}' is already registered; pass "
            "allow_override=True to replace it"
        )
    TRAINER_STRATEGY_REGISTRY[key] = {
        "class": trainer_class,
        "fixed_kwargs": dict(fixed_kwargs or {}),
    }


def get_available_trainer_strategies() -> list[str]:
    """Return supported trainer strategy names, including those of plugins."""
    load_plugins()
    return list(TRAINER_STRATEGY_REGISTRY.keys()) + SPECIAL_TRAINER_STRATEGIES


def _resolve_trainer_class(class_ref: type[Trainer] | str) -> type[Trainer]:
    if not isinstance(class_ref, str):
        return class_ref
    if ":" not in class_ref:
        return globals()[class_ref]  # a trainer of this module
    module_name, _, class_name = class_ref.partition(":")
    trainer_cls = getattr(importlib.import_module(module_name), class_name)
    if not (isinstance(trainer_cls, type) and issubclass(trainer_cls, Trainer)):
        raise TypeError(f"{class_ref} is not a Trainer subclass")
    return trainer_cls


def _build_multi_stage_config(raw_config: Any) -> MultiStageTrainingConfig:
    """Build the typed multi-stage config, checking every key."""
    if raw_config is None:
        raise ValueError(
            "strategy='multi_stage' requires 'multi_stage_config' in trainer config."
        )
    config = build_section(
        MultiStageTrainingConfig,
        raw_config,
        section="training.main.strategies.multi_stage",
    )
    if not config.stages:
        raise ValueError("multi_stage_config must contain at least one stage.")
    return config


_FLATTENED_STRATEGY_CONFIG_KEYS = (
    "voltage_stabilization_config",
    "homeostatic_control_config",
    "ei_equilibrium_config",
    "freeze_layers_config",
    "freeze_branches_config",
    "freeze_branch_kl_config",
)


def _pop_flat_strategy_config(
    enhanced_configs: dict[str, Any],
    trainer_config_dict: dict[str, Any],
    key: str,
) -> None:
    """Pop a nested strategy config and flatten dict values into trainer kwargs."""
    if key not in trainer_config_dict:
        return
    nested_config = trainer_config_dict.pop(key)
    if not isinstance(nested_config, dict):
        raise TypeError(
            f"trainer option {key} must be a mapping, got "
            f"{type(nested_config).__name__}"
        )
    enhanced_configs.update(nested_config)


def _extract_enhanced_configs(trainer_config_dict: dict[str, Any]) -> dict[str, Any]:
    """
    Extract enhanced configuration objects from trainer config dict.

    Args:
        trainer_config_dict: Configuration dictionary

    Returns:
        Dictionary containing enhanced config objects
    """
    enhanced_configs = {}

    # Extract optimizer config
    if "optimizer" in trainer_config_dict and isinstance(
        trainer_config_dict["optimizer"], dict
    ):
        enhanced_configs["optimizer_config"] = OptimizerConfig(
            **trainer_config_dict["optimizer"]
        )
        # Don't pass the dict version to avoid conflicts
        trainer_config_dict.pop("optimizer", None)

    # Extract regularization config
    if "regularization" in trainer_config_dict:
        reg_config = trainer_config_dict.pop("regularization")
        if isinstance(reg_config, dict):
            # Handle nested SelectiveConfig properly
            if "selective" in reg_config and isinstance(reg_config["selective"], dict):
                from dendritic_modeling.config.regularization import SelectiveConfig

                reg_config["selective"] = SelectiveConfig(**reg_config["selective"])
            enhanced_configs["regularization_config"] = RegularizationConfig(
                **reg_config
            )
        else:
            enhanced_configs["regularization_config"] = reg_config
    else:
        # Create default regularization config and check for E/I parameters
        reg_config_dict = {}

        # Check for E/I regularization parameters in trainer config
        ei_params = [
            "enforce_ei_weight_ratio",
            "target_ei_weight_ratio",
            "ei_ratio_loss_weight",
            "ei_ratio_scope",
            "ei_ratio_metric",
        ]
        for param in ei_params:
            if param in trainer_config_dict:
                reg_config_dict[param] = trainer_config_dict.pop(param)

        # Create regularization config with parameters if found
        if reg_config_dict:
            enhanced_configs["regularization_config"] = RegularizationConfig(
                **reg_config_dict
            )

    # Extract pruning config
    if "pruning" in trainer_config_dict:
        prune_config = trainer_config_dict.pop("pruning")
        if isinstance(prune_config, dict):
            # Handle nested SelectiveConfig properly
            if "selective" in prune_config and isinstance(
                prune_config["selective"], dict
            ):
                from dendritic_modeling.config.regularization import SelectiveConfig

                prune_config["selective"] = SelectiveConfig(**prune_config["selective"])
            enhanced_configs["pruning_config"] = PruningConfig(**prune_config)
        else:
            enhanced_configs["pruning_config"] = prune_config

    if "reporting" in trainer_config_dict:
        raise ValueError(
            "The trainer option 'reporting' was removed: its values were never "
            "read. The pruning report options are fields of 'pruning'."
        )

    # Extract param_groups config (preserve original parameter grouping behavior)
    if "param_groups" in trainer_config_dict:
        param_groups_config = trainer_config_dict.pop("param_groups")
        if isinstance(param_groups_config, dict):
            param_groups_config = ParamGroupsConfig(
                **check_section_keys(
                    param_groups_config,
                    ParamGroupsConfig.__dataclass_fields__,
                    section="param_groups",
                )
            )
        enhanced_configs["param_groups_config"] = param_groups_config

    # Extract strategy-specific nested configs and flatten into kwargs
    # Local credit assignment
    if "local_rule_config" in trainer_config_dict:
        enhanced_configs["local_rule_config"] = trainer_config_dict.pop(
            "local_rule_config"
        )

    for key in _FLATTENED_STRATEGY_CONFIG_KEYS:
        _pop_flat_strategy_config(enhanced_configs, trainer_config_dict, key)

    return enhanced_configs


def _apply_enhanced_configs(trainer_config_dict: dict[str, Any]) -> None:
    trainer_config_dict.update(_extract_enhanced_configs(trainer_config_dict))


def _validate_lr_schedule_for(
    trainer_cls: type, trainer_config_dict: dict[str, Any]
) -> None:
    """Fail before training when the strategy would not apply the LR schedule.

    ``training.main.common.lr_schedule``/``lr_warmup_epochs`` are applied per
    epoch by ``Trainer.train()``; strategies whose loops bypass it declare
    ``supports_lr_schedule = False``.
    """
    validate_lr_schedule(
        trainer_config_dict.get("lr_schedule", "none"),
        trainer_config_dict.get("lr_warmup_epochs", 0),
        supported=bool(getattr(trainer_cls, "supports_lr_schedule", False)),
        owner=getattr(trainer_cls, "__name__", str(trainer_cls)),
    )


def _build_recurrent_trainer(trainer_config_dict: dict[str, Any]) -> RecurrentTrainer:
    """Build a recurrent trainer while preserving legacy kwarg precedence."""
    # Strategy-specific recurrent config takes precedence over generic fields.
    recurrent_cfg = trainer_config_dict.pop("recurrent_config", None)
    if recurrent_cfg is not None and not isinstance(recurrent_cfg, dict):
        recurrent_cfg = to_plain_dict(recurrent_cfg)
    if isinstance(recurrent_cfg, dict):
        # Set strategy overrides take precedence over common settings; None
        # means "inherit training.main.common".
        trainer_config_dict.update(
            {key: value for key, value in recurrent_cfg.items() if value is not None}
        )

    task = trainer_config_dict.pop("task", "classification")
    use_amp = trainer_config_dict.pop("use_amp", False)
    lr_schedule = trainer_config_dict.pop("lr_schedule", "none")
    lr_warmup_epochs = trainer_config_dict.pop("lr_warmup_epochs", 0)
    validate_lr_schedule(
        lr_schedule, lr_warmup_epochs, supported=True, owner="RecurrentTrainer"
    )
    _apply_enhanced_configs(trainer_config_dict)
    return RecurrentTrainer(
        task=task,
        use_amp=use_amp,
        lr_schedule=lr_schedule,
        lr_warmup_epochs=lr_warmup_epochs,
        **trainer_config_dict,
    )


def _build_multi_stage_trainer(
    trainer_config_dict: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer,
    analysis_manager: Optional[object],
) -> MultiStageTrainer:
    """Build a multi-stage trainer from normalized factory kwargs."""
    raw_multi_stage_config = trainer_config_dict.pop("multi_stage_config", None)
    multi_stage_config = _build_multi_stage_config(raw_multi_stage_config)
    # training.main.optimizer, used by stages that rebuild their optimizer.
    optimizer_config = trainer_config_dict.pop("optimizer_config", None)
    if isinstance(optimizer_config, dict):
        optimizer_config = OptimizerConfig(**optimizer_config)

    _apply_enhanced_configs(trainer_config_dict)
    _validate_lr_schedule_for(MultiStageTrainer, trainer_config_dict)

    base_trainer_config = dict(trainer_config_dict)
    return MultiStageTrainer(
        multi_stage_config=multi_stage_config,
        base_optimizer=optimizer,
        base_trainer_config=base_trainer_config,
        analysis_manager=analysis_manager,
        optimizer_config=optimizer_config,
    )


def get_trainer(
    strategy: str,
    optimizer: torch.optim.Optimizer,
    trainer_config_dict: dict[str, Any],
    analysis_manager: Optional[object] = None,
) -> Trainer:
    """
    Factory function to create training strategies.

    Args:
        strategy: String specifying the training strategy
        optimizer: The optimizer to use
        trainer_config_dict: The trainer configuration dict
        analysis_manager: Optional analysis manager

    Returns:
        An instance of the requested training strategy
    """
    strategy = strategy.lower()
    if strategy in DEPRECATED_STRATEGY_ALIASES:
        target = DEPRECATED_STRATEGY_ALIASES[strategy]
        if strategy not in _WARNED_STRATEGY_ALIASES:
            _WARNED_STRATEGY_ALIASES.add(strategy)
            logger.warning(
                "Strategy %r now runs as %r: %s computes each block's exact local "
                "gradient, which includes the conductance and driving-force terms "
                "%r approximated. Write %r.",
                strategy,
                target,
                target,
                strategy,
                target,
            )
        strategy = target
    trainer_config_dict["analysis_manager"] = analysis_manager
    trainer_config_dict["optimizer"] = optimizer

    if (
        strategy not in TRAINER_STRATEGY_REGISTRY
        and strategy not in SPECIAL_TRAINER_STRATEGIES
    ):
        load_plugins()
    if strategy in TRAINER_STRATEGY_REGISTRY:
        _apply_enhanced_configs(trainer_config_dict)
        spec = TRAINER_STRATEGY_REGISTRY[strategy]
        trainer_config_dict.update(spec.get("fixed_kwargs", {}))
        trainer_cls = _resolve_trainer_class(spec["class"])
        if not getattr(trainer_cls, "applies_own_lr_schedule", False):
            _validate_lr_schedule_for(trainer_cls, trainer_config_dict)
        return trainer_cls(**trainer_config_dict)

    if strategy == "recurrent":
        return _build_recurrent_trainer(trainer_config_dict)
    elif strategy == "multi_stage":
        return _build_multi_stage_trainer(
            trainer_config_dict,
            optimizer=optimizer,
            analysis_manager=analysis_manager,
        )
    else:
        supported = get_available_trainer_strategies()
        raise ValueError(
            f"Unknown training strategy '{strategy}'. Supported strategies: {supported}"
        )
