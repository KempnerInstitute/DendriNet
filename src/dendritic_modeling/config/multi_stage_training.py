"""
Multi-stage training configuration.

Stages run in order on one model, each with its own learning strategy and
settings. ``training.main.strategies.multi_stage`` is read into
``MultiStageStrategyConfig``, which the trainer also uses at run time.
"""

from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional

from dendritic_modeling.config.base import BaseConfig
from dendritic_modeling.config.choices import normalize_choice

# Weights a multi-stage run ends with.
FINAL_STATES = ("last_stage", "best_stage")
# Stage settings written at stage level, never inside ``trainer_config``.
STAGE_LEVEL_KEYS = ("epochs", "batch_size", "param_groups", "local_rule_config")
_SECTION = "training.main.strategies.multi_stage"


@dataclass
class TrainingStageConfig(BaseConfig):
    """Configuration for a single training stage."""

    # Learning strategy for this stage (REQUIRED)
    learning_strategy: str = "standard"

    # Name of the stage's directory and log entries (default stage_<N>).
    stage_name: Optional[str] = None

    # Number of epochs for this stage (default: training.main.common.epochs)
    epochs: Optional[int] = None

    # Trainer options of this stage. They override the training.main.common
    # values for this stage, including synapse_decay_rate and weight_boosting
    # of the stage optimizer wrapper. epochs, batch_size, param_groups and
    # local_rule_config are written at stage level instead.
    trainer_config: dict[str, Any] = field(default_factory=dict)

    # Optional: override batch size for this stage
    batch_size: Optional[int] = None

    # Optional: override param_groups for this stage
    param_groups: Optional[dict[str, Any]] = None

    # Optional: rebuild the optimizer (fresh state) from training.main.optimizer
    # and the param groups before this stage. Otherwise the previous stage's
    # optimizer (the configured one for the first stage) continues.
    reset_optimizer: bool = False

    # Optional: whether to reset model weights before this stage (for ablation
    # studies). Only modules with reset_parameters() and nn.Linear/nn.Conv2d
    # layers can be reset; other parameters (e.g. Top-K pre_w) raise ValueError.
    reset_model: bool = False

    # Optional: checkpoint to load before starting this stage
    load_checkpoint: Optional[str] = None

    # Write <stage directory>/stage_checkpoint.pt (weights, stage config and
    # results) after this stage.
    save_checkpoint: bool = True

    # Local learning rule config of a local_ca stage (default:
    # training.main.learning_strategy_config)
    local_rule_config: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        duplicated = sorted(set(self.trainer_config or {}) & set(STAGE_LEVEL_KEYS))
        if duplicated:
            raise ValueError(
                f"{_SECTION}.stages[].trainer_config sets {duplicated}; write "
                "them at stage level, where they would otherwise override it"
            )
        if self.stage_name is not None:
            name = str(self.stage_name)
            if name in {"", ".", ".."} or "/" in name or "\\" in name:
                raise ValueError(
                    f"{_SECTION}.stages[].stage_name={self.stage_name!r} names the "
                    "stage's directory; use a plain file name"
                )


def stage_names(stages) -> list[str]:
    """Directory and log name of each stage (``stage_<N>`` when unnamed)."""
    return [
        stage.stage_name or f"stage_{index + 1}" for index, stage in enumerate(stages)
    ]


@dataclass
class MultiStageStrategyConfig(BaseConfig):
    """Multi-stage training: stages run in order on one model."""

    RENAMED_FIELDS: ClassVar[dict[str, tuple[str, str]]] = {
        # true selected the best stage and false the last one.
        "use_best_from_each_stage": ("final_state", "2026-10-09"),
    }

    # Whether to run analysis after each stage
    analyze_between_stages: bool = True
    # Weights the run ends with: "last_stage" (the final stage's result) or
    # "best_stage" (the stage with the lowest validation loss, which needs
    # every stage to use the same loss and selection rule).
    final_state: str = "last_stage"
    # Whether to continue to next stage if current stage fails
    continue_on_failure: bool = False
    # Stages to run in order (at least one when the strategy is selected)
    stages: list[TrainingStageConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if isinstance(self.final_state, bool):
            self.final_state = "best_stage" if self.final_state else "last_stage"
        self.final_state = normalize_choice(
            self.final_state, FINAL_STATES, field=f"{_SECTION}.final_state"
        )
        self.stages = [
            stage if isinstance(stage, TrainingStageConfig) else _stage(stage, index)
            for index, stage in enumerate(self.stages)
        ]
        names = stage_names(self.stages)
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(
                f"{_SECTION}.stages name {repeated} more than once; each stage "
                "needs its own directory"
            )


def _stage(stage, index: int) -> TrainingStageConfig:
    from dendritic_modeling.config.keys import build_section

    return build_section(
        TrainingStageConfig, stage, section=f"{_SECTION}.stages[{index}]"
    )


# The YAML schema and the trainer use one class; this is the former runtime name.
MultiStageTrainingConfig = MultiStageStrategyConfig


__all__ = [
    "FINAL_STATES",
    "STAGE_LEVEL_KEYS",
    "MultiStageStrategyConfig",
    "MultiStageTrainingConfig",
    "TrainingStageConfig",
    "stage_names",
]
