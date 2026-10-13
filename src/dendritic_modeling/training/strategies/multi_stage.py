"""
Multi-stage training strategy.

This module implements a trainer that can execute multiple training stages
sequentially, each with different learning strategies and configurations.
"""

import json
import logging
import os
import shutil
from collections.abc import Iterator
from dataclasses import asdict
from typing import Any, Optional

import torch
import torch.nn as nn

from dendritic_modeling.config import ParamGroupsConfig
from dendritic_modeling.config.keys import apply_renamed_fields
from dendritic_modeling.config.multi_stage_training import (
    MultiStageTrainingConfig,
    TrainingStageConfig,
    stage_names,
)
from dendritic_modeling.config.training import (
    LOCAL_RULE_STRATEGIES,
    CommonTrainerConfig,
    OptimizerConfig,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks.checkpoints import (
    atomic_torch_save,
    load_model_state_checkpoint,
)
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer
from dendritic_modeling.training.optimizers.factory import create_optimizer
from dendritic_modeling.training.strategies.standard import Trainer
from dendritic_modeling.utils.epoch_files import epoch_files_by_number
from dendritic_modeling.utils.general import convert_to_serializable
from dendritic_modeling.utils.hooks import iter_modules_matching

logger = logging.getLogger(__name__)


def _remove_epoch_files(epochs_dir: str) -> None:
    existing_files = [f for f in os.listdir(epochs_dir) if f.startswith("epoch")]
    for filename in existing_files:
        os.remove(os.path.join(epochs_dir, filename))


def _epoch_files_by_local_epoch(epochs_dir: str) -> list[tuple[int, str]]:
    return epoch_files_by_number(epochs_dir, require_json=False)


def _json_file_has_payload(path: str) -> bool:
    try:
        with open(path) as f:
            data = json.load(f)
        return bool(data)
    except Exception:
        return False


def _copy_valid_epoch_file(src_path: str, dest_path: str) -> bool:
    try:
        if not os.path.exists(src_path):
            return False
        if not _json_file_has_payload(src_path):
            return False
        shutil.copy(src_path, dest_path)
        return True
    except Exception:
        return False


def _is_resettable_module(module: nn.Module) -> bool:
    """Return whether a module participates in between-stage weight resets."""
    return hasattr(module, "reset_parameters") or isinstance(
        module, (nn.Linear, nn.Conv2d)
    )


def _reset_module_weights(module: nn.Module) -> None:
    """Reset one module using the historical multi-stage reset rule."""
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()
    elif isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def _iter_resettable_modules(model: nn.Module) -> Iterator[nn.Module]:
    """Yield resettable modules in module traversal order."""
    yield from iter_modules_matching(model, _is_resettable_module)


def _unresettable_parameter_names(model: nn.Module) -> list[str]:
    """Names of parameters the between-stage reset rule leaves unchanged.

    The rule re-initializes modules with ``reset_parameters`` and plain
    ``nn.Linear``/``nn.Conv2d`` layers. Parameters owned by other modules
    (for example ``TopKLinear.pre_w``, ``BlockLinear.log_weight``,
    reactivation gates or a learned output scale) are not reset.
    """
    covered: set[int] = set()
    for module in _iter_resettable_modules(model):
        covered.update(id(param) for param in module.parameters(recurse=False))
    return [
        name for name, param in model.named_parameters() if id(param) not in covered
    ]


def _validate_reset_model_stages(model: nn.Module, stages) -> None:
    """Fail before training when ``reset_model`` cannot reset every parameter."""
    if not any(getattr(stage, "reset_model", False) for stage in stages):
        return
    unresettable = _unresettable_parameter_names(model)
    if unresettable:
        shown = ", ".join(unresettable[:5])
        more = f" and {len(unresettable) - 5} more" if len(unresettable) > 5 else ""
        raise ValueError(
            "multi_stage reset_model re-initializes only modules with "
            "reset_parameters() and nn.Linear/nn.Conv2d layers; it would leave "
            f"these parameters unchanged: {shown}{more}. Start a new run from "
            "the model config instead of resetting between stages."
        )


def _concatenate_stage_performance_epochs(
    names: list[str],
    original_save_root: str,
) -> None:
    main_epochs_dir = os.path.join(original_save_root, "performance", "epochs")
    os.makedirs(main_epochs_dir, exist_ok=True)
    _remove_epoch_files(main_epochs_dir)

    global_epoch = 1
    for stage_name in names:
        stage_dir = os.path.join(original_save_root, stage_name)
        stage_epochs_dir = os.path.join(stage_dir, "performance", "epochs")

        for _local_epoch, epoch_file in _epoch_files_by_local_epoch(stage_epochs_dir):
            src_path = os.path.join(stage_epochs_dir, epoch_file)
            dest_file = f"epoch{global_epoch}.json"
            dest_path = os.path.join(main_epochs_dir, dest_file)

            if _copy_valid_epoch_file(src_path, dest_path):
                global_epoch += 1


def _prepare_stage_analysis_root(
    analysis_manager,
    original_save_root: str,
    stage_name: str,
) -> None:
    if analysis_manager is None or original_save_root is None:
        return

    stage_save_root = os.path.join(original_save_root, stage_name)
    os.makedirs(stage_save_root, exist_ok=True)
    analysis_manager.save_root = stage_save_root

    stage_epochs_dir = os.path.join(stage_save_root, "performance", "epochs")
    if os.path.exists(stage_epochs_dir):
        _remove_epoch_files(stage_epochs_dir)


def _run_between_stage_analysis(
    analysis_manager,
    stage_name: str,
    original_save_root: str,
) -> None:
    logger.info(f"Running analysis after stage {stage_name}")
    if original_save_root is not None:
        stage_save_root = os.path.join(original_save_root, stage_name)
        analysis_manager.save_root = stage_save_root

    try:
        analysis_manager.run_analysis(filename=f"{stage_name}", training=False)
    finally:
        if original_save_root is not None:
            analysis_manager.save_root = original_save_root


def _build_stage_trainer_config(
    base_trainer_config: dict[str, Any],
    multi_stage_config: MultiStageTrainingConfig,
    stage_config: TrainingStageConfig,
) -> dict[str, Any]:
    stage_trainer_config = base_trainer_config.copy()
    # training.main.learning_strategy_config serves the local-rule stages
    # that do not set their own; other stages do not read it.
    shared_rule_config = stage_trainer_config.pop("local_rule_config", None)
    stage_trainer_config.update(
        apply_renamed_fields(
            CommonTrainerConfig,
            stage_config.trainer_config,
            section="training.main.multi_stage_config.stages[].trainer_config",
        )
    )

    if stage_config.epochs is not None:
        stage_trainer_config["epochs"] = stage_config.epochs
    if stage_config.batch_size is not None:
        stage_trainer_config["batch_size"] = stage_config.batch_size

    if stage_config.param_groups is not None:
        stage_trainer_config["param_groups"] = stage_config.param_groups

    if stage_config.local_rule_config is not None:
        stage_trainer_config["local_rule_config"] = stage_config.local_rule_config
    elif (
        shared_rule_config is not None
        and str(stage_config.learning_strategy).lower() in LOCAL_RULE_STRATEGIES
    ):
        stage_trainer_config["local_rule_config"] = shared_rule_config

    return stage_trainer_config


def _stage_param_groups_config(
    stage_config: TrainingStageConfig,
    stage_trainer_config: dict[str, Any],
) -> ParamGroupsConfig:
    """Return the stage's param groups, else the configured common groups."""
    if stage_config.param_groups is not None:
        # Unknown stage keys fail here, as before.
        return ParamGroupsConfig(**stage_config.param_groups)
    raw = stage_trainer_config.get(
        "param_groups_config", stage_trainer_config.get("param_groups")
    )
    if isinstance(raw, ParamGroupsConfig):
        return raw
    allowed = set(ParamGroupsConfig.__dataclass_fields__)
    return ParamGroupsConfig(
        **{key: value for key, value in dict(raw or {}).items() if key in allowed}
    )


def _model_param_groups(model, param_groups_config: ParamGroupsConfig) -> list[dict]:
    if hasattr(model, "get_param_groups"):
        return model.get_param_groups(param_groups_config)
    return [{"params": list(model.parameters()), "lr": param_groups_config.lr}]


def _build_stage_optimizer(
    model: BaseModel,
    base_optimizer,
    stage_config: TrainingStageConfig,
    stage_idx: int,
    *,
    stage_trainer_config: Optional[dict[str, Any]] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
):
    """Return the optimizer of one stage, wrapped for sparse maintenance.

    A stage continues the incoming optimizer (and its state) unless it sets
    ``param_groups`` or ``reset_optimizer``. Then the optimizer is rebuilt
    from ``model.get_param_groups`` (the stage's ``param_groups``, else the
    common ones) with the configured ``OptimizerConfig``
    (``training.main.optimizer``); without one, the incoming optimizer's class
    and hyperparameters are reused. The result is always a
    ``CustomWeightDecayOptimizer`` with the stage's ``synapse_decay_rate`` and
    ``weight_boosting`` (the ``training.main.common`` values unless the
    stage's ``trainer_config`` overrides them), so sparse maintenance and
    rewiring run in every stage.
    """
    del stage_idx  # every stage follows the same rule
    stage_trainer_config = stage_trainer_config or {}
    base_wrapper = (
        base_optimizer
        if isinstance(base_optimizer, CustomWeightDecayOptimizer)
        else None
    )
    inner_optimizer = base_wrapper.optimizer if base_wrapper else base_optimizer
    synapse_decay_rate = float(
        stage_trainer_config.get(
            "synapse_decay_rate", base_wrapper.weight_decay if base_wrapper else 0.0
        )
        or 0.0
    )
    weight_boosting = bool(
        stage_trainer_config.get(
            "weight_boosting", base_wrapper.weight_boosting if base_wrapper else False
        )
    )

    rebuild = stage_config.param_groups is not None or stage_config.reset_optimizer
    if not rebuild and (
        base_wrapper is not None
        and base_wrapper.weight_decay == synapse_decay_rate
        and base_wrapper.weight_boosting == weight_boosting
    ):
        return base_wrapper

    if rebuild:
        param_groups = _model_param_groups(
            model, _stage_param_groups_config(stage_config, stage_trainer_config)
        )
        if optimizer_config is not None:
            inner_optimizer = create_optimizer(param_groups, optimizer_config)
        else:
            try:
                inner_optimizer = type(inner_optimizer)(
                    param_groups, **dict(inner_optimizer.defaults)
                )
            except TypeError as exc:
                raise ValueError(
                    "multi_stage cannot rebuild a "
                    f"{type(inner_optimizer).__name__} optimizer from its "
                    "defaults; pass the training.main.optimizer config "
                    "(MultiStageTrainer optimizer_config)."
                ) from exc

    return CustomWeightDecayOptimizer(
        model=model,
        optimizer=inner_optimizer,
        weight_decay=synapse_decay_rate,
        weight_boosting=weight_boosting,
        topology_controllers=getattr(base_wrapper, "topology_controllers", ()),
    )


def _load_stage_checkpoint(model: BaseModel, checkpoint_path: str) -> None:
    """Load a stage checkpoint (or a bare state dict) with the weights-only loader."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    model.load_state_dict(load_model_state_checkpoint(checkpoint_path))


def _build_stage_trainer(
    stage_config: TrainingStageConfig,
    stage_optimizer,
    stage_trainer_config: dict[str, Any],
    analysis_manager,
):
    from dendritic_modeling.training.factory import get_trainer

    return get_trainer(
        strategy=stage_config.learning_strategy,
        optimizer=stage_optimizer,
        trainer_config_dict=stage_trainer_config,
        analysis_manager=analysis_manager,
    )


# File each stage writes into its directory when save_checkpoint is set.
STAGE_CHECKPOINT_NAME = "stage_checkpoint.pt"


def _stage_directory(save_path: str, stage_name: str) -> str:
    """Create and return the directory of one stage's trainer artifacts."""
    directory = os.path.join(save_path, stage_name)
    os.makedirs(directory, exist_ok=True)
    return directory


def _optimizer_state_dict(optimizer):
    if optimizer and hasattr(optimizer, "optimizer"):
        return optimizer.optimizer.state_dict()
    if optimizer:
        return optimizer.state_dict()
    return None


def _stage_checkpoint_optimizer_state(stage_trainer, stage_optimizer):
    if hasattr(stage_trainer, "optimizer"):
        return _optimizer_state_dict(stage_trainer.optimizer)
    return _optimizer_state_dict(stage_optimizer)


def _save_stage_checkpoint(
    model,
    stage_config: TrainingStageConfig,
    stage_results: dict[str, Any],
    stage_trainer,
    stage_optimizer,
    checkpoint_path: str,
) -> None:
    optimizer_state = _stage_checkpoint_optimizer_state(stage_trainer, stage_optimizer)
    # Plain containers only, so the weights-only loader can read the file.
    checkpoint_data = {
        "model_state_dict": model.state_dict(),
        "stage_config": convert_to_serializable(asdict(stage_config)),
        "stage_results": json.loads(
            json.dumps(convert_to_serializable(stage_results), default=str)
        ),
    }
    if optimizer_state is not None:
        checkpoint_data["optimizer_state_dict"] = optimizer_state

    atomic_torch_save(checkpoint_data, checkpoint_path)
    logger.info(f"Saved checkpoint: {checkpoint_path}")


def _initial_multi_stage_results(final_state: str) -> dict[str, Any]:
    return {
        "stages": [],
        "final_state": final_state,
        "final_stage": None,
        "best_loss": float("inf"),
        "best_epoch": None,
        "final_state_dict": None,
    }


def _append_stage_result(
    all_results: dict[str, Any],
    stage_name: str,
    stage_idx: int,
    stage_results: dict[str, Any],
) -> None:
    all_results["stages"].append(
        {
            "stage_name": stage_name,
            "stage_idx": stage_idx,
            "results": stage_results,
        }
    )


def _update_best_stage(
    stage_results: dict[str, Any],
    best_overall_loss: float,
    best_overall_state_dict,
    best_stage_name: Optional[str],
    stage_name: str,
) -> tuple[float, Any, Optional[str]]:
    if stage_results.get("best_loss", float("inf")) < best_overall_loss:
        return (
            stage_results["best_loss"],
            stage_results.get("best_state_dict"),
            stage_name,
        )

    return best_overall_loss, best_overall_state_dict, best_stage_name


def _finalize_multi_stage_results(
    all_results: dict[str, Any],
    model: BaseModel,
    final_stage: Optional[str],
    final_stage_results: dict[str, Any],
) -> None:
    """Record the stage the run ends with and a copy of the final weights."""
    all_results["final_stage"] = final_stage
    all_results["best_loss"] = final_stage_results.get("best_loss", float("inf"))
    all_results["best_epoch"] = final_stage_results.get("best_epoch")
    all_results["final_state_dict"] = {
        name: tensor.detach().to("cpu", copy=True)
        for name, tensor in model.state_dict().items()
    }


def _check_best_stage_comparable(
    stages: list[TrainingStageConfig], stage_trainer_configs: list[dict[str, Any]]
) -> None:
    """Fail when final_state 'best_stage' would compare incomparable losses."""
    losses = {
        str(config.get("loss_function", "ce")).lower()
        for config in stage_trainer_configs
    }
    if len(losses) > 1:
        raise ValueError(
            "multi_stage final_state 'best_stage' compares the stages' validation "
            f"losses, but the stages use different loss functions {sorted(losses)}; "
            "use final_state 'last_stage'"
        )
    selected_differently = [
        stage.learning_strategy
        for stage in stages
        if str(stage.learning_strategy).lower() == "presentation_stream"
    ]
    if selected_differently:
        raise ValueError(
            "multi_stage final_state 'best_stage' compares validation losses, but "
            "presentation_stream stages select their best state by their own "
            "metric; use final_state 'last_stage'"
        )


class MultiStageTrainer(Trainer):
    """
    Multi-stage training strategy that executes multiple training phases sequentially.

    This trainer allows you to:
    - Train with standard backprop first, then switch to local learning
    - Use different hyperparameters for each stage
    - Save/load checkpoints between stages
    - Run analysis between stages
    """

    # Each stage builds its own trainer; a schedule belongs to a stage config.
    supports_lr_schedule = False

    def __init__(
        self,
        multi_stage_config: MultiStageTrainingConfig,
        base_optimizer: torch.optim.Optimizer,
        base_trainer_config: dict[str, Any],
        analysis_manager: Optional[object] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        **kwargs,
    ):
        """
        Initialize multi-stage trainer.

        Args:
            multi_stage_config: Configuration for multi-stage training
            base_optimizer: Base optimizer (can be recreated for each stage)
            base_trainer_config: Base trainer configuration (can be overridden per stage)
            analysis_manager: Optional analysis manager
            optimizer_config: Configured ``training.main.optimizer``; stages
                that rebuild their optimizer use it with ``create_optimizer``
            **kwargs: Additional arguments passed to base Trainer
        """
        # Remove optimizer from base_trainer_config if present to avoid
        # duplicate; the local-rule block is for the stages.
        base_trainer_config_copy = base_trainer_config.copy()
        base_trainer_config_copy.pop("optimizer", None)
        base_trainer_config_copy.pop("local_rule_config", None)

        # Initialize with base config for compatibility
        super().__init__(optimizer=base_optimizer, **base_trainer_config_copy, **kwargs)

        self.multi_stage_config = multi_stage_config
        self.base_optimizer = base_optimizer
        self.optimizer_config = optimizer_config
        self.base_trainer_config = base_trainer_config
        self.analysis_manager = analysis_manager
        self.stage_results = []

    def train(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
    ) -> dict[str, Any]:
        """
        Execute multi-stage training.

        Args:
            model: Model to train
            train_data: Training dataset
            valid_data: Validation dataset

        Returns:
            Dictionary containing results from all stages
        """
        self._require_single_process(
            "its stages load the unwrapped best state into the "
            "DistributedDataParallel wrapper and every rank writes the stage "
            "checkpoints"
        )
        if not self.multi_stage_config.stages:
            raise ValueError("No training stages defined in multi_stage_config")
        _validate_reset_model_stages(model, self.multi_stage_config.stages)

        stages = self.multi_stage_config.stages
        names = stage_names(stages)
        final_state = self.multi_stage_config.final_state
        if final_state == "best_stage":
            _check_best_stage_comparable(
                stages,
                [
                    _build_stage_trainer_config(
                        self.base_trainer_config, self.multi_stage_config, stage
                    )
                    for stage in stages
                ],
            )
        logger.info(f"Starting multi-stage training with {len(stages)} stages")

        all_results = _initial_multi_stage_results(final_state)

        # The lowest-loss stage (final_state 'best_stage') and the last stage
        # that finished
        best_overall_loss = float("inf")
        best_overall_state_dict = None
        best_stage_name = None
        last_stage_name, last_stage_results = None, {}
        stage_results_by_name: dict[str, dict[str, Any]] = {}

        original_save_root = (
            self.analysis_manager.save_root if self.analysis_manager else None
        )

        for stage_idx, (stage_config, stage_name) in enumerate(zip(stages, names)):
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting Stage {stage_idx + 1}/{len(stages)}: {stage_name}")
            logger.info(f"Learning strategy: {stage_config.learning_strategy}")
            logger.info(f"{'='*60}\n")

            try:
                stage_results = self._execute_stage(
                    model,
                    train_data,
                    valid_data,
                    stage_config,
                    stage_idx,
                    stage_name,
                    original_save_root,
                )

                _append_stage_result(
                    all_results,
                    stage_name,
                    stage_idx,
                    stage_results,
                )
                last_stage_name, last_stage_results = stage_name, stage_results
                stage_results_by_name[stage_name] = stage_results

                (
                    best_overall_loss,
                    best_overall_state_dict,
                    best_stage_name,
                ) = _update_best_stage(
                    stage_results,
                    best_overall_loss,
                    best_overall_state_dict,
                    best_stage_name,
                    stage_name,
                )

                if (
                    self.multi_stage_config.analyze_between_stages
                    and self.analysis_manager is not None
                ):
                    _run_between_stage_analysis(
                        self.analysis_manager,
                        stage_name,
                        original_save_root,
                    )

            except Exception as e:
                logger.error(f"Error in stage {stage_name}: {e!s}")
                if not self.multi_stage_config.continue_on_failure:
                    raise
                logger.warning("Continuing to next stage despite failure")

        final_stage = last_stage_name
        if final_state == "best_stage" and best_overall_state_dict is not None:
            logger.info(
                f"Loading the best stage: {best_stage_name} "
                f"(loss: {best_overall_loss:.4f})"
            )
            model.load_state_dict(best_overall_state_dict)
            final_stage = best_stage_name
        _finalize_multi_stage_results(
            all_results,
            model,
            final_stage,
            stage_results_by_name.get(final_stage, last_stage_results),
        )

        if (
            self.multi_stage_config.analyze_between_stages
            and self.analysis_manager is not None
            and original_save_root is not None
        ):
            logger.info("Concatenating performance data across stages")
            _concatenate_stage_performance_epochs(names, original_save_root)

        return all_results

    def _execute_stage(
        self,
        model: BaseModel,
        train_data: torch.utils.data.Dataset,
        valid_data: torch.utils.data.Dataset,
        stage_config: TrainingStageConfig,
        stage_idx: int,
        stage_name: str,
        original_save_root: str,
    ) -> dict[str, Any]:
        """Execute a single training stage."""

        _prepare_stage_analysis_root(
            self.analysis_manager,
            original_save_root,
            stage_name,
        )

        try:
            stage_trainer_config = _build_stage_trainer_config(
                self.base_trainer_config,
                self.multi_stage_config,
                stage_config,
            )

            if stage_config.load_checkpoint is not None:
                _load_stage_checkpoint(model, stage_config.load_checkpoint)

            if stage_config.reset_model:
                logger.warning(f"Resetting model weights for stage {stage_name}")
                self._reset_model_weights(model)

            stage_optimizer = _build_stage_optimizer(
                model,
                self.base_optimizer,
                stage_config,
                stage_idx,
                stage_trainer_config=stage_trainer_config,
                optimizer_config=self.optimizer_config,
            )

            stage_trainer = _build_stage_trainer(
                stage_config,
                stage_optimizer,
                stage_trainer_config,
                self.analysis_manager,
            )

            stage_directory = (
                _stage_directory(self.save_path, stage_name) if self.save_path else None
            )
            if stage_directory is not None:
                # Every stage writes its trainer artifacts into its own directory.
                stage_trainer.save_path = stage_directory

            logger.info(
                f"Training with {stage_config.learning_strategy} for {stage_trainer_config.get('epochs', 'default')} epochs"
            )
            stage_results = stage_trainer.train(model, train_data, valid_data)

            if hasattr(stage_trainer, "optimizer"):
                self.base_optimizer = stage_trainer.optimizer

            if stage_config.save_checkpoint and stage_directory is not None:
                checkpoint_path = os.path.join(stage_directory, STAGE_CHECKPOINT_NAME)

                _save_stage_checkpoint(
                    model,
                    stage_config,
                    stage_results,
                    stage_trainer,
                    stage_optimizer,
                    checkpoint_path,
                )

            return stage_results
        finally:
            if self.analysis_manager is not None and original_save_root is not None:
                self.analysis_manager.save_root = original_save_root

    def _reset_model_weights(self, model: nn.Module):
        """Reset model weights to random initialization."""
        for module in _iter_resettable_modules(model):
            _reset_module_weights(module)
