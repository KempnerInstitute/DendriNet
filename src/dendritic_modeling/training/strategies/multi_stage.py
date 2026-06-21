"""
Multi-stage training strategy.

This module implements a trainer that can execute multiple training stages
sequentially, each with different learning strategies and configurations.
"""

import json
import logging
import os
import shutil
from typing import Any, Optional

import torch
import torch.nn as nn

from dendritic_modeling.config import ParamGroupsConfig
from dendritic_modeling.config.multi_stage_training import (
    MultiStageTrainingConfig,
    TrainingStageConfig,
)
from dendritic_modeling.models import BaseModel
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer
from dendritic_modeling.training.strategies.standard import Trainer

logger = logging.getLogger(__name__)


def _remove_epoch_files(epochs_dir: str) -> None:
    existing_files = [f for f in os.listdir(epochs_dir) if f.startswith("epoch")]
    for filename in existing_files:
        os.remove(os.path.join(epochs_dir, filename))


def _epoch_files_by_local_epoch(epochs_dir: str) -> list[tuple[int, str]]:
    if not os.path.exists(epochs_dir):
        return []

    epoch_files = []
    for filename in os.listdir(epochs_dir):
        if not filename.startswith("epoch"):
            continue

        epoch_str = filename.replace("epoch", "")
        if epoch_str.endswith(".json"):
            epoch_str = epoch_str[:-5]

        if epoch_str.isdigit():
            epoch_files.append((int(epoch_str), filename))

    epoch_files.sort(key=lambda item: item[0])
    return epoch_files


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


def _concatenate_stage_performance_epochs(
    stages: list[TrainingStageConfig],
    original_save_root: str,
) -> None:
    main_epochs_dir = os.path.join(original_save_root, "performance", "epochs")
    os.makedirs(main_epochs_dir, exist_ok=True)
    _remove_epoch_files(main_epochs_dir)

    global_epoch = 1
    for stage_idx, _stage_config in enumerate(stages):
        stage_name = f"stage_{stage_idx + 1}"
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

    analysis_manager.run_analysis(filename=f"{stage_name}", training=False)

    if original_save_root is not None:
        analysis_manager.save_root = original_save_root


def _build_stage_trainer_config(
    base_trainer_config: dict[str, Any],
    multi_stage_config: MultiStageTrainingConfig,
    stage_config: TrainingStageConfig,
) -> dict[str, Any]:
    stage_trainer_config = base_trainer_config.copy()
    stage_trainer_config.update(stage_config.trainer_config)

    if stage_config.epochs is not None:
        stage_trainer_config["epochs"] = stage_config.epochs
    if stage_config.batch_size is not None:
        stage_trainer_config["batch_size"] = stage_config.batch_size
    elif multi_stage_config.global_batch_size is not None:
        stage_trainer_config["batch_size"] = multi_stage_config.global_batch_size

    if stage_config.param_groups is not None:
        stage_trainer_config["param_groups"] = stage_config.param_groups

    stage_trainer_config["shuffle"] = stage_trainer_config.get(
        "shuffle", multi_stage_config.global_shuffle
    )
    stage_trainer_config["grad_clip_value"] = stage_trainer_config.get(
        "grad_clip_value", multi_stage_config.global_grad_clip_value
    )

    if stage_config.local_rule_config is not None:
        stage_trainer_config["local_rule_config"] = stage_config.local_rule_config

    return stage_trainer_config


class MultiStageTrainer(Trainer):
    """
    Multi-stage training strategy that executes multiple training phases sequentially.

    This trainer allows you to:
    - Train with standard backprop first, then switch to local learning
    - Use different hyperparameters for each stage
    - Save/load checkpoints between stages
    - Run analysis between stages
    """

    def __init__(
        self,
        multi_stage_config: MultiStageTrainingConfig,
        base_optimizer: torch.optim.Optimizer,
        base_trainer_config: dict[str, Any],
        analysis_manager: Optional[object] = None,
        **kwargs,
    ):
        """
        Initialize multi-stage trainer.

        Args:
            multi_stage_config: Configuration for multi-stage training
            base_optimizer: Base optimizer (can be recreated for each stage)
            base_trainer_config: Base trainer configuration (can be overridden per stage)
            analysis_manager: Optional analysis manager
            **kwargs: Additional arguments passed to base Trainer
        """
        # Remove optimizer from base_trainer_config if present to avoid duplicate
        base_trainer_config_copy = base_trainer_config.copy()
        base_trainer_config_copy.pop("optimizer", None)

        # Initialize with base config for compatibility
        super().__init__(optimizer=base_optimizer, **base_trainer_config_copy, **kwargs)

        self.multi_stage_config = multi_stage_config
        self.base_optimizer = base_optimizer
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
        if not self.multi_stage_config.stages:
            raise ValueError("No training stages defined in multi_stage_config")

        logger.info(
            f"Starting multi-stage training with {len(self.multi_stage_config.stages)} stages"
        )

        all_results = {
            "stages": [],
            "final_state_dict": None,
            "best_stage": None,
            "best_loss": float("inf"),
        }

        # Keep track of the best model across all stages
        best_overall_loss = float("inf")
        best_overall_state_dict = None
        best_stage_name = None

        original_save_root = (
            self.analysis_manager.save_root if self.analysis_manager else None
        )

        for stage_idx, stage_config in enumerate(self.multi_stage_config.stages):
            stage_name = f"stage_{stage_idx + 1}"

            logger.info(f"\n{'='*60}")
            logger.info(
                f"Starting Stage {stage_idx + 1}/{len(self.multi_stage_config.stages)}: {stage_name}"
            )
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

                all_results["stages"].append(
                    {
                        "stage_name": stage_name,
                        "stage_idx": stage_idx,
                        "results": stage_results,
                    }
                )

                # Track best model across stages
                if stage_results.get("best_loss", float("inf")) < best_overall_loss:
                    best_overall_loss = stage_results["best_loss"]
                    best_overall_state_dict = stage_results.get("best_state_dict")
                    best_stage_name = stage_name

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

        # Set final results
        all_results["final_state_dict"] = model.state_dict()
        all_results["best_stage"] = best_stage_name
        all_results["best_loss"] = best_overall_loss

        # Load best model if requested
        if (
            self.multi_stage_config.use_best_from_each_stage
            and best_overall_state_dict is not None
        ):
            logger.info(
                f"Loading best model from stage: {best_stage_name} (loss: {best_overall_loss:.4f})"
            )
            model.load_state_dict(best_overall_state_dict)

        if (
            self.multi_stage_config.analyze_between_stages
            and self.analysis_manager is not None
            and original_save_root is not None
        ):
            logger.info("Concatenating performance data across stages")
            _concatenate_stage_performance_epochs(
                self.multi_stage_config.stages,
                original_save_root,
            )

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

        stage_trainer_config = _build_stage_trainer_config(
            self.base_trainer_config,
            self.multi_stage_config,
            stage_config,
        )

        if stage_config.load_checkpoint is not None:
            logger.info(f"Loading checkpoint: {stage_config.load_checkpoint}")
            checkpoint = torch.load(stage_config.load_checkpoint)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)

        if stage_config.reset_model:
            logger.warning(f"Resetting model weights for stage {stage_name}")
            self._reset_model_weights(model)
        if stage_config.param_groups is not None:
            param_groups_config = ParamGroupsConfig(**stage_config.param_groups)

            base_opt = self.base_optimizer
            if hasattr(base_opt, "optimizer"):
                actual_optimizer = base_opt.optimizer
                optimizer_class = type(actual_optimizer)
                weight_decay = base_opt.weight_decay
            else:
                optimizer_class = type(base_opt)
                weight_decay = 0.0

            param_groups = model.get_param_groups(param_groups_config)
            new_optimizer = optimizer_class(param_groups)

            if weight_decay > 0:
                stage_optimizer = CustomWeightDecayOptimizer(
                    model=model, optimizer=new_optimizer, weight_decay=weight_decay
                )
            else:
                stage_optimizer = new_optimizer

        elif stage_config.reset_optimizer or stage_idx == 0:
            base_opt = self.base_optimizer
            if hasattr(base_opt, "optimizer"):
                actual_optimizer = base_opt.optimizer
                optimizer_class = type(actual_optimizer)
                weight_decay = base_opt.weight_decay
            else:
                optimizer_class = type(base_opt)
                weight_decay = 0.0

            new_optimizer = optimizer_class(model.parameters(), lr=0.001)

            if weight_decay > 0:
                stage_optimizer = CustomWeightDecayOptimizer(
                    model=model, optimizer=new_optimizer, weight_decay=weight_decay
                )
            else:
                stage_optimizer = new_optimizer
        else:
            stage_optimizer = self.base_optimizer

        from dendritic_modeling.training.factory import get_trainer

        stage_trainer = get_trainer(
            strategy=stage_config.learning_strategy,
            optimizer=stage_optimizer,
            trainer_config_dict=stage_trainer_config,
            analysis_manager=self.analysis_manager,
        )

        if self.save_path and stage_config.save_checkpoint:
            stage_trainer.save_path = f"{self.save_path}/stage_{stage_idx}_{stage_name}"

        logger.info(
            f"Training with {stage_config.learning_strategy} for {stage_trainer_config.get('epochs', 'default')} epochs"
        )
        stage_results = stage_trainer.train(model, train_data, valid_data)

        if hasattr(stage_trainer, "optimizer"):
            self.base_optimizer = stage_trainer.optimizer

        if stage_config.save_checkpoint and self.save_path:
            checkpoint_path = (
                f"{self.save_path}/checkpoint_stage_{stage_idx}_{stage_name}.pt"
            )

            if hasattr(stage_trainer, "optimizer"):
                trainer_optimizer = stage_trainer.optimizer
                if hasattr(trainer_optimizer, "optimizer"):
                    optimizer_state = trainer_optimizer.optimizer.state_dict()
                else:
                    optimizer_state = trainer_optimizer.state_dict()
            else:
                if stage_optimizer and hasattr(stage_optimizer, "optimizer"):
                    optimizer_state = stage_optimizer.optimizer.state_dict()
                elif stage_optimizer:
                    optimizer_state = stage_optimizer.state_dict()
                else:
                    optimizer_state = None

            checkpoint_data = {
                "model_state_dict": model.state_dict(),
                "stage_config": stage_config,
                "stage_results": stage_results,
            }
            if optimizer_state is not None:
                checkpoint_data["optimizer_state_dict"] = optimizer_state

            torch.save(checkpoint_data, checkpoint_path)
            logger.info(f"Saved checkpoint: {checkpoint_path}")

        if self.analysis_manager is not None and original_save_root is not None:
            self.analysis_manager.save_root = original_save_root

        return stage_results

    def _reset_model_weights(self, model: nn.Module):
        """Reset model weights to random initialization."""
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
