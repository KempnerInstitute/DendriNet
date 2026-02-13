#!/usr/bin/env python
"""
train_experiments.py
====================
Main training script for dendritic modeling experiments (single-GPU only).

This script handles:
- Configuration loading and setup
- Model initialization and training
- Encoder network pretraining (for dendritic models)
- Visualization and analysis
- Results saving and logging

For multi-GPU/FSDP training, use train_experiments_fsdp.py instead.

Usage:
    python train_experiments.py <config_path> [options]

Options:
    --device: Device to use (cuda or cpu)
    --wandb: Enable W&B logging
    --save_model: Save the trained model (deprecated)
    --save_results: Save training results
    --run_analysis: Run post-training analysis
"""

from __future__ import annotations

# Suppress TensorFlow warnings for cleaner output
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "2"  # Suppress TensorFlow INFO and WARNING messages
)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations warnings

# Core imports
import argparse
import logging
from dataclasses import asdict

import torch

# External imports
import torchinfo
import wandb
from omegaconf import DictConfig, OmegaConf

from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.config import load_config
from dendritic_modeling.config.training import ParamGroupsConfig
from dendritic_modeling.models import BaseModel
from dendritic_modeling.networks import Identity
from dendritic_modeling.plotting.visualizations.performance_plots import (
    PerformancePlotter,
)
from dendritic_modeling.scripts.script_utils import config_utils as _config_utils
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment
from dendritic_modeling.scripts.training.train_encoder_network import (
    load_train_encoder_network,
)
from dendritic_modeling.training import CustomWeightDecayOptimizer, get_trainer
from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.training_stability_validator import (
    apply_automatic_safeguards,
    suggest_optimal_config_for_model_size,
    validate_training_stability,
)

logger = logging.getLogger(__name__)

prepare_trainer_config = _config_utils.prepare_trainer_config
normalize_param_groups_config = getattr(
    _config_utils, "normalize_param_groups_config", None
)


if normalize_param_groups_config is None:
    # Backward-compatible fallback for environments where config_utils lacks this helper.
    def normalize_param_groups_config(param_groups):
        if param_groups is None:
            return ParamGroupsConfig()

        if isinstance(param_groups, ParamGroupsConfig):
            return param_groups

        if isinstance(param_groups, DictConfig):
            param_groups = OmegaConf.to_container(param_groups, resolve=True)

        if hasattr(param_groups, "asdict") and callable(param_groups.asdict):
            param_groups = param_groups.asdict()
        elif not isinstance(param_groups, dict) and hasattr(param_groups, "__dict__"):
            param_groups = {
                key: value
                for key, value in vars(param_groups).items()
                if not key.startswith("_")
            }

        if not isinstance(param_groups, dict):
            return ParamGroupsConfig()

        allowed_fields = set(ParamGroupsConfig.__dataclass_fields__.keys())
        sanitized = {k: v for k, v in param_groups.items() if k in allowed_fields}
        return ParamGroupsConfig(**sanitized)


def run_analysis(
    model: BaseModel,
    train_ds: torch.utils.data.Dataset,
    valid_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    run_save_path: str,
    analysis_manager: AnalysisManager | None = None,
    training: bool = False,
    training_results: dict | None = None,
    training_config=None,
):
    """Run analysis using the AnalysisManager."""
    logger.info("Running analyses...")
    if analysis_manager is not None:
        # Set pruning results if available
        if training_results and (
            "pre_pruning_performance" in training_results
            or "post_pruning_performance" in training_results
        ):
            analysis_manager.set_pruning_results(training_results)

        analysis_manager.run_analysis(filename="final", training=training)

        # Generate performance evolution plots if this is final analysis and training tracking was enabled
        if not training and analysis_manager.config.performance_analysis.training:
            performance_dir = os.path.join(run_save_path, "performance", "epochs")
            logger.info(
                f"Checking for performance evolution data in: {performance_dir}"
            )
            if os.path.exists(performance_dir):
                logger.info(
                    "Performance directory exists. Generating performance evolution plots..."
                )
                try:
                    # Get final epoch from training config if available
                    final_epoch = None
                    if (
                        training_config is not None
                        and hasattr(training_config, "main")
                        and hasattr(training_config.main, "common")
                        and hasattr(training_config.main.common, "epochs")
                    ):
                        final_epoch = training_config.main.common.epochs

                    # Get pruning results if available
                    pruning_performance = getattr(
                        analysis_manager, "pruning_results", None
                    )
                    logger.info(
                        f"Pruning results available: {pruning_performance is not None}"
                    )

                    plotter = PerformancePlotter()
                    plotter.plot_performance_evolution(
                        performance_dir=performance_dir,
                        save_path=os.path.join(run_save_path, "performance"),
                        pruning_performance=pruning_performance,
                        final_epoch=final_epoch,
                    )
                    logger.info("Performance evolution plots generated successfully")
                except Exception as e:
                    logger.error(f"Failed to generate performance evolution plots: {e}")
                    import traceback

                    logger.error(traceback.format_exc())
            else:
                logger.warning(
                    f"Performance directory does not exist: {performance_dir}"
                )

        logger.info("Analysis completed")


def main(config_path: str, args=None):
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    logger.info("Validating configuration...")
    validation_passed = validate_training_stability(config)
    if not validation_passed:
        logger.error(
            "Configuration validation failed. Please address the errors above."
        )
        raise ValueError("Invalid configuration - critical errors detected")

    apply_automatic_safeguards(config)
    logger.info("Configuration safeguards applied")

    if args:
        if args.learning_strategy:
            config.training.main.strategy = args.learning_strategy
            logger.info(f"Override learning strategy: {args.learning_strategy}")
        if args.output_dir:
            config.outputs.results_dir = args.output_dir
            logger.info(f"Override output directory: {args.output_dir}")
        if args.run_name:
            config.outputs.run_name = args.run_name
            logger.info(f"Override run name: {args.run_name}")

    # Extract config sections
    experiment_config = config.experiment
    data_config = config.data
    model_config = config.model
    training_config = config.training
    analysis_config = config.analysis
    wandb_config = config.wandb
    outputs_config = config.outputs

    (run_save_path, train_ds, valid_ds, test_ds, model, encoder_network) = (
        setup_environment(
            model_config=model_config,
            training_config=training_config,
            data_config=data_config,
            wandb_config=wandb_config,
            outputs_config=outputs_config,
            experiment_config=experiment_config,
        )
    )

    logger.info("Network architecture summary before training:")
    try:
        sample_input: torch.Tensor = train_ds[0][0]
        summary = torchinfo.summary(
            model,
            input_size=(100, *sample_input.shape),
            col_names=["input_size", "output_size", "num_params"],
            col_width=25,
            row_settings=["depth"],
            depth=7,
        )
        logger.info("Model summary generated (see console output)")
        if wandb.run:
            wandb.log(
                {
                    "total_params": summary.total_params,
                    "trainable_params": summary.trainable_params,
                }
            )
    except Exception as e:
        logger.error(f"Failed to generate model summary before training: {e}")

    save_dict(asdict(config), run_save_path, "config.json")

    (encoder_network, encoded_train_ds, encoded_valid_ds, encoded_test_ds) = (
        load_train_encoder_network(
            encoder_network=encoder_network,
            encoder_train_config=training_config.encoder,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            encoder_network_config=model_config.encoder,
            task_config=data_config,
        )
    )

    encoded_data = {
        "train": encoded_train_ds,
        "valid": encoded_valid_ds,
        "test": encoded_test_ds,
    }
    analysis_manager = AnalysisManager(
        model=model,
        data=encoded_data,
        analysis_config=analysis_config,
        save_root=run_save_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Fix for Identity encoder
    if isinstance(encoder_network, Identity):
        model.encoder_network = encoder_network
    else:
        model.encoder_network = Identity(encoder_network.encoder.output_dim)

    main_net_save_path = os.path.join(run_save_path, "main_network")
    os.makedirs(main_net_save_path, exist_ok=True)

    # Get main training config
    main_train_config = training_config.main

    # Prepare trainer configuration
    main_trainer_config_dict = prepare_trainer_config(
        main_train_config=main_train_config,
        experiment_config=experiment_config,
        save_path=main_net_save_path,
    )

    # Extract config values (handle both dict from sweeps and object from single runs)
    if isinstance(main_train_config, dict):
        common_config = main_train_config.get("common", {})
        strategy = main_train_config.get("strategy", "standard")
    else:
        common_config = main_train_config.common
        strategy = main_train_config.strategy

    # Extract from common_config (also handle dict vs object)
    if isinstance(common_config, dict):
        param_groups = common_config.get("param_groups", {})
        weight_decay_rate = common_config.get("weight_decay_rate", 0.0)
        weight_boosting = common_config.get("weight_boosting", False)
        common_epochs = common_config.get("epochs", 100)
    else:
        param_groups = common_config.param_groups
        weight_decay_rate = common_config.weight_decay_rate
        weight_boosting = getattr(common_config, "weight_boosting", False)
        common_epochs = common_config.epochs

    param_groups = normalize_param_groups_config(param_groups)

    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(model.get_param_groups(param_groups)),
        weight_decay=weight_decay_rate,
        weight_boosting=weight_boosting,
    )

    original_pruning_config = main_trainer_config_dict.get("pruning", {})
    main_trainer_config_dict["pruning"] = {}

    # Check if using multi-stage training
    if strategy == "multi_stage":
        if "multi_stage_config" not in main_trainer_config_dict:
            raise ValueError(
                "Learning strategy is 'multi_stage' but no 'multi_stage_config' found"
            )
        logger.info("Using multi-stage training strategy")

    main_trainer = get_trainer(
        strategy=strategy,
        optimizer=optimizer,
        trainer_config_dict=main_trainer_config_dict,
        analysis_manager=analysis_manager,
    )

    training_results = main_trainer.train(
        model=model, train_data=encoded_train_ds, valid_data=encoded_valid_ds
    )

    nparams = 0
    for param in model.parameters():
        nparams += param.numel()
    if wandb.run is not None:
        wandb.log({"nparams": nparams})
    save_dict({"nparams": nparams}, run_save_path, "nparams")

    suggestions = suggest_optimal_config_for_model_size(nparams)
    if suggestions:
        logger.info(
            f"=== MODEL SIZE RECOMMENDATIONS (Model has {nparams:,} parameters) ==="
        )
        for key, value in suggestions.items():
            current_value = getattr(config.training, key, "Not available")
            if current_value != value:
                logger.info(f"  Consider: {key} = {value} (current: {current_value})")
        logger.info("=" * 50)

    # Run analysis if enabled
    any_analysis_enabled = any(
        [
            analysis_config.performance_analysis.enabled,
            analysis_config.information_analysis.enabled,
            analysis_config.synapse_turnover_analysis.enabled,
            analysis_config.noise_perturbation_analysis.enabled,
            analysis_config.single_layer_contribution.enabled,
            analysis_config.ablation_analysis.enabled,
            analysis_config.weight_analysis.enabled,
            analysis_config.synaptic_activation_analysis.enabled,
            analysis_config.branch_activation_analysis.enabled,
            hasattr(analysis_config, "correlation_analysis")
            and analysis_config.correlation_analysis.enabled,
        ]
    )

    if any_analysis_enabled and strategy != "multi_stage":
        run_analysis(
            model=model,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            run_save_path=run_save_path,
            analysis_manager=analysis_manager,
            training_config=training_config,
            training=False,
        )
    elif strategy == "multi_stage":
        if analysis_config.performance_analysis.training:
            performance_dir = os.path.join(run_save_path, "performance", "epochs")
            if os.path.exists(performance_dir):
                try:
                    # Extract multi_stage config
                    multi_stage = main_train_config.strategies.multi_stage
                    stages = multi_stage.stages

                    total_epochs = sum(
                        getattr(stage, "epochs", common_epochs) for stage in stages
                    )

                    plotter = PerformancePlotter()
                    plotter.plot_performance_evolution(
                        performance_dir=performance_dir,
                        save_path=os.path.join(run_save_path, "performance"),
                        pruning_performance=getattr(
                            analysis_manager, "pruning_results", None
                        ),
                        final_epoch=total_epochs,
                    )
                except Exception as e:
                    logger.error(f"Failed to generate performance plots: {e}")

    # Now apply pruning if enabled (separate step)
    if original_pruning_config and original_pruning_config.get("enabled", False):
        logger.info("Applying pruning as separate step...")
        # Restore pruning config and create new trainer for pruning
        main_trainer_config_dict["pruning"] = original_pruning_config
        main_trainer_config_dict["reporting"] = main_trainer_config_dict.get(
            "reporting", {}
        )

        pruning_trainer = get_trainer(
            strategy=strategy,
            optimizer=optimizer,
            trainer_config_dict=main_trainer_config_dict,
            analysis_manager=analysis_manager,
        )

        # Initialize trainer attributes (device, etc.) without training
        pruning_trainer._initialize_attributes(model, common_epochs)

        # Apply pruning and evaluate
        pruning_results = pruning_trainer._apply_pruning_and_evaluate(
            model=model, train_data=encoded_train_ds, valid_data=encoded_valid_ds
        )
        training_results.update(pruning_results)  # Merge results
        logger.info("Pruning completed")

    # Fix for Identity encoder
    if isinstance(encoder_network, Identity):
        model.encoder_network = encoder_network
    else:
        model.encoder_network = encoder_network.encoder
    torch.save(model.state_dict(), os.path.join(run_save_path, "final_model.pt"))

    if (
        hasattr(wandb_config, "use_wandb")
        and wandb_config.use_wandb
        and wandb.run is not None
    ):
        wandb.finish()

    logger.info("Training experiment completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run dendritic modeling training and evaluation."
    )
    parser.add_argument("config", help="Path to the configuration YAML file.")
    parser.add_argument(
        "--output_dir", help="Override output directory specified in config."
    )
    parser.add_argument("--run_name", help="Override run name specified in config.")
    parser.add_argument(
        "--learning_strategy",
        choices=[
            "mle",
            "freeze_branches",
            "freeze_branches_reversed",
            "freeze_branches_with_pretraining",
            "freeze_branches_reversed_with_pretraining",
            "two_step",
            "two_step_kl",
            "voltage_stabilization",
        ],
        help="Override learning strategy specified in config.",
    )

    args = parser.parse_args()
    main(config_path=args.config, args=args)
