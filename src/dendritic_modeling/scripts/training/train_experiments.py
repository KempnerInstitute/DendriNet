#!/usr/bin/env python
"""
train_experiments.py
====================
Main training script for dendritic modeling experiments.

This script handles:
- Configuration loading and setup
- Model initialization and training
- Encoder network pretraining (for dendritic models)
- Visualization and analysis
- Results saving and logging

Usage:
    python train_experiments.py <config_path> [options]

Options:
    --device: Device to use (cuda or cpu)
    --wandb: Enable W&B logging
    --save_model: Save the trained model
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
import sys
import warnings
from dataclasses import asdict
from datetime import datetime

# Optional imports (with fallbacks)
try:
    import torchinfo
    TORCHINFO_AVAILABLE = True
except ImportError:
    TORCHINFO_AVAILABLE = False
    warnings.warn("torchinfo not available. Model summary will be skipped.")

# Distributed training imports
try:
    import torch.distributed as dist
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

import torch
import wandb
from omegaconf import OmegaConf

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.analysis import AnalysisManager
from dendritic_modeling.config import (
    ModelConfig,
    load_config,
)
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.models import BaseModel, Classifier, Regressor
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.training.train_encoder_network import load_train_encoder_network
from dendritic_modeling.scripts.script_utils.setup_utils import setup_environment, save_model, load_model, initialize_model
from dendritic_modeling.training import CustomWeightDecayOptimizer, get_trainer
from dendritic_modeling.utils import save_dict, validate_config, set_seed
from dendritic_modeling.plotting.visualizations.information_analysis import plot_information_analysis
from dendritic_modeling.plotting.visualizations.general_analysis_plots import plot_synapse_turnover
from dendritic_modeling.utils.training_validation import (
    validate_optimization_combinations,
    suggest_optimal_config_for_model_size,
    apply_automatic_safeguards
)

# FSDP import (conditional)
try:
    from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer
    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False


def _run_fsdp_training(config):
    """
    Intelligent FSDP training delegation.
    
    This function determines the appropriate FSDP training approach based on:
    1. Whether we're in a distributed environment (torchrun/SLURM)
    2. The available resources and model requirements
    3. Single-node vs multi-node setup
    """
    import subprocess
    import sys
    from pathlib import Path
    
    # Check if we're in a distributed environment
    is_torchrun = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    is_slurm = "SLURM_JOB_ID" in os.environ
    
    if not is_torchrun and not is_slurm:
        # Not in distributed environment - provide helpful guidance
        logger.error("FSDP training requires a distributed environment!")
        logger.error("For single-node FSDP, use:")
        logger.error(f"  torchrun --nproc_per_node=<num_gpus> {sys.argv[0]} {' '.join(sys.argv[1:])}")
        logger.error("For multi-node FSDP, use SLURM with generate_slurm_job.py")
        logger.error("Alternatively, use train_experiments_fsdp.py directly")
        raise RuntimeError("FSDP requires distributed environment. See error messages above for proper usage.")
    
    # We're in a distributed environment - delegate to the proper FSDP script
    current_script_dir = Path(__file__).parent
    fsdp_script = current_script_dir / "train_experiments_fsdp.py"
    
    if not fsdp_script.exists():
        raise RuntimeError(f"FSDP script not found at {fsdp_script}")
    
    # Detect environment details for better logging
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    logger.info("Delegating to full FSDP implementation...")
    logger.info(f"Environment: {'SLURM' if is_slurm else 'torchrun'}")
    logger.info(f"World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
    logger.info(f"Using FSDP script: {fsdp_script}")
    
    # Get the config path from command line arguments
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not config_path:
        raise RuntimeError("No config path provided")
    
    logger.info(f"Config file: {config_path}")
    logger.info(f"Arguments being passed: {' '.join(sys.argv)}")
    
    # Delegate to the FSDP script with the same arguments
    try:
        # Use execv to replace current process - cleaner than subprocess for this use case
        os.execv(sys.executable, [sys.executable, str(fsdp_script), config_path])
    except Exception as e:
        logger.error(f"Failed to delegate to FSDP script: {e}")
        raise RuntimeError(f"FSDP delegation failed: {e}")


def run_analysis(
    model: BaseModel,
    train_ds: torch.utils.data.Dataset,
    valid_ds: torch.utils.data.Dataset,
    test_ds: torch.utils.data.Dataset,
    run_save_path: str,
    analysis_manager: AnalysisManager | None = None,
    training: bool = False,
    training_results: dict | None = None,
):
    """Run analysis using the AnalysisManager."""
    logger.info("Running analyses...")
    if analysis_manager is not None:
        # Set pruning results if available
        if training_results and ('pre_pruning_performance' in training_results or 'post_pruning_performance' in training_results):
            analysis_manager.set_pruning_results(training_results)
        
        analysis_manager.run_analysis(filename="final", training=training)
        logger.info("Analysis completed")


def main(config_path: str, args=None):
    config = load_config(config_path)
    logger.info(f"Loaded configuration from {config_path}")

    # Validate configuration before proceeding
    logger.info("Validating configuration...")

    # Convert config to dict for validation
    config_dict = asdict(config) if hasattr(config, "__dataclass_fields__") else config

    if not validate_config(config_dict):
        logger.error("Configuration validation failed. Please fix the errors above.")
        raise ValueError("Invalid configuration")

    logger.info("Configuration validation passed")

    # Apply configuration safeguards and warnings
    logger.info("Applying configuration safeguards...")
    validation_passed = validate_optimization_combinations(config)
    if not validation_passed:
        logger.error("Configuration validation failed. Please address the errors above.")
        raise ValueError("Invalid configuration - critical errors detected")
    
    # Apply automatic safeguards
    apply_automatic_safeguards(config)
    logger.info("Configuration safeguards applied")

    if args:
        if args.learning_strategy:
            config.train.learning_strategy = args.learning_strategy
            logger.info(f"Override learning strategy: {args.learning_strategy}")
        if args.output_dir:
            # Override run_config.dir if it exists, otherwise create run_config
            if hasattr(config, "run_config") and config.run_config:
                config.run_config.dir = args.output_dir
            else:
                # For backwards compatibility, still update outputs.dir if run_config doesn't exist
                config.outputs.dir = args.output_dir
            logger.info(f"Override output directory: {args.output_dir}")
        if args.run_name:
            config.train.run_name = args.run_name
            logger.info(f"Override run name: {args.run_name}")
        if args.use_fsdp:
            # Enable FSDP mode
            logger.info("FSDP mode enabled - initializing distributed training")
            _run_fsdp_training(config)
            return

    model_config = config.model
    train_config = config.train
    task_config = config.task
    wandb_config = config.wandb
    analysis_config = config.analysis
    outputs_config = config.outputs
    run_config = getattr(config, "run_config", None)

    # EI config validation removed - let model initialization handle validation

    (run_save_path, train_ds, valid_ds, test_ds, model, encoder_network) = (
        setup_environment(
            model_config=model_config,
            train_config=train_config,
            task_config=task_config,
            wandb_config=wandb_config,
            outputs_config=outputs_config,
            run_config=run_config,
        )
    )

    logger.info("Network architecture summary before training:")
    try:
        sample_input: torch.Tensor = train_ds[0][0]
        if TORCHINFO_AVAILABLE:
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
        else:
            logger.warning("torchinfo is not available. Model summary not generated.")
    except Exception as e:
        logger.error(f"Failed to generate model summary before training: {e}")

    save_dict(asdict(config), run_save_path, "config.json")

    (encoder_network, encoded_train_ds, encoded_valid_ds, encoded_test_ds) = (
        load_train_encoder_network(
            encoder_network=encoder_network,
            encoder_train_config=train_config.encoder,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            encoder_network_config=model_config.encoder_network,
            task_config=task_config,
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
        train_config=train_config,
    )

    # Fix for Identity encoder
    if isinstance(encoder_network, Identity):
        model.encoder_network = encoder_network
    else:
        model.encoder_network = Identity(encoder_network.encoder.output_dim)

    main_net_save_path = os.path.join(run_save_path, "main_network")
    os.makedirs(main_net_save_path, exist_ok=True)

    main_train_config = train_config.main
    main_train_config.trainer.save_path = main_net_save_path
    main_trainer_config_dict = OmegaConf.to_container(
        main_train_config.trainer, resolve=True
    )

    # Add enhanced configurations for pruning, regularization, reporting, and parameter groups
    # These configurations are needed for the trainer factory to enable advanced features
    if hasattr(main_train_config, 'param_groups'):
        main_trainer_config_dict["param_groups"] = OmegaConf.to_container(
            main_train_config.param_groups, resolve=True
        )
    
    if hasattr(main_train_config, 'regularization'):
        main_trainer_config_dict["regularization"] = OmegaConf.to_container(
            main_train_config.regularization, resolve=True
        )
    
    if hasattr(main_train_config, 'pruning'):
        main_trainer_config_dict["pruning"] = OmegaConf.to_container(
            main_train_config.pruning, resolve=True
        )
    
    if hasattr(main_train_config, 'reporting'):
        main_trainer_config_dict["reporting"] = OmegaConf.to_container(
            main_train_config.reporting, resolve=True
        )

    # Add checkpoint configuration if specified
    if (
        hasattr(main_train_config.trainer, "save_every_n_epochs")
        and main_train_config.trainer.save_every_n_epochs is not None
    ):
        main_trainer_config_dict["save_every_n_epochs"] = (
            main_train_config.trainer.save_every_n_epochs
        )
        main_trainer_config_dict["checkpoint_dir"] = run_save_path

    # Add profiling configuration if specified
    enable_profiling = getattr(config.train, 'enable_profiling', False)
    if enable_profiling:
        main_trainer_config_dict["enable_profiling"] = True
        main_trainer_config_dict["profiling_output_dir"] = os.path.join(run_save_path, "profiling")
        logger.info("Profiling enabled for this training run")
    
    # Add AMP configuration
    enable_amp = getattr(config.train, 'enable_amp', False)
    main_trainer_config_dict["enable_amp"] = enable_amp
    
    # Add hooks configuration
    enable_hooks = getattr(config.train, 'enable_hooks', True)
    main_trainer_config_dict["enable_hooks"] = enable_hooks

    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(
            model.get_param_groups(main_train_config.param_groups)
        ),
        weight_decay=main_train_config.weight_decay_rate,
    )

    main_trainer = get_trainer(
        strategy=main_train_config.learning_strategy,
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
    
    # Provide configuration suggestions based on model size
    suggestions = suggest_optimal_config_for_model_size(nparams)
    if suggestions:
        logger.info(f"=== MODEL SIZE RECOMMENDATIONS (Model has {nparams:,} parameters) ===")
        for key, value in suggestions.items():
            current_value = getattr(config.train, key, "Not set") if hasattr(config.train, key) else "Not available"
            if current_value != value:
                logger.info(f"  Consider: {key} = {value} (current: {current_value})")
        logger.info("=" * 50)

    # Only run analysis if any analysis is enabled (excluding training-only analysis)
    any_analysis_enabled = any([
        analysis_config.performance_analysis.enabled,
        analysis_config.information_analysis.enabled,
        analysis_config.synapse_turnover_analysis.enabled,
        analysis_config.noise_perturbation_analysis.enabled,
        analysis_config.single_layer_contribution.enabled,
        analysis_config.ablation_analysis.enabled,
        analysis_config.weight_analysis.enabled,
        analysis_config.synaptic_activation_analysis.enabled,
        analysis_config.branch_activation_analysis.enabled,
        hasattr(analysis_config, 'correlation_analysis') and analysis_config.correlation_analysis.enabled,
    ])
    
    if any_analysis_enabled:
        run_analysis(
            model=model,
            train_ds=train_ds,
            valid_ds=valid_ds,
            test_ds=test_ds,
            run_save_path=run_save_path,
            analysis_manager=analysis_manager,
            training_results=training_results,
        )
    else:
        logger.info("No analysis enabled - skipping analysis for speed")

    # Fix for Identity encoder
    if isinstance(encoder_network, Identity):
        model.encoder_network = encoder_network
    else:
        model.encoder_network = encoder_network.encoder
    save_model(model, run_save_path, "final_model.pt")

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
    parser.add_argument(
        "--use_fsdp",
        action="store_true",
        help="Enable FSDP (Fully Sharded Data Parallel) training. Requires torchrun or SLURM environment. "
             "For single-node: 'torchrun --nproc_per_node=<gpus> train_experiments.py config.yaml --use_fsdp'. "
             "For multi-node: Use generate_slurm_job.py or train_experiments_fsdp.py directly.",
    )
    args = parser.parse_args()
    main(config_path=args.config, args=args)
