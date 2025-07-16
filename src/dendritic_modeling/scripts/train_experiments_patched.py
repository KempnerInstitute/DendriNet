#!/usr/bin/env python
"""
Patched train_experiments.py with small weight initialization fix.

This version scales down the initial weights in TopKLinear layers
to prevent NaN issues in shunting computation.
"""

import sys
import os

# Get the path to the original train_experiments.py
original_path = os.path.join(os.path.dirname(__file__), 'train_experiments.py')

# Import everything from the original
sys.path.insert(0, os.path.dirname(original_path))
from train_experiments import *

import torch
import torch.nn as nn


def patch_model_initialization(model, scale_factor=0.01):
    """
    Patch model initialization by scaling down TopKLinear weights.
    
    This prevents NaN issues in shunting computation by ensuring
    initial conductances are small.
    
    Args:
        model: The model to patch
        scale_factor: Factor to scale down weights (default: 0.01)
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'pre_w') and 'TopKLinear' in module.__class__.__name__:
                # Scale down the weights in log space
                # If weights are initialized with mean=0, they become exp(0)=1
                # We want them smaller, so subtract a constant in log space
                logger.info(f"Patching {name}: scaling weights by {scale_factor}")
                
                # Subtract log(1/scale_factor) to effectively multiply by scale_factor after exp()
                module.pre_w.data -= torch.log(torch.tensor(1.0 / scale_factor))
                
                # Alternative: directly set to small negative values
                # module.pre_w.data.uniform_(-5.0, -3.0)


# Override the original main function
original_main = main

def main(config_path: str, args=None):
    """Patched main function that applies weight scaling."""
    
    # Load config
    logger.info(f"Loaded configuration from {config_path}")
    config_dict = load_yaml_to_dict(config_path, config_dir=config_dir)
    config = ConfigFactory.create_config(config_dict)
    
    # Set up output directory and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{config.outputs.run_name}_{timestamp}"
    run_dir = Path(config.outputs.dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created output directory: {run_dir}")
    
    # Configure logging
    configure_logging(level=get_log_level())
    configure_logging(
        log_file=run_dir / "dendritic_modeling.log",
        level=get_log_level(),
    )
    
    # Set up file logger for training
    file_handler = logging.FileHandler(run_dir / "train.log")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to {run_dir / 'train.log'}")
    
    # Set random seed
    set_seed(config.train.seed)
    logger.info(f"Set random seed to {config.train.seed}")
    
    # Initialize W&B if enabled
    wandb_run = None
    if config.wandb.use_wandb:
        wandb_run = initialize_wandb(config, config_dict, run_name)
    
    # Load dataset
    logger.info(f"Loading dataset: {config.task.dataset}")
    
    # Convert task and train sections to dict format for compatibility
    task_dict = {
        'dataset': config.task.dataset,
        'data_path': config.task.data_path,
        'parameters': config.task.parameters or {}
    }
    
    train_dict = {
        'train_valid_split': config.train.train_valid_split,
        'seed': config.train.seed
    }
    
    train_ds, valid_ds, test_ds = get_unified_datasets(
        task_cfg=task_dict,
        train_cfg=train_dict,
        outputs_cfg=config.outputs
    )
    logger.info(f"Loaded datasets: {len(train_ds)} training, {len(valid_ds)} validation, {len(test_ds)} test samples")
    
    # Build model
    model_builder = ModelBuilder(config.model)
    model = model_builder.build()
    logger.info(f"Initialized model: {model.__class__.__name__}")
    
    # PATCH: Scale down initial weights
    logger.info("Applying weight scaling patch to prevent NaN...")
    patch_model_initialization(model, scale_factor=0.01)
    
    # Continue with the rest of the original main function
    # (training, analysis, etc.)
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create optimizer
    optimizer = create_optimizer(model, config.train.main)
    
    # Create data loaders
    batch_size = config.train.main.trainer.batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False) if valid_ds else None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False) if test_ds else None
    
    # Create trainer
    trainer_cls = get_trainer_class(config.train.main.learning_strategy)
    trainer = trainer_cls(optimizer=optimizer, **config.train.main.trainer)
    
    # Train model
    results = trainer.train(model, train_loader, valid_loader)
    
    # Save model
    model_path = run_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Run analysis if configured
    if hasattr(config, 'analysis'):
        logger.info("Running analyses...")
        analysis_manager = AnalysisManager(
            model=model,
            data={"train": train_ds, "valid": valid_ds, "test": test_ds},
            analysis_config=config.analysis,
            save_root=run_dir,
            device=device,
            train_config=config.train
        )
        analysis_manager.run_analysis(filename="final", training=False)
    
    # Finish W&B run
    if wandb_run:
        wandb.finish()
    
    logger.info(f"Training complete! Results saved to {run_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train dendritic models with experiments (PATCHED)"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file (YAML)",
    )
    
    args = parser.parse_args()
    main(config_path=args.config, args=args) 