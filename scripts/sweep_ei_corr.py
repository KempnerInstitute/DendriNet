"""
sweep_ei_corr.py
---------------------
Script that performs parameter sweeps for EI correlation experiments.
Supports multiple random seeds per configuration and shunting/non-shunting variants.

EXPECTS TO BE RUN AS A SINGLE TASK within a SLURM array job,
receiving the path to its specific configuration file as the sole argument.
"""
import itertools
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import traceback

import pandas as pd
import torch
from omegaconf import OmegaConf, ListConfig

import wandb
from dendritic_modeling import logger, logger_manager
from dendritic_modeling.config import load_config
from dendritic_modeling.learning_strategies import (
    CustomWeightDecayOptimizer,
    get_trainer,
)
from dendritic_modeling.models import Classifier, ProbabilisticClassifier, DeepstRegressor, Regressor
from dendritic_modeling.networks import ExcitationInhibitionNetwork, MatchedParamMLP
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.utils import evaluate_accuracy, save_dict, set_seed, evaluate_mse

torch.backends.cudnn.deterministic = True


def initialize_model(model_cfg, visualization_enabled=False):
    task = model_cfg.task
    probabilistic = model_cfg.probabilistic
    net_type = model_cfg.network.type
    # Use OmegaConf.to_container to handle potential nested structures cleanly
    net_params_dict = OmegaConf.to_container(model_cfg.network.parameters, resolve=True)
    
    # Extract params safely, providing defaults
    config_branch_outputs = net_params_dict.get("enable_branch_outputs", False)
    enable_branch_outputs = config_branch_outputs or visualization_enabled
    learning_strategy = net_params_dict.get("learning_strategy", "mle")  # Default to mle for classification
    output_dim = net_params_dict.get("output_dim", 2)  # Default to 2 for binary classification

    logger.info(f"EINet branch outputs enabled: {enable_branch_outputs}")
    logger.info(f"Learning strategy: {learning_strategy}")
    logger.info(f"Task type: {task}")

    if task == "classification":
        if net_type == "MLP":
            logger.info(f"Running feedforward with {'probabilistic' if probabilistic else 'simple'} classifier")
            if 'input_dim' not in net_params_dict or 'output_dim' not in net_params_dict:
                raise ValueError("MLP requires input_dim and output_dim in network parameters")
            net = MatchedParamMLP(**net_params_dict)
            if probabilistic:
                model = ProbabilisticClassifier(net, net_params_dict['output_dim'])
            else:
                model = Classifier(net, net_params_dict['output_dim'])
            return model
        elif net_type == "EINet":
            logger.info(f"Running EINet with {'probabilistic' if probabilistic else 'simple'} classifier")

            # Add derived/fixed params back for constructor
            net_params_dict["enable_branch_outputs"] = enable_branch_outputs
            net_params_dict["learning_strategy"] = learning_strategy

            net = ExcitationInhibitionNetwork(**net_params_dict)
            if probabilistic:
                model = ProbabilisticClassifier(net, output_dim)
            else:
                model = Classifier(net, output_dim)
            return model
        else:
            raise ValueError(f"Invalid network type for classification: {net_type}")
    else:
        raise ValueError(f"Invalid task specified in config: {task}")


def train_and_evaluate(model, train_ds, valid_ds, test_ds, train_cfg, run_save_path):
    """Train model and evaluate performance."""
    # Ensure lr is a float, not a list/ListConfig
    lr_value = train_cfg.lr
    if isinstance(lr_value, (list, ListConfig)):
        if len(lr_value) == 1:
            lr_value = lr_value[0]
        else:
             raise TypeError(f"Learning rate must be a single value in task config, found: {lr_value}")
    lr_value = float(lr_value) # Cast to float

    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=lr_value),
        weight_decay=train_cfg.weight_decay_rate,
    )

    # Ensure learning strategy is a string
    learning_strategy = train_cfg.learning_strategy
    if not isinstance(learning_strategy, str):
         raise TypeError(f"Learning strategy must be a string, found: {learning_strategy}")
    
    # include early stopping
    early_stopping = train_cfg.early_stopping
    if early_stopping:
        patience = train_cfg.patience
    else:
        patience = 100

    trainer = get_trainer(learning_strategy, optimizer, suppress_prints=True)

    # Train the model
    trainer.train(
        model=model,
        train_data=train_ds,
        valid_data=valid_ds,
        epochs=train_cfg.epochs,
        batch_size=train_cfg.batch_size,
        grad_clip_value=train_cfg.grad_clip_value,
        shuffle=True,
        load_best_state_dict=True,
        early_stopping=early_stopping,
        patience=patience,
        plot_losses=True,
        save_path=run_save_path,
    )

    # For correlation tasks, we use classification accuracy
    train_acc, valid_acc, test_acc = evaluate_accuracy(
        model,
        train_ds,
        valid_ds,
        test_ds,
        save_path=run_save_path,
        filename="performance",
    )
    return {
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "test_acc": test_acc,
    }


def run_experiment(
    exp_config,
    train_ds,
    valid_ds,
    test_ds,
    run_save_path,
    results_file,
    # Parameters extracted for logging
    ee_synapses,
    ie_synapses,
    ei_synapses,
    excitatory_layer_sizes,
    inhibitory_layer_sizes,
    excitatory_branch_factors,
    lr,
    # New parameters for nonlinearity sweep
    reactivate_val,  # bool
    topk_s,          # string ("conductance_dynamic" or "none")
    blocklinear_s,   # string ("conductance_dynamic" or "none")
    reactivation_s,  # string ("inverse" or "none")
    reactivation_t,  # string (type like "param_tanh")
    topk_type_val,   # string ("standard" or "deepst")
    # W&B related parameters
    wandb_cfg,
    job_id,
    task_id,
    wandb_group_name,
):
    # Ensure seed is an integer
    seed_value = exp_config.train.seed
    if not isinstance(seed_value, int):
        try:
            seed_value = int(seed_value)
        except ValueError:
             raise TypeError(f"Seed must be an integer, found: {seed_value}")
    set_seed(seed_value)

    # WandB initialization
    use_wandb = getattr(wandb_cfg, "use_wandb", False)
    if use_wandb:
        # Ensure scalar for name components
        scalar_seed = get_scalar(exp_config.train.seed)
        base_run_name_from_config = getattr(exp_config.train, "run_name", get_scalar(exp_config.model.network.type))
        scalar_base_run_name = get_scalar(base_run_name_from_config)
        
        strategy_mode_indicator = "dynamic_strat"
        if topk_s == "none" and blocklinear_s == "none" and reactivation_s == "none":
            strategy_mode_indicator = "no_strat"
        
        wandb_run_name = (f"{scalar_base_run_name}_task{task_id}_s{scalar_seed}_"
                          f"act{str(reactivate_val)[0]}_{reactivation_t[:5]}_{strategy_mode_indicator}_topk{topk_type_val[:3]}")

        # Prepare a dictionary of hyperparameters for wandb.config
        flat_config_for_wandb = {
            "model_type": get_scalar(exp_config.model.network.type),
            "seed": scalar_seed,
            "ee_synapses": ee_synapses,
            "ie_synapses": ie_synapses,
            "ei_synapses": ei_synapses,
            "lr": lr,
            "epochs": get_scalar(exp_config.train.epochs),
            "batch_size": get_scalar(exp_config.train.batch_size),
            "weight_decay_rate": get_scalar(exp_config.train.weight_decay_rate),
            "early_stopping": get_scalar(getattr(exp_config.train, "early_stopping", False)),
            "patience": get_scalar(getattr(exp_config.train, "patience", 100)),
            "reactivate": reactivate_val,
            "topk_strategy": topk_s,
            "blocklinear_strategy": blocklinear_s,
            "reactivation_strategy": reactivation_s,
            "reactivation_type": reactivation_t,
            "topk_type": topk_type_val,
            "full_task_config": asdict(exp_config),
            "excitatory_layer_sizes": excitatory_layer_sizes,
            "inhibitory_layer_sizes": inhibitory_layer_sizes,
            "excitatory_branch_factors": excitatory_branch_factors,
        }

        wandb.init(
            project=getattr(wandb_cfg, "project", "dendrinet_corr_sweep"),
            entity=getattr(wandb_cfg, "entity", None),
            group=wandb_group_name,
            name=wandb_run_name,
            config=flat_config_for_wandb,
            tags=getattr(wandb_cfg, "tags", []),
            reinit=True
        )

    try:
        # Initialize and train model
        model = initialize_model(exp_config.model, visualization_enabled=False)
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

        results = train_and_evaluate(
            model, train_ds, valid_ds, test_ds, exp_config.train, run_save_path
        )

        # Log results
        logger.info(
            f"Train accuracy: {results['train_acc']:.3f}, "
            f"Valid accuracy: {results['valid_acc']:.3f}, "
            f"Test accuracy: {results['test_acc']:.3f}"
        )

        # Save results to CSV
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        file_exists = os.path.isfile(results_file)
        with open(results_file, "a") as f:
            if not file_exists:
                f.write("model,seed,ee,ie,ei,excitatory_layer_sizes,inhibitory_layer_sizes,excitatory_branch_factors,lr,reactivate,topk_strategy,blocklinear_strategy,reactivation_strategy,reactivation_type,topk_type,train_acc,valid_acc,test_acc\n")
            
            metrics = f"{results['train_acc']:.4f},{results['valid_acc']:.4f},{results['test_acc']:.4f}"
            
            # Convert list parameters to string representation for CSV
            excit_layers_str = str(excitatory_layer_sizes).replace(' ', '')
            inhib_layers_str = str(inhibitory_layer_sizes).replace(' ', '')
            excit_branch_str = str(excitatory_branch_factors).replace(' ', '')
            
            f.write(
                f"{get_scalar(exp_config.model.network.type)},{scalar_seed},"
                f"{ee_synapses},{ie_synapses},{ei_synapses},"
                f"{excit_layers_str},{inhib_layers_str},{excit_branch_str},"
                f"{lr},"
                f"{reactivate_val},{topk_s},{blocklinear_s},{reactivation_s},{reactivation_t},"
                f"{topk_type_val},"
                f"{metrics}\n"
            )
    finally:
        if use_wandb and wandb.run is not None:
            wandb.finish()


# Helper function to safely extract single value from config
def get_scalar(value):
    if isinstance(value, (list, ListConfig)):
        if not value:
            raise ValueError("Expected a non-empty list or a scalar, got empty list.")
        if len(value) == 1:
            return value[0]
        else:
            raise ValueError(f"Expected scalar or single-item list in config, found list: {value}")
    return value


def main():
    if len(sys.argv) != 2:
        print("Usage: python sweep_ei_corr.py <path_to_task_config>")
        sys.exit(1)

    config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    wandb_cfg = getattr(config, "wandb", OmegaConf.create({}))

    try:
        base_save_path = str(Path(config_path).parent.parent)
    except Exception as e:
        print(f"Error determining base save path from config path '{config_path}': {e}")
        sys.exit(1)

    job_id = os.environ.get('SLURM_ARRAY_JOB_ID', 'local')
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    log_dir = os.path.join(base_save_path, "logs", str(job_id))
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"task_{task_id}.log")
    logger_manager.set_log_file(log_file_path)

    logger.info(f"--- Starting Task {task_id} (Job {job_id}) ---")
    logger.info(f"Task Config loaded from: {config_path}")
    logger.info(f"Base sweep results path: {base_save_path}")

    sweep_run_name_prefix = getattr(config.train, "run_name", Path(config_path).stem)
    sweep_identifier_from_path = Path(base_save_path).name
    wandb_group_name = f"{get_scalar(sweep_run_name_prefix)}_{sweep_identifier_from_path}_job{job_id}"

    results_file = os.path.join(base_save_path, "ei_corr_sweep_results.csv")

    completed_experiments = pd.DataFrame(
         columns=["model", "seed", "ee", "ie", "ei", "lr", "reactivate", "topk_strategy", "blocklinear_strategy", "reactivation_strategy", "reactivation_type", "topk_type", "train_acc", "valid_acc", "test_acc"]
    )
    if os.path.exists(results_file):
        logger.info(f"Checking existing results file: {results_file}")
        try:
             completed_experiments = pd.read_csv(results_file)
             if not all(col in completed_experiments.columns for col in ["model", "seed", "ee", "ie", "ei", "lr", "reactivate", "topk_strategy", "blocklinear_strategy", "reactivation_strategy", "reactivation_type", "topk_type"]):
                 logger.warning("Results CSV missing expected columns. Treating as empty.")
                 completed_experiments = completed_experiments[[]]
        except pd.errors.EmptyDataError:
             logger.warning(f"Results file {results_file} is empty.")
        except Exception as e:
             logger.error(f"Error loading results file {results_file}: {e}. Treating as empty.")
             completed_experiments = completed_experiments[[]]
    else:
        logger.info("No existing results file found. This is the first task or file was moved.")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)

    def experiment_completed(params_dict):
        if completed_experiments.empty:
            return False
        
        query_conditions = []
        for key, value in params_dict.items():
            if key not in completed_experiments.columns:
                logger.warning(f"Parameter '{key}' for completion check not found in CSV columns. Skipping this condition.")
                continue
            
            if completed_experiments[key].dtype == 'object' and not isinstance(value, str):
                query_conditions.append(f"`{key}` == '{str(value)}'")
            elif isinstance(value, bool):
                 query_conditions.append(f"`{key}` == {value}")
            elif isinstance(value, (int, float)):
                 query_conditions.append(f"`{key}` == {value}")
            else:
                 query_conditions.append(f"`{key}` == '{value}'")

        if not query_conditions:
            return False
            
        query_str = " & ".join(query_conditions)
        try:
            return len(completed_experiments.query(query_str)) > 0
        except Exception as e:
            logger.error(f"Error during experiment_completed query: {query_str} with error {e}")
            return False

    try:
        task_cfg = getattr(config, 'task', None)
        train_cfg = getattr(config, 'train', None)
        if task_cfg is None or train_cfg is None:
             raise ValueError("Config must contain 'task' and 'train' sections")
        train_ds, valid_ds, test_ds = get_unified_datasets(task_cfg, train_cfg)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    try:
        config.model.network.parameters.input_dim = train_ds[0][0].shape[-1]
        
        if not hasattr(config.train, 'learning_strategy'):
            config.train.learning_strategy = "mle"
            logger.warning("No learning_strategy found in train config, defaulting to 'mle'")
        config.model.network.parameters.learning_strategy = config.train.learning_strategy

    except Exception as e:
        logger.error(f"Error setting model dimensions/strategy from dataset: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    logger.info("Running single experiment based on loaded config...")

    try:
        model_type = get_scalar(config.model.network.type)
        seed = get_scalar(config.train.seed)
        ee_synapses = get_scalar(config.model.network.parameters.ee_synapses_per_branch_per_layer)
        ie_synapses = get_scalar(config.model.network.parameters.ie_synapses_per_branch_per_layer)
        ei_synapses = get_scalar(config.model.network.parameters.ei_synapses_per_branch_per_layer)
        excitatory_layer_sizes = config.model.network.parameters.excitatory_layer_sizes
        inhibitory_layer_sizes = config.model.network.parameters.inhibitory_layer_sizes
        excitatory_branch_factors = config.model.network.parameters.excitatory_branch_factors
        lr = get_scalar(config.train.lr)
        reactivate_val = get_scalar(getattr(config.model.network.parameters, "reactivate", False))
        topk_s = str(get_scalar(getattr(config.model.network.parameters, "topk_strategy", "none")))
        blocklinear_s = str(get_scalar(getattr(config.model.network.parameters, "blocklinear_strategy", "none")))
        reactivation_s = str(get_scalar(getattr(config.model.network.parameters, "reactivation_strategy", "none")))
        reactivation_t = get_scalar(getattr(config.model.network.parameters, "reactivation_type", "param_tanh"))
        topk_type_val = str(get_scalar(getattr(config.model.network.parameters, "topk_type", "standard")))

    except Exception as e:
        logger.error(f"Error extracting scalar parameters from config: {e}")
        logger.error(f"Config problematic section might be: {config}")
        logger.error(traceback.format_exc())
        sys.exit(1)

    run_save_path = os.path.join(base_save_path, "results", str(job_id), f"task_{task_id}")
    os.makedirs(run_save_path, exist_ok=True)
    logger.info(f"Saving task-specific artifacts (plots, model state) to: {run_save_path}")

    current_params_for_check = {
        "model": model_type,
        "seed": seed,
        "ee": ee_synapses,
        "ie": ie_synapses,
        "ei": ei_synapses,
        "excitatory_layer_sizes": excitatory_layer_sizes,
        "inhibitory_layer_sizes": inhibitory_layer_sizes,
        "excitatory_branch_factors": excitatory_branch_factors,
        "lr": lr,
        "reactivate": reactivate_val,
        "topk_strategy": topk_s,
        "blocklinear_strategy": blocklinear_s,
        "reactivation_strategy": reactivation_s,
        "reactivation_type": reactivation_t,
        "topk_type": topk_type_val,
    }
    if experiment_completed(current_params_for_check):
         param_log_str = ", ".join([f"{k}={v}" for k, v in current_params_for_check.items()])
         logger.info(
             f"Skipping Task {task_id}: {param_log_str} (already completed in {results_file})"
         )
         sys.exit(0)

    try:
        logger.info(f"Executing run_experiment for Task {task_id}...")
        run_experiment(
            config,
            train_ds,
            valid_ds,
            test_ds,
            run_save_path,
            results_file,
            ee_synapses,
            ie_synapses,
            ei_synapses,
            excitatory_layer_sizes,
            inhibitory_layer_sizes,
            excitatory_branch_factors,
            lr,
            reactivate_val,
            topk_s,
            blocklinear_s,
            reactivation_s,
            reactivation_t,
            topk_type_val,
            wandb_cfg,
            job_id,
            task_id,
            wandb_group_name,
        )
    except Exception as e:
        logger.error(
            f"--- Error during run_experiment for Task {task_id} ---"
            f"\nConfig: {config_path}"
            f"\nParams: Seed={seed}, EE={ee_synapses}, IE={ie_synapses}, EI={ei_synapses}, LR={lr}, Reactivate={reactivate_val}, Topk={topk_s}, Blocklinear={blocklinear_s}, Reactivation={reactivation_s}, Reactivation Type={reactivation_t}, Topk Type={topk_type_val}"
            f"\nError: {str(e)}"
        )
        logger.error(traceback.format_exc())
        sys.exit(1)

    logger.info(f"--- Task {task_id} completed successfully. Results appended to: {results_file} ---")


if __name__ == "__main__":
    main()