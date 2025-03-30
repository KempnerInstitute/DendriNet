"""
sweep_ei_experiment.py
---------------------
Script that performs parameter sweeps over excitatory-excitatory and
inhibitory-excitatory synaptic connections while keeping other parameters constant.
Supports multiple random seeds per configuration and shunting/non-shunting variants.
"""
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime

import pandas as pd
import torch

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.config import load_config
from dendritic_modeling.learning_strategies import (
    CustomWeightDecayOptimizer,
    get_trainer,
)
from dendritic_modeling.models import Classifier, ProbabilisticClassifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork, MatchedParamMLP
from dendritic_modeling.plot_utils import plot_branch_sweep_results
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.utils import evaluate_accuracy, save_dict, set_seed


def initialize_model(model_cfg):
    """
    Builds the specified model: 'classification' uses ProbabilisticClassifier or Classifier.
    model_cfg is from config.model, with .task, .probabilistic, .network, etc.
    """
    task = model_cfg.task
    probabilistic = model_cfg.probabilistic
    net_type = model_cfg.network.type
    net_params = model_cfg.network.parameters
    enable_branch_outputs = net_params.pop("enable_branch_outputs", False)
    learning_strategy = net_params.pop("learning_strategy", "mle")

    if task == "classification":
        if net_type == "MLP":
            net = MatchedParamMLP(**net_params)
            output_dim = net_params.get("output_dim", 10)
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim)
            else:
                return Classifier(net)
        elif net_type == "EINet":
            net = ExcitationInhibitionNetwork(
                enable_branch_outputs=enable_branch_outputs,
                learning_strategy=learning_strategy,
                **net_params,
            )

            output_dim = net_params["excitatory_layer_sizes"][-1]
            if probabilistic:
                return ProbabilisticClassifier(net, output_dim)
            else:
                return Classifier(net)
        else:
            raise ValueError(f"Invalid network type: {net_type}")
    else:
        raise ValueError(f"Invalid task: {task}")


def train_and_evaluate(model, train_ds, valid_ds, test_ds, train_cfg, run_save_path):
    """Train model and evaluate performance."""
    optimizer = CustomWeightDecayOptimizer(
        model=model,
        optimizer=torch.optim.Adam(model.parameters(), lr=train_cfg.lr),
        weight_decay=train_cfg.weight_decay_rate,
    )

    trainer = get_trainer(train_cfg.learning_strategy, optimizer, suppress_prints=True)

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
        early_stopping=True,
        patience=100,
        plot_losses=True,
        save_path=run_save_path,
    )

    # Evaluate performance
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
    branch_factors,
    shunting,
):
    set_seed(exp_config.train.seed)
    # Initialize and train model
    model = initialize_model(exp_config.model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    results = train_and_evaluate(
        model, train_ds, valid_ds, test_ds, exp_config.train, run_save_path
    )

    # Calculate total number of parameters
    n_params = sum(p.numel() for p in model.parameters())

    # Log results
    logger.info(
        f"Train acc: {results['train_acc']:.3f}, "
        f"Valid acc: {results['valid_acc']:.3f}, "
        f"Test acc: {results['test_acc']:.3f}, "
        f"Total params: {n_params}"
    )

    # Save results to CSV
    branch_factors_str = "-".join(map(str, branch_factors))
    with open(results_file, "a") as f:
        f.write(
            f"{exp_config.model.network.type},{exp_config.train.seed},"
            f"{branch_factors_str},{shunting},{n_params},"
            f"{results['train_acc']:.4f},{results['valid_acc']:.4f},"
            f"{results['test_acc']:.4f}\n"
        )


def main():
    # Modified argument parsing
    if len(sys.argv) != 3:
        print(
            "Usage: python sweep_branch_experiment.py <path_to_config> <path_to_existing_run>"
        )
        print("Use 'None' for either argument if not provided")
        sys.exit(1)

    config_path = None if sys.argv[1].lower() == "none" else sys.argv[1]
    existing_run_path = None if sys.argv[2].lower() == "none" else sys.argv[2]

    # Load or create config
    if config_path is None:
        if existing_run_path is None:
            raise ValueError("Must provide either config path or existing run path")
        config_path = os.path.join(existing_run_path, "config.yaml")
    config = load_config(config_path)

    # Set up run directory
    if existing_run_path is None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_save_path = os.path.join(config.outputs.dir, f"sweep_branch_{now}")
        os.makedirs(base_save_path, exist_ok=True)

        # Save config
        logger.info("Saving config...")
        save_dict(asdict(config), base_save_path, "config.yaml")
        logger.info(f"Saved config to {base_save_path}/config.yaml")
    else:
        base_save_path = existing_run_path

    # Setup logging
    logger_manager.set_log_file(os.path.join(base_save_path, "sweep_branch.log"))
    logger.info(f"Starting/Resuming parameter sweep experiment")
    logger.info(f"Config loaded from: {config_path}")
    logger.info(f"Results will be saved to: {base_save_path}")

    # Load datasets
    train_ds, valid_ds, test_ds = get_unified_datasets(config.task, config.train)

    config.model.network.parameters.input_dim = train_ds[0][0].shape[-1]
    config.model.network.parameters.output_dim = torch.unique(train_ds[:][1]).shape[-1]
    config.model.network.parameters.learning_strategy = getattr(
        config.train, "learning_strategy", "mle"
    )

    # Initialize or load results file
    results_file = os.path.join(base_save_path, "branch_sweep_results.csv")
    if os.path.exists(results_file):
        logger.info("Loading existing results file")
        completed_experiments = pd.read_csv(results_file)
    else:
        logger.info("Creating new results file")
        completed_experiments = pd.DataFrame(
            columns=[
                "model",
                "seed",
                "branch_factors",
                "shunting",
                "nparams",
                "train",
                "valid",
                "test",
            ]
        )
        completed_experiments.to_csv(results_file, index=False)

    def experiment_completed(model_type, seed, branch_factors, shunting=None):
        """Check if an experiment configuration has already been completed"""
        branch_factors_str = "-".join(map(str, branch_factors))
        query = (
            (completed_experiments["model"] == model_type)
            & (completed_experiments["seed"] == seed)
            & (completed_experiments["branch_factors"] == branch_factors_str)
        )

        if model_type == "EINet":
            query = query & (completed_experiments["shunting"] == shunting)

        return len(completed_experiments[query]) > 0

    # Get branch factor configurations and seeds
    branch_configs = config.model.network.parameters.excitatory_branch_factors
    shunting_configs = [True, False]
    seeds = config.train.seed

    # Count total remaining experiments
    total_remaining = 0
    for branch_config in branch_configs:
        for seed in seeds:
            # Count MLP experiment if not done
            if not experiment_completed("MLP", seed, branch_config):
                total_remaining += 1
            # Count EINet experiments if not done
            for use_shunting in shunting_configs:
                if not experiment_completed("EINet", seed, branch_config, use_shunting):
                    total_remaining += 1

    logger.info(f"Remaining experiments to run: {total_remaining}")
    current = 0

    # Run experiments
    for branch_config in branch_configs:
        # Create base experiment directory for this branch configuration
        exp_name = f"branching_{'_'.join(map(str, branch_config))}"
        exp_base_path = os.path.join(base_save_path, "sweeps", exp_name)
        os.makedirs(exp_base_path, exist_ok=True)

        for seed in seeds:
            # Create seed-specific directory
            seed_path = os.path.join(exp_base_path, f"seed{seed}")
            os.makedirs(seed_path, exist_ok=True)

            # Run MLP experiment for this branch factor and seed
            if experiment_completed("MLP", seed, branch_config):
                logger.info(
                    f"Skipping MLP Branch Factors: {branch_config}, "
                    f"Seed: {seed} (already completed)"
                )
            else:
                current += 1
                logger.info(
                    f"Running combination [{current}/{total_remaining}]"
                    f" MLP, Branch Factors: {branch_config}, Seed: {seed}"
                )

                run_save_path = os.path.join(seed_path, "mlp")
                os.makedirs(run_save_path, exist_ok=True)

                exp_config = deepcopy(config)
                exp_config.model.network.type = "MLP"
                exp_config.model.network.parameters.excitatory_branch_factors = (
                    branch_config
                )
                exp_config.train.seed = seed

                try:
                    run_experiment(
                        exp_config,
                        train_ds,
                        valid_ds,
                        test_ds,
                        run_save_path,
                        results_file,
                        branch_config,
                        None,
                    )
                except Exception as e:
                    logger.error(f"Error in MLP training with seed {seed}: {str(e)}")
                    continue

            # Run EINet experiments
            for use_shunting in shunting_configs:
                if experiment_completed("EINet", seed, branch_config, use_shunting):
                    logger.info(
                        f"Skipping EINet Branch Factors: {branch_config}, "
                        f"Seed: {seed}, Shunting: {use_shunting} (already completed)"
                    )
                    continue

                current += 1
                logger.info(
                    f"Running combination [{current}/{total_remaining}]"
                    f" EINet, Branch Factors: {branch_config}, "
                    f"Seed: {seed}, Shunting: {use_shunting}"
                )

                # Create shunting-specific directory
                shunting_dir = "shunting_true" if use_shunting else "shunting_false"
                run_save_path = os.path.join(seed_path, shunting_dir)
                os.makedirs(run_save_path, exist_ok=True)

                # Update config with current parameter combination
                exp_config = deepcopy(config)
                exp_config.model.network.parameters.excitatory_branch_factors = (
                    branch_config
                )
                exp_config.model.network.parameters.use_shunting = use_shunting
                exp_config.model.network.parameters.reactivation_strategy = (
                    "inverse" if use_shunting else "none"
                )
                exp_config.train.seed = seed

                try:
                    run_experiment(
                        exp_config,
                        train_ds,
                        valid_ds,
                        test_ds,
                        run_save_path,
                        results_file,
                        branch_config,
                        use_shunting,
                    )
                except Exception as e:
                    logger.error(
                        f"Error in combination {exp_name} seed {seed} "
                        f"shunting {use_shunting}: {str(e)}"
                    )
                    continue

    logger.info("Parameter sweep completed. Results saved to:\n" + results_file)

    logger.info("Plotting results...")
    plot_branch_sweep_results(results_file, split="test")
    logger.info("  done...")


if __name__ == "__main__":
    main()
