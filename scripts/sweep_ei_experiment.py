"""
sweep_ei_experiment.py
---------------------
Script that performs parameter sweeps over excitatory-excitatory and
inhibitory-excitatory synaptic connections while keeping other parameters constant.
Supports multiple random seeds per configuration and shunting/non-shunting variants.
"""
import itertools
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime

import pandas as pd
import torch

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.config import load_config
from dendritic_modeling.learning_strategies import (CustomWeightDecayOptimizer,
                                                    get_trainer)
from dendritic_modeling.models import Classifier, ProbabilisticClassifier
from dendritic_modeling.networks import (ExcitationInhibitionNetwork,
                                         MatchedParamMLP)
from dendritic_modeling.plot_utils import plot_ei_sweep_results
from dendritic_modeling.synthetic_datasets import get_unified_datasets
from dendritic_modeling.utils import evaluate_accuracy, save_dict, set_seed

torch.backends.cudnn.deterministic = True


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
    ee_synapses,
    ie_synapses,
    shunting,
):
    set_seed(exp_config.train.seed)
    # Initialize and train model
    model = initialize_model(exp_config.model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    results = train_and_evaluate(
        model, train_ds, valid_ds, test_ds, exp_config.train, run_save_path
    )

    # Log results
    logger.info(
        f"Train acc: {results['train_acc']:.3f}, "
        f"Valid acc: {results['valid_acc']:.3f}, "
        f"Test acc: {results['test_acc']:.3f}"
    )

    # Save results to CSV
    with open(results_file, "a") as f:
        f.write(
            f"{exp_config.model.network.type},{exp_config.train.seed},"
            f"{ee_synapses},{ie_synapses},{shunting},"
            f"{results['train_acc']:.4f},{results['valid_acc']:.4f},"
            f"{results['test_acc']:.4f}\n"
        )


def main():
    # Modified argument parsing
    if len(sys.argv) != 3:
        print(
            "Usage: python sweep_ei_experiment.py <path_to_config> <path_to_existing_run>"
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
        base_save_path = os.path.join(config.outputs.dir, f"sweep_ei_{now}")
        os.makedirs(base_save_path, exist_ok=True)

        # Save config
        logger.info("Saving config...")
        save_dict(asdict(config), base_save_path, "config.yaml")
        logger.info(f"Saved config to {base_save_path}/config.yaml")
    else:
        base_save_path = existing_run_path

    # Setup logging
    logger_manager.set_log_file(os.path.join(base_save_path, "sweep_ei.log"))
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
    results_file = os.path.join(base_save_path, "ei_sweep_results.csv")
    if os.path.exists(results_file):
        logger.info("Loading existing results file")
        completed_experiments = pd.read_csv(results_file)
    else:
        logger.info("Creating new results file")
        completed_experiments = pd.DataFrame(
            columns=[
                "model",
                "seed",
                "exc",
                "inh",
                "shunting",
                "train",
                "valid",
                "test",
            ]
        )
        completed_experiments.to_csv(results_file, index=False)

    def experiment_completed(model_type, seed, ee=None, ie=None, shunting=None):
        """Check if an experiment configuration has already been completed"""
        query = (completed_experiments["model"] == model_type) & (
            completed_experiments["seed"] == seed
        )

        if model_type == "EINet":
            query = (
                query
                & (completed_experiments["exc"] == ee)
                & (completed_experiments["inh"] == ie)
                & (completed_experiments["shunting"] == shunting)
            )

        return len(completed_experiments[query]) > 0

    # Run MLP experiments with each seed
    logger.info("\nRunning/Checking MLP baseline experiments...")
    seeds = config.train.seed

    for seed in seeds:
        if experiment_completed("MLP", seed):
            logger.info(f"Skipping MLP seed {seed} (already completed)")
            continue

        logger.info(f"Training MLP with seed {seed}")

        # Create MLP-specific directory
        run_save_path = os.path.join(base_save_path, "sweeps", "mlp", f"seed{seed}")
        os.makedirs(run_save_path, exist_ok=True)

        # Create MLP config
        exp_config = deepcopy(config)
        exp_config.model.network.type = "MLP"
        exp_config.train.seed = seed

        # Take first configuration from the lists for MLP
        exp_config.model.network.parameters.ee_synapses_per_branch_per_layer = (
            exp_config.model.network.parameters.ee_synapses_per_branch_per_layer[-1]
        )
        exp_config.model.network.parameters.ie_synapses_per_branch_per_layer = (
            exp_config.model.network.parameters.ie_synapses_per_branch_per_layer[-1]
        )

        try:
            run_experiment(
                exp_config,
                train_ds,
                valid_ds,
                test_ds,
                run_save_path,
                results_file,
                None,
                None,
                None,
            )
        except Exception as e:
            logger.error(f"Error in MLP training with seed {seed}: {str(e)}")
            continue

    logger.info("MLP baseline experiments completed.")

    # Get parameter combinations for EINet
    ee_configs = config.model.network.parameters.ee_synapses_per_branch_per_layer
    ie_configs = config.model.network.parameters.ie_synapses_per_branch_per_layer
    shunting_configs = [True, False]

    # Count total remaining experiments
    total_remaining = 0
    for ee_config, ie_config in itertools.product(ee_configs, ie_configs):
        for seed in seeds:
            for use_shunting in shunting_configs:
                if not experiment_completed(
                    "EINet", seed, ee_config[0], ie_config[0], use_shunting
                ):
                    total_remaining += 1

    logger.info(f"\nRemaining experiments to run: {total_remaining}")
    current = 0

    # Run EINet experiments
    for ee_config, ie_config in itertools.product(ee_configs, ie_configs):
        # Create base experiment directory for this EE/IE combination
        exp_name = (
            f"exc{'_'.join(map(str, ee_config))}_inh{'_'.join(map(str, ie_config))}"
        )
        exp_base_path = os.path.join(base_save_path, "sweeps", exp_name)
        os.makedirs(exp_base_path, exist_ok=True)

        for seed in seeds:
            # Create seed-specific directory
            seed_path = os.path.join(exp_base_path, f"seed{seed}")
            os.makedirs(seed_path, exist_ok=True)

            for use_shunting in shunting_configs:
                if experiment_completed(
                    "EINet", seed, ee_config[0], ie_config[0], use_shunting
                ):
                    logger.info(
                        f"Skipping EINet EE: {ee_config[0]}, IE: {ie_config[0]}, "
                        f"Seed: {seed}, Shunting: {use_shunting} (already completed)"
                    )
                    continue

                current += 1
                logger.info(
                    f"\nRunning combination [{current}/{total_remaining}]"
                    f" EE: {ee_config[0]}, IE: {ie_config[0]}, "
                    f"Seed: {seed}, Shunting: {use_shunting}"
                )

                # Create shunting-specific directory
                shunting_dir = "shunting_true" if use_shunting else "shunting_false"
                run_save_path = os.path.join(seed_path, shunting_dir)
                os.makedirs(run_save_path, exist_ok=True)

                # Update config with current parameter combination
                exp_config = deepcopy(config)
                exp_config.model.network.parameters.ee_synapses_per_branch_per_layer = (
                    ee_config
                )
                exp_config.model.network.parameters.ie_synapses_per_branch_per_layer = (
                    ie_config
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
                        ee_config[0],
                        ie_config[0],
                        use_shunting,
                    )
                except Exception as e:
                    logger.error(
                        f"Error in combination {exp_name} seed {seed} "
                        f"shunting {use_shunting}: {str(e)}"
                    )
                    continue

    logger.info("\nParameter sweep completed. Results saved to: " + results_file)

    logger.info("Plotting results...")
    plot_ei_sweep_results(
        results_csv_path=results_file,
        split="test",
        ei_ratio_range=(0, 30),
    )
    logger.info("  done...")


if __name__ == "__main__":
    main()
