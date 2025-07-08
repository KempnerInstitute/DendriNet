#!/usr/bin/env python
"""
Sweep script for noise experiments with EI networks.

This script performs parameter sweeps over noise levels and other hyperparameters
for EI networks on synthetic datasets.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from copy import deepcopy
from typing import Dict

import pandas as pd
import torch
from torch.utils.data import TensorDataset

from dendritic_modeling import logger, logger_manager
from dendritic_modeling.config import load_config
from dendritic_modeling.models import Classifier, ProbabilisticClassifier
from dendritic_modeling.networks import ExcitationInhibitionNetwork, MatchedParamMLP
from dendritic_modeling.plotting.utils import (
    plot_activation_tuning,
    plot_voltage_before_activation,
    split_MNIST_inputs,
)
from dendritic_modeling.datasets import get_unified_datasets
from dendritic_modeling.training.utils.evaluation import evaluate_accuracy
from dendritic_modeling.datasets.augmentation.noise import (
    add_frequency_noise,
    add_gaussian_noise,
    add_poisson_noise,
    add_uniform_noise,
)
from dendritic_modeling.utils import (
    get_data_directory,
    save_dict,
    set_seed,
)


def initialize_model(model_cfg):
    """
    Builds the specified model: 'classification' uses ProbabilisticClassifier or Classifier.
    model_cfg is from config.model, with .task, .probabilistic, .network, etc.
    """
    task = model_cfg.task
    probabilistic = model_cfg.probabilistic
    net_type = model_cfg.network.type
    net_params = model_cfg.network.parameters
    # Branch outputs are now automatically enabled when needed for analysis
    enable_branch_outputs = False
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


def evaluate_noise_robustness(
    model: torch.nn.Module,
    test_ds: torch.utils.data.Dataset,
    n_samples: int,
    gaussian_stds: list,
    uniform_spreads: list,
    poisson_rates: list,
    freq_stds: list,
    n_channels: int = 1,
    image_shape: tuple = (28, 28),
) -> Dict[str, torch.Tensor]:
    """
    Evaluate model performance under different noise conditions and levels.

    Args:
        model: Model to evaluate
        test_ds: Test dataset
        n_samples: Number of noise samples per level
        gaussian_stds: List of standard deviations for Gaussian noise
        uniform_spreads: List of spreads for uniform noise
        poisson_rates: List of rate parameters for Poisson noise
        freq_stds: List of standard deviations for frequency domain noise
        n_channels: Number of image channels
        image_shape: Shape of input images

    Returns:
        Dict containing performance drops for each noise type and level
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    x, y = test_ds[:]
    x = x.to(device)
    y = y.to(device)
    y = y[None, ...].expand(n_samples, -1)

    noise_results = {}

    # Gaussian noise
    for std in gaussian_stds:
        noisy_x = add_gaussian_noise(x, std=std, n_samples=n_samples)
        accs = evaluate_accuracy(
            classifier=model,
            test_ds=TensorDataset(noisy_x, y),
            reduce_dim=1,
            move_device=False,
        )[-1]
        acc_diffs = accs.cpu()
        noise_results[f"gaussian_{std}"] = acc_diffs

    # Uniform noise
    for spread in uniform_spreads:
        noisy_x = add_uniform_noise(x, spread=spread, n_samples=n_samples)
        accs = evaluate_accuracy(
            classifier=model,
            test_ds=TensorDataset(noisy_x, y),
            reduce_dim=1,
            move_device=False,
        )[-1]
        acc_diffs = accs.cpu()
        noise_results[f"uniform_{spread}"] = acc_diffs

    # Poisson noise
    for rate in poisson_rates:
        noisy_x = add_poisson_noise(x, rate=rate, n_samples=n_samples)
        accs = evaluate_accuracy(
            classifier=model,
            test_ds=TensorDataset(noisy_x, y),
            reduce_dim=1,
            move_device=False,
        )[-1]
        acc_diffs = accs.cpu()
        noise_results[f"poisson_{rate}"] = acc_diffs

    # Frequency noise
    for std in freq_stds:
        noisy_x = add_frequency_noise(
            x,
            std=std,
            n_samples=n_samples,
            n_channels=n_channels,
            image_shape=image_shape,
        )
        accs = evaluate_accuracy(
            classifier=model,
            test_ds=TensorDataset(noisy_x, y),
            reduce_dim=1,
            move_device=False,
        )[-1]
        acc_diffs = accs.cpu()
        noise_results[f"frequency_{std}"] = acc_diffs

    return noise_results


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python sweep_noise_experiment.py <results_dir> <n_samples> "
            "[gaussian_stds] [uniform_spreads] [poisson_rates] [freq_stds]"
        )
        sys.exit(1)

    # Parse arguments
    results_dir = sys.argv[1]
    n_samples = int(sys.argv[2])

    # Default noise levels if not provided
    default_gaussian = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
    default_uniform = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    default_poisson = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
    default_freq = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    # Parse noise levels from command line or use defaults
    gaussian_stds = (
        [float(x) for x in sys.argv[3].split(",")]
        if len(sys.argv) > 3
        else default_gaussian
    )
    uniform_spreads = (
        [float(x) for x in sys.argv[4].split(",")]
        if len(sys.argv) > 4
        else default_uniform
    )
    poisson_rates = (
        [float(x) for x in sys.argv[5].split(",")]
        if len(sys.argv) > 5
        else default_poisson
    )
    freq_stds = (
        [float(x) for x in sys.argv[6].split(",")]
        if len(sys.argv) > 6
        else default_freq
    )

    logger.info("Noise levels:")
    logger.info(f"  Gaussian stds: {gaussian_stds}")
    logger.info(f"  Uniform spreads: {uniform_spreads}")
    logger.info(f"  Poisson rates: {poisson_rates}")
    logger.info(f"  Frequency stds: {freq_stds}")

    noise_keys = []
    for std in gaussian_stds:
        noise_keys.append(f"gaussian_{std}")
    for spread in uniform_spreads:
        noise_keys.append(f"uniform_{spread}")
    for rate in poisson_rates:
        noise_keys.append(f"poisson_{rate}")
    for std in freq_stds:
        noise_keys.append(f"frequency_{std}")

    # Load config and results
    config_path = os.path.join(results_dir, "config.yaml")

    logger.info(f"Loading config from: {config_path}")
    config = load_config(config_path)

    # Determine sweep type from config
    is_branch_sweep = (
        len(config.model.network.parameters.ee_synapses_per_branch_per_layer) == 1
    )
    sweep_type = "branch" if is_branch_sweep else "ei"
    logger.info(f"Detected {sweep_type} sweep experiment")

    # Load results based on sweep type
    results_file = (
        "branch_sweep_results.csv" if is_branch_sweep else "ei_sweep_results.csv"
    )
    results_path = os.path.join(results_dir, results_file)

    logger.info(f"Loading results from: {results_path}")
    results_df = pd.read_csv(results_path)

    # Load dataset
    logger.info("Loading test dataset...")
    _, _, test_ds = get_unified_datasets(config.task, config.train)

    config.model.network.parameters.input_dim = test_ds[0][0].shape[-1]
    config.model.network.parameters.output_dim = torch.unique(test_ds[:][1]).shape[-1]
    config.model.network.parameters.learning_strategy = getattr(
        config.train, "learning_strategy", "mle"
    )

    if config.task.dataset in ["mnist", "mnist_modulo10", "mnist_switch2"]:
        n_channels = 1
        image_shape = (28, 28)
    elif config.task.dataset == "mnist_switch10":
        n_channels = 1
        image_shape = (28, 28 * 2)
    elif config.task.dataset in ["cifar10"]:
        n_channels = 3
        image_shape = (32, 32)
    else:
        raise ValueError(f"Invalid dataset: {config.task.dataset}")

    # Initialize results storage
    noise_results = []

    # Group results differently based on sweep type
    if is_branch_sweep:
        results_df = results_df.fillna({"shunting": "none"})
        configs = results_df.groupby(["model", "branch_factors", "shunting"])

        nparams_by_group = {}
        for (model, branch_factors, shunting), group in configs:
            nparams_by_group[(model, branch_factors, shunting)] = group["nparams"].iloc[
                0
            ]
    else:  # ei sweep
        results_df = results_df.fillna(
            {"exc": "none", "inh": "none", "shunting": "none"}
        )
        configs = results_df.groupby(["model", "exc", "inh", "shunting"])

    total_configs = len(configs)
    logger.info(f"Found {total_configs} unique model configurations")

    # Iterate over configurations
    for config_idx, (params, group) in enumerate(configs, 1):
        if is_branch_sweep:
            model_type, branch_factors_str, use_shunting = params
            logger.info(
                f"Evaluating configuration [{config_idx}/{total_configs}] "
                f"Model: {model_type} "
                f"Branch factors: {branch_factors_str} "
                f"Shunting: {use_shunting}"
            )

            branch_factors = [int(x) for x in branch_factors_str.split("-")]
            use_shunting = "none" if use_shunting == "none" else bool(use_shunting)

        else:  # ei sweep
            model_type, ee_syns, ie_syns, use_shunting = params
            logger.info(
                f"Evaluating configuration [{config_idx}/{total_configs}] "
                f"Model: {model_type} "
                f"EE synapses: {ee_syns} "
                f"IE synapses: {ie_syns} "
                f"Shunting: {use_shunting}"
            )

            ee_syns = 1 if ee_syns == "none" else int(ee_syns)
            ie_syns = 1 if ie_syns == "none" else int(ie_syns)
            use_shunting = "none" if use_shunting == "none" else bool(use_shunting)

        # Collect results across seeds
        seed_results = {}
        for noise_key in noise_keys:
            seed_results[noise_key] = []

        # Test each seed
        for seed in config.train.seed:
            set_seed(seed)

            # Initialize model with this configuration
            exp_config = deepcopy(config)
            exp_config.train.seed = seed
            exp_config.model.network.type = model_type
            exp_config.model.network.parameters.use_shunting = (
                False if use_shunting == "none" else use_shunting
            )

            if is_branch_sweep:
                exp_config.model.network.parameters.excitatory_branch_factors = (
                    branch_factors
                )
            else:
                exp_config.model.network.parameters.ee_synapses_per_branch_per_layer = [
                    ee_syns
                ]
                exp_config.model.network.parameters.ie_synapses_per_branch_per_layer = [
                    ie_syns
                ]

            # Initialize and load model
            model = initialize_model(exp_config.model)

            if model_type == "EINet":
                if is_branch_sweep:
                    model_path = os.path.join(
                        results_dir,
                        "sweeps",
                        f'branching_{"_".join(map(str, branch_factors))}',
                        f"seed{seed}",
                        f"shunting_{str(use_shunting).lower()}",
                        "best_model.pt",
                    )
                else:  # ei sweep
                    model_path = os.path.join(
                        results_dir,
                        "sweeps",
                        f"exc{ee_syns}_inh{ie_syns}",
                        f"seed{seed}",
                        f"shunting_{str(use_shunting).lower()}",
                        "best_model.pt",
                    )
            else:  # MLP
                if is_branch_sweep:
                    model_path = os.path.join(
                        results_dir,
                        "sweeps",
                        f'branching_{"_".join(map(str, branch_factors))}',
                        f"seed{seed}",
                        f"mlp",
                        "best_model.pt",
                    )
                else:
                    model_path = os.path.join(
                        results_dir, "sweeps", "mlp", f"seed{seed}", "best_model.pt"
                    )

            model.load_state_dict(torch.load(model_path))
            model.eval()

            # Evaluate noise robustness
            noise_results_seed = evaluate_noise_robustness(
                model,
                test_ds,
                n_samples,
                gaussian_stds,
                uniform_spreads,
                poisson_rates,
                freq_stds,
                n_channels,
                image_shape,
            )

            # Store results for this seed
            for noise_key, results in noise_results_seed.items():
                seed_results[noise_key].append(results)

        # Compute statistics across seeds
        stats = {
            "model": model_type,
            "shunting": use_shunting if model_type == "EINet" else None,
        }

        if is_branch_sweep:
            stats["branch_factors"] = "-".join(map(str, branch_factors))
            stats["nparams"] = nparams_by_group[
                (model_type, "-".join(map(str, branch_factors)), use_shunting)
            ]
        else:
            stats["exc"] = ee_syns
            stats["inh"] = ie_syns

        stats["test_mean"] = group["test"].mean()
        stats["test_std"] = group["test"].std()

        for noise_key in seed_results:
            # Stack results across seeds
            all_results = torch.stack(seed_results[noise_key])

            # Compute mean and std of performance drop
            mean_drop = all_results.mean().item()
            std_drop = all_results.std().item()

            stats[f"{noise_key}_mean"] = mean_drop
            stats[f"{noise_key}_std"] = std_drop

            logger.info(f"    {noise_key}: {mean_drop:.3f} ± {std_drop:.3f}")

        noise_results.append(stats)

    # Save noise analysis results
    output_dir = os.path.join(results_dir, "perturbation_analysis")
    output_path = os.path.join(output_dir, "noise_robustness_results.csv")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving results to:\n{output_path}")
    noise_df = pd.DataFrame(noise_results)
    noise_df.to_csv(output_path, index=False)

    logger.info("Plotting results...")
    if is_branch_sweep:
        for noise_type in ["gaussian", "uniform", "poisson", "frequency"]:
            plot_noise_sweep_results(
                results_csv_path=output_path,
                noise_type=noise_type,
            )
    else:
        for noise_type in ["gaussian", "uniform", "poisson", "frequency"]:
            plot_noise_sweep_results(
                results_csv_path=output_path,
                noise_type=noise_type,
            )
    logger.info("  done...")


if __name__ == "__main__":
    main()
