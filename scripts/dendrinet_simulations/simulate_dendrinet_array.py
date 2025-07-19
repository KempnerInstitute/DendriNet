#!/usr/bin/env python3
"""
simulate_dendrinet_array.py

Array job script for dendrinet simulation that internally handles:
- Correlation combinations (cEE, cII, cEI)
- Information group assignments (E_group, I_group)
- Seeds

This reduces the number of array jobs needed while still exploring the full parameter space.
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Dendrinet simulation script for array jobs"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--output-dir", required=True, help="Directory to save output")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


def setup_logging(output_dir, debug=False):
    """Set up logging to file and console"""
    log_level = logging.DEBUG if debug else logging.INFO
    log_file = os.path.join(output_dir, "simulation.log")

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def run_single_simulation(config, logger):
    """
    Run a simulation with the given parameters and return results.
    In a real implementation, this would call the actual simulation code.
    """
    # Create dummy results (in a real implementation, this would run the actual simulation)
    model_types = config.get("model_types", ["shunting", "linear"])

    # Simulate doing some work
    if logger.level <= logging.DEBUG:
        logger.debug(f"Running simulation with parameters: {config}")
    time.sleep(0.5)  # Reduced for testing

    results = []
    for model_type in model_types:
        result = {
            "model_type": model_type,
            "accuracy": np.random.uniform(0.5, 1.0),
            "final_mi": np.random.uniform(0.1, 0.5),
            "syn_ee": np.random.uniform(-0.1, 0.3),
            "syn_ii": np.random.uniform(-0.1, 0.3),
            "syn_ei": np.random.uniform(-0.1, 0.3),
            "red_ee": np.random.uniform(-0.1, 0.3),
            "red_ii": np.random.uniform(-0.1, 0.3),
            "red_ei": np.random.uniform(-0.1, 0.3),
        }

        # Add config parameters to result
        for key, value in config.items():
            if key not in result and not isinstance(value, (list, dict)):
                result[key] = value

        results.append(result)

    return results


def main():
    start_time = time.time()
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging
    logger = setup_logging(args.output_dir, args.debug)
    logger.info(f"Starting simulation job at {datetime.now()}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load config file
    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("Configuration loaded successfully")

    # Extract the parameters that we'll loop over internally
    seeds = config.pop("seeds", [42])
    correlation_combos = config.pop("correlation_combos", [[0.0, 0.0, 0.0]])
    group_combos = config.pop("group_combos", [[0, 0]])

    logger.info(
        f"Will process {len(seeds)} seeds, {len(correlation_combos)} correlation combinations, "
        f"and {len(group_combos)} group combinations"
    )
    logger.info(
        f"Total combinations to process: {len(seeds) * len(correlation_combos) * len(group_combos)}"
    )

    # Initialize list to store all results
    all_results = []

    # Create progress bar for all combinations
    total_combinations = len(seeds) * len(correlation_combos) * len(group_combos)
    progress_bar = tqdm(total=total_combinations, desc="Processing combinations")

    # Process all combinations
    for seed in seeds:
        for corr_combo in correlation_combos:
            cEE, cII, cEI = corr_combo

            for group_combo in group_combos:
                E_group, I_group = group_combo

                # Create a config for this specific combination
                run_config = config.copy()
                run_config.update(
                    {
                        "cEE": cEE,
                        "cII": cII,
                        "cEI": cEI,
                        "E_group": E_group,
                        "I_group": I_group,
                        "seed": seed,
                    }
                )

                # Run simulation for this combination
                try:
                    results = run_single_simulation(run_config, logger)
                    all_results.extend(results)
                    progress_bar.update(1)
                except Exception as e:
                    logger.error(f"Error running simulation: {e!s}")
                    progress_bar.update(1)
                    continue

    progress_bar.close()

    # Save all results to a single JSON file
    results_file = os.path.join(args.output_dir, "simulation_results.json")
    logger.info(f"Saving {len(all_results)} result entries to {results_file}")
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    # Also save as CSV for easier analysis
    results_df = pd.DataFrame(all_results)
    csv_file = os.path.join(args.output_dir, "simulation_results.csv")
    results_df.to_csv(csv_file, index=False)

    # Log completion
    end_time = time.time()
    duration = end_time - start_time
    logger.info(f"Simulation job completed at {datetime.now()}")
    logger.info(f"Total duration: {duration:.2f} seconds")
    logger.info(f"Processed {len(all_results)} total result entries")

    return 0


if __name__ == "__main__":
    main()
