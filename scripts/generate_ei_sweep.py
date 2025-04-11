#!/usr/bin/env python
"""
generate_ei_sweep.py
------------------------
Custom script to generate sweep configs for the EI sweep with the specific folder structure:
- Creates a unique 'ei_TIMESTAMP' directory for each sweep run
- Sets up subdirectories for configs, results, and plots
- Prepares the SLURM job configuration to organize outputs correctly

Usage:
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml
"""

import argparse
import datetime
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path

from omegaconf import OmegaConf


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate configurations for EI sweep with custom directory structure"
    )
    parser.add_argument("config_file", help="Path to the sweep configuration file")
    return parser.parse_args()


def generate_sweep_configs_manually(config, output_dir, config_filename_init):
    """
    Generate sweep configs manually without relying on generate_sweep_configs.py
    This ensures configs are created in the correct location
    """
    sweep_params = config.sweep_config
    experiment_settings = config.get("experiment_settings", {})
    seeds_per_condition = experiment_settings.get("seeds_per_condition", 1)
    base_seed = experiment_settings.get("base_seed", 42)

    # Extract parameter names and values
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[key] for key in param_names]

    # Generate all combinations
    configs = []
    config_idx = 0

    # Helper function to set nested values
    def set_nested_value(d, path, value):
        keys = path.split(".")
        for key in keys[:-1]:
            if key not in d:
                d[key] = {}
            d = d[key]
        d[keys[-1]] = value

    # Simple product function for all combinations
    def product(*iterables):
        if not iterables:
            yield ()
            return
        for item in iterables[0]:
            for rest in product(*iterables[1:]):
                yield (item,) + rest

    # Generate all parameter combinations
    for params in product(*param_values):
        # For each parameter combination, create configs with different seeds
        for seed_idx in range(seeds_per_condition):
            # Create a new config based on the base config
            new_config = OmegaConf.create(
                OmegaConf.to_container(config.train_config, resolve=True)
            )

            # Set the seed
            seed = base_seed + seed_idx
            new_config.train.seed = seed

            # Set the parameter values
            for i, param_name in enumerate(param_names):
                set_nested_value(new_config, param_name, params[i])

            # Make sure outputs.dir points to the correct directory
            if "outputs" not in new_config:
                new_config.outputs = {}
            new_config.outputs.dir = os.path.join(output_dir, "results")

            # Save the config
            config_path = os.path.join(
                output_dir, "configs", f"{config_filename_init}_{config_idx}.yaml"
            )
            OmegaConf.save(new_config, config_path)
            print(f"Generated config: {config_path}")

            config_idx += 1

    return config_idx  # Return the number of configs generated


def generate_custom_run_script(
    config, output_dir, num_configs, main_script_path="./run.sh"
):
    """Generate a custom run.sh script that properly uses the folder structure"""

    slurm_config = config.slurm_config

    array_job_flag = num_configs > 1

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm_config.run_name}",
        f"#SBATCH --account={slurm_config.account}",
    ]

    # Output paths use a single main job folder with task subfolders
    if array_job_flag:
        lines.extend(
            [
                f"#SBATCH --output={output_dir}/results/%A/task_%a/output.log",
                f"#SBATCH --error={output_dir}/results/%A/task_%a/error.log",
            ]
        )
    else:
        lines.extend(
            [
                f"#SBATCH --output={output_dir}/results/%j/output.log",
                f"#SBATCH --error={output_dir}/results/%j/error.log",
            ]
        )

    # Format time to HH:MM:SS if it's not already
    time_value = slurm_config.time
    if isinstance(time_value, (int, float)) or (
        isinstance(time_value, str) and time_value.isdigit()
    ):
        # Convert seconds to HH:MM:SS
        seconds = int(float(time_value))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        formatted_time = f"{hours}:{minutes:02d}:{seconds:02d}"
    elif isinstance(time_value, str):
        if ":" not in time_value:
            # Assume it's hours if no colons
            formatted_time = f"{time_value}:00:00"
        else:
            formatted_time = time_value
    else:
        # Default to 3 hours if unknown format
        formatted_time = "3:00:00"

    # Add the other SLURM parameters
    lines.extend(
        [
            f"#SBATCH --nodes={slurm_config.nodes}",
            f"#SBATCH --ntasks-per-node={slurm_config.ntasks_per_node}",
            f"#SBATCH --gpus-per-node={slurm_config.gpus_per_node}",
            f"#SBATCH --cpus-per-task={slurm_config.cpus_per_task}",
            f"#SBATCH --time={formatted_time}",
            f"#SBATCH --mem={slurm_config.mem}",
            f"#SBATCH --partition={slurm_config.partition}",
        ]
    )

    # Submit an array job or a single job
    if array_job_flag:
        if num_configs > slurm_config.max_concurrent_jobs:
            lines.append(
                f"#SBATCH --array=0-{num_configs-1}%{slurm_config.max_concurrent_jobs}"
            )
        else:
            lines.append(f"#SBATCH --array=0-{num_configs-1}")

    lines.append("")
    lines.append("# ================================================================ ")
    lines.append("# This file has been generated by generate_ei_sweep.py script ")
    lines.append("#                            ***                                   ")
    lines.append("#       Manual changes to this file may be overwritten.            ")
    lines.append("# ================================================================ ")

    # Add module loading lines
    lines.append("module purge")
    for module in slurm_config.modules_to_load:
        lines.append(f"module load {module}")

    lines.append("")
    lines.append(f"conda activate {slurm_config.conda_env_path}")
    lines.append("")

    # Set up all the path variables
    lines.append("# Path to the config file directory")
    lines.append(f"CONFIG_DIR={output_dir}/configs")
    lines.append("")
    lines.append(f"# Path to main output directory")
    lines.append(f"MAIN_OUTPUT_DIR={output_dir}")
    lines.append("")
    lines.append(f"# Subdirectories")
    lines.append(f"RESULTS_DIR=${{MAIN_OUTPUT_DIR}}/results")
    lines.append(f"PLOTS_DIR=${{MAIN_OUTPUT_DIR}}/plots")
    lines.append("")

    lines.append(f"CONFIG_FILE_NAME_INIT={slurm_config.config_filename_init}")
    lines.append(f"SC_RUN_NAME={slurm_config.run_name}")
    lines.append("")
    lines.append("#Training script: ")
    lines.append(f"TRAINING_SCRIPT={slurm_config.training_script}")
    lines.append("")
    lines.append("#Python path: ")
    lines.append(f"PYTHON={slurm_config.conda_env_path}/bin/python")
    lines.append("")

    # Get the specific config file based on the array job ID
    if array_job_flag:
        lines.append(
            "CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_${SLURM_ARRAY_TASK_ID}.yaml"
        )
        lines.append("TASK_ID=${SLURM_ARRAY_TASK_ID}")
        lines.append("JOB_ID=${SLURM_ARRAY_JOB_ID}")
    else:
        lines.append("CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_0.yaml")
        lines.append("TASK_ID=0")
        lines.append("JOB_ID=${SLURM_JOB_ID}")
    lines.append("")

    # Set up output folders with a single main job folder and task subfolders
    lines.append(f"# Path to the main job results folder")
    lines.append(f"JOB_FOLDER_PATH=${{RESULTS_DIR}}/${{JOB_ID}}")
    lines.append(f"TASK_FOLDER_PATH=${{JOB_FOLDER_PATH}}/task_${{TASK_ID}}")
    lines.append("")

    lines.append("# Create output directory")
    lines.append("mkdir -p ${JOB_FOLDER_PATH}/task_${TASK_ID}")
    lines.append("mkdir -p ${PLOTS_DIR}/${JOB_ID}/task_${TASK_ID}")
    lines.append("")

    # Run the job
    lines.append('echo "Starting running a training job at $(date)"')
    lines.append("start_time=$(date +%s)")
    lines.append("srun \\")
    lines.append("  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\")
    lines.append("  --kill-on-bad-exit \\")

    # Fix the training script path and set environment variables for output paths
    if slurm_config.training_script.startswith("/"):
        # If it's already an absolute path, use it as is
        lines.append("  ${PYTHON} -u ${TRAINING_SCRIPT} ${CONFIG}")
    else:
        # If it's a relative path, prepend the current directory
        lines.append("  ${PYTHON} -u $(pwd)/${TRAINING_SCRIPT} ${CONFIG}")

    lines.append("end_time=$(date +%s)")
    lines.append('echo "Done with running a training job at $(date)"')
    lines.append('echo "Total duration: $((end_time - start_time)) seconds."')

    # Save logs to results directory
    lines.append('echo "Copying logs to results directory"')
    lines.append("cp ${CONFIG} ${JOB_FOLDER_PATH}/task_${TASK_ID}/config_used.yaml")

    # Write to file - now directly to the main script path
    with open(main_script_path, "w") as file:
        for line in lines:
            file.write(line + "\n")

    return main_script_path


def main():
    args = parse_args()
    config_file = args.config_file

    # Read the config file
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")

    # Generate timestamp for unique folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Read the config file
    config = OmegaConf.load(config_file)

    # Check if the config has the necessary sections
    required_sections = ["slurm_config", "train_config", "sweep_config"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config file missing required section: {section}")

    # Get the prefix from config or use default
    subdir_prefix = config.slurm_config.get("subdir_prefix", "ei_")

    # Create the uniquely named output directory
    output_base_dir = config.slurm_config.output_dir
    output_dir = os.path.join(output_base_dir, f"{subdir_prefix}{timestamp}")

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories as specified in the config
    subfolders = config.slurm_config.get(
        "run_subfolders", ["configs", "results", "plots"]
    )
    for subfolder in subfolders:
        subdir_path = os.path.join(output_dir, subfolder)
        os.makedirs(subdir_path, exist_ok=True)
        print(f"Created subdirectory: {subdir_path}")

    # Update the config with the new output directory
    config.slurm_config.output_dir = output_dir

    # Check if the "outputs" block exists, if not add it
    if "outputs" not in config.train_config:
        config.train_config.outputs = {"dir": ""}

    # Ensure outputs.dir is set to the specific results directory
    config.train_config.outputs.dir = os.path.join(output_dir, "results")

    # Save the updated base config to the configs subdirectory
    updated_config_path = os.path.join(output_dir, "configs", "sweep_params.yaml")
    OmegaConf.save(config, updated_config_path)
    print(f"Saved updated config to: {updated_config_path}")

    # Generate all the sweep configurations directly
    config_filename_init = config.slurm_config.config_filename_init
    num_configs = generate_sweep_configs_manually(
        config, output_dir, config_filename_init
    )

    print(
        f"Generated {num_configs} config files in {os.path.join(output_dir, 'configs')}"
    )

    # Generate run.sh directly in the project directory
    run_script_path = generate_custom_run_script(
        config, output_dir, num_configs, "./run.sh"
    )

    # Also generate run_ei_sweep.sh for backward compatibility or if the user prefers that name
    shutil.copy(run_script_path, "./run_ei_sweep.sh")

    print(f"Created run scripts: run.sh and run_ei_sweep.sh")

    print("\nSummary:")
    print(f"1. Created sweep directory: {output_dir}")
    print(
        f"2. Generated {num_configs} config files in: {os.path.join(output_dir, 'configs')}"
    )
    print(
        f"3. Results will be stored in: {os.path.join(output_dir, 'results')}/<job_id>/task_<task_id>/"
    )
    print(
        f"4. Plots will be saved to: {os.path.join(output_dir, 'plots')}/<job_id>/task_<task_id>/"
    )
    print(f"5. To run the sweep, use: sbatch run.sh")


if __name__ == "__main__":
    main()
