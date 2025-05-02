#!/usr/bin/env python
"""
rerun_missing_configs.py
------------------------
Script to check which configurations from an EI sweep don't have results
and generate a new SLURM job script to run only the missing configurations.

Usage:
    python scripts/rerun_missing_configs.py --folder /path/to/ei_TIMESTAMP --output-file rerun_missing.sh
    python scripts/rerun_missing_configs.py --timestamp TIMESTAMP --output-file rerun_missing.sh
    python scripts/rerun_missing_configs.py --timestamp TIMESTAMP --config configs/sweep_params_seeds_ei.yaml
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check for missing results in an EI sweep and create a batch job for them"
    )

    # Either provide a full path or a timestamp
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", help="Full path to the ei_TIMESTAMP folder")
    group.add_argument("--timestamp", help="Timestamp of the ei_TIMESTAMP folder")

    # Add config file parameter
    parser.add_argument(
        "--config",
        default="configs/sweep_params_seeds_ei.yaml",
        help="Path to the sweep configuration file to get output_dir",
    )

    # Where to find the base folder if only timestamp provided and no config file is specified
    parser.add_argument(
        "--base-dir",
        default="",
        help="Base directory where ei_TIMESTAMP folders are located (overridden if config file has output_dir)",
    )

    # Output options
    parser.add_argument(
        "--output-file",
        default="rerun_missing.sh",
        help="Path to save the generated SLURM script",
    )

    # Job settings
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=40,
        help="Maximum number of concurrent jobs",
    )

    # Add verbose mode
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about configs and results",
    )

    # Add option to check log content
    parser.add_argument(
        "--check-log-content",
        action="store_true",
        help="Check log content to verify task completion rather than just file existence",
    )

    # Add option to rerun all configurations with any failed attempts
    parser.add_argument(
        "--rerun-partial-failures",
        action="store_true",
        help="Rerun configs that have any failed attempts, even if they also have successful runs",
    )

    # Add option to specify exactly which config IDs to rerun
    parser.add_argument(
        "--force-config-ids",
        type=str,
        help="Comma-separated list of config IDs to forcibly rerun (e.g., '8,15,23')",
    )

    # Add option to identify any task directory that needs to be rerun
    parser.add_argument(
        "--identify-failed-directories",
        action="store_true",
        help="Identify specific job directories with failed tasks",
    )

    return parser.parse_args()


def get_config_ids(configs_dir, verbose=False):
    """Get all config IDs from the configs directory."""
    config_files = glob.glob(os.path.join(configs_dir, "*.yaml"))
    config_ids = []

    if verbose:
        print(f"Looking for config files in {configs_dir}")
        print(f"Found {len(config_files)} yaml files")

    for config_file in config_files:
        # Extract the config ID from the filename (e.g., ei_sweep_config_42.yaml -> 42)
        filename = os.path.basename(config_file)
        parts = filename.split("_")
        if parts and parts[-1].endswith(".yaml"):
            try:
                config_id = int(parts[-1].replace(".yaml", ""))
                config_ids.append(config_id)
                if verbose:
                    print(f"Found config file: {filename} with ID {config_id}")
            except ValueError:
                if verbose:
                    print(f"Skipping file {filename} - not a numbered config")
                continue

    return sorted(config_ids)


def check_result_exists(results_dir, config_id, verbose=False, check_log_content=False):
    """Check if results exist for a specific config ID by searching all job folders."""
    # Check all job directories for the given task ID
    job_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    if verbose:
        print(f"Checking for config {config_id} in {len(job_dirs)} job directories")

    for job_dir in job_dirs:
        job_path = os.path.join(results_dir, job_dir)
        task_dir = os.path.join(job_path, f"task_{config_id}")

        # Check if this specific task directory exists
        if os.path.isdir(task_dir):
            if verbose:
                print(f"Found task directory for config {config_id}: {task_dir}")

            # Check for result files
            config_used = os.path.join(task_dir, "config_used.yaml")
            output_log = os.path.join(task_dir, "output.log")

            config_exists = os.path.exists(config_used)
            log_exists = os.path.exists(output_log)

            # Default completion status based on file existence
            task_completed = config_exists and log_exists

            # For more thorough checking, verify log content
            if task_completed and check_log_content:
                try:
                    with open(output_log, "r") as f:
                        log_content = f.read()
                        # Look for typical completion markers in the log
                        has_markers = any(
                            marker in log_content
                            for marker in [
                                "Training complete",
                                "Done with running a training job",
                                "Total duration:",
                                "Experiment finished",
                            ]
                        )

                        # Check if log is too small (might indicate failure)
                        too_small = len(log_content.strip()) < 100

                        # Update completion status
                        if too_small or not has_markers:
                            task_completed = False
                            if verbose:
                                reason = (
                                    "log too small"
                                    if too_small
                                    else "no completion markers"
                                )
                                print(f"  Task directory exists but {reason}")
                except Exception as e:
                    if verbose:
                        print(f"  Error reading log file: {e}")

            if verbose:
                status = "COMPLETE" if task_completed else "INCOMPLETE"
                details = []
                if not config_exists:
                    details.append("missing config")
                if not log_exists:
                    details.append("missing log")
                detail_str = f" ({', '.join(details)})" if details else ""
                print(f"  Status: {status}{detail_str}")

            if task_completed:
                return True

    if verbose:
        print(f"No valid results found for config {config_id}")

    return False


def get_base_dir_from_config(config_file, verbose=False):
    """Extract the output_dir from the sweep configuration file."""
    if verbose:
        print(f"Reading base directory from config file: {config_file}")

    try:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)

            if "slurm_config" in config and "output_dir" in config["slurm_config"]:
                base_dir = config["slurm_config"]["output_dir"]
                if base_dir:  # Check if it's not empty
                    if verbose:
                        print(f"Found base directory in config: {base_dir}")
                    return base_dir
                else:
                    if verbose:
                        print("Base directory in config file is empty")
            else:
                if verbose:
                    if "slurm_config" not in config:
                        print("Config file does not have 'slurm_config' section")
                    elif "output_dir" not in config["slurm_config"]:
                        print(
                            "Config file has 'slurm_config' but no 'output_dir' field"
                        )
    except Exception as e:
        print(f"Error reading config file {config_file}: {e}")

    if verbose:
        print("No base directory found in config file or it's empty")

    return None


def find_missing_configs(
    configs_dir,
    results_dir,
    verbose=False,
    check_log_content=False,
    rerun_partial_failures=False,
    force_config_ids=None,
):
    """Find all configs that don't have completed results."""
    # Get all config IDs
    config_ids = get_config_ids(configs_dir, verbose)

    if verbose:
        print(f"Found {len(config_ids)} configuration files")

    # If force_config_ids is specified, only consider those IDs
    if force_config_ids:
        forced_ids = [int(id_str.strip()) for id_str in force_config_ids.split(",")]
        config_ids = [cid for cid in config_ids if cid in forced_ids]
        if verbose:
            print(
                f"Filtering to only consider forced config IDs: {', '.join(map(str, config_ids))}"
            )

    # Track successful and failed attempts for each config
    completed_ids = set()
    partial_failure_ids = set()
    missing_ids = []

    # Track task directories with failures
    failed_task_dirs = []

    for config_id in config_ids:
        success, has_failures, task_failures = check_result_for_config(
            results_dir, config_id, verbose, check_log_content
        )

        # Add any failed task directories to the list
        failed_task_dirs.extend(task_failures)

        if success:
            completed_ids.add(config_id)
            if has_failures and rerun_partial_failures:
                partial_failure_ids.add(config_id)
                if verbose:
                    print(
                        f"Config {config_id} has successful runs but also some failures - marking for rerun"
                    )
        else:
            missing_ids.append(config_id)

    # If rerun_partial_failures is true, add partially failed configs to missing_ids
    if rerun_partial_failures:
        for config_id in partial_failure_ids:
            missing_ids.append(config_id)
            if verbose:
                print(
                    f"Adding config {config_id} to rerun list due to partial failures"
                )

    # Also add forced config IDs
    if force_config_ids:
        for config_id in forced_ids:
            if config_id not in missing_ids and config_id in config_ids:
                missing_ids.append(config_id)
                if verbose:
                    print(
                        f"Adding config {config_id} to rerun list due to --force-config-ids"
                    )

    return config_ids, completed_ids, sorted(missing_ids), failed_task_dirs


def check_result_for_config(
    results_dir, config_id, verbose=False, check_log_content=False
):
    """Check if results exist for a specific config ID by searching all job folders.

    Returns:
        (has_success, has_failures, failed_dirs): Tuple containing:
                                    - Whether there are any successful runs
                                    - Whether there are any failed runs
                                    - List of directories with failed runs
    """
    # Check all job directories for the given task ID
    job_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]

    if verbose:
        print(f"Checking for config {config_id} in {len(job_dirs)} job directories")

    has_success = False
    has_failures = False
    failed_task_dirs = []

    for job_dir in job_dirs:
        job_path = os.path.join(results_dir, job_dir)
        task_dir = os.path.join(job_path, f"task_{config_id}")

        # Check if this specific task directory exists
        if os.path.isdir(task_dir):
            if verbose:
                print(f"Found task directory for config {config_id}: {task_dir}")

            # Check for result files
            config_used = os.path.join(task_dir, "config_used.yaml")
            output_log = os.path.join(task_dir, "output.log")

            config_exists = os.path.exists(config_used)
            log_exists = os.path.exists(output_log)

            # Default completion status based on file existence
            task_completed = config_exists and log_exists

            # For more thorough checking, verify log content
            if task_completed and check_log_content:
                try:
                    with open(output_log, "r") as f:
                        log_content = f.read()
                        # Look for typical completion markers in the log
                        has_markers = any(
                            marker in log_content
                            for marker in [
                                "Training complete",
                                "Done with running a training job",
                                "Total duration:",
                                "Experiment finished",
                            ]
                        )

                        # Check if log is too small (might indicate failure)
                        too_small = len(log_content.strip()) < 100

                        # Update completion status
                        if too_small or not has_markers:
                            task_completed = False
                            if verbose:
                                reason = (
                                    "log too small"
                                    if too_small
                                    else "no completion markers"
                                )
                                print(f"  Task directory exists but {reason}")
                except Exception as e:
                    if verbose:
                        print(f"  Error reading log file: {e}")

            if verbose:
                status = "COMPLETE" if task_completed else "INCOMPLETE"
                details = []
                if not config_exists:
                    details.append("missing config")
                if not log_exists:
                    details.append("missing log")
                detail_str = f" ({', '.join(details)})" if details else ""
                print(f"  Status: {status}{detail_str}")

            if task_completed:
                has_success = True
            else:
                has_failures = True
                failed_task_dirs.append(task_dir)

    if verbose:
        if not has_success and not has_failures:
            print(f"No task directories found for config {config_id}")
        elif has_success:
            if has_failures:
                print(
                    f"Config {config_id} has mixed results: some successful runs, some failed runs"
                )
            else:
                print(f"Config {config_id} has only successful runs")
        else:
            print(f"Config {config_id} has NO successful runs")

    return has_success, has_failures, failed_task_dirs


def get_original_slurm_config(sweep_dir):
    """Get the original SLURM configuration used for this sweep."""
    # Try to find the sweep_params.yaml file
    sweep_params_path = os.path.join(sweep_dir, "configs", "sweep_params.yaml")

    if not os.path.exists(sweep_params_path):
        print(f"Error: Could not find sweep_params.yaml in {sweep_dir}/configs")
        return None

    # Load the sweep config file
    with open(sweep_params_path, "r") as f:
        try:
            config = yaml.safe_load(f)
            if "slurm_config" in config:
                return config["slurm_config"]
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")

    return None


def generate_rerun_script(
    missing_ids,
    sweep_dir,
    output_file,
    max_concurrent,
    failed_task_dirs=None,
    identify_failed_directories=False,
):
    """Generate a SLURM script to rerun missing configurations."""
    # Get the original SLURM settings
    slurm_config = get_original_slurm_config(sweep_dir)
    if not slurm_config:
        print("Error: Could not retrieve original SLURM configuration")
        return False

    # Get script name details
    sweep_timestamp = os.path.basename(sweep_dir)
    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Start generating the script
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=rerun_{sweep_timestamp}",
        f"#SBATCH --account={slurm_config.get('account', 'kempner_dev')}",
    ]

    # Output paths
    lines.extend(
        [
            f"#SBATCH --output={sweep_dir}/results/%A/task_%a/output.log",
            f"#SBATCH --error={sweep_dir}/results/%A/task_%a/error.log",
        ]
    )

    # Resources
    lines.extend(
        [
            f"#SBATCH --nodes={slurm_config.get('nodes', 1)}",
            f"#SBATCH --ntasks-per-node={slurm_config.get('ntasks_per_node', 1)}",
            f"#SBATCH --gpus-per-node={slurm_config.get('gpus_per_node', 1)}",
            f"#SBATCH --cpus-per-task={slurm_config.get('cpus_per_task', 10)}",
            f"#SBATCH --time={slurm_config.get('time', '1:00:00')}",
            f"#SBATCH --mem={slurm_config.get('mem', '64GB')}",
            f"#SBATCH --partition={slurm_config.get('partition', 'kempner_gpu')}",
        ]
    )

    # Handle array job configuration
    if len(missing_ids) > 1:
        # Format array indices as comma-separated list: 1,2,3,4,5
        array_indices = ",".join(map(str, missing_ids))
        lines.append(f"#SBATCH --array={array_indices}%{max_concurrent}")
    else:
        # Single task, no need for array
        single_id = missing_ids[0] if missing_ids else 0
        lines.append(f"#SBATCH --array={single_id}")

    # Add metadata
    lines.extend(
        [
            "",
            "# ================================================================ ",
            "# This file has been generated by rerun_missing_configs.py script ",
            "#                            ***                                   ",
            f"# Rerunning missing configs from: {sweep_timestamp}              ",
            f"# Generated on: {current_timestamp}                              ",
            "# ================================================================ ",
        ]
    )

    # Add module loading and conda environment
    lines.extend(
        [
            "module purge",
        ]
    )

    for module in slurm_config.get("modules_to_load", []):
        lines.append(f"module load {module}")

    lines.extend(
        [
            "",
            f"conda activate {slurm_config.get('conda_env_path', '/path/to/conda/env')}",
            "",
        ]
    )

    # Set up directories
    lines.extend(
        [
            "# Path to the config file directory",
            f"CONFIG_DIR={sweep_dir}/configs",
            "",
            "# Path to main output directory",
            f"MAIN_OUTPUT_DIR={sweep_dir}",
            "",
            "# Subdirectories",
            "RESULTS_DIR=${MAIN_OUTPUT_DIR}/results",
            "PLOTS_DIR=${MAIN_OUTPUT_DIR}/plots",
            "",
            f"CONFIG_FILE_NAME_INIT={slurm_config.get('config_filename_init', 'ei_sweep_config')}",
            f"SC_RUN_NAME=rerun_{slurm_config.get('run_name', 'dendritic_ei_sweep')}",
            "",
            "#Training script: ",
            f"TRAINING_SCRIPT={slurm_config.get('training_script', 'src/dendritic_modeling/train_experiments.py')}",
            "",
            "#Python path: ",
            f"PYTHON={slurm_config.get('conda_env_path', '/path/to/conda/env')}/bin/python",
            "",
        ]
    )

    # Configure task execution
    lines.extend(
        [
            "CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_${SLURM_ARRAY_TASK_ID}.yaml",
            "TASK_ID=${SLURM_ARRAY_TASK_ID}",
            "JOB_ID=${SLURM_ARRAY_JOB_ID}",
            "",
            "# Path to the main job results folder",
            "JOB_FOLDER_PATH=${RESULTS_DIR}/${JOB_ID}",
            "TASK_FOLDER_PATH=${JOB_FOLDER_PATH}/task_${TASK_ID}",
            "",
            "# Create output directory",
            "mkdir -p ${JOB_FOLDER_PATH}/task_${TASK_ID}",
            "mkdir -p ${PLOTS_DIR}/${JOB_ID}/task_${TASK_ID}",
            "",
            'echo "Starting running a training job at $(date)"',
            "start_time=$(date +%s)",
            "",
            "# Unset conflicting SLURM memory environment variables to avoid conflicts",
            "unset SLURM_MEM_PER_CPU",
            "unset SLURM_MEM_PER_GPU",
            "unset SLURM_MEM_PER_NODE",
            "",
        ]
    )

    # Execute command
    lines.extend(
        [
            "srun \\",
            "  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\",
            "  --kill-on-bad-exit \\",
            "  --export=ALL,SLURM_MEM_PER_CPU=,SLURM_MEM_PER_GPU=,SLURM_MEM_PER_NODE= \\",
        ]
    )

    # Training command
    training_script = slurm_config.get("training_script", "")
    if training_script.startswith("/"):
        lines.append("  ${PYTHON} -u ${TRAINING_SCRIPT} ${CONFIG}")
    else:
        lines.append("  ${PYTHON} -u $(pwd)/${TRAINING_SCRIPT} ${CONFIG}")

    # Finish
    lines.extend(
        [
            "end_time=$(date +%s)",
            'echo "Done with running a training job at $(date)"',
            'echo "Total duration: $((end_time - start_time)) seconds."',
            'echo "Copying logs to results directory"',
            "cp ${CONFIG} ${JOB_FOLDER_PATH}/task_${TASK_ID}/config_used.yaml",
        ]
    )

    # Write the script
    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    # Make the script executable
    os.chmod(output_file, 0o755)

    return True


def main():
    args = parse_args()

    # Process force-config-ids if provided
    force_config_ids = None
    if args.force_config_ids:
        force_config_ids = args.force_config_ids
        print(f"Will forcibly rerun the following config IDs: {force_config_ids}")

    # Always enable verbose for debugging
    print(f"Script called with timestamp: {args.timestamp}")
    print(f"Config file path: {args.config}")
    print(f"Rerun partial failures: {args.rerun_partial_failures}")
    print(f"Identify failed directories: {args.identify_failed_directories}")

    # Check if config file exists
    config_exists = os.path.exists(args.config)
    print(f"Config file exists: {config_exists}")

    # Get base directory from config if provided
    base_dir = args.base_dir
    if config_exists:
        config_base_dir = get_base_dir_from_config(args.config, verbose=True)
        if config_base_dir:
            base_dir = config_base_dir
            print(f"Using base directory from config file: {base_dir}")
    else:
        print(f"WARNING: Config file not found at: {args.config}")

    if not base_dir:
        # Default if not provided in args or config
        base_dir = "/n/holylabs/LABS/kempner_dev/Users/hsafaai/results/dendritic-modeling/outputs"
        print(f"Using default base directory: {base_dir}")

    # Determine the sweep directory
    if args.folder:
        sweep_dir = args.folder
    else:
        sweep_dir = os.path.join(base_dir, f"ei_{args.timestamp}")

    print(f"Full sweep directory path: {sweep_dir}")

    # Ensure the directory exists
    if not os.path.isdir(sweep_dir):
        print(f"ERROR: Sweep directory not found: {sweep_dir}")
        print("Please verify the timestamp or base directory is correct.")
        sys.exit(1)

    # Check configuration files and results
    configs_dir = os.path.join(sweep_dir, "configs")
    results_dir = os.path.join(sweep_dir, "results")

    print(f"Checking configs directory: {configs_dir}")
    print(f"Checking results directory: {results_dir}")

    if not os.path.isdir(configs_dir):
        print(f"ERROR: Configs directory not found: {configs_dir}")
        sys.exit(1)

    if not os.path.isdir(results_dir):
        print(f"ERROR: Results directory not found: {results_dir}")
        sys.exit(1)

    # List the config files to verify
    print(f"Config files in directory:")
    config_files = glob.glob(os.path.join(configs_dir, "*.yaml"))
    for config_file in config_files[:5]:  # Show first 5 to avoid clutter
        print(f"  - {os.path.basename(config_file)}")
    if len(config_files) > 5:
        print(f"  - ... and {len(config_files) - 5} more")

    # List task directories to verify
    print(f"Result directories:")
    job_dirs = [
        d
        for d in os.listdir(results_dir)
        if os.path.isdir(os.path.join(results_dir, d))
    ]
    for job_dir in job_dirs[:3]:  # Show first 3 to avoid clutter
        job_path = os.path.join(results_dir, job_dir)
        task_dirs = glob.glob(os.path.join(job_path, "task_*"))
        print(f"  - Job {job_dir}: {len(task_dirs)} tasks")
    if len(job_dirs) > 3:
        print(f"  - ... and {len(job_dirs) - 3} more jobs")

    # Find missing configurations with explicit verbose
    config_ids, completed_ids, missing_ids, failed_task_dirs = find_missing_configs(
        configs_dir,
        results_dir,
        verbose=True,
        check_log_content=args.check_log_content,
        rerun_partial_failures=args.rerun_partial_failures,
        force_config_ids=force_config_ids,
    )

    print(f"\nSUMMARY:")
    print(f"Found {len(config_ids)} total configuration files")
    print(f"Found {len(completed_ids)} configurations with at least one successful run")

    # Count partial failures
    partial_failures = [
        cid
        for cid in completed_ids
        if cid in config_ids
        and any(f"task_{cid}" in failed_dir for failed_dir in failed_task_dirs)
    ]

    print(f"Found {len(partial_failures)} configurations with partial failures")

    # Explain the count
    if args.rerun_partial_failures:
        print(
            f"Found {len(missing_ids)} configurations that are either missing or have partial failures"
        )
    else:
        print(f"Found {len(missing_ids)} configurations with no successful runs")

    # List the task directories with failed runs
    if args.identify_failed_directories and failed_task_dirs:
        print("\nFailed task directories:")
        for task_dir in failed_task_dirs:
            print(f"  - {task_dir}")

    if force_config_ids and missing_ids:
        print(
            f"\nForced config IDs to rerun: {', '.join(map(str, [id for id in missing_ids if id in [int(x.strip()) for x in force_config_ids.split(',')]]))}"
        )

    if not missing_ids:
        if partial_failures and not args.rerun_partial_failures:
            print(
                "\nSome configs have partial failures. Use --rerun-partial-failures to include them."
            )
            print(
                f"Partial failure config IDs: {', '.join(map(str, partial_failures))}"
            )
            print(
                f"Rerun with: python scripts/rerun_missing_configs.py --timestamp {args.timestamp} --force-config-ids {','.join(map(str, partial_failures))}"
            )
        else:
            print("\nAll configurations have results. Nothing to do.")
        sys.exit(0)

    # Generate rerun script
    success = generate_rerun_script(
        missing_ids,
        sweep_dir,
        args.output_file,
        args.max_concurrent,
        failed_task_dirs,
        args.identify_failed_directories,
    )

    if success:
        print(f"Successfully generated rerun script: {args.output_file}")
        print(f"To run missing configurations, use: sbatch {args.output_file}")
    else:
        print("Failed to generate rerun script")
        sys.exit(1)


if __name__ == "__main__":
    main()
