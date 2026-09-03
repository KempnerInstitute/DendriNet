#!/usr/bin/env python
"""
rerun_missing_configs.py
------------------------
Check which configurations from a sweep don't have a successful completion
log line and generate a SLURM array job to rerun only those configs.

Usage:
    python src/dendritic_modeling/scripts/sweeps/rerun_missing_configs.py \\
        --folder /path/to/sweep_run_TIMESTAMP --output-file rerun_missing.sh
    python src/dendritic_modeling/scripts/sweeps/rerun_missing_configs.py \\
        --timestamp TIMESTAMP --config configs/sweep_config.yaml
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

SUCCESS_MESSAGE = "Training experiment completed successfully"
COMPLETION_LOG_FILES = ("dendritic_modeling.log", "train.log")
DEFAULT_TRAINING_SCRIPT = (
    "src/dendritic_modeling/scripts/training/train_experiments.py"
)
REPO_ROOT = Path(__file__).resolve().parents[4]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check for missing results in a unified sweep and create a batch job for them"
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--folder", help="Full path to the sweep output folder")
    group.add_argument(
        "--timestamp",
        help="Timestamp suffix of the {run_name}_{timestamp} sweep folder",
    )

    parser.add_argument(
        "--config",
        default="configs/sweep_config.yaml",
        help="Path to the sweep configuration file used to resolve output_dir and run_name",
    )
    parser.add_argument(
        "--base-dir",
        default="",
        help="Base directory containing sweep folders (overrides values from --config)",
    )
    parser.add_argument(
        "--output-file",
        default="rerun_missing.sh",
        help="Path to save the generated SLURM script",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Maximum number of concurrent array tasks (defaults to slurm_config.max_concurrent_jobs)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information about configs and results",
    )
    parser.add_argument(
        "--check-log-content",
        action="store_true",
        help="Unused; kept for backward compatibility",
    )
    parser.add_argument(
        "--rerun-partial-failures",
        action="store_true",
        help="Unused; kept for backward compatibility",
    )
    parser.add_argument(
        "--force-config-ids",
        type=str,
        help="Comma-separated list of config IDs to forcibly rerun (e.g., '8,15,23')",
    )
    parser.add_argument(
        "--identify-failed-directories",
        action="store_true",
        help="Unused; kept for backward compatibility",
    )

    return parser.parse_args()


def _nested_get(mapping, *keys):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _load_yaml(path):
    with open(path) as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else {}


def _extract_config_id(filename):
    if filename.startswith("sweep_metadata"):
        return None
    stem = filename.replace(".yaml", "")
    for part in reversed(stem.split("_")):
        try:
            return int(part)
        except ValueError:
            continue
    return None


def get_config_file_map(configs_dir, verbose=False):
    """Map config IDs to YAML paths under the sweep configs directory."""
    config_files = glob.glob(os.path.join(configs_dir, "*.yaml"))
    config_map = {}

    if verbose:
        print(f"Looking for config files in {configs_dir}")
        print(f"Found {len(config_files)} yaml files")

    for config_file in config_files:
        filename = os.path.basename(config_file)
        config_id = _extract_config_id(filename)
        if config_id is None:
            continue
        config_map[config_id] = config_file
        if verbose:
            print(f"Found config file: {filename} with ID {config_id}")

    return dict(sorted(config_map.items()))


def get_config_ids(configs_dir, verbose=False):
    """Get all config IDs from the configs directory."""
    return sorted(get_config_file_map(configs_dir, verbose).keys())


def check_result_for_config(
    results_dir, config_id, verbose=False, check_log_content=False
):
    """Return True when the training completion log line is present.

    A saved model without this line is treated as incomplete. ``check_log_content``
    is unused and kept for call-site compatibility.
    """
    del check_log_content
    if verbose:
        print(f"\n===== Checking completion for config ID {config_id} =====")
        print(f"Looking in results directory: {results_dir}")

    config_dir = os.path.join(results_dir, f"config_{config_id}")
    if not os.path.isdir(config_dir):
        if verbose:
            print(f"Config {config_id} incomplete: no config directory found\n")
        return False, True, []

    if verbose:
        print(f"Found config directory: {config_dir}")

    for log_name in COMPLETION_LOG_FILES:
        output_log = os.path.join(config_dir, log_name)
        if not os.path.exists(output_log):
            if verbose:
                print(f"No output log found at {output_log}")
            continue
        if verbose:
            print(f"Found output log: {output_log}")
        try:
            with open(output_log) as f:
                content = f.read()
        except OSError as exc:
            if verbose:
                print(f"Error reading output log: {exc}")
            continue
        if SUCCESS_MESSAGE in content:
            if verbose:
                print(f"FOUND: {SUCCESS_MESSAGE!r} in {output_log}")
            return True, False, []
        if verbose:
            print(f"NOT FOUND: Success message not in {output_log}")

    if verbose:
        print(
            f"Config {config_id} incomplete: no successful completion message found\n"
        )
    return False, True, []


def find_missing_configs(
    configs_dir,
    results_dir,
    verbose=False,
    check_log_content=False,
    rerun_partial_failures=False,
    force_config_ids=None,
):
    """Find all configs that don't have the successful completion log line."""
    del rerun_partial_failures
    config_ids = get_config_ids(configs_dir, verbose)

    if verbose:
        print(f"\nFound {len(config_ids)} total config files in {configs_dir}")

    forced_ids = []
    if force_config_ids:
        forced_ids = [int(id_str.strip()) for id_str in force_config_ids.split(",")]
        config_ids = [cid for cid in config_ids if cid in forced_ids]
        if verbose:
            print(f"Limiting check to forced config IDs: {forced_ids}")

    completed_ids = set()
    missing_ids = []

    if verbose:
        print(
            f"\nChecking {len(config_ids)} configurations for successful completion..."
        )

    for config_id in config_ids:
        is_completed, _, _ = check_result_for_config(
            results_dir, config_id, verbose, check_log_content
        )
        if is_completed:
            completed_ids.add(config_id)
            if verbose:
                print(
                    f"Config {config_id} is COMPLETED (successful completion message found) and will not be rerun"
                )
        else:
            missing_ids.append(config_id)
            if verbose:
                print(
                    f"Config {config_id} is INCOMPLETE (no successful completion message) and will be rerun"
                )

    if force_config_ids:
        for config_id in forced_ids:
            if config_id not in missing_ids and config_id in config_ids:
                missing_ids.append(config_id)
                if verbose:
                    print(
                        f"Config {config_id} is being forcibly added to the rerun list"
                    )

    if verbose:
        print("\nCompletion check summary:")
        print(f"- Total configs checked: {len(config_ids)}")
        print(f"- Completed configs: {len(completed_ids)} - {sorted(completed_ids)}")
        print(
            f"- Missing/incomplete configs: {len(missing_ids)} - {sorted(missing_ids)}"
        )

    return config_ids, completed_ids, sorted(missing_ids), []


def get_sweep_layout_from_config(config_file, verbose=False):
    """Return (base_dir, run_name) using the same lookup as SweepManager."""
    if verbose:
        print(f"Reading sweep layout from config file: {config_file}")

    try:
        config = _load_yaml(config_file)
    except OSError as exc:
        print(f"Error reading config file {config_file}: {exc}")
        return None, "unified_sweep"

    base_dir = (
        config.get("output_dir")
        or _nested_get(config, "outputs", "dir")
        or _nested_get(config, "base_config", "outputs", "results_dir")
        or "output"
    )
    run_name = (
        _nested_get(config, "outputs", "run_name")
        or _nested_get(config, "base_config", "outputs", "run_name")
        or "unified_sweep"
    )

    if verbose:
        print(f"Found base directory in config: {base_dir}")
        print(f"Found run name in config: {run_name}")

    return base_dir, run_name


def get_base_dir_from_config(config_file, verbose=False):
    """Extract the output directory from the sweep configuration file."""
    base_dir, _ = get_sweep_layout_from_config(config_file, verbose=verbose)
    return base_dir


def resolve_sweep_directory(
    folder=None,
    timestamp=None,
    config_file=None,
    base_dir="",
    verbose=False,
):
    """Resolve the sweep output directory from --folder or --timestamp."""
    if folder:
        return folder

    run_name = "unified_sweep"
    if config_file and os.path.exists(config_file):
        config_base_dir, run_name = get_sweep_layout_from_config(
            config_file, verbose=verbose
        )
        if not base_dir:
            base_dir = config_base_dir or "output"
    elif not base_dir:
        base_dir = "output"

    candidates = [
        os.path.join(base_dir, f"{run_name}_{timestamp}"),
        os.path.join(base_dir, f"unified_{timestamp}"),
        os.path.join(base_dir, f"sweep_{timestamp}"),
        os.path.join(base_dir, timestamp),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def get_original_slurm_config(sweep_dir):
    """Get the original SLURM configuration used for this sweep."""
    sweep_params_path = os.path.join(sweep_dir, "original_config.yaml")
    metadata_path = os.path.join(sweep_dir, "configs", "sweep_metadata.yaml")

    config_file = None
    if os.path.exists(sweep_params_path):
        config_file = sweep_params_path
    elif os.path.exists(metadata_path):
        config_file = metadata_path
    else:
        print(f"Error: Could not find sweep configuration in {sweep_dir}")
        return None

    try:
        config = _load_yaml(config_file)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        return None

    slurm_config = config.get("slurm_config")
    return slurm_config if isinstance(slurm_config, dict) else None


def _format_slurm_time(time_value):
    if isinstance(time_value, (int, float)) or (
        isinstance(time_value, str) and time_value.isdigit()
    ):
        seconds = int(float(time_value))
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    if isinstance(time_value, str):
        if ":" not in time_value:
            return f"{time_value}:00:00"
        return time_value
    return "3:00:00"


def generate_rerun_script(
    missing_ids,
    sweep_dir,
    output_file,
    max_concurrent=None,
    failed_task_dirs=None,
    identify_failed_directories=False,
):
    """Generate a SLURM script to rerun missing configurations."""
    del failed_task_dirs, identify_failed_directories

    slurm_config = get_original_slurm_config(sweep_dir)
    if not slurm_config:
        print("Error: Could not retrieve original SLURM configuration")
        return False

    account = slurm_config.get("account")
    partition = slurm_config.get("partition")
    if not account:
        print("Error: SLURM account not set in original_config.yaml slurm_config")
        return False
    if not partition:
        print("Error: SLURM partition not set in original_config.yaml slurm_config")
        return False

    if max_concurrent is None:
        max_concurrent = slurm_config.get("max_concurrent_jobs", 10)

    config_map = get_config_file_map(os.path.join(sweep_dir, "configs"))
    missing_config_paths = {}
    for config_id in missing_ids:
        config_path = config_map.get(config_id)
        if not config_path:
            print(f"Error: Could not find config YAML for config ID {config_id}")
            return False
        missing_config_paths[config_id] = os.path.relpath(
            os.path.abspath(config_path), start=REPO_ROOT
        )

    sweep_timestamp = os.path.basename(sweep_dir)
    current_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    formatted_time = _format_slurm_time(slurm_config.get("time", "02:00:00"))
    mem_value = slurm_config.get("mem", "32G")
    training_script = slurm_config.get("training_script") or DEFAULT_TRAINING_SCRIPT
    conda_env_path = slurm_config.get("conda_env_path") or ""

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=rerun_{sweep_timestamp}",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
    ]
    if slurm_config.get("qos"):
        lines.append(f"#SBATCH --qos={slurm_config['qos']}")
    lines.append(f"#SBATCH --chdir={REPO_ROOT}")
    constraint = slurm_config.get("constraint")
    if constraint:
        lines.append(f"#SBATCH --constraint={constraint}")
    lines.extend(
        [
            f"#SBATCH --time={formatted_time}",
            f"#SBATCH --nodes={slurm_config.get('nodes', 1)}",
            f"#SBATCH --ntasks-per-node={slurm_config.get('ntasks_per_node', 1)}",
            f"#SBATCH --gres=gpu:{slurm_config.get('gpus_per_node', 1)}",
            f"#SBATCH --cpus-per-task={slurm_config.get('cpus_per_task', 4)}",
            f"#SBATCH --mem={mem_value}",
        ]
    )

    exclude_nodes = slurm_config.get("exclude_nodes") or []
    if exclude_nodes:
        lines.append(f"#SBATCH --exclude={','.join(exclude_nodes)}")

    array_indices = ",".join(map(str, missing_ids))
    if len(missing_ids) > 1:
        lines.append(f"#SBATCH --array={array_indices}%{max_concurrent}")
    else:
        single_id = missing_ids[0] if missing_ids else 0
        lines.append(f"#SBATCH --array={single_id}")

    lines.extend(
        [
            f"#SBATCH --output={sweep_dir}/jobs/%A_%a/output_%A_%a.out",
            f"#SBATCH --error={sweep_dir}/jobs/%A_%a/error_%A_%a.err",
            "",
            "set -euo pipefail",
            "",
            "# ================================================================",
            "# This file has been generated by rerun_missing_configs.py script",
            "#                            ***",
            f"# Rerunning missing configs from: {sweep_timestamp}",
            f"# Generated on: {current_timestamp}",
            "# ================================================================",
            "",
            "module purge",
        ]
    )

    for module in slurm_config.get("modules_to_load") or []:
        lines.append(f"module load {module}")

    if conda_env_path:
        lines.extend(
            [
                "",
                f"# Using conda environment at: {conda_env_path}",
                f'PYTHON="{conda_env_path}/bin/python"',
                "",
                "# Set library path for conda environment",
                f'export LD_LIBRARY_PATH="{conda_env_path}/lib:${{LD_LIBRARY_PATH:-}}"',
                "",
                "# Verify environment works",
                '$PYTHON -c "import torch; print(f\'PyTorch version: {torch.__version__}\')"',
            ]
        )
    else:
        lines.extend(
            [
                "",
                "# Using system python",
                'PYTHON="python"',
            ]
        )

    lines.extend(
        [
            "",
            'export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"',
            "",
            f'OUTPUT_DIR="{sweep_dir}"',
            'OUTPUT_FOLDER_PATH="${OUTPUT_DIR}/results/config_${SLURM_ARRAY_TASK_ID}"',
            "",
            'case "${SLURM_ARRAY_TASK_ID}" in',
        ]
    )
    for config_id, rel_config_path in missing_config_paths.items():
        lines.append(f'  {config_id}) CONFIG="{rel_config_path}" ;;')
    lines.extend(
        [
            '  *) echo "Error: No config mapped for task ID ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;',
            "esac",
            "",
            "mkdir -p ${OUTPUT_FOLDER_PATH}",
            "",
            'echo "Starting training job at $(date)"',
            'echo "Array task $SLURM_ARRAY_TASK_ID processing: $CONFIG"',
            "start_time=$(date +%s)",
            "",
            "# Run experiment with srun",
            "srun \\",
            "  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\",
            "  --kill-on-bad-exit \\",
        ]
    )

    if training_script.startswith("/"):
        lines.append(
            f'  $PYTHON -u {training_script} "$CONFIG" --output_dir "$OUTPUT_FOLDER_PATH"'
        )
    else:
        lines.append(
            f'  $PYTHON -u $(pwd)/{training_script} "$CONFIG" --output_dir "$OUTPUT_FOLDER_PATH"'
        )

    lines.extend(
        [
            "",
            "end_time=$(date +%s)",
            'echo "Done with training job at $(date)"',
            'echo "Total duration: $((end_time - start_time)) seconds."',
        ]
    )

    with open(output_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

    os.chmod(output_file, 0o755)
    return True


def main():
    args = parse_args()

    force_config_ids = None
    if args.force_config_ids:
        force_config_ids = args.force_config_ids
        print(f"Will forcibly rerun the following config IDs: {force_config_ids}")

    base_dir = args.base_dir
    config_path = args.config if os.path.exists(args.config) else None
    sweep_dir = resolve_sweep_directory(
        folder=args.folder,
        timestamp=args.timestamp,
        config_file=config_path,
        base_dir=base_dir,
        verbose=args.verbose,
    )

    print(f"Checking sweep directory: {sweep_dir}")

    if not os.path.isdir(sweep_dir):
        print(f"ERROR: Sweep directory not found: {sweep_dir}")
        sys.exit(1)

    configs_dir = os.path.join(sweep_dir, "configs")
    results_dir = os.path.join(sweep_dir, "results")

    if not os.path.isdir(configs_dir) or not os.path.isdir(results_dir):
        print("ERROR: Could not find configs or results directories")
        sys.exit(1)

    config_ids, completed_ids, missing_ids, _ = find_missing_configs(
        configs_dir,
        results_dir,
        verbose=args.verbose,
        check_log_content=args.check_log_content,
        rerun_partial_failures=args.rerun_partial_failures,
        force_config_ids=force_config_ids,
    )

    print("\nSUMMARY:")
    print(f"Found {len(config_ids)} total configurations")
    print(
        f"Found {len(completed_ids)} completed configurations (with successful completion messages)"
    )
    print(f"Found {len(missing_ids)} configurations to rerun (missing or failed)")

    if not missing_ids:
        print("\nAll configurations have completed successfully. Nothing to do.")
        sys.exit(0)

    success = generate_rerun_script(
        missing_ids,
        sweep_dir,
        args.output_file,
        max_concurrent=args.max_concurrent,
    )

    if success:
        print(f"Successfully generated rerun script: {args.output_file}")
        print(f"To run missing configurations, use: sbatch {args.output_file}")
    else:
        print("Failed to generate rerun script")
        sys.exit(1)


if __name__ == "__main__":
    main()
