import argparse
import os
from copy import deepcopy
from itertools import product
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def read_config(directory="configs", param_file_name="sweep_params.yaml"):
    """
    Reads the configuration file from the specified directory.

    Parameters
    ----------

    directory : str
        The directory containing the configuration file. Default is "configs".

    param_file_name : str
        The name of the configuration file. Default is "sweep_params.yaml".
    """

    config_path = os.path.join(directory, param_file_name)

    try:
        assert os.path.exists(config_path), f"Config file not found at {config_path}"
        params = OmegaConf.load(config_path)
        return params
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None


def custom_yaml_dump(data, indent=0):
    """
    Custom YAML dump function to print a dictionary as YAML,
    with lists inline if they are encountered.

    Parameters
    ----------

    data : dict
        The dictionary to print as YAML.

    indent : int
        The current(initial) indentation level. Default is 0.

    Returns
    -------

    yaml_str: str
        The YAML string representation of the dictionary.
    """

    yaml_str = ""
    indent_str = "  " * indent  # Two spaces for each indentation level

    for key, value in data.items():
        if isinstance(value, dict) or isinstance(value, DictConfig):
            # Check if it's an empty dict
            if not value:
                yaml_str += f"{indent_str}{key}: {{}}\n"
            else:
                # Recursively process dictionaries with increased indentation
                yaml_str += f"{indent_str}{key}:\n"
                yaml_str += custom_yaml_dump(value, indent + 1)
        elif isinstance(value, list):
            # Special handling for empty lists
            if not value:
                yaml_str += f"{indent_str}{key}: []\n"
            # For short lists, write them inline with brackets
            elif (
                all(not isinstance(x, (dict, list)) for x in value)
                and len(str(value)) < 60
            ):
                # Remove spaces after commas for consistency
                list_str = str(value).replace(", ", ",")
                yaml_str += f"{indent_str}{key}: {list_str}\n"
            # For longer or complex lists, write each item on a new line
            else:
                yaml_str += f"{indent_str}{key}:\n"
                for item in value:
                    if isinstance(item, (dict, DictConfig)):
                        yaml_str += f"{indent_str}  -\n"
                        yaml_str += custom_yaml_dump(item, indent + 2)
                    else:
                        yaml_str += f"{indent_str}  - {item}\n"
        elif isinstance(value, bool):
            # Output booleans as lowercase 'true' or 'false' for consistency
            yaml_str += f"{indent_str}{key}: {str(value).lower()}\n"
        elif value is None:
            # Handle None values
            yaml_str += f"{indent_str}{key}:\n"
        elif isinstance(value, str):
            # Add quotes around strings that might need them
            if any(c in value for c in ",:{}[]#&*!|>'\"%@`") or value.strip() != value:
                yaml_str += f'{indent_str}{key}: "{value}"\n'
            else:
                yaml_str += f"{indent_str}{key}: {value}\n"
        else:
            # Write scalar values directly
            yaml_str += f"{indent_str}{key}: {value}\n"

    return yaml_str


def validate_params(params):
    """
    Validates the presence of required configurations in the parameters.

    Parameters:
    params (dict): A dictionary containing the configuration parameters.

    Raises:
    ValueError: If any of the required configurations are missing from the parameters.
    """
    required_configs = ["run_config", "train_config", "sweep_config"]
    for config in required_configs:
        if config not in params:
            raise ValueError(f"Missing required config: {config}")


def set_nested_dict_value(d, key_path, value):
    """
    Sets a value in a nested dictionary at the specified key path.

    Parameters
    ----------
    d : dict
        The dictionary to modify
    key_path : str
        The dot-separated path to the key
    value : any
        The value to set
    """
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def generate_config_files(
    base_config,
    sweep_params,
    output_dir="config_files",
    seeds_per_condition=1,
    base_seed=42,
):
    """
    Generates configuration files for each combination of sweep parameters.

    Parameters
    ----------
    base_config : dict
        The base configuration dictionary.

    sweep_params : dict
        The dictionary containing the sweep parameters.

    output_dir : str
        The directory to save the configuration files.
        Default is "config_files".

    seeds_per_condition : int
        Number of random seeds to use per parameter combination.

    base_seed : int
        Base seed value to start from.
    """
    print("[INFO] Generating configuration files...")
    os.makedirs(output_dir, exist_ok=True)

    # Check if seeds_per_condition is a sweep parameter
    seeds_param = None
    if "seeds_per_condition" in sweep_params:
        seeds_param = sweep_params.pop("seeds_per_condition")
        print(f"[INFO] Seeds per condition will be swept: {seeds_param}")

    sweep_keys = list(sweep_params.keys())
    sweep_values = [params for params in sweep_params.values()]

    # Create all possible parameter combinations
    param_combinations = list(product(*sweep_values))
    print(f"[INFO] Generated {len(param_combinations)} parameter combinations")

    configs = []
    sweeps = []

    # If seeds_per_condition is a sweep parameter, we'll iterate through different values
    seeds_list = seeds_param if seeds_param else [seeds_per_condition]

    for seeds_value in seeds_list:
        print(f"[INFO] Processing with {seeds_value} seeds per condition")

        for param_idx, combination in enumerate(param_combinations):
            for seed_idx in range(seeds_value):
                seed_value = base_seed + seed_idx
                new_config = deepcopy(base_config)

                # Set the seed value
                new_config["train"]["seed"] = seed_value

                # Apply the parameter combination
                param_details = {
                    "seed": seed_value,
                }

                if seeds_param:
                    param_details["seeds_per_condition"] = seeds_value

                for key, value in zip(sweep_keys, combination):
                    try:
                        # Handle setting nested dictionary values
                        set_nested_dict_value(new_config, key, value)
                        param_details[key] = value
                    except Exception as e:
                        print(f"[ERROR] Failed to set {key} to {value}: {e}")

                containerized_config = OmegaConf.to_container(new_config, resolve=False)
                configs.append(containerized_config)
                sweeps.append(param_details)

    print(f"[INFO] Successfully generated {len(configs)} configuration files")
    return configs, sweeps


def write_config_files(configs, sweeps, output_dir="configs", base_name="config"):
    """
    Writes the configuration files to disk.

    Parameters
    ----------

    configs : list
        A list of configuration dictionaries.

    sweeps : list
        A list of dictionaries containing the sweep parameters.

    output_dir : str
        The directory to save the configuration files.
        Default is "configs".

    base_name : str
        The base name for the configuration files. Default is "config".
    """

    assert len(configs) == len(sweeps), "Number of configs and sweeps do not match."

    # Create config directory within output_dir
    config_dir = os.path.join(output_dir, "configs")
    os.makedirs(config_dir, exist_ok=True)

    sweep_lines = []
    metadata_dict = {}

    for i, config in enumerate(configs):
        config_name = f"{base_name}_{i}.yaml"
        config_path = os.path.join(config_dir, config_name)
        config_yaml = custom_yaml_dump(config)
        with open(config_path, "w") as f:
            f.write(config_yaml)

        sweep_str = ", ".join([f"{k}={v}" for k, v in sweeps[i].items()])
        sweep_lines.append(f"{config_name}: {sweep_str}\n")
        metadata_dict[config_name] = sweeps[i]

        print(f"[INFO] Configuration saved to {config_path}")

    # Write sweep parameters to text file for reference
    sweep_path = os.path.join(config_dir, f"{base_name}_sweeps.txt")
    with open(sweep_path, "w") as f:
        f.writelines(sweep_lines)

    # Write metadata as YAML for easier parsing by analysis script
    metadata_path = os.path.join(config_dir, f"{base_name}_metadata.yaml")
    metadata_yaml = OmegaConf.create(metadata_dict)
    OmegaConf.save(metadata_yaml, metadata_path)

    print(f"[INFO] Sweep metadata saved to {metadata_path}")

    # Return the config directory path for use by the run script
    return config_dir


def generate_run_sh(
    num_config, config_dir, config_filename_init, run_config, output_dir
):
    """
    Generates a SLURM batch script for running jobs based on the given
    configurations.

    Parameters
    ----------

    num_config: int
        The number of generated configurations.

    config_dir: str
        The directory where the configuration files are located.

    config_file_name_init: str
        The initial name of the configuration files.

    run_config: dict
        A dictionary containing SLURM configuration options.

    output_dir: str
        The main output directory where results will be saved.
    """

    array_job_flag = num_config > 1

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={run_config['run_name']}",
        f"#SBATCH --account={run_config['account']}",
    ]

    if array_job_flag:
        lines.extend(
            [
                f"#SBATCH --output={output_dir}/{run_config['run_name']}_%A/%A_%a/output_%A_%a.out",
                f"#SBATCH --error={output_dir}/{run_config['run_name']}_%A/%A_%a/error_%A_%a.out",
            ]
        )
    else:
        lines.extend(
            [
                f"#SBATCH --output={output_dir}/{run_config['run_name']}_%j/output_%j.out",
                f"#SBATCH --error={output_dir}/{run_config['run_name']}_%j/error_%j.out",
            ]
        )

    # Ensure time is in HH:MM:SS format
    time_value = run_config["time"]
    # If time value is just a number (seconds), convert to HH:MM:SS
    if isinstance(time_value, (int, float)) or (
        isinstance(time_value, str) and time_value.isdigit()
    ):
        seconds = int(time_value)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        formatted_time = f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        # Already in string format, ensure it has colons
        if ":" not in time_value:
            # Assume it's hours if no colons
            formatted_time = f"{time_value}:00:00"
        else:
            formatted_time = time_value

    lines.extend(
        [
            f"#SBATCH --nodes={run_config['nodes']}",
            f"#SBATCH --ntasks-per-node={run_config['ntasks_per_node']}",
            f"#SBATCH --gpus-per-node={run_config['gpus_per_node']}",
            f"#SBATCH --cpus-per-task={run_config['cpus_per_task']}",
            f"#SBATCH --time={formatted_time}",
            f"#SBATCH --mem={run_config['mem']}",
            f"#SBATCH --partition={run_config['partition']}",
        ]
    )

    # Submit an array job or a single job
    if array_job_flag:
        if num_config > run_config["max_concurrent_jobs"]:
            lines.append(
                f"#SBATCH --array=0-{num_config-1}%{run_config['max_concurrent_jobs']}"
            )
        else:
            lines.append(f"#SBATCH --array=0-{num_config-1}")

    lines.append("")

    lines.append("# ================================================================ ")
    lines.append("# This file has been generated by generate_sweep_configs.py script ")
    lines.append("#                            ***                                   ")
    lines.append("#       Manual changes to this file may be overwritten.            ")
    lines.append("# ================================================================ ")

    # Add module loading lines
    lines.append("module purge")
    for module in run_config["modules_to_load"]:
        lines.append(f"module load {module}")

    lines.append("")
    lines.append(f"conda activate {run_config['conda_env_path']}")
    lines.append("")
    lines.append("# Path to the config file directory")
    lines.append(f"CONFIG_DIR={config_dir}")
    lines.append("")
    lines.append(f"# Path to output directory")
    lines.append(f"OUTPUT_DIR={output_dir}")
    lines.append("")
    lines.append(f"CONFIG_FILE_NAME_INIT={config_filename_init}")
    lines.append(f"SC_RUN_NAME={run_config['run_name']}")
    lines.append("")
    lines.append("#Training script: ")
    lines.append(f"TRAINING_SCRIPT={run_config['training_script']}")
    lines.append("")
    lines.append("#Python path: ")
    lines.append(f"PYTHON={run_config['conda_env_path']}/bin/python")
    lines.append("")
    lines.append("# Get the specific config file based on the array job ID")
    if array_job_flag:
        lines.append(
            "CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_${SLURM_ARRAY_TASK_ID}.yaml"
        )
        lines.append(
            "RUN_NAME=${SC_RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
        )
    else:
        lines.append("CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_0.yaml")
        lines.append("RUN_NAME=${SC_RUN_NAME}_${SLURM_JOB_ID}")
    lines.append("")
    lines.append(f"# Path to the folder to save outputs")
    if array_job_flag:
        lines.append(
            f"OUTPUT_FOLDER_PATH=${{OUTPUT_DIR}}/${{SC_RUN_NAME}}_${{SLURM_ARRAY_JOB_ID}}/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
        )
    else:
        lines.append(
            f"OUTPUT_FOLDER_PATH=${{OUTPUT_DIR}}/${{SC_RUN_NAME}}_${{SLURM_JOB_ID}}"
        )
    lines.append("")
    lines.append("# Create output directory")
    lines.append("mkdir -p ${OUTPUT_FOLDER_PATH}")
    lines.append("")
    lines.append('echo "Starting running a training job at $(date)"')
    lines.append("start_time=$(date +%s)")
    lines.append("srun \\")
    lines.append("  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\")
    lines.append("  --kill-on-bad-exit \\")

    # Fix the training script path - use the full path to the script
    # And only pass the config file as an argument, as that's all the script expects
    if run_config["training_script"].startswith("/"):
        # If it's already an absolute path, use it as is
        lines.append("  ${PYTHON} -u ${TRAINING_SCRIPT} ${CONFIG}")
    else:
        # If it's a relative path, prepend the current directory
        lines.append("  ${PYTHON} -u $(pwd)/${TRAINING_SCRIPT} ${CONFIG}")

    lines.append("end_time=$(date +%s)")
    lines.append('echo "Done with running a training job at $(date)"')
    lines.append('echo "Total duration: $((end_time - start_time)) seconds."')

    # Save the run.sh script to the output directory
    run_script_path = os.path.join(output_dir, "run.sh")
    with open(run_script_path, "w") as file:
        for line in lines:
            file.write(line + "\n")

    # Also create a symlink in the current directory for convenience
    try:
        if os.path.exists("run.sh"):
            os.remove("run.sh")
        os.symlink(run_script_path, "run.sh")
    except Exception as e:
        print(f"[WARNING] Could not create symlink to run.sh: {e}")
        # If symlink fails, create a copy
        with open("run.sh", "w") as file:
            for line in lines:
                file.write(line + "\n")

    print(f"[INFO] run.sh file has been generated at {run_script_path}")
    print(f"[INFO] A copy/link has been created in the current directory")


def parse_args():
    """Parse command line arguments, supporting both positional and flag formats for backward compatibility."""

    parser = argparse.ArgumentParser(
        description="Generate configuration files for a sweep."
    )

    # Support two ways of providing arguments:
    # 1. Positional arguments (old style)
    # 2. Flag-based arguments (new style)

    # Add positional arguments first
    parser.add_argument(
        "config_dir_pos",
        nargs="?",  # Make it optional
        default=None,
        help=argparse.SUPPRESS,  # Hide from help
    )

    parser.add_argument(
        "param_filename_pos",
        nargs="?",  # Make it optional
        default=None,
        help=argparse.SUPPRESS,  # Hide from help
    )

    # Add flag-based arguments
    parser.add_argument(
        "--config_dir",
        default="configs",
        type=str,
        help="The directory containing the configuration file.",
    )

    parser.add_argument(
        "--param_filename",
        default="sweep_params.yaml",
        type=str,
        help="The name of the sweep parameters file.",
    )

    args = parser.parse_args()

    # If positional arguments were provided, use them instead of the flag-based ones
    if args.config_dir_pos is not None:
        args.config_dir = args.config_dir_pos

    if args.param_filename_pos is not None:
        args.param_filename = args.param_filename_pos

    return args


if __name__ == "__main__":
    args = parse_args()
    config_dir = args.config_dir
    param_filename = args.param_filename

    print(f"[INFO] Reading config from {config_dir}/{param_filename}...")
    # Read the config file
    params = read_config(config_dir, param_filename)
    if params is None:
        print(f"[ERROR] Failed to read config file")
        exit(1)

    print("[INFO] Validating parameters...")
    validate_params(params)

    # Extract settings
    base_config = params["train_config"]
    sweep_params = params["sweep_config"]
    run_config = params["run_config"]

    # Get experiment settings if available
    experiment_settings = params.get("experiment_settings", {})
    seeds_per_condition = experiment_settings.get("seeds_per_condition", 1)
    base_seed = experiment_settings.get("base_seed", 42)

    # Get output directory - use the one specified in run_config.dir if available
    output_dir = base_config.get("outputs", {}).get("dir", "outputs")

    # Make sure run_config and train_config use the same output directory
    run_config.dir = output_dir
    if "outputs" not in base_config:
        base_config["outputs"] = {}
    base_config["outputs"]["dir"] = output_dir

    # Set run name for run_config
    run_config.run_name = params.train_config.train.run_name
    config_filename_init = run_config["config_filename_init"]

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Using output directory: {output_dir}")

    # Generate configuration files
    configs, sweeps = generate_config_files(
        base_config,
        sweep_params,
        output_dir=output_dir,
        seeds_per_condition=seeds_per_condition,
        base_seed=base_seed,
    )

    # Write the configuration files to disk in a subdirectory of the output directory
    config_subdir = write_config_files(
        configs, sweeps, output_dir=output_dir, base_name=config_filename_init
    )

    # Generate a SLURM batch script for running jobs
    generate_run_sh(
        len(configs),
        config_dir=config_subdir,
        config_filename_init=config_filename_init,
        run_config=run_config,
        output_dir=output_dir,
    )

    # Create a README file with information about the sweep
    readme_path = os.path.join(output_dir, "SWEEP_README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Sweep Configuration: {params.train_config.train.run_name}\n\n")
        f.write(f"Generated on: {os.popen('date').read().strip()}\n\n")
        f.write(f"## Contents\n\n")
        f.write(f"- configs/: Configuration files for each run\n")
        f.write(f"- run.sh: SLURM batch script to run the sweep\n\n")
        f.write(f"## Parameters\n\n")
        f.write(f"Base seed: {base_seed}\n")
        f.write(f"Seeds per condition: {seeds_per_condition}\n")
        f.write(f"Total configurations: {len(configs)}\n\n")
        f.write(f"## Command to run\n\n")
        f.write(f"```bash\nsbatch {output_dir}/run.sh\n```\n")

    print(f"[INFO] Sweep README file created at {readme_path}")
    print(
        f"[INFO] Successfully generated {len(configs)} configuration files and run script"
    )
    print(f"[INFO] To run the sweep, execute: sbatch {output_dir}/run.sh")
    print(f"[INFO] (A copy/link of run.sh is also available in the current directory)")
