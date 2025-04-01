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
    ValueError: If any of the required configurations ('slurm_config',
    'train_config') are missing from the parameters.

    Example:
    try:
        validate_params(params)
    except ValueError as e:
        print(e)
    """
    required_configs = ["slurm_config", "train_config"]
    for config in required_configs:
        if config not in params:
            raise ValueError(f"Missing required config: {config}")


def flatten(config, parent_key="", sep="."):
    """
    Flattens a nested dictionary or OmegaConf DictConfig into a single level
    dictionary with keys as the path to each value.

    Parameters
    ----------

    config : dict or DictConfig
        The nested dictionary or OmegaConf DictConfig to flatten.

    present_key : str
        The base key string for the nested keys. Default is an empty string.

    sep : str
        The separator between keys. Default is '.'.
    """

    if config is None:
        return {}

    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, DictConfig):
            items.extend(flatten(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)


def generate_config_files(base_config, sweep_params, output_dir="config_files"):
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

    """
    print("[DEBUG inside generate_config_files] Function started.")
    os.makedirs(output_dir, exist_ok=True)

    sweep_keys = list(sweep_params.keys())
    sweep_values = [params for params in sweep_params.values()]

    # If nested lists, create all possible combinations
    print(
        f"[DEBUG inside generate_config_files] Generating product of sweep values: {sweep_values}"
    )
    all_combinations = list(product(*sweep_values))
    print(
        f"[DEBUG inside generate_config_files] Generated {len(all_combinations)} combinations."
    )

    configs = []
    sweeps = []

    print(
        f"[DEBUG inside generate_config_files] Starting loop over {len(all_combinations)} combinations..."
    )
    for i, combination in enumerate(all_combinations):
        print(
            f"[DEBUG inside generate_config_files] --- Processing combination {i} --- "
        )
        print(
            f"[DEBUG inside generate_config_files] Combination {i}: Starting deepcopy..."
        )
        new_config = deepcopy(base_config)
        print(
            f"[DEBUG inside generate_config_files] Combination {i}: Deepcopy finished."
        )
        param_details = {}

        print(
            f"[DEBUG inside generate_config_files] Combination {i}: Applying sweep params..."
        )
        for key, value in zip(sweep_keys, combination):
            # Manually handle the specific sweep parameters
            try:
                if key == "train.learning_strategy":
                    print(
                        f"[DEBUG inside generate_config_files] Setting train.learning_strategy to {value}"
                    )
                    new_config["train"]["learning_strategy"] = value
                    param_details[key] = value

                elif key == "model.network.parameters.topk_noise_level":
                    print(
                        f"[DEBUG inside generate_config_files] Setting model.network.parameters.topk_noise_level to {value}"
                    )
                    new_config["model"]["network"]["parameters"][
                        "topk_noise_level"
                    ] = value
                    param_details[key] = value

                else:
                    print(f"[WARNING] Unknown sweep parameter: {key}")
            except Exception as e:
                print(f"[ERROR] Failed to set {key} to {value}: {e}")

        print(
            f"[DEBUG inside generate_config_files] Combination {i}: Sweep params applied."
        )

        print(
            f"[DEBUG inside generate_config_files] Combination {i}: Containerizing config..."
        )
        containerized_config = OmegaConf.to_container(new_config, resolve=False)
        print(
            f"[DEBUG inside generate_config_files] Combination {i}: Config containerized."
        )
        configs.append(containerized_config)
        sweeps.append(param_details)
    print(f"[DEBUG inside generate_config_files] Finished loop.")

    print(
        f"[DEBUG inside generate_config_files] Returning {len(configs)} configs and {len(sweeps)} sweeps."
    )
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

    os.makedirs(output_dir, exist_ok=True)

    sweep_lines = []

    for i, config in enumerate(configs):
        config_name = f"{base_name}_{i}.yaml"
        config_path = os.path.join(output_dir, config_name)
        config_yaml = custom_yaml_dump(config)
        with open(config_path, "w") as f:
            f.write(config_yaml)
        sweep_lines.append(f"{config_name}: {sweeps[i]} \n")
        print(f"Configuration saved to {config_path}")

    sweep_path = os.path.join(output_dir, f"{base_name}_sweeps.txt")
    sweep_lines = "".join(sweep_lines)
    with open(sweep_path, "w") as f:
        f.writelines(sweep_lines)


def generate_run_sh(num_config, config_dir, config_filename_init, slurm_config):
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

    slurm_config: dict
        A dictionary containing SLURM configuration options.
    """

    array_job_flag = False
    if num_config > 1:
        array_job_flag = True

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm_config['run_name']}",
        f"#SBATCH --account={slurm_config['account']}",
    ]

    if array_job_flag:
        lines.extend(
            [
                f"#SBATCH --output={slurm_config['output_dir']}/{slurm_config['run_name']}_%A/%A_%a/output_%A_%a.out",
                f"#SBATCH --error={slurm_config['output_dir']}/{slurm_config['run_name']}_%A/%A_%a/error_%A_%a.out",
            ]
        )
    else:
        lines.extend(
            [
                f"#SBATCH --output={slurm_config['output_dir']}/{slurm_config['run_name']}_%j/output_%j.out",
                f"#SBATCH --error={slurm_config['output_dir']}/{slurm_config['run_name']}_%j/error_%j.out",
            ]
        )

    # Ensure time is in HH:MM:SS format
    time_value = slurm_config["time"]
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
            f"#SBATCH --nodes={slurm_config['nodes']}",
            f"#SBATCH --ntasks-per-node={slurm_config['ntasks_per_node']}",
            f"#SBATCH --gpus-per-node={slurm_config['gpus_per_node']}",
            f"#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}",
            f"#SBATCH --time={formatted_time}",
            f"#SBATCH --mem={slurm_config['mem']}",
            f"#SBATCH --partition={slurm_config['partition']}",
        ]
    )

    # Submit an array job or a single job
    if array_job_flag:
        if num_config > slurm_config["max_concurrent_jobs"]:
            lines.append(
                f"#SBATCH --array=0-{num_config-1}%{slurm_config['max_concurrent_jobs']}"
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
    for module in slurm_config["modules_to_load"]:
        lines.append(f"module load {module}")

    lines.append("")
    lines.append(f"conda activate {slurm_config['conda_env_path']}")
    lines.append("")
    lines.append("# Path to the config file directory")
    lines.append(f"CONFIG_DIR={config_dir}")
    lines.append("")
    lines.append(f"# Path to output directory")
    lines.append(f"OUTPUT_DIR={slurm_config['output_dir']}")
    lines.append("")
    lines.append(f"CONFIG_FILE_NAME_INIT={config_filename_init}")
    lines.append(f"SC_RUN_NAME={slurm_config['run_name']}")
    lines.append("")
    lines.append("#Training script: ")
    lines.append(f"TRAINING_SCRIPT={slurm_config['training_script']}")
    lines.append("")
    lines.append("#Python path: ")
    lines.append(f"PYTHON={slurm_config['conda_env_path']}/bin/python")
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
            f"OUTPUT_FOLDER_PATH={slurm_config['output_dir']}/${{SC_RUN_NAME}}_${{SLURM_ARRAY_JOB_ID}}/${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
        )
    else:
        lines.append(
            f"OUTPUT_FOLDER_PATH={slurm_config['output_dir']}/${{SC_RUN_NAME}}_${{SLURM_JOB_ID}}"
        )
    lines.append("")
    lines.append('echo "Starting running a training job at $(date)"')
    lines.append("start_time=$(date +%s)")
    lines.append("srun \\")
    lines.append("  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\")
    lines.append("  --kill-on-bad-exit \\")

    # Fix the training script path - use the full path to the script
    # And only pass the config file as an argument, as that's all the script expects
    if slurm_config["training_script"].startswith("/"):
        # If it's already an absolute path, use it as is
        lines.append("  ${PYTHON} -u ${TRAINING_SCRIPT} ${CONFIG}")
    else:
        # If it's a relative path, prepend the current directory
        lines.append("  ${PYTHON} -u $(pwd)/${TRAINING_SCRIPT} ${CONFIG}")

    lines.append("end_time=$(date +%s)")
    lines.append('echo "Done with running a training job at $(date)"')
    lines.append('echo "Total duration: $((end_time - start_time)) seconds."')

    with open("run.sh", "w") as file:
        for line in lines:
            file.write(line + "\n")

    print("run.sh file has been generated.")


def parse_args():
    """ """

    parser = argparse.ArgumentParser(
        description="Generate configuration files for a sweep."
    )
    parser.add_argument(
        "config_dir",
        default="configs",
        type=str,
        help="The directory containing the configuration file.",
    )

    parser.add_argument(
        "param_filename",
        default="sweep_params.yaml",
        type=str,
        help="The name of the sweep parameters file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config_dir = args.config_dir
    param_filename = args.param_filename

    print(f"[DEBUG] Reading config from {config_dir}/{param_filename}...")
    # Read the config file
    params = read_config(config_dir, param_filename)
    print("[DEBUG] Config read successfully.")

    # Validate the parameters
    print("[DEBUG] Validating parameters...")
    validate_params(params)
    print("[DEBUG] Parameters validated.")

    params.slurm_config.run_name = params.train_config.train.run_name
    print(f"[DEBUG] Set slurm_config.run_name to {params.slurm_config.run_name}")

    # Prepare for generation
    base_config = params["train_config"]
    sweep_params = params["sweep_config"]
    slurm_config = params["slurm_config"]
    config_filename_init = slurm_config["config_filename_init"]

    # SIMPLIFIED APPROACH FOR SPECIFIC SWEEP PARAMETERS
    # Replace the complex flattening with a direct dictionary for the key sweep parameters
    print("[DEBUG] Creating direct sweep parameter dictionary...")
    manual_sweep_params = {}
    if "train.learning_strategy" in sweep_params:
        manual_sweep_params["train.learning_strategy"] = sweep_params[
            "train.learning_strategy"
        ]
    if "model.network.parameters.topk_noise_level" in sweep_params:
        manual_sweep_params["model.network.parameters.topk_noise_level"] = sweep_params[
            "model.network.parameters.topk_noise_level"
        ]
    if "model.network.parameters.topk_strategy" in sweep_params:
        manual_sweep_params["model.network.parameters.topk_strategy"] = sweep_params[
            "model.network.parameters.topk_strategy"
        ]
    print(f"[DEBUG] Direct sweep parameters = {manual_sweep_params}")

    # Generate configuration files for each combination of sweep parameters
    print("[DEBUG] Generating config combinations...")
    configs, sweeps = generate_config_files(
        base_config, manual_sweep_params, output_dir=config_dir
    )
    print(f"[DEBUG] Generated {len(configs)} config combinations.")

    # Write the configuration files to disk
    print("[DEBUG] Writing config files...")
    write_config_files(
        configs, sweeps, output_dir=config_dir, base_name=config_filename_init
    )
    print("[DEBUG] Finished writing config files.")

    # Generate a SLURM batch script for running jobs
    print("[DEBUG] Generating run.sh script...")
    generate_run_sh(
        len(configs),
        config_dir=config_dir,
        config_filename_init=config_filename_init,
        slurm_config=slurm_config,
    )
    print("[DEBUG] Finished generating run.sh script.")
