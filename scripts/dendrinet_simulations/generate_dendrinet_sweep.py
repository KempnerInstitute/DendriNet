#!/usr/bin/env python
"""
generate_dendrinet_sweep.py
---------------------------
Script to generate configuration files for dendrinet simulation array jobs.

This script:
1. Takes a YAML configuration file as input
2. Generates individual config files for each parameter combination
3. Creates SLURM job scripts to submit as array jobs
4. Organizes output in a structured directory format

Usage:
    python scripts/generate_dendrinet_sweep.py configs/dendrinet_simulation_config.yaml
    python scripts/generate_dendrinet_sweep.py configs/dendrinet_simulation_config.yaml --split-jobs 5
    python scripts/generate_dendrinet_sweep.py configs/dendrinet_simulation_config.yaml --configs-per-job 20
"""

import argparse
import datetime
import itertools
import os
import re
import shutil
from pathlib import Path

import numpy as np

# Try to import yaml first, then fall back to omegaconf if yaml is not available
try:
    import yaml

    USE_OMEGACONF = False
    print("Using PyYAML for configuration handling")
except ImportError:
    try:
        from omegaconf import OmegaConf

        USE_OMEGACONF = True
        print("Using OmegaConf for configuration handling")
    except ImportError:
        raise ImportError(
            "Neither PyYAML nor OmegaConf is installed. Please install one of them."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate configurations for dendrinet simulation array jobs"
    )
    parser.add_argument("config_file", help="Path to the simulation configuration file")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--split-jobs",
        type=int,
        help="Number of job scripts to generate (splits configs evenly)",
    )
    group.add_argument(
        "--configs-per-job",
        type=int,
        help="Maximum number of configurations per job script",
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Print configurations without creating files",
    )
    return parser.parse_args()


def load_config(config_file):
    """Load YAML configuration file using either PyYAML or OmegaConf"""
    if USE_OMEGACONF:
        config = OmegaConf.load(config_file)
        # Convert OmegaConf object to dict for easier handling
        config = OmegaConf.to_container(config, resolve=True)
    else:
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
    return config


def save_config(config, file_path):
    """Save configuration to file using either PyYAML or OmegaConf"""
    if USE_OMEGACONF:
        # Convert dict to OmegaConf object if needed
        if not hasattr(config, "_content"):
            config_obj = OmegaConf.create(config)
        else:
            config_obj = config
        OmegaConf.save(config_obj, file_path)
    else:
        with open(file_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


def expand_ranges(param_value):
    """
    Handle parameter values that are specified as ranges.
    A range is specified as [start, end, step].
    """
    if (
        isinstance(param_value, list)
        and len(param_value) == 3
        and all(isinstance(x, (int, float)) for x in param_value)
    ):
        start, end, step = param_value
        if isinstance(start, int) and isinstance(end, int) and isinstance(step, int):
            return list(range(start, end + 1, step))
        else:
            return np.arange(start, end + step / 2, step).tolist()
    return param_value


def generate_parameter_combinations(config):
    """
    Generate parameter combinations only for main sweep parameters (N_E, N_I, rho).
    Correlation combinations and group combinations will be handled within each job.
    Returns a list of dictionaries, each containing a specific parameter combination.
    """
    sweep_config = config["sweep_config"]
    base_params = config["simulation_params"]
    seeds_per_condition = config["experiment_settings"]["seeds_per_condition"]
    base_seed = config["experiment_settings"]["base_seed"]

    # Extract and expand ranges for single-value parameters (only N_E, N_I, rho)
    param_values = {}
    for key, value in sweep_config.items():
        expanded_value = expand_ranges(value)
        if not isinstance(expanded_value, list):
            expanded_value = [expanded_value]
        param_values[key] = expanded_value

    # Generate the cartesian product of parameter values
    param_keys = list(param_values.keys())
    param_value_lists = [param_values[key] for key in param_keys]

    # Get correlation and group combinations from simulation_params
    correlation_combos = base_params.get("correlation_combos", [[0.0, 0.0, 0.0]])
    group_combos = base_params.get("group_combos", [[0, 0]])

    # Create list to hold all combinations (one for each main parameter combination)
    all_combinations = []

    # Generate all combinations of main parameters (N_E, N_I, rho)
    for values in itertools.product(*param_value_lists):
        param_dict = {key: values[i] for i, key in enumerate(param_keys)}

        # Start with base parameters
        combo = base_params.copy()

        # Update with sweep parameters
        combo.update(param_dict)

        # Add seeds, correlation combos, and group combos as lists
        # These will be processed within each job
        combo.update(
            {
                "seeds": list(range(base_seed, base_seed + seeds_per_condition)),
                # Note: correlation_combos and group_combos are already in the base_params, so we don't need to add them here
            }
        )

        all_combinations.append(combo)

    return all_combinations


def create_timestamp_directory(config):
    """
    Create a timestamped directory for this run.
    Returns the path to the created directory.
    """
    slurm_config = config["slurm_config"]
    output_dir = slurm_config["output_dir"]
    subdir_prefix = slurm_config.get("subdir_prefix", "dendrinet_")

    # Create timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Create directory path
    timestamped_dir = os.path.join(output_dir, f"{subdir_prefix}{timestamp}")

    # Create main directory
    os.makedirs(timestamped_dir, exist_ok=True)

    # Create subdirectories
    for subdir in slurm_config.get("run_subfolders", []):
        os.makedirs(os.path.join(timestamped_dir, subdir), exist_ok=True)

    return timestamped_dir, timestamp


def generate_config_files(
    config, combinations, output_dir, config_filename_init, dryrun=False
):
    """
    Generate individual configuration files for each parameter combination.
    Returns a list of paths to the generated config files.
    """
    configs_dir = os.path.join(output_dir, "configs")
    config_files = []

    if dryrun:
        print(f"Would generate {len(combinations)} config files in {configs_dir}/")

        # In dryrun mode, create one sample config file to demonstrate format
        if len(combinations) > 0:
            sample_file = os.path.join(
                configs_dir, f"{config_filename_init}_sample.yaml"
            )
            save_config(combinations[0], sample_file)
            print(f"Created a sample config file for reference: {sample_file}")

        # Return the list of files that would be created
        return [
            f"{configs_dir}/{config_filename_init}_{i}.yaml"
            for i in range(len(combinations))
        ]

    # Normal mode: create all config files
    for i, combo in enumerate(combinations):
        config_file = os.path.join(configs_dir, f"{config_filename_init}_{i}.yaml")
        save_config(combo, config_file)
        config_files.append(config_file)

    return config_files


def generate_slurm_script(
    config,
    output_dir,
    num_configs,
    max_concurrent_jobs,
    script_name="run_dendrinet_sweep.sh",
    start_idx=0,
    end_idx=None,
    dryrun=False,
):
    """
    Generate a SLURM job script for running the simulation.
    """
    slurm_config = config["slurm_config"]

    # Set up array range
    if end_idx is None:
        end_idx = num_configs - 1
    array_range = f"{start_idx}-{end_idx}%{max_concurrent_jobs}"

    # Handle the simulation script path
    # If it's a relative path, make it relative to the current working directory (not the output dir)
    simulation_script = slurm_config["simulation_script"]
    if not os.path.isabs(simulation_script):
        # Get current working directory
        current_dir = os.getcwd()
        # Create absolute path
        simulation_script = os.path.abspath(
            os.path.join(current_dir, simulation_script)
        )

    # Create script content
    script_content = f"""#!/bin/bash
#SBATCH --job-name={slurm_config['run_name']}
#SBATCH --account={slurm_config['account']}
#SBATCH --output={output_dir}/results/%A/task_%a/output.log
#SBATCH --error={output_dir}/results/%A/task_%a/error.log
#SBATCH --nodes={slurm_config['nodes']}
#SBATCH --ntasks-per-node={slurm_config['ntasks_per_node']}
#SBATCH --gpus-per-node={slurm_config['gpus_per_node']}
#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}
#SBATCH --time={slurm_config['time']}  # Format: HH:MM:SS or D-HH:MM:SS
#SBATCH --mem={slurm_config['mem']}
#SBATCH --partition={slurm_config['partition']}
#SBATCH --array={array_range}

# ================================================================
# This file has been generated by generate_dendrinet_sweep.py script
#                            ***
#       Manual changes to this file may be overwritten.
# ================================================================
module purge
"""

    # Add module loading commands
    for module in slurm_config.get("modules_to_load", []):
        script_content += f"module load {module}\n"

    # Add configuration and output paths
    script_content += f"""
# Path to the config file directory
CONFIG_DIR={output_dir}/configs

# Path to main output directory
MAIN_OUTPUT_DIR={output_dir}

# Subdirectories
RESULTS_DIR=${{MAIN_OUTPUT_DIR}}/results
PLOTS_DIR=${{MAIN_OUTPUT_DIR}}/plots

CONFIG_FILE_NAME_INIT={slurm_config['config_filename_init']}
SC_RUN_NAME={slurm_config['run_name']}

# Get task IDs
TASK_ID=${{SLURM_ARRAY_TASK_ID}}
JOB_ID=${{SLURM_ARRAY_JOB_ID}}

# Path to the main job results folder
JOB_FOLDER_PATH=${{RESULTS_DIR}}/${{JOB_ID}}
TASK_FOLDER_PATH=${{JOB_FOLDER_PATH}}/task_${{TASK_ID}}

# Create output directories immediately
echo "Creating output directories..."
mkdir -p ${{JOB_FOLDER_PATH}}/task_${{TASK_ID}}
mkdir -p ${{PLOTS_DIR}}/${{JOB_ID}}/task_${{TASK_ID}}
"""

    # Add conda environment activation if specified
    if "conda_env_path" in slurm_config and slurm_config["conda_env_path"]:
        script_content += f"""
# Set up conda environment
echo "Setting up conda environment..."
export CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate {slurm_config['conda_env_path']}

# Verify conda environment is activated
ENV_NAME=$(conda info --envs | grep '*' | awk '{{print $1}}')
echo "Activated conda environment: $ENV_NAME"

# Ensure user site-packages is in the Python path
export PYTHONPATH=$HOME/.local/lib/python3.12/site-packages:$PYTHONPATH

# List installed packages for debugging
echo "Listing conda packages (first 10):"
conda list | head -10

# Verify required packages are available
echo "Verifying required packages in conda environment..."

# Check for configuration packages (PyYAML or OmegaConf)
CONFIG_OK=false

# Check python version
echo "Python version:"
python --version

# Check Python path
echo "Python path:"
python -c "import sys; print('\\n'.join(sys.path))"

# Check for PyYAML
echo "Checking for PyYAML..."
python -c "try:
    import yaml
    print('PyYAML is available:', yaml.__file__)
    exit(0)
except ImportError as e:
    print('PyYAML import error:', e)
    exit(1)
"
if [ $? -eq 0 ]; then
    CONFIG_OK=true
else
    # Try OmegaConf as alternative
    echo "Checking for OmegaConf..."
    python -c "try:
    import sys
    print('Python sys.path:')
    for p in sys.path:
        print(' -', p)

    from omegaconf import OmegaConf
    print('OmegaConf is available:', OmegaConf)
    print('OmegaConf file location:', OmegaConf.__file__)
    exit(0)
except ImportError as e:
    print('OmegaConf import error:', e)
    exit(1)
"
    if [ $? -eq 0 ]; then
        CONFIG_OK=true
    else
        echo "ERROR: Neither PyYAML nor OmegaConf is available"
        echo "Please install one of these packages in the {slurm_config['conda_env_path']} conda environment."
        echo "Run: conda install -n {slurm_config['conda_env_path']} pyyaml"
        echo "  or: conda install -n {slurm_config['conda_env_path']} omegaconf"
        exit 1
    fi
fi

# Check other required packages
echo "Checking for numpy, pandas, tqdm..."
python -c "try:
    import numpy
    print('numpy is available:', numpy.__version__)
    import pandas
    print('pandas is available:', pandas.__version__)
    import tqdm
    print('tqdm is available:', tqdm.__version__)
    exit(0)
except ImportError as e:
    print('Error importing core packages:', e)
    exit(1)
"
if [ $? -ne 0 ]; then
    echo "ERROR: Core packages not available in conda environment."
    echo "Please ensure that numpy, pandas, and tqdm are installed in the {slurm_config['conda_env_path']} conda environment."
    exit 1
fi
"""

    # Add remaining configuration
    script_content += f"""
#Simulation script (absolute path):
SIMULATION_SCRIPT={simulation_script}

#Python path:
PYTHON=python

CONFIG=${{CONFIG_DIR}}/${{CONFIG_FILE_NAME_INIT}}_${{SLURM_ARRAY_TASK_ID}}.yaml

echo "Starting simulation job at $(date)"
echo "Using simulation script: ${{SIMULATION_SCRIPT}}"
echo "Config file: ${{CONFIG}}"
echo "Output directory: ${{TASK_FOLDER_PATH}}"

start_time=$(date +%s)
srun \\
  --cpus-per-task=${{SLURM_CPUS_PER_TASK}} \\
  --kill-on-bad-exit \\
  ${{PYTHON}} -u ${{SIMULATION_SCRIPT}} --config ${{CONFIG}} --output-dir ${{TASK_FOLDER_PATH}}
result=$?

end_time=$(date +%s)
echo "Job completed with exit code $result at $(date)"
echo "Total duration: $((end_time - start_time)) seconds."

# Copy config to results directory
echo "Copying config to results directory"
cp ${{CONFIG}} ${{JOB_FOLDER_PATH}}/task_${{TASK_ID}}/config_used.yaml

exit $result
"""

    # Write to file or print for dryrun
    if dryrun:
        print(f"Would generate job script: {output_dir}/{script_name}")
        print(f"Script would contain array range: {array_range}")

        # In dryrun mode, create a sample script file
        sample_script_path = os.path.join(output_dir, f"{script_name}_sample")
        with open(sample_script_path, "w") as f:
            f.write(script_content)
        print(f"Created a sample job script for reference: {sample_script_path}")

        return os.path.join(output_dir, script_name)
    else:
        script_path = os.path.join(output_dir, script_name)
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)  # Make the script executable
        return script_path


def create_analyze_script(output_dir, timestamp, dryrun=False):
    """
    Create a script to analyze the results after all jobs are complete.
    """
    script_content = f"""#!/bin/bash
# analyze_dendrinet_results.sh
# Automatically generated script to analyze dendrinet simulation results

# Set the experiment timestamp
TIMESTAMP={timestamp}

# Run the analysis script
python scripts/analyze_dendrinet_results.py --timestamp $TIMESTAMP

# Check exit code
if [ $? -eq 0 ]; then
    echo "Analysis completed successfully!"
    echo "Results are saved in the plots/ subdirectory of your experiment folder."
else
    echo "Error occurred during analysis."
    echo "Check the log files for more information."
fi
"""

    if dryrun:
        print(
            f"Would generate analysis script: {output_dir}/analyze_dendrinet_results.sh"
        )
        return

    script_path = os.path.join(output_dir, "analyze_dendrinet_results.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)  # Make the script executable
    return script_path


def validate_slurm_time_format(config):
    """
    Validate and fix the time format for SLURM.
    SLURM expects time in format: [days-]hours:minutes:seconds
    """
    slurm_config = config.get("slurm_config", {})
    time_value = slurm_config.get("time", "1:00:00")

    # If time is numeric (seconds), convert to HH:MM:SS
    if isinstance(time_value, (int, float)) or (
        isinstance(time_value, str) and time_value.isdigit()
    ):
        seconds = int(time_value)
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        time_value = f"{hours}:{minutes:02d}:{seconds:02d}"

        # Update config
        slurm_config["time"] = time_value
        print(f"Warning: Converted numeric time value to HH:MM:SS format: {time_value}")

    # Check if time format looks valid
    if isinstance(time_value, str):
        # Simple validation for common formats
        if not (
            # HH:MM:SS
            re.match(r"^\d+:\d{1,2}:\d{1,2}$", time_value)
            or
            # MM:SS
            re.match(r"^\d{1,2}:\d{1,2}$", time_value)
            or
            # D-HH:MM:SS
            re.match(r"^\d+-\d+:\d{1,2}:\d{1,2}$", time_value)
            or
            # D-HH:MM
            re.match(r"^\d+-\d+:\d{1,2}$", time_value)
            or
            # D-HH
            re.match(r"^\d+-\d+$", time_value)
        ):
            print(f"Warning: Time format '{time_value}' may not be valid for SLURM.")
            print("SLURM expects time in format: [days-]hours:minutes:seconds")
            print("Examples: '1:00:00' (1 hour), '1-12:00:00' (1 day, 12 hours)")

    return config


def create_submission_check_script(output_dir, simulation_script, config, dryrun=False):
    """
    Create a script to check the environment and paths before job submission.
    This helps verify that everything is set up correctly.
    """
    slurm_config = config.get("slurm_config", {})
    conda_env_path = slurm_config.get("conda_env_path", "")

    script_content = f"""#!/bin/bash
# check_environment.sh
# Script to verify environment and paths before job submission

echo "==== Environment Check for Dendrinet Simulation ===="
echo "Checking if simulation script exists..."
if [ -f "{simulation_script}" ]; then
    echo "Found simulation script: {simulation_script}"
else
    echo "ERROR: Simulation script not found: {simulation_script}"
    exit 1
fi

echo "Checking output directories..."
if [ -d "{output_dir}" ]; then
    echo "Output directory exists: {output_dir}"
else
    echo "ERROR: Output directory not found: {output_dir}"
    exit 1
fi

if [ -d "{output_dir}/configs" ]; then
    echo "Config directory exists: {output_dir}/configs"
else
    echo "ERROR: Config directory not found: {output_dir}/configs"
    exit 1
fi

echo "Checking if SLURM is available..."
which sbatch
if [ $? -ne 0 ]; then
    echo "WARNING: sbatch command not found - SLURM may not be available"
else
    echo "SLURM is available"
fi

"""

    if conda_env_path:
        script_content += f"""
echo "Checking conda environment '{conda_env_path}'..."
export CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Try to activate the conda environment
conda activate {conda_env_path} 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Could not activate conda environment '{conda_env_path}'"
    echo "Please check if this environment exists"
    exit 1
else
    echo "Successfully activated conda environment '{conda_env_path}'"

    # Check the active environment name
    ENV_NAME=$(conda info --envs | grep '*' | awk '{{print $1}}')
    echo "  Active environment: $ENV_NAME"

    # Display Python version and path
    echo "Python version and path:"
    python --version
    which python

    # Display first few packages in conda environment
    echo "First 5 packages in conda environment:"
    conda list | head -5

    # Check for required packages
    echo "Checking required packages in conda environment..."
    PACKAGES_OK=true
    CONFIG_OK=false

    # Check for PyYAML
    echo "Checking for PyYAML..."
    python -c "
try:
    import yaml
    print(f'PyYAML is available: {{yaml.__file__}}')
    exit(0)
except ImportError as e:
    print(f'PyYAML not available: {{e}}')
    exit(1)
"
    if [ $? -eq 0 ]; then
        echo "PyYAML check passed."
        CONFIG_OK=true
    else
        echo "PyYAML check failed, trying OmegaConf."
    fi

    # Check for OmegaConf as alternative
    if [ "$CONFIG_OK" = false ]; then
        echo "Checking for OmegaConf..."
        python -c "
try:
    import sys
    print('Python paths:')
    for p in sys.path:
        print(f' - {{p}}')

    from omegaconf import OmegaConf
    print(f'OmegaConf is available: {{OmegaConf.__file__}}')
    exit(0)
except ImportError as e:
    print(f'OmegaConf not available: {{e}}')
    exit(1)
"
        if [ $? -eq 0 ]; then
            echo "OmegaConf check passed."
            CONFIG_OK=true
        else
            echo "OmegaConf check failed."
        fi
    fi

    # Config package check
    if [ "$CONFIG_OK" = false ]; then
        echo "ERROR: Neither PyYAML nor OmegaConf is available"
        echo "Please install one of these packages in the '{conda_env_path}' environment:"
        echo "  conda activate {conda_env_path}"
        echo "  conda install pyyaml"
        echo "  or"
        echo "  conda install omegaconf"
        PACKAGES_OK=false
    fi

    # Check other required packages
    echo "Checking for numpy..."
    python -c "
try:
    import numpy
    print(f'numpy is available: {{numpy.__version__}} at {{numpy.__file__}}')
    exit(0)
except ImportError as e:
    print(f'numpy not available: {{e}}')
    exit(1)
"
    if [ $? -ne 0 ]; then
        echo "Package 'numpy' is NOT available in conda environment"
        PACKAGES_OK=false
    fi

    echo "Checking for pandas..."
    python -c "
try:
    import pandas
    print(f'pandas is available: {{pandas.__version__}} at {{pandas.__file__}}')
    exit(0)
except ImportError as e:
    print(f'pandas not available: {{e}}')
    exit(1)
"
    if [ $? -ne 0 ]; then
        echo "Package 'pandas' is NOT available in conda environment"
        PACKAGES_OK=false
    fi

    echo "Checking for tqdm..."
    python -c "
try:
    import tqdm
    print(f'tqdm is available: {{tqdm.__version__}} at {{tqdm.__file__}}')
    exit(0)
except ImportError as e:
    print(f'tqdm not available: {{e}}')
    exit(1)
"
    if [ $? -ne 0 ]; then
        echo "Package 'tqdm' is NOT available in conda environment"
        PACKAGES_OK=false
    fi

    if [ "$PACKAGES_OK" = false ]; then
        echo "ERROR: Some required packages are missing from conda environment"
        echo "Please install the missing packages in the '{conda_env_path}' environment"
        exit 1
    fi

    # Deactivate when done
    conda deactivate
fi
"""

    script_content += f"""
echo "==== Environment check completed successfully ===="
echo "You can now submit the job using:"
echo "sbatch {output_dir}/run_dendrinet_sweep.sh"
"""

    if dryrun:
        print(
            f"Would generate environment check script: {output_dir}/check_environment.sh"
        )
        return

    script_path = os.path.join(output_dir, "check_environment.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)  # Make the script executable

    print(f"Created environment check script: {script_path}")
    print(f"Run this script to verify the environment before submitting the job:")
    print(f"  {script_path}")

    return script_path


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config_file)

    # Validate SLURM time format
    config = validate_slurm_time_format(config)

    # Generate parameter combinations
    combinations = generate_parameter_combinations(config)

    # Create timestamped directory or use dryrun directory
    if not args.dryrun:
        output_dir, timestamp = create_timestamp_directory(config)
    else:
        output_dir = "/tmp/dendrinet_dryrun"
        timestamp = "DRYRUN_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        # In dryrun mode, create the directory structure for clarity
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, "configs")):
            os.makedirs(os.path.join(output_dir, "configs"))

    # Generate config files
    config_filename_init = config["slurm_config"]["config_filename_init"]
    config_files = generate_config_files(
        config, combinations, output_dir, config_filename_init, args.dryrun
    )

    # Calculate job splits
    max_concurrent_jobs = config["slurm_config"].get("max_concurrent_jobs", 40)

    if args.split_jobs:
        num_jobs = args.split_jobs
        configs_per_job = (len(combinations) + num_jobs - 1) // num_jobs
    elif args.configs_per_job:
        configs_per_job = args.configs_per_job
        num_jobs = (len(combinations) + configs_per_job - 1) // configs_per_job
    else:
        num_jobs = 1
        configs_per_job = len(combinations)

    # Get simulation script path
    simulation_script = config["slurm_config"]["simulation_script"]
    if not os.path.isabs(simulation_script):
        current_dir = os.getcwd()
        simulation_script = os.path.abspath(
            os.path.join(current_dir, simulation_script)
        )

    # Generate job scripts
    job_scripts = []
    for i in range(num_jobs):
        start_idx = i * configs_per_job
        end_idx = min((i + 1) * configs_per_job - 1, len(combinations) - 1)

        if num_jobs > 1:
            script_name = f"run_dendrinet_sweep_{i+1}.sh"
        else:
            script_name = "run_dendrinet_sweep.sh"

        script_path = generate_slurm_script(
            config,
            output_dir,
            len(combinations),
            max_concurrent_jobs,
            script_name,
            start_idx,
            end_idx,
            args.dryrun,
        )
        job_scripts.append(script_path)

    # Create analysis script
    create_analyze_script(output_dir, timestamp, args.dryrun)

    # Create environment check script
    create_submission_check_script(output_dir, simulation_script, config, args.dryrun)

    # Copy the original config file to the output directory
    if not args.dryrun:
        shutil.copy(args.config_file, os.path.join(output_dir, "original_config.yaml"))
    else:
        print(
            f"Would copy {args.config_file} to {os.path.join(output_dir, 'original_config.yaml')}"
        )

    # Print summary
    print(f"Generated {len(combinations)} configurations across {num_jobs} job scripts")
    print(f"Output directory: {output_dir}")

    # In dryrun mode, print a sample of what would be generated
    if args.dryrun:
        print("\nSample of configurations that would be generated:")
        for i, combo in enumerate(combinations[:3]):
            print(f"\nConfig {i}:")
            for key, value in combo.items():
                if key in ["correlation_combos", "group_combos", "seeds"]:
                    print(f"  {key}: {len(value)} items")
                else:
                    print(f"  {key}: {value}")

            if i == 2 and len(combinations) > 3:
                print(f"\n... and {len(combinations) - 3} more configurations")

        print("\nEach job will process internally:")
        seeds_count = len(combinations[0].get("seeds", []))
        corr_count = len(combinations[0].get("correlation_combos", []))
        group_count = len(combinations[0].get("group_combos", []))
        internal_combos = seeds_count * corr_count * group_count
        print(f"  - {seeds_count} seeds")
        print(f"  - {corr_count} correlation combinations")
        print(f"  - {group_count} group combinations")
        print(f"  = {internal_combos} internal combinations per job")
        print(
            f"  = {internal_combos * len(combinations)} total combinations across all jobs"
        )

    # Print submission instructions
    if not args.dryrun:
        print("\nBefore submitting, run the environment check script:")
        print(f"  {output_dir}/check_environment.sh")
        print("\nThen submit the jobs:")
        for script in job_scripts:
            print(f"  sbatch {script}")
        print("\nAfter all jobs complete, run the analysis script:")
        print(f"  {output_dir}/analyze_dendrinet_results.sh")
    else:
        print("\nTo create and submit the actual jobs, run without --dryrun:")
        print(f"  python scripts/generate_dendrinet_sweep.py {args.config_file}")


if __name__ == "__main__":
    main()
