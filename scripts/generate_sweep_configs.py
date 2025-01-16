#!/usr/bin/env python

import os
import argparse
from pathlib import Path
from copy import deepcopy
from itertools import product

from omegaconf import OmegaConf, DictConfig


#######################################################
# 1) Reading & Validation
#######################################################

def read_config(directory="configs", param_file_name="sweep_params.yaml"):
    """
    Reads the configuration file (sweep_config.yaml) from the specified directory.
    It must have top-level keys:
      - slurm_config
      - train_config
      - sweep_config
    """
    config_path = os.path.join(directory, param_file_name)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    try:
        params = OmegaConf.load(config_path)
        return params
    except Exception as e:
        print(f"Error reading config file: {e}")
        return None


def validate_params(params):
    """
    Ensures 'slurm_config', 'train_config', and 'sweep_config' exist.
    """
    required_configs = ['slurm_config', 'train_config', 'sweep_config']
    for cfg in required_configs:
        if cfg not in params:
            raise ValueError(f"Missing required config: '{cfg}'")


#######################################################
# 2) Custom YAML Dump
#######################################################

def custom_yaml_dump(data, indent=0):
    """
    Custom YAML dump function to convert nested dict / OmegaConf to YAML lines.
    Lists are printed inline.
    """
    yaml_str = ""
    indent_str = "  " * indent  # two spaces per indent level

    for key, value in data.items():
        if isinstance(value, (dict, DictConfig)):
            yaml_str += f"{indent_str}{key}:\n"
            yaml_str += custom_yaml_dump(value, indent+1)
        elif isinstance(value, list):
            yaml_str += f"{indent_str}{key}: {value}\n"
        else:
            yaml_str += f"{indent_str}{key}: {value}\n"
    return yaml_str


#######################################################
# 3) Flattening "sweep_config"
#######################################################

def flatten(config, parent_key='', sep='.'):
    """
    Flattens a nested dict or OmegaConf into dot-notation keys => values.
    For example:
      {"model": {"topk_init_method": ["xavier","uniform"]}}
    becomes => {"model.topk_init_method": ["xavier","uniform"]}.
    """
    if config is None:
        return {}
    items = []
    for k,v in config.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, DictConfig):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


#######################################################
# 4) Generate config combos
#######################################################

def generate_config_files(base_config, sweep_params, output_dir="config_files"):
    """
    base_config: The dictionary containing 'train_config'
    sweep_params: Flattened dictionary of dot-keys => list of possible values
    """
    os.makedirs(output_dir, exist_ok=True)

    sweep_keys = list(sweep_params.keys())
    sweep_values = [sweep_params[k] for k in sweep_keys]

    # create all combinations with product
    all_combos = list(product(*sweep_values))

    configs = []
    sweeps  = []
    for combo in all_combos:
        new_cfg = deepcopy(base_config)
        param_details = {}
        # For each key in sweep_keys, override new_cfg
        for k, val in zip(sweep_keys, combo):
            path = k.split(".")    # e.g. ["model", "topk_init_method"]
            sub_cfg = new_cfg
            for subkey in path[:-1]:
                sub_cfg = sub_cfg[subkey]
            sub_cfg[path[-1]] = val
            param_details[k] = val

        containerized = OmegaConf.to_container(new_cfg, resolve=False)
        configs.append(containerized)
        sweeps.append(param_details)

    print(f"Generated {len(configs)} configuration files.")
    return configs, sweeps


def write_config_files(configs, sweeps, output_dir="configs", base_name="config"):
    """
    Write each config to disk (YAML) plus a sweep summary text.
    """
    assert len(configs)==len(sweeps), "Mismatch between config count and sweep param combos."

    os.makedirs(output_dir, exist_ok=True)

    summary_lines = []
    for i, cdict in enumerate(configs):
        config_name = f"{base_name}_{i}.yaml"
        config_path = os.path.join(output_dir, config_name)
        config_yaml = custom_yaml_dump(cdict)
        with open(config_path, "w") as f:
            f.write(config_yaml)
        summary_lines.append(f"{config_name}: {sweeps[i]}\n")
        print(f"Configuration saved to {config_path}")

    # write a summary
    sweep_path = os.path.join(output_dir, f"{base_name}_sweeps.txt")
    with open(sweep_path,"w") as f:
        f.writelines(summary_lines)


#######################################################
# 5) Generate run.sh
#######################################################

def generate_run_sh(num_config, config_dir, config_filename_init, slurm_config):
    """
    Writes run.sh for Slurm array or single job, using slurm_config.
    """
    array_job_flag = (num_config>1)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm_config['run_name']}",
        f"#SBATCH --account={slurm_config['account']}"
    ]

    # Output/error path
    if array_job_flag:
        lines.extend([
            f"#SBATCH --output={slurm_config['output_dir']}/{slurm_config['run_name']}_%A/%A_%a/output_%A_%a.out",
            f"#SBATCH --error={slurm_config['output_dir']}/{slurm_config['run_name']}_%A/%A_%a/error_%A_%a.out",
        ])
    else:
        lines.extend([
            f"#SBATCH --output={slurm_config['output_dir']}/{slurm_config['run_name']}_%j/output_%j.out",
            f"#SBATCH --error={slurm_config['output_dir']}/{slurm_config['run_name']}_%j/error_%j.out",
        ])

    # Resources
    lines.extend([
        f"#SBATCH --nodes={slurm_config['nodes']}",
        f"#SBATCH --ntasks-per-node={slurm_config['ntasks_per_node']}",
        f"#SBATCH --gpus-per-node={slurm_config['gpus_per_node']}",
        f"#SBATCH --cpus-per-task={slurm_config['cpus_per_task']}",
        f"#SBATCH --time={slurm_config['time']}",
        f"#SBATCH --mem={slurm_config['mem']}",
        f"#SBATCH --partition={slurm_config['partition']}"
    ])

    # If multiple combos => array job
    if array_job_flag:
        if num_config>slurm_config['max_concurrent_jobs']:
            lines.append(f"#SBATCH --array=0-{num_config-1}%{slurm_config['max_concurrent_jobs']}")
        else:
            lines.append(f"#SBATCH --array=0-{num_config-1}")

    lines.append("")
    lines.append("# ================================================================ ")
    lines.append("# This file has been generated by generate_sweep_configs.py script ")
    lines.append("# ================================================================ ")
    lines.append("module purge")
    for mod in slurm_config['modules_to_load']:
        lines.append(f"module load {mod}")
    lines.append("")
    lines.append(f"conda activate {slurm_config['conda_env_path']}")
    lines.append("")
    lines.append(f"CONFIG_DIR={config_dir}")
    lines.append(f"OUTPUT_DIR={slurm_config['output_dir']}")
    lines.append(f"CONFIG_FILE_NAME_INIT={config_filename_init}")
    lines.append(f"SC_RUN_NAME={slurm_config['run_name']}")
    lines.append("")
    lines.append(f"TRAINING_SCRIPT={slurm_config['training_script']}")
    lines.append(f"PYTHON={slurm_config['conda_env_path']}/bin/python")
    lines.append("")
    if array_job_flag:
        lines.append('CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_${SLURM_ARRAY_TASK_ID}.yaml')
        lines.append('RUN_NAME=${SC_RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}')
        lines.append("OUTPUT_FOLDER_PATH=${OUTPUT_DIR}/${SC_RUN_NAME}_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}")
    else:
        lines.append('CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_0.yaml')
        lines.append('RUN_NAME=${SC_RUN_NAME}_${SLURM_JOB_ID}')
        lines.append("OUTPUT_FOLDER_PATH=${OUTPUT_DIR}/${SC_RUN_NAME}_${SLURM_JOB_ID}")
    lines.append("")
    lines.append('echo "Starting training job at $(date)"')
    lines.append("start_time=$(date +%s)")
    lines.append("srun \\")
    lines.append("  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\")
    lines.append("  --kill-on-bad-exit \\")
    lines.append("  ${PYTHON} -u ${TRAINING_SCRIPT} ${CONFIG} \\")
    lines.append("  --output_dir=${OUTPUT_FOLDER_PATH} \\")
    lines.append("  --experiment_name=${RUN_NAME} \\")
    lines.append("  ${@}")
    lines.append("end_time=$(date +%s)")
    lines.append('echo "Done with running a training job at $(date)"')
    lines.append('echo "Total duration: $((end_time - start_time)) seconds."')

    with open("run.sh","w") as f:
        for line in lines:
            f.write(line+"\n")
    print("run.sh file has been generated.")


#######################################################
# 6) parse_args & main
#######################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate configuration files for a sweep from a single param file."
    )
    parser.add_argument("config_dir", default="configs", type=str,
                        help="Directory containing the sweep_config.yaml.")
    parser.add_argument("param_filename", default="sweep_params.yaml", type=str,
                        help="Name of the sweep config file with slurm_config, train_config, sweep_config.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_dir = args.config_dir
    param_filename = args.param_filename

    # 1) read sweep config
    params = read_config(config_dir, param_filename)
    validate_params(params)

    # 2) Extract top-level dicts
    slurm_cfg    = params['slurm_config']
    base_config  = params['train_config']   # This is the base config
    sweep_dict   = params['sweep_config']   # The param combos

    # Optionally, set run_name
    if 'run_name' not in slurm_cfg:
        # if user wants to copy from base_config, e.g. base_config['train']['run_name']
        if 'train' in base_config and 'run_name' in base_config['train']:
            slurm_cfg['run_name'] = base_config['train']['run_name']
        else:
            slurm_cfg['run_name'] = "my_sweep"

    # 3) flatten the sweep config
    flattened_sweep = flatten(sweep_dict)

    # 4) generate combinations => final configs
    config_filename_init = slurm_cfg['config_filename_init']  # e.g. "test_config"
    configs, sweeps = generate_config_files(base_config, flattened_sweep, output_dir=config_dir)

    # 5) write them out
    write_config_files(configs, sweeps, output_dir=config_dir, base_name=config_filename_init)

    # 6) run.sh
    generate_run_sh(num_config=len(configs),
                    config_dir=config_dir,
                    config_filename_init=config_filename_init,
                    slurm_config=slurm_cfg)

if __name__=="__main__":
    main()