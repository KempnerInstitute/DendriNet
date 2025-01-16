#!/usr/bin/env python

import os
import argparse
from pathlib import Path
from copy import deepcopy
from itertools import product

from omegaconf import OmegaConf, DictConfig


def read_sweep_params(config_dir="configs", param_filename="sweep_params.yaml"):
    """
    Reads the sweep_config.yaml file, expecting keys:
      - slurm_config
      - base_config
      - sweep_config
    """
    config_path = os.path.join(config_dir, param_filename)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No such file: {config_path}")
    try:
        params = OmegaConf.load(config_path)
        return params
    except Exception as e:
        print(f"Error reading config: {e}")
        return None

def validate_sweep_params(params):
    req_keys = ["slurm_config","base_config","sweep_config"]
    for k in req_keys:
        if k not in params:
            raise ValueError(f"Missing required key: {k}")

def custom_yaml_dump(data, indent=0):
    """
    Dump nested dict to YAML lines. Lists are inline.
    Ensures empty dictionaries are printed as '{}'.
    """
    yaml_str = ""
    prefix = "  " * indent

    # If data is None or not a dict, handle separately,
    # but typically we expect a dict or OmegaConf container here.
    if not isinstance(data, (dict, DictConfig)):
        return f"{prefix}{data}\n"

    for key, value in data.items():
        if isinstance(value, (dict, DictConfig)):
            # Check if the dict is empty
            if len(value) == 0:
                # print an empty dictionary
                yaml_str += f"{prefix}{key}: {{}}\n"
            else:
                # non-empty, recurse
                yaml_str += f"{prefix}{key}:\n"
                yaml_str += custom_yaml_dump(value, indent+1)
        elif isinstance(value, list):
            # inline list
            yaml_str += f"{prefix}{key}: {value}\n"
        else:
            # scalar
            yaml_str += f"{prefix}{key}: {value}\n"

    return yaml_str

def flatten_dict(cfg, parent="", sep="."):
    """
    Flattens nested dict or OmegaConf with dot-notation keys => values list
    used for sweeping.
    """
    items = []
    for k,v in cfg.items():
        new_key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, DictConfig):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def generate_configs(base_cfg, sweep_params):
    """
    base_cfg: (dict or OmegaConf) => the entire nested structure
    sweep_params: flattened dict => each value is list of possible values
    returns => list of final configs + param combos
    """
    from itertools import product
    keys = list(sweep_params.keys())
    values = [sweep_params[k] for k in keys]
    combos = list(product(*values))

    final_configs = []
    combos_info   = []

    for c in combos:
        # deep copy the base
        new_cfg = deepcopy(base_cfg)
        param_info = {}
        for k, val in zip(keys, c):
            path = k.split(".")
            sub = new_cfg
            for subkey in path[:-1]:
                sub = sub[subkey]
            # set final
            sub[path[-1]] = val
            param_info[k] = val
        container = OmegaConf.to_container(new_cfg, resolve=False)
        final_configs.append(container)
        combos_info.append(param_info)
    return final_configs, combos_info

def write_configs(final_configs, combos_info, outdir="configs", base_name="test_config"):
    """
    Writes final config files as .yaml + a summary
    """
    os.makedirs(outdir, exist_ok=True)

    summary_lines = []
    for i, cdict in enumerate(final_configs):
        fname = f"{base_name}_{i}.yaml"
        fpath = os.path.join(outdir, fname)
        yaml_out = custom_yaml_dump(cdict)
        with open(fpath,"w") as f:
            f.write(yaml_out)
        summary_lines.append(f"{fname}: {combos_info[i]}\n")
        print(f"Saved {fpath}")

    summary_path = os.path.join(outdir,f"{base_name}_sweeps.txt")
    with open(summary_path,"w") as f:
        f.writelines(summary_lines)

def generate_run_sh(num_configs, config_dir, base_name, slurm_cfg):
    """
    Writes run.sh for launching the jobs with Slurm
    """
    is_array = (num_configs>1)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={slurm_cfg['run_name']}",
        f"#SBATCH --account={slurm_cfg['account']}"
    ]
    if is_array:
        lines.extend([
            f"#SBATCH --output={slurm_cfg['output_dir']}/{slurm_cfg['run_name']}_%A/%A_%a/output_%A_%a.out",
            f"#SBATCH --error={slurm_cfg['output_dir']}/{slurm_cfg['run_name']}_%A/%A_%a/error_%A_%a.out",
        ])
    else:
        lines.extend([
            f"#SBATCH --output={slurm_cfg['output_dir']}/{slurm_cfg['run_name']}_%j/output_%j.out",
            f"#SBATCH --error={slurm_cfg['output_dir']}/{slurm_cfg['run_name']}_%j/error_%j.out",
        ])
    lines.extend([
        f"#SBATCH --nodes={slurm_cfg['nodes']}",
        f"#SBATCH --ntasks-per-node={slurm_cfg['ntasks_per_node']}",
        f"#SBATCH --gpus-per-node={slurm_cfg['gpus_per_node']}",
        f"#SBATCH --cpus-per-task={slurm_cfg['cpus_per_task']}",
        f"#SBATCH --time={slurm_cfg['time']}",
        f"#SBATCH --mem={slurm_cfg['mem']}",
        f"#SBATCH --partition={slurm_cfg['partition']}"
    ])
    if is_array:
        if num_configs>slurm_cfg['max_concurrent_jobs']:
            lines.append(f"#SBATCH --array=0-{num_configs-1}%{slurm_cfg['max_concurrent_jobs']}")
        else:
            lines.append(f"#SBATCH --array=0-{num_configs-1}")

    lines.append("")
    lines.append("#====================================================")
    lines.append("# Auto-generated by generate_sweep_configs.py")
    lines.append("#====================================================")
    lines.append("module purge")
    for m in slurm_cfg['modules_to_load']:
        lines.append(f"module load {m}")
    lines.append("")
    lines.append(f"conda activate {slurm_cfg['conda_env_path']}")
    lines.append("")
    lines.append(f"CONFIG_DIR={config_dir}")
    lines.append(f"OUTPUT_DIR={slurm_cfg['output_dir']}")
    lines.append(f"CONFIG_FILE_NAME_INIT={base_name}")
    lines.append(f"SC_RUN_NAME={slurm_cfg['run_name']}")
    lines.append("")
    lines.append(f"TRAINING_SCRIPT={slurm_cfg['training_script']}")
    lines.append(f"PYTHON={slurm_cfg['conda_env_path']}/bin/python")
    lines.append("")
    if is_array:
        lines.append('CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_${SLURM_ARRAY_TASK_ID}.yaml')
        lines.append('RUN_NAME=${SC_RUN_NAME}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}')
        lines.append('OUTPUT_FOLDER_PATH=${OUTPUT_DIR}/${SC_RUN_NAME}_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}')
    else:
        lines.append('CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_0.yaml')
        lines.append('RUN_NAME=${SC_RUN_NAME}_${SLURM_JOB_ID}')
        lines.append('OUTPUT_FOLDER_PATH=${OUTPUT_DIR}/${SC_RUN_NAME}_${SLURM_JOB_ID}')
    lines.append("")
    lines.append('echo "Starting job at $(date)"')
    lines.append("start_time=$(date +%s)")
    lines.append("srun \\")
    lines.append("  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\")
    lines.append("  --kill-on-bad-exit \\")
    lines.append("  ${PYTHON} -u ${TRAINING_SCRIPT} ${CONFIG} \\")
    lines.append("  --output_dir=${OUTPUT_FOLDER_PATH} \\")
    lines.append("  --experiment_name=${RUN_NAME} \\")
    lines.append("  ${@}")
    lines.append("end_time=$(date +%s)")
    lines.append('echo "Done at $(date)"')
    lines.append('echo "Total duration: $((end_time - start_time)) seconds."')

    with open("run.sh","w") as f:
        for line in lines:
            f.write(line+"\n")
    print("run.sh file has been generated.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_dir", type=str,
                        help="Directory containing the sweep_config.yaml")
    parser.add_argument("param_filename", type=str,
                        help="Name of that file, e.g. sweep_config.yaml")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1) read the sweep_config
    data = read_sweep_params(args.config_dir, args.param_filename)
    validate_sweep_params(data)

    slurm_cfg   = data['slurm_config']
    base_config = data['base_config']      # entire nested structure
    sweep_cfg   = data['sweep_config']     # dot-notation combos

    # If user didn't specify run_name in slurm_config, default it
    if 'run_name' not in slurm_cfg:
        slurm_cfg['run_name'] = "my_sweep_job"

    # 2) flatten sweep_config
    flat_sweep = flatten_dict(sweep_cfg)

    # 3) generate combos
    configs, combos = generate_configs(base_config, flat_sweep)

    # 4) write them
    out_dir = args.config_dir
    base_name = slurm_cfg['config_filename_init']  # e.g. "test_config"
    write_configs(configs, combos, outdir=out_dir, base_name=base_name)

    # 5) make run.sh
    generate_run_sh(len(configs), out_dir, base_name, slurm_cfg)

if __name__=="__main__":
    main()