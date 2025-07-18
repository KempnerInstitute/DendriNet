#!/usr/bin/env python
"""
generate_ei_sweep.py
------------------------
Custom script to generate sweep configs for the EI sweep with the specific folder structure:
- Creates a unique 'ei_TIMESTAMP' directory for each sweep run
- Sets up subdirectories for configs, results, and plots
- Prepares the SLURM job configuration to organize outputs correctly
- Optionally splits sweep into multiple smaller job scripts
- Optionally filters E-I combinations based on specified constraints
- Supports network type comparison (EINet with/without shunting vs. MLP)

Filtering Modes:
- all: Generate all E-I combinations (default)
- ei-sum: Only include combinations where E+I equals a constant (E+I=K)
- e-const: Only include combinations with a specific constant E value
- i-const: Only include combinations with a specific constant I value
- ei-ratio: Only include combinations where E/I approximates a constant ratio

Network Type Comparison:
The script supports comparing different network architectures in the same sweep by using:
`model.core_network.parameters.use_shunting: [true, false, "MLP"]` in your sweep parameters.
- true: Creates an EINet with shunting inhibition enabled
- false: Creates an EINet with shunting inhibition disabled
- "MLP": Creates a standard MLP network using the same EI parameters

This enables direct comparison between dendritic (EINet) and non-dendritic (MLP)
architectures. The analyze_ei_sweep.py script will automatically detect these network
types and generate grouped visualizations to compare performance.

Usage:
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --split-jobs 10
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --configs-per-job 50

Filter Examples:
    # Generate only combinations where E+I=52
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-sum --filter-value 52

    # Generate only combinations where E=26
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode e-const --filter-value 26

    # Generate only combinations where I=26
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode i-const --filter-value 26

    # Generate only combinations where E/I ratio is approximately 2 (with default tolerance of 0.1)
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-ratio --filter-value 2

    # Generate only combinations where E/I ratio is approximately 2 with a custom tolerance of 0.2
    python scripts/generate_ei_sweep.py configs/sweep_params_seeds_ei.yaml --filter-mode ei-ratio --filter-value 2 --filter-tolerance 0.2
"""

import argparse
import datetime
import glob
import math
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

    # Add new filter arguments
    parser.add_argument(
        "--filter-mode",
        type=str,
        choices=["all", "ei-sum", "e-const", "i-const", "ei-ratio"],
        default="all",
        help="Filter mode for E-I combinations: all (default), ei-sum (E+I=constant), e-const (E=constant), "
        "i-const (I=constant), or ei-ratio (E/I=constant)",
    )
    parser.add_argument(
        "--filter-value",
        type=float,
        help="Value to use for filtering (constant sum, constant E, constant I, or constant ratio)",
    )
    parser.add_argument(
        "--filter-tolerance",
        type=float,
        default=0.1,
        help="Tolerance for ratio filtering (only used with ei-ratio mode)",
    )

    return parser.parse_args()


def filter_ei_combinations(
    e_values, i_values, filter_mode, filter_value, filter_tolerance=0.1
):
    """
    Filter E-I combinations based on the specified filter mode

    Args:
        e_values: List of E synapse values
        i_values: List of I synapse values
        filter_mode: Filtering mode (all, ei-sum, e-const, i-const, ei-ratio)
        filter_value: Value to use for filtering
        filter_tolerance: Tolerance for ratio filtering

    Returns:
        List of (e, i) tuples that satisfy the filter condition
    """
    if filter_mode == "all" or filter_value is None:
        # Return all combinations
        return [(e, i) for e in e_values for i in i_values]

    filtered_combinations = []

    for e in e_values:
        for i in i_values:
            # Extract the values (they are wrapped in lists)
            # Handle different types of values (lists, ListConfig objects, etc.)
            if isinstance(e, list) and len(e) > 0:
                e_val = e[0]
            elif hasattr(e, "__getitem__") and len(e) > 0:
                e_val = e[0]
            else:
                e_val = e

            if isinstance(i, list) and len(i) > 0:
                i_val = i[0]
            elif hasattr(i, "__getitem__") and len(i) > 0:
                i_val = i[0]
            else:
                i_val = i

            # Convert to float for comparison
            try:
                e_val = float(e_val)
                i_val = float(i_val)
            except (ValueError, TypeError):
                print(f"Warning: Could not convert values to float: {e_val}, {i_val}")
                continue

            if filter_mode == "ei-sum" and abs(e_val + i_val - filter_value) < 1e-6:
                # E + I = constant
                filtered_combinations.append((e, i))
            elif filter_mode == "e-const" and abs(e_val - filter_value) < 1e-6:
                # E = constant
                filtered_combinations.append((e, i))
            elif filter_mode == "i-const" and abs(i_val - filter_value) < 1e-6:
                # I = constant
                filtered_combinations.append((e, i))
            elif (
                filter_mode == "ei-ratio"
                and i_val > 0
                and abs(e_val / i_val - filter_value) <= filter_tolerance
            ):
                # E/I = constant within tolerance
                filtered_combinations.append((e, i))

    return filtered_combinations


def generate_sweep_configs_manually(
    config,
    output_dir,
    config_filename_init,
    filter_mode="all",
    filter_value=None,
    filter_tolerance=0.1,
):
    """
    Generate sweep configs manually without relying on generate_sweep_configs.py
    This ensures configs are created in the correct location

    Args:
        config: The configuration object
        output_dir: Output directory for configs
        config_filename_init: Base filename for config files
        filter_mode: Mode for filtering E-I combinations
        filter_value: Value for filtering (dependent on filter_mode)
        filter_tolerance: Tolerance for ratio filtering
    """
    sweep_params = config.sweep_config
    experiment_settings = config.get("experiment_settings", {})
    seeds_per_condition = experiment_settings.get("seeds_per_condition", 1)
    base_seed = experiment_settings.get("base_seed", 42)

    # Extract parameter names and values
    einet_param_names = list(sweep_params.keys())

    # Check if model.core_network.type is specified in sweep params
    force_network_type = None
    if "model.core_network.type" in einet_param_names:
        einet_param_names.remove("model.core_network.type")

    # Check if use_shunting is in the sweep parameters and contains "MLP"
    use_shunting_param_name = "model.core_network.parameters.use_shunting"
    has_mlp_option = False
    if use_shunting_param_name in sweep_params:
        # Check if any value is "MLP" (string) in the use_shunting parameter
        for value in sweep_params[use_shunting_param_name]:
            if isinstance(value, str) and value.upper() == "MLP":
                has_mlp_option = True
                break

    # Prepare parameter values for all parameters in the sweep
    # Ensure we handle potential OmegaConf objects correctly
    einet_param_values = []
    for key in einet_param_names:
        param_values = sweep_params[key]
        # Convert to simple Python types if it's an OmegaConf object
        if hasattr(param_values, "__len__") and not isinstance(
            param_values, (str, dict)
        ):
            # Convert to a list if it's a sequence type
            param_values = list(param_values)
        elif hasattr(param_values, "__iter__") and not isinstance(
            param_values, (str, dict)
        ):
            # Convert iterable to list
            param_values = list(param_values)
        else:
            # If it's a single value, wrap it in a list
            param_values = [param_values]

        einet_param_values.append(param_values)

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

        # Get the first iterable and ensure it's a proper list/tuple
        first_iterable = iterables[0]
        if hasattr(first_iterable, "__len__") and not isinstance(
            first_iterable, (str, dict)
        ):
            # It's already a sequence-like object
            pass
        elif hasattr(first_iterable, "__iter__") and not isinstance(
            first_iterable, (str, dict)
        ):
            # Convert iterator to list
            first_iterable = list(first_iterable)
        else:
            # Single value
            first_iterable = [first_iterable]

        # Debug output for the first iterable
        print(
            f"Processing parameter: {einet_param_names[0] if einet_param_names else 'unknown'}"
        )
        print(f"Values: {first_iterable} (type: {type(first_iterable)})")

        for item in first_iterable:
            for rest in product(*iterables[1:]):
                yield (item,) + rest

    # Check if we need to filter E-I combinations
    ee_param_name = "model.core_network.parameters.ee_synapses_per_branch_per_layer"
    ie_param_name = "model.core_network.parameters.ie_synapses_per_branch_per_layer"

    if (
        filter_mode != "all"
        and filter_value is not None
        and ee_param_name in sweep_params
        and ie_param_name in sweep_params
    ):
        # Find indices of E and I parameters
        ee_idx = einet_param_names.index(ee_param_name)
        ie_idx = einet_param_names.index(ie_param_name)

        # Extract E and I values
        ee_values = sweep_params[ee_param_name]
        ie_values = sweep_params[ie_param_name]

        # Filter combinations
        filtered_ei_combinations = filter_ei_combinations(
            ee_values, ie_values, filter_mode, filter_value, filter_tolerance
        )

        print(
            f"Filtered {len(filtered_ei_combinations)} E-I combinations using mode '{filter_mode}' with value {filter_value}"
        )

        # Generate filtered configurations
        for ee, ie in filtered_ei_combinations:
            # For each filtered combination
            for seed_idx in range(seeds_per_condition):
                # Determine the list of shunting values to iterate over.
                # If use_shunting_param_name is not in sweep_params, this loop will run once with a placeholder [None].
                shunting_values_to_iterate = [None]
                if use_shunting_param_name in sweep_params:
                    _temp_shunting_vals = sweep_params[use_shunting_param_name]
                    if isinstance(_temp_shunting_vals, list) or OmegaConf.is_list(
                        _temp_shunting_vals
                    ):
                        shunting_values_to_iterate = list(
                            _temp_shunting_vals
                        )  # Ensure it's a Python list
                    else:
                        shunting_values_to_iterate = [
                            _temp_shunting_vals
                        ]  # Wrap single value in a list

                for current_shunting_val in shunting_values_to_iterate:
                    # Create a new config based on the base config
                    new_config = OmegaConf.create(
                        OmegaConf.to_container(config.train_config, resolve=True)
                    )

                    # Set the seed
                    seed = base_seed + seed_idx
                    new_config.train.seed = seed

                    # Set network type and use_shunting based on current_shunting_val
                    # This logic assumes 'model.core_network.type' is not meant to be independently swept when filtering E/I and use_shunting.
                    # It prioritizes 'use_shunting' to determine network type if 'MLP' is involved.
                    if (
                        use_shunting_param_name in sweep_params
                    ):  # Check if use_shunting was part of the sweep definition
                        if (
                            isinstance(current_shunting_val, str)
                            and current_shunting_val.upper() == "MLP"
                        ):
                            new_config.model.core_network.type = "MLP"
                            set_nested_value(
                                new_config, use_shunting_param_name, False
                            )  # MLP implies shunting is conceptually False for EINet
                        else:
                            # Default to EINet if not MLP, or could take from base train_config if more sophisticated logic is needed
                            new_config.model.core_network.type = "EINet"
                            set_nested_value(
                                new_config,
                                use_shunting_param_name,
                                current_shunting_val,
                            )
                    # else: use_shunting is not in sweep_params. The value from train_config (the base) will be inherited.
                    # We also need to ensure model.network.type from train_config is respected if not overridden by MLP logic.
                    elif (
                        "model" in new_config
                        and "core_network" in new_config.model
                        and "type" in new_config.model.core_network
                    ):
                        pass  # It's already set from train_config
                    else:  # Ensure a default if not in sweep and not in train_config
                        new_config.model.core_network.type = "EINet"

                    # Set the filtered E and I parameters
                    set_nested_value(new_config, ee_param_name, ee)
                    set_nested_value(new_config, ie_param_name, ie)

                    # Set any other parameters from sweep_params (these take their first value in filtered mode)
                    for i, param_name_other in enumerate(einet_param_names):
                        # Skip params already handled (E, I, use_shunting if swept, network.type)
                        if param_name_other not in [
                            ee_param_name,
                            ie_param_name,
                            use_shunting_param_name,  # if use_shunting_param_name in sweep_params, it's handled above
                            # if not, it might be a fixed value in train_config, not listed in einet_param_names from sweep_params
                            "model.core_network.type",  # Handled by MLP logic or inherited
                        ]:
                            # Check if this param_name_other is actually use_shunting_param_name AND it was NOT in sweep_params
                            # This edge case should not occur if einet_param_names is from sweep_params.keys()

                            param_values_list = einet_param_values[
                                i
                            ]  # Values from sweep_config for this param_name_other

                            # Take the first value for these "other" swept parameters
                            param_value_to_set = (
                                param_values_list[0]
                                if (
                                    isinstance(param_values_list, list)
                                    or OmegaConf.is_list(param_values_list)
                                )
                                and param_values_list
                                else param_values_list
                            )
                            set_nested_value(
                                new_config, param_name_other, param_value_to_set
                            )

                    # If use_shunting_param_name was NOT in sweep_params, ensure its value from train_config is present if not already set.
                    # new_config is a copy of train_config, so it should be there.
                    # The logic above for MLP might set it to False. If not MLP and not in sweep, it's inherited.
                    if use_shunting_param_name not in sweep_params and hasattr(
                        config.train_config.model.core_network.parameters,
                        use_shunting_param_name.split(".")[-1],
                    ):
                        if not (
                            isinstance(current_shunting_val, str)
                            and current_shunting_val.upper() == "MLP"
                        ):  # if not already set by MLP logic
                            # Ensure it's explicitly set from train_config if not swept (defensive)
                            base_shunting_val = OmegaConf.select(
                                config.train_config, use_shunting_param_name
                            )
                            if base_shunting_val is not None:
                                set_nested_value(
                                    new_config,
                                    use_shunting_param_name,
                                    base_shunting_val,
                                )

                    # Make sure outputs.dir points to the correct directory
                    if "outputs" not in new_config:
                        new_config.outputs = {}

                    # Store config_idx in the output path for consistent location between runs and reruns
                    new_config.outputs.dir = os.path.join(
                        output_dir, "results", f"config_{config_idx}"
                    )
                    new_config.outputs.run_name = config.run_config.run_name

                    # Save the config
                    config_path = os.path.join(
                        output_dir,
                        "configs",
                        f"{config_filename_init}_{config_idx}.yaml",
                    )
                    OmegaConf.save(new_config, config_path)
                    config_idx += 1
    else:
        # This is the block for when filter_mode == "all" (NO E/I filtering)
        # This block should already correctly handle iterating all combinations from `product(*einet_param_values)`
        # including all variations of use_shunting if it's in sweep_params.
        # The original logic from line 420 onwards should be preserved.
        # Generate all parameter combinations without filtering
        for params in product(*einet_param_values):
            # For each parameter combination, create configs with different seeds
            for seed_idx in range(seeds_per_condition):
                # Create a new config based on the base config
                new_config = OmegaConf.create(
                    OmegaConf.to_container(config.train_config, resolve=True)
                )

                # Set the seed
                seed = base_seed + seed_idx
                new_config.train.seed = seed

                # Check if this is a configuration with use_shunting="MLP"
                is_mlp_config = False
                if has_mlp_option and use_shunting_param_name in einet_param_names:
                    idx = einet_param_names.index(use_shunting_param_name)
                    shunting_value = params[idx]

                    # Fix: Convert shunting_value to a proper single value
                    # In case it's still a list or other container
                    if hasattr(shunting_value, "__len__") and not isinstance(
                        shunting_value, str
                    ):
                        print(
                            f"Warning: use_shunting parameter is a container: {shunting_value}, using first value"
                        )
                        if len(shunting_value) > 0:
                            shunting_value = shunting_value[0]
                        else:
                            shunting_value = True  # Default

                    if (
                        isinstance(shunting_value, str)
                        and shunting_value.upper() == "MLP"
                    ):
                        # Set network type to MLP for this configuration
                        new_config.model.core_network.type = "MLP"
                        is_mlp_config = True
                        # Set use_shunting to False for MLP
                        set_nested_value(new_config, use_shunting_param_name, False)

                        # For MLP, keep the ee and ie parameters from the sweep
                        # and set all other parameters
                        for i, param_name in enumerate(einet_param_names):
                            # Fix: Make sure we set single values not lists
                            param_value = params[i]
                            # Convert parameter value if it's a container
                            if hasattr(param_value, "__len__") and not isinstance(
                                param_value, str
                            ):
                                if len(param_value) > 0 and param_name not in [
                                    "model.core_network.parameters.excitatory_branch_factors",
                                    "model.core_network.parameters.inhibitory_branch_factors",
                                    "model.core_network.parameters.ee_synapses_per_branch_per_layer",
                                    "model.core_network.parameters.ie_synapses_per_branch_per_layer",
                                ]:
                                    param_value = param_value[0]
                            set_nested_value(new_config, param_name, param_value)
                    else:
                        # This is an EINet config with specific use_shunting value
                        new_config.model.core_network.type = "EINet"
                        # Set all parameters, including use_shunting
                        for i, param_name in enumerate(einet_param_names):
                            # Fix: Make sure we set single values not lists
                            param_value = params[i]
                            # Convert parameter value if it's a container
                            if hasattr(param_value, "__len__") and not isinstance(
                                param_value, str
                            ):
                                if len(param_value) > 0 and param_name not in [
                                    "model.core_network.parameters.excitatory_branch_factors",
                                    "model.core_network.parameters.inhibitory_branch_factors",
                                    "model.core_network.parameters.ee_synapses_per_branch_per_layer",
                                    "model.core_network.parameters.ie_synapses_per_branch_per_layer",
                                ]:
                                    param_value = param_value[0]
                            set_nested_value(new_config, param_name, param_value)
                else:
                    # Standard EINet config (no MLP option)
                    # Set network type to EINet if specified in sweep_params
                    if "model.core_network.type" in sweep_params:
                        new_config.model.core_network.type = "EINet"

                    # Set all the parameter values
                    for i, param_name in enumerate(einet_param_names):
                        # Fix: Make sure we set single values not lists
                        param_value = params[i]
                        # Convert parameter value if it's a container
                        if hasattr(param_value, "__len__") and not isinstance(
                            param_value, str
                        ):
                            if len(param_value) > 0 and param_name not in [
                                "model.core_network.parameters.excitatory_branch_factors",
                                "model.core_network.parameters.inhibitory_branch_factors",
                                "model.core_network.parameters.ee_synapses_per_branch_per_layer",
                                "model.core_network.parameters.ie_synapses_per_branch_per_layer",
                            ]:
                                param_value = param_value[0]
                        set_nested_value(new_config, param_name, param_value)

                # Make sure outputs.dir points to the correct directory
                if "outputs" not in new_config:
                    new_config.outputs = {}

                # Store config_idx in the output path for consistent location between runs and reruns
                new_config.outputs.dir = os.path.join(
                    output_dir, "results", f"config_{config_idx}"
                )
                new_config.outputs.run_name = config.run_config.run_name

                # Save the config
                config_path = os.path.join(
                    output_dir, "configs", f"{config_filename_init}_{config_idx}.yaml"
                )
                OmegaConf.save(new_config, config_path)

                # Comment out the detailed print statement for each config
                # network_type_str = new_config.model.network.type if hasattr(new_config.model.network, "type") else "unknown"
                # use_shunting_val = "unknown"
                # if hasattr(new_config.model.network, "parameters") and hasattr(new_config.model.network.parameters, "use_shunting"):
                #     use_shunting_val = new_config.model.network.parameters.use_shunting
                # print(f"Generated config {config_idx}: {network_type_str} with use_shunting={use_shunting_val}")

                config_idx += 1

    # The MLP networks are now handled in the main parameter generation loop
    # via the use_shunting="MLP" option, so we can remove this separate MLP handling
    # unless we still want to generate MLP configurations from model.network.type
    net_types = sweep_params.get("model.core_network.type", ["EINet"])
    if "MLP" in net_types and not has_mlp_option:
        # set default values for MLP
        ee = [1]
        ie = [1]
        use_shunting = False
        for seed_idx in range(seeds_per_condition):
            # Create a new config based on the base config
            new_config = OmegaConf.create(
                OmegaConf.to_container(config.train_config, resolve=True)
            )

            seed = base_seed + seed_idx
            new_config.train.seed = seed
            new_config.model.core_network.type = "MLP"
            new_config.model.core_network.parameters.ee_synapses_per_branch_per_layer = ee
            new_config.model.core_network.parameters.ie_synapses_per_branch_per_layer = ie
            new_config.model.core_network.parameters.use_shunting = use_shunting

            # Make sure outputs.dir points to the correct directory
            if "outputs" not in new_config:
                new_config.outputs = {}

            # Store config_idx in the output path for consistent location
            new_config.outputs.dir = os.path.join(
                output_dir, "results", f"config_{config_idx}"
            )
            new_config.outputs.run_name = config.run_config.run_name

            # Save the config
            config_path = os.path.join(
                output_dir, "configs", f"{config_filename_init}_{config_idx}.yaml"
            )
            OmegaConf.save(new_config, config_path)

            # Comment out the detailed print statement for each config
            # network_type_str = new_config.model.network.type if hasattr(new_config.model.network, "type") else "unknown"
            # use_shunting_val = "unknown"
            # if hasattr(new_config.model.network, "parameters") and hasattr(new_config.model.network.parameters, "use_shunting"):
            #     use_shunting_val = new_config.model.network.parameters.use_shunting
            # print(f"Generated config {config_idx}: {network_type_str} with use_shunting={use_shunting_val}")

            config_idx += 1

    return config_idx  # Return the number of configs generated


def generate_custom_run_script(
    config,
    output_dir,
    num_configs,
    main_script_path="./run.sh",
    start_idx=0,
    end_idx=None,
):
    """Generate a custom run.sh script that properly uses the folder structure

    Args:
        config: The configuration object
        output_dir: Output directory for the sweep
        num_configs: Total number of configurations
        main_script_path: Path where to save the script
        start_idx: Starting configuration index for this job script
        end_idx: Ending configuration index for this job script (inclusive)
    """

    run_config = config.run_config

    # Adjust the range of configurations for this specific job script
    if end_idx is None or end_idx >= num_configs:
        end_idx = num_configs - 1

    # Determine if this is an array job (more than one config)
    array_job_flag = end_idx >= start_idx
    array_size = end_idx - start_idx + 1

    # Extract base script name without extension for job naming
    script_base_name = os.path.splitext(os.path.basename(main_script_path))[0]
    job_suffix = (
        f"_{start_idx}_{end_idx}" if start_idx > 0 or end_idx < num_configs - 1 else ""
    )
    job_name = f"{run_config.run_name}{job_suffix}"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={run_config.account}",
    ]

    # Output paths for logs - store in the config-specific directory
    if array_job_flag:
        lines.extend(
            [
                f"#SBATCH --output={output_dir}/results/config_%a/output.log",
                f"#SBATCH --error={output_dir}/results/config_%a/error.log",
            ]
        )
    else:
        lines.extend(
            [
                f"#SBATCH --output={output_dir}/results/config_{start_idx}/output.log",
                f"#SBATCH --error={output_dir}/results/config_{start_idx}/error.log",
            ]
        )

    # Format time to HH:MM:SS if it's not already
    time_value = run_config.time
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
            f"#SBATCH --nodes={run_config.nodes}",
            f"#SBATCH --ntasks-per-node={run_config.ntasks_per_node}",
            f"#SBATCH --gpus-per-node={run_config.gpus_per_node}",
            f"#SBATCH --cpus-per-task={run_config.cpus_per_task}",
            f"#SBATCH --time={formatted_time}",
        ]
    )

    # Properly handle memory specification - use only one method
    # Check if the memory specification has a unit suffix
    mem_value = run_config.mem
    if isinstance(mem_value, str):
        # If it has a unit suffix (GB, MB, etc.), use it as is for total memory
        if any(unit in mem_value.upper() for unit in ["GB", "MB", "G", "M", "K", "KB"]):
            lines.append(f"#SBATCH --mem={mem_value}")
        else:
            # If no unit, assume it's in MB and add the unit
            lines.append(f"#SBATCH --mem={mem_value}MB")
    else:
        # If it's a number, assume it's in MB
        lines.append(f"#SBATCH --mem={mem_value}MB")

    lines.append(f"#SBATCH --partition={run_config.partition}")

    # Submit an array job or a single job
    if array_job_flag and array_size > 1:
        max_concurrent = min(run_config.max_concurrent_jobs, array_size)
        if max_concurrent < array_size:
            lines.append(f"#SBATCH --array={start_idx}-{end_idx}%{max_concurrent}")
        else:
            lines.append(f"#SBATCH --array={start_idx}-{end_idx}")
    # For single task, don't use array
    elif start_idx == end_idx:
        # Nothing to add, will be a regular job
        pass

    lines.append("")
    lines.append("# ================================================================ ")
    lines.append("# This file has been generated by generate_ei_sweep.py script ")
    lines.append("#                            ***                                   ")
    lines.append("#       Manual changes to this file may be overwritten.            ")
    lines.append("# ================================================================ ")

    # Add module loading lines
    lines.append("module purge")
    for module in run_config.modules_to_load:
        lines.append(f"module load {module}")

    lines.append("")

    # Handle conda activation - simplified approach
    conda_env = run_config.conda_env_path
    lines.append("# Initialize conda and activate environment")
    lines.append("source $(conda info --base)/etc/profile.d/conda.sh")
    lines.append(f"conda activate {conda_env}")

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

    lines.append(f"CONFIG_FILE_NAME_INIT={run_config.config_filename_init}")
    lines.append(f"SC_RUN_NAME={run_config.run_name}")
    lines.append("")
    lines.append("#Training script: ")
    lines.append(f"TRAINING_SCRIPT={run_config.training_script}")
    lines.append("")
    lines.append("#Python path: ")

    # Handle Python path - check if conda_env_path is a path or environment name
    conda_env = run_config.conda_env_path
    if conda_env.startswith("/"):
        # It's a full path, use it directly
        lines.append(f"PYTHON={conda_env}/bin/python")
    else:
        # It's an environment name, use conda to find the path or just use python from PATH
        lines.append(
            "PYTHON=python"
        )  # Will use the python from the activated environment

    lines.append("")

    # Get the specific config file based on the array job ID
    if array_job_flag and array_size > 1:
        lines.append(
            "CONFIG=${CONFIG_DIR}/${CONFIG_FILE_NAME_INIT}_${SLURM_ARRAY_TASK_ID}.yaml"
        )
        lines.append("CONFIG_IDX=${SLURM_ARRAY_TASK_ID}")
    else:
        # For single task, use the specific index
        lines.append(
            f"CONFIG=${{CONFIG_DIR}}/${{CONFIG_FILE_NAME_INIT}}_{start_idx}.yaml"
        )
        lines.append(f"CONFIG_IDX={start_idx}")
    lines.append("")

    # Set up output folders with config-based directories instead of job/task based
    lines.append(f"# Path to the config-specific results folder")
    lines.append(f"CONFIG_FOLDER_PATH=${{RESULTS_DIR}}/config_${{CONFIG_IDX}}")
    lines.append(f"PLOT_FOLDER_PATH=${{PLOTS_DIR}}")
    lines.append("")

    # Create output directory
    lines.append("# Create output directory")
    lines.append("mkdir -p ${CONFIG_FOLDER_PATH}")
    lines.append("")

    # Run the job - Fix the srun command to avoid memory specification conflicts
    lines.append('echo "Starting running a training job at $(date)"')
    lines.append("start_time=$(date +%s)")

    # Unset conflicting SLURM memory environment variables before running srun
    lines.append(
        "# Unset conflicting SLURM memory environment variables to avoid conflicts"
    )
    lines.append("unset SLURM_MEM_PER_CPU")
    lines.append("unset SLURM_MEM_PER_GPU")
    lines.append("unset SLURM_MEM_PER_NODE")
    lines.append("")

    # Use srun without inheriting memory flags that might conflict
    lines.append("srun \\")
    lines.append("  --cpus-per-task=${SLURM_CPUS_PER_TASK} \\")
    lines.append("  --kill-on-bad-exit \\")
    lines.append(
        "  --export=ALL,SLURM_MEM_PER_CPU=,SLURM_MEM_PER_GPU=,SLURM_MEM_PER_NODE= \\"
    )

    # Fix the training script path and set environment variables for output paths
    if run_config.training_script.startswith("/"):
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
    lines.append("cp ${CONFIG} ${CONFIG_FOLDER_PATH}/config_used.yaml")

    # Write to file - now directly to the main script path
    with open(main_script_path, "w") as file:
        for line in lines:
            file.write(line + "\n")

    # Make the script executable
    os.chmod(main_script_path, 0o755)

    return main_script_path


def create_main_run_script(job_scripts, output_dir):
    """Create a main run.sh script that submits all the job scripts in a loop

    Args:
        job_scripts: List of generated job script paths
        output_dir: Output directory for the sweep

    Returns:
        Path to the created main run script
    """
    script_path = "./run.sh"

    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Main script to submit all EI sweep jobs\n\n")
        f.write(f"# Output directory: {output_dir}\n\n")

        # Get directory names relatively to make it more portable
        output_dir_basename = os.path.basename(output_dir)
        output_parent = os.path.dirname(output_dir)
        output_parent_basename = os.path.basename(output_parent)

        # Calculate relative path to job scripts
        relative_path = f"{output_parent_basename}/{output_dir_basename}/jobs"

        f.write("# Set up vars\n")
        f.write(f"SWEEP_DIR={output_dir}\n")
        f.write("JOBS_DIR=${SWEEP_DIR}/jobs\n\n")

        f.write("# Create jobs directory if it doesn't exist\n")
        f.write("mkdir -p ${JOBS_DIR}\n\n")

        f.write("# Submit all jobs in order with a slight delay between submissions\n")
        f.write('echo "Starting job submissions at $(date)"\n')
        f.write("for job_script in ${JOBS_DIR}/run_ei_sweep_*.sh; do\n")
        f.write('    echo "Submitting $job_script"\n')
        f.write("    sbatch $job_script\n")
        f.write("    # Sleep to avoid overwhelming the scheduler\n")
        f.write("    sleep 2\n")
        f.write("done\n\n")

        f.write('echo "All jobs submitted at $(date)"\n')
        f.write('echo "Check job status with: squeue -u $USER"\n')

    # Make the script executable
    os.chmod(script_path, 0o755)
    return script_path


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
    required_sections = ["run_config", "train_config", "sweep_config"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Config file missing required section: {section}")

    # Get the prefix from config or use default
    subdir_prefix = config.run_config.get("subdir_prefix", "ei_")

    # Create the uniquely named output directory
    output_base_dir = config.run_config.output_dir
    # Handle case where output_dir is not set in config
    if output_base_dir is None:
        output_base_dir = os.path.join(os.getcwd(), "outputs")
        print(f"Output directory not set in config, using default: {output_base_dir}")

    output_dir = os.path.join(output_base_dir, f"{subdir_prefix}{timestamp}")

    print(f"Creating output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories as specified in the config
    subfolders = config.run_config.get(
        "run_subfolders", ["configs", "results", "plots"]
    )
    for subfolder in subfolders:
        subdir_path = os.path.join(output_dir, subfolder)
        os.makedirs(subdir_path, exist_ok=True)
        print(f"Created subdirectory: {subdir_path}")

    # Create a jobs directory to store all job scripts
    jobs_dir = os.path.join(output_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    print(f"Created jobs directory: {jobs_dir}")

    # Update the config with the new output directory
    config.run_config.output_dir = output_dir

    # Check if the "outputs" block exists, if not add it
    if "outputs" not in config.train_config:
        config.train_config.outputs = {
            "dir": "",
            "run_name": config.run_config.run_name,
        }

    # Ensure outputs.dir is set to the specific results directory
    config.train_config.outputs.dir = os.path.join(output_dir, "results")

    # Save the updated base config to the configs subdirectory
    updated_config_path = os.path.join(output_dir, "configs", "sweep_params.yaml")
    OmegaConf.save(config, updated_config_path)
    print(f"Saved updated config to: {updated_config_path}")

    # Generate all the sweep configurations with optional filtering
    config_filename_init = config.run_config.config_filename_init
    num_configs = generate_sweep_configs_manually(
        config,
        output_dir,
        config_filename_init,
        filter_mode=args.filter_mode,
        filter_value=args.filter_value,
        filter_tolerance=args.filter_tolerance,
    )

    print(
        f"Generated {num_configs} config files in {os.path.join(output_dir, 'configs')}"
    )

    # Determine how to split the jobs
    split_jobs = args.split_jobs
    configs_per_job = args.configs_per_job

    # If neither split option is provided, create a single job (current behavior)
    if split_jobs is None and configs_per_job is None:
        # Generate a single run script but save it in the jobs directory
        job_script_path = os.path.join(jobs_dir, "run_ei_sweep_0.sh")
        generate_custom_run_script(config, output_dir, num_configs, job_script_path)
        job_scripts = [job_script_path]

    else:
        # Split into multiple job scripts
        job_scripts = []

        # Calculate how many configs per job
        if configs_per_job is not None:
            # User specified configs per job
            configs_per_job = min(configs_per_job, num_configs)
            num_jobs = math.ceil(num_configs / configs_per_job)
        else:
            # User specified number of jobs
            num_jobs = min(split_jobs, num_configs)
            configs_per_job = math.ceil(num_configs / num_jobs)

        print(
            f"Splitting sweep into {num_jobs} separate job scripts with approximately {configs_per_job} configs each"
        )

        # Generate each job script
        for i in range(num_jobs):
            start_idx = i * configs_per_job
            end_idx = min((i + 1) * configs_per_job - 1, num_configs - 1)

            # Create job script with appropriate range, saving in the jobs directory
            job_script_path = os.path.join(jobs_dir, f"run_ei_sweep_{i}.sh")
            generate_custom_run_script(
                config, output_dir, num_configs, job_script_path, start_idx, end_idx
            )

            job_scripts.append(job_script_path)
            print(
                f"Created job script {job_script_path} for configs {start_idx}-{end_idx}"
            )

    # Create a main run.sh script in the main directory
    main_script = create_main_run_script(job_scripts, output_dir)
    print(f"Created main script {main_script} to submit all jobs")

    print("\nSummary:")
    print(f"1. Created sweep directory: {output_dir}")
    print(
        f"2. Generated {num_configs} config files in: {os.path.join(output_dir, 'configs')}"
    )
    print(f"3. Saved {len(job_scripts)} job scripts in: {jobs_dir}")
    print(
        f"4. Results will be stored in: {os.path.join(output_dir, 'results')}/config_<config_idx>/"
    )
    print(
        f"5. Plots will be saved to: {os.path.join(output_dir, 'plots')}/config_<config_idx>/"
    )
    print(f"6. To run the complete sweep, use: ./run.sh")
    if args.filter_mode != "all":
        print(
            f"7. Applied filtering mode '{args.filter_mode}' with value {args.filter_value}"
        )


if __name__ == "__main__":
    main()
