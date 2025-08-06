"""
Unified Sweep Implementation
===========================

A single generator that handles all sweep types (EI, branch, noise, general)
through YAML configuration. No need for separate sweep type scripts.

All sweeps are defined through:
- sweep_config: Parameter values to sweep over
- filter_config: Special modes and filtering options
- experiment_settings: Seeds, base configurations

This replaces ei_sweep.py, noise_sweep.py, branch_sweep.py, etc.
"""

import itertools
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Use absolute imports to avoid relative import issues
try:
    from dendritic_modeling.scripts.sweeps.unified.base_sweep import BaseSweepGenerator
except ImportError:
    # Fallback for when run as script
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from base_sweep import BaseSweepGenerator


def expand_ranges(param_value):
    """
    Handle parameter values that are specified as ranges.
    A range is specified as [start, end, step].
    From the old branch sweep code - handles both int and float ranges.
    Works with both regular lists and OmegaConf ListConfig objects.
    """
    # Convert OmegaConf objects to regular Python objects for processing
    if hasattr(param_value, "_content"):  # OmegaConf object
        param_value = OmegaConf.to_container(param_value, resolve=True)

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


class UnifiedSweepGenerator(BaseSweepGenerator):
    """
    Universal generator for all sweep types.

    Handles all parameter sweeps through sweep_config and filter_config:
    - EI sweeps: Define ee_synapses_per_branch_per_layer, ie_synapses_per_branch_per_layer
    - Branch sweeps: Use filter_config.mode: branch_sweep with depth/width ranges
    - Noise sweeps: Define noise parameters in sweep_config
    - Mixed sweeps: Combine any parameters together

    Filter modes:
    - default: Generate all combinations
    - branch_sweep: Generate systematic branch architecture combinations
    - ei-sum, ei-ratio, e-const, i-const: EI filtering modes (legacy)
    """

    def __init__(self):
        super().__init__("unified")

    def validate_config(self, config: DictConfig) -> bool:
        """Validate unified sweep configuration."""
        # Check for sweep_config or filter_config (at least one must generate parameters)
        sweep_config = config.get("sweep_config", {})
        filter_config = config.get("filter_config", {})

        # If sweep_config is empty, check if filter_config can generate parameters
        if not sweep_config:
            filter_mode = filter_config.get("mode", "default")
            if filter_mode not in ["branch_sweep"]:
                raise ValueError(
                    "Either sweep_config must have parameters or filter_config.mode must be 'branch_sweep'"
                )

        return True

    def _apply_ei_filter(
        self, ee_values: list, ie_values: list, filter_config: dict
    ) -> list[tuple]:
        """
        Apply EI-specific filtering modes using unified 'value' parameter.

        value meaning by mode:
        - ei-sum: target sum (E + I = value)
        - e-const: constant E value (E = value)
        - i-const: constant I value (I = value)
        - ei-ratio: target E/I ratio (E/I = value)
        """
        mode = filter_config.get("mode", "all")
        value = filter_config.get("value", None)

        if mode == "all":
            return list(itertools.product(ee_values, ie_values))

        elif mode == "ei-sum":
            target_sum = value if value is not None else 10
            tolerance = filter_config.get("tolerance", 1)
            return [
                (ee, ie)
                for ee in ee_values
                for ie in ie_values
                if abs((ee + ie) - target_sum) <= tolerance
            ]

        elif mode == "e-const":
            const_e = value if value is not None else ee_values[0]
            return [(const_e, ie) for ie in ie_values if const_e in ee_values]

        elif mode == "i-const":
            const_i = value if value is not None else ie_values[0]
            return [(ee, const_i) for ee in ee_values if const_i in ie_values]

        elif mode == "ei-ratio":
            target_ratio = value if value is not None else 0.5
            tolerance = filter_config.get("tolerance", 0.1)
            return [
                (ee, ie)
                for ee in ee_values
                for ie in ie_values
                if ie > 0 and abs((ee / ie) - target_ratio) <= tolerance
            ]

        else:
            return list(itertools.product(ee_values, ie_values))

    def _generate_branch_combinations(self, filter_config: dict) -> dict[str, list]:
        """
        Generate branch architecture combinations based on filter_config.

        Supports multiple branching patterns:
        1. depth_sweep: Vary depth while keeping branch counts constant/uniform
        2. width_sweep: Vary branch counts while keeping depth constant
        3. uniform_sweep: Both depth and width vary together
        4. pattern_sweep: Specific branching patterns (decreasing, increasing, etc.)
        """
        branch_params = {}
        sweep_mode = filter_config.get("branch_sweep_mode", "uniform_sweep")

        if sweep_mode == "depth_sweep":
            # Focus on depth variation: [1,1,1...] to [n,n,n...]
            depths = expand_ranges(filter_config.get("dendrite_tree_depth", [1, 5, 1]))
            uniform_branches = filter_config.get("uniform_branches_per_layer", 4)

            branch_params["model.core_network.parameters.dendrite_tree_depth"] = depths
            branch_params[
                "model.core_network.parameters.dendrite_branches_per_layer"
            ] = [uniform_branches] * len(depths)

        elif sweep_mode == "width_sweep":
            # Focus on width variation: keep depth constant, vary branches
            fixed_depth = filter_config.get("fixed_dendrite_tree_depth", 2)
            branch_counts = expand_ranges(
                filter_config.get("dendrite_branches_per_layer", [1, 8, 1])
            )

            branch_params["model.core_network.parameters.dendrite_tree_depth"] = [
                fixed_depth
            ] * len(branch_counts)
            branch_params[
                "model.core_network.parameters.dendrite_branches_per_layer"
            ] = branch_counts

        elif sweep_mode == "pattern_sweep":
            # Specific branching patterns
            patterns = filter_config.get(
                "branch_patterns", ["uniform", "decreasing", "increasing"]
            )
            depths = expand_ranges(filter_config.get("dendrite_tree_depth", [2, 4, 1]))
            base_branches = filter_config.get("base_branches", 4)

            all_depth_configs = []
            all_branch_configs = []

            for depth in depths:
                for pattern in patterns:
                    if pattern == "uniform":
                        # [n, n, n, ...] - same branches per layer
                        branch_config = [base_branches] * depth
                    elif pattern == "decreasing":
                        # [n, n-1, n-2, ...] - decreasing branches with depth
                        branch_config = [
                            max(1, base_branches - i) for i in range(depth)
                        ]
                    elif pattern == "increasing":
                        # [1, 2, 3, ...] - increasing branches with depth
                        branch_config = [i + 1 for i in range(depth)]
                    elif pattern == "pyramid":
                        # [1, 2, 4, 2, 1] - pyramid shape (if depth allows)
                        mid = depth // 2
                        branch_config = []
                        for i in range(depth):
                            if i <= mid:
                                branch_config.append(i + 1)
                            else:
                                branch_config.append(depth - i)
                    else:
                        branch_config = [base_branches] * depth

                    all_depth_configs.append(depth)
                    all_branch_configs.append(branch_config)

            branch_params["model.core_network.parameters.dendrite_tree_depth"] = (
                all_depth_configs
            )
            branch_params[
                "model.core_network.parameters.dendrite_branches_per_layer"
            ] = all_branch_configs

        elif sweep_mode == "custom_configs":
            # Direct specification of (depth, branch_list) pairs
            custom_configs = filter_config.get(
                "custom_branch_configs",
                [
                    {"depth": 2, "branches": [2, 4]},
                    {"depth": 3, "branches": [1, 2, 1]},
                ],
            )

            depths = []
            branch_lists = []
            for config in custom_configs:
                depths.append(config["depth"])
                branch_lists.append(config["branches"])

            branch_params["model.core_network.parameters.dendrite_tree_depth"] = depths
            branch_params[
                "model.core_network.parameters.dendrite_branches_per_layer"
            ] = branch_lists

        else:  # uniform_sweep (default) - old behavior
            # Generate all combinations of depth x uniform branch count
            depths = expand_ranges(filter_config.get("dendrite_tree_depth", [1, 3, 1]))
            branch_counts = expand_ranges(
                filter_config.get("dendrite_branches_per_layer", [2, 4, 1])
            )

            # Generate cartesian product
            branch_combos = list(itertools.product(depths, branch_counts))

            if branch_combos:
                branch_params["model.core_network.parameters.dendrite_tree_depth"] = [
                    combo[0] for combo in branch_combos
                ]
                branch_params[
                    "model.core_network.parameters.dendrite_branches_per_layer"
                ] = [combo[1] for combo in branch_combos]

        return branch_params

    def _generate_branch_configurations(self, filter_config: dict) -> list[dict]:
        """
        Generate branch configurations as paired depth+branch combinations.
        Returns a list of dictionaries, each containing a complete branch configuration.
        """
        branch_configs = []
        sweep_mode = filter_config.get("branch_sweep_mode", "uniform_sweep")

        if sweep_mode == "depth_sweep":
            depths = expand_ranges(filter_config.get("dendrite_tree_depth", [1, 5, 1]))
            uniform_branches = filter_config.get("uniform_branches_per_layer", 4)

            for depth in depths:
                branch_configs.append(
                    {
                        "dendrite_tree_depth": depth,
                        "dendrite_branches_per_layer": uniform_branches,
                    }
                )

        elif sweep_mode == "width_sweep":
            fixed_depth = filter_config.get("fixed_dendrite_tree_depth", 2)
            branch_counts = expand_ranges(
                filter_config.get("dendrite_branches_per_layer", [1, 8, 1])
            )

            for branch_count in branch_counts:
                branch_configs.append(
                    {
                        "dendrite_tree_depth": fixed_depth,
                        "dendrite_branches_per_layer": branch_count,
                    }
                )

        elif sweep_mode == "pattern_sweep":
            patterns = filter_config.get(
                "branch_patterns", ["uniform", "decreasing", "increasing"]
            )
            depths = expand_ranges(filter_config.get("dendrite_tree_depth", [2, 4, 1]))
            base_branches = filter_config.get("base_branches", 4)

            for depth in depths:
                for pattern in patterns:
                    if pattern == "uniform":
                        branch_config = [base_branches] * depth
                    elif pattern == "decreasing":
                        branch_config = [
                            max(1, base_branches - i) for i in range(depth)
                        ]
                    elif pattern == "increasing":
                        branch_config = [i + 1 for i in range(depth)]
                    elif pattern == "pyramid":
                        mid = depth // 2
                        branch_config = []
                        for i in range(depth):
                            if i <= mid:
                                branch_config.append(i + 1)
                            else:
                                branch_config.append(depth - i)
                    else:
                        branch_config = [base_branches] * depth

                    branch_configs.append(
                        {
                            "dendrite_tree_depth": depth,
                            "dendrite_branches_per_layer": branch_config,
                        }
                    )

        elif sweep_mode == "custom_configs":
            custom_configs = filter_config.get(
                "custom_branch_configs",
                [
                    {"depth": 2, "branches": [2, 4]},
                    {"depth": 3, "branches": [1, 2, 1]},
                ],
            )

            for config in custom_configs:
                branch_configs.append(
                    {
                        "dendrite_tree_depth": config["depth"],
                        "dendrite_branches_per_layer": config["branches"],
                    }
                )

        else:  # uniform_sweep (default)
            depths = expand_ranges(filter_config.get("dendrite_tree_depth", [1, 3, 1]))
            branch_counts = expand_ranges(
                filter_config.get("dendrite_branches_per_layer", [2, 4, 1])
            )

            for depth in depths:
                for branch_count in branch_counts:
                    branch_configs.append(
                        {
                            "dendrite_tree_depth": depth,
                            "dendrite_branches_per_layer": branch_count,
                        }
                    )

        return branch_configs

    def generate_configs(self, base_config: DictConfig) -> list[DictConfig]:
        """Generate unified sweep configurations."""
        sweep_params = base_config.get("sweep_config", {})
        experiment_settings = base_config.get("experiment_settings", {})
        filter_config = base_config.get("filter_config", {})

        # Create complete base training config by copying the full structure
        base_train_config = {
            "model": base_config.get("base_config", {}).get(
                "model", base_config.get("model", {})
            ),
            "train": base_config.get("train", {"seed": 42}),
            "task": base_config.get(
                "task",
                {
                    "dataset": "cifar10",
                    "data_path": "",
                    "parameters": {"flatten": True, "normalize": False},
                },
            ),
            "wandb": base_config.get("wandb", {"use_wandb": False}),
            "analysis": base_config.get("analysis", {}),
            "outputs": base_config.get("outputs", {}),
        }

        # Expand sweep parameters (handle ranges like [start, end, step])
        param_values = {}
        if sweep_params:  # Only iterate if sweep_params is not None/empty
            for key, value in sweep_params.items():
                expanded = expand_ranges(value)
                if not isinstance(expanded, list):
                    expanded = [expanded]
                param_values[key] = expanded

        # Handle filter_config modes
        mode = filter_config.get("mode", "default")

        # Handle branch sweeps if specified (can be combined with EI filtering)
        if mode == "branch_sweep" or filter_config.get("branch_sweep_mode"):
            # Generate branch combinations as paired configurations
            branch_configs = self._generate_branch_configurations(filter_config)
            # Add as a single parameter that will be unpacked later
            param_values["_branch_config"] = branch_configs

        # Handle EI filtering if EI parameters are present
        ee_param = "model.core_network.parameters.ee_synapses_per_branch_per_layer"
        ie_param = "model.core_network.parameters.ie_synapses_per_branch_per_layer"

        # Apply EI filtering if EI parameters present AND filtering mode specified
        if (
            ee_param in param_values
            and ie_param in param_values
            and mode in ["ei-sum", "ei-ratio", "e-const", "i-const"]
        ):
            # Apply EI filtering
            ee_values = param_values[ee_param]
            ie_values = param_values[ie_param]

            # Convert to scalars if they're single-element lists (for filtering)
            ee_scalars = [
                v[0] if isinstance(v, list) and len(v) == 1 else v for v in ee_values
            ]
            ie_scalars = [
                v[0] if isinstance(v, list) and len(v) == 1 else v for v in ie_values
            ]

            filtered_combos = self._apply_ei_filter(
                ee_scalars, ie_scalars, filter_config
            )

            # Remove individual EI params and add as paired combinations
            del param_values[ee_param]
            del param_values[ie_param]
            # Add as a special paired parameter
            param_values["_ei_config"] = [
                {"ee": combo[0], "ie": combo[1]} for combo in filtered_combos
            ]

        # Generate cartesian product of all sweep params
        param_keys = list(param_values.keys())
        param_value_lists = [param_values[key] for key in param_keys]

        if param_keys:
            all_combos = list(itertools.product(*param_value_lists))
        else:
            # No parameters to sweep - generate single config
            all_combos = [()]

        # Handle seeds and other experiment settings
        seeds_per_condition = experiment_settings.get("seeds_per_condition", 1)
        base_seed = experiment_settings.get("base_seed", 42)

        # Generate configs
        configs = []
        for combo in all_combos:
            # Create new config from base
            new_config = OmegaConf.create(deepcopy(base_train_config))

            # Apply parameter values (only if there are parameters to set)
            if param_keys:
                for key, val in zip(param_keys, combo):
                    if key == "_branch_config":
                        # Special handling for branch configurations
                        # val is a dict with dendrite_tree_depth and dendrite_branches_per_layer
                        if "model" not in new_config:
                            new_config["model"] = {}
                        if "core_network" not in new_config.model:
                            new_config.model["core_network"] = {}
                        if "parameters" not in new_config.model.core_network:
                            new_config.model.core_network["parameters"] = {}

                        new_config.model.core_network.parameters.dendrite_tree_depth = (
                            val["dendrite_tree_depth"]
                        )
                        new_config.model.core_network.parameters.dendrite_branches_per_layer = val[
                            "dendrite_branches_per_layer"
                        ]
                    elif key == "_ei_config":
                        # Special handling for EI configurations
                        # val is a dict with ee and ie values
                        if "model" not in new_config:
                            new_config["model"] = {}
                        if "core_network" not in new_config.model:
                            new_config.model["core_network"] = {}
                        if "parameters" not in new_config.model.core_network:
                            new_config.model.core_network["parameters"] = {}

                        new_config.model.core_network.parameters.ee_synapses_per_branch_per_layer = [
                            val["ee"]
                        ]
                        new_config.model.core_network.parameters.ie_synapses_per_branch_per_layer = [
                            val["ie"]
                        ]
                    else:
                        # Set nested params (e.g., model.core_network.parameters.foo = val)
                        keys = key.split(".")
                        target = new_config
                        for k in keys[:-1]:
                            if k not in target:
                                target[k] = {}
                            target = target[k]
                        target[keys[-1]] = val

            # Handle multiple seeds if specified
            if seeds_per_condition > 1:
                for seed_offset in range(seeds_per_condition):
                    seed_config = OmegaConf.create(deepcopy(new_config))
                    seed_config.train.seed = base_seed + seed_offset
                    configs.append(seed_config)
            else:
                new_config.train.seed = base_seed
                configs.append(new_config)

        return configs


# Note: Analysis should be done using the unified result_analyzer.py
# which has full capabilities including those from experiments/dendrinet_simulations/analyze_dendrinet_results.py
