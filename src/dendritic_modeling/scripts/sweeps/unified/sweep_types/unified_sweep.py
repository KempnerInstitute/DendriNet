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
import logging
from copy import deepcopy

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from dendritic_modeling.scripts.sweeps.unified.base_sweep import BaseSweepGenerator

logger = logging.getLogger(__name__)


_RANGE_FORMS = ("range", "linspace", "geomspace", "values")


def _numeric(value) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    )


def _reads_as_legacy_range(values) -> bool:
    """Whether earlier versions silently expanded this list as [start, stop, step]."""
    if len(values) != 3 or not all(_numeric(x) for x in values):
        return False
    start, stop, step = values
    if step > 0:
        return start <= stop and step <= (stop - start + 1e-9)
    if step < 0:
        return start >= stop and abs(step) <= (start - stop + 1e-9)
    return False


def _range_values(spec: dict, name: str) -> list:
    """Inclusive arithmetic progression from start to stop."""
    try:
        start, stop, step = spec["start"], spec["stop"], spec["step"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"{name}: range needs start, stop and step") from error
    if not all(_numeric(x) for x in (start, stop, step)) or step == 0:
        raise ValueError(
            f"{name}: range start, stop and step must be numbers, step != 0"
        )
    count = int(np.floor((stop - start) / step + 1e-9)) + 1
    if count < 1:
        raise ValueError(f"{name}: range {spec} is empty")
    if all(isinstance(x, (int, np.integer)) for x in (start, stop, step)):
        return [int(start + index * step) for index in range(count)]
    return [round(float(start + index * step), 12) for index in range(count)]


def expand_axis(value, *, name: str = "sweep axis") -> list:
    """Return the values of one sweep axis.

    A list is taken literally. Ranges are written explicitly as a one-key
    mapping: ``{range: {start, stop, step}}`` (inclusive stop),
    ``{linspace: {start, stop, num}}``, ``{geomspace: {start, stop, num}}`` or
    ``{values: [...]}``. A scalar is a one-value axis. A plain three-number
    list that earlier versions silently read as ``[start, stop, step]`` is
    ambiguous and raises; write it with ``range`` or ``values``.
    """
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if (
        isinstance(value, dict)
        and len(value) == 1
        and next(iter(value)) in _RANGE_FORMS
    ):
        form, spec = next(iter(value.items()))
        if form == "values":
            values = list(spec)
        elif form == "range":
            values = _range_values(spec, name)
        else:
            try:
                start, stop, num = spec["start"], spec["stop"], int(spec["num"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"{name}: {form} needs start, stop and num") from error
            space = np.linspace if form == "linspace" else np.geomspace
            values = [round(float(x), 12) for x in space(start, stop, num)]
        if not values:
            raise ValueError(f"{name}: the axis has no values")
        return values
    if isinstance(value, (list, tuple)):
        values = list(value)
        if _reads_as_legacy_range(values):
            raise ValueError(
                f"{name}={values!r} is ambiguous: earlier versions expanded it as "
                "[start, stop, step]. Write {range: {start, stop, step}} for a "
                "range or {values: [...]} for these literal values."
            )
        if not values:
            raise ValueError(f"{name}: the axis has no values")
        return values
    return [value]


_FILTER_MODES = frozenset(
    {"default", "all", "branch_sweep", "ei-sum", "ei-ratio", "e-const", "i-const"}
)
_BRANCH_SWEEP_MODES = frozenset(
    {"uniform_sweep", "depth_sweep", "width_sweep", "pattern_sweep", "custom_configs"}
)
_BRANCH_PATTERNS = frozenset({"uniform", "decreasing", "increasing", "pyramid"})


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
    - ei-sum, ei-ratio, e-const, i-const: EI filtering modes
    """

    def __init__(self):
        super().__init__("unified")

    @staticmethod
    def _dot_path_index(part: str, full_key: str) -> int:
        if not str(part).isdigit():
            raise ValueError(
                f"Sweep path {full_key!r} enters a list but {part!r} is not "
                "a non-negative integer index."
            )
        return int(part)

    @classmethod
    def _set_by_dot_path(cls, config: DictConfig, key: str, value) -> None:
        """Set a dot-path value, supporting numeric indices inside list nodes."""
        parts = str(key).split(".")
        if not parts or any(part == "" for part in parts):
            raise ValueError(f"Invalid empty component in sweep parameter {key!r}")

        target = config
        for idx, part in enumerate(parts[:-1]):
            next_part = parts[idx + 1]
            if isinstance(target, (list, ListConfig)):
                list_idx = cls._dot_path_index(part, key)
                if list_idx >= len(target):
                    raise IndexError(
                        f"Sweep path {key!r} index {list_idx} is out of range "
                        f"for list component {parts[:idx]!r}"
                    )
                target = target[list_idx]
                continue

            if part not in target or target[part] is None:
                target[part] = [] if next_part.isdigit() else {}
            target = target[part]

        last = parts[-1]
        if isinstance(target, (list, ListConfig)):
            list_idx = cls._dot_path_index(last, key)
            if list_idx >= len(target):
                raise IndexError(f"Sweep path {key!r} index {list_idx} is out of range")
            target[list_idx] = value
        else:
            target[last] = value

    def _warn_if_param_group_lr_axis_is_effectively_noop(
        self, sweep_params: DictConfig, base_train_config: DictConfig
    ) -> None:
        """Warn if sweeping param_groups.lr likely has little effect.

        When `split_params=true` and group-specific learning rates are fixed
        (`topk_lr`, `blocklinear_lr`, `reactivation_lr`, `decoder_lr`), sweeping
        only `param_groups.lr` mostly affects uncategorized params.
        """
        lr_key = "training.main.common.param_groups.lr"
        group_lr_keys = [
            "training.main.common.param_groups.topk_lr",
            "training.main.common.param_groups.blocklinear_lr",
            "training.main.common.param_groups.reactivation_lr",
            "training.main.common.param_groups.decoder_lr",
        ]

        if lr_key not in sweep_params:
            return
        if any(key in sweep_params for key in group_lr_keys):
            return

        split_params = OmegaConf.select(
            base_train_config, "training.main.common.param_groups.split_params"
        )
        if split_params is not True:
            return

        fixed_group_lrs = {}
        for key in group_lr_keys:
            value = OmegaConf.select(base_train_config, key)
            if value is not None:
                fixed_group_lrs[key.split(".")[-1]] = value

        if not fixed_group_lrs:
            return

        logger.warning(
            "Sweep parameter '%s' may be mostly a no-op with split_params=true; "
            "group-specific LRs are fixed and not swept: %s",
            lr_key,
            fixed_group_lrs,
        )

    def _warn_if_optimizer_lr_axis_is_effectively_noop(
        self, sweep_params: DictConfig, base_train_config: DictConfig
    ) -> None:
        """Warn when an optimizer-LR sweep is shadowed by parameter-group LRs.

        ``BaseModel.get_param_groups`` attaches an explicit ``lr`` to every
        optimizer parameter group. PyTorch gives those group-level values
        precedence over ``training.main.optimizer.lr``. Consequently, varying
        only the optimizer-level value does not change the effective learning
        rate when the standard parameter-group configuration is present.
        """
        optimizer_lr_key = "training.main.optimizer.lr"
        if optimizer_lr_key not in sweep_params:
            return

        param_groups = OmegaConf.select(
            base_train_config, "training.main.common.param_groups"
        )
        if param_groups is None:
            return

        configured_lrs = {}
        for key in (
            "lr",
            "topk_lr",
            "blocklinear_lr",
            "reactivation_lr",
            "decoder_lr",
            "encoder_lr",
        ):
            value = OmegaConf.select(param_groups, key)
            if value is not None:
                configured_lrs[key] = value

        if not configured_lrs:
            return

        logger.warning(
            "Sweep parameter '%s' is ineffective with the current standard "
            "training path because model parameter groups carry explicit "
            "learning rates: %s. Sweep the relevant "
            "training.main.common.param_groups.* fields instead.",
            optimizer_lr_key,
            configured_lrs,
        )

    def validate_config(self, config: DictConfig) -> bool:
        """Validate unified sweep configuration."""
        # Check for sweep_config or filter_config (at least one must generate parameters)
        sweep_config = config.get("sweep_config", {})
        sweep_variants = config.get("sweep_variants", [])
        filter_config = config.get("filter_config", {})

        self._normalize_sweep_variants(sweep_variants)

        filter_mode = filter_config.get("mode", "default")
        if filter_mode not in _FILTER_MODES:
            raise ValueError(
                f"filter_config.mode={filter_mode!r} is not supported; choose one of "
                + ", ".join(sorted(_FILTER_MODES))
            )
        branch_mode = filter_config.get("branch_sweep_mode")
        if branch_mode is not None and branch_mode not in _BRANCH_SWEEP_MODES:
            raise ValueError(
                f"filter_config.branch_sweep_mode={branch_mode!r} is not supported; "
                "choose one of " + ", ".join(sorted(_BRANCH_SWEEP_MODES))
            )

        # If the ordinary grid and named variants are empty, check whether the
        # filter can generate parameters.
        if not sweep_config and not sweep_variants:
            if filter_mode not in ["branch_sweep"]:
                raise ValueError(
                    "At least one of sweep_config, sweep_variants, or "
                    "filter_config.mode='branch_sweep' is required"
                )

        return True

    @staticmethod
    def _normalize_sweep_variants(variants) -> list[dict]:
        """Validate named correlated override bundles.

        A variant is one condition, not another Cartesian axis internally. Its
        dot-path overrides are applied together before ordinary ``sweep_config``
        axes, which may therefore refine a value shared across all variants.
        """

        if variants is None:
            return []
        if hasattr(variants, "_content"):
            variants = OmegaConf.to_container(variants, resolve=True)
        if not isinstance(variants, list):
            raise TypeError("sweep_variants must be a list of named variants")

        normalized: list[dict] = []
        names: set[str] = set()
        for index, declaration in enumerate(variants):
            if not isinstance(declaration, dict):
                raise TypeError(
                    f"sweep_variants[{index}] must be a mapping with name and overrides"
                )
            name = str(declaration.get("name", "")).strip()
            if not name:
                raise ValueError(f"sweep_variants[{index}].name cannot be empty")
            if name in names:
                raise ValueError(f"Duplicate sweep variant name: {name!r}")
            names.add(name)

            overrides = declaration.get("overrides", {})
            if not isinstance(overrides, dict):
                raise TypeError(
                    f"sweep_variants[{index}].overrides must be a dot-path mapping"
                )
            for path in overrides:
                if not str(path).strip() or str(path).startswith("_"):
                    raise ValueError(
                        f"Invalid override path {path!r} in sweep variant {name!r}"
                    )
            normalized.append({"name": name, "overrides": dict(overrides)})
        return normalized

    @classmethod
    def _set_condition_seed(
        cls,
        config: DictConfig,
        *,
        seed: int,
        seed_paths,
    ) -> None:
        """Set the replicate seed and optional fixed-topology seed aliases."""

        if "experiment" not in config:
            config["experiment"] = {}
        config.experiment.seed = int(seed)

        if seed_paths is None:
            return
        if hasattr(seed_paths, "_content"):
            seed_paths = OmegaConf.to_container(seed_paths, resolve=True)
        if not isinstance(seed_paths, list) or any(
            not isinstance(path, str) or not path.strip() for path in seed_paths
        ):
            raise TypeError(
                "experiment_settings.seed_paths must be a list of non-empty dot paths"
            )
        for path in seed_paths:
            cls._set_by_dot_path(config, path, int(seed))

    def _apply_ei_filter(
        self, ee_values: list, ie_values: list, filter_config: dict
    ) -> list[tuple]:
        """
        Apply EI-specific filtering modes using unified 'value' parameter.

        value meaning by mode:
        - ei-sum: target sum (E + I = value), within ``tolerance``
        - e-const: constant E value (E = value), which must be on the E axis
        - i-const: constant I value (I = value), which must be on the I axis
        - ei-ratio: target E/I ratio (E/I = value), within ``tolerance``
        """
        mode = filter_config.get("mode", "all")
        if mode in {"all", "default"}:
            return list(itertools.product(ee_values, ie_values))

        value = filter_config.get("value")
        if value is None:
            raise ValueError(f"filter_config.mode={mode!r} needs filter_config.value")
        if mode in {"ei-sum", "ei-ratio"}:
            tolerance = filter_config.get("tolerance")
            if tolerance is None:
                raise ValueError(
                    f"filter_config.mode={mode!r} needs filter_config.tolerance"
                )
        if mode == "ei-sum":
            combos = [
                (ee, ie)
                for ee in ee_values
                for ie in ie_values
                if abs((ee + ie) - value) <= tolerance
            ]
        elif mode == "e-const":
            if value not in ee_values:
                raise ValueError(f"e-const value {value!r} is not on the E axis")
            combos = [(value, ie) for ie in ie_values]
        elif mode == "i-const":
            if value not in ie_values:
                raise ValueError(f"i-const value {value!r} is not on the I axis")
            combos = [(ee, value) for ee in ee_values]
        elif mode == "ei-ratio":
            combos = [
                (ee, ie)
                for ee in ee_values
                for ie in ie_values
                if ie > 0 and abs((ee / ie) - value) <= tolerance
            ]
        else:
            raise ValueError(f"filter_config.mode={mode!r} is not an E/I filter")
        if not combos:
            raise ValueError(
                f"filter_config.mode={mode!r} with value {value!r} selects no E/I "
                "combination"
            )
        return combos

    def _generate_branch_configurations(self, filter_config: dict) -> list[dict]:
        """
        Generate branch configurations as paired depth+branch combinations.
        Returns a list of dictionaries, each containing a complete branch configuration.
        """
        branch_configs = []
        sweep_mode = filter_config.get("branch_sweep_mode", "uniform_sweep")

        def required(key: str):
            if key not in filter_config:
                raise ValueError(
                    f"filter_config.branch_sweep_mode={sweep_mode!r} needs "
                    f"filter_config.{key}"
                )
            return filter_config[key]

        def axis(key: str) -> list:
            return expand_axis(required(key), name=f"filter_config.{key}")

        if sweep_mode == "depth_sweep":
            depths = axis("dendrite_tree_depth")
            uniform_branches = required("uniform_branches_per_layer")

            for depth in depths:
                branch_configs.append(
                    {
                        "dendrite_tree_depth": depth,
                        "dendrite_branches_per_layer": uniform_branches,
                    }
                )

        elif sweep_mode == "width_sweep":
            fixed_depth = required("fixed_dendrite_tree_depth")
            branch_counts = axis("dendrite_branches_per_layer")

            for branch_count in branch_counts:
                branch_configs.append(
                    {
                        "dendrite_tree_depth": fixed_depth,
                        "dendrite_branches_per_layer": branch_count,
                    }
                )

        elif sweep_mode == "pattern_sweep":
            patterns = list(required("branch_patterns"))
            unknown = sorted(set(patterns) - _BRANCH_PATTERNS)
            if unknown:
                raise ValueError(
                    f"Unknown filter_config.branch_patterns {unknown}; choose from "
                    + ", ".join(sorted(_BRANCH_PATTERNS))
                )
            depths = axis("dendrite_tree_depth")
            base_branches = required("base_branches")

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
                    else:  # pyramid
                        mid = depth // 2
                        branch_config = []
                        for i in range(depth):
                            if i <= mid:
                                branch_config.append(i + 1)
                            else:
                                branch_config.append(depth - i)

                    branch_configs.append(
                        {
                            "dendrite_tree_depth": depth,
                            "dendrite_branches_per_layer": branch_config,
                        }
                    )

        elif sweep_mode == "custom_configs":
            custom_configs = required("custom_branch_configs")

            for config in custom_configs:
                branch_configs.append(
                    {
                        "dendrite_tree_depth": config["depth"],
                        "dendrite_branches_per_layer": config["branches"],
                    }
                )

        else:  # uniform_sweep (default)
            depths = axis("dendrite_tree_depth")
            branch_counts = axis("dendrite_branches_per_layer")

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
        sweep_variants = self._normalize_sweep_variants(
            base_config.get("sweep_variants", [])
        )
        experiment_settings = base_config.get("experiment_settings", {})
        filter_config = base_config.get("filter_config", {})

        # Every generated config starts from the sweep file's base_config.
        base_train_config = base_config.get("base_config")
        if not base_train_config:
            raise ValueError(
                "Sweep files need a base_config section holding the complete "
                "training config that the sweep axes modify"
            )
        base_train_config = OmegaConf.create(deepcopy(base_train_config))

        # Expand sweep axes (lists are literal; ranges are explicit mappings).
        param_values = {}
        if sweep_variants:
            param_values["_sweep_variant"] = sweep_variants
        self._warn_if_param_group_lr_axis_is_effectively_noop(
            sweep_params=sweep_params,
            base_train_config=base_train_config,
        )
        self._warn_if_optimizer_lr_axis_is_effectively_noop(
            sweep_params=sweep_params,
            base_train_config=base_train_config,
        )
        if sweep_params:  # Only iterate if sweep_params is not None/empty
            for key, value in sweep_params.items():
                param_values[key] = expand_axis(value, name=f"sweep_config.{key}")

        # Handle filter_config modes
        mode = filter_config.get("mode", "default")

        # Handle branch sweeps if specified (can be combined with EI filtering)
        if mode == "branch_sweep" or filter_config.get("branch_sweep_mode"):
            # Generate branch combinations as paired configurations
            branch_configs = self._generate_branch_configurations(filter_config)
            # Add as a single parameter that will be unpacked later
            param_values["_branch_config"] = branch_configs

        # Handle EI filtering if EI parameters are present (structured E/I
        # connectivity or the population-network defaults path).
        ee_param_new = "model.core.connectivity.ee_synapses_per_branch_per_layer"
        ie_param_new = "model.core.connectivity.ie_synapses_per_branch_per_layer"
        ee_param_population = next(
            (
                key
                for key in param_values
                if key.endswith(".population_defaults.ff_excitatory_synapses")
            ),
            None,
        )
        ie_param_population = next(
            (
                key
                for key in param_values
                if key.endswith(".population_defaults.ff_inhibitory_synapses")
            ),
            None,
        )

        # Determine which parameter format is being used
        if ee_param_population is not None and ie_param_population is not None:
            ee_param = ee_param_population
            ie_param = ie_param_population
        else:
            ee_param = ee_param_new
            ie_param = ie_param_new

        # Apply EI filtering if EI parameters present AND filtering mode specified
        ei_filter = mode in ["ei-sum", "ei-ratio", "e-const", "i-const"]
        if ei_filter and not (ee_param in param_values and ie_param in param_values):
            raise ValueError(
                f"filter_config.mode={mode!r} needs both E/I axes in sweep_config "
                "(model.core.connectivity.ee/ie_synapses_per_branch_per_layer or "
                "the population_defaults ff_excitatory/ff_inhibitory_synapses paths)"
            )
        if ei_filter:
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
                {"ee": combo[0], "ie": combo[1], "ee_key": ee_param, "ie_key": ie_param}
                for combo in filtered_combos
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
        seed_paths = experiment_settings.get("seed_paths", [])

        # Generate configs
        configs = []
        for combo in all_combos:
            # Create new config from base
            new_config = OmegaConf.create(deepcopy(base_train_config))

            # Apply parameter values (only if there are parameters to set)
            if param_keys:
                # Correlated overrides define the condition recipe. Apply them
                # first; ordinary sweep axes can then refine shared settings
                # such as learning rate or batch size across every recipe.
                if "_sweep_variant" in param_keys:
                    variant = combo[param_keys.index("_sweep_variant")]
                    for path, value in variant["overrides"].items():
                        self._set_by_dot_path(new_config, path, value)
                    new_config["_sweep_variant"] = variant["name"]

                for key, val in zip(param_keys, combo):
                    if key == "_sweep_variant":
                        continue
                    if key == "_branch_config":
                        # Special handling for branch configurations
                        # val is a dict with dendrite_tree_depth and dendrite_branches_per_layer
                        if "model" not in new_config:
                            new_config["model"] = {}

                        branches = val.get("dendrite_branches_per_layer")
                        depth = val.get("dendrite_tree_depth")
                        # Allow either a scalar (uniform branches) or an
                        # explicit per-level list.
                        if isinstance(branches, (int, np.integer)):
                            if depth is None:
                                branches_list = [int(branches)]
                            else:
                                branches_list = [int(branches)] * int(depth)
                        else:
                            branches_list = branches

                        core_cfg = new_config.model.get("core", {})
                        if (
                            isinstance(core_cfg, (dict, DictConfig))
                            and core_cfg.get("type") == "population_network"
                        ):
                            layers = core_cfg.get("population_network", {}).get(
                                "layers", []
                            )
                            for layer in layers:
                                for population in layer.get("populations", []):
                                    polarity = str(
                                        population.get("polarity", "excitatory")
                                    ).lower()
                                    if polarity == "excitatory":
                                        population["branch_factors"] = branches_list
                        else:
                            # Structured E/I config - branch factors live in
                            # model.core.architecture.
                            if "core" not in new_config.model:
                                new_config.model["core"] = {}
                            if "architecture" not in new_config.model.core:
                                new_config.model.core["architecture"] = {}
                            new_config.model.core.architecture.excitatory_branch_factors = (
                                branches_list
                            )
                    elif key == "_ei_config":
                        # Special handling for EI configurations
                        # val is a dict with ee and ie values
                        if "model" not in new_config:
                            new_config["model"] = {}

                        ee_key = val["ee_key"]
                        ie_key = val["ie_key"]
                        if "population_network" in ee_key:
                            self._set_by_dot_path(new_config, ee_key, val["ee"])
                            self._set_by_dot_path(new_config, ie_key, val["ie"])
                        else:
                            self._set_by_dot_path(new_config, ee_key, [val["ee"]])
                            self._set_by_dot_path(new_config, ie_key, [val["ie"]])
                    else:
                        # Set nested params (e.g., model.core.layers.0.foo = val).
                        self._set_by_dot_path(new_config, key, val)

            # Handle multiple seeds if specified
            if seeds_per_condition > 1:
                for seed_offset in range(seeds_per_condition):
                    seed_config = OmegaConf.create(deepcopy(new_config))
                    self._set_condition_seed(
                        seed_config,
                        seed=base_seed + seed_offset,
                        seed_paths=seed_paths,
                    )

                    # Add metadata for sweep result tracking
                    seed_config["_sweep_config_id"] = f"config_{len(configs)}"
                    configs.append(seed_config)
            else:
                self._set_condition_seed(
                    new_config,
                    seed=base_seed,
                    seed_paths=seed_paths,
                )

                # Add metadata for sweep result tracking
                new_config["_sweep_config_id"] = f"config_{len(configs)}"
                configs.append(new_config)

        return configs


# Analysis should be done using the unified result_analyzer.py
