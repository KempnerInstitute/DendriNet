"""Data collection from sweep result directories."""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from .config_extractor import ConfigExtractor

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects data from sweep result directories."""

    # Canonical directory structure:
    # sweep_results/
    #   ├── configs/config_*.yaml
    #   └── results/config_*/
    #       ├── performance/final.json
    #       ├── information_analysis/final (no .json)
    #       ├── weight_analysis/final
    #       └── ...

    def __init__(self, results_dir: Path):
        """
        Initialize data collector.

        Args:
            results_dir: Root directory containing sweep results
        """
        self.results_dir = Path(results_dir)
        self.configs_dir = self.results_dir / "configs"
        self.results_subdir = self.results_dir / "results"
        self.config_extractor = ConfigExtractor()

    def collect_all(self) -> pd.DataFrame:
        """
        Collect data from all configurations in the sweep.

        Returns:
            DataFrame with one row per configuration
        """
        logger.info(f"Collecting data from {self.results_dir}")

        # Find all config directories
        config_dirs = self._get_config_dirs()
        logger.info(f"Found {len(config_dirs)} configuration directories")

        if not config_dirs:
            logger.warning("No configuration directories found")
            return pd.DataFrame()

        # Collect data from each config
        all_data = []
        for i, config_dir in enumerate(config_dirs):
            if (i + 1) % 50 == 0:
                logger.info(f"Processing {i+1}/{len(config_dirs)}...")

            data = self._collect_single_config(config_dir)
            if data:
                all_data.append(data)

        logger.info(
            f"Successfully collected data from {len(all_data)}/{len(config_dirs)} configs"
        )

        return pd.DataFrame(all_data)

    def _get_config_dirs(self) -> list[Path]:
        """Get all configuration directories."""
        if not self.results_subdir.exists():
            logger.error(f"Results directory not found: {self.results_subdir}")
            return []

        # Find all config_* directories
        config_dirs = sorted(self.results_subdir.glob("config_*"))
        return [d for d in config_dirs if d.is_dir()]

    def _collect_single_config(self, config_dir: Path) -> Optional[dict]:
        """
        Collect all data for a single configuration.

        Args:
            config_dir: Path to config_X directory

        Returns:
            Dictionary with all collected data, or None if failed
        """
        # Extract config ID
        match = re.search(r"config_(\d+)", config_dir.name)
        if not match:
            logger.warning(f"Could not extract ID from {config_dir.name}")
            return None

        config_id = int(match.group(1))

        # Load config parameters
        config_yaml = self.configs_dir / f"config_{config_id}.yaml"
        if not config_yaml.exists():
            # Try alternate naming patterns
            possible_patterns = [
                f"unified_config_{config_id}.yaml",
                f"ei_sweep_config_{config_id}.yaml",
            ]
            for pattern in possible_patterns:
                alt_path = self.configs_dir / pattern
                if alt_path.exists():
                    config_yaml = alt_path
                    break

        if not config_yaml.exists():
            logger.warning(f"Config file not found for config_{config_id}")
            return None

        # Extract parameters from config
        params = self.config_extractor.extract_from_file(config_yaml)
        params["config_id"] = config_id
        params["config_file"] = str(config_yaml)

        # Collect analysis results
        params.update(self._load_performance(config_dir))
        params.update(self._load_information(config_dir))
        params.update(self._load_weights(config_dir))
        params.update(self._load_ablation(config_dir))
        params.update(self._load_snr(config_dir))
        params.update(self._load_gain(config_dir))
        params.update(self._load_noise(config_dir))
        params.update(self._load_nparams(config_dir))

        return params

    def _load_json_file(self, filepath: Path) -> Optional[dict]:
        """Load a JSON file, handling both .json and no extension."""
        # Try exact path first
        if filepath.exists():
            try:
                with open(filepath) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading {filepath}: {e}")

        # Try without .json extension (if path has .json, try removing it)
        if filepath.suffix == ".json":
            no_ext_path = filepath.with_suffix("")
            if no_ext_path.exists():
                try:
                    with open(no_ext_path) as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading {no_ext_path}: {e}")

        # Try adding .json extension (if path doesn't have it)
        if filepath.suffix != ".json":
            json_path = filepath.with_suffix(".json")
            if json_path.exists():
                try:
                    with open(json_path) as f:
                        return json.load(f)
                except Exception as e:
                    logger.warning(f"Error reading {json_path}: {e}")

        return None

    def _load_performance(self, config_dir: Path) -> dict:
        """Load performance metrics."""
        # perf_file = config_dir / "performance" / "final.json"
        # Search recursively for performance/final.json
        perf_files = list(config_dir.rglob("performance/final.json"))

        if not perf_files:
            # Try without .json extension
            perf_files = list(config_dir.rglob("performance/final"))

        if perf_files:
            perf_file = perf_files[0]  # Use first match found
        else:
            # Fallback to original path
            perf_file = config_dir / "performance" / "final.json"

        data = self._load_json_file(perf_file)

        if not data:
            return {}

        # Flatten performance data
        result = {}
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested metrics (e.g., {'accuracy': ..., 'auc': ...})
                for metric, val in value.items():
                    result[f"{metric}_{key}"] = val  # e.g., accuracy_test
            else:
                result[key] = value

        return result

    def _load_information(self, config_dir: Path) -> dict:
        """Load information analysis metrics."""
        # Search recursively for information_analysis/final.json
        info_files = list(config_dir.rglob("information_analysis/final.json"))

        if not info_files:
            # Try without .json extension
            info_files = list(config_dir.rglob("information_analysis/final"))

        if info_files:
            info_file = info_files[0]  # Use first match found
        else:
            # Fallback to original path
            info_file = config_dir / "information_analysis" / "final"

        data = self._load_json_file(info_file)

        if not data:
            return {}

        result = {}

        # Basic MI
        if "basic_mi" in data and isinstance(data["basic_mi"], dict):
            for mi_key, mi_value in data["basic_mi"].items():
                if isinstance(mi_value, (int, float)):
                    # Convert I(E;C) format to mi_E_C
                    # Also handles: I(Vb;C) -> mi_Vb_C, I(E_lin;C) -> mi_E_lin_C
                    normalized_key = (
                        mi_key.replace("I(", "mi_")
                        .replace(";", "_")
                        .replace(")", "")
                        .replace("Vout", "V")
                        .replace(",", "_")
                    )
                    result[normalized_key] = mi_value

        # Pairwise MI
        if "pairwise_mi" in data and isinstance(data["pairwise_mi"], dict):
            for mi_key, mi_value in data["pairwise_mi"].items():
                if isinstance(mi_value, (int, float)):
                    normalized_key = (
                        mi_key.replace("I(", "mi_")
                        .replace(";", "_")
                        .replace(")", "")
                        .replace("Vout", "V")
                    )
                    result[normalized_key] = mi_value

        # Conditional MI
        if "conditional_mi" in data and isinstance(data["conditional_mi"], dict):
            for mi_key, mi_value in data["conditional_mi"].items():
                if isinstance(mi_value, (int, float)):
                    normalized_key = (
                        mi_key.replace("I(", "mi_")
                        .replace(";", "_")
                        .replace(")", "")
                        .replace("|", "_given_")
                        .replace("Vout", "V")
                    )
                    result[normalized_key] = mi_value

        # Layer statistics
        if "layer_statistics" in data and isinstance(data["layer_statistics"], dict):
            layer_names: list[str] = [
                name for name in data["layer_statistics"].keys() 
                if "inhibitory_cells" not in name
            ]
            depths: list[int] = list(range(len(layer_names)))
            depths.reverse()
            for i, layer_name in enumerate(layer_names):
                layer_data: dict[str, float] = data["layer_statistics"][layer_name]
                if isinstance(layer_data, dict):
                    # Extract depth from layer name
                    # depth = self._extract_depth(layer_name)
                    depth = depths[i]
                    for mi_key, mi_value in layer_data.items():
                        if isinstance(mi_value, (int, float)):
                            base_key = (
                                mi_key.replace("I(", "layer_mi_")
                                .replace(";", "_")
                                .replace(")", "")
                                .replace(",", "_")
                                .replace("|", "_given_")
                                .replace("Vout", "V")
                                .replace("_mean", "")
                            )
                            result[f"{base_key}_depth{depth}"] = mi_value

        return result

    def _load_weights(self, config_dir: Path) -> dict:
        """Load weight analysis metrics."""
        # weight_file = config_dir / "weight_analysis" / "final"
        # Search recursively for performance/final.json
        weight_files = list(config_dir.rglob("weight_analysis/final.json"))

        if not weight_files:
            # Try without .json extension
            weight_files = list(config_dir.rglob("weight_analysis/final"))

        if weight_files:
            weight_file = weight_files[0]  # Use first match found
        else:
            # Fallback to original path
            weight_file = config_dir / "weight_analysis" / "final"

        data = self._load_json_file(weight_file)

        if not data:
            return {}

        result = {}

        # Global statistics
        if "global_statistics" in data and isinstance(data["global_statistics"], dict):
            for key, value in data["global_statistics"].items():
                if isinstance(value, (int, float)):
                    result[f"weight_global_{key}"] = value

        # Layer statistics
        if "layer_statistics" in data and isinstance(data["layer_statistics"], dict):
            for layer_name, layer_data in data["layer_statistics"].items():
                if isinstance(layer_data, dict):
                    layer_key = self._format_layer_key(layer_name)
                    for metric, value in layer_data.items():
                        if isinstance(value, (int, float)):
                            safe_metric = metric.replace(".", "_").replace(" ", "_")
                            result[f"weight_{safe_metric}_{layer_key}"] = value

        return result

    def _load_ablation(self, config_dir: Path) -> dict:
        """Load ablation analysis metrics.

        Loads the canonical 3-level ablation schema
        (method -> target -> module -> metrics) and flattens to sweep-ready
        columns via :func:`ablation_columns.flatten_ablation_results`.
        """
        from dendritic_modeling.scripts.sweeps.utils.ablation_columns import (
            flatten_ablation_module_results,
            flatten_ablation_results,
        )

        ablation_files = list(config_dir.rglob("ablation_analysis/final.json"))
        if not ablation_files:
            ablation_files = list(config_dir.rglob("ablation_analysis/final"))
        if ablation_files:
            ablation_file = ablation_files[0]
        else:
            ablation_file = config_dir / "ablation_analysis" / "final"

        data = self._load_json_file(ablation_file)
        if not data:
            return {}

        result: dict[str, float] = {}

        if "layer" in data and isinstance(data["layer"], dict):
            for method, method_data in data["layer"].items():
                for target, target_data in method_data.items():
                    for layer_data in target_data.values():
                        depth = layer_data.get("depth", 0)
                        for metric, val in layer_data.items():
                            if metric != "depth":
                                col_name = "ablation_"
                                col_name += method.replace("_", " ") + "_"
                                col_name += target.replace("_", " ") + "_"
                                col_name += metric.replace("_", " ") + "_"
                                col_name += f"depth{depth}"
                                result[col_name] = val
        return result

    @staticmethod
    def _ensure_3level(layer_dict: dict) -> dict:
        """Convert legacy 2-level ablation schema to 3-level if needed.

        2-level: ``{"lesion_excitation": {module: metrics}}``
        3-level: ``{"lesion": {"excitation": {module: metrics}}}``

        Already 3-level dicts are returned unchanged.
        """
        known_targets = {"all_synapses", "excitation", "inhibition", "upstream"}
        # Convert any legacy 2-level entries while preserving already-canonical
        # 3-level entries in mixed payloads.
        nested: dict = {}
        for key, modules in layer_dict.items():
            if not isinstance(modules, dict):
                continue
            if set(modules.keys()) & known_targets:
                nested[key] = modules
                continue
            key_norm = str(key).replace(" ", "_")
            matched = False
            for target in known_targets:
                suffix = f"_{target}"
                if key_norm.endswith(suffix):
                    method = key_norm[: -len(suffix)]
                    if method:
                        nested.setdefault(method, {})[target] = modules
                        matched = True
                        break
            if not matched:
                nested[key] = modules
        return nested

    def _load_snr(self, config_dir: Path) -> dict:
        """Load compartment SNR analysis metrics."""
        # compartment_snr_file = config_dir / "compartment_snr_analysis" / "final"
        # Search recursively for performance/final.json
        compartment_snr_files = list(
            config_dir.rglob("compartment_snr_analysis/final.json")
        )

        if not compartment_snr_files:
            # Try without .json extension
            compartment_snr_files = list(
                config_dir.rglob("compartment_snr_analysis/final")
            )

        if compartment_snr_files:
            compartment_snr_file = compartment_snr_files[0]  # Use first match found
        else:
            # Fallback to original path
            compartment_snr_file = config_dir / "compartment_snr_analysis" / "final"

        data: dict[str, dict[str, dict[str, float]]] = self._load_json_file(
            compartment_snr_file
        )

        if not data:
            return {}

        result = {}

        prefixes = ["exc", "inh", "upstream", "vout"]
        metrics = ["d_max", "d_total", "n_sig_dirs", "ratio_sig_dirs"]

        if "input_analysis" in data.keys():
            input_data = data["input_analysis"]
            for prefix in ["exc", "inh"]:
                for metric in metrics:
                    field_name = f"{prefix}_inputs_{metric}"
                    if field_name in input_data:
                        result[f"snr_{field_name}"] = input_data[field_name]

        # Handle layer analysis format
        if "layer_analysis" in data.keys():
            if "layer_statistics" in data["layer_analysis"].keys():
                for layer_data in data["layer_analysis"]["layer_statistics"].values():
                    depth = layer_data["depth"]
                    for prefix in prefixes:
                        for metric in metrics:
                            field_name = f"{prefix}_{metric}"
                            if field_name in layer_data:
                                result[f"snr_{field_name}_depth{depth}"] = layer_data[
                                    field_name
                                ]

        return result

    def _load_gain(self, config_dir: Path) -> dict:
        """Load multiplicative gain analysis metrics."""
        multiplicative_gain_files = list(
            config_dir.rglob("multiplicative_gain_analysis/final.json")
        )

        if not multiplicative_gain_files:
            multiplicative_gain_files = list(
                config_dir.rglob("multiplicative_gain_analysis/final")
            )

        if multiplicative_gain_files:
            multiplicative_gain_file = multiplicative_gain_files[0]
        else:
            multiplicative_gain_file = (
                config_dir / "multiplicative_gain_analysis" / "final"
            )

        data = self._load_json_file(multiplicative_gain_file)

        if not data:
            return {}

        result = {}

        gain_factors: list[float] = data["gain_factors"]
        
        raw_scores: dict[str, list[float]] = data["raw_scores"]
        
        id_scores: dict[str, float] = data["ID"]
        ood_lower_scores: dict[str, float] = data["OOD_lower"]
        ood_upper_scores: dict[str, float] = data["OOD_upper"]

        metrics = list(raw_scores.keys())
        for metric in metrics:
            # collect raw scores
            for i, gain_factor in enumerate(gain_factors):
                result[f"mult_gain_{metric}_gf{gain_factor}"] = raw_scores[metric][i]
            
            # collect id scores
            result[f"mult_gain_{metric}_id"] = id_scores[metric]

            # collect ood lower scores
            result[f"mult_gain_{metric}_ood_lower"] = ood_lower_scores[metric]

            # collect ood upper scores
            result[f"mult_gain_{metric}_ood_upper"] = ood_upper_scores[metric]

        return result

    def _load_noise(self, config_dir: Path) -> dict:
        """Load noise perturbation metrics."""
        # noise_file = config_dir / "noise_perturbation_analysis" / "final"
        # Search recursively for performance/final.json
        noise_files = list(config_dir.rglob("noise_perturbation_analysis/final.json"))

        if not noise_files:
            # Try without .json extension
            noise_files = list(config_dir.rglob("noise_perturbation_analysis/final"))

        if noise_files:
            noise_file = noise_files[0]  # Use first match found
        else:
            # Fallback to original path
            noise_file = config_dir / "noise_perturbation_analysis" / "final"

        data = self._load_json_file(noise_file)

        if not data:
            return {}

        result = {}

        # Handle noise perturbation format
        for noise_type, noise_dict in data.items():
            if isinstance(noise_dict, dict):
                for noise_level, noise_data in noise_dict.items():
                    if isinstance(noise_data, dict):
                        for metric, value in noise_data.items():
                            metric_clean = metric.replace("_", "")
                            if isinstance(value, list):
                                for i, v in enumerate(value):
                                    result[
                                        f"noise_{noise_type}_{metric_clean}_{noise_level}_sample{i}"
                                    ] = v
                            else:
                                result[
                                    f"noise_{noise_type}_{metric_clean}_{noise_level}"
                                ] = value

        return result

    def _load_nparams(self, config_dir: Path) -> dict:
        """Load parameter count."""
        # nparams_file = config_dir / "nparams"
        # Search recursively for performance/final.json
        nparams_files = list(config_dir.rglob("nparams.json"))

        if not nparams_files:
            # Try without .json extension
            nparams_files = list(config_dir.rglob("nparams"))

        if nparams_files:
            nparams_file = nparams_files[0]  # Use first match found
        else:
            # Fallback to original path
            nparams_file = config_dir / "nparams"

        data = self._load_json_file(nparams_file)

        if data and "nparams" in data:
            return {"nparams": data["nparams"]}

        return {}

    def _extract_depth(self, layer_name: str) -> int:
        """Extract depth/layer index from layer name."""
        try:
            if "branch_layers." in layer_name:
                return int(layer_name.split("branch_layers.")[-1].split(".")[0])
            elif "layer_" in layer_name:
                return int(layer_name.split("layer_")[-1])
            return 0
        except (ValueError, IndexError):
            return 0

    def _format_layer_key(self, layer_name: str) -> str:
        """Format layer name to a consistent key."""
        if "branch_layers." in layer_name:
            branch_idx = int(layer_name.split("branch_layers.")[1].split(".")[0])
            return f"layer_{branch_idx}"
        elif layer_name == "synthetic_soma":
            return "soma"
        else:
            return layer_name.replace(".", "_")
