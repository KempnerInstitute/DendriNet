"""Configuration parameter extraction from sweep configs."""

import logging
from pathlib import Path

from omegaconf import ListConfig, OmegaConf

logger = logging.getLogger(__name__)


class ConfigExtractor:
    """Extracts parameters from configuration files."""

    def __init__(self):
        """Initialize the config extractor."""

    def extract_from_file(self, config_path: Path) -> dict:
        """
        Extract parameters from a config file.

        Args:
            config_path: Path to config YAML file

        Returns:
            Dictionary of extracted parameters
        """
        try:
            config = OmegaConf.load(config_path)
            return self.extract_from_config(config)
        except Exception as e:
            logger.error(f"Failed to load config {config_path}: {e}")
            return {}

    def extract_from_config(self, config) -> dict:
        """
        Extract all relevant parameters from a loaded config.

        Args:
            config: OmegaConf config object

        Returns:
            Dictionary of extracted parameters
        """
        params = {}

        # Extract model parameters
        params.update(self._extract_model_params(config))

        # Extract training parameters
        params.update(self._extract_training_params(config))

        # Extract experiment parameters
        params.update(self._extract_experiment_params(config))

        return params

    def _extract_model_params(self, config) -> dict:
        """Extract model-related parameters."""
        params = {}

        try:
            if hasattr(config, "model") and hasattr(config.model, "core"):
                core = config.model.core
                if hasattr(core, "type"):
                    # Canonical architecture selector (preferred over legacy booleans)
                    core_type = str(core.type)
                    params["network_type"] = core_type

                # Connectivity parameters
                if hasattr(core, "connectivity"):
                    conn = core.connectivity
                    if hasattr(conn, "ee_synapses_per_branch_per_layer"):
                        ee_values = conn.ee_synapses_per_branch_per_layer
                        if isinstance(ee_values, ListConfig):
                            ee_values = OmegaConf.to_container(ee_values, resolve=True)

                        params["ee_value"] = (
                            ee_values[0] if isinstance(ee_values, list) else ee_values
                        )

                    if hasattr(conn, "ie_synapses_per_branch_per_layer"):
                        ie_values = conn.ie_synapses_per_branch_per_layer
                        if isinstance(ie_values, ListConfig):
                            ie_values = OmegaConf.to_container(ie_values, resolve=True)

                        params["ie_value"] = (
                            ie_values[0] if isinstance(ie_values, list) else ie_values
                        )

                # Architecture parameters
                if hasattr(core, "architecture"):
                    arch = core.architecture
                    if hasattr(arch, "excitatory_layer_sizes"):
                        params["excitatory_layers"] = arch.excitatory_layer_sizes
                    if hasattr(arch, "inhibitory_layer_sizes"):
                        params["inhibitory_layers"] = arch.inhibitory_layer_sizes
                    if hasattr(arch, "excitatory_branch_factors"):
                        branch_factors = arch.excitatory_branch_factors
                        if (
                            isinstance(branch_factors, list)
                            and len(branch_factors) == 1
                        ):
                            params["branch_factors"] = branch_factors[0]
                        else:
                            params["branch_factors"] = "-".join(
                                str(x) for x in branch_factors
                            )

                # Morphology parameters
                if hasattr(core, "morphology"):
                    morph = core.morphology
                    if hasattr(morph, "use_shunting"):
                        params["use_shunting"] = morph.use_shunting

                # If core.type explicitly specifies the network variant, make sure the
                # extracted `use_shunting` matches the *actual* instantiated behavior.
                if "network_type" in params:
                    nt = str(params["network_type"]).lower()
                    if nt == "dendritic_shunting":
                        params["use_shunting"] = True
                    elif nt in [
                        "dendritic_additive",
                        "dendritic_mlp",
                        "mlp",
                        "point_mlp",
                    ]:
                        params["use_shunting"] = False

        except Exception as e:
            logger.warning(f"Error extracting model params: {e}")

        return params

    def _extract_training_params(self, config) -> dict:
        """Extract training-related parameters."""
        params = {}

        try:
            if hasattr(config, "training") and hasattr(config.training, "main"):
                main = config.training.main

                # Strategy
                if hasattr(main, "strategy"):
                    params["strategy"] = main.strategy

                # Common training params
                if hasattr(main, "common"):
                    common = main.common
                    if hasattr(common, "lr") or hasattr(common.param_groups, "lr"):
                        lr = getattr(common, "lr", None) or getattr(
                            common.param_groups, "lr", None
                        )
                        if lr:
                            params["learning_rate"] = lr
                    if hasattr(common, "batch_size"):
                        params["batch_size"] = common.batch_size

                # Local learning parameters (new structure)
                if (
                    hasattr(main, "learning_strategy_config")
                    and main.learning_strategy_config
                ):
                    local_cfg = main.learning_strategy_config
                    params.update(self._extract_local_learning_params(local_cfg))

                # Fallback to old structure
                elif hasattr(main, "strategies") and hasattr(
                    main.strategies, "local_ca"
                ):
                    local_cfg = main.strategies.local_ca
                    params.update(self._extract_local_learning_params(local_cfg))

        except Exception as e:
            logger.warning(f"Error extracting training params: {e}")

        return params

    def _extract_local_learning_params(self, local_cfg) -> dict:
        """Extract local learning rule parameters."""
        params = {}

        try:
            if hasattr(local_cfg, "rule_variant"):
                params["rule_variant"] = local_cfg.rule_variant
            if hasattr(local_cfg, "error_broadcast_mode"):
                params["error_broadcast_mode"] = local_cfg.error_broadcast_mode
            if hasattr(local_cfg, "update_inactive_weights"):
                params["update_inactive_weights"] = local_cfg.update_inactive_weights

            # Extract from nested four_factor config
            if hasattr(local_cfg, "four_factor"):
                four_f = local_cfg.four_factor
                if hasattr(four_f, "rho_mode"):
                    params["rho_mode"] = four_f.rho_mode
                if hasattr(four_f, "rho_estimator"):
                    params["rho_estimator"] = four_f.rho_estimator

            # Extract from nested five_factor config
            if hasattr(local_cfg, "five_factor"):
                five_f = local_cfg.five_factor
                if hasattr(five_f, "phi_mode"):
                    params["phi_mode"] = five_f.phi_mode
                if hasattr(five_f, "phi_estimator"):
                    params["phi_estimator"] = five_f.phi_estimator

            # Extract from nested morphology_aware config
            if hasattr(local_cfg, "morphology_aware"):
                morph = local_cfg.morphology_aware
                if hasattr(morph, "use_dendritic_normalization"):
                    params["use_dendritic_normalization"] = (
                        morph.use_dendritic_normalization
                    )
                if hasattr(morph, "use_path_propagation"):
                    params["use_path_propagation"] = morph.use_path_propagation
                if hasattr(morph, "morphology_modulator_mode"):
                    params["morphology_modulator_mode"] = (
                        morph.morphology_modulator_mode
                    )

            # Extract from nested HSIC config
            if hasattr(local_cfg, "hsic"):
                hsic = local_cfg.hsic
                if hasattr(hsic, "enabled"):
                    params["hsic_enabled"] = hsic.enabled
                if hasattr(hsic, "weight"):
                    params["hsic_weight"] = hsic.weight

        except Exception as e:
            logger.warning(f"Error extracting local learning params: {e}")

        return params

    def _extract_experiment_params(self, config) -> dict:
        """Extract experiment-related parameters."""
        params = {}

        try:
            if hasattr(config, "experiment"):
                exp = config.experiment
                if hasattr(exp, "seed"):
                    params["seed"] = exp.seed

            if hasattr(config, "data"):
                data = config.data
                if hasattr(data, "dataset_name"):
                    dataset_name = data.dataset_name
                    params["dataset"] = dataset_name

                    if hasattr(data, "dataset_params"):
                        dataset_params = data.dataset_params
                        if dataset_name in dataset_params:
                            for key, value in dataset_params[dataset_name].items():
                                params[f"{key}"] = value

        except Exception as e:
            logger.warning(f"Error extracting experiment params: {e}")

        return params
