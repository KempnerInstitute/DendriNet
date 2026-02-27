#!/usr/bin/env python3
"""
Model Architecture Validator for Dendritic Networks
==================================================

This module validates the structural/architectural parameters of dendritic models
BEFORE model creation. It ensures:
- Layer dimensions are compatible
- Synapse counts match layer configurations
- TopK constraints are satisfied
- Branch factors are consistent
- Transfer learning dimensions align
- Biological plausibility constraints

This is separate from training stability validation and runs during model setup.
"""

import logging
from typing import Any, Union

logger = logging.getLogger(__name__)


def _get_config_value(
    config: Union[dict, object], key: str, default: Any = None
) -> Any:
    """Safely get a value from a config object, handling both dict and attribute access."""
    if isinstance(config, dict):
        return config.get(key, default)
    else:
        return getattr(config, key, default)


class ModelArchitectureValidator:
    """Validates structural/architectural parameters for dendritic networks"""

    def __init__(self, config):
        self.config = config

        # Extract network configuration using standard structure
        if hasattr(config, "model") and hasattr(config.model, "core"):
            core_config = config.model.core
            # Combine all network parameters from different sections
            self.network_config = {}

            # Add architecture parameters
            if hasattr(core_config, "architecture"):
                arch_config = core_config.architecture
                if hasattr(arch_config, "__dict__"):
                    self.network_config.update(arch_config.__dict__)
                else:
                    self.network_config.update(arch_config)

            # Add connectivity parameters
            if hasattr(core_config, "connectivity"):
                conn_config = core_config.connectivity
                if hasattr(conn_config, "__dict__"):
                    self.network_config.update(conn_config.__dict__)
                else:
                    self.network_config.update(conn_config)

            # Add transfer parameters (source of input_mode/independent_pathways).
            if hasattr(core_config, "transfer"):
                transfer_config = core_config.transfer
                if hasattr(transfer_config, "__dict__"):
                    self.network_config["transfer_params"] = dict(
                        transfer_config.__dict__
                    )
                else:
                    self.network_config["transfer_params"] = dict(transfer_config)

            # Add morphology parameters
            if hasattr(core_config, "morphology"):
                morph_config = core_config.morphology
                if hasattr(morph_config, "__dict__"):
                    self.network_config.update(morph_config.__dict__)
                else:
                    self.network_config.update(morph_config)
        else:
            self.network_config = {}

        self.errors: list[str] = []
        self.warnings: list[str] = []

    def _get_config_value(self, key: str, default=None):
        """Safely get configuration value from Config object or dict."""
        return _get_config_value(self.config, key, default)

    def validate(self) -> bool:
        """Validate all constraints and return True if config is valid"""
        logger.info("Starting configuration validation")

        # Extract key parameters
        self.extract_parameters()

        # Run all checks
        self.check_input_mode_consistency()
        self.check_layer_dimensions()
        self.check_synapses_per_branch()
        self.check_topk_constraints()
        self.check_branch_factors()
        self.check_empty_lists()
        self.check_transfer_dimensions()
        self.check_biological_plausibility()

        # Report results
        self.report_results()

        return len(self.errors) == 0

    def extract_parameters(self):
        """Extract key parameters from config"""
        # Get transfer params first to determine input_dim
        self.transfer_params = self.network_config.get("transfer_params", {})
        self.input_mode = self.transfer_params.get("input_mode", 0)
        self.independent_pathways = self.transfer_params.get(
            "independent_pathways", False
        )

        # Get dataset name from data config
        data_config = self._get_config_value("data", {})
        dataset_name = _get_config_value(data_config, "dataset_name", "")

        task_config = self._get_config_value("task", {})
        task_params = _get_config_value(task_config, "parameters", {})

        if dataset_name == "cifar10":
            self.input_dim = 3072  # 32x32x3
        elif dataset_name == "imagenet":
            self.input_dim = 150528  # 224x224x3
        elif dataset_name == "mnist":
            self.input_dim = 784  # 28x28
        elif dataset_name == "multixor":
            # MultiXOR typically has small input dimensions (2-8 bits per input)
            # Default to a reasonable size for XOR-like problems
            n_bits = task_params.get("n_bits", 4)  # Default 4-bit XOR
            self.input_dim = n_bits * 2  # Two inputs of n_bits each
        else:
            # Try to get input_dim from various sources
            self.input_dim = (
                self.network_config.get("input_dim")
                or task_params.get("input_dim")
                or self.transfer_params.get("excitatory_dim")
                or 64  # Reasonable default for unknown datasets
            )

        self.excitatory_layer_sizes = self.network_config.get(
            "excitatory_layer_sizes", []
        )
        self.inhibitory_layer_sizes = self.network_config.get(
            "inhibitory_layer_sizes", []
        )

        self.excitatory_branch_factors = self.network_config.get(
            "excitatory_branch_factors", []
        )
        self.inhibitory_branch_factors = self.network_config.get(
            "inhibitory_branch_factors", []
        )

        self.ee_synapses = self.network_config.get(
            "ee_synapses_per_branch_per_layer", []
        )
        self.ei_synapses = self.network_config.get(
            "ei_synapses_per_branch_per_layer", []
        )
        self.ie_synapses = self.network_config.get(
            "ie_synapses_per_branch_per_layer", []
        )
        self.ii_synapses = self.network_config.get(
            "ii_synapses_per_branch_per_layer", []
        )

        self.num_layers = len(self.excitatory_layer_sizes)

        # Transfer dimensions
        self.excitatory_dim = self.transfer_params.get("excitatory_dim", None)
        self.inhibitory_dim = self.transfer_params.get("inhibitory_dim", None)

    def check_input_mode_consistency(self):
        """Check input_mode and independent_pathways consistency"""
        if self.input_mode == 0 and self.independent_pathways:
            self.errors.append("input_mode=0 requires independent_pathways=False")

    def check_layer_dimensions(self):
        """Check layer size consistency"""
        exc_layers = len(self.excitatory_layer_sizes)
        inh_layers = len(self.inhibitory_layer_sizes)

        if self.input_mode == 1:
            # For input_mode=1, inhibitory layers can be:
            # - Empty (no inhibitory layers)
            # - Equal to excitatory layers (allowing 0 values for some layers)
            # - One less than excitatory layers
            if inh_layers not in [0, exc_layers - 1, exc_layers]:
                self.errors.append(
                    f"For input_mode=1, inhibitory layer count ({inh_layers}) must be "
                    f"0, {exc_layers-1}, or {exc_layers} (excitatory count: {exc_layers})"
                )
        else:
            # For other input modes, they should match
            if exc_layers != inh_layers:
                self.errors.append(
                    f"Mismatch in layer counts - "
                    f"excitatory: {exc_layers}, "
                    f"inhibitory: {inh_layers}"
                )

        # Check for zero-sized layers
        for i, exc in enumerate(self.excitatory_layer_sizes):
            if exc <= 0:
                self.errors.append(f"Layer {i} has non-positive excitatory size: {exc}")

        # Only check inhibitory if they exist
        for i, inh in enumerate(self.inhibitory_layer_sizes):
            if inh < 0:
                self.errors.append(f"Layer {i} has negative inhibitory size: {inh}")

    def check_synapses_per_branch(self):
        """Check synapses per branch constraints"""
        # Determine expected lengths (empty lists will be extended with default=1)
        expected_len = self.num_layers

        # Check if there are any inhibitory neurons in the network
        any(size for size in self.inhibitory_layer_sizes if size and size > 0)

        # Check each synapse type
        for layer_idx in range(expected_len):
            # Get dimensions for this layer
            if layer_idx == 0:
                # First layer receives from transfer/encoder
                if self.independent_pathways:
                    exc_input = (
                        self.excitatory_dim
                        if self.excitatory_dim
                        else self.input_dim // 2
                    )
                    inh_input = (
                        self.inhibitory_dim
                        if self.inhibitory_dim
                        else self.input_dim // 2
                    )
                else:
                    exc_input = self.input_dim
                    inh_input = self.input_dim
            else:
                # Subsequent layers receive from previous layer
                exc_input = self.excitatory_layer_sizes[layer_idx - 1]
                # Only get inhibitory input if inhibitory layers exist
                if layer_idx - 1 < len(self.inhibitory_layer_sizes):
                    inh_input = self.inhibitory_layer_sizes[layer_idx - 1]
                else:
                    inh_input = 0

            # Inhibitory input to excitatory cells comes from same layer (if exists)
            if layer_idx < len(self.inhibitory_layer_sizes):
                inh_to_exc = self.inhibitory_layer_sizes[layer_idx]
            else:
                inh_to_exc = 0

            # Check EE synapses
            ee_val = (
                self.ee_synapses[layer_idx]
                if layer_idx < len(self.ee_synapses)
                else (self.ee_synapses[-1] if self.ee_synapses else 1)
            )
            if ee_val > exc_input:
                self.errors.append(
                    f"Layer {layer_idx} - ee_synapses ({ee_val}) > "
                    f"excitatory input dim ({exc_input})"
                )

            # Check EI synapses (only if ei_synapses list is not empty)
            if self.ei_synapses:
                ei_val = (
                    self.ei_synapses[layer_idx]
                    if layer_idx < len(self.ei_synapses)
                    else self.ei_synapses[-1]
                )
                if inh_to_exc > 0 and ei_val > inh_to_exc:
                    self.errors.append(
                        f"Layer {layer_idx} - ei_synapses ({ei_val}) > "
                        f"inhibitory neurons in layer ({inh_to_exc})"
                    )
                elif inh_to_exc == 0 and ei_val > 0:
                    self.errors.append(
                        f"Layer {layer_idx} has ei_synapses={ei_val} but no inhibitory neurons exist"
                    )

            # Check IE synapses (inhibitory TO excitatory connections) - only if list is not empty
            if self.ie_synapses:
                ie_val = (
                    self.ie_synapses[layer_idx]
                    if layer_idx < len(self.ie_synapses)
                    else self.ie_synapses[-1]
                )

                # Determine the source of inhibitory input for IE constraint checking
                if self.input_mode == 1:
                    # In input_mode=1, ALL layers can receive inhibitory input from transfer function
                    if self.independent_pathways:
                        inhibitory_source_dim = (
                            self.inhibitory_dim
                            if self.inhibitory_dim
                            else self.input_dim // 2
                        )
                    else:
                        # When independent_pathways=false, input feeds both E and I synapses
                        # So inhibitory input dimension equals excitatory input dimension
                        inhibitory_source_dim = (
                            self.excitatory_dim
                            if self.excitatory_dim
                            else self.input_dim
                        )
                elif layer_idx == 0:
                    # For input_mode=0: first layer excitatory neurons can receive inhibitory input
                    # from same layer's inhibitory neurons (if they exist)
                    if (
                        layer_idx < len(self.inhibitory_layer_sizes)
                        and self.inhibitory_layer_sizes[layer_idx] > 0
                    ):
                        inhibitory_source_dim = self.inhibitory_layer_sizes[layer_idx]
                    else:
                        inhibitory_source_dim = 0
                else:
                    # Subsequent layers get inhibitory input from previous layer's inhibitory neurons
                    if layer_idx - 1 < len(self.inhibitory_layer_sizes):
                        inhibitory_source_dim = self.inhibitory_layer_sizes[
                            layer_idx - 1
                        ]
                    else:
                        inhibitory_source_dim = 0

                # Check IE constraint
                if inhibitory_source_dim > 0 and ie_val > inhibitory_source_dim:
                    self.errors.append(
                        f"Layer {layer_idx} - ie_synapses ({ie_val}) > "
                        f"inhibitory input dim ({inhibitory_source_dim})"
                    )
                elif inhibitory_source_dim == 0 and ie_val > 0:
                    self.errors.append(
                        f"Layer {layer_idx} has ie_synapses={ie_val} but no inhibitory input available"
                    )

            # Check II synapses (only if inhibitory neurons exist)
            if (
                layer_idx < len(self.inhibitory_layer_sizes)
                and self.inhibitory_layer_sizes[layer_idx] > 0
            ):
                ii_val = (
                    self.ii_synapses[layer_idx]
                    if layer_idx < len(self.ii_synapses)
                    else (self.ii_synapses[-1] if self.ii_synapses else 1)
                )
                if ii_val > inh_input:
                    self.errors.append(
                        f"Layer {layer_idx} - ii_synapses ({ii_val}) > "
                        f"inhibitory input dim ({inh_input})"
                    )

    def check_topk_constraints(self):
        """Check TopK constraints"""
        # EE synapses must always be positive (excitatory neurons always exist)
        for i, val in enumerate(self.ee_synapses):
            if val <= 0:
                self.errors.append(f"ee_synapses[{i}] must be positive, got {val}")

        # Other synapse types can be 0 when target neurons don't exist
        for syn_type, syn_list in [
            ("ei", self.ei_synapses),
            ("ie", self.ie_synapses),
            ("ii", self.ii_synapses),
        ]:
            for i, val in enumerate(syn_list):
                if val < 0:  # Allow 0, but not negative
                    self.errors.append(
                        f"{syn_type}_synapses[{i}] must be non-negative, got {val}"
                    )
                elif val == 0:
                    # Check if 0 makes sense given the layer configuration
                    if syn_type == "ei" and i < len(self.inhibitory_layer_sizes):
                        # EI: excitatory TO inhibitory connections
                        if self.inhibitory_layer_sizes[i] > 0:
                            self.warnings.append(
                                f"ei_synapses[{i}] is 0 but layer {i} has "
                                f"{self.inhibitory_layer_sizes[i]} inhibitory neurons"
                            )
                    elif syn_type == "ie" and i < len(self.inhibitory_layer_sizes):
                        # IE: inhibitory TO excitatory connections
                        # In input_mode=1, first layer gets inhibitory input from transfer function
                        # Subsequent layers get inhibitory input from previous layer's inhibitory neurons
                        if i == 0:
                            # First layer can have IE synapses from transfer function
                            if self.input_mode == 1:
                                self.warnings.append(
                                    "ie_synapses[0] is 0 - layer 0 excitatory neurons won't receive "
                                    "inhibitory input from transfer function"
                                )
                        else:
                            # Other layers need previous layer to have inhibitory neurons
                            if (
                                i - 1 < len(self.inhibitory_layer_sizes)
                                and self.inhibitory_layer_sizes[i - 1] > 0
                            ):
                                self.warnings.append(
                                    f"ie_synapses[{i}] is 0 but previous layer {i-1} has "
                                    f"{self.inhibitory_layer_sizes[i-1]} inhibitory neurons"
                                )
                    elif syn_type == "ii" and i < len(self.inhibitory_layer_sizes):
                        # For II synapses, check if previous layer had inhibitory neurons
                        if i > 0 and i - 1 < len(self.inhibitory_layer_sizes):
                            if (
                                self.inhibitory_layer_sizes[i - 1] > 0
                                and self.inhibitory_layer_sizes[i] > 0
                            ):
                                self.warnings.append(
                                    f"ii_synapses[{i}] is 0 but both layer {i-1} and {i} have inhibitory neurons"
                                )

    def check_branch_factors(self):
        """Check branch factor constraints"""
        # Branch factors represent hierarchical branching structure
        # e.g., [2, 2] means 2 branches that each split into 2 more (total 4 branches)
        # They don't need to match the number of layers

        # Just check that they are valid positive integers
        for i, bf in enumerate(self.excitatory_branch_factors):
            if bf <= 0:
                self.errors.append(
                    f"excitatory_branch_factors[{i}] must be positive, got {bf}"
                )

        for i, bf in enumerate(self.inhibitory_branch_factors):
            if bf <= 0:
                self.errors.append(
                    f"inhibitory_branch_factors[{i}] must be positive, got {bf}"
                )

    def check_empty_lists(self):
        """Check for empty synapse lists and validate they're appropriate for the configuration"""
        has_inhibitory_neurons = any(
            size for size in self.inhibitory_layer_sizes if size and size > 0
        )

        # EE synapses are always required
        if not self.ee_synapses:
            self.errors.append(
                "ee_synapses_per_branch_per_layer cannot be empty - excitatory connections are required"
            )

        # EI synapses: only allow empty if no inhibitory neurons exist
        if not self.ei_synapses:
            if has_inhibitory_neurons:
                self.errors.append(
                    "ei_synapses_per_branch_per_layer cannot be empty when inhibitory neurons exist"
                )
            # If no inhibitory neurons, empty ei_synapses is valid (no E→I connections needed)

        # IE synapses: check based on input_mode and layer configuration
        if not self.ie_synapses:
            if self.input_mode == 1:
                # input_mode=1 provides inhibitory input, so IE connections are expected
                self.errors.append(
                    "ie_synapses_per_branch_per_layer cannot be empty for input_mode=1 (inhibitory input available)"
                )
            elif has_inhibitory_neurons:
                # Other modes with inhibitory neurons should have IE connections
                self.errors.append(
                    "ie_synapses_per_branch_per_layer cannot be empty when inhibitory neurons exist"
                )
            # If no inhibitory input available, empty ie_synapses is valid

        # II synapses: only allow empty if no inhibitory neurons exist
        if not self.ii_synapses:
            if has_inhibitory_neurons:
                self.errors.append(
                    "ii_synapses_per_branch_per_layer cannot be empty when inhibitory neurons exist"
                )
            # If no inhibitory neurons, empty ii_synapses is valid (no I→I connections needed)

    def check_transfer_dimensions(self):
        """Check transfer function dimension settings"""
        if self.independent_pathways:
            # With independent pathways, check if dimensions are auto-set
            if not self.excitatory_dim or not self.inhibitory_dim:
                self.warnings.append(
                    "Transfer dimensions not set, will auto-set to input_dim/2 each"
                )
        else:
            # With shared pathway, dimensions should match input
            if self.excitatory_dim and self.excitatory_dim != self.input_dim:
                self.warnings.append(
                    f"With shared pathway, excitatory_dim ({self.excitatory_dim}) "
                    f"should match input_dim ({self.input_dim})"
                )

    def check_biological_plausibility(self):
        """Check biological plausibility constraints"""
        # Warn if too many synapses per branch
        for syn_type, syn_list in [
            ("ee", self.ee_synapses),
            ("ei", self.ei_synapses),
            ("ie", self.ie_synapses),
            ("ii", self.ii_synapses),
        ]:
            for i, val in enumerate(syn_list):
                if val > 100:
                    self.warnings.append(
                        f"{syn_type}_synapses[{i}] = {val} is very high. "
                        "Biological neurons typically have 10-50 synapses per dendrite"
                    )

    def report_results(self):
        """Report all errors and warnings"""
        logger.info("Configuration validation summary:")
        logger.info(f"  Input mode: {self.input_mode}")
        logger.info(f"  Independent pathways: {self.independent_pathways}")
        logger.info(f"  Number of layers: {self.num_layers}")

        if self.errors:
            logger.error(f"Configuration validation found {len(self.errors)} error(s):")
            for error in self.errors:
                logger.error(f"  {error}")

        if self.warnings:
            logger.warning(
                f"Configuration validation found {len(self.warnings)} warning(s):"
            )
            for warning in self.warnings:
                logger.warning(f"  {warning}")

        if not self.errors and not self.warnings:
            logger.info("Configuration validation passed - all constraints satisfied")
        elif not self.errors:
            logger.info("Configuration validation passed with warnings")


def validate_model_architecture(config: dict) -> bool:
    """Validate model architecture configuration.

    Args:
        config: Configuration dictionary

    Returns:
        bool: True if architecture is valid
    """
    validator = ModelArchitectureValidator(config)
    return validator.validate()
