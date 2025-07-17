#!/usr/bin/env python3
"""Configuration validation for dendritic networks"""

import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configuration parameters for dendritic networks"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Try different paths for network config
        if 'model' in self.config and 'core_network' in self.config['model']:
            self.network_config = self.config['model']['core_network'].get('parameters', {})
        elif 'network' in self.config:
            self.network_config = self.config.get('network', {})
        else:
            self.network_config = {}
            
        self.errors = []
        self.warnings = []
        
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
        self.transfer_params = self.network_config.get('transfer_params', {})
        self.input_mode = self.transfer_params.get('input_mode', 0)
        self.independent_pathways = self.transfer_params.get('independent_pathways', False)
        
        # Get input dimension from data config
        data_config = self.config.get('data', {})
        dataset_name = data_config.get('dataset_name', '')
        if dataset_name == 'cifar10':
            self.input_dim = 3072  # 32x32x3
        elif dataset_name == 'imagenet':
            self.input_dim = 150528  # 224x224x3
        else:
            self.input_dim = self.network_config.get('input_dim', 0)
        
        self.excitatory_layer_sizes = self.network_config.get('excitatory_layer_sizes', [])
        self.inhibitory_layer_sizes = self.network_config.get('inhibitory_layer_sizes', [])
        
        self.excitatory_branch_factors = self.network_config.get('excitatory_branch_factors', [])
        self.inhibitory_branch_factors = self.network_config.get('inhibitory_branch_factors', [])
        
        self.ee_synapses = self.network_config.get('ee_synapses_per_branch_per_layer', [])
        self.ei_synapses = self.network_config.get('ei_synapses_per_branch_per_layer', [])
        self.ie_synapses = self.network_config.get('ie_synapses_per_branch_per_layer', [])
        self.ii_synapses = self.network_config.get('ii_synapses_per_branch_per_layer', [])
        
        self.num_layers = len(self.excitatory_layer_sizes)
        
        # Transfer dimensions
        self.excitatory_dim = self.transfer_params.get('excitatory_dim', None)
        self.inhibitory_dim = self.transfer_params.get('inhibitory_dim', None)
        
    def check_input_mode_consistency(self):
        """Check input_mode and independent_pathways consistency"""
        if self.input_mode == 0 and self.independent_pathways:
            self.errors.append("input_mode=0 requires independent_pathways=False")
        elif self.input_mode == 1 and not self.independent_pathways:
            self.errors.append("input_mode=1 requires independent_pathways=True")
            
    def check_layer_dimensions(self):
        """Check layer size consistency"""
        if len(self.excitatory_layer_sizes) != len(self.inhibitory_layer_sizes):
            self.errors.append(
                f"Mismatch in layer counts - "
                f"excitatory: {len(self.excitatory_layer_sizes)}, "
                f"inhibitory: {len(self.inhibitory_layer_sizes)}"
            )
            
        # Check for zero-sized layers
        for i, (exc, inh) in enumerate(zip(self.excitatory_layer_sizes, self.inhibitory_layer_sizes)):
            if exc <= 0:
                self.errors.append(f"Layer {i} has non-positive excitatory size: {exc}")
            if inh < 0:
                self.errors.append(f"Layer {i} has negative inhibitory size: {inh}")
                
    def check_synapses_per_branch(self):
        """Check synapses per branch constraints"""
        # Determine expected lengths (empty lists will be extended with default=1)
        expected_len = self.num_layers
        
        # Check each synapse type
        for layer_idx in range(expected_len):
            # Get dimensions for this layer
            if layer_idx == 0:
                # First layer receives from transfer/encoder
                if self.independent_pathways:
                    exc_input = self.excitatory_dim if self.excitatory_dim else self.input_dim // 2
                    inh_input = self.inhibitory_dim if self.inhibitory_dim else self.input_dim // 2
                else:
                    exc_input = self.input_dim
                    inh_input = self.input_dim
            else:
                # Subsequent layers receive from previous layer
                exc_input = self.excitatory_layer_sizes[layer_idx - 1]
                inh_input = self.inhibitory_layer_sizes[layer_idx - 1]
                
            # Inhibitory input to excitatory cells comes from same layer
            inh_to_exc = self.inhibitory_layer_sizes[layer_idx]
            
            # Check EE synapses
            ee_val = self.ee_synapses[layer_idx] if layer_idx < len(self.ee_synapses) else (self.ee_synapses[-1] if self.ee_synapses else 1)
            if ee_val > exc_input:
                self.errors.append(
                    f"Layer {layer_idx} - ee_synapses ({ee_val}) > "
                    f"excitatory input dim ({exc_input})"
                )
                
            # Check EI synapses
            ei_val = self.ei_synapses[layer_idx] if layer_idx < len(self.ei_synapses) else (self.ei_synapses[-1] if self.ei_synapses else 1)
            if inh_to_exc > 0 and ei_val > inh_to_exc:
                self.errors.append(
                    f"Layer {layer_idx} - ei_synapses ({ei_val}) > "
                    f"inhibitory neurons in layer ({inh_to_exc})"
                )
            elif inh_to_exc == 0 and ei_val > 0:
                self.warnings.append(
                    f"Layer {layer_idx} has ei_synapses={ei_val} but no inhibitory neurons"
                )
                
            # Check IE synapses (only if inhibitory neurons exist)
            if self.inhibitory_layer_sizes[layer_idx] > 0:
                ie_val = self.ie_synapses[layer_idx] if layer_idx < len(self.ie_synapses) else (self.ie_synapses[-1] if self.ie_synapses else 1)
                if ie_val > exc_input:
                    self.errors.append(
                        f"Layer {layer_idx} - ie_synapses ({ie_val}) > "
                        f"excitatory input dim ({exc_input})"
                    )
                    
            # Check II synapses
            if self.inhibitory_layer_sizes[layer_idx] > 0:
                ii_val = self.ii_synapses[layer_idx] if layer_idx < len(self.ii_synapses) else (self.ii_synapses[-1] if self.ii_synapses else 1)
                if ii_val > inh_input:
                    self.errors.append(
                        f"Layer {layer_idx} - ii_synapses ({ii_val}) > "
                        f"inhibitory input dim ({inh_input})"
                    )
                    
    def check_topk_constraints(self):
        """Check TopK constraints"""
        # TopK values (synapses per branch) must be positive
        for syn_type, syn_list in [
            ('ee', self.ee_synapses), 
            ('ei', self.ei_synapses), 
            ('ie', self.ie_synapses), 
            ('ii', self.ii_synapses)
        ]:
            for i, val in enumerate(syn_list):
                if val <= 0:
                    self.errors.append(f"{syn_type}_synapses[{i}] must be positive, got {val}")
                    
    def check_branch_factors(self):
        """Check branch factor constraints"""
        # Branch factors represent hierarchical branching structure
        # e.g., [2, 2] means 2 branches that each split into 2 more (total 4 branches)
        # They don't need to match the number of layers
        
        # Just check that they are valid positive integers
        for i, bf in enumerate(self.excitatory_branch_factors):
            if bf <= 0:
                self.errors.append(f"excitatory_branch_factors[{i}] must be positive, got {bf}")
                
        for i, bf in enumerate(self.inhibitory_branch_factors):
            if bf <= 0:
                self.errors.append(f"inhibitory_branch_factors[{i}] must be positive, got {bf}")
            
    def check_empty_lists(self):
        """Check and warn about empty lists that will use defaults"""
        if not self.ee_synapses:
            self.warnings.append("ee_synapses_per_branch_per_layer is empty, will default to 1")
        if not self.ei_synapses:
            self.warnings.append("ei_synapses_per_branch_per_layer is empty, will default to 1")
        if not self.ie_synapses:
            self.warnings.append("ie_synapses_per_branch_per_layer is empty, will default to 1")
        if not self.ii_synapses:
            self.warnings.append("ii_synapses_per_branch_per_layer is empty, will default to 1")
            
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
        for syn_type, syn_list in [('ee', self.ee_synapses), ('ei', self.ei_synapses), 
                                   ('ie', self.ie_synapses), ('ii', self.ii_synapses)]:
            for i, val in enumerate(syn_list):
                if val > 100:
                    self.warnings.append(
                        f"{syn_type}_synapses[{i}] = {val} is very high. "
                        "Biological neurons typically have 10-50 synapses per dendrite"
                    )
                    
        # Check E/I balance
        for i, (exc, inh) in enumerate(zip(self.excitatory_layer_sizes, self.inhibitory_layer_sizes)):
            if inh > 0:
                ratio = exc / inh
                if ratio < 2 or ratio > 10:
                    self.warnings.append(
                        f"Layer {i} E/I ratio = {ratio:.2f} ({exc}E/{inh}I). "
                        "Biological cortical networks typically have 4:1 to 5:1 (80-83% excitatory)"
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
            logger.warning(f"Configuration validation found {len(self.warnings)} warning(s):")
            for warning in self.warnings:
                logger.warning(f"  {warning}")
                
        if not self.errors and not self.warnings:
            logger.info("Configuration validation passed - all constraints satisfied")
        elif not self.errors:
            logger.info("Configuration validation passed with warnings")


def validate_config(config: Dict) -> bool:
    """Validate a configuration dictionary
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if config is valid, False otherwise
    """
    validator = ConfigValidator(config)
    return validator.validate() 