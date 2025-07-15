#!/usr/bin/env python3
"""
Estimate memory requirements for the 4-node EI model configuration.
"""

import argparse
from pathlib import Path
import yaml
import numpy as np


def estimate_ei_model_params(config):
    """Estimate parameter count for EI network."""
    params = config['model']['core_network']['parameters']
    
    # Get layer sizes
    exc_sizes = params['excitatory_layer_sizes']
    inh_sizes = params['inhibitory_layer_sizes']
    
    # Get branch factors
    exc_branch_factors = params['excitatory_branch_factors']
    inh_branch_factors = params['inhibitory_branch_factors']
    
    # Get synapse counts
    ee_synapses = params['ee_synapses_per_branch_per_layer']
    ei_synapses = params['ei_synapses_per_branch_per_layer']
    ie_synapses = params['ie_synapses_per_branch_per_layer']
    ii_synapses = params['ii_synapses_per_branch_per_layer']
    
    # Transfer function parameters
    transfer_params = params['transfer_params']
    input_dim = 150528  # ImageNet: 224*224*3
    exc_transfer_dim = transfer_params['excitatory_dim']
    inh_transfer_dim = transfer_params['inhibitory_dim']
    
    total_params = 0
    
    # Transfer function (input to exc/inh pathways)
    total_params += input_dim * (exc_transfer_dim + inh_transfer_dim)
    
    # Calculate parameters for each layer
    for layer_idx in range(len(exc_sizes)):
        # Current layer sizes
        n_exc = exc_sizes[layer_idx]
        n_inh = inh_sizes[layer_idx] if layer_idx < len(inh_sizes) else 0
        
        # Calculate number of branches
        exc_branches = np.prod(exc_branch_factors[:layer_idx+1]) if layer_idx < len(exc_branch_factors) else 1
        inh_branches = np.prod(inh_branch_factors[:layer_idx+1]) if layer_idx < len(inh_branch_factors) and n_inh > 0 else 1
        
        # Input dimensions
        if layer_idx == 0:
            exc_input_dim = exc_transfer_dim
            inh_input_dim = inh_transfer_dim
        else:
            exc_input_dim = exc_sizes[layer_idx - 1]
            inh_input_dim = inh_sizes[layer_idx - 1] if layer_idx - 1 < len(inh_sizes) else 0
        
        # E→E connections
        if layer_idx < len(ee_synapses):
            ee_params = n_exc * exc_branches * ee_synapses[layer_idx] * exc_input_dim
            total_params += ee_params
            print(f"Layer {layer_idx} E→E: {n_exc} neurons × {exc_branches} branches × {ee_synapses[layer_idx]} synapses × {exc_input_dim} = {ee_params:,}")
        
        # E→I connections
        if n_inh > 0 and layer_idx < len(ei_synapses):
            ei_params = n_inh * inh_branches * ei_synapses[layer_idx] * exc_input_dim
            total_params += ei_params
            print(f"Layer {layer_idx} E→I: {n_inh} neurons × {inh_branches} branches × {ei_synapses[layer_idx]} synapses × {exc_input_dim} = {ei_params:,}")
        
        # I→E connections
        if inh_input_dim > 0 and layer_idx < len(ie_synapses):
            ie_params = n_exc * exc_branches * ie_synapses[layer_idx] * inh_input_dim
            total_params += ie_params
            print(f"Layer {layer_idx} I→E: {n_exc} neurons × {exc_branches} branches × {ie_synapses[layer_idx]} synapses × {inh_input_dim} = {ie_params:,}")
        
        # I→I connections
        if n_inh > 0 and inh_input_dim > 0 and layer_idx < len(ii_synapses):
            ii_params = n_inh * inh_branches * ii_synapses[layer_idx] * inh_input_dim
            total_params += ii_params
            print(f"Layer {layer_idx} I→I: {n_inh} neurons × {inh_branches} branches × {ii_synapses[layer_idx]} synapses × {inh_input_dim} = {ii_params:,}")
        
        # Branch to output connections
        branch_to_output_params = n_exc * exc_branches
        if n_inh > 0:
            branch_to_output_params += n_inh * inh_branches
        total_params += branch_to_output_params
        
        # Reactivation parameters (if enabled)
        if params.get('reactivate', True):
            total_params += 2 * (n_exc + n_inh)  # m and b parameters
    
    # Decoder network
    decoder_params = config['model']['decoder_network']['parameters']
    decoder_input = exc_sizes[-1]  # Last excitatory layer
    decoder_hidden = decoder_params['hidden_dims']
    decoder_output = decoder_params['output_dim']
    
    # Decoder layers
    prev_dim = decoder_input
    for hidden_dim in decoder_hidden:
        total_params += prev_dim * hidden_dim + hidden_dim  # weights + bias
        prev_dim = hidden_dim
    total_params += prev_dim * decoder_output + decoder_output  # final layer
    
    return total_params


def estimate_memory_requirements(n_params, batch_size=256, precision='fp16'):
    """Estimate memory requirements for training."""
    bytes_per_param = 2 if precision == 'fp16' else 4
    
    # Model parameters
    param_memory = n_params * bytes_per_param
    
    # Gradients (same size as parameters)
    gradient_memory = param_memory
    
    # Optimizer states (Adam uses 2x parameters for momentum and variance)
    optimizer_memory = 2 * param_memory
    
    # Activations (rough estimate - depends on architecture)
    # For deep networks with many branches, this can be substantial
    activation_memory = param_memory * 0.5  # Conservative estimate
    
    # Total per GPU (with FSDP, model is sharded)
    total_memory = param_memory + gradient_memory + optimizer_memory + activation_memory
    
    return {
        'parameters': param_memory / 1e9,  # GB
        'gradients': gradient_memory / 1e9,
        'optimizer': optimizer_memory / 1e9,
        'activations': activation_memory / 1e9,
        'total': total_memory / 1e9
    }


def main():
    parser = argparse.ArgumentParser(description='Estimate model size for 4-node configuration')
    parser.add_argument('--config', type=str, default='configs/ei_fsdp_4nodes.yaml',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\nEstimating model size for: {args.config}")
    print("=" * 80)
    
    # Estimate parameters
    n_params = estimate_ei_model_params(config)
    print(f"\nTotal parameters: {n_params:,}")
    print(f"Billions of parameters: {n_params/1e9:.2f}B")
    
    # Estimate memory
    print("\nMemory Requirements (FP16):")
    print("-" * 40)
    mem = estimate_memory_requirements(n_params, precision='fp16')
    for key, value in mem.items():
        print(f"{key.capitalize():.<20} {value:.2f} GB")
    
    # FSDP sharding across GPUs
    n_gpus = 16  # 4 nodes × 4 GPUs
    print(f"\nWith FSDP across {n_gpus} GPUs:")
    print(f"Memory per GPU: {mem['total']/n_gpus:.2f} GB")
    print(f"H100 capacity: 80 GB")
    print(f"Memory utilization: {(mem['total']/n_gpus/80)*100:.1f}%")
    
    # Check if it fits
    if mem['total']/n_gpus > 70:  # Leave some headroom
        print("\n⚠️  WARNING: Model may be too large for available GPU memory!")
        print("Consider reducing layer sizes or branch factors.")
    else:
        print("\n✓ Model should fit comfortably in available GPU memory.")


if __name__ == '__main__':
    main() 