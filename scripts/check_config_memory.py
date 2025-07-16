#!/usr/bin/env python
"""
Check if a specific EI network configuration can fit on 4 nodes.
"""

import yaml
import argparse
import math
from typing import Dict, List


def calculate_dendritic_branches(branch_factors: List[int]) -> int:
    """Calculate total number of branches from branch factors."""
    if not branch_factors:
        return 1
    total = 1
    for factor in branch_factors:
        total *= factor
    return total


def estimate_ei_model_parameters(config: Dict) -> Dict[str, float]:
    """Estimate parameters for an EI model from config."""
    model_config = config['model']['core_network']['parameters']
    
    # Get layer sizes
    e_layers = model_config['excitatory_layer_sizes']
    i_layers = model_config.get('inhibitory_layer_sizes', [])
    
    # Get branch factors
    e_branch_factors = model_config.get('excitatory_branch_factors', [])
    i_branch_factors = model_config.get('inhibitory_branch_factors', [])
    
    # Calculate total branches
    e_branches = calculate_dendritic_branches(e_branch_factors)
    i_branches = calculate_dendritic_branches(i_branch_factors) if i_branch_factors else 0
    
    # Get synapses per branch
    ee_synapses = model_config.get('ee_synapses_per_branch_per_layer', [])
    ei_synapses = model_config.get('ei_synapses_per_branch_per_layer', [])
    ie_synapses = model_config.get('ie_synapses_per_branch_per_layer', [])
    ii_synapses = model_config.get('ii_synapses_per_branch_per_layer', [])
    
    # Get transfer function dims
    transfer_params = model_config.get('transfer_params', {})
    e_transfer_dim = transfer_params.get('excitatory_dim', 256)
    i_transfer_dim = transfer_params.get('inhibitory_dim', 256)
    
    # Get input dimension (ImageNet flattened)
    input_dim = 224 * 224 * 3  # ImageNet
    
    total_params = 0
    
    # Transfer function parameters
    # Input to transfer function
    transfer_input_params = input_dim * (e_transfer_dim + i_transfer_dim)
    total_params += transfer_input_params
    
    # Transfer to first layer
    if e_layers:
        transfer_to_e_params = e_transfer_dim * e_layers[0]
        total_params += transfer_to_e_params
    
    if i_layers:
        transfer_to_i_params = i_transfer_dim * i_layers[0]
        total_params += transfer_to_i_params
    
    # E-E connections
    for i, e_out in enumerate(e_layers):
        if i > 0 and i-1 < len(ee_synapses) and ee_synapses[i-1] > 0:
            e_in = e_layers[i-1]
            # Each output neuron has branches, each branch has synapses
            ee_params = e_out * e_branches * ee_synapses[i-1]
            total_params += ee_params
    
    # I-E connections
    for i, e_out in enumerate(e_layers):
        if i < len(ie_synapses) and ie_synapses[i] > 0 and i_layers:
            i_in = i_layers[0] if i_layers else 0
            ie_params = e_out * e_branches * ie_synapses[i]
            total_params += ie_params
    
    # E-I and I-I connections would be similar...
    
    # Decoder parameters
    decoder_config = config['model']['decoder_network']['parameters']
    hidden_dims = decoder_config['hidden_dims']
    output_dim = decoder_config['output_dim']
    
    # Decoder MLP
    decoder_input = e_layers[-1] if e_layers else 0
    prev_dim = decoder_input
    for hidden_dim in hidden_dims:
        total_params += prev_dim * hidden_dim + hidden_dim  # weights + bias
        prev_dim = hidden_dim
    total_params += prev_dim * output_dim + output_dim  # final layer
    
    # Calculate memory requirements
    dtype_bytes = 2  # FP16
    param_memory_gb = (total_params * dtype_bytes) / (1024**3)
    
    # With optimizer, gradients, and activations
    total_memory_gb = param_memory_gb * 4  # Conservative estimate
    
    # With FSDP on 16 GPUs (4 nodes)
    memory_per_gpu = total_memory_gb / 16
    
    return {
        'total_parameters': total_params,
        'param_memory_gb': param_memory_gb,
        'total_memory_gb': total_memory_gb,
        'memory_per_gpu_16': memory_per_gpu,
        'gpu_utilization_pct': (memory_per_gpu / 80) * 100  # H100 has 80GB
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path to YAML config file')
    parser.add_argument('--nodes', type=int, default=4, help='Number of nodes')
    parser.add_argument('--gpus-per-node', type=int, default=4, help='GPUs per node')
    args = parser.parse_args()
    
    # Load config
    with open(args.config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Estimate model size
    results = estimate_ei_model_parameters(config)
    
    total_gpus = args.nodes * args.gpus_per_node
    
    print(f"\nModel Configuration Analysis")
    print("=" * 60)
    print(f"Config file: {args.config_file}")
    print(f"Target hardware: {args.nodes} nodes × {args.gpus_per_node} GPUs = {total_gpus} H100 GPUs")
    print()
    
    print("Model Architecture:")
    model_params = config['model']['core_network']['parameters']
    print(f"  Excitatory layers: {model_params['excitatory_layer_sizes']}")
    print(f"  Inhibitory layers: {model_params.get('inhibitory_layer_sizes', [])}")
    print(f"  Excitatory branches: {calculate_dendritic_branches(model_params.get('excitatory_branch_factors', []))}")
    print(f"  Synapses per branch: {model_params.get('ee_synapses_per_branch_per_layer', [])}")
    print()
    
    print("Memory Requirements:")
    print(f"  Total parameters: {results['total_parameters']:,}")
    print(f"  Parameter memory: {results['param_memory_gb']:.2f} GB")
    print(f"  Total memory (with optimizer): {results['total_memory_gb']:.2f} GB")
    print()
    
    print(f"With FSDP on {total_gpus} GPUs:")
    print(f"  Memory per GPU: {results['memory_per_gpu_16']:.2f} GB")
    print(f"  GPU utilization: {results['gpu_utilization_pct']:.1f}%")
    print()
    
    # Check if it fits
    if results['memory_per_gpu_16'] <= 70:  # Leave some headroom
        print("✓ This model WILL FIT on 4 nodes with comfortable headroom")
    elif results['memory_per_gpu_16'] <= 80:
        print("⚠ This model will fit but with tight memory constraints")
    else:
        print("✗ This model will NOT FIT - requires memory reduction")
        print(f"  Excess memory per GPU: {results['memory_per_gpu_16'] - 80:.2f} GB")


if __name__ == '__main__':
    main() 