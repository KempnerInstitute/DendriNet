#!/usr/bin/env python
"""
Detailed EI network parameter calculation for a specific config.
"""

import yaml
import argparse
from typing import Dict, List


def calculate_dendritic_branches(branch_factors: List[int]) -> int:
    """Calculate total number of branches from branch factors."""
    if not branch_factors:
        return 1
    total = 1
    for factor in branch_factors:
        total *= factor
    return total


def estimate_ei_model_detailed(config: Dict) -> Dict[str, float]:
    """Detailed parameter estimation for EI model."""
    model_config = config['model']['core_network']['parameters']

    # Get layer sizes
    e_layers = model_config['excitatory_layer_sizes']
    i_layers = model_config.get('inhibitory_layer_sizes', [])

    # Get branch factors
    e_branch_factors = model_config.get('excitatory_branch_factors', [])
    i_branch_factors = model_config.get('inhibitory_branch_factors', [])

    # Calculate total branches per neuron
    e_branches_per_neuron = calculate_dendritic_branches(e_branch_factors)
    i_branches_per_neuron = calculate_dendritic_branches(i_branch_factors) if i_branch_factors else 0

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
    input_dim = 224 * 224 * 3  # 150,528 for ImageNet

    print("\nDetailed Parameter Calculation:")
    print("=" * 60)

    total_params = 0

    # 1. Transfer function: Input to transfer dims
    transfer_input_params = input_dim * (e_transfer_dim + i_transfer_dim)
    total_params += transfer_input_params
    print("Transfer function (input -> transfer dims):")
    print(f"  {input_dim} × ({e_transfer_dim} + {i_transfer_dim}) = {transfer_input_params:,}")

    # 2. Transfer to first layer
    transfer_to_layer_params = 0
    if e_layers:
        transfer_to_e_params = e_transfer_dim * e_layers[0]
        transfer_to_layer_params += transfer_to_e_params
        print("Transfer -> E layer:")
        print(f"  {e_transfer_dim} × {e_layers[0]} = {transfer_to_e_params:,}")

    if i_layers and len(i_layers) > 0 and i_layers[0] > 0:
        transfer_to_i_params = i_transfer_dim * i_layers[0]
        transfer_to_layer_params += transfer_to_i_params
        print("Transfer -> I layer:")
        print(f"  {i_transfer_dim} × {i_layers[0]} = {transfer_to_i_params:,}")

    total_params += transfer_to_layer_params

    # 3. Dendritic connections
    # For single layer architecture, we only have I->E connections
    dendritic_params = 0

    # I-E connections (from input through inhibitory pathway to excitatory neurons)
    if e_layers and len(e_layers) > 0 and ie_synapses and len(ie_synapses) > 0:
        # Each excitatory neuron has branches, each branch can receive synapses
        ie_params = e_layers[0] * e_branches_per_neuron * ie_synapses[0]
        dendritic_params += ie_params
        print("I->E dendritic connections:")
        print(f"  {e_layers[0]} neurons × {e_branches_per_neuron} branches × {ie_synapses[0]} synapses")
        print(f"  = {ie_params:,}")

    total_params += dendritic_params

    # 4. Reactivation parameters (2 per neuron for ParametricTanh)
    total_neurons = sum(e_layers) + sum(i_layers)
    reactivation_params = total_neurons * 2
    total_params += reactivation_params
    print("Reactivation parameters:")
    print(f"  {total_neurons} neurons × 2 = {reactivation_params:,}")

    # 5. Decoder parameters
    decoder_config = config['model']['decoder_network']['parameters']
    hidden_dims = decoder_config['hidden_dims']
    output_dim = decoder_config['output_dim']

    decoder_params = 0
    decoder_input = e_layers[-1] if e_layers else 0
    prev_dim = decoder_input

    print("Decoder MLP:")
    for i, hidden_dim in enumerate(hidden_dims):
        layer_params = prev_dim * hidden_dim + hidden_dim  # weights + bias
        decoder_params += layer_params
        print(f"  Layer {i+1}: {prev_dim} × {hidden_dim} + {hidden_dim} = {layer_params:,}")
        prev_dim = hidden_dim

    # Output layer
    output_params = prev_dim * output_dim + output_dim
    decoder_params += output_params
    print(f"  Output: {prev_dim} × {output_dim} + {output_dim} = {output_params:,}")

    total_params += decoder_params

    print("\nTotal Parameters Breakdown:")
    print(f"  Transfer function: {transfer_input_params + transfer_to_layer_params:,}")
    print(f"  Dendritic connections: {dendritic_params:,}")
    print(f"  Reactivation: {reactivation_params:,}")
    print(f"  Decoder: {decoder_params:,}")
    print(f"  TOTAL: {total_params:,}")

    # Memory calculations
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
    args = parser.parse_args()

    # Load config
    with open(args.config_file) as f:
        config = yaml.safe_load(f)

    # Estimate model size
    results = estimate_ei_model_detailed(config)

    print("\n" + "=" * 60)
    print(f"Final Summary for: {args.config_file}")
    print("=" * 60)

    model_params = config['model']['core_network']['parameters']
    print("Architecture:")
    print(f"  Excitatory layers: {model_params['excitatory_layer_sizes']}")
    print(f"  Excitatory branches per neuron: {calculate_dendritic_branches(model_params.get('excitatory_branch_factors', []))}")
    print(f"  Synapses per branch: {model_params.get('ie_synapses_per_branch_per_layer', [])}")
    print(f"  Transfer dims: E={model_params['transfer_params']['excitatory_dim']}, I={model_params['transfer_params']['inhibitory_dim']}")

    print("\nModel Size:")
    print(f"  Total parameters: {results['total_parameters']:,}")
    print(f"  Parameter memory: {results['param_memory_gb']:.2f} GB")
    print(f"  Total memory (with optimizer): {results['total_memory_gb']:.2f} GB")

    print("\nWith FSDP on 16 GPUs (4 nodes):")
    print(f"  Memory per GPU: {results['memory_per_gpu_16']:.2f} GB")
    print(f"  GPU utilization: {results['gpu_utilization_pct']:.1f}%")

    # Check if it fits
    if results['memory_per_gpu_16'] <= 70:
        print("\n✓ This model WILL FIT on 4 nodes with comfortable headroom")
    else:
        print("\n✗ This model will NOT FIT - requires memory reduction")


if __name__ == '__main__':
    main() 