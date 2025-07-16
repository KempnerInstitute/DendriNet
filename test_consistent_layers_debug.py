#!/usr/bin/env python
"""Debug version to trace dimension mismatches."""

import sys
sys.path.insert(0, 'src')

import torch
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork
)

# Enable detailed debugging
import logging
logging.basicConfig(level=logging.DEBUG)

print("\n" + "="*60)
print("Debugging Dimension Mismatch")
print("="*60 + "\n")

# Simple test case
params = {
    'input_dim': 784,  # MNIST input
    'excitatory_layer_sizes': [100, 50],
    'inhibitory_layer_sizes': [20, 10],
    'excitatory_branch_factors': [2, 2],
    'inhibitory_branch_factors': [2, 2],
    'ee_synapses_per_branch_per_layer': [10],
    'ei_synapses_per_branch_per_layer': [5],
    'ie_synapses_per_branch_per_layer': [5],
    'ii_synapses_per_branch_per_layer': [3],
    'transfer_params': {
        'input_mode': 0,
        'independent_pathways': False
    }
}

network = ExcitationInhibitionNetwork(**params)

# Print detailed structure
print("Network structure:")
print(f"Transfer excitatory dim: {network.transfer_fn.excitatory_dim}")
print(f"Transfer inhibitory dim: {network.transfer_fn.inhibitory_dim}")

for i, layer in enumerate(network.layers):
    print(f"\nLayer {i}:")
    if hasattr(layer, 'inhibitory_cells'):
        print(f"  Inhibitory cells:")
        for j, branch_layer in enumerate(layer.inhibitory_cells.branch_layers):
            print(f"    Branch layer {j}:")
            if branch_layer.input_excitatory and branch_layer.branch_excitation:
                print(f"      Excitation input: {branch_layer.excitatory_input_dim}")
                print(f"      Excitation shape: {branch_layer.branch_excitation.pre_w.shape}")
            if branch_layer.input_inhibitory and branch_layer.branch_inhibition:
                print(f"      Inhibition input: {branch_layer.inhibitory_input_dim}")
                print(f"      Inhibition shape: {branch_layer.branch_inhibition.pre_w.shape}")
    
    print(f"  Excitatory cells:")
    for j, branch_layer in enumerate(layer.excitatory_cells.branch_layers):
        print(f"    Branch layer {j}:")
        if branch_layer.input_excitatory and branch_layer.branch_excitation:
            print(f"      Excitation input: {branch_layer.excitatory_input_dim}")
            print(f"      Excitation shape: {branch_layer.branch_excitation.pre_w.shape}")
        if branch_layer.input_inhibitory and branch_layer.branch_inhibition:
            print(f"      Inhibition input: {branch_layer.inhibitory_input_dim}")
            print(f"      Inhibition shape: {branch_layer.branch_inhibition.pre_w.shape}")

# Test forward pass with hooks
print("\n\nTesting forward pass with shape tracking:")
x = torch.randn(32, 784)  # Batch of 32 MNIST images

def print_hook(name):
    def hook(module, input, output):
        if isinstance(input, tuple):
            in_shapes = [inp.shape if inp is not None else None for inp in input]
        else:
            in_shapes = input.shape
        
        if isinstance(output, tuple):
            out_shapes = [out.shape if out is not None else None for out in output]
        else:
            out_shapes = output.shape
            
        print(f"{name}: input={in_shapes}, output={out_shapes}")
    return hook

# Add hooks to track shapes
network.transfer_fn.register_forward_hook(print_hook("TransferLayer"))
for i, layer in enumerate(network.layers):
    layer.register_forward_hook(print_hook(f"EILayer_{i}"))
    if hasattr(layer, 'inhibitory_cells'):
        layer.inhibitory_cells.register_forward_hook(print_hook(f"Layer{i}_InhibitoryCells"))
    layer.excitatory_cells.register_forward_hook(print_hook(f"Layer{i}_ExcitatoryCells"))

try:
    out = network(x)
    print(f"\nFinal output shape: {out.shape}")
except Exception as e:
    print(f"\nError occurred: {e}")
    import traceback
    traceback.print_exc() 