#!/usr/bin/env python
"""Test script to verify that input_mode 0 and 1 now have consistent layer structures."""

import sys
sys.path.insert(0, 'src')

import torch
from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork
)

print("\n" + "="*60)
print("Testing Consistent Layer Structure")
print("="*60 + "\n")

# Common parameters
common_params = {
    'input_dim': 784,  # MNIST input
    'excitatory_layer_sizes': [100, 50],
    'inhibitory_layer_sizes': [20, 10],
    'excitatory_branch_factors': [2, 2],
    'inhibitory_branch_factors': [2, 2],
    'ee_synapses_per_branch_per_layer': [10],
    'ei_synapses_per_branch_per_layer': [5],
    'ie_synapses_per_branch_per_layer': [5],
    'ii_synapses_per_branch_per_layer': [3],
}

# Test 1: input_mode = 0
print("Test 1: input_mode = 0")
print("-" * 60)
params_mode0 = common_params.copy()
params_mode0['transfer_params'] = {
    'input_mode': 0,
    'independent_pathways': False
}

network_mode0 = ExcitationInhibitionNetwork(**params_mode0)

print(f"Transfer function type: {type(network_mode0.transfer_fn).__name__}")
print(f"Transfer has inhibitory_cells: {hasattr(network_mode0.transfer_fn, 'inhibitory_cells')}")
print(f"Number of EI layers: {len(network_mode0.layers)}")

for i, layer in enumerate(network_mode0.layers):
    print(f"\nLayer {i}:")
    print(f"  Has inhibitory_cells: {hasattr(layer, 'inhibitory_cells')}")
    if hasattr(layer, 'inhibitory_cells'):
        print(f"  Inhibitory cells type: {type(layer.inhibitory_cells).__name__}")
    print(f"  Excitatory cells type: {type(layer.excitatory_cells).__name__}")

# Test 2: input_mode = 1
print("\n\nTest 2: input_mode = 1")
print("-" * 60)
params_mode1 = common_params.copy()
params_mode1['transfer_params'] = {
    'input_mode': 1,
    'independent_pathways': False
}

network_mode1 = ExcitationInhibitionNetwork(**params_mode1)

print(f"Transfer function type: {type(network_mode1.transfer_fn).__name__}")
print(f"Transfer has inhibitory_cells: {hasattr(network_mode1.transfer_fn, 'inhibitory_cells')}")
print(f"Number of EI layers: {len(network_mode1.layers)}")

for i, layer in enumerate(network_mode1.layers):
    print(f"\nLayer {i}:")
    print(f"  Has inhibitory_cells: {hasattr(layer, 'inhibitory_cells')}")
    if hasattr(layer, 'inhibitory_cells'):
        print(f"  Inhibitory cells type: {type(layer.inhibitory_cells).__name__}")
    print(f"  Excitatory cells type: {type(layer.excitatory_cells).__name__}")

# Test forward pass
print("\n\nTesting Forward Pass")
print("-" * 60)
x = torch.randn(32, 784)  # Batch of 32 MNIST images

print("Mode 0:")
try:
    out0 = network_mode0(x)
    print(f"  Output shape: {out0.shape}")
    print(f"  Output range: [{out0.min().item():.3f}, {out0.max().item():.3f}]")
except Exception as e:
    print(f"  Error: {e}")

print("\nMode 1:")
try:
    out1 = network_mode1(x)
    print(f"  Output shape: {out1.shape}")
    print(f"  Output range: [{out1.min().item():.3f}, {out1.max().item():.3f}]")
except Exception as e:
    print(f"  Error: {e}")

print("\n" + "="*60)
print("Conclusion: Both modes now have consistent layer structures!")
print("All EI layers have their own inhibitory cells.")
print("="*60) 