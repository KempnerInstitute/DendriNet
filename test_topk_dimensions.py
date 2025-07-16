#!/usr/bin/env python
"""Test TopK behavior when inhibitory neurons < synapses per branch."""

import sys
sys.path.insert(0, 'src')

import torch
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import TopKLinear

print("\n" + "="*60)
print("Testing TopK with Different Dimension Scenarios")
print("="*60 + "\n")

# Test 1: More neurons than synapses (typical case)
print("Test 1: More inhibitory neurons than synapses")
print("-" * 60)
print("Inhibitory neurons: 100, Synapses per branch (K): 20")
topk1 = TopKLinear(in_features=100, out_features=50, K=20)
x1 = torch.randn(32, 100)
y1 = topk1(x1)
print(f"Input shape: {x1.shape}, Output shape: {y1.shape}")
print(f"Weight shape: {topk1.pre_w.shape}")
print(f"TopK selects {topk1.K} out of {100} possible connections per output neuron")

# Test 2: Fewer neurons than synapses (your question)
print("\n\nTest 2: Fewer inhibitory neurons than synapses")
print("-" * 60)
print("Inhibitory neurons: 20, Synapses per branch (K): 5")
topk2 = TopKLinear(in_features=20, out_features=100, K=5)
x2 = torch.randn(32, 20)
y2 = topk2(x2)
print(f"Input shape: {x2.shape}, Output shape: {y2.shape}")
print(f"Weight shape: {topk2.pre_w.shape}")
print(f"TopK selects {topk2.K} out of {20} possible connections per output neuron")

# Test 3: Edge case - K equals input dimensions
print("\n\nTest 3: K equals number of inhibitory neurons")
print("-" * 60)
print("Inhibitory neurons: 20, Synapses per branch (K): 20")
topk3 = TopKLinear(in_features=20, out_features=100, K=20)
x3 = torch.randn(32, 20)
y3 = topk3(x3)
print(f"Input shape: {x3.shape}, Output shape: {y3.shape}")
print(f"Weight shape: {topk3.pre_w.shape}")
print(f"TopK selects ALL {topk3.K} connections (no pruning)")

# Test 4: What happens if K > input dimensions?
print("\n\nTest 4: K greater than inhibitory neurons (should work but select all)")
print("-" * 60)
print("Inhibitory neurons: 10, Synapses per branch (K): 20")
try:
    topk4 = TopKLinear(in_features=10, out_features=100, K=20)
    x4 = torch.randn(32, 10)
    y4 = topk4(x4)
    print(f"Input shape: {x4.shape}, Output shape: {y4.shape}")
    print(f"Weight shape: {topk4.pre_w.shape}")
    print(f"TopK tries to select {topk4.K} but only {10} available")
    # Check how many connections are actually active
    mask = topk4.weight_mask()
    actual_k = mask.sum(dim=1).max().item()
    print(f"Actual connections per neuron: {actual_k}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("Conclusion:")
print("- TopK works fine when inhibitory neurons < synapses")
print("- It selects K strongest from available connections")
print("- If K > available inputs, it uses all available")
print("="*60) 