#!/usr/bin/env python
"""
A simple test script to validate that gradient hooks are being called correctly.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from src.dendritic_modeling.dendrinet import BlockLinear, DendriNet, TopKLinear
from src.dendritic_modeling.gradient_scaling import GradientScaler


def test_hook_call():
    print("Testing hook functionality...")

    # Create a simple model with hooks
    topk_layer = TopKLinear(in_features=10, out_features=4, K=3)
    # Fix: Make sure in_features is divisible by out_features
    block_layer = BlockLinear(in_features=4, out_features=2)

    # Initialize the layers
    topk_layer.initialize()
    block_layer.initialize()

    # Register hooks
    gradient_scaler = GradientScaler(
        topk_strategy="conductance_dynamic",
        blocklinear_strategy="conductance_dynamic",
        scale_factor=1.0,
        layer_idx=0,
    )

    print("Registering hooks...")
    gradient_scaler.register_topk_dynamic(topk_layer)
    gradient_scaler.register_block_linear_dynamic(block_layer)

    # Set layers to training mode
    topk_layer.train()
    block_layer.train()

    # Create a simple input
    x = torch.randn(3, 10, requires_grad=True)

    # Forward pass
    print("\nPerforming forward pass...")
    y1 = topk_layer(x)
    print(f"y1 shape: {y1.shape}, requires_grad: {y1.requires_grad}")
    # Ensure y1 retains gradient
    y1.retain_grad()

    # Dimensions now match (no need for adapter)
    y2 = block_layer(y1)
    print(f"y2 shape: {y2.shape}, requires_grad: {y2.requires_grad}")
    y2.retain_grad()

    # Loss and backward pass
    loss = y2.sum()
    print("\nPerforming backward pass...")
    loss.backward(retain_graph=True)

    # Print gradient shapes
    print(f"\nx.grad shape: {x.grad.shape if x.grad is not None else None}")
    print(f"y1.grad shape: {y1.grad.shape if y1.grad is not None else None}")
    print(f"y2.grad shape: {y2.grad.shape if y2.grad is not None else None}")

    print(
        "\nBackward pass completed. If hooks were called, you should see the hook debugging messages above."
    )

    return True


def test_dendrinet_hooks():
    print("\n\nTesting DendriNet hook functionality...")

    # Create a simple DendriNet model
    model = DendriNet(
        n_soma=10,
        branch_factors=[2],  # This means input to output ratio is 2:1
        excitatory_input_dim=20,
        excitatory_synapses_per_branch=5,
        inhibitory_input_dim=10,
        inhibitory_synapses_per_branch=3,
        topk_strategy="conductance_dynamic",
        blocklinear_strategy="conductance_dynamic",
    )

    # Set model to training mode
    model.train()

    # Create inputs
    x = torch.randn(3, 20, requires_grad=True)
    inhibitory_input = torch.randn(3, 10, requires_grad=True)

    # Forward pass
    print("\nPerforming DendriNet forward pass...")
    y = model(x, inhibitory_input)
    print(f"Output shape: {y.shape}, requires_grad: {y.requires_grad}")
    # Ensure intermediate outputs retain gradients
    y.retain_grad()

    # Loss and backward pass
    loss = y.sum()
    print("\nPerforming DendriNet backward pass...")
    loss.backward()

    # Check gradients
    print(f"\nx.grad shape: {x.grad.shape if x.grad is not None else None}")
    print(
        f"inhibitory_input.grad shape: {inhibitory_input.grad.shape if inhibitory_input.grad is not None else None}"
    )

    print(
        "\nDendriNet backward pass completed. If hooks were called, you should see the hook debugging messages above."
    )

    return True


if __name__ == "__main__":
    # Test individual layer hooks
    test_hook_call()

    # Test DendriNet hooks
    test_dendrinet_hooks()

    print("\n\nTests completed. Check the output above to confirm hook calls.")
