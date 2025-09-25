#!/usr/bin/env python
"""Test EINet with correct branch factors."""

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork,
)

print("Testing with CORRECT branch factors (matching number of layers)...")

# Correct params - 1 layer should have 1 branch factor
params = {
    "input_dim": 784,
    "transfer_params": {
        "function_type": "linear",
        "augment_dim": 0,
        "activation": "relu",
    },
    "excitatory_layer_sizes": [20],  # 1 layer
    "inhibitory_layer_sizes": [],
    "excitatory_branch_factors": [3],  # 1 branch factor for 1 layer
    "inhibitory_branch_factors": [],
    "ee_synapses_per_branch_per_layer": [40],
    "ei_synapses_per_branch_per_layer": [0],
    "ie_synapses_per_branch_per_layer": [10],
    "ii_synapses_per_branch_per_layer": [0],
    "somatic_synapses": True,
    "dbl_init_method": "analytical_expectation",
    "inhibitory_network_type": "dendritic",
    "use_shunting": True,
    "topk_strategy": "none",
    "reactivation_strategy": "none",
    "blocklinear_strategy": "none",
    "topk_init_method": "xavier_normal",
    "topk_noise_level": 0.0,
    "topk_type": "standard",
    "reactivation_type": "param_tanh",
    "reactivation_init_m": 1.5,
    "reactivation_init_b": 0.5,
    "efficient_blocklinear": False,
}

try:
    net = ExcitationInhibitionNetwork(**params)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"Success! Network created with {n_params:,} parameters")

    # Test forward pass
    x = torch.randn(32, 784)
    with torch.no_grad():
        out = net(x)
    print(f"Forward pass successful! Output shape: {out.shape}")

    # Compare with incorrect config
    print("\n\nNow testing with INCORRECT branch factors (4 factors for 1 layer)...")
    params["excitatory_branch_factors"] = [3, 3, 3, 3]  # 4 factors for 1 layer!

    net2 = ExcitationInhibitionNetwork(**params)
    n_params2 = sum(p.numel() for p in net2.parameters())
    print(f"Network created with {n_params2:,} parameters")
    print(f"That's {n_params2 / n_params:.1f}x larger!")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
