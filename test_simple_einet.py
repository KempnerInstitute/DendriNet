#!/usr/bin/env python
"""Simple test to create EINet and check for issues."""

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition import (
    ExcitationInhibitionNetwork,
)

print("Creating EINet directly with minimal params...")

# Minimal params for EINet
params = {
    "input_dim": 784,
    "transfer_params": {
        "function_type": "linear",
        "augment_dim": 0,
        "activation": "relu",
    },
    "excitatory_layer_sizes": [128, 128, 128],
    "inhibitory_layer_sizes": [32, 32, 32],
    "excitatory_branch_factors": [10, 10, 10],
    "inhibitory_branch_factors": [5, 5, 5],
    "ee_synapses_per_branch_per_layer": [10, 10, 10],
    "ei_synapses_per_branch_per_layer": [5, 5, 5],
    "ie_synapses_per_branch_per_layer": [10, 10, 10],
    "ii_synapses_per_branch_per_layer": [5, 5, 5],
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
    print(
        f"Success! Network created with {sum(p.numel() for p in net.parameters())} parameters"
    )

    # Test forward pass
    x = torch.randn(1, 784)
    with torch.no_grad():
        out = net(x)
    print(f"Forward pass successful! Output shape: {out.shape}")

except Exception as e:
    print(f"Error creating network: {e}")
    import traceback

    traceback.print_exc()
