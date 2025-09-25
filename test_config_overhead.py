f  #!/usr/bin/env python
"""Test if config conversion is causing overhead."""

import time

import torch
from omegaconf import OmegaConf

from dendritic_modeling.networks.architectures.excitation_inhibition.config_adapter import (
    _config_cache,
)
from dendritic_modeling.networks.architectures.factory import get_architecture

# Create a typical EINet config
config = OmegaConf.create(
    {
        "type": "einet",
        "architecture": {
            "excitatory_layer_sizes": [128, 128, 128],
            "inhibitory_layer_sizes": [32, 32, 32],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [10, 10, 10],
            "ei_synapses_per_branch_per_layer": [5, 5, 5],
            "ie_synapses_per_branch_per_layer": [10, 10, 10],
            "ii_synapses_per_branch_per_layer": [5, 5, 5],
        },
        "transfer": {
            "function_type": "linear",
            "augment_dim": 0,
            "activation": "relu",
        },
        "morphology": {
            "excitatory_branch_factors": [10, 10, 10],
            "inhibitory_branch_factors": [5, 5, 5],
            "somatic_synapses": True,
        },
        "implementation": {
            "dbl_init_method": "analytical_expectation",
            "inhibitory_network_type": "dendritic",
            "use_shunting": True,
        },
        "sparsity": {
            "gradient_scaling": "none",
            "topk_init_method": "xavier_normal",
            "topk_noise_level": 0.0,
            "topk_type": "standard",
        },
        "reactivation": {
            "enabled": True,
            "gradient_scaling": "none",
            "type": "param_tanh",
            "init_m": 1.5,
            "init_b": 0.5,
        },
        "blocklinear": {
            "gradient_scaling": "none",
            "efficient": False,
        },
    }
)

# Test 1: Create multiple networks and see if caching works
print("Creating 10 EINet instances...")
print(f"Initial cache size: {len(_config_cache)}")

times = []
for i in range(10):
    start = time.time()
    net = get_architecture("einet", config, input_dim=784)
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"  Instance {i+1}: {elapsed:.3f}s (cache size: {len(_config_cache)})")

print(f"\nAverage time: {sum(times)/len(times):.3f}s")
print(f"First instance: {times[0]:.3f}s")
print(f"Subsequent instances: {sum(times[1:])/len(times[1:]):.3f}s")

# Test 2: Check if the config is being converted multiple times
print("\n\nTesting OmegaConf.to_container overhead...")
start = time.time()
for i in range(100):
    _ = OmegaConf.to_container(config, resolve=True)
elapsed = time.time() - start
print(
    f"100 OmegaConf.to_container calls: {elapsed:.3f}s ({elapsed/100*1000:.1f}ms per call)"
)

# Test 3: Create a simple forward pass to check runtime overhead
print("\n\nTesting forward pass performance...")
net = get_architecture("einet", config, input_dim=784)
net.eval()
x = torch.randn(32, 784)  # batch_size=32

# Warmup
with torch.no_grad():
    for _ in range(5):
        _ = net(x)

# Time forward passes
times = []
with torch.no_grad():
    for i in range(50):
        start = time.time()
        _ = net(x)
        elapsed = time.time() - start
        times.append(elapsed)

print(f"Average forward pass time: {sum(times)/len(times)*1000:.2f}ms")
print(f"Min/Max: {min(times)*1000:.2f}ms / {max(times)*1000:.2f}ms")
