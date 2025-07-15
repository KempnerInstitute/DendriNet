#!/usr/bin/env python
"""
Estimate EI Network model sizes for different configurations.
"""

import math
from typing import List, Dict, Tuple


def estimate_ei_network_memory(
    excitatory_layers: List[int],
    inhibitory_layers: List[int],
    excitatory_branches: List[int],
    inhibitory_branches: List[int],
    ee_synapses: List[int],
    ei_synapses: List[int],
    ie_synapses: List[int],
    ii_synapses: List[int],
    input_dim: int,
    dtype_bytes: int = 2  # 2 for fp16, 4 for fp32
) -> Dict[str, float]:
    """
    Estimate memory requirements for an EI network.
    
    Returns memory estimates in GB.
    """
    total_params = 0
    
    # Input layer connections
    # Split input to E and I pathways
    input_e_params = input_dim // 2 * excitatory_layers[0]
    input_i_params = input_dim // 2 * inhibitory_layers[0]
    total_params += input_e_params + input_i_params
    
    # Layer-wise connections
    for i in range(len(excitatory_layers) - 1):
        e_in = excitatory_layers[i]
        e_out = excitatory_layers[i + 1]
        i_in = inhibitory_layers[i] if i < len(inhibitory_layers) else 0
        i_out = inhibitory_layers[i + 1] if i + 1 < len(inhibitory_layers) else 0
        
        e_branches = excitatory_branches[i + 1] if i + 1 < len(excitatory_branches) else 1
        i_branches = inhibitory_branches[i + 1] if i + 1 < len(inhibitory_branches) else 1
        
        # E-E connections (TopK)
        if e_out > 0 and ee_synapses[i] > 0:
            # Each output neuron has branches, each branch has TopK connections
            ee_params = e_in * e_out * e_branches
            total_params += ee_params
        
        # E-I connections
        if i_out > 0 and ei_synapses[i] > 0:
            ei_params = e_in * i_out * i_branches
            total_params += ei_params
        
        # I-E connections
        if e_out > 0 and i_in > 0 and ie_synapses[i] > 0:
            ie_params = i_in * e_out * e_branches
            total_params += ie_params
        
        # I-I connections
        if i_out > 0 and i_in > 0 and ii_synapses[i] > 0:
            ii_params = i_in * i_out * i_branches
            total_params += ii_params
        
        # Branch aggregation parameters
        if i > 0:
            # Previous layer branches to current layer branches
            prev_e_branches = excitatory_branches[i] if i < len(excitatory_branches) else 1
            if e_out > 0 and prev_e_branches > 1:
                total_params += prev_e_branches * e_branches * e_out
            
            prev_i_branches = inhibitory_branches[i] if i < len(inhibitory_branches) else 1
            if i_out > 0 and prev_i_branches > 1:
                total_params += prev_i_branches * i_branches * i_out
    
    # Reactivation parameters (2 per neuron for ParametricTanh)
    total_neurons = sum(excitatory_layers) + sum(inhibitory_layers)
    reactivation_params = total_neurons * 2
    total_params += reactivation_params
    
    # Memory calculations
    param_memory_gb = (total_params * dtype_bytes) / 1e9
    gradient_memory_gb = param_memory_gb  # Same as parameters
    optimizer_memory_gb = 2 * param_memory_gb  # Adam states
    activation_memory_gb = 0.15 * param_memory_gb  # Conservative estimate
    
    total_memory_gb = (
        param_memory_gb + gradient_memory_gb + 
        optimizer_memory_gb + activation_memory_gb
    )
    
    return {
        "total_params": total_params,
        "param_memory_gb": param_memory_gb,
        "gradient_memory_gb": gradient_memory_gb,
        "optimizer_memory_gb": optimizer_memory_gb,
        "activation_memory_gb": activation_memory_gb,
        "total_memory_gb": total_memory_gb,
        "memory_per_param_bytes": dtype_bytes * 4.3,  # Includes all overheads
    }


def print_model_config(name: str, config: Dict, estimates: Dict):
    """Print model configuration and estimates."""
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Excitatory layers: {config['excitatory_layers']}")
    print(f"  Inhibitory layers: {config['inhibitory_layers']}")
    print(f"  Excitatory branches: {config['excitatory_branches']}")
    print(f"  Total parameters: {estimates['total_params']:,}")
    print(f"\nMemory Requirements:")
    print(f"  Parameters: {estimates['param_memory_gb']:.2f} GB")
    print(f"  Gradients: {estimates['gradient_memory_gb']:.2f} GB")
    print(f"  Optimizer: {estimates['optimizer_memory_gb']:.2f} GB")
    print(f"  Activations: {estimates['activation_memory_gb']:.2f} GB")
    print(f"  Total: {estimates['total_memory_gb']:.2f} GB")
    
    # GPU recommendations
    gpu_memory = 80  # H100 80GB
    gpus_needed = math.ceil(estimates['total_memory_gb'] / (gpu_memory * 0.9))
    nodes_needed = math.ceil(gpus_needed / 4)  # 4 GPUs per node
    
    print(f"\nHardware Requirements (H100 80GB):")
    print(f"  Minimum GPUs: {gpus_needed}")
    print(f"  Minimum nodes: {nodes_needed}")
    
    # With FSDP
    fsdp_memory_per_gpu = estimates['total_memory_gb'] / gpus_needed
    print(f"\nWith FSDP:")
    print(f"  Memory per GPU: {fsdp_memory_per_gpu:.2f} GB")
    print(f"  Utilization: {(fsdp_memory_per_gpu / gpu_memory) * 100:.1f}%")


def main():
    """Estimate sizes for various EI network configurations."""
    
    # Common parameters
    input_dim = 3072  # CIFAR10 flattened
    dtype_bytes = 2  # FP16
    
    # Small model (fits on 1 GPU)
    small_config = {
        "excitatory_layers": [512, 256, 128, 10],
        "inhibitory_layers": [256, 128, 64],
        "excitatory_branches": [8, 4, 2, 1],
        "inhibitory_branches": [4, 2, 1],
        "ee_synapses": [200, 100, 50, 10],
        "ei_synapses": [100, 50, 25, 0],
        "ie_synapses": [50, 25, 10, 0],
        "ii_synapses": [25, 10, 5, 0],
    }
    
    # Medium model (needs 2-4 GPUs)
    medium_config = {
        "excitatory_layers": [2048, 1024, 512, 256, 10],
        "inhibitory_layers": [1024, 512, 256, 128],
        "excitatory_branches": [16, 8, 4, 2, 1],
        "inhibitory_branches": [8, 4, 2, 1],
        "ee_synapses": [500, 300, 200, 100, 50],
        "ei_synapses": [250, 150, 100, 50, 0],
        "ie_synapses": [150, 100, 50, 25, 0],
        "ii_synapses": [100, 50, 25, 10, 0],
    }
    
    # Large model (needs 4-8 GPUs)
    large_config = {
        "excitatory_layers": [4096, 2048, 1024, 512, 256, 10],
        "inhibitory_layers": [2048, 1024, 512, 256, 128],
        "excitatory_branches": [32, 16, 8, 4, 2, 1],
        "inhibitory_branches": [16, 8, 4, 2, 1],
        "ee_synapses": [1000, 500, 300, 200, 100, 50],
        "ei_synapses": [500, 250, 150, 100, 50, 0],
        "ie_synapses": [300, 200, 100, 50, 25, 0],
        "ii_synapses": [200, 100, 50, 25, 10, 0],
    }
    
    # XLarge model (needs 8-16 GPUs / 2-4 nodes)
    xlarge_config = {
        "excitatory_layers": [8192, 4096, 2048, 1024, 512, 256, 10],
        "inhibitory_layers": [4096, 2048, 1024, 512, 256, 128],
        "excitatory_branches": [64, 32, 16, 8, 4, 2, 1],
        "inhibitory_branches": [32, 16, 8, 4, 2, 1],
        "ee_synapses": [2000, 1000, 500, 300, 200, 100, 50],
        "ei_synapses": [1000, 500, 250, 150, 100, 50, 0],
        "ie_synapses": [600, 400, 200, 100, 50, 25, 0],
        "ii_synapses": [400, 200, 100, 50, 25, 10, 0],
    }
    
    # Extreme model (needs 16+ GPUs / 4+ nodes)
    extreme_config = {
        "excitatory_layers": [16384, 8192, 4096, 2048, 1024, 512, 256, 10],
        "inhibitory_layers": [8192, 4096, 2048, 1024, 512, 256, 128],
        "excitatory_branches": [128, 64, 32, 16, 8, 4, 2, 1],
        "inhibitory_branches": [64, 32, 16, 8, 4, 2, 1],
        "ee_synapses": [4000, 2000, 1000, 500, 300, 200, 100, 50],
        "ei_synapses": [2000, 1000, 500, 250, 150, 100, 50, 0],
        "ie_synapses": [1200, 800, 400, 200, 100, 50, 25, 0],
        "ii_synapses": [800, 400, 200, 100, 50, 25, 10, 0],
    }
    
    configs = [
        ("Small EI Network", small_config),
        ("Medium EI Network", medium_config),
        ("Large EI Network", large_config),
        ("XLarge EI Network", xlarge_config),
        ("Extreme EI Network", extreme_config),
    ]
    
    print("EI Network Size Estimations")
    print("=" * 60)
    print(f"Input dimension: {input_dim}")
    print(f"Precision: FP16 ({dtype_bytes} bytes per parameter)")
    print(f"Node memory: 1300 GB")
    print(f"GPUs per node: 4")
    print(f"GPU memory: 80 GB (H100)")
    
    for name, config in configs:
        estimates = estimate_ei_network_memory(
            config["excitatory_layers"],
            config["inhibitory_layers"],
            config["excitatory_branches"],
            config["inhibitory_branches"],
            config["ee_synapses"],
            config["ei_synapses"],
            config["ie_synapses"],
            config["ii_synapses"],
            input_dim,
            dtype_bytes
        )
        print_model_config(name, config, estimates)
    
    # Maximum model for given hardware
    print("\n" + "="*60)
    print("Maximum Model Estimates for Your Hardware")
    print("="*60)
    
    # 1 node (4 GPUs, 320 GB GPU memory + 1300 GB system memory)
    # Conservative: use 70% of GPU memory for model
    single_node_memory = 4 * 80 * 0.7  # 224 GB
    print(f"\n1 Node (4 H100 GPUs):")
    print(f"  Usable GPU memory: {single_node_memory:.0f} GB")
    print(f"  Max parameters: ~{int(single_node_memory / 4.3 * 1e9):,} ({single_node_memory / 4.3:.1f}B)")
    
    # 4 nodes (16 GPUs)
    four_node_memory = 16 * 80 * 0.7  # 896 GB
    print(f"\n4 Nodes (16 H100 GPUs):")
    print(f"  Usable GPU memory: {four_node_memory:.0f} GB")
    print(f"  Max parameters: ~{int(four_node_memory / 4.3 * 1e9):,} ({four_node_memory / 4.3:.1f}B)")


if __name__ == "__main__":
    main() 