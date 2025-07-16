#!/usr/bin/env python3
"""
Estimate model sizes and memory requirements for ImageNet EI networks.
"""

def estimate_ei_network_size(
    input_dim=150528,  # ImageNet flattened: 224*224*3
    excitatory_dim=1024,
    inhibitory_dim=1024,
    excitatory_layer_sizes=[4096],
    inhibitory_layer_sizes=[0],
    excitatory_branch_factors=[8, 4, 2, 2],
    inhibitory_branch_factors=[],
    ee_synapses_per_branch_per_layer=[48],
    ei_synapses_per_branch_per_layer=[],
    ie_synapses_per_branch_per_layer=[48],
    ii_synapses_per_branch_per_layer=[],
    decoder_hidden_dims=[2048, 1024, 512],
    output_dim=1000
):
    """Estimate parameters and memory for EI network."""
    
    total_params = 0
    
    # Transfer function parameters
    transfer_ee_params = input_dim * excitatory_dim
    transfer_ie_params = input_dim * inhibitory_dim
    total_params += transfer_ee_params + transfer_ie_params
    
    print(f"Transfer function parameters: {(transfer_ee_params + transfer_ie_params):,}")
    
    # Layer parameters
    for layer_idx, (exc_size, inh_size) in enumerate(zip(excitatory_layer_sizes, inhibitory_layer_sizes + [0])):
        # Calculate branches per neuron
        exc_branches = 1
        for factor in excitatory_branch_factors:
            exc_branches *= factor
        
        inh_branches = 1
        if inhibitory_branch_factors:
            for factor in inhibitory_branch_factors:
                inh_branches *= factor
        
        # Excitatory neuron parameters
        if layer_idx < len(ee_synapses_per_branch_per_layer):
            ee_synapses = ee_synapses_per_branch_per_layer[layer_idx]
            exc_params = exc_size * exc_branches * ee_synapses
            total_params += exc_params
            print(f"Layer {layer_idx} excitatory: {exc_size} neurons * {exc_branches} branches * {ee_synapses} synapses = {exc_params:,} params")
        
        # Inhibitory neuron parameters (if any)
        if inh_size > 0 and layer_idx < len(ii_synapses_per_branch_per_layer):
            ii_synapses = ii_synapses_per_branch_per_layer[layer_idx]
            inh_params = inh_size * inh_branches * ii_synapses
            total_params += inh_params
            print(f"Layer {layer_idx} inhibitory: {inh_size} neurons * {inh_branches} branches * {ii_synapses} synapses = {inh_params:,} params")
    
    # Decoder parameters
    decoder_params = 0
    prev_dim = excitatory_layer_sizes[-1]  # Last excitatory layer size
    for hidden_dim in decoder_hidden_dims:
        decoder_params += prev_dim * hidden_dim + hidden_dim  # weights + bias
        prev_dim = hidden_dim
    
    # Final output layer
    decoder_params += prev_dim * output_dim + output_dim
    total_params += decoder_params
    
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Total parameters: {total_params:,}")
    
    # Memory estimation (float32 = 4 bytes per parameter)
    memory_gb = total_params * 4 / (1024**3)
    
    # Account for gradients, optimizer states (Adam: 2x params), activations
    # Conservative estimate: 4x parameters for training
    training_memory_gb = memory_gb * 4
    
    print(f"Model size: {memory_gb:.2f} GB")
    print(f"Training memory (est): {training_memory_gb:.2f} GB")
    
    return total_params, memory_gb, training_memory_gb

print("=== 2-Node ImageNet Configuration (4K excitatory neurons) ===")
params_2node, mem_2node, train_mem_2node = estimate_ei_network_size(
    excitatory_layer_sizes=[4096],
    inhibitory_layer_sizes=[0],
    decoder_hidden_dims=[2048, 1024, 512]
)

print(f"\nPer GPU (8 GPUs): {train_mem_2node/8:.2f} GB")

print("\n" + "="*60)
print("=== 4-Node ImageNet Configuration (8K excitatory neurons) ===")
params_4node, mem_4node, train_mem_4node = estimate_ei_network_size(
    excitatory_layer_sizes=[8192],
    inhibitory_layer_sizes=[0],
    decoder_hidden_dims=[4096, 2048, 1024]
)

print(f"\nPer GPU (16 GPUs): {train_mem_4node/16:.2f} GB")

print("\n" + "="*60)
print("=== Summary ===")
print(f"2-node model: {params_2node/1e9:.2f}B parameters, {train_mem_2node/8:.1f} GB per GPU")
print(f"4-node model: {params_4node/1e9:.2f}B parameters, {train_mem_4node/16:.1f} GB per GPU")
print(f"GPU capacity (H100): 80 GB")
print(f"2-node utilization: {(train_mem_2node/8)/80*100:.1f}%")
print(f"4-node utilization: {(train_mem_4node/16)/80*100:.1f}%") 