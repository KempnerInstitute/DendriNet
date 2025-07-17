#!/usr/bin/env python3
"""
Verify the parameter count for the example large FSDP model.
"""

import numpy as np

def calculate_ei_network_params():
    """Calculate parameters for the EI network in example_large_model_fsdp.yaml"""
    
    # Network architecture
    input_dim = 150528  # ImageNet flattened (224*224*3)
    excitatory_layers = [4096, 2048, 1024, 512]
    inhibitory_layers = [1024, 512, 256, 128]
    excitatory_branch_factors = [4, 4, 2]  # Cumulative: 4, 16, 32
    inhibitory_branch_factors = [2, 2, 2]   # Cumulative: 2, 4, 8
    
    # Synapses per branch
    ee_synapses = [100, 80, 60, 40]
    ei_synapses = [50, 40, 30, 20]
    ie_synapses = [80, 60, 40, 20]
    ii_synapses = [40, 30, 20, 10]
    
    # Transfer function dimensions (auto-calculated as input_dim/2 each)
    excitatory_dim = input_dim // 2  # 75264
    inhibitory_dim = input_dim // 2  # 75264
    
    total_params = 0
    
    print("=== PARAMETER CALCULATION ===\n")
    
    # 1. Transfer function (input -> E/I pathways)
    transfer_e_params = input_dim * excitatory_dim
    transfer_i_params = input_dim * inhibitory_dim
    total_params += transfer_e_params + transfer_i_params
    print(f"Transfer function:")
    print(f"  Input -> Excitatory: {input_dim:,} × {excitatory_dim:,} = {transfer_e_params:,}")
    print(f"  Input -> Inhibitory: {input_dim:,} × {inhibitory_dim:,} = {transfer_i_params:,}")
    print(f"  Subtotal: {transfer_e_params + transfer_i_params:,}")
    
    # 2. First layer (transfer outputs -> first layer)
    print(f"\nLayer 0 (Transfer -> First layer):")
    
    # Calculate branches for first layer
    exc_branches_0 = excitatory_branch_factors[0] if len(excitatory_branch_factors) > 0 else 1
    inh_branches_0 = inhibitory_branch_factors[0] if len(inhibitory_branch_factors) > 0 else 1
    
    # E pathway: transfer -> first E layer
    e0_params = excitatory_dim * excitatory_layers[0] * exc_branches_0 * ee_synapses[0]
    print(f"  E pathway: {excitatory_dim:,} × {excitatory_layers[0]:,} × {exc_branches_0} branches × {ee_synapses[0]} synapses = {e0_params:,}")
    total_params += e0_params
    
    # I pathway: transfer -> first I layer  
    i0_params = inhibitory_dim * inhibitory_layers[0] * inh_branches_0 * ii_synapses[0]
    print(f"  I pathway: {inhibitory_dim:,} × {inhibitory_layers[0]:,} × {inh_branches_0} branches × {ii_synapses[0]} synapses = {i0_params:,}")
    total_params += i0_params
    
    # Cross connections in first layer
    ei0_params = excitatory_dim * inhibitory_layers[0] * inh_branches_0 * ei_synapses[0]
    ie0_params = inhibitory_dim * excitatory_layers[0] * exc_branches_0 * ie_synapses[0]
    print(f"  E->I: {excitatory_dim:,} × {inhibitory_layers[0]:,} × {inh_branches_0} × {ei_synapses[0]} = {ei0_params:,}")
    print(f"  I->E: {inhibitory_dim:,} × {excitatory_layers[0]:,} × {exc_branches_0} × {ie_synapses[0]} = {ie0_params:,}")
    total_params += ei0_params + ie0_params
    
    # 3. Subsequent layers
    for i in range(1, len(excitatory_layers)):
        print(f"\nLayer {i}:")
        
        # Get previous layer sizes
        prev_exc = excitatory_layers[i-1]
        prev_inh = inhibitory_layers[i-1] if i-1 < len(inhibitory_layers) else 0
        
        # Current layer sizes
        curr_exc = excitatory_layers[i]
        curr_inh = inhibitory_layers[i] if i < len(inhibitory_layers) else 0
        
        # Calculate cumulative branch factors
        exc_branches = np.prod(excitatory_branch_factors[:i+1]) if i < len(excitatory_branch_factors) else np.prod(excitatory_branch_factors)
        inh_branches = np.prod(inhibitory_branch_factors[:i+1]) if i < len(inhibitory_branch_factors) else np.prod(inhibitory_branch_factors)
        
        # E->E connections
        if i < len(ee_synapses):
            ee_params = prev_exc * curr_exc * exc_branches * ee_synapses[i]
            print(f"  E->E: {prev_exc:,} × {curr_exc:,} × {exc_branches} branches × {ee_synapses[i]} synapses = {ee_params:,}")
            total_params += ee_params
        
        # E->I connections
        if curr_inh > 0 and i < len(ei_synapses):
            ei_params = prev_exc * curr_inh * inh_branches * ei_synapses[i]
            print(f"  E->I: {prev_exc:,} × {curr_inh:,} × {inh_branches} branches × {ei_synapses[i]} synapses = {ei_params:,}")
            total_params += ei_params
        
        # I->E connections
        if prev_inh > 0 and i < len(ie_synapses):
            ie_params = prev_inh * curr_exc * exc_branches * ie_synapses[i]
            print(f"  I->E: {prev_inh:,} × {curr_exc:,} × {exc_branches} branches × {ie_synapses[i]} synapses = {ie_params:,}")
            total_params += ie_params
        
        # I->I connections
        if prev_inh > 0 and curr_inh > 0 and i < len(ii_synapses):
            ii_params = prev_inh * curr_inh * inh_branches * ii_synapses[i]
            print(f"  I->I: {prev_inh:,} × {curr_inh:,} × {inh_branches} branches × {ii_synapses[i]} synapses = {ii_params:,}")
            total_params += ii_params
    
    # 4. Reactivation parameters (2 per neuron)
    total_neurons = sum(excitatory_layers) + sum(inhibitory_layers)
    reactivation_params = total_neurons * 2
    print(f"\nReactivation parameters: {total_neurons:,} neurons × 2 = {reactivation_params:,}")
    total_params += reactivation_params
    
    # 5. Decoder MLP
    decoder_input = excitatory_layers[-1]  # 512
    decoder_hidden = [2048, 1024]
    decoder_output = 1000  # ImageNet classes
    
    print(f"\nDecoder MLP:")
    # First layer
    dec1_params = decoder_input * decoder_hidden[0] + decoder_hidden[0]
    print(f"  {decoder_input} -> {decoder_hidden[0]}: {dec1_params:,}")
    total_params += dec1_params
    
    # Second layer
    dec2_params = decoder_hidden[0] * decoder_hidden[1] + decoder_hidden[1]
    print(f"  {decoder_hidden[0]} -> {decoder_hidden[1]}: {dec2_params:,}")
    total_params += dec2_params
    
    # Output layer
    dec3_params = decoder_hidden[1] * decoder_output + decoder_output
    print(f"  {decoder_hidden[1]} -> {decoder_output}: {dec3_params:,}")
    total_params += dec3_params
    
    print(f"\n=== TOTAL PARAMETERS: {total_params:,} ===")
    print(f"    In billions: {total_params/1e9:.2f}B")
    
    # Calculate ACTIVE parameters (due to TopK)
    print(f"\n=== ACTIVE PARAMETERS (TopK Sparsity) ===")
    active_params = 0
    
    # Transfer function - these are fully connected, no sparsity
    active_params += transfer_e_params + transfer_i_params
    print(f"Transfer function (fully connected): {transfer_e_params + transfer_i_params:,}")
    
    # First layer TopK connections
    # Instead of full connectivity, only K connections per output neuron
    e0_active = excitatory_layers[0] * exc_branches_0 * ee_synapses[0]
    i0_active = inhibitory_layers[0] * inh_branches_0 * ii_synapses[0]
    ei0_active = inhibitory_layers[0] * inh_branches_0 * ei_synapses[0]
    ie0_active = excitatory_layers[0] * exc_branches_0 * ie_synapses[0]
    
    print(f"\nLayer 0 active connections:")
    print(f"  E pathway: {excitatory_layers[0]:,} × {exc_branches_0} × {ee_synapses[0]} = {e0_active:,}")
    print(f"  I pathway: {inhibitory_layers[0]:,} × {inh_branches_0} × {ii_synapses[0]} = {i0_active:,}")
    print(f"  E->I: {inhibitory_layers[0]:,} × {inh_branches_0} × {ei_synapses[0]} = {ei0_active:,}")
    print(f"  I->E: {excitatory_layers[0]:,} × {exc_branches_0} × {ie_synapses[0]} = {ie0_active:,}")
    
    active_params += e0_active + i0_active + ei0_active + ie0_active
    
    # Subsequent layers
    for i in range(1, len(excitatory_layers)):
        curr_exc = excitatory_layers[i]
        curr_inh = inhibitory_layers[i] if i < len(inhibitory_layers) else 0
        
        exc_branches = np.prod(excitatory_branch_factors[:i+1]) if i < len(excitatory_branch_factors) else np.prod(excitatory_branch_factors)
        inh_branches = np.prod(inhibitory_branch_factors[:i+1]) if i < len(inhibitory_branch_factors) else np.prod(inhibitory_branch_factors)
        
        # Active connections (output_neurons × branches × K)
        if i < len(ee_synapses):
            ee_active = curr_exc * exc_branches * ee_synapses[i]
            active_params += ee_active
        
        if curr_inh > 0 and i < len(ei_synapses):
            ei_active = curr_inh * inh_branches * ei_synapses[i]
            active_params += ei_active
        
        if curr_exc > 0 and i < len(ie_synapses):
            ie_active = curr_exc * exc_branches * ie_synapses[i]
            active_params += ie_active
        
        if curr_inh > 0 and i < len(ii_synapses):
            ii_active = curr_inh * inh_branches * ii_synapses[i]
            active_params += ii_active
    
    # Add reactivation and decoder (these are not sparse)
    active_params += reactivation_params
    active_params += dec1_params + dec2_params + dec3_params
    
    print(f"\nReactivation + Decoder (fully connected): {reactivation_params + dec1_params + dec2_params + dec3_params:,}")
    print(f"\nTOTAL ACTIVE PARAMETERS: {active_params:,}")
    print(f"Active ratio: {active_params/total_params*100:.2f}%")
    print(f"Active parameters in billions: {active_params/1e9:.2f}B")
    
    # Memory calculation
    bytes_per_param_fp16 = 2
    param_memory_gb = (total_params * bytes_per_param_fp16) / 1e9
    
    print(f"\n=== MEMORY REQUIREMENTS (FP16) ===")
    print(f"Parameters only: {param_memory_gb:.2f} GB")
    print(f"With gradients: {param_memory_gb * 2:.2f} GB")
    print(f"With Adam optimizer: {param_memory_gb * 4:.2f} GB")
    print(f"With activations (estimate): {param_memory_gb * 4.5:.2f} GB")
    
    return total_params

if __name__ == "__main__":
    calculate_ei_network_params() 