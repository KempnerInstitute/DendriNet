"""
DendriNet architectures for dendritic neural networks.

This module implements the DendriNet and DendriNetWithOutputs classes
which generate linear layers of neurons with sequential dendritic branches.
"""

import torch
import torch.nn as nn
from copy import deepcopy

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import TOPK_INIT_METHODS


class DeepstNet(nn.Module):
    """
    Generates a linear layer of neurons, each of which consists of a sequential
    structure of dendritic branches and inhibitory/excitatory inputs.

    This network models the complex interactions between excitatory and
    inhibitory inputs across multiple dendritic branches, simulating dendritic
    processing in biological neurons.
    """

    def __init__(
        self,
        n_soma,
        branch_factors,
        excitatory_input_dim,
        excitatory_synapses_per_branch,
        inhibitory_input_dim=None,
        inhibitory_synapses_per_branch=None,
        reactivate=False,
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        topk_noise_level=0.0,
        topk_type="standard",  # New parameter for TopK strategy
        weight_transform="exp",  # Weight transformation for positive weights
        topk_temperature=0.5,  # For stochastic TopK
        topk_ultrafast=False,  # For much faster stochastic TopK
        topk_strategy="none",
        use_shunting=True,
        reactivation_type="param_tanh",
        reactivation_strategy="none",
        blocklinear_strategy="none",
        print_hooks=False,
        reactivation_init_m = 1,
        reactivation_init_b = 0.5,
        dbl_init_method="analytical_expectation",
        # DEEPST specific parameters, clean up later
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode='global',
        rewire_frequency=1,
        synapses_per_branch=None,
        efficient_blocklinear: bool = False,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        freeze_all_connectivity=False,
        init_method=None,
        weight_threshold=1e-6,
        weight_init=0.01,
        input_noise=0,
        adaptive_initialization=True,  # Add adaptive initialization parameter
        **kwargs,
    ):
        super(DeepstNet, self).__init__()

        if not isinstance(n_soma, int):
            raise TypeError("n_soma must be an integer")
        if n_soma < 1:
            raise ValueError("n_soma must be >= 1")

        layer_sizes = [n_soma]

        for bf in branch_factors:
            if not isinstance(bf, int):
                raise TypeError("branch_factors must be integers")
            if bf < 1:
                raise ValueError("branch_factors must be >= 1")
        
        # Store branch factors for network-aware initialization
        self.branch_factors = deepcopy(branch_factors)

        if (
            inhibitory_input_dim is not None
            and inhibitory_synapses_per_branch is not None
        ):
            if not isinstance(inhibitory_synapses_per_branch, int):
                raise TypeError("inhibitory_synapses_per_branch must be int")
            # if inhibitory_synapses_per_branch < 1:
            #     raise ValueError("inhibitory_synapses_per_branch must be >= 1")

        if (
            excitatory_input_dim is not None
            and excitatory_synapses_per_branch is not None
        ):
            if not isinstance(excitatory_synapses_per_branch, int):
                raise TypeError("excitatory_synapses_per_branch must be int")
            if excitatory_synapses_per_branch < 1:
                raise ValueError("excitatory_synapses_per_branch must be >= 1")

        branch_factors: list = deepcopy(branch_factors)
        self.n_branch_layers = len(branch_factors)
        for i in range(self.n_branch_layers):
            layer_sizes.append(layer_sizes[i] * branch_factors[i])
        layer_sizes.reverse()
        branch_factors.reverse()
        branch_factors.insert(0, None)
        branch_layers = []
        for i in range(self.n_branch_layers):
            lyr = DendriticBranchLayer(
                output_dim=layer_sizes[i],
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=excitatory_synapses_per_branch,
                inhibitory_input_dim=inhibitory_input_dim,
                inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
                input_branch_factor=branch_factors[i],
                topk_init_method=topk_init_method,
                topk_noise_level=topk_noise_level,
                topk_type=topk_type,  # Pass the parameter
                weight_transform=weight_transform,
                topk_temperature=topk_temperature,  # Pass the temperature
                topk_ultrafast=topk_ultrafast,  # Pass the ultrafast flag
                topk_strategy=topk_strategy,
                use_shunting=use_shunting,
                reactivate=reactivate,
                reactivation_init_m=reactivation_init_m,
                reactivation_init_b=reactivation_init_b,
                dbl_init_method=dbl_init_method,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
                efficient_blocklinear=efficient_blocklinear,
                layer_idx=self.n_branch_layers - i,
                print_hooks=print_hooks,
                use_noise=use_noise,
                sigma=sigma,
                rewiring_mode=rewiring_mode,
                rewire_frequency=rewire_frequency,
                synapses_per_branch=synapses_per_branch,
                excitatory_target_density=excitatory_target_density,
                inhibitory_target_density=inhibitory_target_density,
                freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                init_method=init_method,
                weight_threshold=weight_threshold,
                weight_init=weight_init,
                input_noise=input_noise,
                adaptive_initialization=adaptive_initialization,
                branch_factors=self.branch_factors,  # Pass branch_factors
                **kwargs
            )
            branch_layers.append(lyr)

        if somatic_synapses:
            branch_layers.append(
                DendriticBranchLayer(
                    output_dim=n_soma,
                    excitatory_input_dim=excitatory_input_dim,
                    excitatory_synapses_per_branch=excitatory_synapses_per_branch,
                    inhibitory_input_dim=inhibitory_input_dim,
                    inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
                    input_branch_factor=branch_factors[-1],
                    topk_init_method=topk_init_method,
                    topk_noise_level=topk_noise_level,
                    topk_type=topk_type,  # Pass the parameter
                    weight_transform=weight_transform,
                    topk_temperature=topk_temperature,  # Pass the temperature
                    topk_ultrafast=topk_ultrafast,  # Pass the ultrafast flag
                    topk_strategy=topk_strategy,
                    use_shunting=use_shunting,
                    reactivate=reactivate,
                    reactivation_init_m=reactivation_init_m,
                    reactivation_init_b=reactivation_init_b,
                    dbl_init_method=dbl_init_method,
                    reactivation_type=reactivation_type,
                    reactivation_strategy=reactivation_strategy,
                    blocklinear_strategy=blocklinear_strategy,
                    efficient_blocklinear=efficient_blocklinear,
                    layer_idx=0,
                    print_hooks=print_hooks,
                    use_noise=use_noise,
                    sigma=sigma,
                    rewiring_mode=rewiring_mode,
                    rewire_frequency=rewire_frequency,
                    synapses_per_branch=synapses_per_branch,
                    excitatory_target_density=excitatory_target_density,
                    inhibitory_target_density=inhibitory_target_density,
                    freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                    freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                    init_method=init_method,
                    weight_threshold=weight_threshold,
                    weight_init=weight_init,
                    input_noise=input_noise,
                    adaptive_initialization=adaptive_initialization,
                    branch_factors=self.branch_factors,  # Pass branch_factors
                    **kwargs
                )
            )
        else:
            branch_layers.append(
                DendriticBranchLayer(
                    output_dim=n_soma,
                    excitatory_input_dim=None,
                    excitatory_synapses_per_branch=None,
                    inhibitory_input_dim=None,
                    inhibitory_synapses_per_branch=None,
                    input_branch_factor=branch_factors[-1],
                    topk_init_method=topk_init_method,
                    topk_noise_level=topk_noise_level,
                    topk_type=topk_type,  # Pass the parameter
                    weight_transform=weight_transform,
                    topk_temperature=topk_temperature,  # Pass the temperature
                    topk_ultrafast=topk_ultrafast,  # Pass the ultrafast flag
                    topk_strategy=topk_strategy,
                    use_shunting=use_shunting,
                    reactivate=reactivate,
                    reactivation_init_m=reactivation_init_m,
                    reactivation_init_b=reactivation_init_b,
                    dbl_init_method=dbl_init_method,
                    reactivation_type=reactivation_type,
                    reactivation_strategy=reactivation_strategy,
                    blocklinear_strategy=blocklinear_strategy,
                    efficient_blocklinear=efficient_blocklinear,
                    layer_idx=0,
                    print_hooks=print_hooks,
                    use_noise=use_noise,
                    sigma=sigma,
                    rewiring_mode=rewiring_mode,
                    rewire_frequency=rewire_frequency,
                    synapses_per_branch=synapses_per_branch,
                    excitatory_target_density=excitatory_target_density,
                    inhibitory_target_density=inhibitory_target_density,
                    freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                    freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                    init_method=init_method,
                    weight_threshold=weight_threshold,
                    weight_init=weight_init,
                    input_noise=input_noise,
                    adaptive_initialization=adaptive_initialization,
                    branch_factors=self.branch_factors,  # Pass branch_factors
                    **kwargs
                )
            )

        self.branch_layers = nn.ModuleList(branch_layers)

        self.reactivate = reactivate
        self.layer_sizes = layer_sizes
        self.input_inhibitory = inhibitory_input_dim is not None
        self.n_soma = n_soma
        self.somatic_synapses = somatic_synapses
        self.efficient_blocklinear = efficient_blocklinear
        
        # Network-level rewiring parameters
        self.rewiring_mode = rewiring_mode
        self.rewire_frequency = rewire_frequency
        self.step_counter = 0
        self.freeze_all_connectivity = freeze_all_connectivity
        
        # Create branch-to-neuron mapping for local rewiring
        self._create_branch_to_neuron_mapping()
        
        # Pre-compute DeepST layers for efficient network rewiring
        self._collect_deepst_layers()
    
    def _create_branch_to_neuron_mapping(self):
        """Create mapping from (layer_idx, branch_idx) to neuron_idx."""
        self.branch_to_neuron_mapping = {}
        
        # Calculate layer sizes
        layer_sizes = [self.n_soma]
        for bf in self.branch_factors:
            if bf is not None:  # Add this check
                layer_sizes.append(layer_sizes[-1] * bf)
            else:
                # Handle None branch factors - skip or use default
                layer_sizes.append(layer_sizes[-1])  # Keep same size
        
        layer_sizes.reverse()
        
        # Create mapping for each layer
        current_neuron = 0
        for layer_idx, layer_size in enumerate(layer_sizes):
            branches_per_neuron = layer_size // self.n_soma
            
            for branch_idx in range(layer_size):
                neuron_idx = branch_idx // branches_per_neuron
                if neuron_idx < self.n_soma:  # Ensure we don't exceed neuron count
                    self.branch_to_neuron_mapping[(layer_idx, branch_idx)] = neuron_idx
    
    def _collect_deepst_layers(self):
        """Collect all DeepST layers within this DeepstNet for network-level coordination."""
        self.deepst_layers = []
        self.layer_types = []  # Track whether each layer is excitatory or inhibitory
        
        def find_deepst_layers(module, parent_name=""):
            for name, child in module.named_children():
                current_name = f"{parent_name}.{name}" if parent_name else name
                
                if hasattr(child, 'mask') and hasattr(child, 'weight') and hasattr(child, 'weight_threshold'):
                    # This is a DeepST layer
                    self.deepst_layers.append(child)
                    
                    # Determine if this is excitatory or inhibitory based on the layer name
                    if 'branch_excitation' in current_name:
                        self.layer_types.append('excitatory')
                    elif 'branch_inhibition' in current_name:
                        self.layer_types.append('inhibitory')
                    else:
                        # Default to inhibitory if we can't determine
                        self.layer_types.append('inhibitory')
                        print(f"WARNING: Could not determine layer type for: {current_name}")
                
                # Recursively search child modules
                find_deepst_layers(child, current_name)
        
        find_deepst_layers(self)
    
    def apply_rewiring(self):
        """Apply network-level rewiring across all DeepST layers in this DeepstNet."""
        self.step_counter += 1
        if self.step_counter % self.rewire_frequency != 0:
            return
        if self.freeze_all_connectivity: ## TODO: ADDED, need to test
            return
        if self.rewiring_mode == 'global':
            self._apply_global_network_rewiring()
        elif self.rewiring_mode == 'local':
            self._apply_local_network_rewiring() ######### TODO: NOT FUNCTIONING CORRECTLY
        else:
            # Fallback to individual layer rewiring
            for layer in self.branch_layers:
                layer.apply_rewiring()
    
    def _apply_global_network_rewiring(self):
        """Apply global rewiring across all DeepST layers."""
        if not self.deepst_layers:
            return
        
        excitatory_layers, inhibitory_layers = self._separate_layers_by_type()
        
        if excitatory_layers:
            self._apply_global_rewiring_by_type(excitatory_layers, "excitatory")
        if inhibitory_layers:
            self._apply_global_rewiring_by_type(inhibitory_layers, "inhibitory")

    def _apply_global_rewiring_by_type(self, layers, synapse_type):
        """Apply global rewiring only within the same synapse type."""
        # Collect pruned and potential masks for this synapse type only
        all_pruned_masks = []
        all_potential_masks = []
        all_layers = []
        
        for deepst_layer in layers:
            # Apply noise and get current weights
            current_weights = deepst_layer.weight()
            
            # Apply noise if enabled
            if hasattr(deepst_layer, 'use_noise') and deepst_layer.use_noise:
                noise = torch.randn_like(current_weights) * deepst_layer.sigma
                current_weights = current_weights + noise * deepst_layer.mask
            
            # Find pruned and potential connections
            pruned_mask = (current_weights <= deepst_layer.weight_threshold) & deepst_layer.mask
            potential_mask = ~deepst_layer.mask
            
            all_pruned_masks.append(pruned_mask)
            all_potential_masks.append(potential_mask)
            all_layers.append(deepst_layer)
        
        # Calculate totals for this synapse type only
        total_pruned = sum(mask.sum().item() for mask in all_pruned_masks)
        total_potential = sum(mask.sum().item() for mask in all_potential_masks)
        
        if total_pruned == 0 or total_potential == 0:
            return
        
        # Step 3: Create global potential pool with precise index tracking
        global_to_layer_mapping = []
        
        for layer_idx, potential_mask in enumerate(all_potential_masks):
            num_potential = potential_mask.sum().item()
            for local_idx in range(num_potential):
                global_to_layer_mapping.append((layer_idx, local_idx))
        
        # Step 4: Ensure we add exactly the same number of connections that we prune
        num_to_add = total_pruned
        
        # Safety check: ensure we don't exceed available potential connections
        if len(global_to_layer_mapping) < num_to_add:
            num_to_add = len(global_to_layer_mapping)
            print(f"Warning: Only {num_to_add} connections available, but {total_pruned} were pruned")
        
        # Randomly select global indices
        selected_global_indices = torch.randperm(len(global_to_layer_mapping))[:num_to_add]
        
        # Step 5: Distribute selected connections back to layers using vectorized operations
        total_added = 0  # Track total connections added
        
        # VECTORIZED: Pre-compute layer assignments for all selected indices
        selected_layer_indices = torch.tensor([global_to_layer_mapping[global_idx][0] for global_idx in selected_global_indices], 
                                            device=selected_global_indices.device)
        
        for i, (deepst_layer, pruned_mask, potential_mask) in enumerate(zip(all_layers, all_pruned_masks, all_potential_masks)):
            # Remove pruned connections first
            num_pruned_this_layer = pruned_mask.sum().item()
            deepst_layer.mask[pruned_mask] = False
            
            # VECTORIZED: Find which selected global indices belong to this layer
            layer_mask = (selected_layer_indices == i)
            
            # Get the local indices for this layer
            layer_selected_global = selected_global_indices[layer_mask]
            layer_selected_local = torch.tensor([global_to_layer_mapping[global_idx][1] for global_idx in layer_selected_global], 
                                              device=potential_mask.device)
            
            # Initialize new_connections_mask to empty
            new_connections_mask = torch.zeros_like(potential_mask, dtype=torch.bool)
            
            # GUARANTEED: Convert local flat indices back to 2D indices with precision
            potential_indices_2d = torch.nonzero(potential_mask, as_tuple=False)
            if len(layer_selected_local) > 0 and len(potential_indices_2d) > 0:
                # GUARANTEE: These indices MUST be valid since they come from the same potential_mask
                # that generated potential_indices_2d. The mapping is 1:1.
                selected_2d_positions = potential_indices_2d[layer_selected_local]
                rows = selected_2d_positions[:, 0]
                cols = selected_2d_positions[:, 1]
                
                # GUARANTEE: Create new connections mask using vectorized operations
                new_connections_mask[rows, cols] = True
                
                # GUARANTEE: We should add exactly the number of indices we had
                expected_added = len(layer_selected_local)
                actual_added = new_connections_mask.sum().item()
                
                # VERIFICATION: Ensure we added exactly what we expected
                if actual_added != expected_added:
                    print(f"ERROR: Layer {i} expected {expected_added} but added {actual_added}")
                    # Force the correct number by adding any missing connections
                    missing_count = expected_added - actual_added
                    if missing_count > 0:
                        # Find any remaining potential positions and add them
                        remaining_potential = potential_mask & ~new_connections_mask
                        remaining_indices = torch.nonzero(remaining_potential, as_tuple=False)
                        if len(remaining_indices) >= missing_count:
                            additional_indices = remaining_indices[:missing_count]
                            new_connections_mask[additional_indices[:, 0], additional_indices[:, 1]] = True
                            actual_added = new_connections_mask.sum().item()
                
                num_added_this_layer = actual_added
                total_added += num_added_this_layer
                
            # Add new connections (new_connections_mask is now always defined)
            deepst_layer.mask[new_connections_mask] = True
            
            # Initialize new weights in batch using vectorized operations
            if new_connections_mask.sum() > 0:
                new_weights = self._initialize_new_weights_batch(
                    deepst_layer, new_connections_mask.sum().item()
                )
                deepst_layer.pre_w.data[new_connections_mask] = new_weights

    def _apply_local_network_rewiring(self):
        """Apply local rewiring within each neuron."""
        if not self.deepst_layers:
            return
        
        excitatory_layers, inhibitory_layers = self._separate_layers_by_type()
        
        if excitatory_layers:
            self._apply_local_rewiring_by_type(excitatory_layers, "excitatory")
        if inhibitory_layers:
            self._apply_local_rewiring_by_type(inhibitory_layers, "inhibitory")

    def _old_apply_local_rewiring_by_type(self, layers, synapse_type):
        """Apply local rewiring using fully vectorized operations."""
        
        # Step 1: Create stacked tensors for all layers
        all_pruned_masks = []
        all_potential_masks = []
        all_layer_indices = []
        all_branch_indices = []
        all_layer_neurons = []
        
        for layer_idx, deepst_layer in enumerate(layers):
            # Apply noise and get current weights
            current_weights = deepst_layer.weight()
            
            if hasattr(deepst_layer, 'use_noise') and deepst_layer.use_noise:
                noise = torch.randn_like(current_weights) * deepst_layer.sigma
                current_weights = current_weights + noise * deepst_layer.mask
            
            # Find pruned and potential connections
            pruned_mask = (current_weights <= deepst_layer.weight_threshold) & deepst_layer.mask
            potential_mask = ~deepst_layer.mask
            
            # Create neuron mapping for this layer
            layer_neurons = torch.zeros(deepst_layer.out_features, dtype=torch.long, device=deepst_layer.mask.device)
            for branch_idx in range(deepst_layer.out_features):
                neuron_idx = self.branch_to_neuron_mapping.get((layer_idx, branch_idx))
                if neuron_idx is not None:
                    layer_neurons[branch_idx] = neuron_idx
            
            all_pruned_masks.append(pruned_mask)
            all_potential_masks.append(potential_mask)
            all_layer_indices.append(torch.full((deepst_layer.out_features,), layer_idx, dtype=torch.long, device=deepst_layer.mask.device))
            all_branch_indices.append(torch.arange(deepst_layer.out_features, dtype=torch.long, device=deepst_layer.mask.device))
            all_layer_neurons.append(layer_neurons)
        
        # Step 2: Stack all masks into large tensors
        stacked_pruned = torch.cat(all_pruned_masks, dim=0)
        stacked_potential = torch.cat(all_potential_masks, dim=0)
        stacked_layer_indices = torch.cat(all_layer_indices, dim=0)
        stacked_branch_indices = torch.cat(all_branch_indices, dim=0)
        stacked_neurons = torch.cat(all_layer_neurons, dim=0)
        
        # Step 3: Process ALL neurons at once using vectorized operations
        n_neurons = self.n_soma
        
        # Create neuron masks for all neurons at once - ensure same device
        neuron_masks = (stacked_neurons.unsqueeze(1) == torch.arange(n_neurons, device=stacked_neurons.device).unsqueeze(0))
        
        # Calculate totals for all neurons at once
        neuron_pruned_totals = (stacked_pruned.unsqueeze(2) & neuron_masks.unsqueeze(1)).sum(dim=(0, 1))
        neuron_potential_totals = (stacked_potential.unsqueeze(2) & neuron_masks.unsqueeze(1)).sum(dim=(0, 1))
        
        # Find neurons that have both pruned and potential connections
        active_neurons = (neuron_pruned_totals > 0) & (neuron_potential_totals > 0)
        
        # Step 4: Process all active neurons simultaneously
        active_neuron_indices = torch.nonzero(active_neurons).squeeze(1)
        
        # For each active neuron, create the global potential pool
        for neuron_idx in active_neuron_indices:
            # Get branches that belong to this neuron
            neuron_mask = (stacked_neurons == neuron_idx)
            
            # Extract pruned and potential connections for this neuron
            neuron_pruned = stacked_pruned[neuron_mask]
            neuron_potential = stacked_potential[neuron_mask]
            neuron_layer_indices = stacked_layer_indices[neuron_mask]
            neuron_branch_indices = stacked_branch_indices[neuron_mask]
            
            # Calculate totals for this neuron
            total_pruned = neuron_pruned.sum().item()
            total_potential = neuron_potential.sum().item()
            
            # CRITICAL FIX: Define number to add based EXACTLY on number pruned
            num_to_add = total_pruned  # Direct dependency
            
            # CRITICAL FIX: Safety check - don't exceed available potential
            if total_potential < num_to_add:
                num_to_add = total_potential
            
            # CRITICAL FIX: Only proceed if we can add at least one connection
            # Step 5: Create global potential pool for this neuron - VECTORIZED
            branch_potential_counts = neuron_potential.sum(dim=1)
            branch_indices = torch.repeat_interleave(torch.arange(len(branch_potential_counts), device=neuron_potential.device), branch_potential_counts)
            
            # Step 6: Randomly select exactly num_to_add connections for this neuron
            selected_global_indices = torch.randperm(total_potential, device=neuron_potential.device)[:num_to_add]
            
            # Step 7: Distribute selected connections back to layers - VECTORIZED
            selected_branch_indices = branch_indices[selected_global_indices]
            
            # VECTORIZED: Group selected indices by branch and process all branches at once
            unique_branches = torch.unique(selected_branch_indices)
            
            # Track total connections added to ensure we don't exceed the number pruned
            total_added = 0
            
            # Create masks for each unique branch
            for branch_idx in unique_branches:
                # Find which selected indices belong to this branch
                branch_mask = (selected_branch_indices == branch_idx)
                
                if not branch_mask.any():
                    continue
                
                # Get the original layer and branch indices
                original_layer_idx = neuron_layer_indices[branch_idx].item()
                original_branch_idx = neuron_branch_indices[branch_idx].item()
                deepst_layer = layers[original_layer_idx]
                
                # Get local indices for this branch
                branch_selected_local = torch.nonzero(branch_mask, as_tuple=False).squeeze(1)
                
                # Remove pruned connections for this branch
                branch_pruned = neuron_pruned[branch_idx]
                deepst_layer.mask[original_branch_idx] &= ~branch_pruned
                
                # Add new connections for this branch
                if len(branch_selected_local) > 0:
                    # Get potential indices for this branch
                    branch_potential = neuron_potential[branch_idx]
                    branch_potential_indices = torch.nonzero(branch_potential, as_tuple=False)
                    
                    if len(branch_potential_indices) > 0:
                        # Select the actual input indices
                        selected_input_indices = branch_potential_indices[branch_selected_local]
                        
                        # Add new connections
                        deepst_layer.mask[original_branch_idx, selected_input_indices] = True
                        
                        # Initialize new weights
                        new_weights = self._initialize_new_weights_batch(deepst_layer, len(selected_input_indices))
                        deepst_layer.pre_w.data[original_branch_idx, selected_input_indices] = new_weights
                        
                        # Track total added
                        total_added += len(selected_input_indices)

    def _apply_local_rewiring_by_type(self, layers, synapse_type):
        """Apply local rewiring using vectorized operations step by step."""
        
        # Step 1: Generate pruned weight masks for each layer
        pruned_masks = []
        potential_masks = []
        layer_neurons = []  # Maps branch -> neuron for each layer
        
        for layer_idx, deepst_layer in enumerate(layers):
            # Apply noise and get current weights
            current_weights = deepst_layer.weight()
            
            if hasattr(deepst_layer, 'use_noise') and deepst_layer.use_noise:
                noise = torch.randn_like(current_weights) * deepst_layer.sigma
                current_weights = current_weights + noise * deepst_layer.mask
            
            # Find pruned connections (weight <= threshold AND currently active)
            pruned_mask = (current_weights <= deepst_layer.weight_threshold) & deepst_layer.mask
            
            # Find potential connections (currently inactive)
            potential_mask = ~deepst_layer.mask
            
            # Create neuron mapping for this layer
            layer_neuron_map = torch.zeros(deepst_layer.out_features, dtype=torch.long, device=deepst_layer.mask.device)
            for branch_idx in range(deepst_layer.out_features):
                neuron_idx = self.branch_to_neuron_mapping.get((layer_idx, branch_idx))
                if neuron_idx is not None:
                    layer_neuron_map[branch_idx] = neuron_idx
            
            pruned_masks.append(pruned_mask)
            potential_masks.append(potential_mask)
            layer_neurons.append(layer_neuron_map)
        
        # Step 2: Stack all masks into large tensors
        stacked_pruned = torch.cat(pruned_masks, dim=0)  # Shape: (total_branches, input_dim)
        stacked_potential = torch.cat(potential_masks, dim=0)  # Shape: (total_branches, input_dim)
        stacked_neurons = torch.cat(layer_neurons, dim=0)  # Shape: (total_branches,)

        # Step 3: Create neuron masks for each neuron
        n_neurons = self.n_soma
        neuron_masks = (stacked_neurons.unsqueeze(1) == torch.arange(n_neurons, device=stacked_neurons.device).unsqueeze(0))
        # Shape: (total_branches, n_neurons) - True where branch belongs to neuron
        
        # Step 4: Element-wise multiplication to get neuron-specific masks
        # Expand neuron_masks to match the weight dimensions
        neuron_masks_expanded = neuron_masks.unsqueeze(2).expand(-1, -1, stacked_pruned.shape[1])
        # Shape: (total_branches, n_neurons, input_dim)
        
        # Pruned mask subset: True only for pruned connections within each neuron
        pruned_mask_subset = neuron_masks_expanded & stacked_pruned.unsqueeze(1)
        # Shape: (total_branches, n_neurons, input_dim)
        
        # Potential mask subset: True only for potential connections within each neuron
        potential_mask_subset = neuron_masks_expanded & stacked_potential.unsqueeze(1)
        # Shape: (total_branches, n_neurons, input_dim)
        
        # Step 5: Calculate number of pruned weights per neuron
        num_pruned_per_neuron = pruned_mask_subset.sum(dim=(0, 2))  # Sum over branches and inputs
        # Shape: (n_neurons,)
        
        # Step 6: Process each neuron INDEPENDENTLY (allow overlap)
        for neuron_idx in range(n_neurons):
            if num_pruned_per_neuron[neuron_idx] == 0:
                continue
            
            # Get potential connections for this neuron (ALL potential connections, not just unused ones)
            neuron_potential_mask = potential_mask_subset[:, neuron_idx, :]  # Shape: (total_branches, input_dim)
            
            # Find indices of potential connections
            potential_indices = torch.nonzero(neuron_potential_mask, as_tuple=False)
            # Shape: (num_potential_connections, 2) - (branch_idx, input_idx)
            
            # FIXED: Always add exactly the number that was pruned (allow overlap)
            num_to_add = num_pruned_per_neuron[neuron_idx].item()
            
            # Step 7: Randomly select potential connections (allow overlap with other neurons)
            selected_indices = torch.randperm(len(potential_indices), device=potential_indices.device)[:num_to_add]
            selected_potential = potential_indices[selected_indices]
            
            # Step 8: Create masks for the selected connections
            new_connections_mask = torch.zeros_like(neuron_potential_mask)
            new_connections_mask[selected_potential[:, 0], selected_potential[:, 1]] = True
            
            # Step 9: Update layers with pruned and new connections
            connections_added = 0
            
            # Process each layer
            branch_offset = 0
            for layer_idx, deepst_layer in enumerate(layers):
                layer_branches = deepst_layer.out_features
                
                # Get the portion of masks for this layer
                layer_pruned_mask = pruned_mask_subset[branch_offset:branch_offset + layer_branches, neuron_idx, :]
                layer_new_connections_mask = new_connections_mask[branch_offset:branch_offset + layer_branches, :]
                
                # Remove pruned connections
                deepst_layer.mask &= ~layer_pruned_mask
                
                # Add new connections (allow overlap - multiple neurons can claim the same synapse)
                deepst_layer.mask |= layer_new_connections_mask
                
                # Initialize weights for new connections
                new_connection_indices = torch.nonzero(layer_new_connections_mask, as_tuple=False)
                if len(new_connection_indices) > 0:
                    new_weights = self._initialize_new_weights_batch(deepst_layer, len(new_connection_indices))
                    deepst_layer.pre_w.data[new_connection_indices[:, 0], new_connection_indices[:, 1]] = new_weights
                    connections_added += len(new_connection_indices)
                
                branch_offset += layer_branches
            

    def _separate_layers_by_type(self):
        """Separate DeepST layers into excitatory and inhibitory."""
        excitatory_layers = []
        inhibitory_layers = []
        
        for i, (deepst_layer, layer_type) in enumerate(zip(self.deepst_layers, self.layer_types)):
            if layer_type == 'excitatory':
                excitatory_layers.append(deepst_layer)
            else:
                inhibitory_layers.append(deepst_layer)
        
        return excitatory_layers, inhibitory_layers

    def _process_layer_masks(self, layers):
        """Process layers to get pruned and potential masks."""
        all_pruned_masks = []
        all_potential_masks = []
        
        for layer_idx, deepst_layer in enumerate(layers):
            # Apply noise and get current weights
            current_weights = deepst_layer.weight()
            
            if hasattr(deepst_layer, 'use_noise') and deepst_layer.use_noise:
                noise = torch.randn_like(current_weights) * deepst_layer.sigma
                current_weights = current_weights + noise * deepst_layer.mask
            
            # Find pruned and potential connections
            pruned_mask = (current_weights <= deepst_layer.weight_threshold) & deepst_layer.mask
            potential_mask = ~deepst_layer.mask
            
            all_pruned_masks.append(pruned_mask)
            all_potential_masks.append(potential_mask)
        
        return all_pruned_masks, all_potential_masks

    def _create_new_connections_mask(self, potential_mask, num_connections, deepst_layer):
        """Create a mask for new connections using vectorized operations."""
        if num_connections == 0:
            return torch.zeros_like(potential_mask, dtype=torch.bool)
        
        # Get indices of potential connections
        potential_indices = torch.nonzero(potential_mask, as_tuple=False)
        
        if len(potential_indices) == 0:
            return torch.zeros_like(potential_mask, dtype=torch.bool)
        
        # Randomly select indices
        num_to_select = min(num_connections, len(potential_indices))
        selected_indices = potential_indices[torch.randperm(len(potential_indices))[:num_to_select]]
        
        # Create mask for selected connections
        new_connections_mask = torch.zeros_like(potential_mask, dtype=torch.bool)
        new_connections_mask[selected_indices[:, 0], selected_indices[:, 1]] = True
        
        return new_connections_mask

    def _initialize_new_weights_batch(self, deepst_layer, num_weights): #TODO: use weight threshold or change topk_init_methods, make consistent regardless
        """Initialize new weights in batch using vectorized operations."""
        if deepst_layer.weight_transform == 'exp':
            values = torch.exp(torch.tensor(deepst_layer.weight_init))
        elif deepst_layer.weight_transform == 'relu':
            values = torch.relu(torch.tensor(deepst_layer.weight_init))
        elif deepst_layer.weight_transform == 'softplus':
            values = torch.nn.functional.softplus(torch.tensor(deepst_layer.weight_init))
        else:
            raise ValueError(f"Unsupported weight transform type: {deepst_layer.weight_transform}")
        
        return values

    def decay_weights(self, weight_decay, weight_boosting=False):
        for branch_layer in self.branch_layers:
            branch_layer.decay_weights(weight_decay, weight_boosting)

    def forward(self, x, inhibitory_input=None):
        output = None
        for i in range(self.n_branch_layers + 1):
            # Store excitatory and inhibitory inputs for visualization if needed
            # The excitatory input for the current layer is always x
            if hasattr(self.branch_layers[i], "excitatory_input"):
                self.branch_layers[i].excitatory_input = x

            # The inhibitory input is whatever was passed in or None
            if hasattr(self.branch_layers[i], "inhibitory_input"):
                self.branch_layers[i].inhibitory_input = inhibitory_input

            output = self.branch_layers[i](x, inhibitory_input, output)
        return output

    def sum_weights(self, pruned=False):
        exc_total = 0
        if self.input_inhibitory:
            inh_total = 0

        n_layers = self.n_branch_layers + 1
        for i in range(n_layers):
            layer: DendriticBranchLayer = self.branch_layers[i]

            if layer.input_inhibitory and layer.input_excitatory:
                exc, inh = layer.get_pruned_weights() if pruned else layer.get_weights()
                exc_chunk = exc.chunk(self.n_soma, dim=0)
                inh_chunk = inh.chunk(self.n_soma, dim=0)

                exc_total = exc_total + torch.stack(
                    [chunk.sum(0) for chunk in exc_chunk], dim=0
                )
                inh_total = inh_total + torch.stack(
                    [chunk.sum(0) for chunk in inh_chunk], dim=0
                )

            elif layer.input_excitatory:
                exc = layer.get_pruned_weights() if pruned else layer.get_weights()
                exc_chunk = exc.chunk(self.n_soma, dim=0)
                exc_total = exc_total + torch.stack(
                    [chunk.sum(0) for chunk in exc_chunk], dim=0
                )

        if self.input_inhibitory:
            return exc_total, inh_total
        else:
            return exc_total

    def log_sum_weights(self, pruned=False):
        if self.input_inhibitory:
            exc_total, inh_total = self.sum_weights(pruned=pruned)
            return (exc_total + 1e-8).log(), (inh_total + 1e-8).log()
        else:
            exc_total = self.sum_weights(pruned=pruned)
            return (exc_total + 1e-8).log()


__all__ = ["DeepstNet"] 