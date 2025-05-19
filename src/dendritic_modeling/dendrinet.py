"""
dendrinet.py
============
This module contains the DendriNet and DendriticBranchLayer classes.
"""

import math
from copy import deepcopy
from math import floor, log

import torch
import torch.nn as nn

from dendritic_modeling.activation_functions import ActivationFactory
from dendritic_modeling.gradient_scaling import GradientScaler
from dendritic_modeling.initialize import initialize_dendritic_branch_layer

TOPK_INIT_METHODS = {
    "xavier_normal": nn.init.xavier_normal_,
    "xavier_uniform": nn.init.xavier_uniform_,
    "kaiming_normal": nn.init.kaiming_normal_,
    "kaiming_uniform": nn.init.kaiming_uniform_,
    "orthogonal": nn.init.orthogonal_,
    "normal": nn.init.normal_,
    "uniform": nn.init.uniform_,
    "eye": nn.init.eye_,
}


class TopKLinear(nn.Module):
    """
    A linear layer that retains only the top K strongest synaptic weights for
    in_features.

    This module implements a linear transformation with weights constrained to
    be positive. For each output neuron, only the top K weights are kept
    , and the rest are set to zero. This simulates a neuron receiving
    inputs only from its strongest synaptic connections.

    Parameters
    ----------

    in_features : int
        The number of input features.

    out_features : int
        The number of output features.

    K : int
        The number of strongest synapses to keep per dendritic branch.

    param_space : str, optional
        The parameter space for the weights. Options are 'log' and 'presigmoid'.
        If `'log'`, the weights are parameterized as exponentials of `pre_w`.
        If `'presigmoid'`, the weights are parameterized as sigmoids of `pre_w`.
        Defaults to 'log'.

    Attributes
    ----------

    pre_w : torch.nn.Parameter
        The raw weights before applying the exponential or sigmoid
        transformation. Initialized with small negative values to ensure
        positive weights after transformation.

    K : int
        The number of strongest synapses to keep per dendritic branch.

    param_space : str
        The parameter space for the weights.

    Methods
    -------

    forward(x)
        Performs a forward pass through the layer.

    weight()
        Returns the transformed synaptic weights after applying the exponential
        or sigmoid.

    weight_mask()
        Returns a mask tensor indicating the top K synaptic connections per
        output neuron.

    pruned_weight()
        Returns the pruned synaptic weights after applying the mask.

    weighted_synapses(cell_weights, prune=False)
        Returns the weighted synapses for a given set of cell

    Notes
    -----
    - All weights are constrained to be positive.
    - The prunning is done dynamically during the forward pass.
    """

    def __init__(
        self,
        in_features,
        out_features,
        K,
        param_space="log",
        init_method="xavier_normal",
        init_gain=1.0,
        noise_level=0.0,
    ):
        super(TopKLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method
        self.init_gain = init_gain
        self.noise_level = noise_level

        self.pre_w = nn.Parameter(
            torch.empty((out_features, in_features)), requires_grad=True
        )

        if not isinstance(K, int):
            K = int(K)  # Ensure K is an integer

        if K < 1:
            raise ValueError("K must be >= 1")

        if K > in_features:
            raise ValueError(
                f"K must be <= number of input features. (K = {K}, in_features = {in_features})"
            )

        self.K = K
        self.param_space = param_space
        self.init_method = init_method
        self.input_scale_vec = None  # may be used for gradient scaling
        self.param_scale_vec = None  # may be used for gradient scaling

    def initialize(self):
        if self.init_method in TOPK_INIT_METHODS:
            init_func = TOPK_INIT_METHODS[self.init_method]
            init_func(self.pre_w)
            self.pre_w.data.mul_(self.init_gain)  # scale
        else:
            raise ValueError(
                f"Invalid initialization method: {self.init_method}. ",
                f"Choose from {list(TOPK_INIT_METHODS.keys())}",
            )

    def decay_weights(self, weight_decay=0.1, weight_boost=False):
        with torch.no_grad():
            weight_mask = self.weight_mask()
            log_decay = log(1 - weight_decay) * weight_mask
            self.pre_w.data += log_decay # suppress active synapses
            if weight_boost:
                log_boost = log(1 + weight_decay) * (1 - weight_mask)
                self.pre_w.data -= log_boost # boost inactive synapses

    def forward(self, x):
        pruned_weight = self.pruned_weight()

        # Ensure x is a 2D tensor for matrix multiplication
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")

        # If x is a 1D tensor, unsqueeze to make it 2D [1, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # If x is not 2D at this point, we can't do matrix multiplication
        if x.dim() != 2:
            raise ValueError(
                f"Expected input to be a 2D tensor, got shape {x.shape} with {x.dim()} dimensions"
            )

        return torch.matmul(x, pruned_weight.t())

    def weight(self):
        if self.param_space == "log":
            return self.pre_w.exp()
        elif self.param_space == "presigmoid":
            return torch.sigmoid(self.pre_w)

    def log_weight(self):
        return self.weight().log()

    def weight_mask(self):
        # Determine which weights to use for topk selection
        weights_for_topk = self.pre_w
        if self.noise_level > 0:
            # Generate Gaussian noise with mean=0 and std=noise_level
            noise = torch.randn_like(self.pre_w) * self.noise_level
            # Add noise to the raw weights
            weights_for_topk = self.pre_w + noise

        # Perform topk on the (potentially noisy) weights
        topK_indices = torch.topk(
            weights_for_topk, self.K, dim=-1, largest=True, sorted=False
        )[1]

        # Create the mask based on the selected indices
        mask = torch.zeros_like(
            self.pre_w,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        mask[torch.arange(self.pre_w.shape[0])[:, None], topK_indices] = 1
        return mask

    def pruned_weight(self):
        return self.weight_mask() * self.weight()

    def log_pruned_weight(self):
        return ((self.weight_mask() - 1) * 10) + self.log_weight()

    def weighted_synapses(self, cell_weights, prune=False):
        if prune:
            synapse_weights = self.pruned_weight()
        else:
            synapse_weights = self.weight()

        weighted_synapses = cell_weights[:, None] * synapse_weights
        return weighted_synapses.sum(dim=0)


# ------------- New TopK Variant Classes -------------
class StochasticTopKLinear(TopKLinear):
    """
    A variant of TopKLinear where we select top-K stochastically based on rank:
    - We rank by pre_w (or pre_w + noise if noise_level>0).
    - Convert rank => probability of being "kept."
    - Sample a Bernoulli, so smaller weights still have a small chance to stay on.
    """

    def __init__(
        self,
        in_features,
        out_features,
        K,
        param_space="log",
        init_method="xavier_normal",
        init_gain=1.0,
        noise_level=0.0,
        temperature=0.5,  # extra hyperparam for sharper or flatter gating
        ultrafast=False,  # Set to True for much faster but approximate implementation
    ):
        super().__init__(
            in_features,
            out_features,
            K,
            param_space=param_space,
            init_method=init_method,
            init_gain=init_gain,
            noise_level=noise_level,
        )
        self.temperature = temperature
        self.ultrafast = ultrafast

    def weight_mask_ultrafast(self):
        """
        Ultra-fast approximation of stochastic TopK that:
        1. Adds noise directly to the weights before standard TopK selection
        2. Uses more noise for lower-ranked weights to approximate the desired stochasticity
        3. Performs a single standard TopK operation with no correction steps

        10-50x faster than the full stochastic method with similar behavior.
        """
        # Ensure K is valid and an integer
        k_value = max(1, int(self.K))

        # Apply scaled and temperature-controlled noise
        w_for_topk = self.pre_w.clone()

        # Generate noise with standard deviation that scales with temperature
        noise_scale = max(0.01, self.temperature) * (1.0 + self.noise_level)
        noise = torch.randn_like(w_for_topk) * noise_scale

        # Simply add noise to weights - simpler but still effective approach
        noisy_weights = w_for_topk + noise

        # Perform standard TopK selection on the noisy weights
        topK_indices = torch.topk(
            noisy_weights, k_value, dim=-1, largest=True, sorted=False
        )[1]

        # Create mask using the standard TopK approach
        mask = torch.zeros_like(
            self.pre_w,
            device=self.pre_w.device,
            dtype=self.pre_w.dtype,
        )
        batch_indices = (
            torch.arange(self.pre_w.shape[0], device=self.pre_w.device)
            .view(-1, 1)
            .expand(-1, k_value)
        )
        mask[batch_indices, topK_indices] = 1

        return mask

    def weight_mask(self):
        """
        Instead of a hard top-k, we rank by magnitude and keep each weight with a probability
        ~ sigmoid((k - rank)/ (k*self.temperature)), so top ranks have high prob ~1.

        This is a vectorized implementation that processes all neurons at once for better performance.
        """
        # Use ultrafast implementation if requested
        if self.ultrafast:
            return self.weight_mask_ultrafast()

        # Apply noise if requested
        w_for_topk = self.pre_w
        if self.noise_level > 0:
            noise = torch.randn_like(self.pre_w) * self.noise_level
            w_for_topk = self.pre_w + noise

        # Ensure K is an integer and at least 1
        k_value = max(1, int(self.K))

        # Get dimensions
        out_features, in_features = w_for_topk.shape

        # Calculate absolute values for ranking
        abs_weights = w_for_topk.abs()

        # For each neuron, sort the input weights and get the ranks
        # - First, get the indices that would sort each row in descending order
        _, sort_indices = torch.sort(abs_weights, dim=1, descending=True)

        # - Create a rank tensor with the same shape as weights
        # Ensure the dtype is the same as what we'll assign to it (Long/Int64)
        arange_indices = torch.arange(in_features, device=w_for_topk.device).expand(
            out_features, -1
        )
        ranks = torch.zeros_like(w_for_topk, dtype=arange_indices.dtype)

        # - For each neuron (row), assign ranks to the positions
        batch_indices = (
            torch.arange(out_features, device=w_for_topk.device)
            .view(-1, 1)
            .expand(-1, in_features)
        )
        ranks[batch_indices, sort_indices] = arange_indices

        # Convert ranks to float for sigmoid calculation
        ranks = ranks.to(torch.float)

        # Calculate selection probabilities using sigmoid
        temperature = max(0.01, self.temperature)  # Prevent division by zero
        p = torch.sigmoid((k_value - ranks) / (k_value * temperature))

        # Sample Bernoulli for all neurons at once
        mask = torch.bernoulli(p)

        # Fast connection count correction to ensure each neuron has reasonable connections
        # 1. Count connections per neuron
        connections_per_neuron = mask.sum(dim=1)

        # 2. For neurons with too few connections (less than K/2), add more
        too_few_mask = connections_per_neuron < (k_value // 2)
        if too_few_mask.any():
            # Safe handling of nonzero indices to avoid 0-d tensor issues
            too_few_indices = too_few_mask.nonzero(as_tuple=False).view(-1)
            for i in too_few_indices:
                # Get the indices of the top-K weights for this neuron
                _, top_indices = torch.topk(abs_weights[i], k_value, largest=True)
                mask[i, top_indices] = 1

        # 3. For neurons with too many connections (more than 2*K), keep only the top 2*K
        too_many_mask = connections_per_neuron > (2 * k_value)
        if too_many_mask.any():
            # Safe handling of nonzero indices to avoid 0-d tensor issues
            too_many_indices = too_many_mask.nonzero(as_tuple=False).view(-1)
            for i in too_many_indices:
                # Find the weakest connections for this neuron
                # First, get indices of all connections that are currently active
                active_indices = mask[i].nonzero(as_tuple=True)[0]
                # Get their weights
                active_weights = abs_weights[i, active_indices]
                # Sort from weakest to strongest
                _, sorted_indices = torch.sort(active_weights)
                # Calculate how many to remove
                to_remove = int(connections_per_neuron[i].item() - k_value)
                # Set the weakest ones to 0
                if to_remove > 0:
                    indices_to_remove = active_indices[sorted_indices[:to_remove]]
                    mask[i, indices_to_remove] = 0

        # 4. Handle neurons with zero connections
        zero_connections = connections_per_neuron == 0
        if zero_connections.any():
            # Safe handling of nonzero indices to avoid 0-d tensor issues
            zero_indices = zero_connections.nonzero(as_tuple=False).view(-1)
            for i in zero_indices:
                # Get the index of the strongest weight for this neuron
                strongest_idx = torch.argmax(abs_weights[i])
                mask[i, strongest_idx] = 1

        return mask


class VarianceTopKLinear(TopKLinear):
    """
    A variance-based TopK approach that tracks input activation statistics over batches
    and selects weights based on the variance of activations at each input.

    This implementation:
    1. Uses exponential moving averages to track input statistics
    2. Computes variance estimates for each input dimension
    3. Selects connections based on input variance * weight magnitude
    """

    def __init__(
        self,
        in_features,
        out_features,
        K,
        param_space="log",
        init_method="xavier_normal",
        init_gain=1.0,
        noise_level=0.0,
        momentum=0.9,  # EMA momentum for tracking statistics
    ):
        super().__init__(
            in_features,
            out_features,
            K,
            param_space=param_space,
            init_method=init_method,
            init_gain=init_gain,
            noise_level=noise_level,
        )
        self.momentum = momentum

        # Initialize tracking buffers for activation statistics (not parameters)
        self.register_buffer("activation_mean", torch.zeros(in_features))
        self.register_buffer("activation_sq_mean", torch.zeros(in_features))
        self.register_buffer("activation_var", torch.ones(in_features))
        self.register_buffer("num_updates", torch.zeros(1))

        # Flag to track if we're in training mode
        self.stats_tracking = True

    def update_stats(self, x):
        """Update activation statistics with current batch."""
        if not self.stats_tracking:
            return

        # If this is a batched input, average over the batch
        if x.dim() > 1:
            batch_means = x.mean(dim=0)
            batch_sq_means = (x**2).mean(dim=0)
        else:
            batch_means = x
            batch_sq_means = x**2

        # Increment counter
        self.num_updates += 1

        # Apply exponential moving average updates
        if self.num_updates == 1:
            # First update, just use the batch stats
            self.activation_mean.copy_(batch_means)
            self.activation_sq_mean.copy_(batch_sq_means)
        else:
            # EMA update
            self.activation_mean.mul_(self.momentum).add_(
                batch_means * (1 - self.momentum)
            )
            self.activation_sq_mean.mul_(self.momentum).add_(
                batch_sq_means * (1 - self.momentum)
            )

        # Compute variance: E[X²] - E[X]²
        self.activation_var = self.activation_sq_mean - self.activation_mean**2

        # Ensure variance is positive (numeric stability)
        self.activation_var.clamp_(min=1e-6)

    def forward(self, x):
        # Update statistics during forward pass
        self.update_stats(x)

        # Use parent class forward implementation to avoid code duplication
        return super().forward(x)

    def weight_mask(self):
        """
        Select TopK connections based on a combination of:
        - Input activation variance
        - Weight magnitude

        The intuition is that we want to keep connections to inputs that:
        1. Have high variance (carry more information)
        2. Have strong weights (model cares about them)
        """
        # Determine weights for topk selection
        w_for_topk = self.pre_w
        if self.noise_level > 0:
            noise = torch.randn_like(self.pre_w) * self.noise_level
            w_for_topk = self.pre_w + noise

        # Calculate importance scores using weights and input variance
        # Each neuron sees the same activation variance, but has different weights
        # Expand variance to match weight shape [out_features, in_features]
        var_expanded = self.activation_var.expand_as(w_for_topk)

        # Score is weight magnitude * input variance
        if self.param_space == "log":
            # For log param space, we use exp(pre_w) * variance
            importance_scores = w_for_topk.exp() * var_expanded
        else:
            # For presigmoid, use sigmoid(pre_w) * variance
            importance_scores = torch.sigmoid(w_for_topk) * var_expanded

        # Perform topk selection per neuron
        topk_indices = torch.topk(
            importance_scores, self.K, dim=-1, largest=True, sorted=False
        )[1]

        # Create mask based on selected indices
        mask = torch.zeros_like(self.pre_w)
        mask[torch.arange(self.pre_w.shape[0])[:, None], topk_indices] = 1

        return mask

    def train(self, mode=True):
        """Override train method to control stats tracking."""
        super().train(mode)
        self.stats_tracking = mode
        return self

    def eval(self):
        """Override eval method to freeze stats in eval mode."""
        super().eval()
        self.stats_tracking = False
        return self


class DeepstLinear(nn.Module):
    def __init__(
            self, 
            in_features, 
            out_features, 
            target_density=0.1,
            sigma=0.05,  # Noise amplitude for the random walk
            use_noise=True,  # Whether to apply Gaussian noise
            rewiring_mode='global',  # 'local', 'global', or 'constant_branch'
            synapses_per_branch=None,  # Required for 'constant_branch' mode
            param_space='log',  # 'log' or 'presigmoid' for weight parameterization
            weight_init_range=None,
            init_method=None,  # Add initialization method parameter
            freeze_connectivity=False,  # New: flag to freeze synapse locations
            weight_threshold=1e-6  # New: threshold for weight pruning
        ):    
        super(DeepstLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.target_density = target_density
        self.sigma = sigma
        self.use_noise = use_noise
        self.rewiring_mode = rewiring_mode
        self.param_space = param_space
        self.init_method = init_method

        # Add K attribute for compatibility with initialize_dendritic_branch_layer
        self.K = int(target_density * in_features) if in_features else None

        # Default initialization ranges
        if weight_init_range is None:
            if param_space == 'log':
                weight_init_range = (-2.1, -2.0)  # Default for log parameterization
            elif param_space == 'presigmoid':
                weight_init_range = (-0.1, 0.1)   # Default for presigmoid
            elif param_space == 'uniform':
                weight_init_range = (0.25, 0.75)   # Default for uniform
        # Store weight initialization range
        self.weight_init_range = weight_init_range
        
        # For constant_branch mode
        if rewiring_mode == 'constant_branch':
            if synapses_per_branch is None:
                raise ValueError("synapses_per_branch must be specified for constant_branch mode")
            if synapses_per_branch > in_features:
                raise ValueError(f"synapses_per_branch ({synapses_per_branch}) cannot exceed in_features ({in_features})")
            self.synapses_per_branch = synapses_per_branch
        
        # Total number of possible connections
        self.total_connections = out_features * in_features
        
        # Target number of active connections based on density
        if rewiring_mode == 'constant_branch':
            self.target_connections = out_features * synapses_per_branch
        else:
            self.target_connections = int(self.total_connections * target_density)
        
        # Initialize weights parameter
        if self.init_method is not None and self.init_method in TOPK_INIT_METHODS:
            # Initialize using PyTorch's initialization methods
            self.pre_w = nn.Parameter(torch.zeros((out_features, in_features)), requires_grad=True)
            # Apply initialization to a temporary tensor (correct shape for initialization)
            temp_weight = torch.zeros((out_features, in_features))
            TOPK_INIT_METHODS[self.init_method](temp_weight)
            # Reshape to our desired shape
            self.pre_w.data = temp_weight.reshape(out_features, in_features)
            
            # If using log or presigmoid parameterization, transform the weights accordingly
            if param_space == 'log':
                # For log parameterization, ensure weights are positive then take log
                self.pre_w.data = torch.abs(self.pre_w.data).clamp(min=1e-6)
                self.pre_w.data = torch.log(self.pre_w.data)
            elif param_space == 'presigmoid':
                # For presigmoid, pass through inverse sigmoid (logit)
                self.pre_w.data = torch.clamp(self.pre_w.data, min=1e-6, max=1-1e-6)
                self.pre_w.data = torch.log(self.pre_w.data / (1 - self.pre_w.data))
            elif param_space == 'uniform':
                # For uniform, pass through uniform initialization
                self.pre_w.data = torch.empty(out_features, in_features).uniform_(
                    self.weight_init_range[0], self.weight_init_range[1])
        else:
            # Use uniform initialization with specified range
            self.pre_w = nn.Parameter(torch.zeros((out_features, in_features)).uniform_(
                self.weight_init_range[0], self.weight_init_range[1]), requires_grad=True)
        
        # Initialize mask
        mask = torch.zeros_like(self.pre_w, dtype=torch.bool)
        # Register it as a buffer to ensure it moves with the model
        self.mask: torch.Tensor
        self.register_buffer('mask', mask)
    
        # Initialize connectivity
        self._initialize_connectivity()
        
        # Store freeze flag
        self.freeze_connectivity = freeze_connectivity
        
        # Store weight threshold for pruning
        self.weight_threshold = weight_threshold
    
    def _initialize_connectivity(self):
        if self.rewiring_mode == 'constant_branch':
            # Initialize with constant number of connections per branch
            for neuron_idx in range(self.out_features):
                # Randomly select synapses_per_branch input connections
                input_indices = torch.randperm(self.in_features)[:self.synapses_per_branch]
                for input_idx in input_indices:
                    self.mask[neuron_idx, input_idx] = True
        else:
            # Standard random initialization for global or local modes
            connections_to_activate = self.target_connections
            
            # Flatten indices for uniform sampling
            all_indices = torch.nonzero(torch.ones_like(self.pre_w), as_tuple=False)
            
            # Randomly select indices to activate
            selected_indices = torch.randperm(len(all_indices))[:connections_to_activate]
            selected_connections = all_indices[selected_indices]
            
            # Activate connections
            with torch.no_grad():
                for i, j in selected_connections:
                    self.mask[i, j] = True
    
    def weight(self):
        """Return the actual weight values (after transformation)"""
        if self.param_space == 'log':
            return torch.exp(self.pre_w)
        elif self.param_space == 'presigmoid':
            return torch.sigmoid(self.pre_w)
        elif self.param_space == 'uniform':
            return self.pre_w
    
    def weight_mask(self):
        """Return the current mask"""
        return self.mask
    
    def pruned_weight(self):
        """Return masked weights"""
        return self.weight() * self.mask
    
    def forward(self, x):
        # Apply mask and sum across dendritic locations
        return torch.matmul(x, self.pruned_weight().t())
    
    def weighted_synapses(self, cell_weights, prune=False):
        """Apply cell weights to synapses (compatibility with DendriNet)"""
        if prune:
            synapse_weights = self.pruned_weight()
        else:
            synapse_weights = self.weight()
        
        weighted_synapses = cell_weights[:, None, None] * synapse_weights
        return weighted_synapses.sum(dim=0)
    
    def apply_noise_and_enforce_constraints(self):
        """
        Apply Brownian motion-like noise and enforce connectivity constraints.
        Modified to support freezing connectivity while maintaining stability.
        """
        with torch.no_grad():
            # Get current weights
            current_weights = self.weight()
            
            # Always apply basic constraints for numerical stability
            if self.param_space == 'log':
                # Use a tighter clamping range for better stability
                self.pre_w.data = torch.clamp(self.pre_w.data, min=-10.0, max=10.0)
            elif self.param_space == 'presigmoid':
                self.pre_w.data = torch.clamp(self.pre_w.data, min=-5.0, max=5.0)
            elif self.param_space == 'uniform':
                pass
            
            # Skip rewiring if connectivity is frozen
            if self.freeze_connectivity:
                return
            
            # Continue with normal rewiring process for non-frozen connectivity
            
            # Apply Gaussian noise if enabled
            if self.use_noise:
                noise = torch.zeros_like(current_weights)
                noise[self.mask] = torch.randn(self.mask.sum(), device=self.mask.device) * self.sigma
                current_weights = current_weights + noise
            
            # Find connections to prune (weights <= weight_threshold)
            to_prune = (current_weights <= self.weight_threshold) & self.mask
            
            # Count pruned connections
            num_pruned = to_prune.sum().item()
            
            # Update mask (remove pruned connections)
            self.mask[to_prune] = False
            
            # Create new random connections
            if num_pruned > 0:
                self._create_new_connections(to_prune)
    
    def _create_new_connections(self, pruned_mask):
        """
        Create new random connections based on the rewiring mode
        """
        if self.rewiring_mode == 'constant_branch':
            # For each branch, maintain a constant number of synapses
            for neuron_idx in range(self.out_features):
                # Current active synapses at this branch
                current_active = self.mask[neuron_idx].sum().item()
                
                # How many synapses need to be added
                to_add = self.synapses_per_branch - current_active
                
                if to_add > 0:
                    # Get potential input neurons for new connections
                    potential_inputs = ~self.mask[neuron_idx]
                    potential_indices = torch.nonzero(potential_inputs, as_tuple=False).squeeze(1)
                    
                    if len(potential_indices) > 0:
                        # Randomly select input neurons to connect
                        num_to_select = min(to_add, len(potential_indices))
                        selected_indices = potential_indices[torch.randperm(len(potential_indices))[:num_to_select]]
                        
                        # Create new connections
                        for input_idx in selected_indices:
                            self.mask[neuron_idx, input_idx] = True
                            self.pre_w.data[neuron_idx, input_idx] = self._initialize_new_weight()
        
        elif self.rewiring_mode == 'local':
            # Local rewiring - only create connections within the same neuron
            for neuron_idx in range(self.out_features):
                # Count pruned connections for this neuron
                pruned_for_neuron = pruned_mask[neuron_idx].sum().item()
                
                if pruned_for_neuron > 0:
                    # Get potential new connections for this neuron
                    potential_connections = ~self.mask[neuron_idx]
                    
                    if potential_connections.sum() > 0:
                        # Get indices of potential connections
                        flat_indices = torch.nonzero(potential_connections, as_tuple=False)
                        
                        # Randomly select indices to add
                        num_to_add = min(pruned_for_neuron, len(flat_indices))
                        if num_to_add > 0:
                            selected_indices = torch.randperm(len(flat_indices))[:num_to_add]
                            selected_connections = flat_indices[selected_indices]
                            
                            # Add new connections
                            for in_idx in selected_connections:
                                self.mask[neuron_idx, in_idx] = True
                                self.pre_w.data[neuron_idx, in_idx] = self._initialize_new_weight()
        
        else:  # 'global' rewiring mode
            # Count total pruned connections
            num_pruned = pruned_mask.sum().item()
            
            # Get all potential connections globally
            potential_connections = ~self.mask
            
            if potential_connections.sum() > 0 and num_pruned > 0:
                flat_indices = torch.nonzero(potential_connections, as_tuple=False)
                
                # Randomly select indices to add
                num_to_add = min(num_pruned, len(flat_indices))
                if num_to_add > 0:
                    selected_indices = torch.randperm(len(flat_indices))[:num_to_add]
                    selected_connections = flat_indices[selected_indices]
                    
                    # Add new connections
                    for i, j in selected_connections:
                        self.mask[i, j] = True
                        self.pre_w.data[i, j] = self._initialize_new_weight()
    
    def _initialize_new_weight(self):
        """Helper method to initialize a new weight according to the chosen method"""
        if self.init_method is not None and self.init_method in TOPK_INIT_METHODS:
            # Create a tensor with at least 2 dimensions for initialization
            temp = torch.zeros(2, 2, device=self.pre_w.device)  # Changed from 1D to 2D
            TOPK_INIT_METHODS[self.init_method](temp)
            
            # Just use one value from the initialized tensor
            value = temp[0, 0].item()
            
            # Apply transformations for log/presigmoid if needed
            if self.param_space == 'log':
            #     value = abs(value)
            #     value = max(value, 1e-6)
            #     value = math.log(value)

                ### EXPERIMENTAL
                value = math.log(self.weight_threshold)
            elif self.param_space == 'presigmoid':
                value = max(min(value, 1-1e-6), 1e-6)
                value = math.log(value / (1 - value))
            elif self.param_space == 'uniform':
                pass
            return value
        else:
            # Default to uniform initialization with the specified range
            return torch.empty(1, device=self.pre_w.device).uniform_(
                self.weight_init_range[0], self.weight_init_range[1]).item()
    
    def get_connectivity_stats(self):
        """
        Return statistics about the network connectivity
        """
        active_connections = self.mask.sum().item()
        density = active_connections / self.total_connections
        
        # Per-branch stats for constant_branch mode
        branch_stats = {}
        if self.rewiring_mode == 'constant_branch':
            branch_counts = torch.sum(self.mask, dim=1)  # Sum over input dimension
            branch_stats = {
                'min_per_branch': branch_counts.min().item(),
                'max_per_branch': branch_counts.max().item(),
                'mean_per_branch': branch_counts.float().mean().item(),
                'target_per_branch': self.synapses_per_branch
            }
            
        return {
            'active_connections': active_connections,
            'total_connections': self.total_connections,
            'density': density,
            'target_density': self.target_density,
            'branch_stats': branch_stats
        }

    def decay_weights(self, weight_decay=0.1):
        """
        Apply weight decay to active synaptic connections.
        
        Args:
            weight_decay: float, weight decay rate (default: 0.1)
        """
        pass


class BlockLinear(nn.Module):
    """
    A custom linear layer that aggregates information from converging branches
    in a previous branch layer.

    Parameters
    ----------

    in_features : int
        The number of input features.

    out_features : int
        The number of output features.

    requires_grad : bool, optional
        If `True`, the weights are trainable. Defaults to `False`.

    Attributes
    ----------
    weight : torch.nn.Parameter
        The weights for the linear transformation.

    in_features : int
        The number of input features.

    out_features : int
        The number of output features.

    block_size : int
        The number of branches converging onto a single branch.

    Methods
    -------
    forward(x)
        Performs a forward pass through the layer.

    sum_conductances()
        Returns the sum of conductances for each output neuron.

    Notes
    -----

    - Block_size equals number of branches converging onto single branch/soma
      in next layer.

    """

    def __init__(self, in_features, out_features):
        super(BlockLinear, self).__init__()

        block_size = floor(in_features / out_features)
        assert (
            in_features == out_features * block_size
        ), "in_features must be divisible by out_features."

        self.log_weight = nn.Parameter(
            torch.empty((out_features, block_size)), requires_grad=True
        )
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.input_scale_vec = None  # may be used for gradient scaling
        self.param_scale_vec = None  # may be used for gradient scaling

    def initialize(self, sum_synapse_conductances=None):
        if sum_synapse_conductances is None:
            nn.init.constant_(self.log_weight, 0)
        else:
            # Ensure sum_synapse_conductances is a 1D tensor
            self.log_weight.data = (
                sum_synapse_conductances.log()[:, None]
                .expand(self.out_features, self.block_size)
                .clone()
            )

    def block(self):
        block = torch.zeros(
            (self.out_features, self.in_features),
            device=self.log_weight.device,
            dtype=self.log_weight.dtype,
        )
        row_ix = torch.arange(self.out_features, device=self.log_weight.device)[:, None]
        col_ix = torch.arange(self.in_features, device=self.log_weight.device).view(
            self.out_features, self.block_size
        )
        block[row_ix, col_ix] = self.log_weight.exp()
        return block

    def log_block(self):
        log_block = (
            torch.ones(
                (self.out_features, self.in_features),
                device=self.log_weight.device,
                dtype=self.log_weight.dtype,
            )
            * -10
        )
        row_ix = torch.arange(self.out_features, device=self.log_weight.device)[:, None]
        col_ix = torch.arange(self.in_features, device=self.log_weight.device).view(
            self.out_features, self.block_size
        )
        log_block[row_ix, col_ix] = self.log_weight
        return log_block

    def grad_block(self):
        grad_block = torch.zeros(
            (self.out_features, self.in_features),
            device=self.log_weight.device,
            dtype=self.log_weight.dtype,
        )
        row_ix = torch.arange(self.out_features, device=self.log_weight.device)[:, None]
        col_ix = torch.arange(self.in_features, device=self.log_weight.device).view(
            self.out_features, self.block_size
        )
        if self.log_weight.grad is not None:
            grad_block[row_ix, col_ix] = self.log_weight.grad
        return grad_block

    def decay_weights(self, weight_decay):
        with torch.no_grad():
            self.log_weight.data -= weight_decay

    def forward(self, x):
        # Ensure x is a 2D tensor for matrix multiplication
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected input to be a tensor, got {type(x)}")

        # If x is a 1D tensor, unsqueeze to make it 2D [1, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # If x is not 2D at this point, we can't do matrix multiplication
        if x.dim() != 2:
            raise ValueError(
                f"Expected input to be a 2D tensor, got shape {x.shape} with {x.dim()} dimensions"
            )

        return torch.matmul(x, self.block().t())

    def sum_conductances(self):
        return self.log_weight.exp().sum(dim=1)


class DendriticBranchLayer(nn.Module):
    """
    Emulates dendritic computation along a layer of branches (equally distant
    from soma).

    This layer models the interaction between excitatory and inhibitory inputs,
    simulating dendritic processing in biological neurons. It aggregates
    information from converging branches in the previous layer and optionally
    incorporates inhibitory modulation.

    Parameters
    ----------
    output_dim : int
        The number of output features.

    excitatory_input_dim : int
        The number of input features from excitatory cells.

    excitatory_synapses_per_branch : int
        The number of excitatory synapses per branch.

    inhibitory_input_dim : int, optional
        The number of input features from inhibitory cells. Defaults to `None`.

    inhibitory_synapses_per_branch : int, optional
        The number of inhibitory synapses per branch. Defaults to `None`.

    input_branch_factor : int, optional
        The number of branches converging onto a single branch. Defaults to
        `None`.
    """

    def __init__(
        self,
        output_dim,
        excitatory_input_dim=None,
        excitatory_synapses_per_branch=None,
        inhibitory_input_dim=None,
        inhibitory_synapses_per_branch=None,
        input_branch_factor=None,
        topk_init_method="xavier_normal",
        topk_noise_level=0.0,
        topk_type="standard",  # New parameter
        topk_temperature=0.5,  # For stochastic TopK
        topk_ultrafast=False,  # For much faster stochastic TopK
        topk_strategy="none",
        use_shunting=True,
        reactivate=False,
        reactivation_type="param_tanh",
        reactivation_init_m=1,
        reactivation_init_b=0.5,
        reactivation_strategy="none",
        blocklinear_strategy="none",
        layer_idx=0,
        print_hooks=False,
        # DEEPST specific parameters, clean up later
        use_noise=True,
        rewiring_mode='global',
        synapses_per_branch=None,
        sigma=0.05,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method=None,
        weight_threshold=1e-6,
        **kwargs,
    ):
        super(DendriticBranchLayer, self).__init__()
        self.layer_idx = layer_idx
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.use_shunting = use_shunting
        self.topk_type = topk_type  # Store the strategy
        self.topk_temperature = topk_temperature  # Store the temperature
        self.topk_ultrafast = topk_ultrafast  # Store the ultrafast flag
        self.topk_strategy = topk_strategy  # Store the strategy
        self.input_excitatory = excitatory_input_dim is not None
        self.print_hooks = print_hooks

        # DEEPST specific parameters, clean up later
        self.use_noise = use_noise
        self.rewiring_mode = rewiring_mode
        self.synapses_per_branch = synapses_per_branch
        self.sigma = sigma
        self.excitatory_target_density = excitatory_target_density
        self.inhibitory_target_density = inhibitory_target_density
        self.freeze_excitatory_connectivity = freeze_excitatory_connectivity
        self.freeze_inhibitory_connectivity = freeze_inhibitory_connectivity
        self.init_method = init_method
        self.weight_threshold = weight_threshold
        
        gradient_scaler = GradientScaler(
            topk_strategy=topk_strategy,
            reactivation_strategy=reactivation_strategy,
            blocklinear_strategy=blocklinear_strategy,
            layer_idx=layer_idx,
            print_hooks=print_hooks,
        )

        if self.input_excitatory:
            self.branch_excitation = self._create_topk_layer(
                in_features=excitatory_input_dim,
                out_features=output_dim,
                K=excitatory_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                cell_type='exc',
            )
            gradient_scaler.register_topk_dynamic(self.branch_excitation)
            self.excitatory_input_dim = excitatory_input_dim
        else:
            self.branch_excitation = None

        self.input_inhibitory = inhibitory_input_dim is not None
        if self.input_inhibitory:
            self.branch_inhibition = self._create_topk_layer(
                in_features=inhibitory_input_dim,
                out_features=output_dim,
                K=inhibitory_synapses_per_branch,
                init_method=topk_init_method,
                noise_level=topk_noise_level,
                cell_type='inh',
            )
            gradient_scaler.register_topk_dynamic(self.branch_inhibition)
            self.inhibitory_input_dim = inhibitory_input_dim
        else:
            self.branch_inhibition = None

        self.input_branches = input_branch_factor is not None
        if self.input_branches:
            self.branches_to_output = BlockLinear(
                output_dim * input_branch_factor, output_dim
            )
            gradient_scaler.register_block_linear_dynamic(self.branches_to_output)
        self.input_branch_factor = input_branch_factor

        if reactivate:
            self.reactivation = ActivationFactory.create(
                act_type=reactivation_type,
                output_dim=output_dim,
                init_m=reactivation_init_m if use_shunting else 0.1,
                init_b=reactivation_init_b if use_shunting else 0.1,
                **kwargs
            )
            if use_shunting:
                gradient_scaler.register_reactivation_inverse(self.reactivation)
        else:
            self.reactivation = nn.Identity()
        self.reactivate = reactivate

        # if topk_type == 'deepst':
        #     initialize_dendritic_branch_layer(self)
        # else:
        #     self.initialize()
        initialize_dendritic_branch_layer(self)

    def _create_topk_layer(
        self,
        in_features,
        out_features,
        K,
        init_method,
        noise_level,
        cell_type='exc',
    ):
        """Helper function that picks which topk class to use based on strategy."""
        # For 'standard' strategy, use the original TopKLinear class exactly as before
        # This provides a direct path with zero overhead compared to the original code
        if self.topk_type is None or self.topk_type == "standard":
            # Create directly without any strategy-specific logic
            topk_layer = TopKLinear(
                in_features=in_features,
                out_features=out_features,
                K=K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
            )
            return topk_layer

        # Alternative strategies only incur overhead if explicitly requested
        elif self.topk_type == "stochastic":
            # Get temperature parameter if available
            temperature = getattr(self, "topk_temperature", 0.5)
            # Check if ultrafast mode is enabled
            ultrafast = getattr(self, "topk_ultrafast", False)

            return StochasticTopKLinear(
                in_features=in_features,
                out_features=out_features,
                K=K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
                temperature=temperature,
                ultrafast=ultrafast,
            )
        elif self.topk_type == "variance":
            return VarianceTopKLinear(
                in_features=in_features,
                out_features=out_features,
                K=K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
            )
        elif self.topk_type == "deepst":
            return DeepstLinear(
                in_features=in_features,
                out_features=out_features,
                target_density=self.excitatory_target_density if cell_type == 'exc' else self.inhibitory_target_density,
                sigma=self.sigma,
                use_noise=self.use_noise,
                rewiring_mode=self.rewiring_mode,
                synapses_per_branch=self.synapses_per_branch,
                param_space='log',
                init_method=self.init_method,
                freeze_connectivity=self.freeze_excitatory_connectivity if cell_type == 'exc' else self.freeze_inhibitory_connectivity,
                weight_threshold=self.weight_threshold,
            )
        else:
            # Fallback for any unknown strategy - use standard
            return TopKLinear(
                in_features=in_features,
                out_features=out_features,
                K=K,
                param_space="log",
                init_method=init_method,
                noise_level=noise_level,
            )

    def initialize(self):
        with torch.no_grad():
            if self.input_excitatory and self.branch_excitation is not None:
                self.branch_excitation.initialize()
            if self.input_inhibitory and self.branch_inhibition is not None:
                self.branch_inhibition.initialize()

            if self.input_branches:
                if (
                    self.input_excitatory or self.input_inhibitory
                ) and self.use_shunting:
                    sum_synapse_conductances = 0
                    if self.input_excitatory and self.branch_excitation is not None:
                        sum_synapse_conductances += self.branch_excitation.weight().sum(
                            dim=1
                        )
                    if self.input_inhibitory and self.branch_inhibition is not None:
                        sum_synapse_conductances += self.branch_inhibition.weight().sum(
                            dim=1
                        )
                else:
                    sum_synapse_conductances = None
                self.branches_to_output.initialize(sum_synapse_conductances)

    def decay_weights(self, weight_decay):
        if self.input_excitatory and self.branch_excitation is not None:
            self.branch_excitation.decay_weights(weight_decay, weight_boost=False)
        if self.input_inhibitory and self.branch_inhibition is not None:
            self.branch_inhibition.decay_weights(weight_decay, weight_boost=False)
        # if self.input_branches:
        #     self.branches_to_output.decay_weights(weight_decay)

    # DEEPST specific method
    def apply_rewiring(self):
        """
        Apply rewiring to all synaptic connections.
        """
        if hasattr(self.branch_excitation, 'apply_noise_and_enforce_constraints'):
            self.branch_excitation.apply_noise_and_enforce_constraints()
        if hasattr(self.branch_inhibition, 'apply_noise_and_enforce_constraints'):
            self.branch_inhibition.apply_noise_and_enforce_constraints()

    def forward(self, x, inhibitory_input=None, branch_input=None):
        if self.use_shunting:
            numerator = 0
            denominator = 1

            if self.input_excitatory and self.branch_excitation is not None:
                excitation = self.branch_excitation(x)
                numerator = numerator + excitation
                denominator = denominator + excitation

            if self.input_branches and branch_input is not None:
                current = self.branches_to_output(branch_input)
                numerator = numerator + current
                conductance = self.branches_to_output.sum_conductances()
                denominator = denominator + conductance

            if (
                self.input_inhibitory
                and self.branch_inhibition is not None
                and inhibitory_input is not None
            ):
                denominator = denominator + self.branch_inhibition(inhibitory_input)

            voltage = numerator / denominator

        else:
            voltage = 0

            if self.input_excitatory and self.branch_excitation is not None:
                voltage = voltage + self.branch_excitation(x)

            if self.input_branches and branch_input is not None:
                voltage = voltage + self.branches_to_output(branch_input)

            if (
                self.input_inhibitory
                and self.branch_inhibition is not None
                and inhibitory_input is not None
            ):
                voltage = voltage - self.branch_inhibition(inhibitory_input)

        if self.training and "conductance_dynamic" in [
            self.topk_strategy, self.blocklinear_strategy]:
            self.compute_grad_scales(
                g_total = denominator if self.use_shunting else None)

        return self.reactivation(voltage)

    def compute_grad_scales(self, g_total = None):
        """
        A conduction-based dynamic scale, returning a flattened vector so each weight
        can get a distinct multiplier. shape [out_features * block_size].
        """
        with torch.no_grad():
            if self.use_shunting:
                if g_total.dim() > 1:
                    g_total = g_total.mean(dim = tuple(range(g_total.dim() - 1)))

                if (
                    hasattr(self, "branches_to_output")
                    and self.blocklinear_strategy == "conductance_dynamic"
                ):
                    self.branches_to_output.input_scale_vec = (
                        g_total[:, None] / self.branches_to_output.log_weight.exp()
                    ).flatten()
                    self.branches_to_output.param_scale_vec = g_total[:, None]
                
                if (self.branch_excitation is not None 
                    and self.topk_strategy == "conductance_dynamic"):
                    self.branch_excitation.input_scale_vec = (
                        g_total[:, None] / self.branch_excitation.weight()
                    ).mean(dim = 0)
                    self.branch_excitation.param_scale_vec = g_total[:, None]

                if (
                    self.branch_inhibition is not None
                    and self.topk_strategy == "conductance_dynamic"
                ):
                    self.branch_inhibition.input_scale_vec = (
                        g_total[:, None] / self.branch_inhibition.weight()
                    ).mean(dim = 0)
                    self.branch_inhibition.param_scale_vec = g_total[:, None]

            else:
                if (
                    hasattr(self, "branches_to_output")
                    and self.blocklinear_strategy == "conductance_dynamic"
                ):
                    input_scale_vec = 1 / self.branches_to_output.log_weight.exp().flatten()
                    self.branches_to_output.input_scale_vec = input_scale_vec

    def get_weights(self):
        if self.input_inhibitory and self.input_excitatory:
            return (self.branch_excitation.weight(), self.branch_inhibition.weight())
        elif self.input_excitatory:
            return self.branch_excitation.weight()
        elif self.input_inhibitory:
            return self.branch_inhibition.weight()
        else:
            return None

    def get_log_weights(self):
        if self.input_inhibitory and self.input_excitatory:
            return (
                self.branch_excitation.log_weight(),
                self.branch_inhibition.log_weight(),
            )
        elif self.input_excitatory:
            return self.branch_excitation.log_weight()
        elif self.input_inhibitory:
            return self.branch_inhibition.log_weight()
        else:
            return None

    def get_mask(self):
        if self.input_inhibitory and self.input_excitatory:
            return (
                self.branch_excitation.weight_mask(),
                self.branch_inhibition.weight_mask(),
            )
        elif self.input_excitatory:
            return self.branch_excitation.weight_mask()
        elif self.input_inhibitory:
            return self.branch_inhibition.weight_mask()
        else:
            return None

    def get_pruned_weights(self):
        if self.input_inhibitory and self.input_excitatory:
            return (
                self.branch_excitation.pruned_weight(),
                self.branch_inhibition.pruned_weight(),
            )
        elif self.input_excitatory:
            return self.branch_excitation.pruned_weight()
        elif self.input_inhibitory:
            return self.branch_inhibition.pruned_weight()
        else:
            return None

    def get_log_pruned_weights(self):
        if self.input_inhibitory and self.input_excitatory:
            return (
                self.branch_excitation.log_pruned_weight(),
                self.branch_inhibition.log_pruned_weight(),
            )
        elif self.input_excitatory:
            return self.branch_excitation.log_pruned_weight()
        elif self.input_inhibitory:
            return self.branch_inhibition.log_pruned_weight()
        else:
            return None


class DendriNet(nn.Module):
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
        # DEEPST specific parameters, clean up later
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode='global',
        synapses_per_branch=None,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method=None,
        weight_threshold=1e-6,
        **kwargs,
    ):
        super(DendriNet, self).__init__()

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

        if (
            inhibitory_input_dim is not None
            and inhibitory_synapses_per_branch is not None
        ):
            if not isinstance(inhibitory_synapses_per_branch, int):
                raise TypeError("inhibitory_synapses_per_branch must be int")
            if inhibitory_synapses_per_branch < 1:
                raise ValueError("inhibitory_synapses_per_branch must be >= 1")

        if (
            excitatory_input_dim is not None
            and excitatory_synapses_per_branch is not None
        ):
            if not isinstance(excitatory_synapses_per_branch, int):
                raise TypeError("excitatory_synapses_per_branch must be int")
            if excitatory_synapses_per_branch < 1:
                raise ValueError("excitatory_synapses_per_branch must be >= 1")

        branch_factors = deepcopy(branch_factors)
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
                topk_temperature=topk_temperature,  # Pass the temperature
                topk_ultrafast=topk_ultrafast,  # Pass the ultrafast flag
                topk_strategy=topk_strategy,
                use_shunting=use_shunting,
                reactivate=reactivate,
                reactivation_init_m=reactivation_init_m,
                reactivation_init_b=reactivation_init_b,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
                layer_idx=i,
                print_hooks=print_hooks,
                use_noise=use_noise,
                rewiring_mode=rewiring_mode,
                synapses_per_branch=synapses_per_branch,
                excitatory_target_density=excitatory_target_density,
                inhibitory_target_density=inhibitory_target_density,
                freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                init_method=init_method,
                weight_threshold=weight_threshold,
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
                    topk_temperature=topk_temperature,  # Pass the temperature
                    topk_ultrafast=topk_ultrafast,  # Pass the ultrafast flag
                    topk_strategy=topk_strategy,
                    use_shunting=use_shunting,
                    reactivate=reactivate,
                    reactivation_init_m=reactivation_init_m,
                    reactivation_init_b=reactivation_init_b,
                    reactivation_type=reactivation_type,
                    reactivation_strategy=reactivation_strategy,
                    blocklinear_strategy=blocklinear_strategy,
                    layer_idx=self.n_branch_layers,
                    print_hooks=print_hooks,
                    use_noise=use_noise,
                    rewiring_mode=rewiring_mode,
                    synapses_per_branch=synapses_per_branch,
                    excitatory_target_density=excitatory_target_density,
                    inhibitory_target_density=inhibitory_target_density,
                    freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                    freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                    init_method=init_method,
                    weight_threshold=weight_threshold,
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
                    topk_temperature=topk_temperature,  # Pass the temperature
                    topk_ultrafast=topk_ultrafast,  # Pass the ultrafast flag
                    topk_strategy=topk_strategy,
                    use_shunting=use_shunting,
                    reactivate=reactivate,
                    reactivation_init_m=reactivation_init_m,
                    reactivation_init_b=reactivation_init_b,
                    reactivation_type=reactivation_type,
                    reactivation_strategy=reactivation_strategy,
                    blocklinear_strategy=blocklinear_strategy,
                    layer_idx=self.n_branch_layers,
                    print_hooks=print_hooks,
                    use_noise=use_noise,
                    rewiring_mode=rewiring_mode,
                    synapses_per_branch=synapses_per_branch,
                    excitatory_target_density=excitatory_target_density,
                    inhibitory_target_density=inhibitory_target_density,
                    freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                    freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                    init_method=init_method,
                    weight_threshold=weight_threshold,
                    **kwargs
                )
            )

        self.branch_layers = nn.ModuleList(branch_layers)

        self.reactivate = reactivate
        self.layer_sizes = layer_sizes
        self.input_inhibitory = inhibitory_input_dim is not None
        self.n_soma = n_soma
        self.somatic_synapses = somatic_synapses
    
    def apply_rewiring(self):
        for layer in self.branch_layers:
            layer.apply_rewiring()

    def decay_weights(self, weight_decay):
        for branch_layer in self.branch_layers:
            branch_layer.decay_weights(weight_decay)

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
            layer = self.branch_layers[i]

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


class DendriNetWithOutputs(DendriNet):
    """
    A version of DendriNet that also returns branch activations from forward pass.
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
        # DEEPST specific parameters, clean up later
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode='global',
        synapses_per_branch=None,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method=None,
        weight_threshold=1e-6,
        **kwargs
    ):
        super().__init__(
            n_soma,
            branch_factors,
            excitatory_input_dim,
            excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            reactivate=reactivate,
            somatic_synapses=somatic_synapses,
            topk_init_method=topk_init_method,
            topk_noise_level=topk_noise_level,
            topk_type=topk_type,  # New parameter for TopK strategy
            topk_temperature=topk_temperature,  # For stochastic TopK
            topk_ultrafast=topk_ultrafast,  # For much faster stochastic TopK
            topk_strategy=topk_strategy,
            use_shunting=use_shunting,
            reactivation_type=reactivation_type,
            reactivation_strategy=reactivation_strategy,
            blocklinear_strategy=blocklinear_strategy,
            print_hooks=print_hooks,
            reactivation_init_m = reactivation_init_m,
            reactivation_init_b = reactivation_init_b,
            # DEEPST specific parameters, clean up later
            excitatory_target_density=excitatory_target_density,
            inhibitory_target_density=inhibitory_target_density,
            use_noise=use_noise,
            sigma=sigma,
            rewiring_mode=rewiring_mode,
            synapses_per_branch=synapses_per_branch,
            freeze_excitatory_connectivity=freeze_excitatory_connectivity,
            freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
            init_method=init_method,
            weight_threshold=weight_threshold,
            **kwargs
        )

    def forward_with_branch_outputs(self, x, inhibitory_input=None):
        """
        Forward pass that returns branch activations for each layer.
        Important: For visualization and information analysis.
        Properly handles somatic_synapses=False configuration.

        Returns:
            final_soma: Tensor of final output
            per_layer_acts: List of processed activations per layer
                Each element is a list with format [excitatory, inhibitory, combined]
                Elements may be None if not available for that layer
        """
        per_layer_acts = []
        branch_input = None

        for i in range(self.n_branch_layers):
            current_layer = self.branch_layers[i]

            # --- Compute excitatory & inhibitory partial transforms ---
            excitatory_input = None
            inhibitory_input_processed = None

            if (
                hasattr(current_layer, "branch_excitation")
                and current_layer.branch_excitation is not None
            ):
                try:
                    excitatory_input = current_layer.branch_excitation(x)
                except Exception:
                    pass

            if (
                inhibitory_input is not None
                and hasattr(current_layer, "branch_inhibition")
                and current_layer.branch_inhibition is not None
            ):
                try:
                    inhibitory_input_processed = current_layer.branch_inhibition(
                        inhibitory_input
                    )
                except Exception:
                    pass

            if hasattr(current_layer, "excitatory_input"):
                current_layer.excitatory_input = excitatory_input  # was 'x' before

            if hasattr(current_layer, "inhibitory_input"):
                current_layer.inhibitory_input = (
                    inhibitory_input_processed  # was 'inhibitory_input' before
                )

            output = current_layer(x, inhibitory_input, branch_input)

            layer_info = [excitatory_input, inhibitory_input_processed, output]
            per_layer_acts.append(layer_info)

            branch_input = output

        # -----------------------------
        # Final soma layer processing
        final_layer = self.branch_layers[-1]

        # Compute final excit/inhib partial transforms for the final layer
        final_excitatory = None
        final_inhibitory = None

        if (
            hasattr(final_layer, "branch_excitation")
            and final_layer.branch_excitation is not None
        ):
            try:
                final_excitatory = final_layer.branch_excitation(x)
            except Exception:
                pass

        if (
            inhibitory_input is not None
            and hasattr(final_layer, "branch_inhibition")
            and final_layer.branch_inhibition is not None
        ):
            try:
                final_inhibitory = final_layer.branch_inhibition(inhibitory_input)
            except Exception:
                pass

        if hasattr(final_layer, "excitatory_input"):
            final_layer.excitatory_input = final_excitatory

        if hasattr(final_layer, "inhibitory_input"):
            final_layer.inhibitory_input = final_inhibitory

        # Now get final soma output:
        final_soma = final_layer(x, inhibitory_input, branch_input)

        # Add the final layer's info to the per_layer_acts
        per_layer_acts.append([final_excitatory, final_inhibitory, final_soma])

        return final_soma, per_layer_acts

    def forward(self, x, inhibitory_input=None):
        """
        The simpler forward pass, typically used during training.
        We still assign excitatory_input / inhibitory_input to the partial transforms
        so that any visualization or ablation code sees them.
        """
        output = None
        for i in range(self.n_branch_layers):
            layer_i = self.branch_layers[i]

            excitatory_in = None
            inhibitory_in = None

            if (
                hasattr(layer_i, "branch_excitation")
                and layer_i.branch_excitation is not None
            ):
                try:
                    excitatory_in = layer_i.branch_excitation(x)
                except Exception:
                    pass

            if (
                inhibitory_input is not None
                and hasattr(layer_i, "branch_inhibition")
                and layer_i.branch_inhibition is not None
            ):
                try:
                    inhibitory_in = layer_i.branch_inhibition(inhibitory_input)
                except Exception:
                    pass

            if hasattr(layer_i, "excitatory_input"):
                layer_i.excitatory_input = excitatory_in  # was 'x'

            if hasattr(layer_i, "inhibitory_input"):
                layer_i.inhibitory_input = inhibitory_in  # was 'inhibitory_input'

            # Actually run the layer
            output = layer_i(x, inhibitory_input, output)

        # For the final layer:
        final_layer = self.branch_layers[-1]

        final_excit = None
        final_inhib = None

        if (
            hasattr(final_layer, "branch_excitation")
            and final_layer.branch_excitation is not None
        ):
            try:
                final_excit = final_layer.branch_excitation(x)
            except Exception:
                pass

        if (
            inhibitory_input is not None
            and hasattr(final_layer, "branch_inhibition")
            and final_layer.branch_inhibition is not None
        ):
            try:
                final_inhib = final_layer.branch_inhibition(inhibitory_input)
            except Exception:
                pass

        if hasattr(final_layer, "excitatory_input"):
            final_layer.excitatory_input = final_excit

        if hasattr(final_layer, "inhibitory_input"):
            final_layer.inhibitory_input = final_inhib

        final_soma = final_layer(x, inhibitory_input, output)
        return final_soma
