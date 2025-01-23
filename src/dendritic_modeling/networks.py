"""
networks.py
========================
This module contains the ExcitationInhibitionLayer and 
ExcitationInhibitionNetwork classes.
"""

from copy import deepcopy
from torch import nn
from typing import List
from dendritic_modeling.dendrinet import DendriNet
from dendritic_modeling import logger

class ExcitationInhibitionLayer(nn.Module):
    """
    ...
    """
    def __init__(
        self, 
        n_excitatory_cells,
        n_inhibitory_cells,
        excitatory_branch_factors, 
        inhibitory_branch_factors,
        excitatory_input_dim, 
        ee_synapses_per_branch, 
        ei_synapses_per_branch, 
        inhibitory_input_dim = None, 
        ie_synapses_per_branch = None, 
        ii_synapses_per_branch = None, 
        reactivate = True, 
        somatic_synapses = True,
        topk_init_method = "xavier_normal",
        use_shunting = True,
        init_gain=1.0,
        init_inhib_offset=0.0,
        gradient_strategy="none",
        gradient_scale_factor=1.0,
        reactivation_initial_m=1.5,
        reactivation_initial_b=1.0,
    ):
        super(ExcitationInhibitionLayer, self).__init__()
        self.init_gain = init_gain
        self.init_inhib_offset = init_inhib_offset
        self.gradient_strategy = gradient_strategy
        self.gradient_scale_factor = gradient_scale_factor

        # inhibitory cells
        self.inhibitory_cells = DendriNet(
            n_soma = n_inhibitory_cells, 
            branch_factors = inhibitory_branch_factors,
            excitatory_input_dim = excitatory_input_dim, 
            excitatory_synapses_per_branch = ei_synapses_per_branch,
            inhibitory_input_dim = inhibitory_input_dim, 
            inhibitory_synapses_per_branch = ii_synapses_per_branch,
            reactivate = reactivate, 
            somatic_synapses = somatic_synapses, 
            topk_init_method = topk_init_method,
            use_shunting=use_shunting,
            reactivation_initial_m=reactivation_initial_m,
            reactivation_initial_b=reactivation_initial_b,
        )

        # excitatory cells
        self.excitatory_cells = DendriNet(
            n_soma = n_excitatory_cells, 
            branch_factors = excitatory_branch_factors,
            excitatory_input_dim = excitatory_input_dim, 
            excitatory_synapses_per_branch = ee_synapses_per_branch,
            inhibitory_input_dim = n_inhibitory_cells, 
            inhibitory_synapses_per_branch = ie_synapses_per_branch,
            reactivate = reactivate, 
            somatic_synapses = somatic_synapses,
            topk_init_method = topk_init_method,
            use_shunting=use_shunting,
            reactivation_initial_m=reactivation_initial_m,
            reactivation_initial_b=reactivation_initial_b,
        )
    
    def decay_weights(self, weight_decay):
        self.inhibitory_cells.decay_weights(weight_decay)
        self.excitatory_cells.decay_weights(weight_decay)

    def forward(self, x, inhibitory_input = None):
        inhibitory_output = self.inhibitory_cells(x, inhibitory_input)
        excitatory_output = self.excitatory_cells(x, inhibitory_output)
        return excitatory_output, inhibitory_output


class ExcitationInhibitionNetwork(nn.Module):
    """
    ...
    """
    def __init__(
        self, 
        input_dim, 
        excitatory_layer_sizes, 
        inhibitory_layer_sizes,
        excitatory_branch_factors, 
        inhibitory_branch_factors, 
        ee_synapses_per_branch_per_layer, 
        ei_synapses_per_branch_per_layer, 
        ie_synapses_per_branch_per_layer, 
        ii_synapses_per_branch_per_layer = [], 
        reactivate = False, 
        somatic_synapses = True,
        topk_init_method = "xavier_normal",
        use_shunting=True,
        init_gain=1.0,
        init_inhib_offset=0.0,
        gradient_strategy="none",
        gradient_scale_factor=1.0,
        weight_decay_rate=0.1,
        reactivation_initial_m=1.5,
        reactivation_initial_b=1.0,

        # NEW FEATURE: local learning
        local_learning="none", 
        local_loss_weight=0.0,
    ):
        super(ExcitationInhibitionNetwork, self).__init__()
        self.weight_decay_rate = weight_decay_rate

        self.n_layers = len(excitatory_layer_sizes)
        self.init_gain = init_gain
        self.init_inhib_offset = init_inhib_offset
        self.gradient_strategy = gradient_strategy
        self.gradient_scale_factor = gradient_scale_factor

        # local learning approach
        self.local_learning = local_learning
        self.local_loss_weight = local_loss_weight

        logger.debug(f'Building ExcitationInhibitionNetwork with {self.n_layers} layer(s).')

        excitatory_layer_sizes = deepcopy(excitatory_layer_sizes)
        inhibitory_layer_sizes = deepcopy(inhibitory_layer_sizes)

        excitatory_layer_sizes.insert(0, input_dim)
        inhibitory_layer_sizes.insert(0, None)

        ii_synapses_per_branch_per_layer = deepcopy(ii_synapses_per_branch_per_layer)
        ii_synapses_per_branch_per_layer.insert(0, None)

        layers = []
        for i in range(self.n_layers):
            layers.append(ExcitationInhibitionLayer(
                n_excitatory_cells = excitatory_layer_sizes[i+1], 
                n_inhibitory_cells = inhibitory_layer_sizes[i+1],
                excitatory_branch_factors = excitatory_branch_factors, 
                inhibitory_branch_factors = inhibitory_branch_factors,
                excitatory_input_dim = excitatory_layer_sizes[i], 
                ee_synapses_per_branch = ee_synapses_per_branch_per_layer[i], 
                ei_synapses_per_branch = ei_synapses_per_branch_per_layer[i],
                inhibitory_input_dim = inhibitory_layer_sizes[i],
                ie_synapses_per_branch = ie_synapses_per_branch_per_layer[i], 
                ii_synapses_per_branch = ii_synapses_per_branch_per_layer[i],
                reactivate = reactivate, 
                somatic_synapses = somatic_synapses,
                topk_init_method = topk_init_method,
                use_shunting=use_shunting,
                init_gain = self.init_gain,
                init_inhib_offset = self.init_inhib_offset,
                gradient_strategy = self.gradient_strategy,
                gradient_scale_factor = self.gradient_scale_factor,
                reactivation_initial_m = reactivation_initial_m,
                reactivation_initial_b = reactivation_initial_b,
            ))
        
        self.layers = nn.ModuleList(layers)
    
    def decay_weights(self):
        """
        Apply custom weight decay to each ExcitationInhibitionLayer
        using self.weight_decay_rate.
        """
        for layer in self.layers:
            layer.decay_weights(self.weight_decay_rate)

    def forward(self, x):
        excitatory = x
        inhibitory = None
        for i, layer in enumerate(self.layers):
            # if gradient_strategy == "distal_upweight", 
            # we can do a custom scale factor that depends on layer index
            if self.gradient_strategy == "distal_upweight":
                # For example: deeper layers get bigger scale
                # This is only conceptual: layer 0 => scale=2, layer 1 => scale=1.5, ...
                # You can store a map or formula:
                scale = 2.0/(i+1)
                layer.gradient_scale_factor = scale

            excitatory, inhibitory = layer(excitatory, inhibitory)
        return excitatory

    # NEW FEATURE: local learning example
    def compute_local_loss(self, x, y=None):
        """
        Demonstration of local objectives at each layer.
        We sum them up with self.local_loss_weight.

        E.g. "backprop_alignment": for each layer, we do partial alignment with 
        final target or random projection. (This is a conceptual placeholder.)
        """
        if self.local_learning == "none":
            return 0.0

        local_losses = []
        # For demonstration: we re-forward pass
        excitatory = x
        inhibitory = None
        for layer_idx, layer in enumerate(self.layers):
            excitatory, inhibitory = layer(excitatory, inhibitory)
            # Possibly define a "local" label or partial target
            if self.local_learning == "backprop_alignment":
                # We can do a silly example: MSE vs. a random vector 
                # or alignment with final label onehot
                # Conceptual only:
                # shape: excitatory => [batch_size, n_excitatory_cells]
                local_obj = excitatory.mean(dim=1).sum()
                local_losses.append(local_obj)

        total_local = torch.stack(local_losses).mean()
        return self.local_loss_weight * total_local
    
class FeedforwardNN(nn.Module):
    """
    A configurable feedforward neural network as a baseline for comparison
    on the flattened MNIST dataset.
    """
    def __init__(
        self,
        input_dim: int = 784,
        hidden_layer_sizes: List[int] = [128, 64],
        activation: str = "relu",
        dropout: float = 0.2,
        output_dim: int = 10,
    ):
        super(FeedforwardNN, self).__init__()

        # Map string activation to PyTorch activation functions
        activation_mapping = {
            "relu": nn.ReLU,
            "sigmoid": nn.Sigmoid,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU
        }

        # Raise an error if the activation is not supported
        if activation not in activation_mapping:
            raise ValueError(f"Unsupported activation function: {activation}. "
                             f"Supported: {list(activation_mapping.keys())}")

        # Define layers dynamically
        layers = []
        current_dim = input_dim

        for layer_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_dim, layer_size))
            layers.append(activation_mapping[activation]())
            if dropout > 0:  # Apply dropout only if it's greater than 0
                layers.append(nn.Dropout(dropout))
            current_dim = layer_size

        # Add the output layer
        layers.append(nn.Linear(current_dim, output_dim))

        # Combine layers into a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
    
class MLPBaseline(nn.Module):
    """
    A simple feedforward MLP with one hidden layer, for baseline comparison.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPBaseline, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # raw logits