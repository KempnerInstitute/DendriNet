"""
networks.py
========================
Modified to allow different input modes for the *first* layer of the
ExcitationInhibitionNetwork.

Requests:
---------
1) input_mode=0:

   - The *first-layer* inhibitory net does get built with input=x,
   - Then the inhibitory net's output is used by the excitatory net's inhibitory synapses
     in *that same layer* (the original behavior).

2) input_mode=1 and 2:

   - Skip making the inhibitory sub-network ONLY in the *first layer*,
     but do allow the inhibitory sub-network for subsequent layers
     (if inhibitory_layer_sizes > 0 for those layers).

3) We still have an MLP transform if input_mode=2 (for the first layer).
"""

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchinfo
from torch import nn

from dendritic_modeling import logger
from dendritic_modeling.dendrinet import DendriNet, DendriNetWithOutputs
from dendritic_modeling.utils import SafeHookModule, safe_register_hooks


###############################################################################
# InputNetTransform
###############################################################################
class InputNetTransform(nn.Module):
    """
    A transform that maps the raw input x to (excitatory_input, inhibitory_input).
    This is a generalized version that can support different network architectures.
    By default, it uses an MLP with configurable hidden dimensions.
    """

    def __init__(
        self, 
        input_dim: Optional[int] = None, 
        hidden_dims: Optional[List[int]] = None, 
        excitatory_dim: Optional[int] = None, 
        inhibitory_dim: Optional[int] = None, 
        input_shape: Optional[Tuple[int, int, int]] = None,
        excitatory_hidden_dims: Optional[List[int]] = None,
        inhibitory_hidden_dims: Optional[List[int]] = None,
        excitatory_kernel_sizes: Optional[List[int]] = None,
        inhibitory_kernel_sizes: Optional[List[int]] = None,
        excitatory_strides: Optional[List[int]] = None,
        inhibitory_strides: Optional[List[int]] = None,
        network_type: str = "MLPInputTransform"
    ):
        super().__init__()
        self.network_type = network_type

        if network_type == "IndexInputTransform":
            self.input_transform = IndexInputTransform(
                input_dim = input_dim,
                excitatory_dim = excitatory_dim,
                inhibitory_dim = inhibitory_dim
            )

        elif network_type == "MLPInputTransform":
            self.input_transform = MLPInputTransform(
                input_dim = input_dim,
                hidden_dims = hidden_dims,
                excitatory_dim = excitatory_dim,
                inhibitory_dim = inhibitory_dim
            )

        elif network_type == "AlexNetTransform":
            self.input_transform = AlexNetInputTransform(
                excitatory_dim = excitatory_dim,
                inhibitory_dim = inhibitory_dim
            )

        elif network_type == "CNNInputTransform":
            self.input_transform = CNNInputTransform(
                input_shape = input_shape,
                excitatory_hidden_dims = excitatory_hidden_dims,
                inhibitory_hidden_dims = inhibitory_hidden_dims,
                excitatory_kernel_sizes = excitatory_kernel_sizes,
                inhibitory_kernel_sizes = inhibitory_kernel_sizes,
                excitatory_strides = excitatory_strides,
                inhibitory_strides = inhibitory_strides
            )

        else:
            raise ValueError(f"Unknown network type: {network_type}")

    def forward(self, x):
        excitatory_input, inhibitory_input = self.input_transform(x)
        return excitatory_input, inhibitory_input

    def load_pretrained_weights(self, source, weights_path=None, pytorch_model=None):
        """
        Load pre-trained weights for the input network.

        Args:
            source (str): Source of weights - 'file', 'pytorch', or 'none'
            weights_path (str, optional): Path to weights file if source is 'file'
            pytorch_model (str, optional): Name of PyTorch model if source is 'pytorch'
        """
        if source == "none":
            logger.info("No pre-trained weights to load.")
            return

        elif source == "file":
            if not weights_path:
                raise ValueError("weights_path must be provided when source is 'file'")
            logger.info(f"Loading weights from file: {weights_path}")
            state_dict = torch.load(weights_path)
            self.load_state_dict(state_dict)

        elif source == "pytorch":
            if not pytorch_model:
                raise ValueError(
                    "pytorch_model must be provided when source is 'pytorch'"
                )
            logger.info(f"Loading pre-trained PyTorch model: {pytorch_model}")

            import torchvision.models as models

            if pytorch_model == "alexnet":
                pretrained_model = models.alexnet(pretrained=True)
                # Copy features to shared backbone
                if self.network_type == "AlexNetTransform":
                    # For AlexNet, we can directly copy the features
                    for i, layer in enumerate(list(pretrained_model.features)):
                        if i < len(list(self.shared)):
                            self.shared[i].load_state_dict(layer.state_dict())
            else:
                raise ValueError(f"Unsupported PyTorch model: {pytorch_model}")
        else:
            raise ValueError(f"Unknown source: {source}")

# For backward compatibility
# MLPInputTransform = InputNetTransform


class IndexInputTransform(nn.Module):
    def __init__(self, input_dim: int, excitatory_dim: int, inhibitory_dim: int):
        super().__init__()
        try:
            assert excitatory_dim + inhibitory_dim == input_dim, \
                'error in IndexInputTransform: excitatory_dim + inhibitory_dim must ' \
                f'equal input_dim, got {excitatory_dim} + {inhibitory_dim} != {input_dim}'
        except AssertionError as e:
            logger.error(e)
            inhibitory_dim = int(input_dim - excitatory_dim)
        
        dims = torch.randperm(input_dim)
        exc_dims = dims[:excitatory_dim]
        inh_dims = dims[excitatory_dim:]

        self.exc_dims : torch.Tensor
        self.register_buffer('exc_dims', exc_dims)
        self.inh_dims : torch.Tensor
        self.register_buffer('inh_dims', inh_dims)

        self.excitatory_dim = excitatory_dim
        self.inhibitory_dim = inhibitory_dim

    def forward(self, x):
        if x.dim() == 1:
            x = x[None, :]
        
        excitatory_input = x[..., self.exc_dims]
        inhibitory_input = x[..., self.inh_dims]
        return excitatory_input, inhibitory_input


class MLPInputTransform(nn.Module):
    """
    A transform that maps the raw input x to (excitatory_input, inhibitory_input).
    This is a generalized version that can support different network architectures.
    By default, it uses an MLP with configurable hidden dimensions.
    """

    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        excitatory_dim: int, 
        inhibitory_dim: int
    ):
        super().__init__()

        # Build MLP backbone
        exc_layers = []
        inh_layers = []
        in_features = input_dim
        for hdim in hidden_dims:
            exc_layer = nn.Linear(in_features, hdim)
            nn.init.kaiming_normal_(exc_layer.weight)
            nn.init.zeros_(exc_layer.bias)
            
            exc_layers.append(exc_layer)
            exc_layers.append(nn.ReLU())

            inh_layer = nn.Linear(in_features, hdim)
            nn.init.kaiming_normal_(inh_layer.weight)
            nn.init.zeros_(inh_layer.bias)
            inh_layers.append(inh_layer)
            inh_layers.append(nn.ReLU())

            in_features = hdim
        
        exc_head = nn.Linear(in_features, excitatory_dim)
        nn.init.xavier_normal_(exc_head.weight)
        nn.init.zeros_(exc_head.bias)
        exc_layers.append(exc_head)
        exc_layers.append(nn.Sigmoid())
        
        inh_head = nn.Linear(in_features, inhibitory_dim)
        nn.init.xavier_normal_(inh_head.weight)
        nn.init.zeros_(inh_head.bias)
        inh_layers.append(inh_head)
        inh_layers.append(nn.Sigmoid())

        self.exc_encoding = nn.Sequential(*exc_layers)
        self.inh_encoding = nn.Sequential(*inh_layers)

        self.excitatory_dim = excitatory_dim
        self.inhibitory_dim = inhibitory_dim

    def forward(self, x):
        excitatory_input = self.exc_encoding(x)
        inhibitory_input = self.inh_encoding(x)
        return excitatory_input, inhibitory_input


class CNNInputTransform(nn.Module):
    """
    A transform that maps the raw input x to (excitatory_input, inhibitory_input)
    using convolutional layers.
    """

    def __init__(
        self, 
        input_shape: Tuple[int, int, int],  # (channels, height, width)
        excitatory_hidden_dims: List[int],           # List of channel sizes for each conv layer
        inhibitory_hidden_dims: List[int],           # List of channel sizes for each conv layer
        excitatory_kernel_sizes: List[int],          # Kernel size for all conv layers
        inhibitory_kernel_sizes: List[int],          # Kernel size for all conv layers
        excitatory_strides: List[int],               # Stride for all conv layers
        inhibitory_strides: List[int],               # Stride for all conv layers
    ):
        super().__init__()
        
        # Input validation
        input_dim, height, width = input_shape
        
        # Build CNN backbone for excitatory pathway
        exc_layers = []
        in_dim = input_dim
        current_h, current_w = height, width
        
        for hidden_dim, kernel_size, stride in zip(
            excitatory_hidden_dims[:-1], excitatory_kernel_sizes[:-1], excitatory_strides[:-1]
        ):
            # Add convolutional layer
            exc_conv = nn.Conv2d(in_dim, hidden_dim, kernel_size, stride)
            nn.init.kaiming_normal_(exc_conv.weight)
            nn.init.zeros_(exc_conv.bias)
            exc_layers.append(exc_conv)
            exc_layers.append(nn.ReLU())
            
            in_dim = hidden_dim
            current_h = ((current_h - kernel_size) // stride) + 1
            current_w = ((current_w - kernel_size) // stride) + 1
        
        # Add final convolutional layer
        exc_conv = nn.Conv2d(in_dim, excitatory_hidden_dims[-1], excitatory_kernel_sizes[-1], excitatory_strides[-1])
        nn.init.xavier_normal_(exc_conv.weight)
        nn.init.zeros_(exc_conv.bias)
        exc_layers.append(exc_conv)
        exc_layers.append(nn.Sigmoid())

        current_h = ((current_h - excitatory_kernel_sizes[-1]) // excitatory_strides[-1]) + 1
        current_w = ((current_w - excitatory_kernel_sizes[-1]) // excitatory_strides[-1]) + 1

        # Flatten layer
        exc_layers.append(nn.Flatten(start_dim = -3))
        
        # Calculate flattened dimension
        self.excitatory_dim = excitatory_hidden_dims[-1] * current_h * current_w
        
        # Build CNN backbone for inhibitory pathway
        inh_layers = []
        in_dim = input_dim
        current_h, current_w = height, width
        
        for hidden_dim, kernel_size, stride in zip(
            inhibitory_hidden_dims[:-1], inhibitory_kernel_sizes[:-1], inhibitory_strides[:-1]
        ):
            # Add convolutional layer
            inh_conv = nn.Conv2d(in_dim, hidden_dim, kernel_size, stride)
            nn.init.kaiming_normal_(inh_conv.weight)
            nn.init.zeros_(inh_conv.bias)
            inh_layers.append(inh_conv)
            inh_layers.append(nn.ReLU())
            
            in_dim = hidden_dim
            current_h = ((current_h - kernel_size) // stride) + 1
            current_w = ((current_w - kernel_size) // stride) + 1
        
        # Add final convolutional layer
        inh_conv = nn.Conv2d(in_dim, inhibitory_hidden_dims[-1], inhibitory_kernel_sizes[-1], inhibitory_strides[-1])
        nn.init.xavier_normal_(inh_conv.weight)
        nn.init.zeros_(inh_conv.bias)
        inh_layers.append(inh_conv)
        inh_layers.append(nn.Sigmoid())

        current_h = ((current_h - inhibitory_kernel_sizes[-1]) // inhibitory_strides[-1]) + 1
        current_w = ((current_w - inhibitory_kernel_sizes[-1]) // inhibitory_strides[-1]) + 1

        # Flatten layer
        inh_layers.append(nn.Flatten(start_dim = -3))
        
        # Calculate flattened dimension
        self.inhibitory_dim = inhibitory_hidden_dims[-1] * current_h * current_w
        
        # Create sequential models
        self.exc_encoding = nn.Sequential(*exc_layers)
        self.inh_encoding = nn.Sequential(*inh_layers)

    def forward(self, x):
        # Ensure input has correct shape (add batch dimension if needed)
        if x.dim() == 3:  # (C, H, W)
            x = x.unsqueeze(0)  # Add batch dimension: (1, C, H, W)
            
        excitatory_input = self.exc_encoding(x)
        inhibitory_input = self.inh_encoding(x)
        return excitatory_input, inhibitory_input


class AlexNetInputTransform(nn.Module):
    """
    A transform that maps the raw input x to (excitatory_input, inhibitory_input)
    using convolutional layers.
    """
    def __init__(self, excitatory_dim: int, inhibitory_dim: int):
        super().__init__()
        import torchvision.models as models

        # Load AlexNet without classifier
        alexnet = models.alexnet(pretrained=True)
        self.shared = nn.Sequential(
            *list(alexnet.features), nn.AdaptiveAvgPool2d((6, 6)), nn.Flatten()
        )

        # Create new heads for excitatory and inhibitory outputs
        self.head_exc = nn.Sequential(nn.Linear(9216, excitatory_dim), nn.Sigmoid())
        self.head_inh = nn.Sequential(nn.Linear(9216, inhibitory_dim), nn.Sigmoid())

        self.excitatory_dim = excitatory_dim
        self.inhibitory_dim = inhibitory_dim

    def forward(self, x):
        x = self.shared(x)
        excitatory_input = self.head_exc(x)
        inhibitory_input = self.head_inh(x)
        return excitatory_input, inhibitory_input


###############################################################################
# SigmoidInputNetTransform
###############################################################################

class SigmoidInputNetTransform(InputNetTransform):
    """
    Extension of InputNetTransform that applies a sigmoid activation to the output,
    ensuring values are in the range [0, 1] before entering the dendrinet.
    """

    def __init__(
        self, input_dim, hidden_dims, excitatory_dim, inhibitory_dim, network_type="MLPInputTransform"
    ):
        super().__init__(input_dim, hidden_dims, excitatory_dim, inhibitory_dim, network_type)

    def forward(self, x):
        # Call the parent's forward method to get outputs
        excit_out, inhib_out = InputNetTransform.forward(self, x)

        # Apply sigmoid activation to ensure outputs are in [0, 1]
        excit_out = torch.sigmoid(excit_out)
        inhib_out = torch.sigmoid(inhib_out)

        return excit_out, inhib_out

###############################################################################
# ExcitationInhibitionLayer
###############################################################################
class ExcitationInhibitionLayer(nn.Module):
    def __init__(
        self,
        n_excitatory_cells,
        n_inhibitory_cells,
        excitatory_branch_factors,
        inhibitory_branch_factors,
        excitatory_input_dim,
        ee_synapses_per_branch,
        ei_synapses_per_branch,
        inhibitory_input_dim=None,
        ie_synapses_per_branch=None,
        ii_synapses_per_branch=None,
        reactivate=True,
        reactivation_init_m=1.5,
        reactivation_init_b=0.5,
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        use_shunting=True,
        reactivation_strategy="none",
        blocklinear_strategy="none",
        reactivation_type="param_tanh",
        enable_branch_outputs=False,
        topk_noise_level=0.0,
        topk_type="standard",
        topk_strategy="none",
        print_hooks=False,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode="global",
        synapses_per_branch=None,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method="xavier_normal",
        weight_threshold=1e-6,
        **kwargs
    ):
        super().__init__()

        # Decide if we build an inhibitory sub-network
        self.has_inhib_subnet = (
            n_inhibitory_cells is not None
            and isinstance(n_inhibitory_cells, int)
            and n_inhibitory_cells > 0
        )

        if self.has_inhib_subnet:
            if enable_branch_outputs:
                self.inhibitory_cells = DendriNetWithOutputs(
                    n_soma=n_inhibitory_cells,
                    branch_factors=inhibitory_branch_factors,
                    excitatory_input_dim=excitatory_input_dim,
                    excitatory_synapses_per_branch=ei_synapses_per_branch,
                    inhibitory_input_dim=inhibitory_input_dim,
                    inhibitory_synapses_per_branch=ii_synapses_per_branch,
                    reactivate=reactivate,
                    somatic_synapses=somatic_synapses,
                    topk_init_method=topk_init_method,
                    use_shunting=use_shunting,
                    reactivation_type=reactivation_type,
                    reactivation_init_m=reactivation_init_m,
                    reactivation_init_b=reactivation_init_b,
                    reactivation_strategy=reactivation_strategy,
                    blocklinear_strategy=blocklinear_strategy,
                    topk_noise_level=topk_noise_level,
                    topk_strategy=topk_strategy,
                    topk_type=topk_type,
                    print_hooks=print_hooks,
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
            else:
                self.inhibitory_cells = DendriNet(
                    n_soma=n_inhibitory_cells,
                    branch_factors=inhibitory_branch_factors,
                    excitatory_input_dim=excitatory_input_dim,
                    excitatory_synapses_per_branch=ei_synapses_per_branch,
                    inhibitory_input_dim=inhibitory_input_dim,
                    inhibitory_synapses_per_branch=ii_synapses_per_branch,
                    reactivate=reactivate,
                    reactivation_init_m=reactivation_init_m,
                    reactivation_init_b=reactivation_init_b,
                    somatic_synapses=somatic_synapses,
                    topk_init_method=topk_init_method,
                    use_shunting=use_shunting,
                    reactivation_type=reactivation_type,
                    reactivation_strategy=reactivation_strategy,
                    blocklinear_strategy=blocklinear_strategy,
                    topk_noise_level=topk_noise_level,
                    topk_strategy=topk_strategy,
                    topk_type=topk_type,
                    print_hooks=print_hooks,
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
        else:
            self.inhibitory_cells = None

        if enable_branch_outputs:
            self.excitatory_cells = DendriNetWithOutputs(
                n_soma=n_excitatory_cells,
                branch_factors=excitatory_branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=ee_synapses_per_branch,
                inhibitory_input_dim=n_inhibitory_cells
                if self.has_inhib_subnet
                else None,
                inhibitory_synapses_per_branch=ie_synapses_per_branch,
                reactivate=reactivate,
                reactivation_init_m=reactivation_init_m,
                reactivation_init_b=reactivation_init_b,
                somatic_synapses=somatic_synapses,
                topk_init_method=topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
                topk_noise_level=topk_noise_level,
                topk_strategy=topk_strategy,
                topk_type=topk_type,
                print_hooks=print_hooks,
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
        else:
            self.excitatory_cells = DendriNet(
                n_soma=n_excitatory_cells,
                branch_factors=excitatory_branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=ee_synapses_per_branch,
                inhibitory_input_dim=n_inhibitory_cells
                if self.has_inhib_subnet
                else None,
                inhibitory_synapses_per_branch=ie_synapses_per_branch,
                reactivate=reactivate,
                reactivation_init_m=reactivation_init_m,
                reactivation_init_b=reactivation_init_b,
                somatic_synapses=somatic_synapses,
                topk_init_method=topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
                topk_noise_level=topk_noise_level,
                topk_strategy=topk_strategy,
                topk_type=topk_type,
                print_hooks=print_hooks,
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

    def decay_weights(self, weight_decay):
        if self.inhibitory_cells is not None:
            self.inhibitory_cells.decay_weights(weight_decay)
        self.excitatory_cells.decay_weights(weight_decay)

    def forward(self, x, inhibitory_input=None):
        """
        x: shape [batch, excitatory_input_dim]
        inhibitory_input: shape [batch, inhibitory_input_dim], or None
        Returns:
            excitatory_output, inhibitory_output
        """
        if self.has_inhib_subnet:
            inhibitory_output = self.inhibitory_cells(x, inhibitory_input)
        else:
            # If no inhibitory subnet, just pass through the inhibitory input
            inhibitory_output = inhibitory_input

        # Pass the inhibitory output to excitatory cells
        # The DendriNet architecture handles the proper connections internally
        # through the inhibitory_synapses_per_branch parameter
        excitatory_output = self.excitatory_cells(x, inhibitory_output)

        return excitatory_output, inhibitory_output

    def forward_with_branch_outputs(self, x, inhibitory_input=None):
        if self.inhibitory_cells is not None and hasattr(
            self.inhibitory_cells, "forward_with_branch_outputs"
        ):
            (
                inhibitory_final,
                inh_acts,
            ) = self.inhibitory_cells.forward_with_branch_outputs(x, inhibitory_input)
        elif self.inhibitory_cells is not None:
            inhibitory_final = self.inhibitory_cells(x, inhibitory_input)
            inh_acts = []
        else:
            # When there are no inhibitory cells, pass the original inhibitory_input
            inhibitory_final = inhibitory_input
            inh_acts = []

        # Additional safety checks for inhibitory_final
        if not isinstance(inhibitory_final, torch.Tensor):
            # Set to None so it gets properly handled in excitatory_cells
            inhibitory_final = None
        elif inhibitory_final.dim() == 0:
            # If it's a scalar tensor, set to None
            inhibitory_final = None
        elif inhibitory_final.dim() == 1:
            # If it's a 1D tensor, make it 2D [1, features]
            inhibitory_final = inhibitory_final.unsqueeze(0)

        # Ensure x is a properly formed 2D tensor
        if x.dim() == 1:
            x = x.unsqueeze(0)

        excitatory_final, exc_acts = self.excitatory_cells.forward_with_branch_outputs(
            x, inhibitory_final
        )

        # Enhance the activation information for better visualization
        # Try to enhance the excitatory activations with information about inhibitory components
        enhanced_acts = []
        for i, act in enumerate(exc_acts):
            if isinstance(act, torch.Tensor):
                # Get corresponding inhibitory activation if available
                inh_act = inh_acts[i] if i < len(inh_acts) else None

                # Create a list containing both excitatory and inhibitory information
                if inh_act is not None:
                    # Include both components plus a placeholder for combined output
                    enhanced_acts.append([act, inh_act, None])
                else:
                    enhanced_acts.append(act)
            else:
                enhanced_acts.append(act)

        return excitatory_final, inhibitory_final, inh_acts, enhanced_acts


###############################################################################
# ExcitationInhibitionNetwork
###############################################################################
class ExcitationInhibitionNetwork(nn.Module):
    """
    A neural network model that implements excitatory and inhibitory pathways.
    This is a placeholder implementation based on common patterns.
    """

    def __init__(
        self,
        input_dim: int,
        excitatory_layer_sizes: List[int],
        inhibitory_layer_sizes: List[int],
        excitatory_branch_factors: List[int],
        inhibitory_branch_factors: List[int],
        output_dim: int = 10,
        reactivate: bool = True,
        reactivation_init_m: float = 1.5,
        reactivation_init_b: float = 0.5,
        somatic_synapses: bool = True,
        ee_synapses_per_branch_per_layer: List[int] = None,
        ei_synapses_per_branch_per_layer: List[int] = None,
        ie_synapses_per_branch_per_layer: List[int] = None,
        ii_synapses_per_branch_per_layer: List[int] = None,
        use_shunting: bool = True,
        reactivation_type: str = "param_tanh",
        reactivation_strategy: str = "inverse",
        blocklinear_strategy: str = "conductance_dynamic",
        enable_branch_outputs: bool = False,
        output_layer: bool = False,
        input_mode: int = 0,
        input_net_dict: Dict[str, Any] = None,
        learning_strategy: str = "mle",
        topk_init_method: str = "xavier_normal",
        topk_strategy: str = "conductance_dynamic",
        topk_type: str = "standard",
        print_hooks: bool = False,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode="global",
        synapses_per_branch=None,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method="xavier_normal",
        weight_threshold=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.excitatory_layer_sizes = excitatory_layer_sizes
        self.inhibitory_layer_sizes = inhibitory_layer_sizes
        self.excitatory_branch_factors = excitatory_branch_factors
        self.inhibitory_branch_factors = inhibitory_branch_factors
        self.reactivate = reactivate
        self.reactivation_init_m = reactivation_init_m
        self.reactivation_init_b = reactivation_init_b
        self.somatic_synapses = somatic_synapses
        self.ee_synapses_per_branch_per_layer = (
            ee_synapses_per_branch_per_layer if ee_synapses_per_branch_per_layer else []
        )
        self.ei_synapses_per_branch_per_layer = (
            ei_synapses_per_branch_per_layer if ei_synapses_per_branch_per_layer else []
        )
        self.ie_synapses_per_branch_per_layer = (
            ie_synapses_per_branch_per_layer if ie_synapses_per_branch_per_layer else []
        )
        self.ii_synapses_per_branch_per_layer = (
            ii_synapses_per_branch_per_layer if ii_synapses_per_branch_per_layer else []
        )
        self.use_shunting = use_shunting
        self.reactivation_type = reactivation_type
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.enable_branch_outputs = enable_branch_outputs
        self.output_layer = output_layer
        self.input_mode = input_mode
        self.input_net_dict = input_net_dict if input_net_dict else {}
        self.learning_strategy = learning_strategy
        self.topk_init_method = topk_init_method
        self.topk_strategy = topk_strategy
        self.topk_type = topk_type
        self.topk_noise_level = kwargs.get("topk_noise_level", 0.0)
        self.print_hooks = print_hooks
        self.excitatory_target_density = excitatory_target_density
        self.inhibitory_target_density = inhibitory_target_density
        self.use_noise = use_noise
        self.sigma = sigma
        self.rewiring_mode = rewiring_mode
        self.synapses_per_branch = synapses_per_branch
        self.freeze_excitatory_connectivity = freeze_excitatory_connectivity
        self.freeze_inhibitory_connectivity = freeze_inhibitory_connectivity
        self.init_method = init_method
        self.weight_threshold = weight_threshold

        # Set up input transformation based on input_mode
        if input_mode == 0:
            # Original mode - raw input
            self.input_transform = nn.Identity()
        elif input_mode == 1:
            # Simple mode - input goes to both exc and inh pathways
            self.input_transform = lambda x: (x, x)
        elif input_mode == 2:
            # Network mode - transform input through a neural network
            if not input_net_dict:
                raise ValueError("input_net_dict must be provided when input_mode=2")
            self.setup_input_network()
        else:
            raise ValueError(f"Unknown input_mode: {input_mode}")

        # Set up network layers
        self.setup_network_layers()

        # Set up output layer if needed
        if self.output_layer:
            self.setup_output_layer()

        # Save branch activations if enabled
        self.branch_activations = [] if enable_branch_outputs else None

    def setup_input_network(self):
        """Set up the input transformation network based on input_net_dict"""
        # This would create the input network based on the configuration
        # Placeholder implementation
        self.input_net = InputNetTransform(
            input_dim=self.input_dim,
            hidden_dims=self.input_net_dict.get("hidden_dims", []),
            excitatory_dim=self.input_net_dict.get("excitatory_dim", 128),
            inhibitory_dim=self.input_net_dict.get("inhibitory_dim", 128),
            input_shape=self.input_net_dict.get("input_shape", None),
            excitatory_hidden_dims=self.input_net_dict.get("excitatory_hidden_dims", []),
            inhibitory_hidden_dims=self.input_net_dict.get("inhibitory_hidden_dims", []),
            excitatory_kernel_sizes=self.input_net_dict.get("excitatory_kernel_sizes", []),
            inhibitory_kernel_sizes=self.input_net_dict.get("inhibitory_kernel_sizes", []),
            excitatory_strides=self.input_net_dict.get("excitatory_strides", []),
            inhibitory_strides=self.input_net_dict.get("inhibitory_strides", []),
            network_type=self.input_net_dict.get("type", "MLPInputTransform")
        )

    def setup_network_layers(self):
        """Set up the excitatory and inhibitory network layers"""
        # Create network layers using ExcitationInhibitionLayer
        layers = []

        # Setup the lists we need
        excitatory_layer_sizes = deepcopy(self.excitatory_layer_sizes)
        inhibitory_layer_sizes = deepcopy(self.inhibitory_layer_sizes)

        # Insert input_dim at the beginning
        excitatory_layer_sizes.insert(0, self.input_dim)
        inhibitory_layer_sizes.insert(0, None)

        # Set up synapses per branch per layer
        ee_syn = list(self.ee_synapses_per_branch_per_layer)
        ei_syn = list(self.ei_synapses_per_branch_per_layer)
        ie_syn = list(self.ie_synapses_per_branch_per_layer)
        ii_syn = list(self.ii_synapses_per_branch_per_layer)

        # Extend lists if needed
        num_layers = len(excitatory_layer_sizes) - 1
        while len(ee_syn) < num_layers:
            ee_syn.append(ee_syn[-1] if ee_syn else 1)
        while len(ei_syn) < num_layers:
            ei_syn.append(ei_syn[-1] if ei_syn else 1)
        while len(ie_syn) < num_layers:
            ie_syn.append(ie_syn[-1] if ie_syn else 1)
        while len(ii_syn) < num_layers:
            ii_syn.append(ii_syn[-1] if ii_syn else 1)

        # Initialize input_net if using input_mode=2
        if self.input_mode == 2 and self.input_net is not None:
            input_net_exc_dim = self.input_net.input_transform.excitatory_dim
            excitatory_layer_sizes[0] = input_net_exc_dim

        # Build each layer
        for i in range(num_layers):
            n_exc = excitatory_layer_sizes[i + 1]

            # Decide n_inh for layer i based on input_mode
            if i == 0:
                if self.input_mode == 0:
                    # input_mode=0 => normal approach => use config's inhibitory_layer_sizes[i+1]
                    n_inh = inhibitory_layer_sizes[i + 1]
                elif self.input_mode in (1, 2):
                    # skip building inhibitory net in the FIRST layer => set n_inh=0
                    n_inh = 0
                else:
                    raise ValueError(f"Unknown input_mode={self.input_mode}")
            else:
                # from 2nd layer onward, use the config's inhibitory_layer_sizes
                n_inh = inhibitory_layer_sizes[i + 1]

            # Determine inhibitory input size
            inhibitory_input_size = inhibitory_layer_sizes[i]
            if i == 0 and self.input_mode == 2:
                # For input_mode=2, use net_out_dim as inhibitory_input_dim
                inhibitory_input_size = self.input_net.input_transform.inhibitory_dim

            # Create ExcitationInhibitionLayer
            layer = ExcitationInhibitionLayer(
                n_excitatory_cells=n_exc,
                n_inhibitory_cells=n_inh,
                excitatory_branch_factors=self.excitatory_branch_factors,
                inhibitory_branch_factors=self.inhibitory_branch_factors,
                excitatory_input_dim=excitatory_layer_sizes[i],
                ee_synapses_per_branch=ee_syn[i],
                ei_synapses_per_branch=(
                    min(ei_syn[i], inhibitory_input_size)
                    if i == 0 and self.input_mode == 2 and inhibitory_input_size
                    else ei_syn[i]
                ),
                inhibitory_input_dim=inhibitory_input_size,
                ie_synapses_per_branch=ie_syn[i],
                ii_synapses_per_branch=ii_syn[i],
                reactivate=self.reactivate,
                reactivation_init_m=self.reactivation_init_m,
                reactivation_init_b=self.reactivation_init_b,
                somatic_synapses=self.somatic_synapses,
                topk_init_method=self.topk_init_method,
                use_shunting=self.use_shunting,
                reactivation_strategy=self.reactivation_strategy,
                blocklinear_strategy=self.blocklinear_strategy,
                reactivation_type=self.reactivation_type,
                enable_branch_outputs=self.enable_branch_outputs,
                topk_noise_level=self.topk_noise_level,
                topk_strategy=self.topk_strategy,
                topk_type=self.topk_type,
                print_hooks=self.print_hooks,
                excitatory_target_density=self.excitatory_target_density,
                inhibitory_target_density=self.inhibitory_target_density,
                use_noise=self.use_noise,
                sigma=self.sigma,
                rewiring_mode=self.rewiring_mode,
                synapses_per_branch=self.synapses_per_branch,
                freeze_excitatory_connectivity=self.freeze_excitatory_connectivity,
                freeze_inhibitory_connectivity=self.freeze_inhibitory_connectivity,
                init_method=self.init_method,
                weight_threshold=self.weight_threshold
            )
            layers.append(layer)

        # Create ModuleList with the layers
        self.layers = nn.ModuleList(layers)

    def setup_output_layer(self):
        """Set up the output classification layer"""
        final_exc_size = (
            self.excitatory_layer_sizes[-1] if self.excitatory_layer_sizes else 0
        )
        self.output_classifier = nn.Linear(final_exc_size, self.output_dim)

    def decay_weights(self, weight_decay):
        for layer in self.layers:
            layer.decay_weights(weight_decay)

    def forward(self, x):
        """
        Forward logic depends on input_mode:
        - mode=0 => excit=x, inhibitory=None at the first layer
        - mode=1 => excit=x, inhibitory=x at the first layer, but we skip building that layer's inhib net => no separate output
        - mode=2 => pass x into self.input_net => produce (excit, inhib)
                    skip that layer's inhib net => no separate output
        Then subsequent layers do normal E–I if their inhib cells exist.
        """
        # Handle the input based on input_mode
        if self.input_mode == 0:
            excitatory_input = x
            inhibitory_input = None
        elif self.input_mode == 1:
            excitatory_input = x
            inhibitory_input = x
        elif self.input_mode == 2:
            if self.input_net is not None:
                excitatory_input, inhibitory_input = self.input_net(x)
            else:
                # Fallback if input_net is not initialized
                excitatory_input = x
                inhibitory_input = None
        else:
            raise ValueError(f"Unknown input_mode={self.input_mode}")

        # Forward through each layer
        excitatory_output, inhibitory_output = self.layers[0](
            excitatory_input, inhibitory_input
        )

        for i in range(1, len(self.layers)):
            # Each layer receives the previous layer's outputs
            excitatory_output, inhibitory_output = self.layers[i](
                excitatory_output, inhibitory_output
            )

        # Apply output layer if enabled
        if self.output_layer and hasattr(self, "output_classifier"):
            output = self.output_classifier(excitatory_output)
        else:
            output = excitatory_output

        # Store branch activations if enabled
        if self.enable_branch_outputs:
            self.branch_activations = {
                "excitatory": [excitatory_output],
                "inhibitory": [inhibitory_output]
                if inhibitory_output is not None
                else [],
            }

        return output

    def get_branch_activations(self):
        """Return branch activations if enabled"""
        if not self.enable_branch_outputs:
            raise ValueError(
                "Branch outputs not enabled. Set enable_branch_outputs=True to use this method."
            )
        return self.branch_activations

    def forward_with_branch_outputs(self, x):
        """Forward pass that extracts activations from all layers.

        Args:
            x: Input tensor

        Returns:
            tuple: (output, per_layer_activations)
                Where per_layer_activations is a list of activation tensors or lists
                from each layer in the network.
        """
        # Apply input transformations and create appropriate pathways
        if self.input_mode == 0:
            excitatory_input = x
            inhibitory_input = None
        elif self.input_mode == 1:
            excitatory_input = x
            inhibitory_input = x
        elif self.input_mode == 2:
            if self.input_net is not None:
                excitatory_input, inhibitory_input = self.input_net(x)
            else:
                # Fallback if input_net is not initialized
                excitatory_input = x
                inhibitory_input = None
        else:
            raise ValueError(f"Unknown input_mode={self.input_mode}")

        # Use forward_with_branch_outputs for each layer
        excitatory_output, inhibitory_output, inh_acts, exc_acts = self.layers[
            0
        ].forward_with_branch_outputs(excitatory_input, inhibitory_input)

        # Process the enhanced activations to create a list of activation tensors or lists
        # that contain information about both excitatory and inhibitory components
        all_acts = []
        for act in exc_acts:
            if isinstance(act, list) and len(act) > 1:
                # This is already an enhanced activation with [exc, inh, output] format
                all_acts.append(act)
            else:
                # Just use the excitatory activation as is
                all_acts.append(act)

        for i in range(1, len(self.layers)):
            (
                excitatory_output,
                inhibitory_output,
                layer_inh_acts,
                layer_exc_acts,
            ) = self.layers[i].forward_with_branch_outputs(
                excitatory_output, inhibitory_output
            )

            # Process this layer's activations similarly
            for act in layer_exc_acts:
                if isinstance(act, list) and len(act) > 1:
                    # Enhanced activation
                    all_acts.append(act)
                else:
                    # Just excitatory
                    all_acts.append(act)

        # Apply output layer if enabled - check if it's actually a callable object not a boolean
        output = excitatory_output
        if hasattr(self, "output_layer") and self.output_layer is not None:
            if isinstance(self.output_layer, bool):
                # If output_layer is just a boolean flag, don't try to call it
                pass  # Just use excitatory_output as is
            elif callable(self.output_layer):
                # If it's a callable module, use it
                if hasattr(self.output_layer, "forward_with_activations"):
                    output, out_acts = self.output_layer.forward_with_activations(
                        excitatory_output
                    )
                    all_acts.extend(out_acts)
                else:
                    try:
                        output = self.output_layer(excitatory_output)
                    except Exception as e:
                        logger.warning(
                            f"Could not call output_layer (type: {type(self.output_layer)}): {e}"
                        )
            else:
                logger.warning(
                    f"output_layer is type {type(self.output_layer)}, not callable or boolean"
                )

        return output, all_acts

    def compute_local_loss(self, x, y=None):
        return 0.0

    @property
    def branch_layers(self):
        all_branch_layers = []
        for ei_layer in self.layers:
            all_branch_layers.extend(ei_layer.excitatory_cells.branch_layers)
            all_branch_layers.extend(ei_layer.inhibitory_cells.branch_layers)
        return all_branch_layers

    @property
    def n_branch_layers(self):
        return len(self.branch_layers)

    def load_input_net_weights(self, input_net_weights):
        """
        Load weights for the input network component.

        Args:
            input_net_weights: State dict or model with state_dict() method
        """
        if self.input_mode != 2 or self.input_net is None:
            logger.warning(
                "Network not configured for input network weight loading (input_mode != 2 or no input_net)"
            )
            return

        logger.info("Loading input network weights")
        if hasattr(input_net_weights, "state_dict"):
            state_dict = input_net_weights.state_dict()
        else:
            # Assuming input_net_weights is already a state_dict
            state_dict = input_net_weights  # Correct assignment

        # Original load call
        self.input_net.load_state_dict(state_dict)


############################################################
# MLPExcInhNetwork
############################################################
class MLPExcInhLayer(nn.Module):
    def __init__(self, in_excit, in_inhib, out_features, activation=None):
        super().__init__()
        self.in_excit = int(in_excit)
        self.in_inhib = int(in_inhib)
        self.out_features = int(out_features)
        if self.in_excit > 0:
            self.excit_pre_w = nn.Parameter(
                torch.zeros(self.out_features, self.in_excit)
            )
            nn.init.xavier_normal_(self.excit_pre_w)
        else:
            self.excit_pre_w = None
        if self.in_inhib > 0:
            self.inhib_pre_w = nn.Parameter(
                torch.zeros(self.out_features, self.in_inhib)
            )
            nn.init.xavier_normal_(self.inhib_pre_w)
        else:
            self.inhib_pre_w = None
        self.bias = nn.Parameter(torch.zeros(self.out_features))
        self.activation = activation

    def decay_weights(self, weight_decay):
        with torch.no_grad():
            if self.excit_pre_w is not None:
                self.excit_pre_w.sub_(weight_decay * self.excit_pre_w)
            if self.inhib_pre_w is not None:
                self.inhib_pre_w.sub_(weight_decay * self.inhib_pre_w)
            self.bias.sub_(weight_decay * self.bias)

    def forward(self, x):
        batch_size = x.size(0)
        if self.in_excit > 0 and self.in_inhib > 0:
            x_excit = x[:, : self.in_excit]
            x_inhib = x[:, self.in_excit : self.in_excit + self.in_inhib]
        elif self.in_excit > 0:
            x_excit = x
            x_inhib = None
        else:
            x_excit = None
            x_inhib = x
        out = torch.zeros(batch_size, self.out_features, device=x.device, dtype=x.dtype)
        if self.excit_pre_w is not None and x_excit is not None:
            w_excit = self.excit_pre_w.exp()
            out += x_excit.matmul(w_excit.t())
        if self.inhib_pre_w is not None and x_inhib is not None:
            w_inhib = -self.inhib_pre_w.exp()
            out += x_inhib.matmul(w_inhib.t())
        out += self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out


class MLPExcInhNetwork(nn.Module):
    """
    A simple MLP network with separate excitatory and inhibitory pathways.
    This is a placeholder implementation.
    """

    def __init__(
        self, input_dim: int, hidden_dims: List[int], output_dim: int = 10, **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        # Create the network
        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            in_excit = layer.in_excit
            in_inhib = layer.in_inhib
            if x.size(1) < in_excit + in_inhib:
                pad_size = (in_excit + in_inhib) - x.size(1)
                x = torch.cat(
                    [x, torch.zeros(x.size(0), pad_size, device=x.device)], dim=1
                )
            x = layer(x)
        if self.classifier is not None:
            x = self.classifier(x)
        return x


class MatchedParamMLP(nn.Module):
    """
    Creates an MLP with at least as many parameters as an equivalent EINet configuration.
    Matches the number of layers to DendriticBranchLayers and minimizes parameters while
    exceeding EINet parameter count.
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
        ii_synapses_per_branch_per_layer=[],
        reactivate=False,
        somatic_synapses=True,
        output_layer=False,
        output_dim=10,
        **kwargs,  # Accept but ignore other EINet parameters
    ):
        super(MatchedParamMLP, self).__init__()

        # Create EINet to get parameter count and number of layers
        einet = ExcitationInhibitionNetwork(
            input_dim=input_dim,
            excitatory_layer_sizes=excitatory_layer_sizes,
            inhibitory_layer_sizes=inhibitory_layer_sizes,
            excitatory_branch_factors=excitatory_branch_factors,
            inhibitory_branch_factors=inhibitory_branch_factors,
            ee_synapses_per_branch_per_layer=ee_synapses_per_branch_per_layer,
            ei_synapses_per_branch_per_layer=ei_synapses_per_branch_per_layer,
            ie_synapses_per_branch_per_layer=ie_synapses_per_branch_per_layer,
            ii_synapses_per_branch_per_layer=ii_synapses_per_branch_per_layer,
            reactivate=reactivate,
            somatic_synapses=somatic_synapses,
            output_layer=output_layer,
            output_dim=output_dim,
            **kwargs
        )

        summary = torchinfo.summary(einet, verbose=0)
        target_params = summary.trainable_params

        # Get number of branch layers
        n_branch_layers = einet.n_branch_layers
        if not somatic_synapses:
            n_branch_layers -= 2 * len(einet.layers)  # Account for somatic layer

        # Calculate required hidden layer widths
        # For n hidden layers (n+1 total layers), we need:
        # input_dim * h1 + h1 * h2 + h2 * h3 + ... + hn * output_dim ≥ target_params
        # Minimize total parameters while satisfying this constraint
        # We'll use equal width hidden layers for simplicity
        # Then: input_dim * h + h * h * (n-1) + h * output_dim ≥ target_params
        # where n is number of hidden layers

        def get_param_count(hidden_width):
            """Calculate total parameters for given hidden width"""
            params = input_dim * hidden_width  # First layer
            params += (
                hidden_width * hidden_width * (n_branch_layers - 1)
            )  # Hidden layers
            params += hidden_width * output_dim  # Output layer
            return params

        # Binary search for minimum hidden width that exceeds target params
        left, right = 1, 10000
        min_width = right

        while left <= right:
            mid = (left + right) // 2
            params = get_param_count(mid)

            if params >= target_params:
                min_width = mid
                right = mid - 1
            else:
                left = mid + 1

        hidden_width = min_width

        # Build MLP layers
        layers = []

        # Input layer
        layers.extend([nn.Linear(input_dim, hidden_width), nn.ReLU()])

        # Hidden layers
        for _ in range(n_branch_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.ReLU()])

        # Output layer
        layers.append(nn.Linear(hidden_width, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SimpleMLPLayer(nn.Module):
    """
    A single layer of a simple MLP with configurable activation function.
    Supports all activation functions from ActivationFactory.
    """
    def __init__(self, in_features, out_features, activation_type=None, trainable_activation=True, 
                 activation_wrapper=None, positive_weights=False, init_m=4.0, use_batch_norm=False,
                 weight_init="kaiming_uniform"):  # Updated default to be more explicit
        super(SimpleMLPLayer, self).__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.init_m = init_m
        self.use_batch_norm = use_batch_norm
        
        # Linear layer
        self.linear = nn.Linear(self.in_features, self.out_features)

        # Initialize weights using selected method
        weight_init = weight_init.lower()  # Normalize to lowercase for comparison
        
        # Kaiming/He initialization variants
        if weight_init == "kaiming_normal" or weight_init == "kaiming" or weight_init == "he_normal":
            nn.init.kaiming_normal_(self.linear.weight)
        elif weight_init == "kaiming_uniform" or weight_init == "he_uniform":
            nn.init.kaiming_uniform_(self.linear.weight)
            
        # Xavier/Glorot initialization variants
        elif weight_init == "xavier_normal" or weight_init == "xavier" or weight_init == "glorot_normal":
            nn.init.xavier_normal_(self.linear.weight)
        elif weight_init == "xavier_uniform" or weight_init == "glorot_uniform":
            nn.init.xavier_uniform_(self.linear.weight)
            
        # Standard normal and uniform initialization
        elif weight_init == "normal":
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        elif weight_init == "uniform":
            nn.init.uniform_(self.linear.weight, -0.01, 0.01)
            
        # Default to kaiming_normal if unrecognized
        else:
            nn.init.kaiming_uniform_(self.linear.weight)
            
        # Scale weights to [0,1] if positive_weights is True
        if positive_weights:
            with torch.no_grad():
                self.linear.weight.data = (self.linear.weight.data + 1) / 2
                
        nn.init.zeros_(self.linear.bias)
        
        # Batch normalization layer
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(self.out_features)
        
        # Activation function from factory
        if activation_type is not None:
            self.activation = ActivationFactory.create(
                act_type=activation_type,
                output_dim=self.out_features,
                init_m=self.init_m,
                init_b=2.0,
                trainable=trainable_activation,
                activation_wrapper=activation_wrapper
            )
        else:
            self.activation = None

    def decay_weights(self, weight_decay):
        """Apply weight decay to linear layer."""
        if weight_decay is not None:
            with torch.no_grad():
                self.linear.weight.sub_(weight_decay * self.linear.weight)
                self.linear.bias.sub_(weight_decay * self.linear.bias)

    def forward(self, x):
        """Forward pass through the layer."""
        x = self.linear(x)
        
        # Apply batch normalization if enabled
        if self.use_batch_norm:
            x = self.batch_norm(x)
            
        # Apply activation if present
        if self.activation is not None:
            x = self.activation(x)
            
        return x


class SimpleMLPNet(nn.Module):
    """
    A modular MLP implementation that uses SimpleMLPLayer.
    Supports all activation functions from ActivationFactory.
    """
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        nb_layers = 2,  # default to 2-layer network
        activation_type="param_linear_sigmoid",
        weight_decay_rate=None,
        learning_strategy="regression",
        trainable_activation=True,
        grad_clip_value=1.0,
        activation_wrapper=None,
        init_m=4.0,
        seed=None,
        use_batch_norm=False,  # Batch normalization parameter
        weight_init="kaiming_uniform",  # Updated default value with more options
        **kwargs
    ):
        super(SimpleMLPNet, self).__init__()
        self.weight_decay_rate = weight_decay_rate
        self.learning_strategy = learning_strategy
        self.input_dim = int(input_dim)
        self.grad_clip_value = grad_clip_value
        self.init_m = init_m
        self.use_batch_norm = use_batch_norm
        self.weight_init = weight_init

        if seed is not None:
            self.set_seed(seed)
            
        # Build multi-layer network
        layers = []
        
        # First layer: input_dim -> hidden_dim
        layers.append(SimpleMLPLayer(
            in_features=input_dim,
            out_features=hidden_dim,
            activation_type=activation_type,
            trainable_activation=trainable_activation,
            activation_wrapper=activation_wrapper,
            init_m=self.init_m,
            use_batch_norm=use_batch_norm,
            weight_init=weight_init
        ))
        
        # Hidden layers: hidden_dim -> hidden_dim
        for _ in range(nb_layers - 2):
            layers.append(SimpleMLPLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                activation_type=activation_type,
                trainable_activation=trainable_activation,
                activation_wrapper=activation_wrapper,
                init_m=self.init_m,
                use_batch_norm=use_batch_norm,
                weight_init=weight_init
            ))
            
        # Output layer: hidden_dim -> output_dim (no activation)
        layers.append(SimpleMLPLayer(
            in_features=hidden_dim,
            out_features=output_dim,
            activation_type=None,
            trainable_activation=trainable_activation,
            activation_wrapper=activation_wrapper,
            init_m=self.init_m,
            use_batch_norm=False,  # No batch norm on output layer
            weight_init=weight_init
        ))
        
        self.layers = nn.ModuleList(layers)

    def decay_weights(self, weight_decay=None):
        """Apply weight decay to all layers."""
        if weight_decay is not None:
            weight_decay = self.weight_decay_rate
            for layer in self.layers:
                layer.decay_weights(weight_decay)
    
    def clip_gradients(self):
        """Apply gradient clipping to all parameters."""
        if self.grad_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_clip_value)
                
    def forward(self, x):
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x)
        return x
    
    def set_seed(self, seed):
        """Set the seed for the network."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @property
    def branch_layers(self):
        """Compatibility with existing codebase expectations."""
        return []

    @property
    def n_branch_layers(self):
        """Compatibility with existing codebase expectations."""
        return 0
    
