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
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

from dendritic_modeling.dendrinet import DendriNet, DendriNetWithOutputs
from dendritic_modeling import logger
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
    def __init__(self, input_dim, hidden_dims, output_dim, network_type="MLPInputTransform"):
        super().__init__()
        self.network_type = network_type
        
        if network_type == "MLPInputTransform":
            # Build MLP backbone
            layers = []
            in_features = input_dim
            for hdim in hidden_dims:
                layers.append(nn.Linear(in_features, hdim))
                layers.append(nn.ReLU())
                in_features = hdim
            self.shared = nn.Sequential(*layers)
            
            # Create excitatory and inhibitory heads
            self.head_exc = nn.Sequential(nn.Linear(in_features, output_dim), nn.Sigmoid())
            self.head_inh = nn.Sequential(nn.Linear(in_features, output_dim), nn.Sigmoid())
            
            # Initialize weights using Kaiming initialization
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        elif network_type == "AlexNetTransform":
            # Example of using a pre-trained model like AlexNet
            import torchvision.models as models
            
            # Load AlexNet without classifier
            alexnet = models.alexnet(pretrained=True)
            self.shared = nn.Sequential(*list(alexnet.features), 
                                       nn.AdaptiveAvgPool2d((6, 6)),
                                       nn.Flatten())
            
            # Create new heads for excitatory and inhibitory outputs
            self.head_exc = nn.Sequential(nn.Linear(9216, output_dim), nn.Sigmoid())
            self.head_inh = nn.Sequential(nn.Linear(9216, output_dim), nn.Sigmoid())
            
        else:
            raise ValueError(f"Unknown network type: {network_type}")

    def forward(self, x):
        x = self.shared(x)
        excitatory_input = self.head_exc(x)
        inhibitory_input = self.head_inh(x)
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
                raise ValueError("pytorch_model must be provided when source is 'pytorch'")
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
MLPInputTransform = InputNetTransform


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
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        use_shunting=True,
        reactivation_strategy="none",
        blocklinear_strategy="none",
        reactivation_type='param_tanh',
        enable_branch_outputs=False
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
                    reactivation_strategy=reactivation_strategy,
                    blocklinear_strategy=blocklinear_strategy,
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
                    somatic_synapses=somatic_synapses,
                    topk_init_method=topk_init_method,
                    use_shunting=use_shunting,
                    reactivation_type=reactivation_type,
                    reactivation_strategy=reactivation_strategy,
                    blocklinear_strategy=blocklinear_strategy,
                )
        else:
            self.inhibitory_cells = None

        if enable_branch_outputs:
            self.excitatory_cells = DendriNetWithOutputs(
                n_soma=n_excitatory_cells,
                branch_factors=excitatory_branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=ee_synapses_per_branch,
                inhibitory_input_dim=n_inhibitory_cells if self.has_inhib_subnet else None,
                inhibitory_synapses_per_branch=ie_synapses_per_branch,
                reactivate=reactivate,
                somatic_synapses=somatic_synapses,
                topk_init_method=topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
            )
        else:
            self.excitatory_cells = DendriNet(
                n_soma=n_excitatory_cells,
                branch_factors=excitatory_branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=ee_synapses_per_branch,
                inhibitory_input_dim=n_inhibitory_cells if self.has_inhib_subnet else None,
                inhibitory_synapses_per_branch=ie_synapses_per_branch,
                reactivate=reactivate,
                somatic_synapses=somatic_synapses,
                topk_init_method=topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
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
        if self.inhibitory_cells is not None:
            inhibitory_output = self.inhibitory_cells(x, inhibitory_input)
        else:
            # When inhibitory_cells is None but we still need to pass something
            # to the excitatory cells, we pass the original inhibitory_input.
            # This ensures compatibility with somatic_synapses=False
            inhibitory_output = inhibitory_input

        # Additional safety checks for inhibitory_output
        if not isinstance(inhibitory_output, torch.Tensor):
            # Set to None so it gets properly handled in excitatory_cells
            inhibitory_output = None
        elif inhibitory_output.dim() == 0:
            # If it's a scalar tensor, set to None
            inhibitory_output = None
        elif inhibitory_output.dim() == 1:
            # If it's a 1D tensor, make it 2D [1, features]
            inhibitory_output = inhibitory_output.unsqueeze(0)
            
        # Ensure x is a properly formed 2D tensor
        if x.dim() == 1:
            x = x.unsqueeze(0)

        excitatory_output = self.excitatory_cells(x, inhibitory_output)
        return excitatory_output, inhibitory_output

    def forward_with_branch_outputs(self, x, inhibitory_input=None):
        if self.inhibitory_cells is not None and hasattr(self.inhibitory_cells, 'forward_with_branch_outputs'):
            inhibitory_final, inh_acts = self.inhibitory_cells.forward_with_branch_outputs(x, inhibitory_input)
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

        excitatory_final, exc_acts = self.excitatory_cells.forward_with_branch_outputs(x, inhibitory_final)
        return excitatory_final, inhibitory_final, inh_acts, exc_acts


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
        somatic_synapses: bool = True,
        ee_synapses_per_branch_per_layer: List[int] = None,
        ei_synapses_per_branch_per_layer: List[int] = None,
        ie_synapses_per_branch_per_layer: List[int] = None,
        ii_synapses_per_branch_per_layer: List[int] = None,
        use_shunting: bool = True,
        reactivation_type: str = "param_tanh",
        reactivation_strategy: str = "inverse",
        blocklinear_strategy: str = "block_conductance_dynamic",
        enable_branch_outputs: bool = False,
        output_layer: bool = False,
        input_mode: int = 0,
        input_net_dict: Dict[str, Any] = None,
        learning_strategy: str = "mle",
        topk_init_method: str = "xavier_normal",
        **kwargs
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.excitatory_layer_sizes = excitatory_layer_sizes
        self.inhibitory_layer_sizes = inhibitory_layer_sizes
        self.excitatory_branch_factors = excitatory_branch_factors
        self.inhibitory_branch_factors = inhibitory_branch_factors
        self.reactivate = reactivate
        self.somatic_synapses = somatic_synapses
        self.ee_synapses_per_branch_per_layer = ee_synapses_per_branch_per_layer if ee_synapses_per_branch_per_layer else []
        self.ei_synapses_per_branch_per_layer = ei_synapses_per_branch_per_layer if ei_synapses_per_branch_per_layer else []
        self.ie_synapses_per_branch_per_layer = ie_synapses_per_branch_per_layer if ie_synapses_per_branch_per_layer else []
        self.ii_synapses_per_branch_per_layer = ii_synapses_per_branch_per_layer if ii_synapses_per_branch_per_layer else []
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
            output_dim=self.input_net_dict.get("output_dim", 128),
            network_type=self.input_net_dict.get("type", "MLPInputTransform")
        )
        
    def setup_network_layers(self):
        """Set up the excitatory and inhibitory network layers"""
        # Create excitatory layers with branches
        self.excitatory_layers = nn.ModuleList()
        prev_size = self.input_dim
        
        for i, layer_size in enumerate(self.excitatory_layer_sizes):
            branch_factor = self.excitatory_branch_factors[i] if i < len(self.excitatory_branch_factors) else 1
            # In a real implementation, this would be a specialized layer type
            # For this placeholder, we're using a simple linear layer
            layer = nn.Linear(prev_size, layer_size * branch_factor)
            self.excitatory_layers.append(layer)
            prev_size = layer_size
            
        # Create inhibitory layers with branches
        self.inhibitory_layers = nn.ModuleList()
        prev_size = self.input_dim
        
        for i, layer_size in enumerate(self.inhibitory_layer_sizes):
            branch_factor = self.inhibitory_branch_factors[i] if i < len(self.inhibitory_branch_factors) else 1
            # In a real implementation, this would be a specialized layer type
            layer = nn.Linear(prev_size, layer_size * branch_factor)
            self.inhibitory_layers.append(layer)
            prev_size = layer_size
    
    def setup_output_layer(self):
        """Set up the output classification layer"""
        final_exc_size = self.excitatory_layer_sizes[-1] if self.excitatory_layer_sizes else 0
        self.output_classifier = nn.Linear(final_exc_size, self.output_dim)
        
    def forward(self, x):
        """Forward pass through the network"""
        batch_size = x.shape[0]
        
        # Apply input transformation
        if self.input_mode == 0:
            x_exc = x
            x_inh = x
        elif self.input_mode == 1:
            x_exc, x_inh = x, x
        elif self.input_mode == 2:
            x_exc, x_inh = self.input_net(x)
            
        # Process through excitatory pathway
        exc_activations = []
        for i, layer in enumerate(self.excitatory_layers):
            x_exc = layer(x_exc)
            # Apply activation
            if self.reactivate:
                if self.reactivation_type == "param_tanh":
                    x_exc = torch.tanh(x_exc)
                elif self.reactivation_type == "relu":
                    x_exc = F.relu(x_exc)
                elif self.reactivation_type == "sigmoid":
                    x_exc = torch.sigmoid(x_exc)
                    
            # Store branch activations if enabled
            if self.enable_branch_outputs:
                exc_activations.append(x_exc)
                
        # Process through inhibitory pathway
        inh_activations = []
        for i, layer in enumerate(self.inhibitory_layers):
            x_inh = layer(x_inh)
            # Apply activation
            if self.reactivate:
                if self.reactivation_type == "param_tanh":
                    x_inh = torch.tanh(x_inh)
                elif self.reactivation_type == "relu":
                    x_inh = F.relu(x_inh)
                elif self.reactivation_type == "sigmoid":
                    x_inh = torch.sigmoid(x_inh)
                    
            # Store branch activations if enabled
            if self.enable_branch_outputs:
                inh_activations.append(x_inh)
                
        # Apply shunting inhibition if enabled
        if self.use_shunting and self.inhibitory_layers:
            # This is a simplified implementation
            # In a real model, this would be more complex
            x_exc = x_exc * (1 - torch.sigmoid(x_inh))
            
        # Apply output layer if enabled
        if self.output_layer:
            output = self.output_classifier(x_exc)
        else:
            output = x_exc
            
        # Store branch activations if enabled
        if self.enable_branch_outputs:
            self.branch_activations = {
                'excitatory': exc_activations,
                'inhibitory': inh_activations
            }
            
        return output
        
    def get_branch_activations(self):
        """Return branch activations if enabled"""
        if not self.enable_branch_outputs:
            raise ValueError("Branch outputs not enabled. Set enable_branch_outputs=True to use this method.")
        return self.branch_activations


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
            self.excit_pre_w = nn.Parameter(torch.zeros(self.out_features, self.in_excit))
            nn.init.xavier_normal_(self.excit_pre_w)
        else:
            self.excit_pre_w = None
        if self.in_inhib > 0:
            self.inhib_pre_w = nn.Parameter(torch.zeros(self.out_features, self.in_inhib))
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
            x_excit = x[:, :self.in_excit]
            x_inhib = x[:, self.in_excit:self.in_excit + self.in_inhib]
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
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int = 10,
        **kwargs
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
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        return self.layers(x)