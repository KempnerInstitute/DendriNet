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
import torchinfo
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
# SigmoidInputNetTransform
###############################################################################
class SigmoidInputNetTransform(InputNetTransform):
    """
    Extension of InputNetTransform that applies a sigmoid activation to the output,
    ensuring values are in the range [0, 1] before entering the dendrinet.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, network_type="MLPInputTransform"):
        super().__init__(input_dim, hidden_dims, output_dim, network_type)
        
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
            net_out_dim = self.input_net_dict.get("output_dim", 64)
            excitatory_layer_sizes[0] = net_out_dim
        
        # Build each layer
        for i in range(num_layers):
            n_exc = excitatory_layer_sizes[i+1]
            
            # Decide n_inh for layer i based on input_mode
            if i == 0:
                if self.input_mode == 0:
                    # input_mode=0 => normal approach => use config's inhibitory_layer_sizes[i+1]
                    n_inh = inhibitory_layer_sizes[i+1]
                elif self.input_mode in (1, 2):
                    # skip building inhibitory net in the FIRST layer => set n_inh=0 
                    n_inh = 0
                else:
                    raise ValueError(f"Unknown input_mode={self.input_mode}")
            else:
                # from 2nd layer onward, use the config's inhibitory_layer_sizes
                n_inh = inhibitory_layer_sizes[i+1]
            
            # Determine inhibitory input size
            inhibitory_input_size = inhibitory_layer_sizes[i]
            if i == 0 and self.input_mode == 2:
                # For input_mode=2, use net_out_dim as inhibitory_input_dim
                inhibitory_input_size = self.input_net_dict.get("output_dim", 64)
            
            # Create ExcitationInhibitionLayer
            layer = ExcitationInhibitionLayer(
                n_excitatory_cells=n_exc,
                n_inhibitory_cells=n_inh,
                excitatory_branch_factors=self.excitatory_branch_factors,
                inhibitory_branch_factors=self.inhibitory_branch_factors,
                excitatory_input_dim=excitatory_layer_sizes[i],
                ee_synapses_per_branch=ee_syn[i],
                ei_synapses_per_branch=(min(ei_syn[i], inhibitory_input_size) 
                                       if i == 0 and self.input_mode == 2 and inhibitory_input_size 
                                       else ei_syn[i]),
                inhibitory_input_dim=inhibitory_input_size,
                ie_synapses_per_branch=ie_syn[i],
                ii_synapses_per_branch=ii_syn[i],
                reactivate=self.reactivate,
                somatic_synapses=self.somatic_synapses,
                topk_init_method=self.topk_init_method,
                use_shunting=self.use_shunting,
                reactivation_strategy=self.reactivation_strategy,
                blocklinear_strategy=self.blocklinear_strategy,
                reactivation_type=self.reactivation_type,
                enable_branch_outputs=self.enable_branch_outputs
            )
            layers.append(layer)
        
        # Create ModuleList with the layers
        self.layers = nn.ModuleList(layers)
        
    def setup_output_layer(self):
        """Set up the output classification layer"""
        final_exc_size = self.excitatory_layer_sizes[-1] if self.excitatory_layer_sizes else 0
        self.output_classifier = nn.Linear(final_exc_size, self.output_dim)
        
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
        excitatory_output, inhibitory_output = self.layers[0](excitatory_input, inhibitory_input)

        for i in range(1, len(self.layers)):
            # Each layer receives the previous layer's outputs
            excitatory_output, inhibitory_output = self.layers[i](excitatory_output, inhibitory_output)
        
        # Apply output layer if enabled
        if self.output_layer and hasattr(self, 'output_classifier'):
            output = self.output_classifier(excitatory_output)
        else:
            output = excitatory_output
            
        # Store branch activations if enabled
        if self.enable_branch_outputs:
            self.branch_activations = {
                'excitatory': [excitatory_output],
                'inhibitory': [inhibitory_output] if inhibitory_output is not None else []
            }
            
        return output
        
    def get_branch_activations(self):
        """Return branch activations if enabled"""
        if not self.enable_branch_outputs:
            raise ValueError("Branch outputs not enabled. Set enable_branch_outputs=True to use this method.")
        return self.branch_activations

    def forward_with_branch_outputs(self, x):
        """
        Forward pass that also returns branch activation outputs.
        Used when branch_outputs=True for visualization and analysis.
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

        # Use forward_with_branch_outputs for each layer
        excitatory_output, inhibitory_output, inh_acts, exc_acts = \
            self.layers[0].forward_with_branch_outputs(excitatory_input, inhibitory_input)
        all_acts = exc_acts

        for i in range(1, len(self.layers)):
            excitatory_output, inhibitory_output, inh_acts, exc_acts = \
                self.layers[i].forward_with_branch_outputs(excitatory_output, inhibitory_output)
            all_acts.extend(exc_acts)

        # Apply output layer if enabled
        if self.output_layer and hasattr(self, 'output_classifier'):
            output = self.output_classifier(excitatory_output)
        return excitatory_output, all_acts

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
            logger.warning("Network not configured for input network weight loading (input_mode != 2 or no input_net)")
            return
            
        logger.info("Loading input network weights")
        if hasattr(input_net_weights, 'state_dict'):
            state_dict = input_net_weights.state_dict()
        else:
            output = excitatory_output

        return output, all_acts


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
        for i, layer in enumerate(self.layers):
            in_excit = layer.in_excit
            in_inhib = layer.in_inhib
            if x.size(1) < in_excit + in_inhib:
                pad_size = (in_excit + in_inhib) - x.size(1)
                x = torch.cat([x, torch.zeros(x.size(0), pad_size, device=x.device)], dim=1)
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
        **kwargs  # Accept but ignore other EINet parameters
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
            params += hidden_width * hidden_width * (n_branch_layers - 1)  # Hidden layers
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
        layers.extend([
            nn.Linear(input_dim, hidden_width),
            nn.ReLU()
        ])
        
        # Hidden layers
        for _ in range(n_branch_layers - 1):
            layers.extend([
                nn.Linear(hidden_width, hidden_width),
                nn.ReLU()
            ])
        
        # Output layer
        layers.append(nn.Linear(hidden_width, output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)