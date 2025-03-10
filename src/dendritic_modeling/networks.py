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

from dendritic_modeling.dendrinet import DendriNet, DendriNetWithOutputs
from dendritic_modeling import logger


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
            inhibitory_output = inhibitory_input

        excitatory_output = self.excitatory_cells(x, inhibitory_output)
        return excitatory_output, inhibitory_output

    def forward_with_branch_outputs(self, x, inhibitory_input=None):
        if self.inhibitory_cells is not None and hasattr(self.inhibitory_cells, 'forward_with_branch_outputs'):
            inhibitory_final, inh_acts = self.inhibitory_cells.forward_with_branch_outputs(x, inhibitory_input)
        elif self.inhibitory_cells is not None:
            inhibitory_final = self.inhibitory_cells(x, inhibitory_input)
            inh_acts = []
        else:
            inhibitory_final = None
            inh_acts = []

        if hasattr(self.excitatory_cells, 'forward_with_branch_outputs'):
            excitatory_final, exc_acts = self.excitatory_cells.forward_with_branch_outputs(x, inhibitory_final)
        else:
            excitatory_final = self.excitatory_cells(x, inhibitory_final)
            exc_acts = []

        return excitatory_final, inhibitory_final, inh_acts, exc_acts


###############################################################################
# ExcitationInhibitionNetwork
###############################################################################
class ExcitationInhibitionNetwork(nn.Module):
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
        topk_init_method="xavier_normal",
        use_shunting=True,
        reactivation_strategy="none",
        blocklinear_strategy="none",
        weight_decay_rate=0.1,
        reactivation_type='param_tanh',
        local_loss_weight=0.0,
        enable_branch_outputs=False,
        output_layer=False,
        output_dim=10,
        learning_strategy="mle",
        input_mode=0,                    # 0=original, 1=direct, 2=input_net
        input_net_dict=None,             # Renamed from mlp_transform_dict
    ):
        """
        Changes:
        --------
        1) If input_mode=0: The first-layer inhibitory net sees input=x (the raw input),
           just as in the "original" approach. 
           So we do NOT skip building that inhibitory sub-network in layer0.

        2) If input_mode=1 or 2: skip building inhibitory sub-network ONLY for the FIRST layer,
           but do build it for subsequent layers if inhibitory_layer_sizes[i+1] > 0.

        3) If input_mode=2: also build an input network from input_net_dict that 
           outputs (excit, inhib). Then feed that into the first layer's excit/inhib inputs.
        """
        super().__init__()
        self.weight_decay_rate = weight_decay_rate
        self.n_layers = len(excitatory_layer_sizes)
        self.reactivation_strategy = reactivation_strategy
        self.blocklinear_strategy = blocklinear_strategy
        self.local_loss_weight = local_loss_weight
        self.learning_strategy = learning_strategy
        self.input_mode = input_mode

        logger.debug(f'Building ExcitationInhibitionNetwork with {self.n_layers} layer(s).')
        logger.debug(f'input_mode={self.input_mode}')

        excitatory_layer_sizes = deepcopy(excitatory_layer_sizes)
        inhibitory_layer_sizes = deepcopy(inhibitory_layer_sizes)

        excitatory_layer_sizes.insert(0, input_dim)
        inhibitory_layer_sizes.insert(0, None)

        ee_syn = list(ee_synapses_per_branch_per_layer)
        ei_syn = list(ei_synapses_per_branch_per_layer)
        ie_syn = list(ie_synapses_per_branch_per_layer)
        ii_syn = list(ii_synapses_per_branch_per_layer)

        while len(ee_syn) < self.n_layers:
            ee_syn.append(ee_syn[-1])
        while len(ei_syn) < self.n_layers:
            ei_syn.append(ei_syn[-1])
        while len(ie_syn) < self.n_layers:
            ie_syn.append(ie_syn[-1])
        while len(ii_syn) < self.n_layers:
            ii_syn.append(ii_syn[-1])

        # Possibly build input network if input_mode=2
        self.input_net = None  # Renamed from mlp_transform
        if self.input_mode == 2:
            if not input_net_dict:
                raise ValueError(
                    "For input_mode=2, must provide input_net_dict with 'type', 'hidden_dims', 'output_dim'."
                )
            net_type = input_net_dict.get("type", "MLPInputTransform")
            hidden_dims = input_net_dict.get("hidden_dims", [128,64])
            net_out_dim = input_net_dict.get("output_dim", 64)
            
            self.input_net = InputNetTransform(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=net_out_dim,
                network_type=net_type
            )
            
            # Note: Pre-trained weights loading is now handled in train_experiments.py
            # based on the train.input_net_training configuration
            
            # The excit first layer sees net_out_dim
            excitatory_layer_sizes[0] = net_out_dim

        # Build each layer
        layers = []
        for i in range(self.n_layers):
            n_exc = excitatory_layer_sizes[i+1]

            ### CHANGED: Decide n_inh for layer i
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

            inhibitory_input_size = inhibitory_layer_sizes[i]
            if i == 0 and self.input_mode == 2:
                # For input_mode=2, use net_out_dim as inhibitory_input_dim
                inhibitory_input_size = input_net_dict.get("output_dim", 64)

            # Fix for newer version of ExcitationInhibitionLayer
            # Now passing all required arguments directly to the layer instead of creating DendriNet objects first
            layers.append(
                ExcitationInhibitionLayer(
                    n_excitatory_cells=n_exc,
                    n_inhibitory_cells=n_inh,
                    excitatory_branch_factors=excitatory_branch_factors,
                    inhibitory_branch_factors=inhibitory_branch_factors,
                    excitatory_input_dim=excitatory_layer_sizes[i],
                    ee_synapses_per_branch=ee_syn[i],
                    ei_synapses_per_branch=min(ei_syn[i], inhibitory_input_size) if i == 0 and self.input_mode == 2 else ei_syn[i],
                    inhibitory_input_dim=inhibitory_input_size,
                    ie_synapses_per_branch=ie_syn[i],
                    ii_synapses_per_branch=ii_syn[i],
                    reactivate=reactivate,
                    somatic_synapses=somatic_synapses,
                    topk_init_method=topk_init_method,
                    use_shunting=use_shunting,
                    reactivation_strategy=reactivation_strategy,
                    blocklinear_strategy=blocklinear_strategy,
                    reactivation_type=reactivation_type,
                    enable_branch_outputs=enable_branch_outputs
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        """
        Forward logic depends on input_mode:
        - mode=0 => excit=x, inhibitory=None at the first layer
        - mode=1 => excit=x, inhibitory=x at the first layer, but we skip building that layer's inhib net => no separate output
        - mode=2 => pass x into self.input_net => produce (excit, inhib)
                    skip that layer's inhib net => no separate output
        Then subsequent layers do normal E–I if their inhib cells exist.
        """
        if self.input_mode == 0:
            excitatory_input = x
            inhibitory_input = None
        elif self.input_mode == 1:
            excitatory_input = x
            inhibitory_input = x
        elif self.input_mode == 2:
            excitatory_input, inhibitory_input = self.input_net(x)
        else:
            raise ValueError(f"Unknown input_mode={self.input_mode}")

        excitatory_output, inhibitory_output = self.layers[0](excitatory_input, inhibitory_input)

        for i in range(1, self.n_layers):
            excitatory_output, inhibitory_output = self.layers[i](excitatory_output, inhibitory_output)

        return excitatory_output

    def forward_with_branch_outputs(self, x):
        if self.input_mode == 0:
            excitatory_input = x
            inhibitory_input = None
        elif self.input_mode == 1:
            excitatory_input = x
            inhibitory_input = x
        elif self.input_mode == 2:
            excitatory_input, inhibitory_input = self.input_net(x)
        else:
            raise ValueError(f"Unknown input_mode={self.input_mode}")

        excitatory_output, inhibitory_output, inh_acts, exc_acts = \
            self.layers[0].forward_with_branch_outputs(excitatory_input, inhibitory_input)
        all_acts = exc_acts

        for i in range(1, self.n_layers):
            excitatory_output, inhibitory_output, inh_acts, exc_acts = \
                self.layers[i].forward_with_branch_outputs(excitatory_output, inhibitory_output)
            all_acts.extend(exc_acts)

        return excitatory_output, all_acts

    def compute_local_loss(self, x, y=None):
        return 0.0

    @property
    def branch_layers(self):
        all_branch_layers = []
        for ei_layer in self.layers:
            all_branch_layers.extend(ei_layer.excitatory_cells.branch_layers)
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
            state_dict = input_net_weights
            
        self.input_net.load_state_dict(state_dict)
        
    def freeze_input_net_weights(self):
        """
        Freeze the weights in the input_net component to prevent them from being updated
        during training. This is useful for transfer learning scenarios.
        """
        if self.input_mode == 2 and self.input_net is not None:
            logger.info("Freezing input network weights")
            for param in self.input_net.parameters():
                param.requires_grad = False
        else:
            logger.warning("No input network to freeze (input_mode != 2 or no input_net)")
            
    def unfreeze_input_net_weights(self):
        """
        Unfreeze the weights in the input_net component to allow them to be updated
        during training.
        """
        if self.input_mode == 2 and self.input_net is not None:
            logger.info("Unfreezing input network weights")
            for param in self.input_net.parameters():
                param.requires_grad = True
        else:
            logger.warning("No input network to unfreeze (input_mode != 2 or no input_net)")


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
    Builds an MLP mimicking the excitatory/inhibitory structure.
    """
    def __init__(
        self,
        input_dim,
        excitatory_layer_sizes,
        inhibitory_layer_sizes,
        excitatory_branch_factors,
        inhibitory_branch_factors,
        ee_synapses_per_branch_per_layer,
        ie_synapses_per_branch_per_layer,
        output_layer=False,
        output_dim=10,
        activation=nn.ReLU(),
        weight_decay_rate=0.1,
        learning_strategy="mle",
        **kwargs
    ):
        super().__init__()
        logger.info("Building MLPExcInhNetwork...")
        self.weight_decay_rate = weight_decay_rate
        self.learning_strategy = learning_strategy
        self.output_layer_flag = output_layer

        input_dim = int(input_dim)
        excitatory_layer_sizes = [int(x) for x in excitatory_layer_sizes]
        inhibitory_layer_sizes = [int(x) for x in inhibitory_layer_sizes] if inhibitory_layer_sizes else []
        ee_synapses_per_branch_per_layer = [int(x) for x in ee_synapses_per_branch_per_layer]
        ie_synapses_per_branch_per_layer = [int(x) for x in ie_synapses_per_branch_per_layer]

        excit_sizes = deepcopy(excitatory_layer_sizes)
        excit_sizes.insert(0, input_dim)
        self.n_layers = len(excit_sizes) - 1

        layers = []
        for i in range(1, len(excit_sizes)):
            excit_count = excit_sizes[i-1] * ee_synapses_per_branch_per_layer[i-1]
            if inhibitory_layer_sizes:
                inhib_count = inhibitory_layer_sizes[i-1] * ie_synapses_per_branch_per_layer[i-1]
            else:
                inhib_count = 0
            in_dim = excit_count + inhib_count
            layer_activation = activation if i < self.n_layers else None
            layer_mod = MLPExcInhLayer(
                in_excit=excit_count,
                in_inhib=inhib_count,
                out_features=excit_sizes[i],
                activation=layer_activation
            )
            layers.append(layer_mod)
        self.layers = nn.ModuleList(layers)

        if self.output_layer_flag:
            self.classifier = nn.Linear(int(excit_sizes[-1]), int(output_dim))
        else:
            self.classifier = None

    def decay_weights(self, weight_decay=None):
        if weight_decay is None:
            weight_decay = self.weight_decay_rate
        for layer in self.layers:
            layer.decay_weights(weight_decay)
        if self.classifier is not None:
            with torch.no_grad():
                self.classifier.weight.sub_(weight_decay * self.classifier.weight)
                self.classifier.bias.sub_(weight_decay * self.classifier.bias)

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