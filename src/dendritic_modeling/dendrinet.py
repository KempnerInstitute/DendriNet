"""
dendrinet.py
============
This module contains the DendriNet and DendriticBranchLayer classes.
"""

from math import floor, log
from copy import deepcopy

import torch
import torch.nn as nn


TOPK_INIT_METHODS = {
    'xavier_normal': nn.init.xavier_normal_,
    'xavier_uniform': nn.init.xavier_uniform_,
    'kaiming_normal': nn.init.kaiming_normal_,
    'kaiming_uniform': nn.init.kaiming_uniform_,
    'orthogonal': nn.init.orthogonal_,
    'normal': nn.init.normal_,
    'uniform': nn.init.uniform_,
    'eye': nn.init.eye_,
}


def scale_layer_gradients(module, grad_input, grad_output, scale_factor):
    return tuple(
        grad * scale_factor if grad is not None else None for grad in grad_input
    )

def normalize_param_gradients(grad, scale_factor):
    return None if grad is None else (grad / grad.norm()) * scale_factor


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

    Examples
    --------

    >>> import torch
    >>> from torch import nn
    >>> topk_linear = TopKLinear(in_features=10, out_features=5, K=3)
    >>> x = torch.randn(4, 10)  # Batch of 4 samples
    >>> output = topk_linear(x)
    >>> print(output.shape)
    torch.Size([4, 5])
    """
    def __init__(
            self, 
            in_features, 
            out_features, 
            K, 
            distance,
            param_space = 'log',
            init_method = 'xavier_normal',
            init_params = None,
        ):
        super(TopKLinear, self).__init__()
        # make all weights positive
        self.pre_w = nn.Parameter(
            torch.empty((out_features, in_features)), requires_grad = True)
        
        # self.scale_factor = log(distance + 2)
        # self.pre_w.register_hook(
        #     lambda grad : normalize_param_gradients(grad, self.scale_factor)
        # )

        if not isinstance(K, int):
            raise TypeError('K must be an integer')
        
        if K < 1:
            raise ValueError('K must be greater than or equal to 1')
        
        if K > in_features:
            raise ValueError(
                f'K must be less than or equal to the number of input ',
                f'features. (K = {K}, in_features = {in_features})')

        self.K = K
        self.param_space = param_space
        self.init_method = init_method

        # self.register_full_backward_hook(normalize_layer_gradients)
            
    def initialize(self):
        if self.init_method in TOPK_INIT_METHODS:
            init_func = TOPK_INIT_METHODS[self.init_method]
            init_func(self.pre_w)
            #TODO: Accept hyperparameters for initialization
            # init_func(self.pre_w, **self.init_params) 
        else:
            raise ValueError(
                f'Invalid initialization method: {self.init_method}. ',
                f'Choose from {list(TOPK_INIT_METHODS.keys())}')
    
    def decay_weights(self, weight_decay = 0.1):
        with torch.no_grad():
            self.pre_w.data -= (weight_decay * self.weight_mask())
        
    def forward(self, x):
        # identify top K strongest synaptic connections onto each dendritic branch
        pruned_weight = self.pruned_weight()
        # matrix multiply inputs and synaptic weights
        return torch.mm(x, pruned_weight.t())
    
    def weight(self):
        if self.param_space == 'log':
            return self.pre_w.exp()
        elif self.param_space == 'presigmoid':
            return torch.sigmoid(self.pre_w)
    
    def log_weight(self):
        return self.weight().log()

    def weight_mask(self):
        topK_indices = torch.topk(self.pre_w, 
                                  self.K, 
                                  dim = -1, 
                                  largest = True, 
                                  sorted = False)[1]
        # initialize and populate masking matrix
        mask = torch.zeros_like(
            self.pre_w, 
            device = self.pre_w.device, 
            dtype = self.pre_w.dtype,
            )
        mask[torch.arange(self.pre_w.shape[0])[:,None], topK_indices] = 1
        return mask

    def pruned_weight(self):
        return self.weight_mask() * self.weight()
    
    def log_pruned_weight(self):
        return ((self.weight_mask() - 1) * 10) + self.log_weight()
    
    def weighted_synapses(self, cell_weights, prune = False):
        if prune:
            synapse_weights = self.pruned_weight()
        else:
            synapse_weights = self.weight()
        
        weighted_synapses = cell_weights[:,None] * synapse_weights
        return weighted_synapses.sum(dim = 0 )


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
    def __init__(
        self, in_features, out_features,
    ):
        super(BlockLinear, self).__init__()

        assert (in_features > 0 and out_features > 0), (
            "in_features and out_features must be greater than 0"
        )

        block_size = floor(in_features / out_features)
        assert(in_features == out_features * block_size), (
            "in_features must be divisible by out_features."
        )

        # branches have equal weights (?)
        self.log_weight = nn.Parameter(
            torch.empty((out_features, block_size)), requires_grad = True)
        # self.log_weight.register_hook(
        #     lambda grad : normalize_param_gradients(grad, 1)
        # )

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        self.scale_factor = 1#300
        self.register_full_backward_hook(
            lambda module, grad_input, grad_output : scale_layer_gradients(module, grad_input, grad_output, self.scale_factor))
    
    def initialize(self, sum_synapse_conductances = None):
        if sum_synapse_conductances is None:
            nn.init.constant_(self.log_weight, 0)
        else:
            self.log_weight.data = (1*sum_synapse_conductances).log()[:,None].expand(
                self.out_features, self.block_size).clone()
    
    def block(self):
        # initialize sparse matrix
        block = torch.zeros(
            (self.out_features, self.in_features),
            device = self.log_weight.device, dtype = self.log_weight.dtype,
        )
        # get row and column indices for sparse matrix which should contain non-zero weights
        row_ix = torch.arange(
            self.out_features, device = self.log_weight.device)[:,None]
        col_ix = torch.arange(
            self.in_features, device = self.log_weight.device).view(self.out_features, self.block_size)
        # assign weights to appropriate locations in sparse matrix
        block[row_ix, col_ix] = self.log_weight.exp()
        return block
    
    def log_block(self):
        log_block = torch.ones(
            (self.out_features, self.in_features),
            device = self.log_weight.device, dtype = self.log_weight.dtype,
        ) * -10
        row_ix = torch.arange(
            self.out_features, device = self.log_weight.device)[:,None]
        col_ix = torch.arange(
            self.in_features, device = self.log_weight.device).view(self.out_features, self.block_size)
        log_block[row_ix, col_ix] = self.log_weight
        return log_block
    
    def grad_block(self):
        grad_block = torch.zeros(
            (self.out_features, self.in_features),
            device = self.log_weight.device, dtype = self.log_weight.dtype,
        )
        row_ix = torch.arange(
            self.out_features, device = self.log_weight.device)[:,None]
        col_ix = torch.arange(
            self.in_features, device = self.log_weight.device).view(self.out_features, self.block_size)
        if self.log_weight.grad is not None:
            grad_block[row_ix, col_ix] = self.log_weight.grad
        return grad_block
    
    def decay_weights(self, weight_decay):
        with torch.no_grad():
            self.log_weight.data -= weight_decay

    def forward(self, x):
        # matrix multiply inputs and sparse matrix
        return torch.mm(x, self.block().t())
    
    def sum_conductances(self):
        sum_g = self.log_weight.exp().sum(dim = 1)
        return sum_g


class DendriticBranchLayer(nn.Module):
    """
    Emulates dendritic computation along a layer of branches (equally distant
    from soma).

    This layer models the interaction between excitatory and inhibitory inputs, 
    simulating dendritic processing in biological neurons. It aggregates 
    information from converging branches in the previous layer and optionally 
    incorporates inhibitory modulation and reactivation dynamics.

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

    reactivate : bool, optional
        If `True`, applies a reactivation function simulating dendritic/somatic 
        spiking behavior. Defaults to `False`.

    Attributes
    ----------

    branch_excitation : TopKLinear
        The module for computing excitation along branches in the layer.

    branch_inhibition : TopKLinear
        The module for computing inhibition along branches in the layer.

    branches_to_output : BlockLinear
        The module for aggregating information from converging branches.

    reactivate : bool
        If `True`, applies a reactivation function simulating neuronal firing
        at the soma layer.


    Methods
    -------

    forward(x, inhibitory_input=None, branch_input=None)
        Performs a forward pass through the layer.

    reactivation(V)
        Applies a reactivation function to the membrane potential.

    get_weights()
        Returns the synaptic weights for the excitatory and inhibitory inputs.

    get_mask()
        Returns the mask for the synaptic weights for the excitatory and
        inhibitory inputs.

    get_pruned_weights()
        Returns the pruned synaptic weights for the excitatory and inhibitory
        inputs.
    """
    def __init__(
        self, 
        distance,
        output_dim, 
        excitatory_input_dim = None, 
        excitatory_synapses_per_branch = None,
        inhibitory_input_dim = None, 
        inhibitory_synapses_per_branch = None, 
        input_branch_factor = None, 
        reactivate = False,
        topk_init_method = 'xavier_normal',
        ):
        
        # initialize dendritic branch layer
        super(DendriticBranchLayer, self).__init__()
        # module for computing excitation along branches in layer
        self.input_excitatory = excitatory_input_dim is not None
        if self.input_excitatory:
            self.branch_excitation = TopKLinear(
                excitatory_input_dim, 
                output_dim, 
                K = excitatory_synapses_per_branch,
                distance = distance,
                init_method = topk_init_method,
            )
        # if receiving inputs from inhibitory cells
        self.input_inhibitory = inhibitory_input_dim is not None
        if self.input_inhibitory:
            # module for computing inhibition along branches in layer
            self.branch_inhibition = TopKLinear(
                inhibitory_input_dim, 
                output_dim, 
                K = inhibitory_synapses_per_branch,
                distance = distance,
                init_method = topk_init_method,
            )
        # if receiving inputs from other dendritic branches
        self.input_branches = input_branch_factor is not None
        if self.input_branches:
            # module for aggregating information from converging branches
            self.branches_to_output = BlockLinear(
                output_dim*input_branch_factor, output_dim,
            )

        if reactivate:
            self.log_m = nn.Parameter(torch.empty((output_dim,)), 
                                      requires_grad = True)
            self.log_b = nn.Parameter(torch.empty((output_dim,)), 
                                      requires_grad = True)
        self.reactivate = reactivate

        # self.register_full_backward_hook(normalize_layer_gradients)

        self.initialize()
    
    def initialize(self):
        with torch.no_grad():
            if self.input_excitatory:
                self.branch_excitation.initialize()
            
            if self.input_inhibitory:
                self.branch_inhibition.initialize()
                self.branch_inhibition.pre_w + log(
                    self.branch_excitation.K / self.branch_inhibition.K
                )
                
            if self.input_branches:
                if self.input_excitatory or self.input_inhibitory:
                    sum_synapse_conductances = 0
                    if self.input_excitatory:
                        sum_synapse_conductances += self.branch_excitation.weight().sum(dim = 1)
                    if self.input_inhibitory:
                        sum_synapse_conductances += self.branch_inhibition.weight().sum(dim = 1)
                else:
                    sum_synapse_conductances = None
                self.branches_to_output.initialize(sum_synapse_conductances)
            
            if self.reactivate:
                # nn.init.constant_(self.log_m, 1.8)
                # nn.init.constant_(self.log_b, -0.7)
                nn.init.constant_(self.log_m, log(3.0))
                nn.init.constant_(self.log_b, log(0.5))
    
    def decay_weights(self, weight_decay = 0.1):
        if self.input_excitatory:
            self.branch_excitation.decay_weights(weight_decay)
        if self.input_inhibitory:
            self.branch_inhibition.decay_weights(weight_decay)
        # if self.input_branches:
        #     self.branches_to_output.decay_weights(weight_decay)

    def forward(self, x, inhibitory_input = None, branch_input = None):
        # initialize numerator and denominator variables
        numerator = 0
        denominator = 1
        # compute excitation and inhibition along each branch in layer
        if self.input_excitatory:
            excitation = self.branch_excitation(x)
            numerator = numerator + excitation
            denominator = denominator + excitation
        # if receiving inhibitory inputs
        if self.input_inhibitory and inhibitory_input is not None:
            denominator = denominator + self.branch_inhibition(inhibitory_input)
        # if receiving inputs from upstream branches
        if self.input_branches:
            # compute current contribution from upstream branches and add to numerator
            numerator = numerator + self.branches_to_output(branch_input)
            # compute total conductance of upstream branches and add to denominator
            denominator = denominator + self.branches_to_output.sum_conductances()
        # current divided by conductance is voltage
        voltage = numerator / denominator
        return self.reactivation(voltage) if self.reactivate else voltage

    def reactivation(self, V):
        m, b = self.log_m.exp(), self.log_b.exp()
        return (torch.tanh(m * (V - b)) + 1) / 2
    
    def get_weights(self):
        if self.input_inhibitory:
            return (self.branch_excitation.weight(), 
                    self.branch_inhibition.weight())
        else:
            return self.branch_excitation.weight()
    
    def get_log_weights(self):
        if self.input_inhibitory:
            return (
                self.branch_excitation.log_weight(), 
                self.branch_inhibition.log_weight()
            )
        else:
            return self.branch_excitation.log_weight()

    def get_mask(self):
        if self.input_inhibitory:
            return (self.branch_excitation.weight_mask(), 
                    self.branch_inhibition.weight_mask())
        else:
            return self.branch_excitation.weight_mask()

    def get_pruned_weights(self):
        if self.input_inhibitory:
            return (self.branch_excitation.pruned_weight(), 
                    self.branch_inhibition.pruned_weight())
        else:
            return self.branch_excitation.pruned_weight()
    
    def get_log_pruned_weights(self):
        if self.input_inhibitory:
            return (
                self.branch_excitation.log_pruned_weight(),
                self.branch_inhibition.log_pruned_weight()
            )
        else:
            return self.branch_excitation.log_pruned_weight()


class DendriNet(nn.Module):
    """
    Generates a linear layer of neurons, each of which consists of a sequential 
    structure of dendrtic branches and inhibitory/excitatory inputs.

    This network models the complex interactions between excitatory and 
    inhibitory inputs across multiple dendritic branches, simulating dendritic 
    processing in biological neurons.

    Parameters
    ----------
    n_soma : int
        The number of soma neurons in the network. Must be positive integer.

    branch_factors : list of int
        The branching factors for each layer of branches. Each element in the 
        list must be a positive integer.

    excitatory_input_dim : int
        The dimensionality of the excitatory inputs.
    
    excitatory_synapses_per_branch : int
        The number of excitatory synapses per branch.

    inhibitory_input_dim : int, optional
        The dimensionality of the inhibitory inputs. Defaults to `None`.

    inhibitory_synapses_per_branch : int, optional 
        The number of inhibitory synapses per branch. Defaults to `None`.

    reactivate : bool, optional
        If `True`, applies a reactivation function simulating neuronal firing 
        at the soma layer. Defaults to `False`.
    
    somatic_synapses : bool, optional
        If `True`, includes direct synaptic inputs to the soma layer. Defaults 
        to `True`.
    
    Attributes
    ----------
    branch_layers : torch.nn.ModuleList
        The list of branch layers in the network.

    soma_layer : DendriticBranchLayer
        The layer representing the soma of neurons.

    input_inhibitory : bool
        If `True`, the network receives inputs from inhibitory cells.

    n_soma : int
        The number of neurons in the network.
    
    reactivate : bool
        If True, applies a reactivation function dendritic / somatic spiking 
        behavior.

    somatic_synapses : bool
        If `True`, includes direct synaptic inputs to the soma layer.

    layer_sizes : list of int
        The number of branches in each layer of the network.

    Methods
    -------

    forward(x, inhibitory_input=None)
        Performs a forward pass through the network.

    sum_weights(pruned=False)
        Returns the sum of synaptic weights for excitatory and inhibitory 
        inputs.
    """
    def __init__(
        self, 
        n_soma, 
        branch_factors, 
        excitatory_input_dim, 
        excitatory_synapses_per_branch,
        inhibitory_input_dim = None, 
        inhibitory_synapses_per_branch = None,
        reactivate = False, 
        somatic_synapses = True,
        topk_init_method = 'xavier_normal',
    ):
        super(DendriNet, self).__init__()
        
        # n_soma should be a positive integer
        if not isinstance(n_soma, int):
            raise TypeError('n_soma must be an integer')
        if n_soma < 1:
            raise ValueError('n_soma must be greater than or equal to 1')
        
        # number of branches in layer closest to soma layer
        layer_sizes = [n_soma]
        # number of branches in upstream layers

        # branch_factors must be integers greater than or equal to 1
        for i in range(len(branch_factors)):
            if not isinstance(branch_factors[i], int):
                raise TypeError('branch_factors must be integers')
            if branch_factors[i] < 1:
                raise ValueError('branch_factors must be greater than or equal to 1')
        
        # Inhibitory synapses must be positive integers
        if inhibitory_synapses_per_branch is not None:
            if not isinstance(inhibitory_synapses_per_branch, int):
                raise TypeError('inhibitory_synapses_per_branch must be an integer')
            if inhibitory_synapses_per_branch < 1:
                raise ValueError('inhibitory_synapses_per_branch must be greater than or equal to 1')
            
        # Excitatory synapses must be positive integers
        if not isinstance(excitatory_synapses_per_branch, int):
            raise TypeError('excitatory_synapses_per_branch must be an integer')
        if excitatory_synapses_per_branch < 1:
            raise ValueError('excitatory_synapses_per_branch must be greater than or equal to 1')

        branch_factors = deepcopy(branch_factors)
        self.n_branch_layers = len(branch_factors)
        for i in range(self.n_branch_layers):
            layer_sizes.append(layer_sizes[i] * branch_factors[i])
        # first item in list corresponds to number of branches in layer furthest from the soma layer
        layer_sizes.reverse()
        branch_factors.reverse()
        # first branch layer (furthest from soma) does not receive any branch inputs
        branch_factors.insert(0, None)
        # create branch layers and append to list
        branch_layers = []
        for i in range(self.n_branch_layers):
            branch_layers.append(DendriticBranchLayer(
                distance = self.n_branch_layers - i,
                output_dim = layer_sizes[i], 
                excitatory_input_dim = excitatory_input_dim, 
                excitatory_synapses_per_branch = excitatory_synapses_per_branch,
                inhibitory_input_dim = inhibitory_input_dim, 
                inhibitory_synapses_per_branch = inhibitory_synapses_per_branch, 
                input_branch_factor = branch_factors[i], 
                reactivate = reactivate,
                topk_init_method = topk_init_method,
            ))
        # wrap list of branch layers in ModuleList
        self.branch_layers = nn.ModuleList(branch_layers)
        
        # branches to soma
        if somatic_synapses:
            self.soma_layer = DendriticBranchLayer(
                distance = 0,
                output_dim = n_soma, 
                excitatory_input_dim = excitatory_input_dim, 
                excitatory_synapses_per_branch = excitatory_synapses_per_branch,
                inhibitory_input_dim = inhibitory_input_dim, 
                inhibitory_synapses_per_branch = inhibitory_synapses_per_branch,
                input_branch_factor = branch_factors[-1], 
                reactivate = reactivate,
                topk_init_method = topk_init_method,
            )
        else:
            self.soma_layer = DendriticBranchLayer(
                distance = 0,
                output_dim = n_soma, 
                excitatory_input_dim = None, 
                excitatory_synapses_per_branch = None,
                inhibitory_input_dim = None, 
                inhibitory_synapses_per_branch = None,
                input_branch_factor = branch_factors[-1], 
                reactivate = reactivate,
                topk_init_method = topk_init_method,
            )
        
        # assign necessary class attributes
        self.input_inhibitory = inhibitory_input_dim is not None
        self.n_soma = n_soma
        self.reactivate = reactivate
        self.somatic_synapses = somatic_synapses
        self.layer_sizes = layer_sizes
    
    def decay_weights(self, weight_decay = 0.1):
        for branch_layer in self.branch_layers:
            branch_layer.decay_weights(weight_decay)
        self.soma_layer.decay_weights(weight_decay)
    
    def forward(self, x, inhibitory_input = None):
        # branch_input to first branch layer is None
        output = None
        # dendritic computation
        for i in range(self.n_branch_layers):
            # recursively update branch_input from one branch layer to the next
            output = self.branch_layers[i](x, inhibitory_input, output)
        # process information at the soma layer
        output = self.soma_layer(x, inhibitory_input, output)
        # return output
        return output
    
    def sum_weights(self, pruned = False):
        exc_total = 0
        if self.input_inhibitory:
            inh_total = 0
        
        n_layers = self.n_branch_layers + 1 if self.somatic_synapses else self.n_branch_layers

        for i in range(n_layers):
            if self.somatic_synapses and i == n_layers - 1:
                layer = self.soma_layer
            else:
                layer = self.branch_layers[i]

            if self.input_inhibitory:
                exc, inh = layer.get_pruned_weights() if pruned else layer.get_weights()

                inh_chunk = inh.chunk(self.n_soma, dim = 0)
                inh_total = inh_total + torch.stack([chunk.sum(0) for chunk in inh_chunk], dim = 0)
            else:
                exc = layer.get_pruned_weights() if pruned else layer.get_weights()
            
            exc_chunk = exc.chunk(self.n_soma, dim = 0)
            exc_total = exc_total + torch.stack([chunk.sum(0) for chunk in exc_chunk], dim = 0)

        if self.input_inhibitory:
            return exc_total, inh_total
        else:
            return exc_total
        
    def log_sum_weights(self, pruned = False):
        if self.input_inhibitory:
            exc_total, inh_total = self.sum_weights(pruned = pruned)
            return (exc_total + 1e-6).log(), (inh_total + 1e-6).log()
        else:
            exc_total = self.sum_weights(pruned = pruned)
            return (exc_total + 1e-6).log()


# class DendriNet(nn.Module):
#     """
#     Generates a linear layer of neurons, each of which consists of a sequential 
#     structure of dendrtic branches and inhibitory/excitatory inputs.

#     This network models the complex interactions between excitatory and 
#     inhibitory inputs across multiple dendritic branches, simulating dendritic 
#     processing in biological neurons.

#     Parameters
#     ----------
#     n_soma : int
#         The number of soma neurons in the network. Must be positive integer.

#     branch_factors : list of int
#         The branching factors for each layer of branches. Each element in the 
#         list must be a positive integer.

#     excitatory_input_dim : int
#         The dimensionality of the excitatory inputs.
    
#     excitatory_synapses_per_branch : int
#         The number of excitatory synapses per branch.

#     inhibitory_input_dim : int, optional
#         The dimensionality of the inhibitory inputs. Defaults to `None`.

#     inhibitory_synapses_per_branch : int, optional 
#         The number of inhibitory synapses per branch. Defaults to `None`.

#     reactivate : bool, optional
#         If `True`, applies a reactivation function simulating neuronal firing 
#         at the soma layer. Defaults to `False`.
    
#     somatic_synapses : bool, optional
#         If `True`, includes direct synaptic inputs to the soma layer. Defaults 
#         to `True`.
    
#     Attributes
#     ----------
#     branch_layers : torch.nn.ModuleList
#         The list of branch layers in the network.

#     soma_layer : DendriticBranchLayer
#         The layer representing the soma of neurons.

#     input_inhibitory : bool
#         If `True`, the network receives inputs from inhibitory cells.

#     n_soma : int
#         The number of neurons in the network.
    
#     reactivate : bool
#         If True, applies a reactivation function dendritic / somatic spiking 
#         behavior.

#     somatic_synapses : bool
#         If `True`, includes direct synaptic inputs to the soma layer.

#     layer_sizes : list of int
#         The number of branches in each layer of the network.

#     Methods
#     -------

#     forward(x, inhibitory_input=None)
#         Performs a forward pass through the network.

#     sum_weights(pruned=False)
#         Returns the sum of synaptic weights for excitatory and inhibitory 
#         inputs.
#     """
#     def __init__(
#         self, 
#         n_soma, 
#         branch_factors, 
#         excitatory_input_dim, 
#         excitatory_synapses_per_branch,
#         inhibitory_input_dim = None, 
#         inhibitory_synapses_per_branch = None,
#         reactivate = False, 
#         somatic_synapses = True,
#         topk_init_method = 'xavier_normal',
#     ):
#         super(DendriNet, self).__init__()
        
#         # n_soma should be a positive integer
#         if not isinstance(n_soma, int):
#             raise TypeError('n_soma must be an integer')
#         if n_soma < 1:
#             raise ValueError('n_soma must be greater than or equal to 1')
        
#         # number of branches in layer closest to soma layer
#         layer_sizes = [n_soma]
#         # number of branches in upstream layers

#         # branch_factors must be integers greater than or equal to 1
#         for i in range(len(branch_factors)):
#             if not isinstance(branch_factors[i], int):
#                 raise TypeError('branch_factors must be integers')
#             if branch_factors[i] < 1:
#                 raise ValueError('branch_factors must be greater than or equal to 1')
        
#         # Inhibitory synapses must be positive integers
#         if inhibitory_synapses_per_branch is not None:
#             if not isinstance(inhibitory_synapses_per_branch, int):
#                 raise TypeError('inhibitory_synapses_per_branch must be an integer')
#             if inhibitory_synapses_per_branch < 1:
#                 raise ValueError('inhibitory_synapses_per_branch must be greater than or equal to 1')
            
#         # Excitatory synapses must be positive integers
#         if not isinstance(excitatory_synapses_per_branch, int):
#             raise TypeError('excitatory_synapses_per_branch must be an integer')
#         if excitatory_synapses_per_branch < 1:
#             raise ValueError('excitatory_synapses_per_branch must be greater than or equal to 1')

#         branch_factors = deepcopy(branch_factors)
#         self.n_branch_layers = len(branch_factors)
#         for i in range(self.n_branch_layers):
#             layer_sizes.append(layer_sizes[i] * branch_factors[i])
#         # first item in list corresponds to number of branches in layer furthest from the soma layer
#         layer_sizes.reverse()
#         branch_factors.reverse()
#         # first branch layer (furthest from soma) does not receive any branch inputs
#         branch_factors.insert(0, None)
#         # create branch layers and append to list
#         branch_layers = []
#         for i in range(self.n_branch_layers):
#             branch_layers.append(DendriticBranchLayer(
#                 output_dim = layer_sizes[i], 
#                 excitatory_input_dim = excitatory_input_dim, 
#                 excitatory_synapses_per_branch = excitatory_synapses_per_branch,
#                 inhibitory_input_dim = inhibitory_input_dim, 
#                 inhibitory_synapses_per_branch = inhibitory_synapses_per_branch, 
#                 input_branch_factor = branch_factors[i], 
#                 reactivate = reactivate,
#                 topk_init_method = topk_init_method,
#             ))
#         # wrap list of branch layers in ModuleList
#         self.branch_layers = nn.ModuleList(branch_layers)
        
#         # branches to soma
#         if somatic_synapses:
#             self.soma_layer = DendriticBranchLayer(
#                 output_dim = n_soma, 
#                 excitatory_input_dim = excitatory_input_dim, 
#                 excitatory_synapses_per_branch = excitatory_synapses_per_branch,
#                 inhibitory_input_dim = inhibitory_input_dim, 
#                 inhibitory_synapses_per_branch = inhibitory_synapses_per_branch,
#                 input_branch_factor = branch_factors[-1], 
#                 reactivate = True,
#                 topk_init_method = topk_init_method,
#             )
#         else:
#             self.soma_layer = BlockLinear(layer_sizes[-2], n_soma)
        
#         # assign necessary class attributes
#         self.input_inhibitory = inhibitory_input_dim is not None
#         self.n_soma = n_soma
#         self.reactivate = reactivate
#         self.somatic_synapses = somatic_synapses
#         self.layer_sizes = layer_sizes
    
#     def decay_weights(self, weight_decay = 0.1):
#         for branch_layer in self.branch_layers:
#             branch_layer.decay_weights(weight_decay)
#         self.soma_layer.decay_weights(weight_decay)
    
#     def forward(self, x, inhibitory_input = None):
#         # branch_input to first branch layer is None
#         output = None
#         # dendritic computation
#         for i in range(self.n_branch_layers):
#             # recursively update branch_input from one branch layer to the next
#             output = self.branch_layers[i](x, inhibitory_input, output)
#         # process information at the soma layer
#         if self.somatic_synapses:
#             output = self.soma_layer(x, inhibitory_input, output)
#         else:
#             output = self.soma_layer(output) / (self.soma_layer.sum_conductances() + 1)
#         # return output
#         return output
    
#     def sum_weights(self, pruned = False):
#         exc_total = 0
#         if self.input_inhibitory:
#             inh_total = 0
        
#         n_layers = self.n_branch_layers + 1 if self.somatic_synapses else self.n_branch_layers

#         for i in range(n_layers):
#             if self.somatic_synapses and i == n_layers - 1:
#                 layer = self.soma_layer
#             else:
#                 layer = self.branch_layers[i]

#             if self.input_inhibitory:
#                 exc, inh = layer.get_pruned_weights() if pruned else layer.get_weights()

#                 inh_chunk = inh.chunk(self.n_soma, dim = 0)
#                 inh_total = inh_total + torch.stack([chunk.sum(0) for chunk in inh_chunk], dim = 0)
#             else:
#                 exc = layer.get_pruned_weights() if pruned else layer.get_weights()
            
#             exc_chunk = exc.chunk(self.n_soma, dim = 0)
#             exc_total = exc_total + torch.stack([chunk.sum(0) for chunk in exc_chunk], dim = 0)

#         if self.input_inhibitory:
#             return exc_total, inh_total
#         else:
#             return exc_total
        
#     def log_sum_weights(self, pruned = False):
#         if self.input_inhibitory:
#             exc_total, inh_total = self.sum_weights(pruned = pruned)
#             return (exc_total + 1e-6).log(), (inh_total + 1e-6).log()
#         else:
#             exc_total = self.sum_weights(pruned = pruned)
#             return (exc_total + 1e-6).log()
