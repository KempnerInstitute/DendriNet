import torch
import torch.nn as nn

from copy import deepcopy

from math import floor


# custom linear layer which only uses the strongest K synaptic weights from in_features 
class TopKLinear(nn.Module):
    def __init__(self, in_features, out_features, K, param_space = 'sigmoid'):    
        # initialize super Linear layer
        super(TopKLinear, self).__init__()
        # make all weights positive
        self.pre_w = nn.Parameter(torch.empty((out_features, in_features)), requires_grad = True)
        # assign positive integer attribute K, denoting the number of strongest synapses to keep per dendritic branch
        self.K = K
        self.param_space = param_space
    
    def initialize(self):
        nn.init.xavier_normal_(self.pre_w)
    
    def forward(self, x):
        # identify top K strongest synaptic connections onto each dendritic branch
        pruned_weight = self.pruned_weight()
        # matrix multiply inputs and synaptic weights
        return torch.mm(x, pruned_weight.t())
    
    def weight(self):
        if self.param_space == 'log':
            return self.pre_w.exp()
        elif self.param_space == 'sigmoid':
            return torch.sigmoid(self.pre_w)
    
    def log_weight(self):
        return self.weight().log()

    def weight_mask(self):
        topK_indices = torch.topk(self.pre_w, self.K, dim = -1, largest = True, sorted = False)[1]
        # initialize and populate masking matrix
        mask = torch.zeros_like(
            self.pre_w, device = self.pre_w.device, dtype = self.pre_w.dtype,
            )
        mask[torch.arange(self.pre_w.shape[0])[:,None], topK_indices] = 1
        return mask

    def pruned_weight(self):
        # apply mask to weight matrix for manual pruning effect
        return self.weight_mask() * self.weight()
    
    def log_pruned_weight(self):
        return ((self.weight_mask() - 1) * 2) + self.log_weight()
    
    def weighted_synapses(self, cell_weights, prune = False):
        if prune:
            synapse_weights = self.pruned_weight()
        else:
            synapse_weights = self.weight()
        
        weighted_synapses = cell_weights[:,None] * synapse_weights
        return weighted_synapses.sum(dim = 0 )


# custom linear layer aggregating information from converging branches in previous branch layer
class BlockLinear(nn.Module):
    def __init__(self, in_features, out_features, requires_grad = False):
        super(BlockLinear, self).__init__()
        # block_size equals number of branches converging onto single branch / soma in next layer
        block_size = floor(in_features / out_features)
        assert(in_features == out_features * block_size)

        # branches have equal weights (?)
        self.log_weight = nn.Parameter(torch.empty((out_features, block_size)), requires_grad = requires_grad)

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

    def initialize(self, sum_synapse_conductance):
        self.log_weight.data = sum_synapse_conductance.log()[:,None].expand(*self.log_weight.shape).clone()
        
    def forward(self, x):
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
        # matrix multiply inputs and sparse matrix
        return torch.mm(x, block.t())
    
    def sum_conductances(self):
        sum_g = self.log_weight.exp().sum(dim = 1)
        return sum_g


# emulates dendritic computation along a layer of branches (equally distant from soma)
class DendriticBranchLayer(nn.Module):
    def __init__(
        self, output_dim, excitatory_input_dim, excitatory_synapses_per_branch,
        inhibitory_input_dim = None, inhibitory_synapses_per_branch = None, 
        input_branch_factor = None, reactivate = False,
        ):
        # initialize dendritic branch layer
        super(DendriticBranchLayer, self).__init__()
        # module for computing excitation along branches in layer
        self.branch_excitation = TopKLinear(excitatory_input_dim, output_dim, K = excitatory_synapses_per_branch)
        # if receiving inputs from inhibitory cells
        self.input_inhibitory = inhibitory_input_dim is not None
        if self.input_inhibitory:
            # module for computing inhibition along branches in layer
            self.branch_inhibition = TopKLinear(inhibitory_input_dim, output_dim, K = inhibitory_synapses_per_branch)
        # if receiving inputs from other dendritic branches
        self.input_branches = input_branch_factor is not None
        if self.input_branches:
            # module for aggregating information from converging branches
            self.branches_to_output = BlockLinear(
                output_dim*input_branch_factor, output_dim, requires_grad = True,
            )
        # if using branch reactivation
        if reactivate:
            self.log_m = nn.Parameter(torch.empty((output_dim,)), requires_grad = True)
            self.log_b = nn.Parameter(torch.empty((output_dim,)), requires_grad = True)
        self.reactivate = reactivate

        self.initialize()

    def initialize(self):
        self.branch_excitation.initialize()
        if self.input_inhibitory:
            self.branch_inhibition.initialize()

        if self.input_branches:
            sum_synapse_conductance = self.branch_excitation.pruned_weight().data.sum(dim = 1)
            if self.input_inhibitory:
                sum_synapse_conductance += self.branch_inhibition.pruned_weight().data.sum(dim = 1)
            self.branches_to_output.initialize(sum_synapse_conductance)
        
        if self.reactivate:
            nn.init.zeros_(self.log_m)
            nn.init.zeros_(self.log_b)
    
    def forward(self, x, inhibitory_input = None, branch_input = None):
        # compute excitation and inhibition along each branch in layer
        excitation = self.branch_excitation(x)
        # initialize numerator and denominator variables
        numerator = excitation
        denominator = excitation + 1
        # if receiving inputs from upstream branches
        if self.input_branches:
            # compute current contribution from upstream branches and add to numerator
            current = self.branches_to_output(branch_input)
            numerator = numerator + current
            # compute total conductance of upstream branches and add to denominator
            conductance = self.branches_to_output.sum_conductances()
            denominator = denominator + conductance
        # if receiving inhibitory inputs
        if self.input_inhibitory:
            denominator = denominator + self.branch_inhibition(inhibitory_input)
        # current divided by conductance is voltage
        voltage = numerator / denominator
        return self.reactivation(voltage) if self.reactivate else voltage

    def reactivation(self, V):
        m, b = self.log_m.exp(), self.log_b.exp()
        return (torch.tanh(m * V - b) + 1) / 2
    
    def get_weights(self):
        if self.input_inhibitory:
            return self.branch_excitation.weight(), self.branch_inhibition.weight()
        else:
            return self.branch_excitation.weight()
        
    def get_log_weights(self):
        if self.input_inhibitory:
            return self.branch_excitation.log_weight(), self.branch_inhibition.log_weight()
        else:
            return self.branch_excitation.log_weight()

    def get_mask(self):
        if self.input_inhibitory:
            return self.branch_excitation.weight_mask(), self.branch_inhibition.weight_mask()
        else:
            return self.branch_excitation.weight_mask()

    def get_pruned_weights(self):
        if self.input_inhibitory:
            return self.branch_excitation.pruned_weight(), self.branch_inhibition.pruned_weight()
        else:
            return self.branch_excitation.pruned_weight()
    
    def get_log_pruned_weights(self):
        if self.input_inhibitory:
            return self.branch_excitation.log_pruned_weight(), self.branch_inhibition.log_pruned_weight()
        else:
            return self.branch_excitation.log_pruned_weight()


# emulates branching dendritic tree strctures into an artficial neural network analogous to the MLP 
class DendriNet(nn.Module):
    def __init__(
        self, n_soma, branch_factors, excitatory_input_dim, excitatory_synapses_per_branch,
        inhibitory_input_dim = None, inhibitory_synapses_per_branch = None,
         reactivate = False, somatic_synapses = True,
    ):
        super(DendriNet, self).__init__()
        # number of branches in layer closest to soma layer
        layer_sizes = [n_soma]
        # number of branches in upstream layers
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
                output_dim = layer_sizes[i], 
                excitatory_input_dim = excitatory_input_dim, excitatory_synapses_per_branch = excitatory_synapses_per_branch,
                inhibitory_input_dim = inhibitory_input_dim, inhibitory_synapses_per_branch = inhibitory_synapses_per_branch, 
                input_branch_factor = branch_factors[i], reactivate = reactivate,
            ))
        # wrap list of branch layers in ModuleList
        self.branch_layers = nn.ModuleList(branch_layers)
        # branches to soma
        if somatic_synapses:
            self.soma_layer = DendriticBranchLayer(
                output_dim = n_soma, 
                excitatory_input_dim = excitatory_input_dim, excitatory_synapses_per_branch = excitatory_synapses_per_branch,
                inhibitory_input_dim = inhibitory_input_dim, inhibitory_synapses_per_branch = inhibitory_synapses_per_branch,
                input_branch_factor = branch_factors[-1], reactivate = True,
            )
        else:
            self.soma_layer = BlockLinear(layer_sizes[-2], n_soma, requires_grad = True)
        # assign necessary class attributes
        self.input_inhibitory = inhibitory_input_dim is not None
        self.n_soma = n_soma
        self.reactivate = reactivate
        self.somatic_synapses = somatic_synapses
        self.layer_sizes = layer_sizes
    
    def forward(self, x, inhibitory_input = None):
        # branch_input to first branch layer is None
        output = None
        # dendritic computation
        for i in range(self.n_branch_layers):
            # recursively update branch_input from one branch layer to the next
            output = self.branch_layers[i](x, inhibitory_input, output)
        # process information at the soma layer
        if self.somatic_synapses:
            output = self.soma_layer(x, inhibitory_input, output)
        else:
            output = self.soma_layer(output)
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
            return (exc_total + 0.5).log(), (inh_total + 1e-1).log()
        else:
            exc_total = self.sum_weights(pruned = pruned)
            return (exc_total + 0.5).log()

