import torch
import torch.nn as nn

from math import floor


# new implementation of TopKLinear where only topk weights for each output node are trained
class newTopKLinear(nn.Module):
    def __init__(
            self, in_features, out_features, K, bias = False,
            ):
        # super initialize
        super(newTopKLinear, self).__init__()
        # hold each scalar value in weight matrix as an individual parameter in ParameterList object
        #   initialize params in [0,1) interval with requires_grad set to False
        self.params = nn.ParameterList([
            nn.Parameter(torch.rand(1), requires_grad = False) for _ in range(in_features*out_features)
        ])
        # initialize class attributes
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        # get indices of parameters in self.params which should be trainable
        _, params_ix = self.pruned_weight_and_indices()
        # for specified parameters, set requires_grad = True
        for i in range(params_ix.shape[0]):
            self.params[params_ix[i]].requires_grad_(True)
        # this class attribute is referenced during each forward call to determine which 
        #   parameters should have their requires_grad field inverted, if any
        self.params_ix = params_ix
        # if a bias term is to be included
        if bias:
            self.b = nn.Parameter(torch.rand(out_features), requires_grad = True)
        self.bias = bias

    def forward(self, x):
        # get pruned weight matrix and indices of parameters which are used
        pruned_weight, params_ix = self.pruned_weight_and_indices()
        # logic for whether or not to update requires_grad fields of parameters in self.params
        if params_ix != self.params_ix:
            # function call to update requires_grad fields
            self.update_freeze(new_ix = params_ix)
        # matrix multiply inputs and pruned weight matrix
        output = torch.mm(x, pruned_weight.t())
        # if bias term is included
        if self.bias:
            # add bias to output
            output = output + self.b.expand(output.shape[0], -1)
        # return layer output
        return output
    
    def pruned_weight_and_indices(self):
        # reformat self.params (list of Parameters) into shape of weight matrix
        weight = torch.cat(self.params, dim = 0).view(self.out_features, self.in_features)
        # determine indices of the topk weights for each output node which are to be used
        mask_ix = torch.topk(weight, self.K, dim = -1, largest = True, sorted = False)[1]
        # convert mask indices to those of corresponding parameters in self.params
        params_ix = ((self.in_features*torch.arange(self.out_features)).expand(-1, self.K) + mask_ix).flatten().sort()
        # initialize mask as zeros matrix
        mask = torch.zeros_like(
            weight, device = weight.device, dtype = weight.dtype,
        )
        # assign 1 values to mask based on topk indices
        mask[torch.arange(self.out_features), mask_ix] = 1
        # use mask to prune weight matrix
        pruned_weight = mask * weight
        # return pruned weight matrix and indices of parameters in self.params which are used
        return pruned_weight, params_ix
    
    def update_freeze(self, new_ix):
        # vector of indices of parameters in self.params which will have requires_grad field set to opposite
        invert = torch.cat([
            new_ix[~torch.isin(new_ix, self.params_ix)], # new parameters being used; requires_grad is currently False, set to True
            self.params_ix[~torch.isin(self.params_ix, new_ix)], # parameters no longer being used; set requires_grad to False
        ], dim = 0)
        # iterate over indices of parameters in self.params which will have requires_grad field inverted
        for i in range(invert.shape[0]):
            # invert requires_grad field
            self.params[invert[i]].requires_grad_(~self.params[invert[i]].requires_grad)
        # update self.params_ix attribute
        self.params_ix = new_ix


# custom linear layer which only uses the strongest K synaptic weights from in_features 
class TopKLinear(nn.Linear):
    def __init__(
            self, in_features, out_features, K, bias = False,
            ):    
        # initialize super Linear layer
        super().__init__(in_features, out_features, bias = bias)
        # make all weights positive
        with torch.no_grad():
            self.weight = torch.sign(self.weight) * self.weight
        # initialize self.topK_indices
        self.topK_indices = torch.topk(self.weight, K, dim = -1, largest = True, sorted = False)[1]
        # assign positive integer attribute K, denoting the number of strongest synapses to keep per dendritic branch
        self.K = K

    
    def forward(self, x):
        # identify top K strongest synaptic connections onto each dendritic branch
        topK_indices = torch.topk(self.weight, self.K, dim = -1, largest = True, sorted = False)[1]
        # initialize and populate masking matrix
        mask = torch.zeros_like(
            self.weight, device = self.weight.device, dtype = self.weight.dtype,
            requires_grad = self.weight.requires_grad,
            )
        mask[torch.arange(self.weight.shape[0]), topK_indices] = 1
        # apply mask to weight matrix for manual pruning effect
        pruned_weight = mask * self.weight
        # matrix multiply inputs and synaptic weights
        return torch.mm(x, pruned_weight.t())
    
    def get_mask(self):
        # TBD
        pass


# custom linear layer aggregating information from converging branches in previous branch layer
class BlockLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias = False, requires_grad = False):
        # block_size equals number of branches converging onto single branch / soma in next layer???
        block_size = floor(in_features / out_features)
        assert(in_features == out_features * block_size)
        self.block_size = block_size
        # initialize super linear layer
        super().__init__(block_size, out_features, bias = bias)
        # these parameters are not learned
        self.requires_grad_(requires_grad)
        # branches have equal weights (?)
        with torch.no_grad():
            self.weight = 1
    
    def forward(self, x):
        # initialize sparse matrix
        block = torch.zeros(
            (self.out_features, self.in_features),
            device = self.weight.device, dtype = self.weight.dtype,
            requires_grad = self.weight.requires_grad,
            )
        # get row and column indices for sparse matrix which should contain non-zero weights
        row_ix = torch.arange(
            self.out_features, device = self.weight.device)#.unsqueeze(1)
        col_ix = torch.arange(
            self.in_features, device = self.weight.device).view(self.out_features, self.block_size)
        # assign weights to appropriate locations in sparse matrix
        block[row_ix, col_ix] = self.weight
        # matrix multiply inputs and sparse matrix
        return torch.mm(x, block.t())


# emulates dendritic computation along a layer of branches (equally distant from soma)
class DendriticBranchLayer(nn.Module):
    def __init__(
        self, input_dim, n_branches, excitatory_synapses_per_branch,
        n_inhibitory_cells, inhibitory_synapses_per_branch, 
        shunting = True, reactivate = False, n_input_branches = None,
        ):
        # initialize dendritic branch layer
        super(DendriticBranchLayer, self).__init__()
        # module for computing excitation along branches in layer
        self.branch_excitation = TopKLinear(
            input_dim, n_branches, K = excitatory_synapses_per_branch
        )
        # module for computing inhibition along branches in layer
        self.branch_inhibition = TopKLinear(
            n_inhibitory_cells, n_branches, K = inhibitory_synapses_per_branch
        )
        # if receiving inputs from other dendritic branches
        self.input_branches = n_input_branches is not None
        if self.input_branches:
            # module for aggregating information from converging branches
            self.branches_to_branch = BlockLinear(n_input_branches, n_branches)            
        # initialize sigmoid activation if shunting activation is not used
        if not shunting:
            self.sigmoid = nn.Sigmoid()
        # assign necessary class attributes
        self.shunting = shunting
        self.reactivate = reactivate
        self.e_i_ratio = excitatory_synapses_per_branch / inhibitory_synapses_per_branch
    
    def forward(self, x, inhibitory_cell_activation, branch_input = None):
        # compute excitation and inhibition along each branch in layer
        g_excitation = self.branch_excitation(x)
        g_inhibition = self.branch_inhibition(inhibitory_cell_activation)
        # if receiving inputs from upstream branches
        if self.input_branches:
            # compute depolarization effect and update excitation
            depolarization = self.branches_to_branch(branch_input)
            g_excitation = g_excitation + depolarization
        # if shunting activation is used
        if self.shunting:
            # compute shunting activation
            output = g_excitation / (g_excitation + self.e_i_ratio * g_inhibition + 1)
        else:
            # compute sigmoidal activation
            output = self.sigmoid(g_excitation - self.e_i_ratio * g_inhibition - 3)
        # reactivation
        if self.reactivate:
            output = output / (output + 1)
        # return activation along branches as output of layer
        return output


# emulates branching dendritic tree strctures into an artficial neural network analogous to the MLP 
class DendriNet(nn.Module):
    def __init__(
        self, input_dim, n_soma,
        n_branch_layers, branch_factors, # change this to root branches or something like that
        n_inhibitory_cells, synapses_per_inhibitory_cell, 
        inhibitory_synapses_per_branch, excitatory_synapses_per_branch, shunting, reactivate,
        ):
        # module for propagating inputs to inhibitory cells
        self.input_inhibitory_cells = TopKLinear(input_dim, n_inhibitory_cells, synapses_per_inhibitory_cell)
        # number of branches in layer closest to soma layer
        layer_sizes = [n_soma * branch_factors[0]]
        # number of branches in upstream layers
        for i in range(1, n_branch_layers):
            layer_sizes.append(layer_sizes[i-1] * branch_factors[i])
        # first item in list corresponds to number of branches in layer furthest from the soma layer
        layer_sizes.reverse()
        # first branch layer (furthest from soma) does not receive any branch inputs
        layer_sizes.insert(0, None)
        # create branch layers and append to list
        branch_layers = []
        for i in range(n_branch_layers):
            branch_layers.append(DendriticBranchLayer(
                input_dim = input_dim, n_branches = layer_sizes[i+1], excitatory_synapses_per_branch = excitatory_synapses_per_branch,
                n_inhibitory_cells = n_inhibitory_cells, inhibitory_synapses_per_branch = inhibitory_synapses_per_branch,
                shunting = shunting, reactivate = reactivate, n_input_branches = layer_sizes[i],
            ))
        # wrap list of branch layers in ModuleList
        self.branch_layers = nn.ModuleList(branch_layers)
        # propagate outputs of dendritic computation to soma layer
        self.branches_to_soma = BlockLinear(layer_sizes[-1], n_soma)
        # if shunting activation is not used
        if not shunting:
            self.sigmoid = nn.Sigmoid()
        # assign necessary class attributes
        self.shunting = shunting
        self.n_branch_layers = n_branch_layers
    
    def forward(self, x):
        # compute output from inhibitory cells
        g_inhibition = self.input_inhibitory_cells(x)
        # if shunting is used, compute shunting activation for inhibitory cell output
        if self.shunting:
            v_inhibitory = g_inhibition / (g_inhibition + 1) # hardcoding sigmoid with midpoint at 1
        # if no shunting, compute sigmoidal activation for inhibitory cell output 
        else:
            v_inhibitory = self.sigmoid(g_inhibition - 3)
        # branch_input to first branch layer is None
        branch_input = None
        # dendritic computation
        for i in range(self.n_branch_layers):
            # recursively update branch_input from one branch layer to the next
            branch_input = self.branch_layers[i](x, v_inhibitory, branch_input)
        # return information processed at the soma layer
        return self.branches_to_soma(branch_input)
        
        













