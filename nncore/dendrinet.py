import torch
import torch.nn as nn

from math import floor


# custom linear layer which only uses the strongest K synaptic weights from in_features 
class TopKLinear(nn.Linear):
    def __init__(self, in_features, out_features, K, bias = False):    
        # initialize super Linear layer
        super().__init__(in_features, out_features, bias = bias)
        # assign positive integer attribute K, denoting the number of strongest synapses to keep per dendritic branch
        self.K = K
        # make all weights positive
        with torch.no_grad():
            self.weight = torch.sign(self.weight) * self.weight

    
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
    
    def freeze_params(self):
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
        shunting = True, n_input_branches = None,
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
        self.e_i_ratio = excitatory_synapses_per_branch / inhibitory_synapses_per_branch
    
    def forward(self, x, inhibitory_cell_activation, branch_input = None):
        # compute excitation and inhibition along each branch in layer
        excitation = self.branch_excitation(x)
        inhibition = self.branch_inhibition(inhibitory_cell_activation)
        # if receiving inputs from upstream branches
        if self.input_branches:
            # compute depolarization effect and update excitation
            depolarization = self.branches_to_branch(branch_input)
            excitation = excitation + depolarization
        # if shunting activation is used
        if self.shunting:
            # compute shunting activation
            activation = excitation / (excitation + self.e_i_ratio * inhibition + 1)
        else:
            # compute sigmoidal activation
            activation = self.sigmoid(excitation - self.e_i_ratio * inhibition - 3)
        # return activation along branches as output of layer
        return activation


# emulates branching dendritic tree strctures into an artficial neural network analogous to the MLP 
class DendriNet(nn.Module):
    def __init__(
        self, input_dim, n_soma,
        n_branch_layers, branch_factor, init_factor, 
        n_inhibitory_cells, synapses_per_inhibitory_cell, 
        inhibitory_synapses_per_branch, excitatory_synapses_per_branch, shunting,
        ):
        # module for propagating inputs to inhibitory cells
        self.input_inhibitory_cells = TopKLinear(input_dim, n_inhibitory_cells, synapses_per_inhibitory_cell)
        # number of branches in layer closest to soma layer
        layer_sizes = [n_soma * branch_factor**init_factor]
        # number of branches in upstream layers
        for i in range(1, n_branch_layers):
            layer_sizes.append(layer_sizes[i-1] * branch_factor)
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
                shunting = shunting, n_input_branches = layer_sizes[i],
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
        inhibition = self.input_inhibitory_cells(x)
        # if shunting is used, compute shunting activation for inhibitory cell output
        if self.shunting:
            inhibitory_cell_activation = inhibition / (inhibition + 1)
        # if no shunting, compute sigmoidal activation for inhibitory cell output
        else:
            inhibitory_cell_activation = self.sigmoid(inhibition - 3)
        # branch_input to first branch layer is None
        branch_input = None
        # dendritic computation
        for i in range(self.n_branch_layers):
            # recursively update branch_input from one branch layer to the next
            branch_input = self.branch_layers[i](x, inhibitory_cell_activation, branch_input)
        # return information processed at the soma layer
        return self.branches_to_soma(branch_input)
        
        













