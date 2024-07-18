import torch
import torch.nn as nn

import numpy as np
from math import floor



#@title Define layer class that only keeps K top weights
class TopKLinear(nn.Linear):
    """ Custom Linear layer that only uses the stongest K weights """
    def __init__(self, in_features, out_features, keep_in_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        self.keep_in_features = keep_in_features
        self.topk_indices=[]
        self.mask=0*self.weight.data
        self.net_weight=0*self.weight.data

    def forward(self, x):
        self.mask=0*self.weight.data
        self.topk_indices=torch.topk(self.weight, self.keep_in_features, 1, largest=True)[1]
        self.mask[torch.arange(self.weight.shape[0])[:,None], self.topk_indices] = 1  ##  Change those values to 1
        self.net_weight=self.mask*self.weight
        w_times_x= torch.mm(x, self.net_weight.t())
        return w_times_x



#@title Define layer class that uses a block structure
class BlockLinear(nn.Linear):
    """ Custom Linear layer that only uses the stongest K weights """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__(in_features, out_features, bias=bias)
        block_size=floor(in_features/out_features)
        assert(block_size*out_features==in_features)
        self.block_size = block_size

        self.mask=torch.zeros(out_features, in_features, device='cuda:0')
        for counter in range(self.out_features):
            self.mask[counter, torch.arange(counter*self.block_size, (1+counter)*self.block_size)]=1

    def forward(self, x):
        self.net_weight=self.weight*self.mask
        w_times_x= torch.mm(x, self.net_weight.t())
        return w_times_x



#@title Define dendrite branch layer
class DendriteBranchLayer(nn.Module):
    """ """
    def __init__(
        self, inhibitory_units_n, dendrite_branch_n, inhibitory_synapses_per_dendrite_branch,
        excitatory_synapses_per_dendrite_branch, shunting = False, branch_input = False, input_dendrite_branch_n = None,
        ):
        super(DendriteBranchLayer, self).__init__()

        self.branch_input = branch_input
        self.shunting = shunting

        self.inputs_to_dendrite_branch = TopKLinear(input_size, dendrite_branch_n, excitatory_synapses_per_dendrite_branch)

        self.inhibition_to_dendrite_branch = TopKLinear(inhibitory_units_n, dendrite_branch_n, inhibitory_synapses_per_dendrite_branch)

        if self.branch_input:
            self.dendrite_branch_inputs_to_dendrite_branch = BlockLinear(input_dendrite_branch_n, dendrite_branch_n)
            self.dendrite_branch_inputs_to_dendrite_branch.requires_grad_(False) # we don't train this.  All dendrites have the same weight

        self.sigmoid=nn.Sigmoid()

    def forward(self, input, inhibition_outputs, dendrite_branch_outputs=None):

        self.dendrite_inhibition = self.inhibition_to_dendrite_branch(inhibition_outputs)
        self.dendrite_excitation = self.inputs_to_dendrite_branch(input)

        if self.branch_input:
            self.dendrite_depolarization = self.dendrite_branch_inputs_to_dendrite_branch(dendrite_branch_outputs)
            total_excitatory_input = self.dendrite_excitation + self.dendrite_depolarization
        else:
            total_excitatory_input = self.dendrite_excitation

        if self.shunting:
            self.dendrite_act = (total_excitatory_input)/     \
            (total_excitatory_input + e_to_i_ratio*self.dendrite_inhibition + 1)
            self.dendrite_out=self.dendrite_act
        else:
            self.dendrite_act = total_excitatory_input - e_to_i_ratio*self.dendrite_inhibition - 3
            self.dendrite_out = self.sigmoid(self.dendrite_act)

        return self.dendrite_out



#@title Define network structure

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        # inhibitory cell layer
        self.inputs_to_inhibition=TopKLinear(input_size, inhibitory_units_n, synapses_per_inhibitory_unit) # only XX inputs to each inhibitory cell

        # dendrite layers
        self.dendrite_layers = nn.ModuleList([None]*dendrite_layers_n)
        for i in range(dendrite_layers_n - 1 , -1, -1):

            is_outermost_layer = (i == (dendrite_layers_n - 1))
            branch_input = False if is_outermost_layer else True
            input_dendrite_branch_n = None if is_outermost_layer else dendrite_branch_n[i+1]

            self.dendrite_layers[i] = DendriteBranchLayer(
                inhibitory_units_n, dendrite_branch_n[i], inhibitory_synapses_per_dendrite_branch,
                excitatory_synapses_per_dendrite_branch, shunting, branch_input, input_dendrite_branch_n,
                )

        # soma
        self.dendrite_to_soma=BlockLinear(dendrite_branch_n[0], soma_n)
        self.dendrite_to_soma.requires_grad_(False) # we don't train this.  All dendrites have the same weight

        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

    def initWeights(self):

        with torch.no_grad():
            self.inputs_to_inhibition.weight.data=torch.sign(self.inputs_to_inhibition.weight)*self.inputs_to_inhibition.weight

        for i in range(dendrite_layers_n - 1 , -1, -1):

            self.dendrite_layers[i].inputs_to_dendrite_branch.weight.data=torch.sign(self.dendrite_layers[i].inputs_to_dendrite_branch.weight)*self.dendrite_layers[i].inputs_to_dendrite_branch.weight # all +
            self.dendrite_layers[i].inhibition_to_dendrite_branch.weight.data=torch.sign(self.dendrite_layers[i].inhibition_to_dendrite_branch.weight)*self.dendrite_layers[i].inhibition_to_dendrite_branch.weight # all +

            is_outermost_layer = (i == (dendrite_layers_n - 1))
            if not is_outermost_layer:
                self.dendrite_layers[i].dendrite_branch_inputs_to_dendrite_branch.weight.data=1.0*self.dendrite_layers[i].dendrite_branch_inputs_to_dendrite_branch.mask # all 1

        self.dendrite_to_soma.weight.data=1.0*self.dendrite_to_soma.mask # all 1

    def trimWeights(self):

        with torch.no_grad():
            self.inputs_to_inhibition.weight.data=torch.maximum(self.inputs_to_inhibition.weight, 0*self.inputs_to_inhibition.weight) # all +

        for i in range(dendrite_layers_n - 1 , -1, -1):
            self.dendrite_layers[i].inputs_to_dendrite_branch.weight.data=torch.maximum(self.dendrite_layers[i].inputs_to_dendrite_branch.weight, 0*self.dendrite_layers[i].inputs_to_dendrite_branch.weight) # all +
            self.dendrite_layers[i].inhibition_to_dendrite_branch.weight.data=torch.maximum(self.dendrite_layers[i].inhibition_to_dendrite_branch.weight, 0*self.dendrite_layers[i].inhibition_to_dendrite_branch.weight) # all +


    def forward(self, input):
        self.inhibition_act = self.inputs_to_inhibition(input)
        
        if shunting:
            self.inhibition_out = self.inhibition_act / (1 + self.inhibition_act)
        else:
            self.inhibition_out = self.sigmoid(self.inhibition_act - sigmoid_offset)

        self.dendrite_branch_out = [None] * dendrite_layers_n

        for i in range(dendrite_layers_n - 1 , -1, -1):

            is_outermost_layer = (i == (dendrite_layers_n - 1))
            this_dendrite_branch_out = None if is_outermost_layer else self.dendrite_branch_out[i + 1]

            self.dendrite_branch_out[i] = self.dendrite_layers[i](input, self.inhibition_out, this_dendrite_branch_out)

        self.soma_act = self.dendrite_to_soma(self.dendrite_branch_out[0])
        # self.soma_out = self.sigmoid(self.soma_act - sigmoid_offset)

        return self.soma_act












