from torch import nn


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.output_dim: int

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclass.")
    
    def get_output_dim(self):
        raise NotImplementedError("get_output_dim method must be implemented in subclass.")