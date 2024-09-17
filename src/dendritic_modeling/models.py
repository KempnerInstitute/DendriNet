import torch
import torch.nn as nn
from torch.distributions import Categorical

from dendrinet import *


class ProbabilisticClassifier(nn.Module):

    def predict(self, x, deterministic = True):
        distribution = self.forward(x)

        if deterministic:
            return distribution.logits.argmax(dim = -1)
        else:
            return distribution.sample()
    
    def log_prob(self, x, y):
        distribution = self.forward(x)
        return distribution.log_prob(y)


class TopKLogisticRegression(ProbabilisticClassifier):

    def __init__(self, input_dim, n_classes, K, param_freezing = False):
        if param_freezing:
            self.lin = newTopKLinear(input_dim, n_classes, K = K)
        else:
            self.lin = TopKLinear(input_dim, n_classes, K = K)
        
        self.final = nn.LogSoftmax(dim = -1)
    
    def forward(self, x):
        logits = self.final(self.lin(x))
        return Categorical(logits = logits)
    

































