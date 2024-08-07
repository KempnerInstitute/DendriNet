import torch
import torch.nn as nn
from torch.distributions import Categorical

from dendritic_modeling.dendrinet import *



class ProbabilisticModel(nn.Module):

    def log_prob(self, x, y):
        distribution = self.forward(x)
        return distribution.log_prob(y)


class ProbabilisticClassifier(ProbabilisticModel):

    def predict(self, x, stochastic = False):
        distribution = self.forward(x)

        if stochastic:
            return distribution.sample()
        else:
            return distribution.logits.argmax(dim = -1)


class ProbabilisticRegressor(ProbabilisticModel):

    def predict(self, x, stochastic = True):
        distribution = self.forward(x)

        if stochastic:
            return distribution.sample()
        else:
            return distribution.mean


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
    

































