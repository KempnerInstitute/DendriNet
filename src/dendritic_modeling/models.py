"""
models.py
=========
This module contains the neural network models for the dendritic_modeling.
"""

import torch.nn as nn
from torch.distributions import Categorical

from dendritic_modeling.excitation_inhibition import ExcitationInhibitionNetwork


class ProbabilisticModel(nn.Module):
    """
    This class is designed to represent models that return a conditional 
    probability distribution of observing labels given their corresponding 
    inputs.
    
    Methods
    -------
    log_prob(x, y)
        Computes the log-probability of the data `y` given input `x` by 
        applying the model's forward method to `x` to obtain a probability
        distribution and then evaluating the log probability of `y`.
    """

    def log_prob(self, x, y):
        """
        Computes the log-probability of `y` given `x`.
        
        Parameters
        ----------
        x : torch.Tensor
            The input data which will be passed through the model's forward 
            method to generate a probability distribution.
        y : torch.Tensor
            The observed data for which we want to compute the log-probability.
        
        Returns
        -------
        torch.Tensor
            The log-probability of `y` given `x` according to the distribution 
            output by the model's forward pass.
        """
        distribution = self.forward(x)
        return distribution.log_prob(y)


class ProbabilisticClassifier(ProbabilisticModel):
    """
    A base class for probabilistic classifiers.

    This class extends `ProbabilisticModel` and provides a common interface for 
    classifiers that output probability distributions over classes. It includes 
    a method for making predictions either deterministically or stochastically 
    based on the output distribution.

    Methods
    -------
    predict(x, stochastic=False)
        Predicts class labels for the given input data.

    See Also
    --------
    ProbabilisticModel : Base class for probabilistic models.

    """

    def predict(self, x, stochastic = False):
        distribution = self.forward(x)

        if stochastic:
            return distribution.sample()
        else:
            return distribution.logits.argmax(dim = -1)


class ProbabilisticRegressor(ProbabilisticModel):
    """
    A base class for probabilistic regressors.

    This class extends `ProbabilisticModel` and provides a common interface for 
    regressors that output probability distributions over continuous targets. 
    It includes a method for making predictions either stochastically (by 
    sampling from the distribution) or deterministically (using the mean of 
    the distribution).

    Methods
    -------
    predict(x, stochastic=True)
        Predicts continuous target values for the given input data.

    See Also
    --------
    ProbabilisticModel : Base class for probabilistic models.
    """

    def predict(self, x, stochastic = True):
        distribution = self.forward(x)

        if stochastic:
            return distribution.sample()
        else:
            return distribution.mean


class EINetClassifier(ProbabilisticClassifier):
    """
    A probabilistic classifier based on an Excitation-Inhibition Network (EINet) 
    architecture.

    This classifier utilizes an underlying `ExcitationInhibitionNetwork` to 
    model complex interactions between excitatory and inhibitory neurons for 
    classification tasks.

    Parameters
    ----------
    input_dim : int
        The dimensionality of the input features.

    excitatory_layer_sizes : list of int
        A list specifying the number of excitatory neurons in each layer.

    inhibitory_layer_sizes : list of int
       A list specifying the number of inhibitory neurons in each layer.

    excitatory_branch_factors : list of int
        Branching factors for excitatory neurons at each layer, determining 
        how they branch out to the next layer.

    inhibitory_branch_factors : list of int
        Branching factors for inhibitory neurons at each layer, determining
        how they branch out to the next layer.

    ee_synapses_per_branch_per_layer : list of int
        Number of excitatory-to-excitatory synapses per branch at each layer.

    ei_synapses_per_branch_per_layer : list of int
        Number of excitatory-to-inhibitory synapses per branch at each layer.

    ie_synapses_per_branch_per_layer : list of int
        Number of inhibitory-to-excitatory synapses per branch at each layer.

    ii_synapses_per_branch_per_layer : list of int, optional
        Number of inhibitory-to-inhibitory synapses per branch at each layer. 
        Defaults to an empty list `[]`.

    reactivate : bool, optional
        If True, applies a reactivation function simulating dendritic/somatic 
        spiking behavior. Defaults to `False`.

    somatic_synapses : bool, optional
        If `True`, includes direct synaptic inputs to the soma layer. Defaults 
        to `True`.

    Methods
    -------
    forward(x)
        Performs a forward pass through the network and returns a categorical 
        distribution over the classes.

    Examples
    --------
    
    >>> import torch
    >>> classifier = EINetClassifier(
    ...     input_dim=784,
    ...     excitatory_layer_sizes=[10,],
    ...     inhibitory_layer_sizes=[20,],
    ...     excitatory_branch_factors=[24,],
    ...     inhibitory_branch_factors=[],
    ...     ee_synapses_per_branch_per_layer=[24,],
    ...     ei_synapses_per_branch_per_layer=[100,],
    ...     ie_synapses_per_branch_per_layer=[1,],
    ...     ii_synapses_per_branch_per_layer=[],
    ...     reactivate=True,
    ...     somatic_synapses=True,
    ... )
    >>> x = torch.randn(10, 784)  # Batch of 10 samples with 784 features
    >>> output = classifier.forward(x)
    >>> predicted_classes = output.sample()
    >>> print(predicted_classes.shape)
    torch.Size([10])
    """

    def __init__(
        self, input_dim, excitatory_layer_sizes, inhibitory_layer_sizes,
        excitatory_branch_factors, inhibitory_branch_factors, 
        ee_synapses_per_branch_per_layer, ei_synapses_per_branch_per_layer, 
        ie_synapses_per_branch_per_layer, ii_synapses_per_branch_per_layer = [], 
        reactivate = False, somatic_synapses = True,
    ):
        super(EINetClassifier, self).__init__()

        self.net = ExcitationInhibitionNetwork(
            input_dim = input_dim, 
            excitatory_layer_sizes = excitatory_layer_sizes, 
            inhibitory_layer_sizes = inhibitory_layer_sizes,
            excitatory_branch_factors = excitatory_branch_factors, 
            inhibitory_branch_factors = inhibitory_branch_factors,
            ee_synapses_per_branch_per_layer = ee_synapses_per_branch_per_layer, 
            ei_synapses_per_branch_per_layer = ei_synapses_per_branch_per_layer,
            ie_synapses_per_branch_per_layer = ie_synapses_per_branch_per_layer, 
            ii_synapses_per_branch_per_layer = ii_synapses_per_branch_per_layer,
            reactivate = reactivate, somatic_synapses = somatic_synapses,
        )
        self.final = nn.LogSoftmax(dim = -1)
    
    def forward(self, x):
        return Categorical(logits = self.final(self.net(x)))


