"""
models.py
=========
This module contains the neural network models for the dendritic_modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from itertools import chain

from dendritic_modeling.networks import ExcitationInhibitionNetwork


class BaseModel(nn.Module):
    def __init__(self, net):
        super(BaseModel, self).__init__()
        self.net = net

    def forward(self, x):
        raise NotImplementedError("Forward method must be implemented in subclass.")

    def predict(self, x):
        raise NotImplementedError("Predict method must be implemented in subclass.")
    
    def decay_weights(self, weight_decay):
        """
        Forward this call to self.net if it implements decay_weights.
        """
        if hasattr(self.net, 'decay_weights'):
            self.net.decay_weights(weight_decay)
    
    def get_param_groups(self, optim_dict):
        """
        Get parameter groups for optimizer.
        """
        split_params = optim_dict.get('split_params', False)
        if split_params and isinstance(self.net, ExcitationInhibitionNetwork):
            topk_params = []
            blocklinear_params = []
            reactivation_params = []
            output_layer_params = []

            for ei_layer in self.net.layers:
                exc_cells = ei_layer.excitatory_cells
                inh_cells = ei_layer.inhibitory_cells

                for dendritic_layer in exc_cells.branch_layers:
                    if dendritic_layer.branch_excitation is not None:
                        topk_params.append(dendritic_layer.branch_excitation.parameters())
                    if dendritic_layer.branch_inhibition is not None:
                        topk_params.append(dendritic_layer.branch_inhibition.parameters())
                    
                    if hasattr(dendritic_layer, 'branches_to_output'):
                        blocklinear_params.append(dendritic_layer.branches_to_output.parameters())
                    
                    if dendritic_layer.reactivate:
                        reactivation_params.append(dendritic_layer.reactivation.parameters())
                
                if inh_cells is not None:
                    for dendritic_layer in inh_cells.branch_layers:
                        if dendritic_layer.branch_excitation is not None:
                            topk_params.append(dendritic_layer.branch_excitation.parameters())
                        if dendritic_layer.branch_inhibition is not None:
                            topk_params.append(dendritic_layer.branch_inhibition.parameters())
                        
                        if hasattr(dendritic_layer, 'branches_to_output'):
                            blocklinear_params.append(dendritic_layer.branches_to_output.parameters())
                        
                        if dendritic_layer.reactivate:
                            reactivation_params.append(dendritic_layer.reactivation.parameters())
            
            if hasattr(self.net, 'output_classifier'):
                output_layer_params.append(self.net.output_classifier.parameters())
            if hasattr(self, 'final_fc'):
                output_layer_params.append(self.final_fc.parameters())
            if hasattr(self, 'log_output_scale'):
                output_layer_params.append([self.log_output_scale])

            param_groups = []
            if topk_params:
                param_groups.append({'params': chain(*topk_params), 'lr': optim_dict.get('topk_lr', 0.001)})
            if blocklinear_params:
                param_groups.append({'params': chain(*blocklinear_params), 'lr': optim_dict.get('blocklinear_lr', 0.001)})
            if reactivation_params:
                param_groups.append({'params': chain(*reactivation_params), 'lr': optim_dict.get('reactivation_lr', 0.001)})
            if hasattr(self.net, 'input_net'):
                param_groups.append({'params': self.net.input_net.parameters(), 'lr': optim_dict.get('input_net_lr', 0.001)})
            if output_layer_params:
                param_groups.append({'params': chain(*output_layer_params), 'lr': optim_dict.get('output_layer_lr', 0.001)})
            
            return param_groups
        else:
            return [{'params': self.parameters(), 'lr': optim_dict.get('lr', 0.001)}]
                


class ProbabilisticClassifier(BaseModel):
    """
    A probabilistic classifier that outputs a categorical distribution
    via LogSoftmax.
    """

    def __init__(self, net, output_dim=10, use_output_layer=False):
        super(ProbabilisticClassifier, self).__init__(net)
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer

        # if we want a final FC, create it here
        if self.use_output_layer:
            self.final_fc = nn.Linear(
                net.forward(torch.zeros(1, net.n_branch_layers)).shape[1],
                self.output_dim,
            )
        else:
            self.log_output_scale = nn.Parameter(
                torch.zeros((self.output_dim,)), requires_grad=True
            )

        self.final = nn.LogSoftmax(dim=-1)

    def log_prob(self, x, y):
        dist = self.forward(x, sampler=True)
        return dist.log_prob(y)

    def forward(self, x, y=None, sampler=False):
        # net_output is shape [batch_size, last_excit_size]
        net_output = self.net(x)
        if hasattr(self, "final_fc"):
            net_output = self.final_fc(net_output)  # optional
        else:
            net_output = self.log_output_scale.exp() * net_output

        final_logits = self.final(net_output)
        categorical_dist = Categorical(logits=final_logits)
        if y is None:
            return categorical_dist if sampler else categorical_dist.logits
        else:
            nll = -categorical_dist.log_prob(y)
            return nll

    def predict(self, x, stochastic=False):
        distribution = self.forward(x, y=None, sampler=True)
        if stochastic:
            return distribution.sample()
        else:
            return distribution.logits.argmax(dim=-1)

    def compute_loss(self, x, y):
        nll = self.forward(x, y=y)
        main_loss = nll.mean(dim=0)
        return main_loss

    def forward_with_branch_outputs(self, x):
        net_output, per_layer_acts = self.net.forward_with_branch_outputs(x)
        if hasattr(self, "final_fc"):
            net_output = self.final_fc(net_output)
        return net_output, per_layer_acts


class Classifier(BaseModel):
    """
    A standard classifier that returns raw logits;
    cross-entropy is used for compute_loss.
    """

    def __init__(self, net, output_dim=10, use_output_layer=False):
        super(Classifier, self).__init__(net)
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer

        # Fix: Use the same approach as ProbabilisticClassifier for final_fc
        if self.use_output_layer:
            # Get the output dimension from the network
            with torch.no_grad():
                if hasattr(net, "n_branch_layers"):
                    test_input = torch.zeros(1, net.n_branch_layers)
                else:
                    test_input = torch.zeros(
                        1, net.input_dim if hasattr(net, "input_dim") else 784
                    )
                net_out_dim = net.forward(test_input).shape[1]
                self.final_fc = nn.Linear(net_out_dim, self.output_dim)
        else:
            self.log_output_scale = nn.Parameter(
                torch.zeros((self.output_dim,)), requires_grad=True
            )
        
    def forward(self, x, y=None):
        logits = self.net(x)
        if hasattr(self, "final_fc"):
            logits = self.final_fc(logits)
        else:
            logits = self.log_output_scale.exp() * logits

        return logits

    def predict(self, x, stochastic=False):
        with torch.no_grad():
            logits = self.forward(x)
            return logits.argmax(dim=-1)

    def compute_loss(self, x, y):
        logits = self.forward(x)
        loss_fn = nn.CrossEntropyLoss()
        main_loss = loss_fn(logits, y)
        return main_loss

    def forward_with_branch_outputs(self, x):
        net_output, per_layer_acts = self.net.forward_with_branch_outputs(x)
        if hasattr(self, "final_fc"):
            net_output = self.final_fc(net_output)
        return net_output, per_layer_acts


class Regressor(BaseModel):
    """
    A standard regressor that returns continuous outputs;
    MSE is used for compute_loss.
    """
    def __init__(self, net, output_dim=None, use_output_layer=False, seed=None):
        super(Regressor, self).__init__(net)
        
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer
        self.seed = seed
        if self.use_output_layer:
            # Use a dummy input with correct input dimension
            dummy_input = torch.zeros(1, net.input_dim)  # First layer size is input dim
            self.final_fc = nn.Linear(
                net.forward(dummy_input).shape[1], 
                self.output_dim
            )
            
        if self.seed is not None:
            self.net.set_seed(self.seed)

    def forward(self, x):
        outputs = self.net(x)
        if hasattr(self, 'final_fc'):
            outputs = self.final_fc(outputs)
        return outputs

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def compute_loss(self, x, y):
        outputs = self.forward(x)
        loss_fn = nn.MSELoss()
        main_loss = loss_fn(outputs, y)
        return main_loss

    def forward_with_branch_outputs(self, x):
        net_output, per_layer_acts = self.net.forward_with_branch_outputs(x)
        if hasattr(self, 'final_fc'):
            net_output = self.final_fc(net_output)
        return net_output, per_layer_acts

    def set_seed(self, seed):
        """Set the seed for the network."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class DeepstRegressor(BaseModel):
    """
    A regressor model that incorporates the Bellec 2018 dynamic rewiring algorithm.
    Uses MSE for compute_loss like standard Regressor.
    """
    def __init__(self, net, output_dim=None, use_output_layer=False, seed=None, 
                 rewire_frequency=1):
        super(DeepstRegressor, self).__init__(net)
        
        self.output_dim = output_dim
        self.use_output_layer = use_output_layer
        self.seed = seed
        self.rewire_frequency = rewire_frequency
        self.step_counter = 0
        
        if self.use_output_layer:
            # Use a dummy input with correct input dimension
            dummy_input = torch.zeros(1, net.input_dim)  # First layer size is input dim
            self.final_fc = nn.Linear(
                net.forward(dummy_input).shape[1], 
                self.output_dim
            )
            
        if self.seed is not None:
            self.set_seed(self.seed)

    def forward(self, x):
        outputs = self.net(x)
        if hasattr(self, 'final_fc'):
            outputs = self.final_fc(outputs)
        return outputs

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)

    def compute_loss(self, x, y):
        outputs = self.forward(x)
        loss_fn = nn.MSELoss()
        main_loss = loss_fn(outputs, y)
        return main_loss
    
    def training_step(self, batch, optimizer):
        """Custom training step that incorporates rewiring"""
        x, y = batch
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass and compute loss
        loss = self.compute_loss(x, y)
        
        # Backward pass
        loss.backward()
        
        # Apply rewiring based on frequency
        self.step_counter += 1
        if hasattr(self.net, 'apply_rewiring') and self.step_counter % self.rewire_frequency == 0:
            self.net.apply_rewiring()
        
        # Optimizer step
        optimizer.step()
        
        return loss.item()

    def forward_with_branch_outputs(self, x):
        net_output, per_layer_acts = self.net.forward_with_branch_outputs(x)
        if hasattr(self, 'final_fc'):
            net_output = self.final_fc(net_output)
        return net_output, per_layer_acts

    def set_seed(self, seed):
        """Set the seed for the network."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SimpleMLP(nn.Module):
    """
    A simple PyTorch MLP that's optimized for MNIST classification.
    This will be used to demonstrate good performance on MNIST.
    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Create hidden layers
        for h_dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.BatchNorm1d(h_dim))
            prev_dim = h_dim

        # Create output layer
        layers.append(torch.nn.Linear(prev_dim, output_dim))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        # Ensure input is flattened
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        return self.network(x)


class SimpleClassifier(nn.Module):
    """
    A simple classifier that wraps our SimpleMLP for better performance.
    """

    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.net = SimpleMLP(input_dim, hidden_dims, output_dim)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.net(x)

    def compute_loss(self, x, y):
        logits = self.forward(x)
        return self.criterion(logits, y)


class MLPWithClassificationHead(nn.Module):
    """
    A wrapper around an MLP that adds a classification head for training,
    but allows accessing the feature extractor part separately.
    """

    def __init__(self, feature_extractor, num_features, num_classes):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classification_head = torch.nn.Linear(num_features, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classification_head(features)

    def extract_features(self, x):
        return self.feature_extractor(x)

    def compute_loss(self, x, y):
        logits = self.forward(x)
        return self.criterion(logits, y)
