"""
DendriNet architectures for dendritic neural networks.

This module implements the DendriNet and DendriNetWithOutputs classes
which generate linear layers of neurons with sequential dendritic branches.
"""

from copy import deepcopy

import torch
import torch.nn as nn

from dendritic_modeling.networks.activations import resolve_dendritic_activation
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.branch_layer import (
    DendriticBranchLayer,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.synapse_config import (
    DendriticSynapseConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.structured_mask import (
    sample_configured_mask,
)


class DendriNet(nn.Module):
    """
    Generates a linear layer of neurons, each of which consists of a sequential
    structure of dendritic branches and inhibitory/excitatory inputs.

    This network models the complex interactions between excitatory and
    inhibitory inputs across multiple dendritic branches, simulating dendritic
    processing in biological neurons.
    """

    def __init__(
        self,
        n_soma,
        branch_factors,
        excitatory_input_dim,
        excitatory_synapses_per_branch,
        inhibitory_input_dim=None,
        inhibitory_synapses_per_branch=None,
        reactivate=False,
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        topk_noise_level=0.0,
        topk_type="standard",  # New parameter for TopK strategy
        topk_weight_norm_order=None,
        topk_gamma=1.0,
        weight_transform="exp",  # Weight transformation for positive weights
        topk_temperature=0.5,  # For stochastic TopK
        topk_ultrafast=False,  # For much faster stochastic TopK
        topk_strategy="none",
        use_shunting=True,
        reactivation_type="param_tanh",
        dendritic_activation=None,
        reactivation_strategy="none",
        blocklinear_strategy="none",
        print_hooks=False,
        reactivation_init_m=1,
        reactivation_init_b=0.5,
        reactivation_init_policy="analytical",
        reactivation_sigma_aware_k=0.25,
        dbl_init_method="analytical_expectation",
        # DEEPST specific parameters, clean up later
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode="global",
        rewire_frequency=1,
        synapses_per_branch=None,
        efficient_blocklinear: bool = False,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method=None,
        weight_threshold=1e-6,
        adaptive_initialization=True,  # Add adaptive initialization parameter
        adaptive_initialization_policy="legacy_scale",
        adaptive_target_conductance=5.0,
        structured_connectivity=None,
        structured_layer_idx: int = 0,
        structured_excitatory_pathway: str = "ee",
        structured_inhibitory_pathway: str = "ie",
        **kwargs,
    ):
        super().__init__()
        reactivate, reactivation_type = resolve_dendritic_activation(
            dendritic_activation,
            reactivate,
            reactivation_type,
        )

        if not isinstance(n_soma, int):
            raise TypeError("n_soma must be an integer")
        if n_soma < 1:
            raise ValueError("n_soma must be >= 1")

        layer_sizes = [n_soma]

        for bf in branch_factors:
            if not isinstance(bf, int):
                raise TypeError("branch_factors must be integers")
            if bf < 1:
                raise ValueError("branch_factors must be >= 1")

        # Store branch factors for network-aware initialization
        self.branch_factors = deepcopy(branch_factors)
        self.dendritic_activation = reactivation_type if reactivate else "none"

        if (
            inhibitory_input_dim is not None
            and inhibitory_synapses_per_branch is not None
        ):
            if not isinstance(inhibitory_synapses_per_branch, int):
                raise TypeError("inhibitory_synapses_per_branch must be int")
            # if inhibitory_synapses_per_branch < 1:
            #     raise ValueError("inhibitory_synapses_per_branch must be >= 1")

        if (
            excitatory_input_dim is not None
            and excitatory_synapses_per_branch is not None
        ):
            if not isinstance(excitatory_synapses_per_branch, int):
                raise TypeError("excitatory_synapses_per_branch must be int")
            if excitatory_synapses_per_branch < 1:
                raise ValueError("excitatory_synapses_per_branch must be >= 1")

        branch_factors: list = deepcopy(branch_factors)
        self.n_branch_layers = len(branch_factors)
        for i in range(self.n_branch_layers):
            layer_sizes.append(layer_sizes[i] * branch_factors[i])
        layer_sizes.reverse()
        branch_factors.reverse()
        branch_factors.insert(0, None)
        branch_layers = []
        formal_synapse_kwargs = {
            "topk_init_method": topk_init_method,
            "topk_noise_level": topk_noise_level,
            "topk_type": topk_type,
            "topk_weight_norm_order": topk_weight_norm_order,
            "topk_gamma": topk_gamma,
            "topk_temperature": topk_temperature,
            "topk_ultrafast": topk_ultrafast,
            "topk_strategy": topk_strategy,
            "use_shunting": use_shunting,
            "reactivate": reactivate,
            "reactivation_type": reactivation_type,
            "dendritic_activation": dendritic_activation,
            "reactivation_strategy": reactivation_strategy,
            "blocklinear_strategy": blocklinear_strategy,
            "print_hooks": print_hooks,
            "reactivation_init_m": reactivation_init_m,
            "reactivation_init_b": reactivation_init_b,
            "reactivation_init_policy": reactivation_init_policy,
            "reactivation_sigma_aware_k": reactivation_sigma_aware_k,
            "dbl_init_method": dbl_init_method,
            "excitatory_target_density": excitatory_target_density,
            "inhibitory_target_density": inhibitory_target_density,
            "use_noise": use_noise,
            "sigma": sigma,
            "rewiring_mode": rewiring_mode,
            "rewire_frequency": rewire_frequency,
            "synapses_per_branch": synapses_per_branch,
            "efficient_blocklinear": efficient_blocklinear,
            "freeze_excitatory_connectivity": freeze_excitatory_connectivity,
            "freeze_inhibitory_connectivity": freeze_inhibitory_connectivity,
            "init_method": init_method,
            "weight_threshold": weight_threshold,
            "adaptive_initialization": adaptive_initialization,
            "adaptive_initialization_policy": adaptive_initialization_policy,
            "adaptive_target_conductance": adaptive_target_conductance,
            "branch_factors": self.branch_factors,
            "weight_transform": weight_transform,
        }
        synapse_config = DendriticSynapseConfig.from_kwargs(
            {**kwargs, **formal_synapse_kwargs}
        )
        _, branch_extra_kwargs = DendriticSynapseConfig.split_kwargs(kwargs)

        def _structured_mask(pathway, out_dim, in_dim, synapses_per_branch, level_idx):
            if (
                in_dim is None
                or synapses_per_branch is None
                or synapses_per_branch <= 0
            ):
                return None
            return sample_configured_mask(
                structured_connectivity,
                pathway=pathway,
                out_features=out_dim,
                in_features=in_dim,
                layer_idx=structured_layer_idx,
                level_idx=level_idx,
            )

        for i in range(self.n_branch_layers):
            excitatory_connection_mask = _structured_mask(
                structured_excitatory_pathway,
                layer_sizes[i],
                excitatory_input_dim,
                excitatory_synapses_per_branch,
                i,
            )
            inhibitory_connection_mask = _structured_mask(
                structured_inhibitory_pathway,
                layer_sizes[i],
                inhibitory_input_dim,
                inhibitory_synapses_per_branch,
                i,
            )
            lyr = DendriticBranchLayer(
                output_dim=layer_sizes[i],
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=excitatory_synapses_per_branch,
                inhibitory_input_dim=inhibitory_input_dim,
                inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
                excitatory_connection_mask=excitatory_connection_mask,
                inhibitory_connection_mask=inhibitory_connection_mask,
                input_branch_factor=branch_factors[i],
                layer_idx=self.n_branch_layers - i,
                synapse_config=synapse_config,
                **branch_extra_kwargs,
            )
            branch_layers.append(lyr)

        if somatic_synapses:
            soma_level_idx = self.n_branch_layers
            excitatory_connection_mask = _structured_mask(
                structured_excitatory_pathway,
                n_soma,
                excitatory_input_dim,
                excitatory_synapses_per_branch,
                soma_level_idx,
            )
            inhibitory_connection_mask = _structured_mask(
                structured_inhibitory_pathway,
                n_soma,
                inhibitory_input_dim,
                inhibitory_synapses_per_branch,
                soma_level_idx,
            )
            branch_layers.append(
                DendriticBranchLayer(
                    output_dim=n_soma,
                    excitatory_input_dim=excitatory_input_dim,
                    excitatory_synapses_per_branch=excitatory_synapses_per_branch,
                    inhibitory_input_dim=inhibitory_input_dim,
                    inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
                    excitatory_connection_mask=excitatory_connection_mask,
                    inhibitory_connection_mask=inhibitory_connection_mask,
                    input_branch_factor=branch_factors[-1],
                    layer_idx=0,
                    synapse_config=synapse_config,
                    **branch_extra_kwargs,
                )
            )
        else:
            branch_layers.append(
                DendriticBranchLayer(
                    output_dim=n_soma,
                    excitatory_input_dim=None,
                    excitatory_synapses_per_branch=None,
                    inhibitory_input_dim=None,
                    inhibitory_synapses_per_branch=None,
                    input_branch_factor=branch_factors[-1],
                    layer_idx=0,
                    synapse_config=synapse_config,
                    **branch_extra_kwargs,
                )
            )

        self.branch_layers = nn.ModuleList(branch_layers)

        self.reactivate = reactivate
        self.layer_sizes = layer_sizes
        self.input_inhibitory = inhibitory_input_dim is not None
        self.n_soma = n_soma
        self.somatic_synapses = somatic_synapses
        self.efficient_blocklinear = efficient_blocklinear

    def apply_rewiring(self):
        for layer in self.branch_layers:
            layer.apply_rewiring()

    def decay_weights(self, weight_decay, weight_boosting=False):
        for branch_layer in self.branch_layers:
            branch_layer.decay_weights(weight_decay, weight_boosting)

    def forward(self, x, inhibitory_input=None):
        output = None
        for i in range(self.n_branch_layers + 1):
            # Store excitatory and inhibitory inputs for visualization if needed
            # The excitatory input for the current layer is always x
            if hasattr(self.branch_layers[i], "excitatory_input"):
                self.branch_layers[i].excitatory_input = x

            # The inhibitory input is whatever was passed in or None
            if hasattr(self.branch_layers[i], "inhibitory_input"):
                self.branch_layers[i].inhibitory_input = inhibitory_input

            output = self.branch_layers[i](x, inhibitory_input, output)
        return output

    def sum_weights(self, pruned=False):
        exc_total = 0
        if self.input_inhibitory:
            inh_total = 0

        n_layers = self.n_branch_layers + 1
        for i in range(n_layers):
            layer: DendriticBranchLayer = self.branch_layers[i]

            if layer.input_inhibitory and layer.input_excitatory:
                weights = layer.get_pruned_weights() if pruned else layer.get_weights()
                exc = weights["exc"]
                inh = weights["inh"]
                exc_chunk = exc.chunk(self.n_soma, dim=0)
                inh_chunk = inh.chunk(self.n_soma, dim=0)

                exc_total = exc_total + torch.stack(
                    [chunk.sum(0) for chunk in exc_chunk], dim=0
                )
                inh_total = inh_total + torch.stack(
                    [chunk.sum(0) for chunk in inh_chunk], dim=0
                )

            elif layer.input_excitatory:
                weights = layer.get_pruned_weights() if pruned else layer.get_weights()
                exc = weights["exc"]
                exc_chunk = exc.chunk(self.n_soma, dim=0)
                exc_total = exc_total + torch.stack(
                    [chunk.sum(0) for chunk in exc_chunk], dim=0
                )

        if self.input_inhibitory:
            return exc_total, inh_total
        else:
            return exc_total

    def log_sum_weights(self, pruned=False):
        if self.input_inhibitory:
            exc_total, inh_total = self.sum_weights(pruned=pruned)
            return (exc_total + 1e-8).log(), (inh_total + 1e-8).log()
        else:
            exc_total = self.sum_weights(pruned=pruned)
            return (exc_total + 1e-8).log()


__all__ = ["DendriNet"]
