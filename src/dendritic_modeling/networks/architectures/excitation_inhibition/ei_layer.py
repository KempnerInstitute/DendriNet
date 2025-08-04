"""
Excitation-Inhibition Layer for dendritic networks.

This module implements the ExcitationInhibitionLayer which models
excitatory and inhibitory cell populations.
"""

from typing import Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.dendrinet import (
    DendriNet,
)


class ExcitationInhibitionLayer(nn.Module):
    def __init__(
        self,
        n_excitatory_cells,
        n_inhibitory_cells,
        excitatory_branch_factors,
        inhibitory_branch_factors,
        excitatory_input_dim,
        ee_synapses_per_branch,
        ei_synapses_per_branch,
        build_inhibitory_cells=True,
        inhibitory_input_dim=None,
        ie_synapses_per_branch=None,
        ii_synapses_per_branch=None,
        reactivate=True,
        reactivation_init_m=1.5,
        reactivation_init_b=0.5,
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        use_shunting=True,
        reactivation_strategy="none",
        blocklinear_strategy="none",
        reactivation_type="param_tanh",
        topk_noise_level=0.0,
        topk_type="standard",
        topk_strategy="none",
        print_hooks=False,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode="global",
        rewire_frequency=1,
        synapses_per_branch=None,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method="xavier_normal",
        weight_threshold=1e-6,
        **kwargs,
    ):
        super().__init__()

        if build_inhibitory_cells:
            self.inhibitory_cells = DendriNet(
                n_soma=n_inhibitory_cells,
                branch_factors=inhibitory_branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=ei_synapses_per_branch,
                inhibitory_input_dim=inhibitory_input_dim,
                inhibitory_synapses_per_branch=ii_synapses_per_branch,
                reactivate=reactivate,
                reactivation_init_m=reactivation_init_m,
                reactivation_init_b=reactivation_init_b,
                somatic_synapses=somatic_synapses,
                topk_init_method=topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
                topk_noise_level=topk_noise_level,
                topk_strategy=topk_strategy,
                topk_type=topk_type,
                print_hooks=print_hooks,
                excitatory_target_density=excitatory_target_density,
                inhibitory_target_density=inhibitory_target_density,
                use_noise=use_noise,
                sigma=sigma,
                rewiring_mode=rewiring_mode,
                rewire_frequency=rewire_frequency,
                synapses_per_branch=synapses_per_branch,
                freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                init_method=init_method,
                weight_threshold=weight_threshold,
                **kwargs,
            )

        self.excitatory_cells = DendriNet(
            n_soma=n_excitatory_cells,
            branch_factors=excitatory_branch_factors,
            excitatory_input_dim=excitatory_input_dim,
            excitatory_synapses_per_branch=ee_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=ie_synapses_per_branch,
            reactivate=reactivate,
            reactivation_init_m=reactivation_init_m,
            reactivation_init_b=reactivation_init_b,
            somatic_synapses=somatic_synapses,
            topk_init_method=topk_init_method,
            use_shunting=use_shunting,
            reactivation_type=reactivation_type,
            reactivation_strategy=reactivation_strategy,
            blocklinear_strategy=blocklinear_strategy,
            topk_noise_level=topk_noise_level,
            topk_strategy=topk_strategy,
            topk_type=topk_type,
            print_hooks=print_hooks,
            excitatory_target_density=excitatory_target_density,
            inhibitory_target_density=inhibitory_target_density,
            use_noise=use_noise,
            sigma=sigma,
            rewiring_mode=rewiring_mode,
            rewire_frequency=rewire_frequency,
            synapses_per_branch=synapses_per_branch,
            freeze_excitatory_connectivity=freeze_excitatory_connectivity,
            freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
            init_method=init_method,
            weight_threshold=weight_threshold,
            **kwargs,
        )

    def decay_weights(self, weight_decay: float):
        if hasattr(self, "inhibitory_cells"):
            self.inhibitory_cells.decay_weights(weight_decay)
        self.excitatory_cells.decay_weights(weight_decay)

    def apply_rewiring(self):
        if hasattr(self, "inhibitory_cells"):
            self.inhibitory_cells.apply_rewiring()
        self.excitatory_cells.apply_rewiring()

    def forward(
        self,
        excitatory_x: torch.Tensor,
        inhibitory_x: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: shape [batch, excitatory_input_dim]
        inhibitory_input: shape [batch, inhibitory_input_dim], or None
        Returns:
            excitatory_output, inhibitory_output
        """
        if hasattr(self, "inhibitory_cells"):
            inhibitory_x = self.inhibitory_cells(excitatory_x, inhibitory_x)

        excitatory_x = self.excitatory_cells(excitatory_x, inhibitory_x)

        return excitatory_x, inhibitory_x


__all__ = ["ExcitationInhibitionLayer"]
