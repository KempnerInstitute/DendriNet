"""
Excitation-Inhibition Layer for dendritic networks.

This module implements the ExcitationInhibitionLayer which models
excitatory and inhibitory cell populations.
"""

from typing import Any, Optional

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP
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
        inhibitory_network_type="dendritic",
        inhibitory_input_dim=None,
        ie_synapses_per_branch=None,
        ii_synapses_per_branch=None,
        reactivate=True,
        reactivation_init_m=1.5,
        reactivation_init_b=0.5,
        dbl_init_method="analytical_expectation",
        somatic_synapses=True,
        topk_init_method="xavier_normal",
        use_shunting=True,
        synapse_mode: str = "ei",
        reactivation_strategy="none",
        blocklinear_strategy="none",
        reactivation_type="param_tanh",
        topk_noise_level=0.0,
        topk_type="standard",
        topk_weight_norm_order=None,
        topk_gamma=1.0,
        weight_transform="exp",  # Weight transformation for positive weights
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
        adaptive_initialization=True,
        **kwargs,
    ):
        super().__init__()
        if synapse_mode not in ["ei", "mlp"]:
            raise ValueError(f"synapse_mode must be 'ei' or 'mlp', got: {synapse_mode}")
        self.synapse_mode = synapse_mode

        # Only build inhibitory cells if n_inhibitory_cells is not None and build_inhibitory_cells is True
        if build_inhibitory_cells and n_inhibitory_cells is not None:
            assert inhibitory_network_type in ["dendritic", "mlp"], (
                "inhibitory_network_type must be either 'dendritic' or 'mlp'."
                + f"Got inhibitory_network_type: {inhibitory_network_type}"
            )
            if inhibitory_network_type == "dendritic":
                if self.synapse_mode == "ei":
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
                        dbl_init_method=dbl_init_method,
                        somatic_synapses=somatic_synapses,
                        topk_init_method=topk_init_method,
                        use_shunting=use_shunting,
                        reactivation_type=reactivation_type,
                        reactivation_strategy=reactivation_strategy,
                        blocklinear_strategy=blocklinear_strategy,
                        topk_noise_level=topk_noise_level,
                        topk_strategy=topk_strategy,
                        topk_type=topk_type,
                        topk_weight_norm_order=topk_weight_norm_order,
                        topk_gamma=topk_gamma,
                        weight_transform=weight_transform,
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
                        adaptive_initialization=adaptive_initialization,
                        **kwargs,
                    )
                else:
                    # "dendritic_mlp": merge E/I inputs into one signed synapse bank
                    inh_in_dim = int(excitatory_input_dim) + int(
                        inhibitory_input_dim or 0
                    )
                    k_total = int(ei_synapses_per_branch or 0) + int(
                        ii_synapses_per_branch or 0
                    )
                    self.inhibitory_cells = DendriNet(
                        n_soma=n_inhibitory_cells,
                        branch_factors=inhibitory_branch_factors,
                        excitatory_input_dim=inh_in_dim,
                        excitatory_synapses_per_branch=k_total,
                        inhibitory_input_dim=None,
                        inhibitory_synapses_per_branch=None,
                        reactivate=reactivate,
                        reactivation_init_m=reactivation_init_m,
                        reactivation_init_b=reactivation_init_b,
                        dbl_init_method=dbl_init_method,
                        somatic_synapses=somatic_synapses,
                        topk_init_method=topk_init_method,
                        use_shunting=use_shunting,
                        reactivation_type=reactivation_type,
                        reactivation_strategy=reactivation_strategy,
                        blocklinear_strategy=blocklinear_strategy,
                        topk_noise_level=topk_noise_level,
                        topk_strategy=topk_strategy,
                        topk_type=topk_type,
                        topk_weight_norm_order=topk_weight_norm_order,
                        topk_gamma=topk_gamma,
                        weight_transform=weight_transform,
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
                        adaptive_initialization=adaptive_initialization,
                        **kwargs,
                    )
            elif inhibitory_network_type == "mlp":
                inh_mlp_kwargs: dict[str, Any] = kwargs.get(
                    "mlp_inhibitory_network_params", {}
                )
                inh_mlp = MLP(
                    input_dim=excitatory_input_dim,
                    hidden_dims=inh_mlp_kwargs.get("hidden_dims", []),
                    activation=inh_mlp_kwargs.get("activation", "relu"),
                    output_dim=n_inhibitory_cells,
                )
                nn.init.xavier_normal_(inh_mlp.layers[-1].weight)
                nn.init.zeros_(inh_mlp.layers[-1].bias)
                self.inhibitory_cells = nn.Sequential(inh_mlp, nn.Sigmoid())
        else:
            self.inhibitory_cells = None

        # Excitatory cells may receive inhibitory input from the same layer or transferred pathway.
        # In MLP-synapse mode this must be gated by IE connectivity; otherwise concatenating
        # stale inhibitory tensors can create shape mismatches across layers.
        excitatory_inhibitory_input_dim = (
            n_inhibitory_cells
            if (build_inhibitory_cells and n_inhibitory_cells is not None)
            else inhibitory_input_dim
        )
        self.concat_inhibitory_to_excitatory = bool(
            excitatory_inhibitory_input_dim is not None
            and int(ie_synapses_per_branch or 0) > 0
        )
        if not self.concat_inhibitory_to_excitatory:
            excitatory_inhibitory_input_dim = None

        if self.synapse_mode == "ei":
            self.excitatory_cells = DendriNet(
                n_soma=n_excitatory_cells,
                branch_factors=excitatory_branch_factors,
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=ee_synapses_per_branch,
                inhibitory_input_dim=excitatory_inhibitory_input_dim,
                inhibitory_synapses_per_branch=ie_synapses_per_branch,
                reactivate=reactivate,
                reactivation_init_m=reactivation_init_m,
                reactivation_init_b=reactivation_init_b,
                dbl_init_method=dbl_init_method,
                somatic_synapses=somatic_synapses,
                topk_init_method=topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
                topk_noise_level=topk_noise_level,
                topk_strategy=topk_strategy,
                topk_type=topk_type,
                topk_weight_norm_order=topk_weight_norm_order,
                topk_gamma=topk_gamma,
                weight_transform=weight_transform,
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
                adaptive_initialization=adaptive_initialization,
                **kwargs,
            )
        else:
            exc_in_dim = int(excitatory_input_dim) + int(
                excitatory_inhibitory_input_dim or 0
            )
            k_total = int(ee_synapses_per_branch or 0) + int(
                ie_synapses_per_branch or 0
            )
            self.excitatory_cells = DendriNet(
                n_soma=n_excitatory_cells,
                branch_factors=excitatory_branch_factors,
                excitatory_input_dim=exc_in_dim,
                excitatory_synapses_per_branch=k_total,
                inhibitory_input_dim=None,
                inhibitory_synapses_per_branch=None,
                reactivate=reactivate,
                reactivation_init_m=reactivation_init_m,
                reactivation_init_b=reactivation_init_b,
                dbl_init_method=dbl_init_method,
                somatic_synapses=somatic_synapses,
                topk_init_method=topk_init_method,
                use_shunting=use_shunting,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
                topk_noise_level=topk_noise_level,
                topk_strategy=topk_strategy,
                topk_type=topk_type,
                topk_weight_norm_order=topk_weight_norm_order,
                topk_gamma=topk_gamma,
                weight_transform=weight_transform,
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
                adaptive_initialization=adaptive_initialization,
                **kwargs,
            )

    def decay_weights(self, weight_decay: float, weight_boosting=False):
        if self.inhibitory_cells is not None and hasattr(
            self.inhibitory_cells, "decay_weights"
        ):
            self.inhibitory_cells.decay_weights(weight_decay, weight_boosting)
        self.excitatory_cells.decay_weights(weight_decay, weight_boosting)

    def apply_rewiring(self):
        if self.inhibitory_cells is not None and hasattr(
            self.inhibitory_cells, "apply_rewiring"
        ):
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
        if self.synapse_mode == "mlp":
            # Merge inputs into one signed synapse bank.
            def _cat(a: torch.Tensor, b: Optional[torch.Tensor]) -> torch.Tensor:
                return torch.cat([a, b], dim=-1) if b is not None else a

            if isinstance(self.inhibitory_cells, DendriNet):
                inh_in = _cat(excitatory_x, inhibitory_x)
                inhibitory_x = self.inhibitory_cells(inh_in, None)
            elif isinstance(self.inhibitory_cells, nn.Sequential):
                inhibitory_x = self.inhibitory_cells(excitatory_x)

            exc_in = (
                _cat(excitatory_x, inhibitory_x)
                if self.concat_inhibitory_to_excitatory
                else excitatory_x
            )
            excitatory_x = self.excitatory_cells(exc_in, None)
        else:
            if isinstance(self.inhibitory_cells, DendriNet):
                inhibitory_x = self.inhibitory_cells(excitatory_x, inhibitory_x)
            elif isinstance(self.inhibitory_cells, nn.Sequential):
                inhibitory_x = self.inhibitory_cells(excitatory_x)

            excitatory_x = self.excitatory_cells(excitatory_x, inhibitory_x)

        return excitatory_x, inhibitory_x


__all__ = ["ExcitationInhibitionLayer"]
