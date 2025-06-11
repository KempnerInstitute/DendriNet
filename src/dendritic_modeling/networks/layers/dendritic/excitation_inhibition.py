"""
Excitation-Inhibition Layer for dendritic networks.

This module implements the ExcitationInhibitionLayer which models
excitatory and inhibitory cell populations.
"""

import torch
import torch.nn as nn

from dendritic_modeling.dendrinet import DendriNet, DendriNetWithOutputs


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
        ii_connectivity=False,
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
        enable_branch_outputs=False,
        topk_noise_level=0.0,
        topk_type="standard",
        topk_strategy="none",
        print_hooks=False,
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode="global",
        synapses_per_branch=None,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method="xavier_normal",
        weight_threshold=1e-6,
        **kwargs,
    ):
        super().__init__()

        # Decide if we build an inhibitory sub-network
        self.has_inhib_subnet = (
            n_inhibitory_cells is not None
            and isinstance(n_inhibitory_cells, int)
            and n_inhibitory_cells > 0
        )

        if self.has_inhib_subnet:
            if enable_branch_outputs:
                self.inhibitory_cells = DendriNetWithOutputs(
                    n_soma=n_inhibitory_cells,
                    branch_factors=inhibitory_branch_factors,
                    excitatory_input_dim=excitatory_input_dim,
                    excitatory_synapses_per_branch=ei_synapses_per_branch,
                    inhibitory_input_dim=inhibitory_input_dim
                    if ii_connectivity
                    else None,
                    inhibitory_synapses_per_branch=ii_synapses_per_branch,
                    reactivate=reactivate,
                    somatic_synapses=somatic_synapses,
                    topk_init_method=topk_init_method,
                    use_shunting=use_shunting,
                    reactivation_type=reactivation_type,
                    reactivation_init_m=reactivation_init_m,
                    reactivation_init_b=reactivation_init_b,
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
                    synapses_per_branch=synapses_per_branch,
                    freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                    freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                    init_method=init_method,
                    weight_threshold=weight_threshold,
                    **kwargs,
                )
            else:
                self.inhibitory_cells = DendriNet(
                    n_soma=n_inhibitory_cells,
                    branch_factors=inhibitory_branch_factors,
                    excitatory_input_dim=excitatory_input_dim,
                    excitatory_synapses_per_branch=ei_synapses_per_branch,
                    inhibitory_input_dim=inhibitory_input_dim
                    if ii_connectivity
                    else None,
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
                    synapses_per_branch=synapses_per_branch,
                    freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                    freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                    init_method=init_method,
                    weight_threshold=weight_threshold,
                    **kwargs,
                )
        else:
            self.inhibitory_cells = None

        if enable_branch_outputs:
            self.excitatory_cells = DendriNetWithOutputs(
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
                synapses_per_branch=synapses_per_branch,
                freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                init_method=init_method,
                weight_threshold=weight_threshold,
                **kwargs,
            )
        else:
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
                synapses_per_branch=synapses_per_branch,
                freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                init_method=init_method,
                weight_threshold=weight_threshold,
                **kwargs,
            )

    def decay_weights(self, weight_decay):
        if self.inhibitory_cells is not None:
            self.inhibitory_cells.decay_weights(weight_decay)
        self.excitatory_cells.decay_weights(weight_decay)

    def forward(self, x, inhibitory_input=None):
        """
        x: shape [batch, excitatory_input_dim]
        inhibitory_input: shape [batch, inhibitory_input_dim], or None
        Returns:
            excitatory_output, inhibitory_output
        """
        if self.has_inhib_subnet:
            inhibitory_output = self.inhibitory_cells(x, inhibitory_input)
        else:
            # If no inhibitory subnet, just pass through the inhibitory input
            inhibitory_output = inhibitory_input

        # Pass the inhibitory output to excitatory cells
        # The DendriNet architecture handles the proper connections internally
        # through the inhibitory_synapses_per_branch parameter
        excitatory_output = self.excitatory_cells(x, inhibitory_output)

        return excitatory_output, inhibitory_output

    def forward_with_branch_outputs(self, x, inhibitory_input=None):
        if self.inhibitory_cells is not None and hasattr(
            self.inhibitory_cells, "forward_with_branch_outputs"
        ):
            (
                inhibitory_final,
                inh_acts,
            ) = self.inhibitory_cells.forward_with_branch_outputs(x, inhibitory_input)
        elif self.inhibitory_cells is not None:
            inhibitory_final = self.inhibitory_cells(x, inhibitory_input)
            inh_acts = []
        else:
            # When there are no inhibitory cells, pass the original inhibitory_input
            inhibitory_final = inhibitory_input
            inh_acts = []

        # Additional safety checks for inhibitory_final
        if not isinstance(inhibitory_final, torch.Tensor):
            # Set to None so it gets properly handled in excitatory_cells
            inhibitory_final = None
        elif inhibitory_final.dim() == 0:
            # If it's a scalar tensor, set to None
            inhibitory_final = None
        elif inhibitory_final.dim() == 1:
            # If it's a 1D tensor, make it 2D [1, features]
            inhibitory_final = inhibitory_final.unsqueeze(0)

        # Ensure x is a properly formed 2D tensor
        if x.dim() == 1:
            x = x.unsqueeze(0)

        excitatory_final, exc_acts = self.excitatory_cells.forward_with_branch_outputs(
            x, inhibitory_final
        )

        # Enhance the activation information for better visualization
        # Try to enhance the excitatory activations with information about inhibitory components
        enhanced_acts = []
        for i, act in enumerate(exc_acts):
            if isinstance(act, torch.Tensor):
                # Get corresponding inhibitory activation if available
                inh_act = inh_acts[i] if i < len(inh_acts) else None

                # Create a list containing both excitatory and inhibitory information
                if inh_act is not None:
                    # Include both components plus a placeholder for combined output
                    enhanced_acts.append([act, inh_act, None])
                else:
                    enhanced_acts.append(act)
            else:
                enhanced_acts.append(act)

        return excitatory_final, inhibitory_final, inh_acts, enhanced_acts


__all__ = ["ExcitationInhibitionLayer"] 