"""
DendriNet architectures for dendritic neural networks.

This module implements the DendriNet and DendriNetWithOutputs classes
which generate linear layers of neurons with sequential dendritic branches.
"""

import torch
import torch.nn as nn
from copy import deepcopy

from ...layers.dendritic import DendriticBranchLayer


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
        topk_temperature=0.5,  # For stochastic TopK
        topk_ultrafast=False,  # For much faster stochastic TopK
        topk_strategy="none",
        use_shunting=True,
        reactivation_type="param_tanh",
        reactivation_strategy="none",
        blocklinear_strategy="none",
        print_hooks=False,
        reactivation_init_m = 1,
        reactivation_init_b = 0.5,
        # DEEPST specific parameters, clean up later
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode='global',
        synapses_per_branch=None,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method=None,
        weight_threshold=1e-6,
        **kwargs,
    ):
        super(DendriNet, self).__init__()

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

        if (
            inhibitory_input_dim is not None
            and inhibitory_synapses_per_branch is not None
        ):
            if not isinstance(inhibitory_synapses_per_branch, int):
                raise TypeError("inhibitory_synapses_per_branch must be int")
            if inhibitory_synapses_per_branch < 1:
                raise ValueError("inhibitory_synapses_per_branch must be >= 1")

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
        for i in range(self.n_branch_layers):
            lyr = DendriticBranchLayer(
                output_dim=layer_sizes[i],
                excitatory_input_dim=excitatory_input_dim,
                excitatory_synapses_per_branch=excitatory_synapses_per_branch,
                inhibitory_input_dim=inhibitory_input_dim,
                inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
                input_branch_factor=branch_factors[i],
                topk_init_method=topk_init_method,
                topk_noise_level=topk_noise_level,
                topk_type=topk_type,  # Pass the parameter
                topk_temperature=topk_temperature,  # Pass the temperature
                topk_ultrafast=topk_ultrafast,  # Pass the ultrafast flag
                topk_strategy=topk_strategy,
                use_shunting=use_shunting,
                reactivate=reactivate,
                reactivation_init_m=reactivation_init_m,
                reactivation_init_b=reactivation_init_b,
                reactivation_type=reactivation_type,
                reactivation_strategy=reactivation_strategy,
                blocklinear_strategy=blocklinear_strategy,
                layer_idx=self.n_branch_layers - i,
                print_hooks=print_hooks,
                use_noise=use_noise,
                rewiring_mode=rewiring_mode,
                synapses_per_branch=synapses_per_branch,
                excitatory_target_density=excitatory_target_density,
                inhibitory_target_density=inhibitory_target_density,
                freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                init_method=init_method,
                weight_threshold=weight_threshold,
                **kwargs
            )
            branch_layers.append(lyr)

        if somatic_synapses:
            branch_layers.append(
                DendriticBranchLayer(
                    output_dim=n_soma,
                    excitatory_input_dim=excitatory_input_dim,
                    excitatory_synapses_per_branch=excitatory_synapses_per_branch,
                    inhibitory_input_dim=inhibitory_input_dim,
                    inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
                    input_branch_factor=branch_factors[-1],
                    topk_init_method=topk_init_method,
                    topk_noise_level=topk_noise_level,
                    topk_type=topk_type,  # Pass the parameter
                    topk_temperature=topk_temperature,  # Pass the temperature
                    topk_ultrafast=topk_ultrafast,  # Pass the ultrafast flag
                    topk_strategy=topk_strategy,
                    use_shunting=use_shunting,
                    reactivate=reactivate,
                    reactivation_init_m=reactivation_init_m,
                    reactivation_init_b=reactivation_init_b,
                    reactivation_type=reactivation_type,
                    reactivation_strategy=reactivation_strategy,
                    blocklinear_strategy=blocklinear_strategy,
                    layer_idx=0,
                    print_hooks=print_hooks,
                    use_noise=use_noise,
                    rewiring_mode=rewiring_mode,
                    synapses_per_branch=synapses_per_branch,
                    excitatory_target_density=excitatory_target_density,
                    inhibitory_target_density=inhibitory_target_density,
                    freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                    freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                    init_method=init_method,
                    weight_threshold=weight_threshold,
                    **kwargs
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
                    topk_init_method=topk_init_method,
                    topk_noise_level=topk_noise_level,
                    topk_type=topk_type,  # Pass the parameter
                    topk_temperature=topk_temperature,  # Pass the temperature
                    topk_ultrafast=topk_ultrafast,  # Pass the ultrafast flag
                    topk_strategy=topk_strategy,
                    use_shunting=use_shunting,
                    reactivate=reactivate,
                    reactivation_init_m=reactivation_init_m,
                    reactivation_init_b=reactivation_init_b,
                    reactivation_type=reactivation_type,
                    reactivation_strategy=reactivation_strategy,
                    blocklinear_strategy=blocklinear_strategy,
                    layer_idx=0,
                    print_hooks=print_hooks,
                    use_noise=use_noise,
                    rewiring_mode=rewiring_mode,
                    synapses_per_branch=synapses_per_branch,
                    excitatory_target_density=excitatory_target_density,
                    inhibitory_target_density=inhibitory_target_density,
                    freeze_excitatory_connectivity=freeze_excitatory_connectivity,
                    freeze_inhibitory_connectivity=freeze_inhibitory_connectivity,
                    init_method=init_method,
                    weight_threshold=weight_threshold,
                    **kwargs
                )
            )

        self.branch_layers = nn.ModuleList(branch_layers)

        self.reactivate = reactivate
        self.layer_sizes = layer_sizes
        self.input_inhibitory = inhibitory_input_dim is not None
        self.n_soma = n_soma
        self.somatic_synapses = somatic_synapses
    
    def apply_rewiring(self):
        for layer in self.branch_layers:
            layer.apply_rewiring()

    def decay_weights(self, weight_decay):
        for branch_layer in self.branch_layers:
            branch_layer.decay_weights(weight_decay)

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
            layer = self.branch_layers[i]

            if layer.input_inhibitory and layer.input_excitatory:
                exc, inh = layer.get_pruned_weights() if pruned else layer.get_weights()
                exc_chunk = exc.chunk(self.n_soma, dim=0)
                inh_chunk = inh.chunk(self.n_soma, dim=0)

                exc_total = exc_total + torch.stack(
                    [chunk.sum(0) for chunk in exc_chunk], dim=0
                )
                inh_total = inh_total + torch.stack(
                    [chunk.sum(0) for chunk in inh_chunk], dim=0
                )

            elif layer.input_excitatory:
                exc = layer.get_pruned_weights() if pruned else layer.get_weights()
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


class DendriNetWithOutputs(DendriNet):
    """
    A version of DendriNet that also returns branch activations from forward pass.
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
        topk_temperature=0.5,  # For stochastic TopK
        topk_ultrafast=False,  # For much faster stochastic TopK
        topk_strategy="none",
        use_shunting=True,
        reactivation_type="param_tanh",
        reactivation_strategy="none",
        blocklinear_strategy="none",
        print_hooks=False,
        reactivation_init_m = 1,
        reactivation_init_b = 0.5,
        # DEEPST specific parameters, clean up later
        excitatory_target_density=0.1,
        inhibitory_target_density=0.1,
        use_noise=True,
        sigma=0.05,
        rewiring_mode='global',
        synapses_per_branch=None,
        freeze_excitatory_connectivity=False,
        freeze_inhibitory_connectivity=False,
        init_method=None,
        weight_threshold=1e-6,
        **kwargs
    ):
        super().__init__(
            n_soma,
            branch_factors,
            excitatory_input_dim,
            excitatory_synapses_per_branch,
            inhibitory_input_dim=inhibitory_input_dim,
            inhibitory_synapses_per_branch=inhibitory_synapses_per_branch,
            reactivate=reactivate,
            somatic_synapses=somatic_synapses,
            topk_init_method=topk_init_method,
            topk_noise_level=topk_noise_level,
            topk_type=topk_type,  # New parameter for TopK strategy
            topk_temperature=topk_temperature,  # For stochastic TopK
            topk_ultrafast=topk_ultrafast,  # For much faster stochastic TopK
            topk_strategy=topk_strategy,
            use_shunting=use_shunting,
            reactivation_type=reactivation_type,
            reactivation_strategy=reactivation_strategy,
            blocklinear_strategy=blocklinear_strategy,
            print_hooks=print_hooks,
            reactivation_init_m = reactivation_init_m,
            reactivation_init_b = reactivation_init_b,
            # DEEPST specific parameters, clean up later
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
            **kwargs
        )

    def forward_with_branch_outputs(self, x, inhibitory_input=None):
        """
        Forward pass that returns branch activations for each layer.
        Important: For visualization and information analysis.
        Properly handles somatic_synapses=False configuration.

        Returns:
            final_soma: Tensor of final output
            per_layer_acts: List of processed activations per layer
                Each element is a list with format [excitatory, inhibitory, combined]
                Elements may be None if not available for that layer
        """
        per_layer_acts = []
        branch_input = None

        for i in range(self.n_branch_layers):
            current_layer = self.branch_layers[i]

            # --- Compute excitatory & inhibitory partial transforms ---
            excitatory_input = None
            inhibitory_input_processed = None

            if (
                hasattr(current_layer, "branch_excitation")
                and current_layer.branch_excitation is not None
            ):
                try:
                    excitatory_input = current_layer.branch_excitation(x)
                except Exception:
                    pass

            if (
                inhibitory_input is not None
                and hasattr(current_layer, "branch_inhibition")
                and current_layer.branch_inhibition is not None
            ):
                try:
                    inhibitory_input_processed = current_layer.branch_inhibition(
                        inhibitory_input
                    )
                except Exception:
                    pass

            if hasattr(current_layer, "excitatory_input"):
                current_layer.excitatory_input = excitatory_input  # was 'x' before

            if hasattr(current_layer, "inhibitory_input"):
                current_layer.inhibitory_input = (
                    inhibitory_input_processed  # was 'inhibitory_input' before
                )

            output = current_layer(x, inhibitory_input, branch_input)

            layer_info = [excitatory_input, inhibitory_input_processed, output]
            per_layer_acts.append(layer_info)

            branch_input = output

        # -----------------------------
        # Final soma layer processing
        final_layer = self.branch_layers[-1]

        # Compute final excit/inhib partial transforms for the final layer
        final_excitatory = None
        final_inhibitory = None

        if (
            hasattr(final_layer, "branch_excitation")
            and final_layer.branch_excitation is not None
        ):
            try:
                final_excitatory = final_layer.branch_excitation(x)
            except Exception:
                pass

        if (
            inhibitory_input is not None
            and hasattr(final_layer, "branch_inhibition")
            and final_layer.branch_inhibition is not None
        ):
            try:
                final_inhibitory = final_layer.branch_inhibition(inhibitory_input)
            except Exception:
                pass

        if hasattr(final_layer, "excitatory_input"):
            final_layer.excitatory_input = final_excitatory

        if hasattr(final_layer, "inhibitory_input"):
            final_layer.inhibitory_input = final_inhibitory

        # Now get final soma output:
        final_soma = final_layer(x, inhibitory_input, branch_input)

        # Add the final layer's info to the per_layer_acts
        per_layer_acts.append([final_excitatory, final_inhibitory, final_soma])

        return final_soma, per_layer_acts

    def forward(self, x, inhibitory_input=None):
        """
        The simpler forward pass, typically used during training.
        We still assign excitatory_input / inhibitory_input to the partial transforms
        so that any visualization or ablation code sees them.
        """
        output = None
        for i in range(self.n_branch_layers):
            layer_i = self.branch_layers[i]

            excitatory_in = None
            inhibitory_in = None

            if (
                hasattr(layer_i, "branch_excitation")
                and layer_i.branch_excitation is not None
            ):
                try:
                    excitatory_in = layer_i.branch_excitation(x)
                except Exception:
                    pass

            if (
                inhibitory_input is not None
                and hasattr(layer_i, "branch_inhibition")
                and layer_i.branch_inhibition is not None
            ):
                try:
                    inhibitory_in = layer_i.branch_inhibition(inhibitory_input)
                except Exception:
                    pass

            if hasattr(layer_i, "excitatory_input"):
                layer_i.excitatory_input = excitatory_in  # was 'x'

            if hasattr(layer_i, "inhibitory_input"):
                layer_i.inhibitory_input = inhibitory_in  # was 'inhibitory_input'

            # Actually run the layer
            output = layer_i(x, inhibitory_input, output)

        # For the final layer:
        final_layer = self.branch_layers[-1]

        final_excit = None
        final_inhib = None

        if (
            hasattr(final_layer, "branch_excitation")
            and final_layer.branch_excitation is not None
        ):
            try:
                final_excit = final_layer.branch_excitation(x)
            except Exception:
                pass

        if (
            inhibitory_input is not None
            and hasattr(final_layer, "branch_inhibition")
            and final_layer.branch_inhibition is not None
        ):
            try:
                final_inhib = final_layer.branch_inhibition(inhibitory_input)
            except Exception:
                pass

        if hasattr(final_layer, "excitatory_input"):
            final_layer.excitatory_input = final_excit

        if hasattr(final_layer, "inhibitory_input"):
            final_layer.inhibitory_input = final_inhib

        final_soma = final_layer(x, inhibitory_input, output)
        return final_soma


__all__ = ["DendriNet", "DendriNetWithOutputs"] 