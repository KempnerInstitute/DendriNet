"""Signal aggregation helpers for information analysis."""

from __future__ import annotations

import os

import numpy as np
import torch

from dendritic_modeling.utils import save_dict
from dendritic_modeling.utils.lda import (
    apply_nonnegative_lda,
    apply_random_weights,
    fit_nonnegative_lda,
)


class InformationSignalAggregationMixin:
    """Builds LDA, random-weight, per-neuron, and E/I signal summaries."""

    def _fit_nonnegative_lda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight_norm_order: int | None = None,
        gamma: float = 1.0,
    ) -> tuple[np.ndarray, float]:
        """Fit LDA with non-negative weight constraint.

        Delegates to the utility function in dendritic_modeling.utils.lda.
        """
        return fit_nonnegative_lda(X, y, weight_norm_order, gamma)

    def _apply_nonnegative_lda(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight_norm_order: int | None = None,
        gamma: float = 1.0,
    ) -> np.ndarray:
        """Apply non-negative LDA to get projected values.

        Delegates to the utility function in dendritic_modeling.utils.lda.
        """
        return apply_nonnegative_lda(X, y, weight_norm_order, gamma)

    def compute_lda_aggregated_signals(
        self,
        layer_data: dict,
        C: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute LDA-weighted versions of E, I, Vb, and Vout using TopK-masked features.

        For each branch, fit non-negative constrained LDA only on the input
        dimensions that pass through the TopK mask (i.e., the synapses that are
        actually connected). Non-negative weights match the network's positive
        synaptic weights.

        Vout_lin is computed by applying the same dendritic computation
        (shunting or additive) to E_lin, I_lin, Vb_lin, then passing through
        the network's reactivation function.

        Args:
            layer_data: Dictionary containing raw inputs and module
            C: Class labels, shape [n_samples,]

        Returns:
            Tuple of (E_lin, I_lin, Vb_lin, Vout_lin) each shape [n_samples, n_branches]
        """
        module = layer_data.get("module")
        if module is None:
            return None, None, None, None

        raw_exc = layer_data.get("raw_excitatory_input")
        raw_inh = layer_data.get("raw_inhibitory_input")
        raw_branch = layer_data.get("raw_branch_input")

        n_samples = C.shape[0]
        n_branches = layer_data["excitation"].shape[1]

        E_lin = np.zeros((n_samples, n_branches))
        I_lin = np.zeros((n_samples, n_branches))
        Vb_lin = np.zeros((n_samples, n_branches))
        Vout_lin = np.zeros((n_samples, n_branches))

        # Check if we have enough classes for LDA
        if len(np.unique(C)) <= 1:
            # Fall back to network weights
            E_lin = layer_data["excitation"].detach().cpu().numpy()
            I_lin = layer_data["inhibition"].detach().cpu().numpy()
            vb_fallback = layer_data.get("branch_input")
            if vb_fallback is not None:
                Vb_lin = vb_fallback.detach().cpu().numpy()
            else:
                Vb_lin = np.zeros_like(E_lin)
            Vout_lin = layer_data["output"].detach().cpu().numpy()
            return E_lin, I_lin, Vb_lin, Vout_lin

        # For E: Fit LDA on TopK-masked excitatory inputs for each branch
        if raw_exc is not None and module.branch_excitation is not None:
            raw_exc_np = raw_exc.detach().cpu().numpy()  # [n_samples, input_dim]

            # Get weight normalization parameters from the module (matching TopKLinear)
            exc_weight_norm_order = getattr(
                module.branch_excitation, "weight_norm_order", None
            )
            exc_gamma = getattr(module.branch_excitation, "gamma", 1.0)

            # Get the TopK mask for excitatory branches [n_branches, input_dim]
            exc_mask = module.branch_excitation.weight_mask().detach().cpu().numpy()

            # Fit separate LDA for each branch using its masked inputs
            for branch_idx in range(n_branches):
                branch_mask = exc_mask[branch_idx, :]  # [input_dim]
                active_indices = np.where(branch_mask > 0)[0]

                if len(active_indices) > 0:
                    # Extract only the active synaptic inputs for this branch
                    branch_inputs = raw_exc_np[:, active_indices]  # [n_samples, K]

                    try:
                        # Fit non-negative LDA on the K active synapses
                        # Use same weight normalization as the network's TopKLinear
                        E_lin[:, branch_idx] = self._apply_nonnegative_lda(
                            branch_inputs,
                            C,
                            weight_norm_order=exc_weight_norm_order,
                            gamma=exc_gamma,
                        )

                    except Exception as e:
                        self.logger.warning(
                            f"LDA failed for E branch {branch_idx}: {e}. Using network weights."
                        )
                        E_lin[:, branch_idx] = (
                            layer_data["excitation"][:, branch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                else:
                    E_lin[:, branch_idx] = (
                        layer_data["excitation"][:, branch_idx].detach().cpu().numpy()
                    )

        # For I: Fit LDA on TopK-masked inhibitory inputs for each branch
        if raw_inh is not None and module.branch_inhibition is not None:
            raw_inh_np = raw_inh.detach().cpu().numpy()  # [n_samples, input_dim]

            # Get weight normalization parameters from the module (matching TopKLinear)
            inh_weight_norm_order = getattr(
                module.branch_inhibition, "weight_norm_order", None
            )
            inh_gamma = getattr(module.branch_inhibition, "gamma", 1.0)

            # Get the TopK mask for inhibitory branches [n_branches, input_dim]
            inh_mask = module.branch_inhibition.weight_mask().detach().cpu().numpy()

            # Fit separate LDA for each branch using its masked inputs
            for branch_idx in range(n_branches):
                branch_mask = inh_mask[branch_idx, :]  # [input_dim]
                active_indices = np.where(branch_mask > 0)[0]

                if len(active_indices) > 0:
                    # Extract only the active synaptic inputs for this branch
                    branch_inputs = raw_inh_np[:, active_indices]  # [n_samples, K]

                    try:
                        # Fit non-negative LDA on the K active synapses
                        # Use same weight normalization as the network's TopKLinear
                        I_lin[:, branch_idx] = self._apply_nonnegative_lda(
                            branch_inputs,
                            C,
                            weight_norm_order=inh_weight_norm_order,
                            gamma=inh_gamma,
                        )

                    except Exception as e:
                        self.logger.warning(
                            f"LDA failed for I branch {branch_idx}: {e}. Using network weights."
                        )
                        I_lin[:, branch_idx] = (
                            layer_data["inhibition"][:, branch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
                else:
                    I_lin[:, branch_idx] = (
                        layer_data["inhibition"][:, branch_idx].detach().cpu().numpy()
                    )

        # For Vb: Fit LDA on branch inputs using BlockLinear's block structure
        # branches_to_output is a BlockLinear where each output branch receives input
        # from a contiguous block of upstream branches
        if raw_branch is not None and module.input_branches:
            raw_branch_np = (
                raw_branch.detach().cpu().numpy()
            )  # [n_samples, n_upstream_branches]

            # BlockLinear doesn't have weight_norm_order/gamma like TopKLinear
            # Use defaults (L1 norm, gamma=1.0)
            vb_weight_norm_order = getattr(
                module.branches_to_output, "weight_norm_order", None
            )
            vb_gamma = getattr(module.branches_to_output, "gamma", 1.0)

            # Get block_size from branches_to_output (BlockLinear)
            if hasattr(module.branches_to_output, "block_size"):
                block_size = module.branches_to_output.block_size

                # Fit separate LDA for each output branch using its upstream block
                for branch_idx in range(n_branches):
                    # Each output branch receives from upstream indices [branch_idx * block_size : (branch_idx+1) * block_size]
                    start_idx = branch_idx * block_size
                    end_idx = min((branch_idx + 1) * block_size, raw_branch_np.shape[1])

                    if start_idx < raw_branch_np.shape[1]:
                        # Extract the upstream branches for this output branch
                        branch_inputs = raw_branch_np[:, start_idx:end_idx]

                        if branch_inputs.shape[1] > 0:
                            try:
                                # Fit non-negative LDA on the upstream branches
                                # Use same weight normalization as the network
                                Vb_lin[:, branch_idx] = self._apply_nonnegative_lda(
                                    branch_inputs,
                                    C,
                                    weight_norm_order=vb_weight_norm_order,
                                    gamma=vb_gamma,
                                )

                            except Exception as e:
                                self.logger.warning(
                                    f"LDA failed for Vb branch {branch_idx}: {e}. Using network weights."
                                )
                                Vb_lin[:, branch_idx] = (
                                    layer_data["branch_input"][:, branch_idx]
                                    .detach()
                                    .cpu()
                                    .numpy()
                                )
                        else:
                            Vb_lin[:, branch_idx] = (
                                layer_data["branch_input"][:, branch_idx]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                    else:
                        Vb_lin[:, branch_idx] = (
                            layer_data["branch_input"][:, branch_idx]
                            .detach()
                            .cpu()
                            .numpy()
                        )
            else:
                # Fallback: no block structure, use network weights
                Vb_lin = layer_data["branch_input"].detach().cpu().numpy()

        # Compute Vout_lin using the same dendritic computation as the network
        # With non-negative LDA weights, E_lin, I_lin, Vb_lin are all non-negative
        # so shunting inhibition formula works properly
        use_shunting = getattr(module, "use_shunting", True)
        epsilon = getattr(module, "epsilon", 1e-8)

        if use_shunting:
            # Shunting inhibition: voltage = (E + Vb) / (1 + E + I + Vb)
            # With non-negative inputs, this normalizes properly
            numerator = E_lin + Vb_lin
            denominator = 1.0 + E_lin + I_lin + Vb_lin
            voltage_lin = numerator / (denominator + epsilon)
        else:
            # Additive: voltage = E + Vb - I
            voltage_lin = E_lin + Vb_lin - I_lin

        # Apply reactivation function if available
        if hasattr(module, "reactivation") and module.reactivation is not None:
            import torch

            with torch.no_grad():
                voltage_tensor = torch.tensor(voltage_lin, dtype=torch.float32)
                if next(module.reactivation.parameters(), None) is not None:
                    device = next(module.reactivation.parameters()).device
                    voltage_tensor = voltage_tensor.to(device)
                Vout_lin = module.reactivation(voltage_tensor).cpu().numpy()
        else:
            # No reactivation, use voltage directly
            Vout_lin = voltage_lin

        return E_lin, I_lin, Vb_lin, Vout_lin

    def compute_shuffled_aggregated_signals(
        self,
        layer_data: dict,
        C: np.ndarray,
        shuffle_idx: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute random-weighted versions of E, I, Vb for baseline comparison.

        Uses random non-negative weights with same normalization as LDA weights.
        This provides a random baseline to compare against the LDA-derived signals,
        showing how much improvement LDA provides over chance.

        Args:
            layer_data: Dictionary containing raw inputs and module
            C: Class labels, shape [n_samples,] (unused but kept for API consistency)
            shuffle_idx: Random seed offset for reproducibility across shuffles

        Returns:
            Tuple of (E_sh, I_sh, Vb_sh, Vout_sh) each shape [n_samples, n_branches]
        """
        module = layer_data.get("module")
        if module is None:
            return None, None, None, None

        raw_exc = layer_data.get("raw_excitatory_input")
        raw_inh = layer_data.get("raw_inhibitory_input")
        raw_branch = layer_data.get("raw_branch_input")

        n_samples = C.shape[0]
        n_branches = layer_data["excitation"].shape[1]

        E_sh = np.zeros((n_samples, n_branches))
        I_sh = np.zeros((n_samples, n_branches))
        Vb_sh = np.zeros((n_samples, n_branches))
        Vout_sh = np.zeros((n_samples, n_branches))

        # Create RNG with reproducible seed
        rng = np.random.default_rng(int(self.random_weights_seed) + int(shuffle_idx))

        # For E: Apply random weights on TopK-masked excitatory inputs
        if raw_exc is not None and module.branch_excitation is not None:
            raw_exc_np = raw_exc.detach().cpu().numpy()
            exc_weight_norm_order = getattr(
                module.branch_excitation, "weight_norm_order", None
            )
            exc_gamma = getattr(module.branch_excitation, "gamma", 1.0)
            exc_mask = module.branch_excitation.weight_mask().detach().cpu().numpy()

            for branch_idx in range(n_branches):
                branch_mask = exc_mask[branch_idx, :]
                active_indices = np.where(branch_mask > 0)[0]
                if len(active_indices) > 0:
                    branch_inputs = raw_exc_np[:, active_indices]
                    E_sh[:, branch_idx] = apply_random_weights(
                        branch_inputs,
                        weight_norm_order=exc_weight_norm_order,
                        gamma=exc_gamma,
                        rng=rng,
                    )

        # For I: Apply random weights on TopK-masked inhibitory inputs
        if raw_inh is not None and module.branch_inhibition is not None:
            raw_inh_np = raw_inh.detach().cpu().numpy()
            inh_weight_norm_order = getattr(
                module.branch_inhibition, "weight_norm_order", None
            )
            inh_gamma = getattr(module.branch_inhibition, "gamma", 1.0)
            inh_mask = module.branch_inhibition.weight_mask().detach().cpu().numpy()

            for branch_idx in range(n_branches):
                branch_mask = inh_mask[branch_idx, :]
                active_indices = np.where(branch_mask > 0)[0]
                if len(active_indices) > 0:
                    branch_inputs = raw_inh_np[:, active_indices]
                    I_sh[:, branch_idx] = apply_random_weights(
                        branch_inputs,
                        weight_norm_order=inh_weight_norm_order,
                        gamma=inh_gamma,
                        rng=rng,
                    )

        # For Vb: Apply random weights on branch inputs using BlockLinear structure
        if raw_branch is not None and module.input_branches:
            raw_branch_np = raw_branch.detach().cpu().numpy()
            vb_weight_norm_order = getattr(
                module.branches_to_output, "weight_norm_order", None
            )
            vb_gamma = getattr(module.branches_to_output, "gamma", 1.0)

            if hasattr(module.branches_to_output, "block_size"):
                block_size = module.branches_to_output.block_size
                for branch_idx in range(n_branches):
                    start_idx = branch_idx * block_size
                    end_idx = min((branch_idx + 1) * block_size, raw_branch_np.shape[1])
                    if start_idx < raw_branch_np.shape[1]:
                        branch_inputs = raw_branch_np[:, start_idx:end_idx]
                        if branch_inputs.shape[1] > 0:
                            Vb_sh[:, branch_idx] = apply_random_weights(
                                branch_inputs,
                                weight_norm_order=vb_weight_norm_order,
                                gamma=vb_gamma,
                                rng=rng,
                            )

        # Compute Vout_sh using the same dendritic computation as the network
        use_shunting = getattr(module, "use_shunting", True)
        epsilon = getattr(module, "epsilon", 1e-8)

        if use_shunting:
            numerator = E_sh + Vb_sh
            denominator = 1.0 + E_sh + I_sh + Vb_sh
            voltage_sh = numerator / (denominator + epsilon)
        else:
            voltage_sh = E_sh + Vb_sh - I_sh

        # Apply reactivation function if available
        if hasattr(module, "reactivation") and module.reactivation is not None:
            import torch

            with torch.no_grad():
                voltage_tensor = torch.tensor(voltage_sh, dtype=torch.float32)
                if next(module.reactivation.parameters(), None) is not None:
                    device = next(module.reactivation.parameters()).device
                    voltage_tensor = voltage_tensor.to(device)
                Vout_sh = module.reactivation(voltage_tensor).cpu().numpy()
        else:
            Vout_sh = voltage_sh

        return E_sh, I_sh, Vb_sh, Vout_sh

    def compute_lda_aggregated_signals_chained(
        self,
        layer_data: dict,
        C: np.ndarray,
        prev_layer_vout_lin: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute LDA-weighted signals with optional chained branch input.

        This is similar to compute_lda_aggregated_signals but allows overriding
        the branch input with Vout_lin from a previous layer, enabling proper
        chained LDA comparison across all network layers.

        Uses non-negative LDA weights to match network's non-negative synaptic weights.

        Args:
            layer_data: Dictionary containing raw inputs and module
            C: Class labels, shape [n_samples,]
            prev_layer_vout_lin: Vout_lin from previous layer, to use as branch input.
                                 If None, uses the raw_branch_input from layer_data.

        Returns:
            Tuple of (E_lin, I_lin, Vb_lin, Vout_lin) each shape [n_samples, n_branches]
        """
        # The "chained" path is identical to the standard LDA aggregation, except
        # that the upstream branch input can be replaced with the previous layer's
        # LDA-computed output (Vout_lin). To avoid duplicated logic (and to ensure
        # we still respect TopK masks / BlockLinear block structure), we delegate
        # to `compute_lda_aggregated_signals` after injecting the overridden
        # `raw_branch_input` when provided.

        if prev_layer_vout_lin is None:
            return self.compute_lda_aggregated_signals(layer_data, C)

        try:
            # Convert override to a torch Tensor so downstream code can `.detach()`.
            if isinstance(prev_layer_vout_lin, torch.Tensor):
                raw_branch_override = prev_layer_vout_lin
            else:
                raw_branch_override = torch.as_tensor(
                    prev_layer_vout_lin, dtype=torch.float32
                )

            # Move override to the same device as existing layer tensors if possible.
            device = None
            for key in (
                "raw_excitatory_input",
                "raw_inhibitory_input",
                "excitation",
                "inhibition",
                "output",
            ):
                t = layer_data.get(key)
                if isinstance(t, torch.Tensor):
                    device = t.device
                    break
            if device is not None:
                raw_branch_override = raw_branch_override.to(device)

            tmp = dict(layer_data)
            tmp["raw_branch_input"] = raw_branch_override
            return self.compute_lda_aggregated_signals(tmp, C)
        except Exception as e:
            self.logger.warning(
                "Chained LDA branch-input override failed; falling back to raw branch input. Error: %s",
                e,
            )
            return self.compute_lda_aggregated_signals(layer_data, C)

    def ensure_inhibitory_data(
        self, excitation_data: torch.Tensor, inhibition_data: torch.Tensor
    ):
        """Ensure inhibitory data is available, replacing with zeros if missing.

        Args:
            excitation_data: Shape [n_samples, n_branches] - excitatory activations
            inhibition_data: Shape [n_samples, n_branches] - inhibitory activations (can be None)

        Returns:
            Tuple of (excitation_data, inhibition_data) with guaranteed inhibitory data
        """
        # Handle case where inhibitory data might be missing/None
        if inhibition_data is None:
            inhibition_data = torch.zeros_like(excitation_data)
            self.logger.warning(
                "Inhibitory data is missing - using zeros for inhibitory activations"
            )

        return excitation_data, inhibition_data

    def has_inhibitory_data(self, layer_data_dict: dict) -> bool:
        """Check if the network has valid inhibitory data.

        Args:
            layer_data_dict: Dictionary containing layer data from all layers

        Returns:
            True if inhibitory data exists and is not all zeros
        """
        for layer_data in layer_data_dict.values():
            if "inhibition" in layer_data and layer_data["inhibition"] is not None:
                # Check if inhibitory data is not all zeros
                if torch.any(layer_data["inhibition"] != 0):
                    return True
        return False

    def compute_per_neuron_mi(
        self,
        E_data: np.ndarray,
        I_data: np.ndarray,
        Vout_data: np.ndarray,
        C: np.ndarray,
    ):
        """Compute MI for each neuron separately.

        Args:
            E_data: Shape [n_samples, n_neurons] - excitatory activations
            I_data: Shape [n_samples, n_neurons] - inhibitory activations
            Vout_data: Shape [n_samples, n_neurons] - output activations
            C: Shape [n_samples,] - class labels

        Returns:
            Dictionary with per-neuron MI results
        """
        _, n_neurons = E_data.shape
        per_neuron_results = []

        self.logger.info(f"Computing MI for {n_neurons} neurons individually...")

        for neuron_idx in range(n_neurons):
            # Extract data for single neuron
            E_neuron = E_data[:, neuron_idx : neuron_idx + 1]  # Keep 2D shape
            I_neuron = I_data[:, neuron_idx : neuron_idx + 1]
            Vout_neuron = Vout_data[:, neuron_idx : neuron_idx + 1]

            # Compute MI for this neuron
            neuron_results = self.compute_metrics(
                E_neuron, I_neuron, Vout_neuron, C, None, None
            )
            neuron_results["neuron_index"] = neuron_idx
            per_neuron_results.append(neuron_results)

        # Also compute average across neurons
        averaged_results = self._average_mi_results(per_neuron_results)

        return {
            "per_neuron_results": per_neuron_results,
            "averaged_results": averaged_results,
            "num_neurons": n_neurons,
        }

    def compute_ei_specific_mi(
        self,
        E_data: np.ndarray,
        I_data: np.ndarray,
        Vout_data: np.ndarray,
        C: np.ndarray,
    ):
        """Compute MI separately for excitatory and inhibitory networks.

        Args:
            E_data: Shape [n_samples, n_neurons] - excitatory activations
            I_data: Shape [n_samples, n_neurons] - inhibitory activations
            Vout_data: Shape [n_samples, n_neurons] - output activations
            C: Shape [n_samples,] - class labels

        Returns:
            Dictionary with separate E and I network MI results
        """
        self.logger.info(
            "Computing MI separately for excitatory and inhibitory networks..."
        )

        results = {}

        # Always compute excitatory network MI
        e_results = self.compute_metrics(
            E_data, np.zeros_like(I_data), Vout_data, C, None, None
        )
        e_results["network_type"] = "excitatory"
        results["excitatory_results"] = e_results

        # Check if inhibitory data is meaningful (not all zeros)
        has_inhibitory = np.any(I_data != 0)

        if has_inhibitory:
            # Compute MI for inhibitory network only
            i_results = self.compute_metrics(
                np.zeros_like(E_data), I_data, Vout_data, C, None, None
            )
            i_results["network_type"] = "inhibitory"
            results["inhibitory_results"] = i_results

            # Compute MI for combined E+I network (current default)
            combined_results = self.compute_metrics(
                E_data, I_data, Vout_data, C, None, None
            )
            combined_results["network_type"] = "combined"
            results["combined_results"] = combined_results

            self.logger.info(
                "Computed MI for excitatory, inhibitory, and combined networks"
            )
        else:
            # No inhibitory data available
            self.logger.warning(
                "No inhibitory data detected - skipping inhibitory and combined network analysis"
            )
            results["inhibitory_results"] = None
            results["combined_results"] = (
                e_results  # Use excitatory results as combined
            )
            results["inhibitory_data_available"] = False

        results["inhibitory_data_available"] = has_inhibitory
        return results

    def compute_per_layer_enhanced_analysis(self, C: np.ndarray):
        """Compute enhanced analysis (per-neuron and EI-specific) for each layer separately.

        Args:
            C: Class labels for all samples

        Returns:
            Dictionary with per-layer enhanced analysis results
        """
        self.logger.info("Computing per-layer enhanced analysis...")

        per_layer_results = {}

        for layer_name, layer_data in self.data_dict.items():
            if "excitation" not in layer_data:
                continue

            self.logger.info(f"Processing enhanced analysis for layer: {layer_name}")

            # Ensure inhibitory data is available for this layer
            E_filtered, I_filtered = self.ensure_inhibitory_data(
                layer_data["excitation"], layer_data["inhibition"]
            )

            E_layer = E_filtered.detach().cpu().numpy()
            I_layer = I_filtered.detach().cpu().numpy()
            Vout_layer = layer_data["output"].detach().cpu().numpy()

            layer_results = {}

            # Add per-neuron analysis for this layer
            if self.per_neuron_analysis:
                per_neuron_results = self.compute_per_neuron_mi(
                    E_layer, I_layer, Vout_layer, C
                )
                layer_results["per_neuron_analysis"] = per_neuron_results

            # Add EI-network analysis for this layer
            if self.per_einet_analysis:
                ei_specific_results = self.compute_ei_specific_mi(
                    E_layer, I_layer, Vout_layer, C
                )
                layer_results["per_einet_analysis"] = ei_specific_results

            # Add layer metadata
            layer_results["layer_name"] = layer_name
            layer_results["layer_shape"] = {
                "excitation": E_layer.shape,
                "inhibition": I_layer.shape,
                "output": Vout_layer.shape,
            }
            layer_results["has_inhibitory_data"] = np.any(I_layer != 0)

            per_layer_results[layer_name] = layer_results

        return per_layer_results

    def _save_per_layer_ei_summaries(
        self, per_layer_data: dict, save_path: str, base_filename: str
    ):
        """Generate and save individual summaries for each layer's E and I networks.

        Args:
            per_layer_data: Dictionary containing per-layer analysis results
            save_path: Directory to save summary files
            base_filename: Base filename for the summaries
        """
        self.logger.info("Generating individual layer E/I network summaries...")

        # Create subfolder for per-layer results
        per_layer_path = os.path.join(save_path, "per_layer_results")
        os.makedirs(per_layer_path, exist_ok=True)

        for layer_name, layer_results in per_layer_data.items():
            if "per_einet_analysis" in layer_results:
                ei_analysis = layer_results["per_einet_analysis"]

                # Clean layer name for filename
                clean_layer_name = layer_name.replace(".", "_").replace("/", "_")

                # Generate excitatory network summary
                if "excitatory_results" in ei_analysis:
                    exc_summary = self.get_summary(ei_analysis["excitatory_results"])
                    exc_summary_path = os.path.join(
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_excitatory_summary.txt",
                    )
                    with open(exc_summary_path, "w") as f:
                        f.write(f"Excitatory Network Summary - {layer_name}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(exc_summary)

                    # Also save the results as JSON for plotting
                    save_dict(
                        ei_analysis["excitatory_results"],
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_excitatory_results",
                    )

                # Generate inhibitory network summary
                if ei_analysis.get("inhibitory_results") is not None:
                    inh_summary = self.get_summary(ei_analysis["inhibitory_results"])
                    inh_summary_path = os.path.join(
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_inhibitory_summary.txt",
                    )
                    with open(inh_summary_path, "w") as f:
                        f.write(f"Inhibitory Network Summary - {layer_name}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(inh_summary)

                    # Also save the results as JSON for plotting
                    save_dict(
                        ei_analysis["inhibitory_results"],
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_inhibitory_results",
                    )

                # Generate combined network summary
                if ei_analysis.get("combined_results") is not None:
                    combined_summary = self.get_summary(ei_analysis["combined_results"])
                    combined_summary_path = os.path.join(
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_combined_summary.txt",
                    )
                    with open(combined_summary_path, "w") as f:
                        f.write(f"Combined E+I Network Summary - {layer_name}\n")
                        f.write("=" * 50 + "\n\n")
                        f.write(combined_summary)

                    # Also save the results as JSON for plotting
                    save_dict(
                        ei_analysis["combined_results"],
                        per_layer_path,
                        f"{base_filename}_{clean_layer_name}_combined_results",
                    )

        self.logger.info(
            f"Generated individual summaries for {len(per_layer_data)} layers"
        )


__all__ = ["InformationSignalAggregationMixin"]
