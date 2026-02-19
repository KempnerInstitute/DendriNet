"""
Tests for freeze_ei_connectivity: verify that when freeze_ei_connectivity=True,
the Exc (input) -> Inh (EI) synapse layer is frozen during rewiring, while
EE and IE layers continue to rewire.
"""

import unittest

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.ei_network import (
    ConfigurableEINetwork,
)


def _get_ei_ee_ie_masks(ei_network):
    """Extract EI, EE, IE masks from first layer."""
    layer = ei_network.layers[0]
    ei_masks, ee_masks, ie_masks = [], [], []
    if hasattr(layer, "inhibitory_cells") and layer.inhibitory_cells is not None:
        for bl in layer.inhibitory_cells.branch_layers:
            if bl.branch_excitation is not None and hasattr(bl.branch_excitation, "mask"):
                ei_masks.append(bl.branch_excitation.mask.clone())
    for bl in layer.excitatory_cells.branch_layers:
        if bl.branch_excitation is not None and hasattr(bl.branch_excitation, "mask"):
            ee_masks.append(bl.branch_excitation.mask.clone())
        if bl.branch_inhibition is not None and hasattr(bl.branch_inhibition, "mask"):
            ie_masks.append(bl.branch_inhibition.mask.clone())
    return ei_masks, ee_masks, ie_masks


def _force_rewiring_by_zeroing_weights(ei_network, zero_ee_ie_only=True):
    """Zero out some weights to force pruning and rewiring. Optionally skip EI."""
    layer = ei_network.layers[0]
    # Zero EE and IE weights (not EI) to force rewiring
    for bl in layer.excitatory_cells.branch_layers:
        if bl.branch_excitation is not None and hasattr(bl.branch_excitation, "pre_w"):
            mask = bl.branch_excitation.mask
            with torch.no_grad():
                # Zero ~half of active weights to trigger pruning
                active = torch.nonzero(mask, as_tuple=False)
                n_zero = max(1, len(active) // 2)
                idx = torch.randperm(len(active))[:n_zero]
                for i in idx:
                    r, c = active[i]
                    bl.branch_excitation.pre_w.data[r, c] = -100.0  # exp(-100) ~ 0
        if bl.branch_inhibition is not None and hasattr(bl.branch_inhibition, "pre_w"):
            mask = bl.branch_inhibition.mask
            with torch.no_grad():
                active = torch.nonzero(mask, as_tuple=False)
                n_zero = max(1, len(active) // 2)
                idx = torch.randperm(len(active))[:n_zero]
                for i in idx:
                    r, c = active[i]
                    bl.branch_inhibition.pre_w.data[r, c] = -100.0
    if not zero_ee_ie_only and hasattr(layer, "inhibitory_cells") and layer.inhibitory_cells is not None:
        for bl in layer.inhibitory_cells.branch_layers:
            if bl.branch_excitation is not None and hasattr(bl.branch_excitation, "pre_w"):
                mask = bl.branch_excitation.mask
                with torch.no_grad():
                    active = torch.nonzero(mask, as_tuple=False)
                    n_zero = max(1, len(active) // 2)
                    idx = torch.randperm(len(active))[:n_zero]
                    for i in idx:
                        r, c = active[i]
                        bl.branch_excitation.pre_w.data[r, c] = -100.0


class TestFreezeEIConnectivity(unittest.TestCase):
    """Test that freeze_ei_connectivity correctly freezes EI layer during rewiring."""

    def _make_network(self, freeze_ei_connectivity, topk_type="deepst"):
        """Create EINetwork with lateral inhibition and DeepST via config."""
        config = {
            "architecture": {
                "excitatory_layer_sizes": [4],
                "inhibitory_layer_sizes": [20],
                "excitatory_branch_factors": [2, 2],
                "inhibitory_branch_factors": [1],
                "inhibitory_network_type": "dendritic",
            },
            "connectivity": {
                "ee_synapses_per_branch_per_layer": [20],
                "ei_synapses_per_branch_per_layer": [20],
                "ie_synapses_per_branch_per_layer": [20],
                "ii_synapses_per_branch_per_layer": [0],
            },
            "transfer": {"input_mode": 0, "independent_pathways": True},
            "morphology": {
                "somatic_synapses": True,
                "use_shunting": True,
                "weight_transform": "softplus",
                "dbl_init_method": "analytical_expectation",
            },
            "sparsity": {
                "type": topk_type,
                "deepst": {
                    "freeze_excitatory_connectivity": False,
                    "freeze_inhibitory_connectivity": False,
                    "freeze_ei_connectivity": freeze_ei_connectivity,
                    "weight_threshold": 1e-6,
                },
            },
        }
        return ConfigurableEINetwork(config=config, input_dim=100)

    def test_ei_frozen_when_freeze_ei_connectivity_true(self):
        """When freeze_ei_connectivity=True, EI mask must not change after rewiring."""
        ei_network = self._make_network(freeze_ei_connectivity=True)
        ei_masks_before, ee_masks_before, ie_masks_before = _get_ei_ee_ie_masks(ei_network)

        # Force rewiring by zeroing EE and IE weights (not EI)
        _force_rewiring_by_zeroing_weights(ei_network, zero_ee_ie_only=True)

        # Trigger rewiring multiple times
        for _ in range(10):
            ei_network.apply_rewiring()

        ei_masks_after, ee_masks_after, ie_masks_after = _get_ei_ee_ie_masks(ei_network)

        # EI masks must be unchanged
        self.assertEqual(len(ei_masks_before), len(ei_masks_after))
        for before, after in zip(ei_masks_before, ei_masks_after):
            self.assertTrue(
                torch.equal(before, after),
                "EI mask changed when freeze_ei_connectivity=True",
            )

        # EE or IE masks should have changed (rewiring occurred)
        ee_changed = any(
            not torch.equal(b, a)
            for b, a in zip(ee_masks_before, ee_masks_after)
        )
        ie_changed = any(
            not torch.equal(b, a)
            for b, a in zip(ie_masks_before, ie_masks_after)
        )
        self.assertTrue(
            ee_changed or ie_changed,
            "Expected EE or IE masks to change after rewiring",
        )

    def test_ee_ie_not_frozen_when_freeze_ei_connectivity_true(self):
        """EE and IE connectivity must not be frozen when only freeze_ei_connectivity=True."""
        ei_network = self._make_network(freeze_ei_connectivity=True)
        _force_rewiring_by_zeroing_weights(ei_network, zero_ee_ie_only=True)
        ee_before, ie_before = [], []
        for bl in ei_network.layers[0].excitatory_cells.branch_layers:
            if bl.branch_excitation is not None:
                ee_before.append(bl.branch_excitation.mask.clone())
            if bl.branch_inhibition is not None:
                ie_before.append(bl.branch_inhibition.mask.clone())

        for _ in range(10):
            ei_network.apply_rewiring()

        ee_after, ie_after = [], []
        for bl in ei_network.layers[0].excitatory_cells.branch_layers:
            if bl.branch_excitation is not None:
                ee_after.append(bl.branch_excitation.mask.clone())
            if bl.branch_inhibition is not None:
                ie_after.append(bl.branch_inhibition.mask.clone())

        ee_changed = any(not torch.equal(b, a) for b, a in zip(ee_before, ee_after))
        ie_changed = any(not torch.equal(b, a) for b, a in zip(ie_before, ie_after))
        self.assertTrue(ee_changed or ie_changed, "EE and IE should rewire when only EI is frozen")

    def test_ei_can_change_when_freeze_ei_connectivity_false(self):
        """When freeze_ei_connectivity=False, EI mask can change after rewiring."""
        ei_network = self._make_network(freeze_ei_connectivity=False)
        _force_rewiring_by_zeroing_weights(ei_network, zero_ee_ie_only=False)

        ei_masks_before, _, _ = _get_ei_ee_ie_masks(ei_network)
        for _ in range(10):
            ei_network.apply_rewiring()
        ei_masks_after, _, _ = _get_ei_ee_ie_masks(ei_network)

        ei_changed = any(
            not torch.equal(b, a)
            for b, a in zip(ei_masks_before, ei_masks_after)
        )
        self.assertTrue(
            ei_changed,
            "EI mask should change when freeze_ei_connectivity=False",
        )
