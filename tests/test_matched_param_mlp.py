"""Tests for SparseStructuredLayer (from matched_param.py)."""

import torch
import torch.nn as nn

from dendritic_modeling.networks.architectures.classical.mlp.matched_param import (
    SparseStructuredLayer,
)

# ---------------------------------------------------------------------------
# SparseStructuredLayer
# ---------------------------------------------------------------------------


class TestSparseStructuredLayer:
    def test_forward_synapse_only(self):
        layer = SparseStructuredLayer(
            output_dim=10,
            synaptic_input_dim=20,
            synaptic_K=5,
        )
        x = torch.randn(4, 20)
        out = layer(x)
        assert out.shape == (4, 10)

    def test_forward_branches_only(self):
        layer = SparseStructuredLayer(
            output_dim=8,
            input_branch_factor=3,
        )
        branch_input = torch.randn(4, 24)  # 8 * 3
        x = torch.randn(4, 10)  # not used since no synapse
        out = layer(x, branch_input=branch_input)
        assert out.shape == (4, 8)

    def test_forward_synapse_and_branches(self):
        layer = SparseStructuredLayer(
            output_dim=8,
            synaptic_input_dim=20,
            synaptic_K=5,
            input_branch_factor=3,
        )
        x = torch.randn(4, 20)
        branch = torch.randn(4, 24)
        out = layer(x, branch_input=branch)
        assert out.shape == (4, 8)

    def test_with_reactivation(self):
        layer = SparseStructuredLayer(
            output_dim=6,
            synaptic_input_dim=12,
            synaptic_K=4,
            reactivate=True,
        )
        x = torch.randn(4, 12)
        out = layer(x)
        assert out.shape == (4, 6)
        # ParametricTanh output in [0, 1]
        assert (out >= 0).all() and (out <= 1).all()

    def test_without_reactivation(self):
        layer = SparseStructuredLayer(
            output_dim=6,
            synaptic_input_dim=12,
            synaptic_K=4,
            reactivate=False,
        )
        assert isinstance(layer.reactivation, nn.Identity)

    def test_gradient_flow(self):
        layer = SparseStructuredLayer(
            output_dim=8,
            synaptic_input_dim=16,
            synaptic_K=4,
            reactivate=True,
        )
        x = torch.randn(2, 16)
        out = layer(x)
        out.sum().backward()
        assert layer.synapse_layer.pre_w.grad is not None

    def test_decay_weights(self):
        layer = SparseStructuredLayer(
            output_dim=8,
            synaptic_input_dim=16,
            synaptic_K=4,
        )
        w_before = layer.synapse_layer.pre_w.data.clone()
        layer.decay_weights(weight_decay=0.1)
        # Weights should have changed
        assert not torch.equal(w_before, layer.synapse_layer.pre_w.data)

    def test_decay_weights_no_synapse(self):
        layer = SparseStructuredLayer(
            output_dim=8,
            input_branch_factor=2,
        )
        # Should not raise even without synapse layer
        layer.decay_weights(weight_decay=0.1)

    def test_no_synapse_no_branches(self):
        layer = SparseStructuredLayer(output_dim=5)
        x = torch.randn(2, 10)
        out = layer(x)
        # No synapse and no branch → activation is 0 (scalar), Identity(0) = 0
        assert out == 0
