"""Regression tests for two iss367_hs_PR fixes over develop.

1. Data-driven reactivation calibration must fire for *every* model built (and
   trained) in a single Python process, not just the first. On develop,
   ``calibrate_reactivation_from_data`` silently no-op'd (returned an empty
   diagnostics dict, leaving the gate at its init value) for the 2nd+ model in a
   process once a prior model had been built and trained — which breaks sweeps,
   notebooks, and any multi-model driver.

2. Multi-layer EI networks with ``transfer.input_mode == 1`` must build and run
   a forward pass. On develop this raised a shape-mismatch in the inter-layer
   inhibitory projection (``branch_inhibition`` matmul); only ``input_mode == 0``
   worked for >1 layer. PR supports both.
"""

from __future__ import annotations

import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize import (
    calibrate_reactivation_from_data,
)
from dendritic_modeling.networks.architectures.factory import get_architecture


def _small_additive_net(input_dim: int = 16):
    """A tiny normalized-additive EINet with a data-driven reactivation gate."""
    params = {
        "architecture": {
            "excitatory_layer_sizes": [8],
            "inhibitory_layer_sizes": [4],
            "excitatory_branch_factors": [2],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [4],
            "ei_synapses_per_branch_per_layer": [2],
            "ie_synapses_per_branch_per_layer": [2],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {"input_mode": 1},
        "reactivation": {"enabled": True, "init_policy": "occupancy_quantile"},
    }
    return get_architecture(
        "dendritic_normalized_additive", params, input_dim=input_dim
    )


def _forward(net, x):
    out = net(x)
    return out[0] if isinstance(out, (tuple, list)) else out


def test_calibration_fires_for_every_model_in_one_process():
    """Build + calibrate + train a model, then build + calibrate a second one
    in the SAME process. Both must calibrate (non-empty diagnostics)."""
    torch.manual_seed(0)
    x = torch.randn(48, 16)
    batches = [x[:16], x[16:32], x[32:]]

    calibrated_layer_counts = []
    for i in range(2):
        net = _small_additive_net()
        net.eval()
        with torch.no_grad():
            diag = calibrate_reactivation_from_data(
                net, batches, mode="occupancy_quantile", quantile_aggregation="global"
            )
        calibrated_layer_counts.append(len(diag) if diag else 0)

        # Train the FIRST model briefly: on develop this in-process training was
        # the trigger that made the *next* model's calibration silently no-op.
        if i == 0:
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            net.train()
            target = torch.randint(0, 8, (32,))
            for _ in range(3):
                opt.zero_grad()
                logits = _forward(net, x[:32])
                torch.nn.functional.cross_entropy(logits, target).backward()
                opt.step()

    assert calibrated_layer_counts[0] > 0, "first model did not calibrate"
    assert calibrated_layer_counts[1] > 0, (
        "second model in the same process did not calibrate "
        f"(regression: {calibrated_layer_counts})"
    )
    assert calibrated_layer_counts[0] == calibrated_layer_counts[1]


def test_two_layer_input_mode_1_builds_and_forwards():
    """A 2-layer EI network with input_mode=1 must build and forward cleanly."""
    params = {
        "architecture": {
            "excitatory_layer_sizes": [16, 8],
            "inhibitory_layer_sizes": [4, 4],
            "excitatory_branch_factors": [2, 2],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [4, 4],
            "ei_synapses_per_branch_per_layer": [2, 2],
            "ie_synapses_per_branch_per_layer": [2, 2],
            "ii_synapses_per_branch_per_layer": [0, 0],
        },
        "transfer": {"input_mode": 1},  # the PR fix: input_mode=1 with >1 layer
        "morphology": {"use_shunting": True},
    }
    net = get_architecture("einet", params, input_dim=16)

    # shunting requires non-negative (conductance-like) inputs
    x = torch.rand(4, 16)
    y = _forward(net, x)
    assert y.shape[0] == 4
    assert torch.isfinite(y).all()


def test_two_layer_input_mode_1_additive_builds_and_forwards():
    """Same as above for raw additive morphology (covers branch_inhibition path)."""
    params = {
        "architecture": {
            "excitatory_layer_sizes": [16, 8],
            "inhibitory_layer_sizes": [4, 4],
            "excitatory_branch_factors": [2, 2],
            "inhibitory_branch_factors": [1],
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [4, 4],
            "ei_synapses_per_branch_per_layer": [2, 2],
            "ie_synapses_per_branch_per_layer": [2, 2],
            "ii_synapses_per_branch_per_layer": [0, 0],
        },
        "transfer": {"input_mode": 1},
    }
    net = get_architecture("dendritic_additive", params, input_dim=16)

    x = torch.randn(4, 16)
    y = _forward(net, x)
    assert y.shape[0] == 4
    assert torch.isfinite(y).all()
