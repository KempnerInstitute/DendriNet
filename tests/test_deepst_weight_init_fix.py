"""
Tests for the _initialize_new_weight fix in DeepstLinear.

Bug: the EXPERIMENTAL block in _initialize_new_weight used
  value = math.log(self.weight_threshold)
instead of
  value = math.log(max(self.weight_init, 1e-10))

This caused every new connection to be born at exactly the death threshold,
making it immediately eligible for re-pruning on the very next step.

Fix: replace with math.log(max(self.weight_init, 1e-10)).

Tests
-----
1. test_new_weight_value_equals_weight_init
   Direct unit test: forces a pruning event on a DeepstLinear and checks that
   the new pre_w values equal log(weight_init), not log(weight_threshold).

2. test_new_weight_above_threshold_after_pruning
   Structural sanity: after a pruning+reinit cycle, no active connection should
   have a weight == weight_threshold exactly (the pre-fix symptom).

3. test_rewiring_survives_200_epochs
   Integration test: trains a tiny EINet for 200 epochs with a high
   weight_threshold (1e-3) and weight_init (1e-2) and verifies that:
     - at least one rewiring event occurred (confirms high threshold triggers pruning)
     - no active connection was ever born at log(weight_threshold)
     - the final network still produces valid outputs (no NaN/Inf)
"""

import math
import unittest

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# DeepstLinear import
# ---------------------------------------------------------------------------
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
    DeepstLinear,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_layer(
    in_features: int = 20,
    out_features: int = 4,
    weight_threshold: float = 1e-3,
    weight_init: float = 1e-2,
    target_density: float = 0.5,
) -> DeepstLinear:
    layer = DeepstLinear(
        in_features=in_features,
        out_features=out_features,
        param_space="log",
        weight_threshold=weight_threshold,
        weight_init=weight_init,
        target_density=target_density,
        rewiring_mode="global",
        use_noise=False,
        init_method="xavier_normal",
        freeze_connectivity=False,
    )
    return layer


def _force_prune_some_weights(layer: DeepstLinear, fraction: float = 0.3) -> int:
    """
    Manually set a fraction of active pre_w values to log(weight_threshold) - epsilon
    so they will be pruned on the next apply_noise_and_enforce_constraints call.
    Returns the number of connections forced below threshold.
    """
    with torch.no_grad():
        active_indices = layer.mask.nonzero(as_tuple=False)
        n_to_force = max(1, int(len(active_indices) * fraction))
        chosen = active_indices[:n_to_force]
        below = math.log(max(layer.weight_threshold * 0.5, 1e-15))
        for idx in chosen:
            layer.pre_w.data[idx[0], idx[1]] = below
    return n_to_force


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestWeightInitFix(unittest.TestCase):

    def test_new_weight_value_equals_weight_init(self):
        """
        After a forced pruning event, new pre_w values should equal
        log(weight_init), not log(weight_threshold).
        """
        weight_threshold = 1e-3
        weight_init = 1e-2
        layer = _make_layer(weight_threshold=weight_threshold, weight_init=weight_init)

        # Record mask before pruning
        mask_before = layer.mask.clone()
        n_forced = _force_prune_some_weights(layer, fraction=0.4)
        self.assertGreater(n_forced, 0, "No connections forced below threshold")

        # Trigger rewiring
        layer.apply_noise_and_enforce_constraints()

        # Find connections that are active now but were not before the prune
        # (these are the newly created connections)
        mask_after = layer.mask.clone()
        new_connections = mask_after & ~mask_before

        if new_connections.sum() == 0:
            self.skipTest(
                "No new connections were created (all pruned slots may have been "
                "refilled from within the existing active set). "
                "Increase target_density or fraction."
            )

        expected_pre_w = math.log(max(weight_init, 1e-10))
        expected_bug_value = math.log(max(weight_threshold, 1e-10))
        tol = 1e-5

        new_pre_w = layer.pre_w.data[new_connections]
        for i, val in enumerate(new_pre_w.tolist()):
            with self.subTest(connection=i):
                self.assertAlmostEqual(
                    val, expected_pre_w, places=4,
                    msg=(
                        f"New connection pre_w={val:.6f} should equal "
                        f"log(weight_init)={expected_pre_w:.6f}. "
                        f"If it equals log(weight_threshold)={expected_bug_value:.6f} "
                        f"the bug is still present."
                    ),
                )
                # Explicit check that the old bug value is NOT present
                self.assertFalse(
                    abs(val - expected_bug_value) < tol,
                    msg=(
                        f"New connection pre_w={val:.6f} equals "
                        f"log(weight_threshold)={expected_bug_value:.6f} — "
                        f"the bug has NOT been fixed."
                    ),
                )

    def test_new_weight_above_threshold_after_pruning(self):
        """
        After rewiring, no active connection should have a weight exactly equal
        to weight_threshold (the symptom of the bug — born-at-threshold connections).
        """
        weight_threshold = 1e-3
        weight_init = 1e-2
        layer = _make_layer(weight_threshold=weight_threshold, weight_init=weight_init)

        _force_prune_some_weights(layer, fraction=0.4)
        layer.apply_noise_and_enforce_constraints()

        active_weights = layer.weight()[layer.mask]
        at_threshold = (active_weights - weight_threshold).abs() < 1e-9
        n_at_threshold = at_threshold.sum().item()

        self.assertEqual(
            n_at_threshold, 0,
            msg=(
                f"{n_at_threshold} active connection(s) have weight exactly equal to "
                f"weight_threshold={weight_threshold}. This is the symptom of the bug."
            ),
        )

    def test_all_active_weights_above_threshold(self):
        """
        After rewiring, all active connections should have weight > weight_threshold.
        """
        layer = _make_layer(weight_threshold=1e-3, weight_init=1e-2)
        _force_prune_some_weights(layer, fraction=0.5)
        layer.apply_noise_and_enforce_constraints()

        active_weights = layer.weight()[layer.mask]
        below = (active_weights <= 1e-3).sum().item()
        self.assertEqual(
            below, 0,
            msg=(
                f"{below} active connection(s) have weight <= weight_threshold after "
                f"rewiring. New connections should be born above the threshold."
            ),
        )

    def test_rewiring_survives_200_epochs(self):
        """
        Integration test: train a tiny 1-layer network for 200 epochs using a
        high weight_threshold (1e-3) and weight_init (1e-2).

        Design note: natural gradient descent may not reliably drive log-space
        weights from their initial values (~exp(-2.0) ≈ 0.13) below the 1e-3
        threshold within 200 steps.  To guarantee at least one rewiring event —
        which is the interesting condition to test — we artificially force a
        batch of weights below threshold at epoch 0, then let the remaining
        199 epochs train naturally with weight_decay to reinforce further
        natural pruning.

        Checks:
          - At least one rewiring event occurred (from the forced pruning at epoch 0).
          - Zero new connections were ever born at log(weight_threshold)
            (the symptom of the pre-fix bug).
          - Final outputs contain no NaN or Inf values.
        """
        torch.manual_seed(42)

        in_features      = 20
        out_features     = 4
        weight_threshold = 1e-3
        weight_init      = 1e-2
        n_epochs         = 200
        batch_size       = 16

        layer = _make_layer(
            in_features=in_features,
            out_features=out_features,
            weight_threshold=weight_threshold,
            weight_init=weight_init,
            target_density=0.5,
        )

        # weight_decay pushes log-space weights toward 0, helping natural pruning
        optimizer = torch.optim.SGD(layer.parameters(), lr=0.01, weight_decay=0.05)
        loss_fn = nn.MSELoss()

        log_wi = math.log(max(weight_init, 1e-10))
        log_wt = math.log(max(weight_threshold, 1e-10))
        tol    = 1e-4

        total_rewire_events = 0
        bug_births_seen     = 0
        mask_prev = layer.mask.clone()

        for epoch in range(n_epochs):
            # At epoch 0 force some weights below threshold to guarantee at least
            # one observable rewiring event in the 200-epoch run.
            if epoch == 0:
                _force_prune_some_weights(layer, fraction=0.3)

            x = torch.randn(batch_size, in_features)
            y_target = torch.randn(batch_size, out_features)

            optimizer.zero_grad()
            y_pred = layer(x)
            loss = loss_fn(y_pred, y_target)
            loss.backward()
            optimizer.step()

            # Rewiring is not called inside forward(); trigger it explicitly.
            layer.apply_noise_and_enforce_constraints()

            mask_now = layer.mask.clone()
            new_conns = mask_now & ~mask_prev

            if new_conns.sum() > 0:
                total_rewire_events += new_conns.sum().item()
                new_pre_w = layer.pre_w.data[new_conns]
                born_at_threshold = (new_pre_w - log_wt).abs() < tol
                bug_births_seen += born_at_threshold.sum().item()

            mask_prev = mask_now

        # The forced pruning at epoch 0 guarantees at least one rewiring event.
        self.assertGreater(
            total_rewire_events, 0,
            msg="No rewiring events observed — check apply_noise_and_enforce_constraints.",
        )

        # Core assertion: new connections are born at log(weight_init), not log(weight_threshold).
        self.assertEqual(
            bug_births_seen, 0,
            msg=(
                f"{bug_births_seen} new connection(s) were born at "
                f"pre_w ≈ log(weight_threshold) = {log_wt:.4f} "
                f"(expected log(weight_init) = {log_wi:.4f}). "
                f"The bug is still present."
            ),
        )

        # Final outputs should be numerically valid.
        with torch.no_grad():
            out = layer(torch.randn(8, in_features))
        self.assertFalse(torch.isnan(out).any().item(), "Final output contains NaN.")
        self.assertFalse(torch.isinf(out).any().item(), "Final output contains Inf.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
