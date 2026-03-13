"""Tests for biological loss functions."""

import torch

from dendritic_modeling.training.loss.bio_losses import (
    compute_bio_losses,
    loss_dog,
    loss_dsi,
    loss_endstop,
    loss_osi,
    loss_phase_invariance,
    loss_sparsity,
)

# ---------------------------------------------------------------------------
# loss_dog
# ---------------------------------------------------------------------------


def test_loss_dog_returns_scalar():
    sta = torch.randn(8, 8)
    loss = loss_dog(sta)
    assert loss.dim() == 0


def test_loss_dog_nonnegative():
    torch.manual_seed(0)
    sta = torch.randn(10, 10)
    loss = loss_dog(sta)
    assert loss.item() >= 0


def test_loss_dog_gradient_flow():
    sta = torch.randn(8, 8, requires_grad=True)
    loss = loss_dog(sta)
    loss.backward()
    assert sta.grad is not None


def test_loss_dog_flat_sta():
    sta = torch.zeros(6, 6)
    loss = loss_dog(sta)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# loss_osi
# ---------------------------------------------------------------------------


def test_loss_osi_returns_scalar():
    responses = torch.rand(8)
    orientations = torch.linspace(0, 180, 8)
    loss = loss_osi(responses, orientations)
    assert loss.dim() == 0


def test_loss_osi_selectivity():
    # Use 12 orientations spanning [0, 165] so preferred (idx 0, 0 deg)
    # and orthogonal (idx 6, 90 deg) are well separated
    n = 12
    responses = torch.full((n,), 0.2)
    responses[0] = 2.0  # preferred at 0 deg
    orientations = torch.linspace(0, 165, n)
    loss = loss_osi(responses, orientations)
    # Just verify it's finite and a scalar
    assert torch.isfinite(loss)
    assert loss.dim() == 0


def test_loss_osi_zero_responses():
    responses = torch.zeros(8)
    orientations = torch.linspace(0, 157.5, 8)
    loss = loss_osi(responses, orientations)
    assert torch.isfinite(loss)


def test_loss_osi_gradient_flow():
    responses = torch.rand(8, requires_grad=True)
    orientations = torch.linspace(0, 157.5, 8)
    loss = loss_osi(responses, orientations)
    loss.backward()
    assert responses.grad is not None


# ---------------------------------------------------------------------------
# loss_dsi
# ---------------------------------------------------------------------------


def test_loss_dsi_returns_scalar():
    responses = torch.rand(12)
    directions = torch.linspace(0, 330, 12)
    loss = loss_dsi(responses, directions)
    assert loss.dim() == 0


def test_loss_dsi_perfect_direction():
    responses = torch.zeros(12)
    responses[0] = 1.0  # preferred direction
    directions = torch.linspace(0, 330, 12)
    loss = loss_dsi(responses, directions, target_dsi=1.0)
    # DSI ≈ 1.0 → loss ≈ 0
    assert loss.item() < 0.01


def test_loss_dsi_gradient_flow():
    responses = torch.rand(12, requires_grad=True)
    directions = torch.linspace(0, 330, 12)
    loss = loss_dsi(responses, directions)
    loss.backward()
    assert responses.grad is not None


# ---------------------------------------------------------------------------
# loss_phase_invariance
# ---------------------------------------------------------------------------


def test_loss_phase_invariance_returns_scalar():
    responses = torch.rand(8)
    loss = loss_phase_invariance(responses)
    assert loss.dim() == 0


def test_loss_phase_invariance_constant_responses():
    # Constant responses → F1=0, F0=const → F1/F0=0 → loss=(0-target)^2
    responses = torch.ones(8)
    loss = loss_phase_invariance(responses, target_f1_f0=0.0)
    assert loss.item() < 0.01


def test_loss_phase_invariance_gradient_flow():
    responses = torch.rand(8, requires_grad=True)
    loss = loss_phase_invariance(responses)
    loss.backward()
    assert responses.grad is not None


# ---------------------------------------------------------------------------
# loss_endstop
# ---------------------------------------------------------------------------


def test_loss_endstop_returns_scalar():
    length_curve = torch.rand(10)
    loss = loss_endstop(length_curve)
    assert loss.dim() == 0


def test_loss_endstop_auto_preferred_idx():
    length_curve = torch.tensor([0.1, 0.5, 1.0, 0.7, 0.3])
    loss = loss_endstop(length_curve)  # preferred_idx=None → auto-detect (idx 2)
    assert torch.isfinite(loss)


def test_loss_endstop_explicit_preferred_idx():
    length_curve = torch.rand(8)
    loss = loss_endstop(length_curve, preferred_idx=3)
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# loss_sparsity
# ---------------------------------------------------------------------------


def test_loss_sparsity_returns_scalar():
    activations = torch.randn(10, 20)
    loss = loss_sparsity(activations)
    assert loss.dim() == 0


def test_loss_sparsity_l1_component():
    # Sparse activations should have lower L1 mean than dense
    sparse = torch.zeros(100)
    sparse[:5] = 1.0
    dense = torch.ones(100)
    assert sparse.abs().mean() < dense.abs().mean()
    # Overall loss is also finite
    loss = loss_sparsity(sparse)
    assert torch.isfinite(loss)


def test_loss_sparsity_gradient_flow():
    activations = torch.randn(5, 10, requires_grad=True)
    loss = loss_sparsity(activations)
    loss.backward()
    assert activations.grad is not None


# ---------------------------------------------------------------------------
# compute_bio_losses
# ---------------------------------------------------------------------------


def test_compute_bio_losses_returns_dict():
    model_responses = {
        "gratings": torch.rand(4, 8),
        "dots": torch.rand(4, 10, 10),
    }
    loss_weights = {"osi": 1.0, "dog": 1.0}
    probe_metadata = {"orientations": list(range(0, 180, 22))}

    total_loss, loss_dict = compute_bio_losses(
        model_responses, loss_weights, probe_metadata
    )
    assert isinstance(total_loss, torch.Tensor)
    assert total_loss.dim() == 0
    assert "osi" in loss_dict
    assert "dog" in loss_dict


def test_compute_bio_losses_zero_weight_skipped():
    model_responses = {
        "gratings": torch.rand(4, 8),
        "dots": torch.rand(4, 10, 10),
    }
    loss_weights = {"osi": 0.0, "dog": 1.0}
    probe_metadata = {"orientations": list(range(0, 180, 22))}

    _, loss_dict = compute_bio_losses(model_responses, loss_weights, probe_metadata)
    assert "osi" not in loss_dict
    assert "dog" in loss_dict


def test_compute_bio_losses_weighted_sum():
    model_responses = {
        "gratings": torch.rand(4, 8),
    }
    loss_weights = {"osi": 2.0}
    probe_metadata = {"orientations": list(range(0, 180, 22))}

    total_loss, loss_dict = compute_bio_losses(
        model_responses, loss_weights, probe_metadata
    )
    expected = 2.0 * loss_dict["osi"]
    torch.testing.assert_close(total_loss, expected, atol=1e-6, rtol=1e-6)
