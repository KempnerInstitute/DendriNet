"""Unit tests for the extracted pure broadcast / pathway-role helpers."""

from __future__ import annotations

import logging
import math
from types import SimpleNamespace

import torch

from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast import (
    apply_broadcast_bandwidth,
    compute_local_mismatch_broadcast,
    compute_role_selectivity,
    expand_parent_signal_to_children,
    expand_soma_error_to_compartments,
    normalize_role_mass,
    reduce_error_to_scalar,
    resolve_recent_role_profile,
    role_mass_from_weight_matrix,
)
from dendritic_modeling.training.strategies.local_learning_parts.local_learning_broadcast_transport_mixin import (
    LocalLearningPathTransportBroadcastMixin,
    _align_parent_path,
    _block_size,
    _default_path_factor,
    _edge_weights_like_child,
    _expand_scalar_like,
    _expected_child_out_features,
    _feedback_child_seed,
    _iter_distal_parent_records,
    _path_branch_factor,
    _scalar_edge_mean_factor,
    _soma_seed_from_delta,
    _transport_child_error,
    _warn_child_out_mismatch,
)


def test_apply_broadcast_bandwidth_modes():
    e = torch.tensor([[-2.0, 1.0, 0.0, 3.0]])
    # sign_only: sign(e) * mean(|e|)
    out = apply_broadcast_bandwidth(SimpleNamespace(), e, "sign_only")
    torch.testing.assert_close(out, torch.sign(e) * e.abs().mean())
    # unknown bandwidth -> passthrough
    torch.testing.assert_close(
        apply_broadcast_bandwidth(SimpleNamespace(), e, "none"), e
    )
    # sparse_topk keeps only the top fraction by magnitude
    cfg = SimpleNamespace(broadcast_topk_fraction=0.5)
    sp = apply_broadcast_bandwidth(cfg, e, "sparse_topk")
    assert int((sp != 0).sum()) == 2  # 50% of 4
    assert sp[0, 0] == -2.0 and sp[0, 3] == 3.0  # largest-|.| retained


def test_reduce_error_to_scalar_preserves_mean_and_near_zero_fallback():
    delta = torch.tensor([[2.0, 4.0], [1.0, -1.0], [-3.0, 0.5]])

    out = reduce_error_to_scalar(delta)

    torch.testing.assert_close(out, torch.tensor([[3.0], [1.0], [-1.25]]))


def test_reduce_error_to_scalar_flattens_higher_rank_errors():
    delta = torch.tensor([[[1.0, -1.0], [0.25, -0.25]], [[2.0, 4.0], [6.0, 8.0]]])

    out = reduce_error_to_scalar(delta)

    torch.testing.assert_close(out, torch.tensor([[1.0], [5.0]]))


def test_compute_local_mismatch_broadcast_matches_stabilized_formula():
    delta_scalar = torch.tensor([[0.5], [-0.25]], dtype=torch.float32)
    v_n = torch.tensor([[0.0, 0.1], [0.2, 0.3]], dtype=torch.float32)
    parent = torch.tensor([[0.4, 0.1], [0.2, -0.1]], dtype=torch.float32)

    out = compute_local_mismatch_broadcast(
        delta_scalar=delta_scalar,
        v_n=v_n,
        parent=parent,
        residual_fraction=0.25,
        clip_value=3.0,
    )

    base = delta_scalar.expand_as(v_n)
    mismatch = parent - v_n
    mismatch = mismatch - mismatch.mean(dim=0, keepdim=True)
    rms = mismatch.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-6)
    expected = 0.75 * base * (mismatch / rms).clamp(-3.0, 3.0) + 0.25 * base
    torch.testing.assert_close(out, expected)


def test_normalize_role_mass_rows_sum_to_one():
    rm = torch.tensor([[1.0, 3.0], [0.0, 0.0]])
    out = normalize_role_mass(rm)
    torch.testing.assert_close(out[0], torch.tensor([0.25, 0.75]))
    assert torch.isfinite(out).all()  # zero row handled via clamp


def test_compute_role_selectivity_uniform_vs_onehot():
    uniform = torch.full((1, 4), 0.25)
    onehot = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    assert compute_role_selectivity(uniform).item() < 1e-4  # max entropy -> ~0
    assert compute_role_selectivity(onehot).item() > 0.99  # min entropy -> ~1
    # single role -> selectivity 1
    assert compute_role_selectivity(torch.ones(2, 1)).tolist() == [1.0, 1.0]


def test_expand_parent_signal_to_children_repeat_and_truncate():
    sig = torch.tensor([[1.0, 2.0]])
    # block_size=2 -> [1,1,2,2]; truncate to 3
    torch.testing.assert_close(
        expand_parent_signal_to_children(sig, 2, 3), torch.tensor([[1.0, 1.0, 2.0]])
    )
    # tile up to 6
    torch.testing.assert_close(
        expand_parent_signal_to_children(sig, 1, 6),
        torch.tensor([[1.0, 2.0, 1.0, 2.0, 1.0, 2.0]]),
    )


def test_expand_soma_error_to_compartments_preserves_ancestry_blocks():
    delta = torch.tensor([[1.0, 2.0], [-1.0, 0.5]])

    expanded = expand_soma_error_to_compartments(delta, out_features=6)

    assert expanded is not None
    torch.testing.assert_close(
        expanded,
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                [-1.0, -1.0, -1.0, 0.5, 0.5, 0.5],
            ]
        ),
    )


def test_expand_soma_error_to_compartments_rejects_ambiguous_width():
    delta = torch.ones(2, 3)

    assert expand_soma_error_to_compartments(delta, out_features=8) is None


def test_soma_seed_from_delta_preserves_vector_and_scalar_paths():
    soma_v = torch.zeros(2, 3, dtype=torch.float64)
    delta = torch.tensor([[1.0, 2.0, 3.0], [0.5, 0.25, -1.0]])
    delta_scalar = torch.tensor([[0.1], [0.2]])

    torch.testing.assert_close(
        _soma_seed_from_delta(soma_v, delta, delta_scalar),
        delta.to(dtype=soma_v.dtype),
    )

    mismatched_delta = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    torch.testing.assert_close(
        _soma_seed_from_delta(soma_v, mismatched_delta, delta_scalar),
        delta_scalar.to(dtype=soma_v.dtype).expand(-1, soma_v.size(1)),
    )


def test_align_parent_path_preserves_legacy_alignment_branches():
    parent_v = torch.zeros(2, 3, dtype=torch.float64)

    torch.testing.assert_close(
        _align_parent_path(1.0, parent_v),
        torch.ones_like(parent_v),
    )

    scalar_path = torch.tensor([[0.5], [2.0]], dtype=torch.float64)
    torch.testing.assert_close(
        _align_parent_path(scalar_path, parent_v),
        scalar_path.expand(-1, parent_v.size(1)),
    )

    vector_path = torch.tensor(
        [[0.5, 0.75, 1.0], [2.0, 3.0, 4.0]],
        dtype=torch.float64,
    )
    assert _align_parent_path(vector_path, parent_v) is vector_path


def test_default_path_factor_preserves_scalar_tensor_and_missing_paths():
    child_v = torch.zeros(2, 3, dtype=torch.float64)

    scalar_factor = _default_path_factor(child_v, "scalar_mean")
    torch.testing.assert_close(scalar_factor, torch.ones(2, 1, dtype=child_v.dtype))

    vector_factor = _default_path_factor(child_v, "per_branch")
    torch.testing.assert_close(vector_factor, torch.ones_like(child_v))

    assert _default_path_factor(None, "per_branch") == 1.0


def test_precompute_path_propagation_uses_default_soma_factor_modes():
    soma_v = torch.zeros(2, 3, dtype=torch.float64)
    transport = LocalLearningPathTransportBroadcastMixin()

    transport.local_cfg = SimpleNamespace(
        morphology_aware=SimpleNamespace(path_factor_mode="per_branch")
    )
    per_branch = transport._precompute_path_propagation_factors([{"v_n": soma_v}])
    torch.testing.assert_close(per_branch[-1], torch.ones_like(soma_v))

    transport.local_cfg = SimpleNamespace(
        morphology_aware=SimpleNamespace(path_factor_mode="scalar_mean")
    )
    scalar = transport._precompute_path_propagation_factors([{"v_n": soma_v}])
    torch.testing.assert_close(
        scalar[-1],
        torch.ones(soma_v.size(0), 1, dtype=soma_v.dtype),
    )


def test_iter_distal_parent_records_preserves_child_parent_order():
    distal = {"name": "distal"}
    middle = {"name": "middle"}
    soma = {"name": "soma"}
    records = [distal, middle, soma]

    pairs = list(_iter_distal_parent_records(records))

    assert pairs == [(1, middle, soma), (0, distal, middle)]
    assert list(_iter_distal_parent_records([])) == []
    assert list(_iter_distal_parent_records([soma])) == []


def test_expand_scalar_like_matches_reference_shape_dtype_and_device():
    reference = torch.zeros(2, 4, dtype=torch.float64)
    scalar = torch.tensor([[0.25], [-0.5]], dtype=torch.float32)

    expanded = _expand_scalar_like(reference, scalar)

    torch.testing.assert_close(
        expanded,
        scalar.to(dtype=reference.dtype).expand(-1, reference.size(1)),
    )
    assert expanded.dtype == reference.dtype
    assert expanded.device == reference.device


def test_block_size_preserves_default_and_int_coercion():
    assert _block_size(SimpleNamespace()) == 1
    assert _block_size(SimpleNamespace(block_size="3")) == 3


def test_edge_weights_like_child_detaches_and_matches_child_dtype_device():
    class _Block:
        def __init__(self):
            self.raw_weight = torch.tensor([1.0, -2.0], requires_grad=True)

        def weight(self):
            return self.raw_weight

    child_v = torch.zeros(3, 2, dtype=torch.float64)
    edge_weights = _edge_weights_like_child(_Block(), child_v)

    torch.testing.assert_close(
        edge_weights,
        torch.tensor([1.0, -2.0], dtype=child_v.dtype),
    )
    assert edge_weights.requires_grad is False
    assert edge_weights.dtype == child_v.dtype
    assert edge_weights.device == child_v.device


def test_expected_child_out_features_uses_edge_weight_count():
    assert _expected_child_out_features(torch.zeros(5)) == 5
    assert _expected_child_out_features(torch.zeros(2, 3)) == 6


def test_warn_child_out_mismatch_preserves_legacy_message(caplog):
    child_v = torch.zeros(2, 3)
    logger_name = (
        "dendritic_modeling.training.strategies."
        "local_learning_broadcast_transport_mixin"
    )

    with caplog.at_level(logging.WARNING, logger=logger_name):
        _warn_child_out_mismatch(
            "Path transport",
            child_v=child_v,
            expected_child_out=5,
        )

    assert (
        "Path transport shape mismatch: child_out=3, expected=5. "
        "Falling back to scalar averaging."
    ) in caplog.text


def test_parent_activation_derivative_helper_preserves_disabled_and_fallback_paths():
    transport = LocalLearningPathTransportBroadcastMixin()
    parent_rec = {"name": "parent"}
    parent_v = torch.zeros(2, 3, dtype=torch.float64)

    def fail_if_called(*_args):
        raise AssertionError("activation derivative lookup should be skipped")

    transport._get_layer_activation_derivative = fail_if_called
    torch.testing.assert_close(
        transport._parent_activation_derivative_or_ones(
            parent_rec,
            parent_v,
            enabled=False,
        ),
        torch.ones_like(parent_v),
    )

    calls = []

    def tensor_derivative(rec, v_n):
        calls.append((rec, v_n))
        return torch.full_like(v_n.float(), 0.25)

    transport._get_layer_activation_derivative = tensor_derivative
    derivative = transport._parent_activation_derivative_or_ones(parent_rec, parent_v)
    torch.testing.assert_close(derivative, torch.full_like(parent_v, 0.25))
    assert len(calls) == 1
    assert calls[0][0] is parent_rec
    assert calls[0][1] is parent_v
    assert derivative.dtype == parent_v.dtype

    transport._get_layer_activation_derivative = lambda *_args: None
    torch.testing.assert_close(
        transport._parent_activation_derivative_or_ones(parent_rec, parent_v),
        torch.ones_like(parent_v),
    )


def test_parent_resistance_total_preserves_stabilized_formula():
    transport = LocalLearningPathTransportBroadcastMixin()
    parent_rec = {"name": "parent"}
    parent_v = torch.zeros(2, 2, dtype=torch.float64)
    conductance = torch.tensor([[1.0, 3.0], [0.5, 2.0]], dtype=torch.float64)
    calls = []

    def total_conductance(rec, v_n):
        calls.append((rec, v_n))
        return conductance

    transport._compute_layer_total_conductance = total_conductance

    resistance = transport._parent_resistance_total(parent_rec, parent_v)

    torch.testing.assert_close(resistance, 1.0 / (conductance + 1e-8))
    assert len(calls) == 1
    assert calls[0][0] is parent_rec
    assert calls[0][1] is parent_v


def test_scalar_edge_mean_factor_preserves_shape_and_formula():
    parent_signal = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    parent_act_deriv = torch.tensor([[0.5, 0.25], [1.0, 0.75]])
    parent_r_tot = torch.tensor([[2.0, 3.0], [0.5, 1.5]])
    edge_weights = torch.tensor([0.25, 0.5, 0.75])

    factor = _scalar_edge_mean_factor(
        parent_signal,
        parent_act_deriv,
        parent_r_tot,
        edge_weights,
    )

    expected = (parent_signal * parent_act_deriv * parent_r_tot).mean(
        dim=1, keepdim=True
    ) * edge_weights.mean()
    torch.testing.assert_close(factor, expected)
    assert factor.shape == (2, 1)


def test_path_branch_factor_preserves_expansion_product_and_scalar_mode():
    parent_path = torch.tensor([[1.0, 2.0]])
    parent_act_deriv = torch.tensor([[0.5, 0.25]])
    parent_r_tot = torch.tensor([[2.0, 3.0]])
    edge_weights = torch.tensor([0.25, 0.5, 0.75, 1.0])

    per_branch = _path_branch_factor(
        parent_path=parent_path,
        parent_act_deriv=parent_act_deriv,
        parent_r_tot=parent_r_tot,
        edge_weights=edge_weights,
        block_size=2,
        child_out_features=4,
        mode="per_branch",
    )
    scalar_mean = _path_branch_factor(
        parent_path=parent_path,
        parent_act_deriv=parent_act_deriv,
        parent_r_tot=parent_r_tot,
        edge_weights=edge_weights,
        block_size=2,
        child_out_features=4,
        mode="scalar_mean",
    )

    expected = (
        torch.tensor([[1.0, 1.0, 2.0, 2.0]])
        * torch.tensor([[0.5, 0.5, 0.25, 0.25]])
        * torch.tensor([[2.0, 2.0, 3.0, 3.0]])
        * edge_weights.reshape(1, -1)
    )
    torch.testing.assert_close(per_branch, expected)
    torch.testing.assert_close(scalar_mean, expected.mean(dim=1, keepdim=True))


def test_transport_child_error_preserves_expansion_product_dtype_and_device():
    parent_error = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    parent_act_deriv = torch.tensor([[0.5, 0.25]], dtype=torch.float32)
    parent_r_tot = torch.tensor([[2.0, 3.0]], dtype=torch.float32)
    edge_weights = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    child_v = torch.zeros(1, 4, dtype=torch.float64)

    transported = _transport_child_error(
        parent_error=parent_error,
        parent_act_deriv=parent_act_deriv,
        parent_r_tot=parent_r_tot,
        edge_weights=edge_weights,
        block_size=2,
        child_v=child_v,
    )

    expected = (
        torch.tensor([[0.5, 0.5, 0.5, 0.5]], dtype=torch.float64)
        * torch.tensor([[2.0, 2.0, 3.0, 3.0]], dtype=torch.float64)
        * edge_weights.reshape(1, -1)
    )
    torch.testing.assert_close(transported, expected)
    assert transported.dtype == child_v.dtype
    assert transported.device == child_v.device


def test_feedback_child_seed_preserves_expansion_product_dtype_and_device():
    parent_feedback = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    edge_weights = torch.tensor([0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
    child_v = torch.zeros(1, 4, dtype=torch.float64)

    feedback = _feedback_child_seed(
        parent_feedback=parent_feedback,
        edge_weights=edge_weights,
        block_size=2,
        child_v=child_v,
    )

    expected = torch.tensor([[1.0, 1.0, 2.0, 2.0]], dtype=torch.float64)
    expected = expected * edge_weights.reshape(1, -1)
    torch.testing.assert_close(feedback, expected)
    assert feedback.dtype == child_v.dtype
    assert feedback.device == child_v.device


def test_soma_seed_from_delta_preserves_legacy_delta_reshaping():
    one_dim_soma = torch.zeros(2, 1)
    one_dim_delta = torch.tensor([0.5, -0.25])
    delta_scalar = torch.tensor([[1.0], [2.0]])

    torch.testing.assert_close(
        _soma_seed_from_delta(one_dim_soma, one_dim_delta, delta_scalar),
        one_dim_delta.unsqueeze(-1),
    )

    high_rank_soma = torch.zeros(2, 4)
    high_rank_delta = torch.arange(8, dtype=torch.float32).view(2, 2, 2)
    torch.testing.assert_close(
        _soma_seed_from_delta(high_rank_soma, high_rank_delta, delta_scalar),
        high_rank_delta.view(2, -1),
    )

    assert _soma_seed_from_delta(None, high_rank_delta, delta_scalar) is None


def test_role_mass_from_weight_matrix():
    w = torch.tensor([[1.0, -2.0]])  # out=1, in=2
    roles = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # in=2, roles=2
    torch.testing.assert_close(
        role_mass_from_weight_matrix(w, roles), torch.tensor([[1.0, 2.0]])
    )
    assert role_mass_from_weight_matrix(w, None) is None
    assert role_mass_from_weight_matrix(w, torch.ones(3, 2)) is None  # dim mismatch


def test_resolve_recent_role_profile_prefers_matching_base_then_recent():
    base = torch.ones(4, 2)
    assert resolve_recent_role_profile(4, base, []) is base
    # base wrong size -> search prior outputs (reversed) for a 3-row profile
    prior = [torch.ones(5, 2), torch.ones(3, 2)]
    out = resolve_recent_role_profile(3, base, prior)
    assert out is prior[1]
    assert resolve_recent_role_profile(9, None, prior) is None


def test_max_entropy_constant_matches_log():
    # guards the entropy normalization used in selectivity
    assert math.isclose(math.log(4.0), math.log(4))
