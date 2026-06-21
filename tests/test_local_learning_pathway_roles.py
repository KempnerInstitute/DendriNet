"""Tests for pathway-aligned LocalCA teaching signals and branch roles."""

from __future__ import annotations

import torch
import torch.nn as nn

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.training.strategies.local_learning import (
    LocalCreditAssignment,
    _LayerStats,
)
from dendritic_modeling.training.strategies.local_learning_broadcast import (
    compute_pathway_seed_projection,
)
from dendritic_modeling.training.strategies.local_learning_morphology_mixin import (
    _apply_branch_length_factor,
    _apply_dendritic_length_factor,
    _centered_alignment_gain,
    _default_branch_depths,
    _default_branch_types,
    _normalized_inverse_depth_factor,
    _reset_branch_role_metadata,
    _role_specialization_scale,
    _sum_role_mass_terms,
)


class _RoutedSingleBranchModel(nn.Module):
    """Tiny routed model exposing one dendritic branch layer."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder_network = PathwayRouter(
            input_dim=4,
            router_mode="fixed",
            shared_indices=[],
            pathway_groups=[[0, 1], [2, 3]],
            pathway_dim=2,
            router_activation="none",
        )
        self.branch = DendriticBranchLayer(
            output_dim=2,
            excitatory_input_dim=4,
            excitatory_synapses_per_branch=2,
            inhibitory_input_dim=None,
            inhibitory_synapses_per_branch=None,
            input_branch_factor=None,
            use_shunting=True,
            reactivate=False,
            topk_noise_level=0.0,
            weight_transform="identity",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routed = self.encoder_network(x)
        return self.branch(routed)


def _make_trainer(
    *,
    error_broadcast_mode: str = "pathway_vector",
    pathway_broadcast_residual: float = 0.0,
    pathway_activity_gate_strength: float = 0.0,
    use_branch_role_rules: bool = False,
    specialized_branch_scale: float = 1.0,
    mixed_branch_scale: float = 1.0,
    branch_role_alignment_weight: float = 0.0,
) -> LocalCreditAssignment:
    trainer = LocalCreditAssignment.__new__(LocalCreditAssignment)
    trainer.local_cfg = trainer._build_local_rule_config(
        {
            "rule_variant": "3f",
            "error_broadcast_mode": error_broadcast_mode,
            "pathway_broadcast_residual": pathway_broadcast_residual,
            "pathway_activity_gate_strength": pathway_activity_gate_strength,
            "normalize_by_batch": True,
            "update_inactive_weights": True,
            "three_factor": {
                "dynamics_mode": "conductance",
                "use_conductance_scaling": True,
                "use_driving_force": True,
                "theta": 0.0,
                "e_rev_exc": 1.0,
                "e_rev_inh": 0.0,
            },
            "morphology_aware": {
                "use_branch_role_rules": use_branch_role_rules,
                "specialized_branch_scale": specialized_branch_scale,
                "mixed_branch_scale": mixed_branch_scale,
                "branch_role_power": 1.0,
                "branch_role_alignment_weight": branch_role_alignment_weight,
            },
            "inhibitory_homeostasis": {"enabled": False, "weight": 0.0},
        }
    )
    trainer._layer_stats = {}
    trainer._decoder_cache = {}
    trainer._broadcast_cache = {}
    return trainer


def test_compute_pathway_seed_projection_matches_legacy_projection() -> None:
    role_profile = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float64,
    )
    seed_error = torch.tensor(
        [[[1.0, -1.0], [0.5, -0.5]]],
        dtype=torch.float32,
    )
    delta_scalar = torch.tensor([[0.0]], dtype=torch.float32)

    pathway_error, base = compute_pathway_seed_projection(
        seed_error=seed_error,
        delta_scalar=delta_scalar,
        role_profile=role_profile,
        out_features=4,
    )

    expected_seed = seed_error.view(1, -1).to(dtype=torch.float64)
    expected_pathway_error = (expected_seed @ role_profile) / torch.tensor(
        [[2.0, 2.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(base, expected_seed)
    torch.testing.assert_close(pathway_error, expected_pathway_error)


def test_compute_pathway_seed_projection_falls_back_to_scalar_broadcast() -> None:
    role_profile = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    seed_error = torch.ones(2, 3)
    delta_scalar = torch.tensor([[0.25], [-0.5]])

    pathway_error, base = compute_pathway_seed_projection(
        seed_error=seed_error,
        delta_scalar=delta_scalar,
        role_profile=role_profile,
        out_features=4,
    )

    torch.testing.assert_close(pathway_error, delta_scalar.expand(-1, 2))
    torch.testing.assert_close(base, delta_scalar.expand(-1, 4))


def test_normalized_inverse_depth_factor_preserves_unit_mean_formula() -> None:
    depths = torch.tensor([1.0, 2.0, 4.0])
    offset = 0.5

    factor = _normalized_inverse_depth_factor(depths, offset)
    expected = 1.0 / (depths + offset)
    expected = expected / (expected.mean() + 1e-8)

    torch.testing.assert_close(factor, expected)
    torch.testing.assert_close(factor.mean(), torch.tensor(1.0))


def test_branch_type_scale_uses_normalized_inverse_depth_factor() -> None:
    trainer = _make_trainer()
    fake_layer = nn.Identity()
    stats = _LayerStats()
    stats.branch_types = torch.tensor([0.0, 1.0])
    stats.branch_depths = torch.tensor([1.0, 3.0])
    trainer._layer_stats[id(fake_layer)] = stats
    trainer.local_cfg.morphology_aware.use_branch_length_modulation = True
    trainer.local_cfg.morphology_aware.morphology_depth_offset = 1.0
    trainer.local_cfg.morphology_aware.basal_branch_scale = 0.5
    trainer.local_cfg.morphology_aware.apical_branch_scale = 2.0

    scale = trainer._get_branch_type_scale({"layer": fake_layer})

    assert isinstance(scale, torch.Tensor)
    base = torch.tensor([0.5, 2.0])
    expected = base * _normalized_inverse_depth_factor(stats.branch_depths, 1.0)
    torch.testing.assert_close(scale, expected)


def test_role_specialization_scale_interpolates_selectivity() -> None:
    selectivity = torch.tensor([0.0, 0.5, 1.0])

    scale = _role_specialization_scale(
        selectivity,
        power=2.0,
        specialized_scale=1.4,
        mixed_scale=0.6,
    )

    torch.testing.assert_close(
        scale,
        torch.tensor([0.6, 0.8, 1.4]),
    )


def test_centered_alignment_gain_preserves_center_and_clamp_formula() -> None:
    alignment = torch.tensor([[1.0, 3.0], [-4.0, 4.0]])

    gain = _centered_alignment_gain(alignment, weight=0.5)
    centered = alignment - alignment.mean(dim=1, keepdim=True)
    expected = torch.clamp(1.0 + 0.5 * centered, min=0.0)

    torch.testing.assert_close(gain, expected)
    assert gain[1, 0].item() == 0.0


def test_apply_branch_length_factor_preserves_scalar_and_tensor_paths() -> None:
    length_factor = torch.tensor([0.5, 2.0])

    torch.testing.assert_close(
        _apply_branch_length_factor(1.5, length_factor),
        torch.tensor([0.75, 3.0]),
    )
    torch.testing.assert_close(
        _apply_branch_length_factor(torch.tensor([2.0, 4.0]), length_factor),
        torch.tensor([1.0, 8.0]),
    )


def test_apply_dendritic_length_factor_preserves_branch_and_block_paths() -> None:
    length_factor = torch.tensor([0.5, 2.0])

    torch.testing.assert_close(
        _apply_dendritic_length_factor(torch.tensor([2.0, 4.0]), length_factor),
        torch.tensor([1.0, 8.0]),
    )
    torch.testing.assert_close(
        _apply_dendritic_length_factor(
            torch.tensor([[2.0, 4.0], [6.0, 8.0]]),
            length_factor,
        ),
        torch.tensor([[1.0, 2.0], [12.0, 16.0]]),
    )


def test_default_branch_depths_preserve_layer_index_heuristic() -> None:
    depths = _default_branch_depths(
        3,
        layer_idx=2,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert depths.dtype == torch.float64
    torch.testing.assert_close(depths, torch.full((3,), 3.0, dtype=torch.float64))


def test_default_branch_types_preserve_apical_threshold() -> None:
    basal = _default_branch_types(
        2,
        layer_idx=1,
        n_layers=4,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    apical = _default_branch_types(
        2,
        layer_idx=2,
        n_layers=4,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert basal.dtype == torch.float64
    torch.testing.assert_close(basal, torch.zeros(2, dtype=torch.float64))
    torch.testing.assert_close(apical, torch.ones(2, dtype=torch.float64))


def test_reset_branch_role_metadata_preserves_morphology_annotations() -> None:
    stats = _LayerStats()
    stats.branch_depths = torch.tensor([1.0, 2.0])
    stats.branch_types = torch.tensor([0.0, 1.0])
    stats.branch_role_profile = torch.ones(2, 2)
    stats.branch_role_selectivity = torch.ones(2)
    stats.input_role_profile = torch.ones(3, 2)
    stats.block_input_role_profile = torch.ones(4, 2)
    stats.branch_block_roles = torch.ones(2, 2, 2)
    stats.branch_block_selectivity = torch.ones(2, 2)

    _reset_branch_role_metadata(stats)

    torch.testing.assert_close(stats.branch_depths, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(stats.branch_types, torch.tensor([0.0, 1.0]))
    assert stats.branch_role_profile is None
    assert stats.branch_role_selectivity is None
    assert stats.input_role_profile is None
    assert stats.block_input_role_profile is None
    assert stats.branch_block_roles is None
    assert stats.branch_block_selectivity is None


def test_sum_role_mass_terms_preserves_left_to_right_addition() -> None:
    terms = [
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64),
        torch.tensor([[0.5, -1.0], [2.0, -3.0]], dtype=torch.float64),
        torch.tensor([[4.0, 1.0], [-2.0, 0.25]], dtype=torch.float64),
    ]

    expected = terms[0]
    for extra in terms[1:]:
        expected = expected + extra

    torch.testing.assert_close(_sum_role_mass_terms(terms), expected)


def test_pathway_vector_broadcast_projects_error_into_role_subspace() -> None:
    trainer = _make_trainer(
        error_broadcast_mode="pathway_vector",
        pathway_broadcast_residual=0.0,
        pathway_activity_gate_strength=0.0,
    )
    fake_layer = nn.Identity()
    trainer._layer_stats[id(fake_layer)] = _LayerStats()
    trainer._layer_stats[id(fake_layer)].branch_role_profile = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    delta = torch.tensor([[1.0, -1.0, 0.5, -0.5]], dtype=torch.float32)
    delta_scalar = trainer._reduce_error_to_scalar(delta)
    v_n = torch.zeros(1, 4)
    rec = {"layer": fake_layer}

    broadcast = trainer._compute_pathway_vector_broadcast(
        seed_error=delta,
        delta=delta,
        delta_scalar=delta_scalar,
        rec=rec,
        v_n=v_n,
    )

    expected = torch.tensor([[0.75, -0.75, 0.75, -0.75]], dtype=torch.float32)
    torch.testing.assert_close(broadcast, expected)


def test_branch_role_scales_follow_router_connectivity() -> None:
    torch.manual_seed(0)
    model = _RoutedSingleBranchModel()
    trainer = _make_trainer(
        use_branch_role_rules=True,
        specialized_branch_scale=1.4,
        mixed_branch_scale=0.6,
        branch_role_alignment_weight=0.5,
    )

    with torch.no_grad():
        model.branch.branch_excitation.pre_w.copy_(
            torch.tensor(
                [
                    [5.0, 4.0, 0.1, 0.0],
                    [5.0, 0.1, 4.9, 0.0],
                ],
                dtype=torch.float32,
            )
        )

    records, handles = trainer._attach_local_recorders(model)
    try:
        _ = model(torch.tensor([[1.0, 0.5, 0.2, 0.1]], dtype=torch.float32))
    finally:
        for handle in handles:
            handle.remove()

    assert len(records) == 1
    rec = records[0]
    trainer._initialize_branch_morphology(model, records)

    stats = trainer._layer_stats[id(model.branch)]
    assert stats.branch_role_profile is not None
    torch.testing.assert_close(
        stats.branch_role_profile[0],
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        atol=1e-6,
        rtol=1e-6,
    )
    assert stats.branch_role_profile[1, 0].item() > 0.45
    assert stats.branch_role_profile[1, 1].item() > 0.45

    branch_scale, dendritic_scale = trainer._get_branch_role_scales(rec)
    syn_alignment, block_alignment = trainer._get_branch_role_alignment_factors(rec)

    assert isinstance(branch_scale, torch.Tensor)
    assert branch_scale.shape == (2,)
    assert branch_scale[0].item() > branch_scale[1].item()
    assert branch_scale[0].item() > 1.0
    assert branch_scale[1].item() < 1.0
    assert torch.allclose(branch_scale, dendritic_scale)
    assert syn_alignment is not None
    assert block_alignment is None
    assert syn_alignment[0, 0].item() > 1.0
    assert syn_alignment[0, 2].item() < 1.0


def test_branch_role_scales_use_block_selectivity_for_dendritic_scale() -> None:
    trainer = _make_trainer(
        use_branch_role_rules=True,
        specialized_branch_scale=1.5,
        mixed_branch_scale=0.5,
    )
    fake_layer = nn.Identity()
    stats = _LayerStats()
    stats.branch_role_selectivity = torch.tensor([0.0, 1.0])
    stats.branch_block_selectivity = torch.tensor([[0.0, 0.5], [1.0, 0.25]])
    trainer._layer_stats[id(fake_layer)] = stats

    branch_scale, dendritic_scale = trainer._get_branch_role_scales(
        {"layer": fake_layer}
    )

    assert isinstance(branch_scale, torch.Tensor)
    assert isinstance(dendritic_scale, torch.Tensor)
    torch.testing.assert_close(branch_scale, torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(
        dendritic_scale,
        torch.tensor([[0.5, 1.0], [1.5, 0.75]]),
    )


def test_branch_role_alignment_factors_include_block_alignment() -> None:
    trainer = _make_trainer(branch_role_alignment_weight=0.5)
    fake_layer = nn.Identity()
    stats = _LayerStats()
    stats.branch_role_profile = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    stats.input_role_profile = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    stats.branch_block_roles = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    trainer._layer_stats[id(fake_layer)] = stats

    syn_factor, block_factor = trainer._get_branch_role_alignment_factors(
        {"layer": fake_layer}
    )

    assert syn_factor is not None
    assert block_factor is not None
    torch.testing.assert_close(
        syn_factor,
        _centered_alignment_gain(
            stats.branch_role_profile @ stats.input_role_profile.t(),
            0.5,
        ),
    )
    raw_block_alignment = (
        stats.branch_block_roles * stats.branch_role_profile.unsqueeze(1)
    ).sum(dim=-1)
    torch.testing.assert_close(
        block_factor,
        _centered_alignment_gain(raw_block_alignment, 0.5),
    )


def test_feedback_seeds_unpool_parent_error_through_blocklinear() -> None:
    trainer = _make_trainer(error_broadcast_mode="pathway_vector")
    child_v = torch.zeros(1, 4)
    parent_v = torch.zeros(1, 2)
    blk = BlockLinear(4, 2, weight_transform="identity")
    with torch.no_grad():
        blk.log_weight.copy_(torch.tensor([[0.2, 0.4], [0.6, 0.8]]))

    layer_records = [
        {"v_n": child_v},
        {"v_n": parent_v, "blk_module": blk},
    ]
    delta = torch.tensor([[1.0, -2.0]], dtype=torch.float32)
    delta_scalar = trainer._reduce_error_to_scalar(delta)

    feedback = trainer._precompute_feedback_seeds(
        layer_records=layer_records,
        delta=delta,
        delta_scalar=delta_scalar,
    )

    expected_child = torch.tensor([[0.2, 0.4, -1.2, -1.6]], dtype=torch.float32)
    torch.testing.assert_close(feedback[1], delta)
    torch.testing.assert_close(feedback[0], expected_child)
