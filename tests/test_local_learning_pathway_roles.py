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
