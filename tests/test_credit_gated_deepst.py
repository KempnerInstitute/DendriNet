import torch

from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    AdaptiveInitConfig,
    DeepSTConfig,
    DendriticBranchLayer,
    DendriticSynapseConfig,
    TopKConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    CreditGatedDeepstLinear,
    DeepstLinear,
)
from dendritic_modeling.training.optimizers.custom import (
    apply_indexed_rewiring_after_step,
)


def _credit_layer(**kwargs) -> CreditGatedDeepstLinear:
    defaults = {
        "in_features": 5,
        "out_features": 1,
        "rewiring_mode": "constant_branch",
        "synapses_per_branch": 2,
        "use_noise": False,
        "weight_transform": "exp",
        "credit_candidate_pool_size": 3,
        "credit_turnover_fraction": 0.5,
        "credit_min_observations": 1,
        "credit_warmup_steps": 0,
    }
    defaults.update(kwargs)
    return CreditGatedDeepstLinear(**defaults)


def test_credit_trace_observes_only_active_and_candidate_contacts():
    layer = _credit_layer()
    layer.mask.zero_()
    layer.mask[0, [0, 1]] = True
    layer.credit_candidate_mask.zero_()
    layer.credit_candidate_mask[0, [2, 3]] = True

    gradient = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    layer.record_local_credit(gradient)

    assert torch.equal(layer.credit_trace, torch.tensor([[1.0, 2.0, 3.0, 4.0, 0.0]]))
    assert torch.equal(layer.credit_observations, torch.tensor([[1, 1, 1, 1, 0]]))


def test_credit_rewiring_selects_positive_first_order_swap_and_preserves_budget():
    layer = _credit_layer(credit_swap_margin=0.0)
    layer.mask.zero_()
    layer.mask[0, [0, 1]] = True
    layer.credit_candidate_mask.zero_()
    layer.credit_candidate_mask[0, [2, 3]] = True
    layer.credit_observations[0, [0, 1, 2, 3]] = 4
    layer.credit_trace[0] = torch.tensor([1.0, 0.0, -1.0, 0.5, 0.0])
    layer.pre_w.data[0] = torch.log(torch.tensor([0.4, 0.8, 0.2, 0.2, 0.2]))
    transferred = layer.pre_w.data[0, 0].clone()

    layer.apply_noise_and_enforce_constraints()

    assert layer.mask.sum().item() == 2
    assert not bool(layer.mask[0, 0])
    assert bool(layer.mask[0, 2])
    assert torch.equal(layer.pre_w.data[0, 2], transferred)
    assert layer.credit_total_swaps.item() == 1
    rewired = layer.consume_rewired_mask()
    assert rewired is not None
    assert bool(rewired[0, 0]) and bool(rewired[0, 2])


def test_negative_credit_benefit_does_not_force_turnover():
    layer = _credit_layer()
    layer.mask.zero_()
    layer.mask[0, [0, 1]] = True
    layer.credit_candidate_mask.zero_()
    layer.credit_candidate_mask[0, [2, 3]] = True
    layer.credit_observations[0, [0, 1, 2, 3]] = 1
    layer.credit_trace[0] = torch.tensor([-2.0, -1.0, 0.0, 1.0, 0.0])

    original = layer.mask.clone()
    layer.apply_noise_and_enforce_constraints()

    assert torch.equal(layer.mask, original)
    assert layer.credit_total_swaps.item() == 0


def test_rewiring_clears_adam_moments_for_removed_and_grown_slots():
    layer = _credit_layer(credit_force_turnover=True, credit_selection="random")
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    layer(torch.randn(4, 5)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    state = optimizer.state[layer.pre_w]
    state["exp_avg"].fill_(1.0)
    state["exp_avg_sq"].fill_(1.0)

    layer.credit_observations[layer.mask | layer.credit_candidate_mask] = 1
    count = apply_indexed_rewiring_after_step(layer, optimizer)

    assert count == 1
    changed = (state["exp_avg"] == 0) & (state["exp_avg_sq"] == 0)
    assert changed.sum().item() == 2


def test_typed_config_constructs_credit_gated_branch_layer():
    config = DendriticSynapseConfig(
        topk=TopKConfig(type="credit_deepst"),
        initialization=AdaptiveInitConfig(
            dbl_init_method="default", adaptive_initialization=False
        ),
        deepst=DeepSTConfig(
            use_noise=False,
            rewiring_mode="constant_branch",
            synapses_per_branch=2,
            credit_candidate_pool_size=3,
        ),
    )
    branch = DendriticBranchLayer(
        output_dim=2,
        excitatory_input_dim=5,
        excitatory_synapses_per_branch=2,
        synapse_config=config,
    )

    assert isinstance(branch.branch_excitation, CreditGatedDeepstLinear)
    assert branch.branch_excitation.credit_candidate_pool_size == 3
    assert torch.equal(branch.branch_excitation.mask.sum(dim=1), torch.tensor([2, 2]))


def test_standard_deepst_reports_changed_slots_for_optimizer_reset():
    layer = DeepstLinear(
        4,
        1,
        rewiring_mode="constant_branch",
        synapses_per_branch=2,
        use_noise=False,
        weight_threshold=0.5,
        weight_transform="identity",
        weight_init_range=(0.1, 0.2),
    )
    layer.apply_noise_and_enforce_constraints()

    rewired = layer.consume_rewired_mask()
    assert rewired is not None
    assert rewired.sum().item() >= 2
    assert layer.consume_rewired_mask() is None
