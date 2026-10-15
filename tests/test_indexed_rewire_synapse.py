"""IndexedRewireLinear: topology rewiring of fixed-fan-in indexed synapses.

The layer keeps one persistent index set per compartment and replaces its
weakest contacts during training; ``CustomWeightDecayOptimizer`` advances the
rewiring after each step and clears the optimizer state of rewired slots.
"""

from math import ceil

import pytest
import torch

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed import (
    IndexedRewireLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _connection_generator,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.optimizers.custom import (
    CustomWeightDecayOptimizer,
    _iter_rewired_optimizer_modules,
)
from dendritic_modeling.utils.sparse_ops import get_available_sparse_types


def test_indexed_rewire_registered_and_stores_only_active_slots():
    assert "indexed_rewire" in get_available_sparse_types()
    layer = IndexedRewireLinear(
        in_features=9,
        out_features=4,
        K=3,
        seed=11,
        weight_transform="softplus",
    )

    assert layer.pre_w.shape == (4, 3)
    assert layer.connection_indices.shape == (4, 3)
    assert layer.weight_mask().sum().item() == 12


def test_indexed_rewire_requires_persistent_topology():
    with pytest.raises(ValueError, match="requires persistent_indices=True"):
        IndexedRewireLinear(
            in_features=6,
            out_features=3,
            K=2,
            seed=7,
            persistent_indices=False,
        )


def test_indexed_rewire_replaces_weakest_slots_per_dendrite():
    layer = IndexedRewireLinear(
        in_features=8,
        out_features=2,
        K=4,
        connection_indices=torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        weight_transform="identity",
        seed=7,
        rewire_frequency=1,
        rewire_quantile=0.5,
    )
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor([[0.1, 0.2, 3.0, 4.0], [0.1, 0.2, 3.0, 4.0]]))

    before = layer.connection_indices.clone()
    layer.apply_rewiring()

    assert int(layer.rewire_step.item()) == 1
    assert torch.all(layer._last_rewired_mask.sum(dim=1) == 2)
    assert torch.equal(layer.connection_indices[:, 2:], before[:, 2:])
    assert not torch.equal(layer.connection_indices[:, :2], before[:, :2])
    assert torch.all(layer.weight_mask().sum(dim=1) == 4)
    for row in layer.connection_indices:
        assert row.unique().numel() == row.numel()


def test_indexed_rewire_inherits_pruned_positive_weight_scale_by_default():
    layer = IndexedRewireLinear(
        in_features=8,
        out_features=1,
        K=4,
        connection_indices=torch.tensor([[0, 1, 2, 3]]),
        weight_transform="softplus",
        seed=7,
        rewire_frequency=1,
        rewire_quantile=0.5,
    )
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor([[-12.0, -10.0, 1.0, 2.0]]))
    raw_before = layer.pre_w.detach().clone()
    effective_before = layer.sparse_weight().detach().clone()

    layer.apply_rewiring()

    assert torch.equal(layer.pre_w, raw_before)
    assert torch.equal(layer.sparse_weight(), effective_before)
    assert torch.all(layer._last_rewired_mask[0, :2])


def test_indexed_rewire_can_request_legacy_fresh_weight_initialization():
    layer = IndexedRewireLinear(
        in_features=8,
        out_features=1,
        K=4,
        connection_indices=torch.tensor([[0, 1, 2, 3]]),
        weight_transform="softplus",
        seed=7,
        rewire_frequency=1,
        rewire_quantile=0.5,
        rewire_init_policy="fresh",
    )
    with torch.no_grad():
        layer.pre_w.copy_(torch.tensor([[-12.0, -10.0, 1.0, 2.0]]))

    layer.apply_rewiring()

    assert not torch.equal(layer.pre_w[0, :2], torch.tensor([-12.0, -10.0]))


def test_indexed_rewire_checkpoint_restores_frequency_and_cutoff_schedule():
    source = IndexedRewireLinear(
        in_features=8,
        out_features=1,
        K=2,
        connection_indices=torch.tensor([[0, 1]]),
        weight_transform="identity",
        seed=17,
        rewire_frequency=3,
        rewire_quantile=0.5,
        rewire_until_step=6,
    )
    for _ in range(4):
        source.apply_rewiring()
    assert int(source.rewire_step.item()) == 4

    restored = IndexedRewireLinear(
        in_features=8,
        out_features=1,
        K=2,
        connection_indices=torch.tensor([[6, 7]]),
        weight_transform="identity",
        seed=17,
        rewire_frequency=3,
        rewire_quantile=0.5,
        rewire_until_step=6,
    )
    restored.load_state_dict(source.state_dict())
    assert restored._rewire_step_host == 4

    indices_at_step_4 = restored.connection_indices.clone()
    restored.apply_rewiring()
    assert restored._rewire_step_host == 5
    assert int(restored.rewire_step.item()) == 5
    assert torch.equal(restored.connection_indices, indices_at_step_4)

    restored.apply_rewiring()
    indices_at_step_6 = restored.connection_indices.clone()
    assert restored._rewire_step_host == 6
    assert int(restored.rewire_step.item()) == 6
    assert not torch.equal(indices_at_step_6, indices_at_step_4)

    restored.apply_rewiring()
    assert restored._rewire_step_host == 7
    assert int(restored.rewire_step.item()) == 7
    assert torch.equal(restored.connection_indices, indices_at_step_6)


def test_indexed_rewire_detects_direct_checkpoint_counter_copy():
    layer = IndexedRewireLinear(
        in_features=8,
        out_features=1,
        K=2,
        connection_indices=torch.tensor([[0, 1]]),
        weight_transform="identity",
        seed=19,
        rewire_frequency=3,
        rewire_quantile=0.5,
    )
    # Some legacy loaders copy into tensors returned by state_dict rather than
    # calling load_state_dict.  The buffer version catches that mutation once.
    layer.state_dict()["rewire_step"].copy_(torch.tensor(2))
    before = layer.connection_indices.clone()

    layer.apply_rewiring()

    assert layer._rewire_step_host == 3
    assert int(layer.rewire_step.item()) == 3
    assert not torch.equal(layer.connection_indices, before)


def test_indexed_rewire_preserves_direct_counter_copy_across_tensor_move():
    layer = IndexedRewireLinear(
        in_features=8,
        out_features=1,
        K=2,
        connection_indices=torch.tensor([[0, 1]]),
        weight_transform="identity",
        seed=23,
        rewire_frequency=3,
        rewire_quantile=0.5,
    )
    layer.state_dict()["rewire_step"].copy_(torch.tensor(2))
    # Cloning every tensor exercises the same tensor-replacement path as a
    # module device transfer without requiring CUDA in the unit test.
    layer._apply(lambda tensor: tensor.clone())
    before = layer.connection_indices.clone()

    layer.apply_rewiring()

    assert layer._rewire_step_host == 3
    assert int(layer.rewire_step.item()) == 3
    assert not torch.equal(layer.connection_indices, before)


def test_indexed_rewire_steady_state_does_not_read_counter_tensor(monkeypatch):
    layer = IndexedRewireLinear(
        in_features=8,
        out_features=1,
        K=2,
        weight_transform="identity",
        seed=29,
        rewire_frequency=1,
        rewire_quantile=0.5,
        rewire_until_step=0,
    )

    def fail_if_synchronized():
        raise AssertionError("steady-state rewiring synchronized the counter tensor")

    monkeypatch.setattr(
        layer,
        "_sync_rewire_step_host_from_buffer",
        fail_if_synchronized,
    )
    for _ in range(10):
        layer.apply_rewiring()

    assert layer._rewire_step_host == 10
    assert int(layer.rewire_step.item()) == 10


def test_indexed_rewire_preserves_auto_index_dtype_at_rewire_boundary():
    layer = IndexedRewireLinear(
        in_features=8,
        out_features=2,
        K=4,
        index_dtype="auto",
        weight_transform="identity",
        seed=7,
        rewire_frequency=1,
        rewire_quantile=0.5,
    )

    assert layer.connection_indices.dtype == torch.int32
    layer.apply_rewiring()
    assert layer.connection_indices.dtype == torch.int32
    assert bool(layer._last_rewired_mask.any())


def test_indexed_rewire_respects_connection_mask():
    connection_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [False, False, True, True, True, True],
        ]
    )
    layer = IndexedRewireLinear(
        in_features=6,
        out_features=2,
        K=2,
        connection_indices=torch.tensor([[0, 1], [2, 3]]),
        connection_mask=connection_mask,
        weight_transform="identity",
        seed=5,
        rewire_frequency=1,
        rewire_quantile=1.0,
    )

    layer.apply_rewiring()

    row = torch.arange(2)[:, None]
    assert bool(connection_mask[row, layer.connection_indices.cpu()].all())
    assert torch.all(layer.weight_mask().sum(dim=1) == 2)


def _legacy_rowwise_rewire_boundary(layer: IndexedRewireLinear) -> None:
    """Test oracle for the row-wise mutation used before batched commits."""

    with torch.no_grad():
        layer._last_rewired_mask.zero_()
        step = layer._rewire_step_value()
        n_rewire = min(layer.K, max(1, ceil(layer.K * layer.rewire_quantile)))
        prune_slots = torch.topk(
            layer.sparse_weight().detach().abs(),
            n_rewire,
            dim=1,
            largest=False,
            sorted=False,
        ).indices
        fresh_weights = (
            layer._new_pre_weights(
                (layer.out_features, n_rewire), layer._rewire_seed + 104729 * step
            )
            if layer.rewire_init_policy == "fresh"
            else None
        )
        conn_cpu = layer.connection_indices.detach().cpu()
        seed_base = layer._rewire_seed + 104729 * step
        for row_idx in range(layer.out_features):
            available_mask = torch.zeros(layer.in_features, dtype=torch.bool)
            allowed = layer._allowed_indices_for_row_cpu(row_idx)
            available_mask[allowed] = True
            available_mask[conn_cpu[row_idx]] = False
            available = torch.nonzero(available_mask, as_tuple=False).flatten()
            if available.numel() == 0:
                continue
            n_select = min(n_rewire, int(available.numel()))
            generator = _connection_generator(seed_base + row_idx)
            selected = available[
                torch.randperm(available.numel(), generator=generator)[:n_select]
            ].to(
                device=layer.connection_indices.device,
                dtype=layer.connection_indices.dtype,
            )
            target_slots = prune_slots[row_idx, :n_select]
            layer.connection_indices[row_idx, target_slots] = selected
            if fresh_weights is not None:
                layer.pre_w.data[row_idx, target_slots] = fresh_weights[
                    row_idx, :n_select
                ]
            layer._last_rewired_mask[row_idx, target_slots] = True


@pytest.mark.parametrize("rewire_init_policy", ["inherit_pruned", "fresh"])
def test_batched_rewire_commit_is_bitwise_identical_to_rowwise_oracle(
    rewire_init_policy,
):
    connection_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False, False, False],
            [False, True, True, True, True, True, False, False, False],
            [False, False, True, True, True, True, True, False, False],
        ]
    )
    kwargs = {
        "in_features": 9,
        "out_features": 3,
        "K": 3,
        "connection_indices": torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4]]),
        "connection_mask": connection_mask,
        "forbidden_input_index_per_output": torch.tensor([4, 5, 6]),
        "weight_transform": "identity",
        "seed": 41,
        "rewire_frequency": 1,
        "rewire_quantile": 2 / 3,
        "rewire_init_policy": rewire_init_policy,
    }
    batched = IndexedRewireLinear(**kwargs)
    rowwise = IndexedRewireLinear(**kwargs)
    rowwise.load_state_dict(batched.state_dict())
    with torch.no_grad():
        weights = torch.tensor([[0.1, 0.2, 3.0], [0.3, 0.1, 4.0], [0.2, 5.0, 0.1]])
        batched.pre_w.copy_(weights)
        rowwise.pre_w.copy_(weights)

    torch.manual_seed(20260831)
    batched.apply_rewiring()
    torch.manual_seed(20260831)
    _legacy_rowwise_rewire_boundary(rowwise)

    assert torch.equal(batched.connection_indices, rowwise.connection_indices)
    assert torch.equal(batched.pre_w, rowwise.pre_w)
    assert torch.equal(batched._last_rewired_mask, rowwise._last_rewired_mask)
    assert batched._last_rewired_any_host is True


def test_consume_rewired_mask_uses_host_state_without_device_reduction(monkeypatch):
    layer = IndexedRewireLinear(
        in_features=8,
        out_features=2,
        K=4,
        seed=7,
        weight_transform="identity",
        rewire_frequency=3,
        rewire_quantile=0.5,
    )

    def fail_any(*_args, **_kwargs):
        raise AssertionError("mask reduction synchronized")

    monkeypatch.setattr(torch.Tensor, "any", fail_any)
    assert layer.consume_rewired_mask() is None


def test_dendritic_branch_layer_builds_indexed_rewire():
    layer = DendriticBranchLayer(
        output_dim=3,
        excitatory_input_dim=8,
        excitatory_synapses_per_branch=2,
        topk_type="indexed_rewire",
        indexed_seed=3,
        indexed_rewire_frequency=1,
        indexed_rewire_quantile=0.5,
        weight_transform="softplus",
        use_shunting=False,
        reactivate=False,
    )

    assert isinstance(layer.branch_excitation, IndexedRewireLinear)
    y = layer(torch.rand(2, 8))
    assert y.shape == (2, 3)
    before = layer.branch_excitation.connection_indices.clone()
    layer.apply_rewiring()
    assert not torch.equal(before, layer.branch_excitation.connection_indices)


def test_custom_optimizer_rewires_after_step_and_clears_adam_state():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.syn = IndexedRewireLinear(
                in_features=6,
                out_features=1,
                K=2,
                connection_indices=torch.tensor([[0, 1]]),
                weight_transform="identity",
                seed=9,
                rewire_frequency=1,
                rewire_quantile=0.5,
            )

        def forward(self, x):
            return self.syn(x)

        def decay_weights(self, weight_decay: float, weight_boosting: bool = False):
            return None

        def apply_rewiring(self):
            self.syn.apply_rewiring()

    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    wrapped = CustomWeightDecayOptimizer(model, optimizer)

    x = torch.randn(4, 6)
    model(x).sum().backward()
    wrapped.step()

    state = optimizer.state[model.syn.pre_w]
    rewired = model.syn._last_rewired_mask
    assert not bool(rewired.any())  # consumed by optimizer-state reset
    assert int((state["exp_avg"] == 0).sum().item()) >= 1
    assert int((state["exp_avg"] != 0).sum().item()) >= 1


def test_rewired_optimizer_module_iterator_preserves_duck_typed_order():
    class RewiredModule(torch.nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.pre_w = torch.nn.Parameter(torch.ones(size))

        def consume_rewired_mask(self):
            return None

    first = RewiredModule(2)
    second = RewiredModule(3)
    missing_weight = torch.nn.Module()
    missing_weight.consume_rewired_mask = lambda: None
    model = torch.nn.Sequential(first, missing_weight, torch.nn.Sequential(second))

    assert list(_iter_rewired_optimizer_modules(model)) == [
        (first, first.pre_w),
        (second, second.pre_w),
    ]


def test_indexed_rewire_compares_to_standard_topk_on_tiny_regression():
    torch.manual_seed(0)
    x = torch.rand(64, 10)
    target = torch.stack(
        [
            x[:, 0] + 0.5 * x[:, 1],
            x[:, 2] + 0.5 * x[:, 3],
            x[:, 4] + 0.5 * x[:, 5],
        ],
        dim=1,
    )

    def train(layer):
        opt = torch.optim.SGD(layer.parameters(), lr=0.1)
        first_loss = None
        for _ in range(30):
            opt.zero_grad()
            loss = (layer(x) - target).square().mean()
            if first_loss is None:
                first_loss = float(loss.detach())
            loss.backward()
            opt.step()
            if hasattr(layer, "apply_rewiring"):
                layer.apply_rewiring()
        return first_loss, float(((layer(x) - target).square().mean()).detach())

    standard = TopKLinear(
        in_features=10,
        out_features=3,
        K=4,
        weight_transform="softplus",
        init_method="xavier_normal",
    )
    indexed_rewire = IndexedRewireLinear(
        in_features=10,
        out_features=3,
        K=4,
        weight_transform="softplus",
        init_method="xavier_normal",
        seed=0,
        rewire_frequency=1,
        rewire_quantile=0.25,
    )

    standard_start, standard_end = train(standard)
    _, rewire_end = train(indexed_rewire)

    assert torch.isfinite(torch.tensor([standard_end, rewire_end])).all()
    assert standard_end < standard_start
    assert indexed_rewire.pre_w.numel() < standard.pre_w.numel()


def test_configured_rewiring_regrows_contacts_with_the_pruned_weight():
    """Configurations select the inherit_pruned policy, whatever init_method is."""
    from dendritic_modeling.config.model import SparsityConfig
    from dendritic_modeling.networks.architectures.factory import get_architecture

    # The indexed options have no rewire_init_policy key.
    assert "rewire_init_policy" not in SparsityConfig().indexed
    states = []
    for init_method in ("xavier_normal", "uniform"):
        layer = {
            "name": "layer0",
            "readout_population": "e",
            "population_defaults": {
                "ff_excitatory_synapses": 2,
                "ff_inhibitory_synapses": 0,
                "topk_type": "indexed_rewire",
                "topk_init_method": init_method,
                "indexed_seed": 5,
                "indexed_rewire_frequency": 1,
                "indexed_rewire_quantile": 0.5,
                "initialization_seed": 3,
            },
            "populations": [{"name": "e", "polarity": "excitatory", "n_neurons": 3}],
            "connections": [{"source": "input", "target": "e"}],
        }
        network = get_architecture(
            "population_network",
            {"biological_neuron": False, "population_network": {"layers": [layer]}},
            input_dim=6,
        )
        rewired = [m for m in network.modules() if isinstance(m, IndexedRewireLinear)]
        assert rewired
        for synapses in rewired:
            assert synapses.rewire_init_policy == "inherit_pruned"
            weights = synapses.pre_w.detach().clone()
            indices = synapses.connection_indices.clone()
            synapses.apply_rewiring()
            assert not torch.equal(synapses.connection_indices, indices)
            assert torch.equal(synapses.pre_w, weights)
        states.append(network.state_dict())
    assert states[0].keys() == states[1].keys()
    assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
