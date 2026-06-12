import torch

from dendritic_modeling.networks import DendriticBranchLayer
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed import (
    IndexedRewireLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer
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
