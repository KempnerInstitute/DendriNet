"""Sparse selection, maintenance, rewiring schedules and checkpoint state."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from test_operator_config_contract import _config

from dendritic_modeling.config.affine_mlp import AffineMLPConfig
from dendritic_modeling.models.base import BaseModel
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.classical.autoencoder.vae import (
    MLPVariationalAutoencoder,
)
from dendritic_modeling.networks.architectures.classical.mlp.affine_mlp import (
    AffineMLP,
    build_affine_mlp,
)
from dendritic_modeling.networks.architectures.classical.mlp.matched_param import (
    SparseStructuredMLP,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic import (
    AdaptiveInitConfig,
    DeepSTConfig,
    DendriticBranchLayer,
    DendriticSynapseConfig,
    TopKConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.blocklinear import (
    BlockLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse import (
    kernels as kernel_backends,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.credit_deepst import (
    CreditGatedDeepstLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.deepst import (
    DeepstLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.dense_to_sparse import (
    DenseToSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.impact_rewiring import (
    CellwiseImpactRewiring,
    ImpactRewiringConfig,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_common import (
    _dispatch_indexed_sparse_projection,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_dynamic import (
    IndexedDynamicTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_release import (
    IndexedReleaseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_rewire import (
    IndexedRewireLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_sparse import (
    IndexedSparseLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.kernels import (
    backend as backend_module,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.stochastic import (
    StochasticTopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.topk import (
    TopKLinear,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.variance import (
    VarianceTopKLinear,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent import (
    FixedLegendreMemory,
    FixedLegendreMemoryConfig,
    HeterogeneousLeakCTRNN,
    HeterogeneousLeakCTRNNConfig,
)
from dendritic_modeling.training.optimizers.custom import (
    CustomWeightDecayOptimizer,
    apply_sparse_topology_updates_after_step,
)

RATE = 0.1


def _deepst_branch(kind="deepst", **deepst):
    config = DendriticSynapseConfig(
        topk=TopKConfig(type=kind),
        initialization=AdaptiveInitConfig(
            dbl_init_method="default", adaptive_initialization=False
        ),
        deepst=DeepSTConfig(**deepst),
    )
    return DendriticBranchLayer(
        output_dim=4,
        excitatory_input_dim=10,
        excitatory_synapses_per_branch=2,
        synapse_config=config,
    )


def _count_rewiring_events(layer):
    events = []
    original = layer.apply_noise_and_enforce_constraints

    def counted():
        events.append(int(layer.rewire_step))
        return original()

    layer.apply_noise_and_enforce_constraints = counted
    return events


# 1. Fixed-index layers have no unselected candidates: boosting is ignored.
@pytest.mark.parametrize("cls", [IndexedSparseLinear, IndexedRewireLinear])
@pytest.mark.parametrize("transform", ["exp", "softplus", "identity"])
def test_fixed_index_decay_ignores_boosting(cls, transform):
    torch.manual_seed(0)
    plain = cls(10, 4, 3, weight_transform=transform, seed=5)
    boosted = copy.deepcopy(plain)
    before = plain.sparse_weight().detach().clone()
    plain.decay_weights(RATE, False)
    boosted.decay_weights(RATE, True)
    assert torch.equal(plain.pre_w, boosted.pre_w)
    torch.testing.assert_close(boosted.sparse_weight().detach(), before * (1 - RATE))


# 2. DeepST rewires exactly every rewire_frequency optimizer steps on every
# path, with a checkpointed counter; frequency 1 is unchanged.
def test_deepst_rewire_frequency_holds_on_branch_and_direct_paths():
    torch.manual_seed(0)
    branch = _deepst_branch(rewire_frequency=3, use_noise=True)
    events = _count_rewiring_events(branch.branch_excitation)
    for _ in range(7):
        branch.apply_rewiring()
    assert events == [3, 6]

    torch.manual_seed(1)
    layer = DeepstLinear(8, 3, target_density=0.4, rewire_frequency=3)
    direct = _count_rewiring_events(layer)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
    for _ in range(7):
        apply_sparse_topology_updates_after_step(layer, optimizer)
    assert direct == [3, 6]
    assert int(layer.rewire_step) == 7

    restored = DeepstLinear(8, 3, target_density=0.4, rewire_frequency=3)
    restored.load_state_dict(layer.state_dict())
    assert int(restored.rewire_step) == 7
    legacy = {k: v for k, v in layer.state_dict().items() if k != "rewire_step"}
    DeepstLinear(8, 3, target_density=0.4).load_state_dict(legacy)

    with pytest.raises(ValueError, match="rewire_frequency"):
        DeepstLinear(8, 3, rewire_frequency=0)


def test_deepst_default_frequency_matches_unconditional_rewiring():
    torch.manual_seed(2)
    reference = DeepstLinear(8, 3, target_density=0.4, sigma=0.5, weight_threshold=0.3)
    candidate = copy.deepcopy(reference)
    torch.manual_seed(3)
    for _ in range(4):
        reference.apply_noise_and_enforce_constraints()
    torch.manual_seed(3)
    for _ in range(4):
        candidate.apply_rewiring()
    assert torch.equal(reference.mask, candidate.mask)
    assert torch.equal(reference.pre_w, candidate.pre_w)


# A. The branch-layer step counter survives a state_dict round trip.
def test_branch_step_counter_survives_state_dict_round_trip():
    torch.manual_seed(4)
    branch = _deepst_branch(rewire_frequency=5)
    for _ in range(3):
        branch.apply_rewiring()
    assert branch.step_counter == 3
    torch.manual_seed(4)
    reloaded = _deepst_branch(rewire_frequency=5)
    reloaded.load_state_dict(branch.state_dict())
    assert reloaded.step_counter == 3


# 3. Maintenance skips frozen layers and layers the optimizer does not own.
def test_custom_weight_decay_skips_frozen_and_unowned_sparse_layers():
    torch.manual_seed(5)
    trained, frozen, unowned = (TopKLinear(6, 3, 2) for _ in range(3))
    for layer in (trained, frozen, unowned):
        layer.initialize()
    rewire = IndexedRewireLinear(
        6, 3, 2, rewire_frequency=1, rewire_quantile=0.5, seed=1
    )
    model = nn.Sequential(trained, frozen, unowned, rewire)
    frozen.pre_w.requires_grad_(False)
    rewire.pre_w.requires_grad_(False)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.SGD([trained.pre_w], lr=1.0)  # ``unowned`` excluded
    CustomWeightDecayOptimizer(
        model, optimizer, weight_decay=RATE, weight_boosting=True
    ).step()
    after = model.state_dict()
    changed = {k for k in after if not torch.equal(before[k], after[k])}
    assert changed == {"0.pre_w"}

    apply_sparse_topology_updates_after_step(model, optimizer)
    assert int(rewire.rewire_step) == 0
    assert torch.equal(rewire.connection_indices, before["3.connection_indices"])


# 4. credit_deepst raises when no LocalCA credit arrives after warmup.
def test_credit_deepst_requires_local_credit_after_warmup():
    def layer(**kwargs):
        torch.manual_seed(6)
        return CreditGatedDeepstLinear(
            in_features=6,
            out_features=2,
            rewiring_mode="constant_branch",
            synapses_per_branch=2,
            use_noise=False,
            credit_warmup_steps=2,
            credit_min_observations=1,
            **kwargs,
        )

    backprop = layer()
    backprop.apply_rewiring()
    backprop.apply_rewiring()
    with pytest.raises(ValueError, match="local_ca"):
        backprop.apply_rewiring()

    local = layer()
    for _ in range(4):
        local.record_local_credit(torch.zeros_like(local.pre_w))
        local.apply_rewiring()

    frozen = layer(freeze_connectivity=True)
    for _ in range(4):
        frozen.apply_rewiring()


# 5. Cores without containers forward maintenance to every sparse leaf once.
def test_ctrnn_legendre_affine_and_ss_mlp_forward_maintenance():
    ctrnn = HeterogeneousLeakCTRNN(
        HeterogeneousLeakCTRNNConfig(
            input_dim=2,
            input_projection_dims=[3],
            n_excitatory_outputs=2,
            n_inhibitory_outputs=1,
            channels_per_output=4,
            tau_values=[9.0, 3.0],
            owner_tau_patterns=[
                {"owner_count": 3, "tau_counts": [2, 2], "excitatory_owner_count": 2}
            ],
            input_contacts=30,
            recurrent_excitatory_contacts=18,
            recurrent_inhibitory_contacts=9,
            readout_contacts=4,
            topology_seed=711,
            initialization_seed=919,
        )
    )
    memory = FixedLegendreMemory(
        FixedLegendreMemoryConfig(
            input_dim=3,
            output_dim=2,
            memory_order=4,
            n_memory_channels=2,
            theta=7.5,
            dt=0.25,
            input_contacts=5,
            readout_contacts=9,
            topology_seed=101,
            initialization_seed=211,
        )
    )
    for core, expected_leaves in ((ctrnn, 4), (memory, 2)):
        leaves = [m for m in core.modules() if isinstance(m, IndexedSparseLinear)]
        assert len(leaves) >= expected_leaves
        before = [leaf.sparse_weight().detach().clone() for leaf in leaves]
        BaseModel(Identity(input_dim=1), core, Identity(input_dim=1)).decay_weights(
            RATE, True
        )
        for leaf, weight in zip(leaves, before):
            torch.testing.assert_close(leaf.sparse_weight().detach(), weight * 0.9)
        core.apply_rewiring()

    torch.manual_seed(7)
    topk = build_affine_mlp(
        AffineMLPConfig(dimensions=[6, 5, 3], mode="topk", fanin=[3, 2])
    )
    assert isinstance(topk, AffineMLP)
    assert list(topk.state_dict()) == list(nn.Sequential(*topk).state_dict())
    banks = [m for m in topk.modules() if isinstance(m, TopKLinear)]
    before = [bank.pre_w.detach().clone() for bank in banks]
    topk.decay_weights(RATE, True)
    assert all(not torch.equal(b.pre_w, w) for b, w in zip(banks, before))
    topk.apply_rewiring()

    assert callable(SparseStructuredMLP.apply_rewiring)


def test_base_model_maintains_affine_topk_decoder_once():
    torch.manual_seed(8)
    core = build_affine_mlp(AffineMLPConfig(dimensions=[6, 4], mode="topk", fanin=[3]))
    decoder = nn.Sequential(
        build_affine_mlp(AffineMLPConfig(dimensions=[4, 3], mode="topk", fanin=[2]))
    )
    model = BaseModel(Identity(input_dim=6), core, decoder)
    leaves = [m for m in model.modules() if isinstance(m, TopKLinear)]
    assert len(leaves) == 2
    masks = [leaf.weight_mask().bool() for leaf in leaves]
    before = [leaf.pre_w.detach().clone() for leaf in leaves]
    model.decay_weights(RATE, False)
    for leaf, mask, weight in zip(leaves, masks, before):
        torch.testing.assert_close(leaf.pre_w[mask], weight[mask] * (1 - RATE))
        assert torch.equal(leaf.pre_w[~mask], weight[~mask])
    model.apply_rewiring()


# 6. Step counters advance once per optimizer step, not once per timestep.
def test_dense_to_sparse_and_release_age_count_steps_not_timesteps():
    config = _config("population_network", "shunting", "param_tanh")
    layer = config["population_network"]["layers"][0]
    layer["recurrent"] = True
    layer["population_defaults"].update(
        topk_type="dense_to_sparse",
        rec_excitatory_synapses=2,
        dense_to_sparse_end_step=1000,
    )
    layer["connections"].append({"source": "e", "target": "e", "timing": "delayed"})
    torch.manual_seed(9)
    network = get_architecture("population_network", config, input_dim=6)
    network.train()
    network(torch.rand(2, 5, 6)).sum().backward()
    steps = {
        int(m.sparsity_step)
        for m in network.modules()
        if isinstance(m, DenseToSparseLinear)
    }
    assert steps == {1}

    feedforward = DenseToSparseLinear(6, 3, 2, initial_k=6)
    feedforward(torch.rand(2, 6))
    feedforward(torch.rand(2, 6))
    assert int(feedforward.sparsity_step) == 2

    bank = IndexedReleaseLinear(12, 4, 4, seed=8, weight_transform="exp")
    for _ in range(5):
        bank(torch.rand(3, 12))
    assert int(bank.contact_age.max()) == 0
    controller = CellwiseImpactRewiring(
        [(2, [bank])], ImpactRewiringConfig(frequency=100, maturity_steps=1)
    )
    controller.step(torch.optim.SGD(bank.parameters(), lr=0.1))
    assert bank.contact_age.eq(1).all()


# 7. Boosting never touches entries that can never be selected.
@pytest.mark.parametrize("cls", [TopKLinear, DenseToSparseLinear])
@pytest.mark.parametrize("transform", ["exp", "softplus", "identity"])
def test_boosting_skips_forbidden_and_masked_entries(cls, transform):
    forbidden = torch.tensor([0, 1, -1, 5])
    connection_mask = torch.ones(4, 8, dtype=torch.bool)
    connection_mask[:, 6] = False
    kwargs = {"initial_k": 2} if cls is DenseToSparseLinear else {}
    torch.manual_seed(10)
    layer = cls(
        8,
        4,
        2,
        weight_transform=transform,
        forbidden_input_index_per_output=forbidden,
        connection_mask=connection_mask,
        **kwargs,
    )
    layer.initialize()
    blocked = ~connection_mask
    blocked[torch.tensor([0, 1, 3]), torch.tensor([0, 1, 5])] = True
    mask = layer.weight_mask().bool()
    pre = layer.pre_w.detach().clone()
    weight = layer.weight().detach().clone()
    layer.decay_weights(RATE, True)
    assert torch.equal(layer.pre_w[blocked], pre[blocked])
    candidates = ~mask & ~blocked
    torch.testing.assert_close(
        layer.weight().detach()[candidates],
        weight[candidates] * (1 + RATE),
        rtol=1e-4,
        atol=1e-6,
    )
    # Forbidden entries used to be boosted on every call and overflowed after
    # about 950 calls at this rate.
    for _ in range(1200):
        layer.decay_weights(RATE, True)
    assert torch.isfinite(layer.pre_w).all()
    assert torch.equal(layer.pre_w[blocked], pre[blocked])


# 8. Signed (identity) weights are ranked and pruned by magnitude.
def test_identity_transform_selection_uses_magnitude():
    row = torch.tensor([[-5.0, -4.0, -3.0, 0.1, 0.2, 0.3, 0.4, 0.5]])
    stochastic = StochasticTopKLinear(
        8, 1, 3, weight_transform="identity", temperature=0.01
    )
    stochastic.pre_w.data.copy_(row)
    variance = VarianceTopKLinear(8, 1, 3, weight_transform="identity")
    variance.pre_w.data.copy_(row)
    for layer in (stochastic, variance):
        assert layer.weight_mask()[0].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]

    deepst = DeepstLinear(
        8,
        1,
        target_density=0.5,
        weight_transform="identity",
        use_noise=False,
        init_method="xavier_normal",
        weight_threshold=0.15,
    )
    deepst.mask.zero_()
    deepst.mask[0, [0, 1, 3, 7]] = True
    deepst.pre_w.data.copy_(row)
    deepst.apply_noise_and_enforce_constraints()
    kept = torch.zeros(8, dtype=torch.bool)
    kept[[0, 1, 7]] = True  # |w| = 5, 4, 0.5 exceed the threshold; 0.1 does not
    assert deepst.mask[0][kept].all() and int(deepst.mask.sum()) == 4
    grown = deepst.mask[0] & ~kept  # the regrown contact starts at +threshold
    assert torch.equal(deepst.pre_w.detach()[0][grown], torch.tensor([0.15]))


# 9. Selection noise and stochastic selection apply in training mode only.
def test_eval_selection_is_deterministic_and_draws_no_random_numbers():
    torch.manual_seed(11)
    layers = [
        TopKLinear(20, 6, 4, noise_level=0.5),
        StochasticTopKLinear(20, 6, 4),
        StochasticTopKLinear(20, 6, 4, ultrafast=False),
        DenseToSparseLinear(20, 6, 4, initial_k=8, noise_level=0.5),
        VarianceTopKLinear(20, 6, 4, noise_level=0.5),
        IndexedDynamicTopKLinear(20, 6, 4, candidate_size=10, noise_level=0.5, seed=1),
        IndexedDynamicTopKLinear(
            20, 6, 4, candidate_size=10, selection="stochastic", seed=1
        ),
    ]
    for layer in layers:
        if isinstance(layer, TopKLinear):
            layer.initialize()
    x = torch.rand(3, 20)
    for layer in layers:
        layer.eval()
        state = torch.get_rng_state()
        with torch.no_grad():
            first, second = layer(x), layer(x)
        assert torch.equal(first, second), type(layer).__name__
        assert torch.equal(state, torch.get_rng_state()), type(layer).__name__
    noisy = layers[0].train()
    assert len({tuple(noisy.weight_mask().flatten().tolist()) for _ in range(10)}) > 1


# B. The VAE encoder returns the latent mean in eval mode.
def test_vae_encoder_samples_only_in_training():
    torch.manual_seed(12)
    vae = MLPVariationalAutoencoder(
        input_dim=6, encoder_hidden_dims=[5], latent_dim=3, decoder_hidden_dims=[5]
    )
    x = torch.rand(4, 6)
    vae.eval()
    with torch.no_grad():
        mean, _ = vae.encoder.latent_distribution(x)
        assert torch.equal(vae.encoder(x), torch.sigmoid(mean))
        assert torch.equal(vae(x), vae(x))
    vae.train()
    assert not torch.equal(vae.encoder(x), vae.encoder(x))


# 10. Variance statistics are buffers outside autograd.
def test_variance_running_stats_do_not_join_autograd():
    torch.manual_seed(13)
    layer = VarianceTopKLinear(8, 3, 2)
    layer.initialize()
    upstream = nn.Linear(4, 8)
    for _ in range(3):
        layer(upstream(torch.randn(5, 4))).sum().backward()
    for name in ("activation_mean", "activation_sq_mean", "activation_var"):
        assert getattr(layer, name).grad_fn is None
    copy.deepcopy(layer)


# 11. Kernels with int32 offsets are never run on shapes that overflow them.
def test_int32_offset_kernels_are_guarded(monkeypatch):
    violations = kernel_backends.indexed_kernel_shape_violations
    big = {"batch_size": 16384, "in_features": 64, "out_features": 140000}
    assert violations("triton_transposed", synapses_per_output=4, **big)
    assert violations("triton_transposed_bf16_compat", synapses_per_output=4, **big)
    assert not violations("triton_fused", synapses_per_output=4, **big)
    gather = {"batch_size": 2100, "in_features": 2**20, "out_features": 256}
    assert violations("triton_fused", synapses_per_output=64, **gather)
    small = {"batch_size": 64, "in_features": 300, "out_features": 129}
    assert not violations("triton_transposed", synapses_per_output=17, **small)
    assert not violations("recompute", synapses_per_output=4, **big)

    monkeypatch.setattr(
        backend_module,
        "indexed_projection_backend_available",
        lambda backend, device=None: True,
    )
    resolve = backend_module.resolve_indexed_projection_backend
    assert resolve("auto", device="cuda", synapses_per_output=4, **big) == "recompute"
    assert (
        resolve("auto", device="cuda", synapses_per_output=17, **small)
        == "triton_transposed"
    )
    with pytest.raises(ValueError, match="int32"):
        resolve("triton_transposed", device="cuda", synapses_per_output=4, **big)

    # A cached transposed resolution of an auto request runs a fitting kernel
    # for an overflowing call (the Triton kernel cannot run on CPU, so a
    # correct output proves the fallback) and keeps the cached resolution.
    monkeypatch.setattr(
        kernel_backends,
        "indexed_kernel_shape_violations",
        lambda backend, **shape: ("simulated",) if backend.startswith("triton") else (),
    )
    x = torch.rand(3, 5)
    indices = torch.tensor([[0, 1], [2, 4]])
    weight = torch.rand(2, 2)
    call = {
        "flat_x": x,
        "weight": weight,
        "connection_indices": indices,
        "output_chunk_size": 8,
        "projection_backend": "triton_transposed",
        "backend_is_resolved": True,
    }
    output, resolved = _dispatch_indexed_sparse_projection(**call, auto_requested=True)
    assert resolved == "triton_transposed"
    torch.testing.assert_close(output, (x[:, indices] * weight).sum(-1))
    with pytest.raises(ValueError, match="int32"):
        _dispatch_indexed_sparse_projection(**call)


# 12. The eligible-fraction floor is the specified behavior (documented).
def test_impact_rewiring_eligible_fraction_floor_is_documented():
    assert "floor(n_eligible * eligible_fraction)" in ImpactRewiringConfig.__doc__
    bank = IndexedReleaseLinear(12, 4, 4, seed=9, weight_transform="exp")
    bank.contact_age.fill_(1000)
    bank.impact.fill_(1.0)
    bank.impact[:, 0] = 0.0
    controller = CellwiseImpactRewiring(
        [(2, [bank])], ImpactRewiringConfig(frequency=1, maturity_steps=1)
    )
    assert controller.step(torch.optim.SGD(bank.parameters(), lr=0.1)) == 0


# C. Rewiring layers checkpoint the allowed-connection mask.
def test_rewiring_layers_checkpoint_allowed_connection_mask():
    mask_a = torch.zeros(3, 12, dtype=torch.bool)
    mask_a[:, :6] = True
    mask_b = torch.zeros(3, 12, dtype=torch.bool)
    mask_b[:, 6:] = True
    for cls in (IndexedRewireLinear, IndexedReleaseLinear):
        kwargs = {"rewire_frequency": 1, "rewire_quantile": 0.5}
        if cls is IndexedReleaseLinear:
            kwargs = {}
        original = cls(12, 3, 2, seed=3, connection_mask=mask_a, **kwargs)
        assert "_allowed_connection_mask" in original.state_dict()
        resumed = cls(12, 3, 2, seed=4, connection_mask=mask_b, **kwargs)
        resumed.load_state_dict(original.state_dict())
        assert torch.equal(resumed._allowed_connection_mask, mask_a)
        if cls is IndexedRewireLinear:
            for _ in range(5):
                resumed.apply_rewiring()
            rows = torch.arange(3)[:, None]
            assert mask_a[rows, resumed.connection_indices.long()].all()
        legacy = {
            k: v
            for k, v in original.state_dict().items()
            if k != "_allowed_connection_mask"
        }
        cls(12, 3, 2, seed=4, connection_mask=mask_b, **kwargs).load_state_dict(legacy)
    assert (
        "_allowed_connection_mask"
        not in IndexedSparseLinear(
            12, 3, 2, seed=3, connection_mask=mask_a
        ).state_dict()
    )

    torch.manual_seed(14)
    deepst = DeepstLinear(12, 3, target_density=0.3, connection_mask=mask_a)
    other = DeepstLinear(12, 3, target_density=0.3, connection_mask=mask_b)
    other.load_state_dict(deepst.state_dict())
    assert torch.equal(other._forbidden_connection_mask, ~mask_a)


# D. Deep copies work after a training step and keep gradient-scaling hooks.
def test_deepcopy_keeps_gradient_scaling_hooks_and_drops_graph_tensors():
    config = _config("einet", "shunting", "param_tanh")
    config["sparsity"] = {"gradient_scaling": "conductance_dynamic"}
    config["blocklinear"] = {"gradient_scaling": "conductance_dynamic"}
    torch.manual_seed(15)
    original = get_architecture("einet", config, input_dim=6)
    original.eval()
    with torch.no_grad():
        original(torch.rand(2, 6))
    copied = copy.deepcopy(original)

    def hooked(model):
        return sum(bool(p._backward_hooks) for p in model.parameters())

    assert hooked(original) > 0 and hooked(copied) == hooked(original)
    x = torch.rand(5, 6)
    grads = []
    for model in (original, copied):
        model.train()
        model.zero_grad()
        model(x).square().sum().backward()
        grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
    for a, b in zip(*grads):
        assert torch.equal(a, b)
    copy.deepcopy(original)  # after a training forward/backward


# E. Unseeded rewiring seeds are checkpointed; train() undoes an inference
# fold's freezing; indexed-dynamic re-resolves a changed backend.
def test_unseeded_rewire_seed_is_checkpointed():
    torch.manual_seed(16)
    original = IndexedRewireLinear(16, 4, 3, rewire_frequency=1, rewire_quantile=0.5)
    resumed = IndexedRewireLinear(16, 4, 3, rewire_frequency=1, rewire_quantile=0.5)
    resumed.load_state_dict(original.state_dict())
    for _ in range(3):
        original.apply_rewiring()
        resumed.apply_rewiring()
    assert torch.equal(original.connection_indices, resumed.connection_indices)


def test_train_after_inference_fold_restores_requires_grad():
    block = BlockLinear(8, 4)
    block.initialize(1.0)
    sparse = IndexedSparseLinear(8, 4, 3, seed=1)
    frozen = IndexedSparseLinear(8, 4, 3, seed=1)
    frozen.pre_w.requires_grad_(False)
    for layer, name in ((block, "log_weight"), (sparse, "pre_w"), (frozen, "pre_w")):
        expected = getattr(layer, name).requires_grad
        layer.freeze_for_inference()
        assert not getattr(layer, name).requires_grad
        layer.train()
        assert getattr(layer, name).requires_grad == expected
    sparse(torch.rand(3, 8)).sum().backward()
    assert sparse.pre_w.grad is not None


def test_indexed_dynamic_re_resolves_changed_backend():
    layer = IndexedDynamicTopKLinear(16, 4, 3, seed=1, projection_backend="eager")
    layer(torch.rand(2, 16))
    layer.projection_backend = "scatter_dense"
    layer(torch.rand(2, 16))
    assert layer._last_resolved_projection_backend == "scatter_dense"


# F. The contiguous-blocks layout cache follows the index tensor itself.
def test_contiguous_block_layout_cache_tracks_index_tensor():
    first = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
    second = torch.tensor([[6, 7], [4, 5], [2, 3], [0, 1]])
    layer = IndexedSparseLinear(
        8, 4, 2, connection_indices=first, projection_backend="contiguous_blocks"
    )
    x = torch.rand(3, 8)
    layer(x)
    assert layer._contiguous_block_layout_cache[0] is layer.connection_indices
    state = layer.state_dict()
    state["connection_indices"] = second.to(layer.index_dtype)
    layer.load_state_dict(state, assign=True)
    reference = IndexedSparseLinear(
        8, 4, 2, connection_indices=second, projection_backend="eager"
    )
    reference.pre_w.data.copy_(layer.pre_w.data)
    torch.testing.assert_close(layer(x), reference(x))
    assert layer._contiguous_block_layout_cache[0] is layer.connection_indices
