"""Config isolation, shared primitives and numerical contracts of paired routing."""

from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.nn import functional as F

from dendritic_modeling.config import (
    CoreConfig,
    RoutedEIConfig,
    config_from_mapping,
    load_config,
)
from dendritic_modeling.config.regularization import RegularizationConfig
from dendritic_modeling.config.seeds import derive_seed
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.routed_ei import (
    build_routed_ei,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.functional import additive_voltage, shunting_voltage
from dendritic_modeling.networks.utils.routing import (
    assignment_statistics,
    routing_assignments,
)
from dendritic_modeling.networks.utils.weight_normalization import normalize_weight_rows
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training.regularization import RegularizationManager

FIXED = [0, 1, 2, 0, 1, 2, 0, 1]


def spec(mechanism="shunting", support="matched", fixed=False):
    return RoutedEIConfig(
        features=8,
        branches=3,
        output_dim=2,
        routing="fixed" if fixed else "learned",
        fixed_assignment=FIXED if fixed else [],
        mechanism=mechanism,
        inhibitory_support=support,
        temperature=1.0 if fixed else 0.8,
        leak_conductance=0.25 if mechanism == "shunting" else 1.0,
        epsilon=0.0 if mechanism == "shunting" else 1e-8,
        weight_normalization_floor=1e-6,
        branch_scale=20.0 if mechanism == "shunting" else 1.0,
        branch_center=4.0 / 8.25 if mechanism == "shunting" else 0.0,
        branch_divisor=1.0 if mechanism == "shunting" else 4.0,
    )


class ReferenceReadout(nn.Module):
    """Independent reduced-model oracle; intentionally no shared computation."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        b, f = config.branches, config.features
        self.route_logits = nn.Parameter(torch.zeros(b, f))
        nn.init.normal_(self.route_logits, std=0.05)
        self.e_weight_raw = nn.Parameter(torch.zeros(b, f))
        self.i_weight_raw = nn.Parameter(torch.zeros(b, f))
        nn.init.normal_(self.e_weight_raw, std=0.04)
        nn.init.normal_(self.i_weight_raw, std=0.04)
        if config.branch_activation == "tanh":
            for name in ("activation_slope_raw", "activation_offset"):
                value = torch.full((b,), float(getattr(config, name)))
                if config.trainable_activation:
                    self.register_parameter(name, nn.Parameter(value))
                else:
                    self.register_buffer(name, value)
        self.decoder = nn.Linear(b, config.output_dim)
        self.register_buffer(
            "fixed_assignment", torch.tensor(config.fixed_assignment, dtype=torch.long)
        )

    def forward(self, e, i, *, hard):
        c = self.config
        soft = torch.softmax(self.route_logits / c.temperature, dim=0)
        if self.fixed_assignment.numel():
            a = F.one_hot(self.fixed_assignment, num_classes=c.branches).T.to(e.dtype)
        elif hard:
            discrete = F.one_hot(soft.argmax(0), num_classes=c.branches).T.to(e.dtype)
            a = discrete + soft - soft.detach() if self.training else discrete
        else:
            a = soft
        ew = F.softplus(self.e_weight_raw) * a
        iw = F.softplus(self.i_weight_raw) * a
        ed = e @ (ew / ew.sum(1, keepdim=True).clamp_min(1e-6)).T
        id = i @ (iw / iw.sum(1, keepdim=True).clamp_min(1e-6)).T
        if c.mechanism == "additive":
            result = (ed - id) / 4.0
        else:
            if c.inhibitory_support == "global_mean":
                id = i.mean(1, keepdim=True)
            result = 20.0 * (ed / (0.25 + ed + id) - 4.0 / 8.25)
        if c.branch_activation == "tanh":
            result = torch.tanh(
                F.softplus(self.activation_slope_raw) * result + self.activation_offset
            )
        return self.decoder(result)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize(
    "mechanism,support",
    [("shunting", "matched"), ("shunting", "global_mean"), ("additive", "matched")],
)
def test_initialization_checkpoints_gradients_and_training_are_exact(
    dtype, fixed, hard, mechanism, support
):
    config = spec(mechanism, support, fixed)
    torch.manual_seed(27)
    expected = ReferenceReadout(config).to(dtype)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(27)
    actual = build_routed_ei(config).to(dtype)
    assert torch.equal(rng, torch.get_rng_state())
    assert list(expected.state_dict()) == list(actual.state_dict())
    assert list(dict(expected.named_parameters())) == list(
        dict(actual.named_parameters())
    )
    e = torch.linspace(0.1, 8.0, 40, dtype=dtype).reshape(5, 8)
    i = e.flip(0) * 0.7
    histories = []
    for model in (expected, actual):
        history = []
        opt = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
        for _ in range(3):
            opt.zero_grad()
            inputs = [x.clone().requires_grad_() for x in (e, i)]
            output = model(*inputs, hard=hard)
            output.square().mean().backward()
            history.append(
                [
                    output.detach().clone(),
                    *(x.grad.clone() for x in inputs),
                    *[p.grad.clone() for p in model.parameters() if p.grad is not None],
                ]
            )
            opt.step()
        histories.append(history)
        model.eval()
    for left, right in zip(*histories, strict=True):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key])
    actual.load_state_dict(expected.state_dict(), strict=True)
    torch.testing.assert_close(
        expected(e, i, hard=hard), actual(e, i, hard=hard), rtol=0, atol=0
    )
    assert torch.equal(actual(e, i), actual(torch.cat([e, i], 1)))


def mapping(config=None):
    return {
        "config_version": 4,
        "model": {
            "output_scale_mode": "fixed",
            "encoder": {"type": "identity", "params": {"input_dim": 16}},
            "core": {"type": "routed_ei", "routed_ei": asdict(config or spec())},
            "decoder": {"type": "identity"},
        },
        "training": {"main": {"strategy": "standard"}},
    }


def test_loader_factory_wrapper_and_state_roundtrip(tmp_path):
    path = tmp_path / "run.yaml"
    OmegaConf.save(OmegaConf.create(mapping()), path)
    cfg = load_config(str(path))
    assert asdict(cfg.model.core.routed_ei) == asdict(spec())
    torch.manual_seed(8)
    direct = build_routed_ei(spec())
    rng = torch.get_rng_state().clone()
    torch.manual_seed(8)
    via = get_architecture(
        "routed_ei", cfg.model.core, input_dim=16, suffix_input_dim=2
    )
    assert torch.equal(rng, torch.get_rng_state())
    via.load_state_dict(direct.state_dict(), strict=True)
    x = torch.ones(5, 16)
    assert torch.equal(via(x), direct(x))
    model, _ = initialize_model(cfg.model)
    model.core_network.load_state_dict(direct.state_dict())
    assert torch.equal(model(x), direct(x))
    with pytest.raises(ValueError, match="encoder"):
        get_architecture("routed_ei", cfg.model.core, input_dim=8)
    with pytest.raises(ValueError, match="decoder"):
        get_architecture("routed_ei", cfg.model.core, suffix_input_dim=3)


@pytest.mark.parametrize(
    "change",
    [
        {"features": True},
        {"branches": 0},
        {"temperature": 0},
        {"temperature": float("nan")},
        {"hard_routing": 1},
        {"routing": "unknown"},
        {"routing": "fixed"},
        {"fixed_assignment": [0] * 8},
        {"leak_conductance": 0},
        {"epsilon": -1},
        {"weight_normalization_floor": 0},
        {"mechanism": "additive", "leak_conductance": 0.25},
        {"mechanism": "additive", "inhibitory_support": "global_mean"},
        {"branch_activation": "param_tanh"},
        {"trainable_activation": True},
        {"activation_slope_raw": 1.0},
        {"activation_offset": 0.5},
        {"branch_activation": "tanh", "trainable_activation": 1},
        {"branch_activation": "tanh", "activation_slope_raw": float("inf")},
        {"branch_activation": "tanh", "activation_offset": float("nan")},
        {"routing": "fixed", "fixed_assignment": FIXED, "temperature": 0.8},
        {"routing": "fixed", "fixed_assignment": FIXED, "hard_routing": False},
        {"routing": "fixed", "fixed_assignment": FIXED, "route_init_std": 0.1},
    ],
)
def test_invalid_config_fails_before_rng_changes(change):
    config = RoutedEIConfig(
        **{**asdict(RoutedEIConfig(features=8, branches=3)), **change}
    )
    rng = torch.get_rng_state().clone()
    with pytest.raises(ValueError):
        build_routed_ei(config)
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "core_type",
    [
        "affine_mlp",
        "dendritic_shunting",
        "population_network",
        "shared_projection_shunting",
    ],
)
def test_foreign_core_rejects_routing_section(core_type):
    value = mapping()
    value["model"]["core"]["type"] = core_type
    with pytest.raises(ValueError, match="routed_ei"):
        config_from_mapping(value)


def test_unknown_keys_unsupported_training_and_ignored_fields_rejected():
    value = mapping()
    value["model"]["core"]["routed_ei"]["temprature"] = 0.5
    with pytest.raises(ValueError, match="temprature"):
        config_from_mapping(value)
    value = mapping()
    value["training"]["main"]["strategy"] = "local_ca"
    with pytest.raises(ValueError, match="backpropagation"):
        config_from_mapping(value)
    with pytest.raises(ValueError, match="morphology"):
        get_architecture(
            "routed_ei",
            {"routed_ei": asdict(spec()), "morphology": {"leak_conductance": 2.0}},
        )


def test_mixed_precision_is_rejected_in_configs_and_under_autocast():
    value = mapping()
    value["training"]["main"]["common"] = {"use_amp": True}
    with pytest.raises(ValueError, match="use_amp"):
        config_from_mapping(value)
    model = build_routed_ei(spec())
    x = torch.ones(2, 16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="autocast"):
            model(x)
    assert torch.isfinite(model(x)).all()


def test_temperature_schedules_apply_to_learned_routing_only():
    learned = build_routed_ei(spec())
    learned.temperature = 0.5
    assert learned.temperature == learned.config.temperature == 0.5
    fixed = build_routed_ei(spec(fixed=True))
    with pytest.raises(ValueError, match="fixed"):
        fixed.temperature = 0.5
    assert fixed.temperature == 1.0


def test_new_section_is_none_and_old_models_are_identical():
    assert CoreConfig().routed_ei is None
    before = config_from_mapping(
        {"config_version": 4, "model": {"core": {"biological_neuron": False}}}
    )
    after = config_from_mapping(
        {
            "config_version": 4,
            "model": {"core": {"biological_neuron": False, "routed_ei": None}},
        }
    )
    torch.manual_seed(47)
    old, _ = initialize_model(before.model)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(47)
    new, _ = initialize_model(after.model)
    assert torch.equal(rng, torch.get_rng_state())
    assert list(old.state_dict()) == list(new.state_dict())
    for key, tensor in old.state_dict().items():
        assert torch.equal(tensor, new.state_dict()[key])


def test_explicit_initialization_seed_selects_a_keyed_stream():
    raw = {"routed_ei": asdict(spec()), "initialization_seed": 4}
    rng = torch.get_rng_state().clone()
    a = get_architecture("routed_ei", raw)
    b = get_architecture("routed_ei", raw)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())
    # Keyed by the core type, so an encoder built after manual_seed(4) is not
    # replayed by the core.
    built = []
    for seed in (derive_seed(4, "core.routed_ei.initialization"), 4):
        torch.manual_seed(seed)
        built.append(build_routed_ei(spec()).route_logits)
    assert torch.equal(a.route_logits, built[0])
    assert not torch.equal(a.route_logits, built[1])


def test_reused_pathway_soft_assignments_are_exact():
    router = PathwayRouter(input_dim=8, router_mode="learned", n_pathways=2)
    expected = torch.softmax(
        router.assignment_logits / router.learned_router_temperature, dim=-1
    )
    assert torch.equal(router._learned_assignments(), expected)
    a = torch.autograd.grad(
        router._learned_assignments().square().sum(), router.assignment_logits
    )[0]
    b = torch.autograd.grad(expected.square().sum(), router.assignment_logits)[0]
    assert torch.equal(a, b)


def test_shared_assignment_forward_and_surrogate_gradient():
    x = torch.tensor(
        [[0.2, -0.3, 0.8], [0.5, 0.3, -0.1]], dtype=torch.float64, requires_grad=True
    )
    hard = routing_assignments(x, 0.8, dim=0, hard=True, training=True)
    soft = torch.softmax(x / 0.8, dim=0)
    a = torch.autograd.grad((hard * x.detach()).sum(), x)[0]
    b = torch.autograd.grad((soft * x.detach()).sum(), x)[0]
    assert torch.equal(a, b)
    exact = routing_assignments(x, 0.8, dim=0, hard=True, training=False)
    assert torch.equal(exact.sum(0), torch.ones(3, dtype=x.dtype))


def test_inferred_assignment_resolves_near_ties_like_the_hard_forward():
    model = build_routed_ei(RoutedEIConfig(features=2, branches=2, output_dim=2))
    with torch.no_grad():
        model.route_logits.fill_(0.25)
        # The next float after 0.25: the softmax rounds both probabilities of
        # source 0 to one half, so the forward picks branch 0.
        model.route_logits[1, 0] = torch.nextafter(
            torch.tensor(0.25), torch.tensor(1.0)
        )
    assert model.route_logits.argmax(dim=0).tolist() == [1, 0]
    hard = model.eval().assignment(hard=True)
    assert model.inferred_assignment().tolist() == hard.argmax(dim=0).tolist()
    assert model.inferred_assignment().tolist() == [0, 0]


def test_assignment_entropy_is_finite_when_float16_probabilities_underflow():
    logits = torch.tensor(
        [[0.0, 20.0], [20.0, 0.0], [-20.0, -20.0]], dtype=torch.float16
    )
    soft = routing_assignments(logits, 0.5, dim=0)
    assert bool((soft == 0).any())
    entropy, balance = assignment_statistics(soft, destination_dim=0)
    assert torch.isfinite(entropy) and torch.isfinite(balance)
    # Other dtypes keep the 1e-8 floor.
    small = torch.tensor([[1e-12, 0.5], [1.0 - 1e-12, 0.5]], dtype=torch.float64)
    expected = -(small * torch.log(small.clamp_min(1e-8))).sum(dim=0).mean()
    assert torch.equal(assignment_statistics(small, destination_dim=0)[0], expected)


def test_clamped_sum_is_explicit_and_default_normalization_is_unchanged():
    w = torch.tensor([[1e-9, 2e-9], [0.0, 0.0], [1.0, 2.0]], requires_grad=True)
    robust = normalize_weight_rows(w, 1, 1.0)
    same = normalize_weight_rows(w, 1, 1.0, sum_floor=None)
    assert torch.equal(robust, same)
    clamped = normalize_weight_rows(w, 1, 1.0, sum_floor=1e-6)
    assert torch.equal(clamped, w / w.sum(-1, keepdim=True).clamp_min(1e-6))
    assert robust[0].sum() > 0.99 and clamped[0].sum() < 0.01
    for kwargs in ({"sum_floor": 0}, {"sum_floor": float("nan")}, {"sum_floor": True}):
        with pytest.raises(ValueError):
            normalize_weight_rows(w, 1, 1.0, **kwargs)
    with pytest.raises(ValueError):
        normalize_weight_rows(w, 2, 1.0, sum_floor=1e-6)


def test_default_normalization_keeps_the_competition_term_at_unselected_entries():
    # d drive_b / d a_bk = softplus(r_bk) (x_k - drive_b) / S_b for every k,
    # including the zero entries of a hard-routed row. The straight-through
    # estimator passes it to the logits through the tempered softmax.
    model = build_routed_ei(
        RoutedEIConfig(features=4, branches=2, output_dim=2, temperature=0.7)
    ).double()
    with torch.no_grad():
        model.route_logits.copy_(
            torch.tensor([[0.3, -0.2, 0.1, 0.4], [-0.1, 0.2, -0.3, 0.0]])
        )
    x = torch.tensor([[0.3, 0.8, 0.5, 0.7], [1.1, 0.2, 0.4, 0.9]], dtype=torch.float64)
    excitation, _ = model.drives(x, x, hard=True)
    excitation.sum().backward()
    with torch.no_grad():
        soft = torch.softmax(model.route_logits / 0.7, dim=0)
        positive = F.softplus(model.e_weight_raw)
        weights = positive * F.one_hot(soft.argmax(0), 2).T
        total = weights.sum(1, keepdim=True)
        drive = x @ (weights / total).T
        upstream = positive * ((x[:, None] - drive[:, :, None]) / total).sum(0)
        expected = soft * (upstream - (soft * upstream).sum(0)) / 0.7
    torch.testing.assert_close(
        model.route_logits.grad, expected, rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("hard", [False, True])
def test_default_and_clamped_sum_agree_while_branch_sums_exceed_the_floor(hard):
    torch.manual_seed(5)
    clamped = build_routed_ei(spec())
    default = build_routed_ei(replace(spec(), weight_normalization_floor=None))
    default.load_state_dict(clamped.state_dict())
    assert clamped.inferred_assignment().unique().numel() == 3  # No empty branch.
    e = torch.linspace(0.1, 8.0, 40).reshape(5, 8)
    outputs = [model(e, e.flip(0) * 0.7, hard=hard) for model in (clamped, default)]
    assert torch.equal(outputs[0], outputs[1])
    for output in outputs:
        output.square().mean().backward()
    for a, b in zip(clamped.parameters(), default.parameters(), strict=True):
        assert torch.equal(a.grad, b.grad)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("support", ["matched", "global_mean"])
def test_only_parameters_the_forward_reads_require_gradients(fixed, support):
    model = build_routed_ei(spec(support=support, fixed=fixed))
    assert list(model.state_dict()) == list(build_routed_ei(spec()).state_dict())
    assert model.route_logits.requires_grad is not fixed
    assert model.i_weight_raw.requires_grad is (support == "matched")
    x = torch.linspace(0.1, 2.0, 48).reshape(3, 16)
    for hard in (False, True):
        model.zero_grad(set_to_none=True)
        model(x, hard=hard).square().mean().backward()
        trainable = {name for name, p in model.named_parameters() if p.requires_grad}
        updated = {name for name, p in model.named_parameters() if p.grad is not None}
        assert trainable == updated
    restored = build_routed_ei(RoutedEIConfig(**asdict(model.config))).double()
    restored.load_state_dict(model.state_dict(), strict=True)
    assert [p.requires_grad for p in restored.parameters()] == [
        p.requires_grad for p in model.parameters()
    ]


@pytest.mark.parametrize("fixed,support", [(True, "matched"), (False, "global_mean")])
def test_distributed_data_parallel_trains_controls_with_unused_parameters(
    tmp_path, fixed, support
):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group'}", rank=0, world_size=1
    )
    try:
        model = DistributedDataParallel(
            build_routed_ei(spec(support=support, fixed=fixed))
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.linspace(0.1, 2.0, 48).reshape(3, 16)
        for _ in range(3):  # A parameter without a gradient raises on step two.
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("hard", [False, True])
@pytest.mark.parametrize(
    "mechanism,support",
    [("shunting", "matched"), ("shunting", "global_mean"), ("additive", "matched")],
)
def test_drives_are_the_currents_the_forward_integrates(
    fixed, hard, mechanism, support
):
    config = spec(mechanism, support, fixed)
    model = build_routed_ei(config)
    e = torch.linspace(0.1, 8.0, 40).reshape(5, 8)
    i = e.flip(0) * 0.7
    e_drive, i_drive = model.drives(e, i, hard=hard)
    response = model.branch_response(e, i, hard=hard)
    assert torch.equal(e_drive, response["e_drive"])
    assert torch.equal(i_drive, response["i_drive"])
    if support == "global_mean":
        assert torch.equal(i_drive, i.mean(dim=1, keepdim=True))
    # The forward's unchecked arithmetic equals the checked public functions.
    voltage = (
        additive_voltage(e_drive, i_drive)
        if mechanism == "additive"
        else shunting_voltage(
            e_drive, i_drive, leak=config.leak_conductance, epsilon=config.epsilon
        )
    )
    assert torch.equal(
        response["preactivation"],
        config.branch_scale * (voltage - config.branch_center) / config.branch_divisor,
    )


@pytest.mark.parametrize("router", ["paired", "pathway"])
def test_regularization_manager_uses_shared_observables(router):
    model = (
        build_routed_ei(spec())
        if router == "paired"
        else PathwayRouter(8, router_mode="learned")
    )
    config = RegularizationConfig(
        routing_entropy_weight=0.002, routing_balance_weight=0.25
    )
    manager = RegularizationManager(config)
    assert manager.has_regularization()
    entropy, balance = model.routing_statistics()
    assert torch.equal(
        manager.compute_regularization_loss(model), 0.002 * entropy + 0.25 * balance
    )
    manager.compute_regularization_loss(model).backward()
    assert any(p.grad is not None for p in model.parameters())
    with pytest.raises(ValueError, match="routing_statistics"):
        manager.compute_regularization_loss(nn.Linear(2, 2))
    if router == "paired":
        assert torch.equal(
            model.routing_regularizer(entropy_weight=0.002, balance_weight=0.25),
            0.002 * entropy + 0.25 * balance,
        )
        with pytest.raises(TypeError):
            model.routing_regularizer()  # The weights are the caller's choice.


@pytest.mark.parametrize("router", ["paired", "pathway"])
def test_routing_penalty_requires_a_learned_router(router):
    model = (
        build_routed_ei(spec(fixed=True))
        if router == "paired"
        else PathwayRouter(8, router_mode="fixed")
    )
    assert model.routing_statistics() is None
    manager = RegularizationManager(RegularizationConfig(routing_entropy_weight=0.002))
    with pytest.raises(ValueError, match="learned router"):
        manager.compute_regularization_loss(model)
    if router == "paired":
        zero = model.routing_regularizer(entropy_weight=0.002, balance_weight=0.25)
        assert zero.item() == 0 and not zero.requires_grad


def test_default_regularization_never_calls_routing_observables():
    class NoRouting(nn.Linear):
        def routing_statistics(self):
            raise AssertionError("disabled routing must not be evaluated")

    manager = RegularizationManager(RegularizationConfig())
    assert not manager.has_regularization()
    assert manager.compute_regularization_loss(NoRouting(2, 2)).item() == 0


@pytest.mark.parametrize(
    "x",
    [
        torch.ones(2, 7),
        -torch.ones(2, 16),
        torch.full((2, 16), float("nan")),
        torch.ones(2, 16, dtype=torch.float64),
    ],
)
def test_input_contract(x):
    with pytest.raises(ValueError):
        build_routed_ei(spec())(x)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mechanism", ["shunting", "additive"])
@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("hard", [False, True])
def test_optional_gate_equations_training_diagnostics_and_state(
    dtype, mechanism, trainable, hard
):
    config = replace(
        spec(mechanism), branch_activation="tanh", trainable_activation=trainable
    )
    torch.manual_seed(128)
    expected = ReferenceReadout(config).to(dtype)
    rng = torch.get_rng_state().clone()
    torch.manual_seed(128)
    actual = build_routed_ei(config).to(dtype)
    assert torch.equal(rng, torch.get_rng_state())
    assert list(expected.state_dict()) == list(actual.state_dict())
    assert list(dict(expected.named_parameters())) == list(
        dict(actual.named_parameters())
    )
    inputs = torch.linspace(0.2, 7.0, 40, dtype=dtype).reshape(5, 8)
    histories = []
    for model in (expected, actual):
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
        history = []
        for _ in range(4):
            optimizer.zero_grad()
            e, i = (x.clone().requires_grad_() for x in (inputs, inputs.flip(0)))
            output = model(e, i, hard=hard)
            output.square().mean().backward()
            history.extend([output.detach().clone(), e.grad.clone(), i.grad.clone()])
            history.extend(
                p.grad.clone() for p in model.parameters() if p.grad is not None
            )
            optimizer.step()
        histories.append(history)
    for a, b in zip(*histories, strict=True):
        assert torch.equal(a, b)
    for key, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key])
    response = actual.branch_response(inputs, inputs.flip(0), hard=hard)
    assert torch.equal(
        actual(inputs, inputs.flip(0), hard=hard),
        actual.decoder(response["activation"]),
    )
    assert torch.equal(
        actual.branch_values(inputs, inputs.flip(0), hard=hard), response["activation"]
    )
    assert torch.equal(
        response["path_slope"],
        F.softplus(actual.activation_slope_raw) * (1 - response["activation"].square()),
    )
    restored = build_routed_ei(RoutedEIConfig(**asdict(actual.config))).to(dtype)
    restored.load_state_dict(actual.state_dict(), strict=True)
    assert torch.equal(
        actual(inputs, inputs, hard=hard), restored(inputs, inputs, hard=hard)
    )


@pytest.mark.parametrize("trainable", [False, True])
def test_gate_config_reaches_all_routed_builders(trainable):
    value = mapping()
    value["model"]["core"]["routed_ei"].update(
        branch_activation="tanh",
        trainable_activation=trainable,
        activation_offset=-0.25,
    )
    cfg = config_from_mapping(value)
    models = [
        build_routed_ei(cfg.model.core.routed_ei),
        get_architecture("routed_ei", cfg.model.core, input_dim=16, suffix_input_dim=2),
        initialize_model(cfg.model)[0].core_network,
    ]
    for model in models:
        assert model.config.trainable_activation is trainable
        assert isinstance(model.activation_slope_raw, nn.Parameter) is trainable
        assert torch.equal(model.activation_offset, torch.full((3,), -0.25))


def test_gate_default_registers_no_state_and_does_not_evaluate_diagnostics():
    raw = asdict(spec())
    for key in (
        "branch_activation",
        "trainable_activation",
        "activation_slope_raw",
        "activation_offset",
    ):
        raw.pop(key)
    torch.manual_seed(84)
    omitted = build_routed_ei(RoutedEIConfig(**raw))
    rng = torch.get_rng_state().clone()
    torch.manual_seed(84)
    explicit = build_routed_ei(spec())
    assert torch.equal(rng, torch.get_rng_state())
    assert all("activation" not in name for name in explicit.state_dict())
    assert list(omitted.state_dict()) == list(explicit.state_dict())
    explicit.branch_response = lambda *args, **kwargs: pytest.fail(
        "forward evaluated diagnostics"
    )
    x = torch.ones(3, 16)
    assert torch.equal(omitted(x), explicit(x))


def test_optional_gate_input_and_parameter_derivatives():
    model = build_routed_ei(
        replace(spec(), branch_activation="tanh", trainable_activation=True)
    ).double()
    e = torch.linspace(0.3, 2.0, 16, dtype=torch.float64).reshape(2, 8).requires_grad_()
    i = (e.detach() * 0.4).requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: model(a, b, hard=False), (e, i))
    assert torch.autograd.gradcheck(
        lambda a, slope, offset: torch.func.functional_call(
            model,
            {"activation_slope_raw": slope, "activation_offset": offset},
            (a, i),
            {"hard": False},
        ),
        (e, model.activation_slope_raw, model.activation_offset),
    )


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("mechanism", ["shunting", "additive"])
def test_soft_forward_has_exact_input_derivative(fixed, mechanism):
    model = build_routed_ei(spec(mechanism=mechanism, fixed=fixed)).double()
    e = torch.linspace(0.3, 2.0, 16, dtype=torch.float64).reshape(2, 8).requires_grad_()
    i = (e.detach() * 0.4).requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: model(a, b, hard=False), (e, i))


def test_default_row_sum_keeps_empty_branches_at_zero_and_round_trips(tmp_path):
    config = RoutedEIConfig(features=4, branches=3, output_dim=2)
    model = build_routed_ei(config)
    with torch.no_grad():
        model.route_logits[0].fill_(4)
        model.route_logits[1:].zero_()
    x = torch.ones(3, 8)
    for drive in model.drives(x):
        assert torch.equal(drive[:, 1:], torch.zeros(3, 2))
        torch.testing.assert_close(drive[:, 0], torch.ones(3))  # Unit weight sum.
    model(x).square().sum().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    path = tmp_path / "model.pt"
    torch.save({"config": asdict(model.config), "state": model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = build_routed_ei(RoutedEIConfig(**saved["config"]))
    restored.load_state_dict(saved["state"], strict=True)
    assert torch.equal(model(x), restored(x))


def _train_one_epoch(run_path, regularization):
    from torch.utils.data import TensorDataset

    from dendritic_modeling.scripts.training.train_experiments import (
        _build_main_trainer,
    )

    value = mapping()
    value["training"]["main"]["common"] = {
        "epochs": 1,
        "batch_size": 4,
        "loss_function": "ce",
        "plot_losses": False,
        "early_stopping": False,
        "synapse_decay_rate": 0.0,
        "weight_boosting": False,
    }
    value["training"]["main"]["regularization"] = regularization
    cfg = config_from_mapping(value)
    torch.manual_seed(61)
    model, _ = initialize_model(cfg.model)
    inputs = torch.linspace(0.1, 2.0, 128).reshape(8, 16)
    dataset = TensorDataset(inputs, torch.arange(8) % 2)
    run_path.mkdir()
    setup = _build_main_trainer(
        unwrapped_model=model,
        training_config=cfg.training,
        model_config=cfg.model,
        experiment_config=cfg.experiment,
        run_save_path=str(run_path),
        encoded_train_ds=dataset,
        analysis_manager=None,
    )
    torch.manual_seed(62)
    results = setup.trainer.train(model, dataset, dataset)
    assert len(results["train_losses"]) == 1
    manager = setup.trainer.regularization_manager
    return manager, model.core_network.route_logits.detach().cpu()


def test_standard_training_adds_the_configured_routing_penalty(tmp_path):
    penalty = {"routing_entropy_weight": 0.002, "routing_balance_weight": 0.25}
    manager, penalized = _train_one_epoch(tmp_path / "penalized", penalty)
    assert manager.routing_balance_weight == 0.25
    _, plain = _train_one_epoch(tmp_path / "plain", {})
    _, repeated = _train_one_epoch(tmp_path / "repeated", {})
    # Training repeats exactly, so only the penalty changes the learned routing.
    assert torch.equal(plain, repeated)
    assert not torch.equal(plain, penalized)


@pytest.mark.parametrize("value", [-1, float("nan"), True])
def test_invalid_routing_regularization_is_rejected(value):
    with pytest.raises(ValueError):
        RegularizationConfig(routing_entropy_weight=value)


def test_example_reuses_existing_dataset_loader():
    from dendritic_modeling.datasets import get_unified_datasets

    path = Path(__file__).resolve().parents[1] / "configs/examples/routed_ei.yaml"
    cfg = load_config(str(path))
    params = cfg.data.dataset_params[cfg.data.dataset_name]
    train, valid, test = get_unified_datasets(
        {
            "dataset": cfg.data.dataset_name,
            "train_valid_split": cfg.experiment.train_valid_split,
            "parameters": {**params, "seed": cfg.experiment.seed},
        }
    )
    assert params["valid_split_mode"] == "train"
    model, _ = initialize_model(cfg.model)
    for dataset in (train, valid, test):
        x, labels = dataset.tensors
        assert x.shape[1] == 2 * cfg.model.core.routed_ei.features
        assert x.min() >= 0 and labels.max() < cfg.model.core.routed_ei.output_dim
        logits = model(x[:4])
        assert logits.shape == (4, 2) and torch.isfinite(logits).all()


def test_training_builds_the_control_from_the_global_stream():
    from dendritic_modeling.scripts.script_utils.setup_utils import (
        _apply_initialization_seed_defaults,
    )

    cfg = config_from_mapping(mapping())
    _apply_initialization_seed_defaults(cfg.model, 41)
    assert cfg.model.core.initialization_seed is None
    torch.manual_seed(41)  # As training's set_seed(model_seed).
    trained = initialize_model(cfg.model)[0].core_network
    torch.manual_seed(41)
    direct = build_routed_ei(spec())
    assert all(
        torch.equal(v, direct.state_dict()[k]) for k, v in trained.state_dict().items()
    )


def test_training_filled_seeds_do_not_warn_about_unused_sections(caplog):
    from dendritic_modeling.scripts.script_utils.setup_utils import (
        _apply_topology_seed_defaults,
    )

    cfg = config_from_mapping(mapping())
    _apply_topology_seed_defaults(cfg.model, 17)
    initialize_model(cfg.model)
    assert "Ignoring sections" not in caplog.text
    direct = get_architecture(
        "ROUTED_EI", {"type": "ROUTED_EI", "routed_ei": asdict(spec())}
    )
    assert direct.input_dim == 16
