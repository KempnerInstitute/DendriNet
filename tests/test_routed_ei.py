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
from dendritic_modeling.networks.architectures.classical.pathway_router import (
    PathwayRouter,
)
from dendritic_modeling.networks.architectures.excitation_inhibition.routed_ei import (
    build_routed_ei,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.utils.routing import routing_assignments
from dendritic_modeling.networks.utils.weight_normalization import normalize_weight_rows
from dendritic_modeling.scripts.script_utils.setup_utils import initialize_model
from dendritic_modeling.training.regularization import RegularizationManager


def spec(mechanism="shunting", support="matched", fixed=False):
    return RoutedEIConfig(
        features=8,
        branches=3,
        output_dim=2,
        routing="fixed" if fixed else "learned",
        fixed_assignment=[0, 1, 2, 0, 1, 2, 0, 1] if fixed else [],
        mechanism=mechanism,
        inhibitory_support=support,
        temperature=0.8,
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


def test_explicit_initialization_seed_preserves_callers_stream():
    raw = {"routed_ei": asdict(spec()), "initialization_seed": 4}
    rng = torch.get_rng_state().clone()
    a = get_architecture("routed_ei", raw)
    b = get_architecture("routed_ei", raw)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


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


def test_robust_default_empty_branch_and_config_checkpoint_roundtrip(tmp_path):
    config = RoutedEIConfig(features=4, branches=3, output_dim=2)
    model = build_routed_ei(config)
    with torch.no_grad():
        model.route_logits[0].fill_(4)
        model.route_logits[1:].zero_()
    x = torch.ones(3, 8)
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


def test_standard_training_uses_configured_routing_penalty(tmp_path):
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
    value["training"]["main"]["regularization"] = {
        "routing_entropy_weight": 0.002,
        "routing_balance_weight": 0.25,
    }
    cfg = config_from_mapping(value)
    model, _ = initialize_model(cfg.model)
    inputs = torch.linspace(0.1, 2.0, 128).reshape(8, 16)
    dataset = TensorDataset(inputs, torch.arange(8) % 2)
    setup = _build_main_trainer(
        unwrapped_model=model,
        training_config=cfg.training,
        model_config=cfg.model,
        experiment_config=cfg.experiment,
        run_save_path=str(tmp_path),
        encoded_train_ds=dataset,
        analysis_manager=None,
    )
    assert setup.trainer.regularization_manager.routing_balance_weight == 0.25
    before = model.core_network.route_logits.detach().clone()
    results = setup.trainer.train(model, dataset, dataset)
    assert len(results["train_losses"]) == 1
    assert not torch.equal(before, model.core_network.route_logits)


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
