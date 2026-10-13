"""Activation calibration, gradient hooks and state across module lifecycles."""

from __future__ import annotations

import copy
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from dendritic_modeling.config import ParamGroupsConfig
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.activations import memory_efficient
from dendritic_modeling.networks.activations.endpoint_tanh import EndpointNormalizedTanh
from dendritic_modeling.networks.activations.parametric import (
    ParametricActivation,
    ParametricTanh,
    ParametricTanhOnlyM,
)
from dendritic_modeling.networks.architectures.classical import pretrained
from dendritic_modeling.networks.architectures.excitation_inhibition.dendritic.initialize_reactivation import (
    calibrate_reactivation_from_data,
)
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.training.strategies.standard import Trainer


def _einet_config(*, reactivation=None, sparsity=None, blocklinear=None):
    """Tiny shunting EINet with explicit inhibitory dendritic cells."""
    config = {
        "architecture": {
            "excitatory_layer_sizes": [6],
            "inhibitory_layer_sizes": [3],
            "excitatory_branch_factors": [2, 2],
            "inhibitory_branch_factors": [2],
            "inhibitory_network_type": "dendritic",
            "mlp_inhibitory_network_params": {"hidden_dims": [], "activation": "relu"},
        },
        "connectivity": {
            "ee_synapses_per_branch_per_layer": [4],
            "ei_synapses_per_branch_per_layer": [3],
            "ie_synapses_per_branch_per_layer": [2],
            "ii_synapses_per_branch_per_layer": [0],
        },
        "transfer": {
            "independent_pathways": False,
            "excitatory_dim": None,
            "inhibitory_dim": None,
            "output_activation": None,
        },
        "morphology": {
            "somatic_synapses": False,
            "use_shunting": True,
            "weight_transform": "softplus",
            "dbl_init_method": "analytical_expectation",
        },
        "sparsity": {
            "init_method": "xavier_normal",
            "noise_level": 0.0,
            "type": "standard",
            "gradient_scaling": "none",
        },
        "reactivation": {
            "enabled": True,
            "type": "param_tanh",
            "init_m": 1.5,
            "init_b": 0.5,
            "gradient_scaling": "none",
        },
        "blocklinear": {"gradient_scaling": "none", "efficient": False},
        "implementation": {"adaptive_initialization": False, "print_hooks": False},
    }
    config["reactivation"].update(reactivation or {})
    config["sparsity"].update(sparsity or {})
    config["blocklinear"].update(blocklinear or {})
    return config


def _build_einet(config, *, seed=0, input_dim=12):
    torch.manual_seed(seed)
    return get_architecture(
        "dendritic_shunting", copy.deepcopy(config), input_dim=input_dim
    )


def _inputs(seed, batch=8, dim=12, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, dim, generator=generator) * scale


# ---------------------------------------------------------------------------
# Item 5: linear-tanh/sigmoid reactivations build through EINet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "act_type",
    ["param_linear_tanh", "param_linear_sigmoid", "linear_tanh", "linear_sigmoid"],
)
def test_linear_blend_reactivations_build_through_einet(act_type):
    model = _build_einet(_einet_config(reactivation={"type": act_type}))
    gates = [
        module.reactivation
        for module in model.modules()
        if hasattr(module, "reactivation") and hasattr(module.reactivation, "log_m")
    ]
    assert gates, "expected parametric reactivation gates"
    output = model(_inputs(1))
    assert torch.isfinite(output).all()
    output.sum().backward()
    trainable = act_type.startswith("param_")
    assert all(gate.log_m.requires_grad is trainable for gate in gates)


# ---------------------------------------------------------------------------
# Item 1: calibrated fixed_b survives state_dict round trips
# ---------------------------------------------------------------------------


def _fixed_b_values(model):
    return [
        module.fixed_b
        for module in model.modules()
        if isinstance(module, ParametricTanhOnlyM)
    ]


def test_calibrated_fixed_b_survives_checkpoint_reload_and_best_state_restore():
    config = _einet_config(reactivation={"type": "param_tanh_only_m"})
    model = _build_einet(config)
    model.eval()
    constructed = _fixed_b_values(model)
    calibrate_reactivation_from_data(
        model, [_inputs(seed, batch=16, scale=3.0) for seed in range(3)]
    )
    calibrated = _fixed_b_values(model)
    assert calibrated != constructed
    x = _inputs(10)
    with torch.no_grad():
        y_calibrated = model(x)

    checkpoint = copy.deepcopy(model.state_dict())
    fresh = _build_einet(config)
    fresh.load_state_dict(checkpoint, strict=True)
    fresh.eval()
    assert _fixed_b_values(fresh) == pytest.approx(calibrated, abs=1e-6)
    with torch.no_grad():
        assert torch.equal(fresh(x), y_calibrated)

    # In-process best-state restore after a later recalibration.
    calibrate_reactivation_from_data(
        model, [_inputs(seed, batch=16, scale=0.5) for seed in range(20, 23)]
    )
    with torch.no_grad():
        assert not torch.equal(model(x), y_calibrated)
    model.load_state_dict(checkpoint)
    with torch.no_grad():
        assert torch.equal(model(x), y_calibrated)


def test_legacy_checkpoint_without_fixed_b_buffer_keeps_constructed_fixed_b():
    activation = ParametricTanhOnlyM(output_dim=3, init_m=1.2, fixed_b=0.4)
    legacy = activation.state_dict()
    legacy.pop("fixed_b_buffer")
    legacy._metadata[""]["version"] = 1  # written before the buffer existed

    reloaded = ParametricTanhOnlyM(output_dim=3, init_m=1.0, fixed_b=0.3)
    with pytest.warns(UserWarning, match="cannot be recovered"):
        reloaded.load_state_dict(legacy, strict=True)
    assert reloaded.fixed_b == 0.3
    assert float(reloaded.fixed_b_buffer) == pytest.approx(0.3)
    torch.testing.assert_close(reloaded.log_m, activation.log_m)

    current = activation.state_dict()
    current.pop("fixed_b_buffer")
    with pytest.raises(RuntimeError, match="fixed_b_buffer"):
        reloaded.load_state_dict(current, strict=True)


# ---------------------------------------------------------------------------
# Item 3: the fused shunt path never bypasses reactivation hooks
# ---------------------------------------------------------------------------


def _count_fused_shunt_calls(monkeypatch):
    calls = []
    original = memory_efficient._MemoryEfficientShuntPositiveTanh.apply

    def counting_apply(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(
        memory_efficient._MemoryEfficientShuntPositiveTanh, "apply", counting_apply
    )
    return calls


def _train_mode_gradients(model, x, *, diagnostics=False):
    for module in model.modules():
        if hasattr(module, "set_branch_diagnostics"):
            module.set_branch_diagnostics(diagnostics)
    model.train()
    model.zero_grad()
    output = model(x)
    (output.square().mean() + 0.1 * output.sum()).backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), gradients


def test_reactivation_gradient_scaling_runs_unfused_and_applies_its_hook(monkeypatch):
    calls = _count_fused_shunt_calls(monkeypatch)
    hooked = _build_einet(
        _einet_config(
            reactivation={
                "memory_efficient": True,
                "gradient_scaling": "straight_through",
            }
        )
    )
    default = _build_einet(_einet_config(reactivation={"memory_efficient": True}))
    for (name, a), (_, b) in zip(
        hooked.state_dict().items(), default.state_dict().items()
    ):
        assert torch.equal(a, b), name
    x = _inputs(3)

    y_default, g_default = _train_mode_gradients(default, x)
    assert calls, "the default memory-efficient shunt must stay on the fused path"
    calls.clear()
    y_hooked, g_hooked = _train_mode_gradients(hooked, x)
    assert not calls, "a reactivation gradient-scaling hook needs the unfused path"
    assert torch.equal(y_hooked, y_default)
    assert any(not torch.equal(g_hooked[name], g_default[name]) for name in g_default)

    _, g_diagnostics = _train_mode_gradients(hooked, x, diagnostics=True)
    assert g_diagnostics.keys() == g_hooked.keys()
    for name, gradient in g_hooked.items():
        assert torch.equal(g_diagnostics[name], gradient), name


def test_calibration_reaches_memory_efficient_shunting_layers():
    diagnostics = {}
    for memory_efficient_gate in (False, True):
        model = _build_einet(
            _einet_config(reactivation={"memory_efficient": memory_efficient_gate})
        )
        diagnostics[memory_efficient_gate] = calibrate_reactivation_from_data(
            model, [_inputs(seed, batch=16, scale=3.0) for seed in range(2)]
        )
    assert len(diagnostics[True]) == len(diagnostics[False]) > 0
    for name, eager in diagnostics[False].items():
        fused_capable = diagnostics[True][name]
        assert fused_capable["b"] == pytest.approx(eager["b"], rel=1e-5, abs=1e-6)
        assert fused_capable["m"] == pytest.approx(eager["m"], rel=1e-4)


def test_any_hook_on_a_memory_efficient_gate_disables_shunt_fusion():
    gate = ParametricTanh(output_dim=3, memory_efficient=True)
    assert memory_efficient.fused_shunt_reactivation(gate) is not None
    for register in (
        lambda: gate.register_forward_pre_hook(lambda module, inputs: None),
        lambda: gate.register_forward_hook(lambda module, inputs, output: None),
        lambda: gate.register_full_backward_hook(lambda module, gi, go: None),
    ):
        handle = register()
        assert memory_efficient.fused_shunt_reactivation(gate) is None
        handle.remove()
        assert memory_efficient.fused_shunt_reactivation(gate) is not None


# ---------------------------------------------------------------------------
# Item 2: gradient-scaling hooks do not rescale eval-mode gradients
# ---------------------------------------------------------------------------


def _eval_gradients(model, x):
    model.eval()
    model.zero_grad()
    x = x.clone().requires_grad_(True)
    model(x).sum().backward()
    gradients = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return x.grad.clone(), gradients


@pytest.mark.parametrize(
    "scaling",
    [
        {
            "sparsity": {"gradient_scaling": "conductance_dynamic"},
            "blocklinear": {"gradient_scaling": "conductance_dynamic"},
        },
        {"blocklinear": {"gradient_scaling": "distal_upweight_by_idx"}},
        {"reactivation": {"gradient_scaling": "straight_through"}},
    ],
    ids=["conductance_dynamic", "distal_upweight_by_idx", "reactivation_inverse"],
)
def test_gradient_scaling_hooks_leave_eval_gradients_unscaled(scaling):
    scaled = _build_einet(_einet_config(**scaling))
    reference = _build_einet(_einet_config(), seed=123)
    optimizer = torch.optim.Adam(scaled.parameters(), lr=0.05)
    for step in range(3):
        optimizer.zero_grad()
        _train_mode_gradients(scaled, _inputs(40 + step))
        optimizer.step()
    reference.load_state_dict(scaled.state_dict())

    # Training mode still applies the configured scaling.
    _, scaled_train = _train_mode_gradients(scaled, _inputs(50))
    _, reference_train = _train_mode_gradients(reference, _inputs(50))
    assert any(
        not torch.allclose(scaled_train[name], reference_train[name])
        for name in reference_train
    )

    x = _inputs(9, batch=4)
    scaled_input_grad, scaled_grads = _eval_gradients(scaled, x)
    reference_input_grad, reference_grads = _eval_gradients(reference, x)
    assert torch.equal(scaled_input_grad, reference_input_grad)
    assert scaled_grads.keys() == reference_grads.keys()
    for name, gradient in reference_grads.items():
        assert torch.equal(scaled_grads[name], gradient), name


# ---------------------------------------------------------------------------
# Item 4: building a backbone split leaves BatchNorm statistics untouched
# ---------------------------------------------------------------------------


class _TinyBatchNormBackbone(torch.nn.Module):
    """Sequential-style torchvision stand-in with BatchNorm in the prefix."""

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(4),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(16, 8),
            torch.nn.BatchNorm1d(8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 3),
        )
        with torch.no_grad():
            for norm in (self.features[1], self.classifier[1]):
                norm.running_mean.uniform_(-1.0, 1.0)
                norm.running_var.uniform_(0.5, 2.0)


def _batchnorm_state(modules):
    return [
        (
            norm.running_mean.clone(),
            norm.running_var.clone(),
            int(norm.num_batches_tracked),
        )
        for module in modules
        for norm in module.modules()
        if isinstance(norm, torch.nn.modules.batchnorm._BatchNorm)
    ]


def test_backbone_split_build_keeps_batchnorm_statistics_and_frozen_segments_eval(
    monkeypatch,
):
    torch.manual_seed(0)
    backbone = _TinyBatchNormBackbone()
    before = _batchnorm_state([backbone])
    monkeypatch.setattr(pretrained, "load_backbone", lambda *args, **kwargs: backbone)

    plan = pretrained.build_split_plan(
        "tiny", target_modules=["classifier.0"], input_shape=[3, 6, 6]
    )
    encoder, decoder = pretrained.build_encoder_decoder_from_plan(
        plan, prefix_freeze=True, suffix_freeze=False
    )
    after = _batchnorm_state([backbone])
    for (mean0, var0, n0), (mean1, var1, n1) in zip(before, after):
        assert torch.equal(mean0, mean1) and torch.equal(var0, var1) and n0 == n1

    # Frozen segments stay in eval mode; trainable segments follow train().
    for segment in (encoder, decoder):
        segment.train()
    assert not any(module.training for module in encoder.layers.modules())
    assert all(module.training for module in decoder.layers.modules())
    encoder(torch.rand(4, 3, 6, 6))
    for (mean0, var0, n0), (mean1, var1, n1) in zip(
        before[:1], _batchnorm_state([encoder])
    ):
        assert torch.equal(mean0, mean1) and torch.equal(var0, var1) and n0 == n1


# ---------------------------------------------------------------------------
# Finding A: endpoint_tanh gates are reactivation modules and reject calibration
# ---------------------------------------------------------------------------


def _endpoint_classifier():
    defaults = {
        "use_shunting": True,
        "ff_excitatory_synapses": 2,
        "ff_inhibitory_synapses": 0,
        "rec_excitatory_synapses": 0,
        "rec_inhibitory_synapses": 0,
        "reactivate": True,
        "reactivation_type": "endpoint_tanh",
        "reactivation_init_policy": "fixed",
        "somatic_synapses": False,
        "adaptive_initialization": False,
        "dbl_init_method": "default",
    }
    config = {
        "population_network": {
            "input_transform": "identity",
            "layers": [
                {
                    "name": "layer0",
                    "readout_population": "e",
                    "population_defaults": defaults,
                    "populations": [
                        {
                            "name": "e",
                            "polarity": "excitatory",
                            "n_neurons": 3,
                            "branch_factors": [2, 2],
                        }
                    ],
                    "connections": [{"source": "input", "target": "e"}],
                }
            ],
        }
    }
    torch.manual_seed(3)
    core = get_architecture("population_network", config, input_dim=6)
    return Classifier(Identity(6), core, torch.nn.Linear(core.output_dim, 2))


def test_endpoint_tanh_gates_join_reactivation_group_and_reject_calibration():
    model = _endpoint_classifier()
    gates = [
        module
        for module in model.modules()
        if isinstance(module, EndpointNormalizedTanh)
    ]
    assert gates
    groups = model.get_param_groups(
        ParamGroupsConfig(lr=0.01, split_params=True, reactivation_lr=1e-4)
    )
    group_of = {
        id(parameter): group["name"]
        for group in groups
        for parameter in group["params"]
    }
    reactivation_lr = {group["name"]: group["lr"] for group in groups}["reactivation"]
    assert reactivation_lr == 1e-4
    for gate in gates:
        assert group_of[id(gate.log_m)] == group_of[id(gate.b)] == "reactivation"

    trainer = Trainer(torch.optim.SGD(model.parameters(), lr=0.1))
    trainer.reactivation_update_mode = "frozen"
    trainer._configure_reactivation_update_mode(model)
    assert not any(p.requires_grad for gate in gates for p in gate.parameters())

    offsets = [gate.b.detach().clone() for gate in gates]
    batches = [torch.rand(16, 6) for _ in range(2)]
    assert calibrate_reactivation_from_data(model, batches, mode=None) == {}
    with pytest.raises(ValueError, match="requires fixed reactivation initialization"):
        calibrate_reactivation_from_data(model, batches, mode="median_mad")
    for gate, offset in zip(gates, offsets):
        assert torch.equal(gate.b, offset)


# ---------------------------------------------------------------------------
# Finding B: DDP ranks share rank 0's recalibrated reactivation state
# ---------------------------------------------------------------------------


def _reactivation_state_vector(model):
    values = []
    for module in model.modules():
        if isinstance(module, ParametricActivation):
            values.extend(t.detach().reshape(-1) for t in module.state_dict().values())
            if isinstance(module, ParametricTanhOnlyM):
                values.append(torch.tensor([module.fixed_b]))
    return torch.cat([value.double() for value in values])


def _ddp_recalibration_worker(rank, world_size, rendezvous_path, result_path):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.set_num_threads(1)
        model = _build_einet(_einet_config(reactivation={"type": "param_tanh_only_m"}))
        before = _reactivation_state_vector(model)
        batches = [_inputs(100 * rank + i, batch=16, scale=3.0) for i in range(2)]
        local = copy.deepcopy(model)
        calibrate_reactivation_from_data(local, batches, mode="median_mad")

        trainer = Trainer(torch.optim.SGD(model.parameters(), lr=0.1))
        trainer.device = "cpu"
        trainer.epoch_counter = 1
        trainer.logger_info_prefix = ""
        trainer.reactivation_recalibration_mode = "median_mad"
        trainer.reactivation_recalibration_num_batches = len(batches)
        trainer._recalibrate_reactivation(DistributedDataParallel(model), batches)

        rows = torch.stack(
            [
                before,
                _reactivation_state_vector(local),
                _reactivation_state_vector(model),
            ]
        )
        gathered = [torch.empty_like(rows) for _ in range(world_size)]
        dist.all_gather(gathered, rows)
        if rank == 0:
            torch.save(torch.stack(gathered), result_path)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.multiprocess
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="CPU gloo distributed backend is unavailable",
)
def test_ddp_recalibration_broadcasts_rank0_reactivation_state(tmp_path):
    world_size = 2
    result_path = tmp_path / "ddp_recalibration.pt"
    mp.spawn(
        _ddp_recalibration_worker,
        args=(world_size, str(tmp_path / "gloo_rendezvous"), str(result_path)),
        nprocs=world_size,
        join=True,
    )
    rows = torch.load(result_path, weights_only=True)  # [rank, (before, local, ddp)]
    before, local, synced = rows[:, 0], rows[:, 1], rows[:, 2]
    assert torch.equal(before[0], before[1])
    assert not torch.equal(local[0], local[1]), "ranks must see different shards"
    assert not torch.equal(synced[0], before[0])
    assert torch.equal(synced[0], local[0])
    assert torch.equal(synced[1], synced[0])
