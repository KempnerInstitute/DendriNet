"""FSDP trains every parameter at the learning rate of single-process training."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from test_biological_constraints import _einet
from test_trainer_configuration import IN_DIM, N_CLASSES
from torch import nn

from dendritic_modeling.config import load_config
from dendritic_modeling.config.training import OptimizerConfig, ParamGroupsConfig
from dendritic_modeling.models import Classifier
from dendritic_modeling.networks import Identity, ParametricActivation, TopKLinear
from dendritic_modeling.networks.activations import ParametricTanh
from dendritic_modeling.networks.activations.endpoint_tanh import EndpointNormalizedTanh
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.scripts.script_utils.config_utils import (
    normalize_param_groups_config,
)
from dendritic_modeling.scripts.training import train_experiments_fsdp as fsdp_script
from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer

RATES = ParamGroupsConfig(
    lr=0.01,
    topk_lr=0.002,
    blocklinear_lr=0.003,
    reactivation_lr=0.004,
    decoder_lr=0.005,
    decoder_input_lr=0.006,
    encoder_lr=0.007,
)
GATE_TYPES = (
    "param_tanh",
    "param_tanh_only_m",
    "param_linear_sigmoid",
    "param_linear_tanh",
    "param_linear_tanh_transition",
    "param_relu",
    "residual_silu",
    "endpoint_tanh",
)
FSDP_PREFIX = "_fsdp_wrapped_module."
REPO = Path(__file__).resolve().parents[1]


def _core(gate="param_tanh"):
    changes = {"reactivation.type": gate}
    if gate == "endpoint_tanh":
        changes["reactivation.init_policy"] = "fixed"
    torch.manual_seed(0)
    return get_architecture("einet", _einet(**changes), input_dim=IN_DIM)


def _model(gate="param_tanh", *, decoder=None, encoder=None, **classifier):
    core = _core(gate)
    if decoder is None:
        decoder = nn.Linear(core.output_dim, N_CLASSES)
    classifier.setdefault("output_scale_mode", "fixed")
    return Classifier(encoder or Identity(IN_DIM), core, decoder, **classifier)


def _rates(model, groups):
    names = {
        id(p): name.replace(FSDP_PREFIX, "") for name, p in model.named_parameters()
    }
    return {
        names[id(p)]: group["lr"]
        for group in groups
        for p in group["params"]
        if p.requires_grad
    }


def _fsdp_optimizer(model, param_groups=RATES, optimizer=None, rank=0):
    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.rank = rank
    trainer.optimizer_config = {
        "optimizer": optimizer or OptimizerConfig(),
        "param_groups": param_groups,
    }
    return trainer.create_optimizer(model)


def _gate_parameter_names(model):
    names = {id(p): name for name, p in model.named_parameters()}
    return [
        names[id(p)]
        for module in model.core_network.modules()
        if isinstance(module, (ParametricActivation, EndpointNormalizedTanh))
        for p in module.parameters()
    ]


@pytest.mark.parametrize("gate", GATE_TYPES)
def test_every_gate_family_trains_at_reactivation_lr(gate):
    model = _model(gate)
    rates = _rates(model, _fsdp_optimizer(model).param_groups)
    assert rates == _rates(model, model.get_param_groups(RATES))
    gate_names = _gate_parameter_names(model)
    assert gate_names
    assert {rates[name] for name in gate_names} == {RATES.reactivation_lr}


@pytest.mark.parametrize("mode", ["per_class", "global"])
def test_learned_output_scales_train_with_the_decoder(mode):
    core = _core()
    model = Classifier(
        Identity(IN_DIM), core, Identity(core.output_dim), output_scale_mode=mode
    )
    name = "log_output_scale" if mode == "per_class" else "log_global_output_scale"
    rates = _rates(model, _fsdp_optimizer(model).param_groups)
    assert rates[name] == RATES.decoder_lr
    assert rates == _rates(model, model.get_param_groups(RATES))


def test_sparse_and_gate_modules_outside_the_core_follow_their_component():
    torch.manual_seed(0)
    encoder = TopKLinear(IN_DIM, IN_DIM, K=2)
    core = _core()
    decoder = nn.Sequential(
        TopKLinear(core.output_dim, 4, K=2),
        ParametricTanh(4),
        nn.Linear(4, N_CLASSES),
    )
    model = Classifier(encoder, core, decoder, output_scale_mode="fixed")
    rates = _rates(model, _fsdp_optimizer(model).param_groups)
    assert rates == _rates(model, model.get_param_groups(RATES))
    assert rates["encoder_network.pre_w"] == RATES.encoder_lr
    assert rates["decoder_network.0.pre_w"] == RATES.decoder_lr
    assert rates["decoder_network.1.log_m"] == RATES.decoder_lr
    # decoder_input_lr applies to the decoder's first Linear weight only.
    assert rates["decoder_network.2.weight"] == RATES.decoder_input_lr
    assert rates["decoder_network.2.bias"] == RATES.decoder_lr


def test_frozen_parameters_and_empty_groups_are_left_out():
    core = _core()
    model = Classifier(
        Identity(IN_DIM), core, Identity(core.output_dim), output_scale_mode="fixed"
    )
    model.core_network.requires_grad_(False)
    with pytest.raises(ValueError, match="at least one trainable parameter"):
        _fsdp_optimizer(model)
    model.core_network.requires_grad_(True)
    for module in model.core_network.modules():
        if isinstance(module, ParametricActivation):
            module.requires_grad_(False)
    optimizer = _fsdp_optimizer(model)
    names = [group["name"] for group in optimizer.param_groups]
    assert "reactivation" not in names
    assert all(group["params"] for group in optimizer.param_groups)


def test_without_split_params_one_group_holds_the_trainable_parameters():
    model = _model()
    model.decoder_network.requires_grad_(False)
    (group,) = _fsdp_optimizer(
        model, ParamGroupsConfig(lr=0.02, split_params=False)
    ).param_groups
    assert group["name"] == "all"
    assert group["lr"] == 0.02
    assert {id(p) for p in group["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad
    }


def test_optimizer_config_takes_only_optimizer_and_param_groups():
    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.optimizer_config = {"lr": 0.01, "split_params": True}
    with pytest.raises(ValueError, match=r"unknown keys \['lr', 'split_params'\]"):
        trainer.create_optimizer(_model())


def test_the_reference_fsdp_config_keeps_its_groups():
    """config_exp_fsdp.yaml groups exactly as before the shared rule."""
    config = load_config(str(REPO / "configs" / "config_exp_fsdp.yaml"))
    torch.manual_seed(0)
    model = fsdp_script.create_large_model(config)
    optimizer = _fsdp_optimizer(
        model,
        normalize_param_groups_config(config.training.main.common.param_groups),
    )
    assert [
        (group["name"], group["lr"], len(group["params"]))
        for group in optimizer.param_groups
    ] == [
        ("topk", 0.001, 10),
        ("blocklinear", 0.0001, 4),
        ("reactivation", 0.0001, 12),
        ("decoder", 0.001, 4),
    ]


def _fsdp(model, sharding, use_orig_params):
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

    return FullyShardedDataParallel(
        model,
        sharding_strategy=getattr(ShardingStrategy, sharding),
        device_id=torch.device("cpu"),
        sync_module_states=False,
        use_orig_params=use_orig_params,
    )


def _wrapped_worker(rank, rendezvous):
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        # Sharded units expose 1-D views of the original parameters; the
        # groups still hold them at single-process training's rates.
        core = _core("residual_silu")
        model = Classifier(
            Identity(IN_DIM),
            core,
            Identity(core.output_dim),
            output_scale_mode="per_class",
        )
        expected = _rates(model, model.get_param_groups(RATES))
        wrapped = _fsdp(model, "FULL_SHARD", use_orig_params=True)
        assert _rates(wrapped, _fsdp_optimizer(wrapped, rank=rank).param_groups) == (
            expected
        )
        with pytest.raises(ValueError, match="muonh needs full 2-D"):
            _fsdp_optimizer(wrapped, optimizer=OptimizerConfig(name="muonh"))

        unsharded = _fsdp(_model(), "NO_SHARD", use_orig_params=True)
        muonh = _fsdp_optimizer(unsharded, optimizer=OptimizerConfig(name="muonh"))
        assert muonh.muon_parameter_count() > 0

        flat = _fsdp(_model(), "FULL_SHARD", use_orig_params=False)
        with pytest.raises(ValueError, match="use_orig_params"):
            _fsdp_optimizer(flat)
        (group,) = _fsdp_optimizer(
            flat, ParamGroupsConfig(lr=0.02, split_params=False)
        ).param_groups
        assert {id(p) for p in group["params"]} == {id(p) for p in flat.parameters()}
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.multiprocess
def test_wrapped_fsdp_groups_match_single_process_training(tmp_path):
    mp.spawn(_wrapped_worker, args=(str(tmp_path / "init"),), nprocs=2)


def _validate(**options):
    config = load_config(str(REPO / "configs" / "config_exp_fsdp.yaml"))
    for path, value in options.items():
        section, key = path.split(".")
        target = (
            config.distributed.fsdp
            if section == "fsdp"
            else (
                config.training.main.common.param_groups
                if section == "param_groups"
                else config.training.main.optimizer
            )
        )
        setattr(target, key, value)
    torch.manual_seed(0)
    fsdp_script._validate_fsdp_training_options(
        config, fsdp_script.create_large_model(config)
    )


@pytest.mark.parametrize("sharding", ["FULL_SHARD", "SHARD_GRAD_OP", "HYBRID_SHARD"])
def test_validation_rejects_muonh_on_sharded_parameters(sharding):
    with pytest.raises(ValueError, match="MuonH needs full 2-D matrices"):
        _validate(**{"optimizer.name": "muonh", "fsdp.sharding_strategy": sharding})


def test_validation_accepts_muonh_without_sharding():
    _validate(**{"optimizer.name": "muonh", "fsdp.sharding_strategy": "NO_SHARD"})
    with pytest.raises(ValueError, match="MuonH needs full 2-D matrices"):
        _validate(
            **{
                "optimizer.name": "muon_hyperball",
                "fsdp.sharding_strategy": "NO_SHARD",
                "fsdp.use_orig_params": False,
                "param_groups.split_params": False,
            }
        )


def test_validation_rejects_scalar_parameters_fsdp_cannot_shard():
    config = load_config(str(REPO / "configs" / "config_exp_fsdp.yaml"))
    core = _core()
    model = Classifier(Identity(IN_DIM), core, Identity(core.output_dim))
    model.temperature = torch.nn.Parameter(torch.zeros(()))
    with pytest.raises(
        ValueError, match=r"scalar parameters \['temperature'\]"
    ) as error:
        fsdp_script._validate_fsdp_training_options(config, model)
    # The global output scale is a one-element vector, which FSDP shards.
    assert "shape of at least (1,)" in str(error.value)
    assert "use per_class or fixed" not in str(error.value)


def test_a_global_output_scale_trains_under_fsdp_validation():
    config = load_config(str(REPO / "configs" / "config_exp_fsdp.yaml"))
    core = _core()
    model = Classifier(
        Identity(IN_DIM), core, Identity(core.output_dim), output_scale_mode="global"
    )
    fsdp_script._validate_fsdp_training_options(config, model)


def test_validation_rejects_split_groups_over_flat_parameters():
    with pytest.raises(ValueError, match="use_orig_params: false with param_groups"):
        _validate(**{"fsdp.use_orig_params": False})
    _validate(**{"fsdp.use_orig_params": False, "param_groups.split_params": False})


def test_validation_rejects_fp16_mixed_precision_without_loss_scaling():
    with pytest.raises(ValueError, match="fp16 FSDP mixed precision"):
        _validate(
            **{"fsdp.mixed_precision": True, "fsdp.mixed_precision_dtype": "fp16"}
        )
    with pytest.raises(ValueError, match="fp16 FSDP mixed precision"):
        _validate(**{"fsdp.mixed_precision": True, "fsdp.reduce_dtype": "fp16"})
    _validate(**{"fsdp.mixed_precision": True})  # bf16 by default


def test_fsdp_wrap_policy_and_threshold_must_agree():
    from dendritic_modeling.config.config import FSDPConfig

    FSDPConfig(auto_wrap_policy="size", min_num_params=5000)
    with pytest.raises(ValueError, match="read only by auto_wrap_policy: size"):
        FSDPConfig(auto_wrap_policy="indexed_synapses", min_num_params=5000)
    with pytest.raises(ValueError, match="auto_wrap_policy must be"):
        FSDPConfig(auto_wrap_policy="size_based")


def test_retired_fsdp_knobs_load_only_when_equivalent(tmp_path):
    def load(fsdp):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({"distributed": {"mode": "fsdp", "fsdp": fsdp}}))
        return load_config(str(path))

    load({"cpu_init": True, "reduce_communication_overhead": True})
    with pytest.raises(ValueError, match="reduce_communication_overhead=False"):
        load({"reduce_communication_overhead": False})
