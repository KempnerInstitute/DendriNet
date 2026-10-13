"""Training strategies honor configured optimization and loss behavior."""

from __future__ import annotations

from copy import deepcopy

import pytest
import torch
import torch.nn as nn
from test_operator_config_contract import _config
from torch.utils.data import TensorDataset

from dendritic_modeling.models import Classifier, RecurrentClassifier
from dendritic_modeling.networks import Identity
from dendritic_modeling.networks.architectures.classical.mlp.mlp import MLP
from dendritic_modeling.networks.architectures.factory import get_architecture
from dendritic_modeling.networks.architectures.recurrent import (
    EILayerConfig,
    EINetwork,
    EINetworkConfig,
    PopulationConfig,
)
from dendritic_modeling.training.factory import get_trainer
from dendritic_modeling.training.loss.functions import get_loss_function
from dendritic_modeling.training.optimizers.custom import CustomWeightDecayOptimizer

IN_DIM = 6
N_CLASSES = 3


# ----------------------------------------------------------------------------
# Tiny model and trainer builders
# ----------------------------------------------------------------------------


def _wrap(core, *, recurrent=False, in_dim=IN_DIM):
    decoder = nn.Linear(core.output_dim, N_CLASSES)
    if recurrent:
        return RecurrentClassifier(Identity(in_dim), core, decoder)
    return Classifier(Identity(in_dim), core, decoder, output_scale_mode="fixed")


def _einet_model(seed=13):
    torch.manual_seed(seed)
    core = get_architecture(
        "einet", deepcopy(_config("einet", "shunting", "param_tanh")), input_dim=IN_DIM
    )
    return _wrap(core)


def _population_model(seed=13, **population_options):
    config = _config("population_network", "shunting", "param_tanh")
    defaults = config["population_network"]["layers"][0]["population_defaults"]
    defaults.update(
        weight_transform="exp",
        ff_excitatory_synapses=3,
        indexed_seed=7,
        **population_options,
    )
    torch.manual_seed(seed)
    core = get_architecture("population_network", config, input_dim=IN_DIM)
    return _wrap(core)


def _einet_model_with_inhibition(seed=13):
    """EINet whose branches receive inhibitory synapses."""
    config = deepcopy(_config("einet", "shunting", "param_tanh"))
    config["architecture"].update(
        inhibitory_layer_sizes=[2], inhibitory_branch_factors=[2]
    )
    config["connectivity"].update(
        ie_synapses_per_branch_per_layer=[2],
        ei_synapses_per_branch_per_layer=[2],
        ii_synapses_per_branch_per_layer=[0],
    )
    torch.manual_seed(seed)
    return _wrap(get_architecture("einet", config, input_dim=IN_DIM))


def _mlp_model(seed=13):
    torch.manual_seed(seed)
    core = MLP(input_dim=IN_DIM, hidden_dims=[8], output_dim=8)
    core.output_dim = 8
    return _wrap(core)


def _recurrent_model(seed=13):
    population = {
        "branch_factors": [2],
        "ff_excitatory_synapses": 3,
        "rec_excitatory_synapses": 2,
        "topk_type": "standard",
    }
    torch.manual_seed(seed)
    core = EINetwork(
        EINetworkConfig(
            input_dim=IN_DIM,
            layers=[
                EILayerConfig(
                    excitatory=PopulationConfig(n_neurons=4, **population),
                    inhibitory=PopulationConfig(n_neurons=2, **population),
                    recurrent=True,
                    dt=1.0,
                )
            ],
            output_mode="last",
        )
    )
    return _wrap(core, recurrent=True)


def _data(n=48, seed=0, *, sequence=False):
    generator = torch.Generator().manual_seed(seed)
    shape = (n, 4, IN_DIM) if sequence else (n, IN_DIM)
    x = torch.rand(*shape, generator=generator)
    y = torch.randint(0, N_CLASSES, (n,), generator=generator)
    return TensorDataset(x, y)


def _optimizer(model, *, lr=0.01, synapse_decay_rate=0.0, weight_boosting=False):
    return CustomWeightDecayOptimizer(
        model,
        torch.optim.Adam(model.parameters(), lr=lr),
        weight_decay=synapse_decay_rate,
        weight_boosting=weight_boosting,
    )


def _trainer_config(**overrides):
    config = {
        "loss_function": "ce",
        "epochs": 2,
        "batch_size": 16,
        "shuffle": False,
        "grad_clip_value": 2.0,
        "early_stopping": False,
        "load_best_state_dict": False,
        "plot_losses": False,
        "suppress_prints": True,
        "save_path": None,
        "num_workers": 0,
        "pin_memory": False,
        "task": "classification",
        "seed": 0,
    }
    config.update(overrides)
    return config


def _make_trainer(strategy, model, optimizer=None, **overrides):
    optimizer = optimizer if optimizer is not None else _optimizer(model)
    trainer = get_trainer(strategy, optimizer, _trainer_config(**overrides))
    trainer.device = "cpu"
    return trainer, optimizer


def _state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _max_abs_diff(first, second):
    return max(
        (
            float((first[name].double() - second[name].double()).abs().max())
            for name in first
            if first[name].is_floating_point() and first[name].numel()
        ),
        default=0.0,
    )


# ----------------------------------------------------------------------------
# 1. grad_clip_value: 0 disables clipping instead of zeroing every gradient
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("clip_value", [0, 0.0, None])
def test_zero_grad_clip_value_disables_clipping_in_standard_trainer(clip_value):
    reference = _mlp_model()
    disabled = deepcopy(reference)
    initial = _state(reference)

    trainer, _ = _make_trainer("standard", reference, grad_clip_value=1e9)
    trainer.train(reference, _data(), _data(seed=1))
    trainer, _ = _make_trainer("standard", disabled, grad_clip_value=clip_value)
    trainer.train(disabled, _data(), _data(seed=1))

    assert _max_abs_diff(_state(disabled), initial) > 0.0
    assert _max_abs_diff(_state(disabled), _state(reference)) == 0.0


def test_zero_grad_clip_value_disables_clipping_in_fsdp_epoch(monkeypatch):
    import torch.distributed as dist

    from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer

    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None: None)
    model = _mlp_model()
    initial = _state(model)
    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer.device = torch.device("cpu")
    trainer.use_amp = False
    trainer.grad_clip_value = 0
    trainer.loss_function = get_loss_function("ce")
    trainer.loss_fn = trainer.loss_function.from_predictions
    trainer._class_logit_loss = True
    trainer.log_first_step_diagnostics = False
    trainer.first_step_diagnostics = None
    trainer._use_precomputed_core = False

    trainer._epoch_train_fsdp(
        model, torch.utils.data.DataLoader(_data(16), batch_size=8)
    )

    assert _max_abs_diff(_state(model), initial) > 0.0


def test_zero_grad_clip_value_disables_clipping_in_two_step_stage_one():
    model = _einet_model()
    initial = _state(model)
    trainer, _ = _make_trainer(
        "freeze_branches", model, grad_clip_value=0, epochs=0, pretrain_epochs=1
    )

    trainer.train(model, _data(), _data(seed=1))

    core_names = [name for name in initial if name.startswith("core_network.")]
    assert (
        max(
            float((initial[name] - model.state_dict()[name]).abs().max())
            for name in core_names
            if initial[name].is_floating_point() and initial[name].numel()
        )
        > 0.0
    )


# ----------------------------------------------------------------------------
# 2. common.lr_schedule / lr_warmup_epochs
# ----------------------------------------------------------------------------


def _record_epoch_learning_rates(trainer):
    learning_rates = []
    original = trainer._epoch_train

    def recording(model, loader):
        learning_rates.append(
            [group["lr"] for group in trainer._get_torch_optimizer().param_groups]
        )
        return original(model, loader)

    trainer._epoch_train = recording
    return learning_rates


# warmup=1, epochs=4: factors 1, 1, 0.5 * (1 + cos(pi / 3)), 0.5 * (1 + cos(2 pi / 3))
_COSINE_FACTORS = [1.0, 1.0, 0.75, 0.25]


@pytest.mark.parametrize(
    ("strategy", "builder"),
    [
        ("standard", _mlp_model),
        ("standard", _population_model),
        ("fa", _mlp_model),
        ("dfa", _einet_model),
        ("shunting_dfa", _einet_model),
        ("soma_dfa", _einet_model),
    ],
)
def test_cosine_lr_schedule_is_applied_per_epoch(strategy, builder):
    model = builder()
    trainer, _ = _make_trainer(
        strategy, model, epochs=4, lr_schedule="cosine", lr_warmup_epochs=1
    )
    learning_rates = _record_epoch_learning_rates(trainer)

    trainer.train(model, _data(), _data(seed=1))

    assert [rates[0] / 0.01 for rates in learning_rates] == pytest.approx(
        _COSINE_FACTORS
    )


def test_standard_cosine_schedule_matches_recurrent_trainer():
    standard_model = _mlp_model()
    standard, _ = _make_trainer(
        "standard", standard_model, epochs=4, lr_schedule="cosine", lr_warmup_epochs=1
    )
    standard_rates = _record_epoch_learning_rates(standard)
    standard.train(standard_model, _data(), _data(seed=1))

    recurrent_model = _recurrent_model()
    recurrent, _ = _make_trainer(
        "recurrent",
        recurrent_model,
        epochs=4,
        lr_schedule="cosine",
        lr_warmup_epochs=1,
    )
    recurrent_rates = _record_epoch_learning_rates(recurrent)
    recurrent.train(recurrent_model, _data(sequence=True), _data(seed=1, sequence=True))

    assert standard_rates == recurrent_rates
    assert [rates[0] / 0.01 for rates in recurrent_rates] == pytest.approx(
        _COSINE_FACTORS
    )


def test_lr_schedule_none_leaves_learning_rate_untouched():
    model = _mlp_model()
    trainer, _ = _make_trainer("standard", model, epochs=3, lr_schedule="none")
    learning_rates = _record_epoch_learning_rates(trainer)

    trainer.train(model, _data(), _data(seed=1))

    assert trainer.scheduler is None
    assert learning_rates == [[0.01]] * 3


@pytest.mark.parametrize(
    "strategy",
    [
        "voltage_stabilization",
        "homeostatic_control",
        "ei_equilibrium",
        "freeze_layers",
        "freeze_branches",
        "freeze_branch_kl",
        "local_ca",
    ],
)
def test_strategies_without_epoch_schedule_reject_lr_schedule(strategy):
    model = _einet_model()
    with pytest.raises(ValueError, match="does not apply lr_schedule='cosine'"):
        _make_trainer(strategy, model, lr_schedule="cosine")


def test_multi_stage_rejects_lr_schedule():
    model = _mlp_model()
    with pytest.raises(ValueError, match="does not apply lr_schedule='cosine'"):
        _make_trainer(
            "multi_stage",
            model,
            lr_schedule="cosine",
            multi_stage_config={"stages": [{"learning_strategy": "standard"}]},
        )


@pytest.mark.parametrize("strategy", ["standard", "recurrent"])
def test_lr_warmup_without_cosine_schedule_is_rejected(strategy):
    model = _recurrent_model() if strategy == "recurrent" else _mlp_model()
    with pytest.raises(ValueError, match="requires lr_schedule='cosine'"):
        _make_trainer(strategy, model, lr_schedule="none", lr_warmup_epochs=2)


# ----------------------------------------------------------------------------
# Typed-config helpers (production YAML loader and trainer-config assembly)
# ----------------------------------------------------------------------------

_TYPED_MAIN = {
    "strategy": "standard",
    "optimizer": {"name": "adam", "lr": 0.001},
    "common": {
        "epochs": 1,
        "batch_size": 16,
        "num_workers": 0,
        "plot_losses": False,
        "suppress_prints": True,
        "loss_function": "ce",
        "early_stopping": False,
        "load_best_state_dict": False,
    },
}


def _deep_update(base, update):
    merged = deepcopy(base)
    for key, value in (update or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _typed_config(tmp_path, main=None, *, experiment=None, extra=None, name="cfg"):
    import yaml

    from dendritic_modeling.config import load_config

    raw = {
        "training": {"main": _deep_update(_TYPED_MAIN, main)},
        "experiment": {"seed": 5, **(experiment or {})},
        "analysis": {},
    }
    raw.update(extra or {})
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(str(path))


def _trainer_kwargs_from_typed(config, tmp_path):
    from dendritic_modeling.scripts.script_utils.config_utils import (
        prepare_trainer_config,
    )

    kwargs = prepare_trainer_config(
        main_train_config=config.training.main,
        experiment_config=config.experiment,
        save_path=str(tmp_path),
    )
    kwargs["task"] = "classification"
    return kwargs


# ----------------------------------------------------------------------------
# 7. Recorded typed strategy defaults equal the runtime trainer defaults
# ----------------------------------------------------------------------------

_WARMUP_FIELDS = (
    "pretrain_epochs",
    "stabilize_mode",
    "freeze_reactivation",
    "target_voltage",
    "target_saturation",
    "loss_metric",
)


@pytest.mark.parametrize(
    ("strategy", "fields"),
    [
        ("voltage_stabilization", _WARMUP_FIELDS),
        ("homeostatic_control", _WARMUP_FIELDS),
        ("ei_equilibrium", ("pretrain_epochs", "freeze_reactivation", "loss_metric")),
        (
            "freeze_layers",
            ("pretrain_epochs", "reverse_training", "epochs_per_layer"),
        ),
        ("freeze_branches", ("pretrain_epochs",)),
        ("freeze_branch_kl", ("pretrain_epochs", "kl_weight")),
    ],
)
def test_recorded_strategy_defaults_match_trainer_runtime(tmp_path, strategy, fields):
    from dataclasses import asdict

    config = _typed_config(tmp_path, {"strategy": strategy})
    recorded = asdict(config)["training"]["main"]["strategies"][strategy]
    model = _einet_model()
    trainer = get_trainer(
        strategy, _optimizer(model), _trainer_kwargs_from_typed(config, tmp_path)
    )

    for field in fields:
        assert getattr(trainer, field) == recorded[field], field


@pytest.mark.parametrize("strategy", ["standard", "soma_dfa", "local_ca"])
def test_the_local_rule_block_reaches_only_the_strategies_that_read_it(
    tmp_path, strategy
):
    config = _typed_config(
        tmp_path,
        {"strategy": strategy, "learning_strategy_config": {"rule_variant": "4f"}},
    )
    kwargs = _trainer_kwargs_from_typed(config, tmp_path)
    model = _einet_model()

    trainer = get_trainer(strategy, _optimizer(model), kwargs)

    if strategy == "local_ca":
        assert trainer.local_cfg.rule_variant == "4f"
    else:
        assert not hasattr(trainer, "local_rule_config")


def test_multi_stage_hands_the_local_rule_block_to_local_stages(tmp_path):
    from dendritic_modeling.training.strategies import multi_stage

    stages = [
        {"learning_strategy": "standard", "epochs": 1},
        {"learning_strategy": "local_ca", "epochs": 1},
    ]
    config = _typed_config(
        tmp_path,
        {
            "strategy": "multi_stage",
            "learning_strategy_config": {"rule_variant": "4f"},
            "strategies": {"multi_stage": {"stages": stages}},
        },
    )
    model = _einet_model()
    trainer = get_trainer(
        "multi_stage", _optimizer(model), _trainer_kwargs_from_typed(config, tmp_path)
    )
    stage_configs = [
        multi_stage._build_stage_trainer_config(
            trainer.base_trainer_config, trainer.multi_stage_config, stage
        )
        for stage in trainer.multi_stage_config.stages
    ]
    assert "local_rule_config" not in stage_configs[0]
    assert stage_configs[1]["local_rule_config"]["rule_variant"] == "4f"

    trainer.device = "cpu"
    trainer.train(model, _data(), _data(seed=1))


@pytest.mark.parametrize(
    "options, message",
    [
        (
            {"reactivation_update_mode": "quantile"},
            "recalibrate_reactivation_every is 0",
        ),
        ({"recalibrate_reactivation_every": -1}, "must be >= 0"),
        ({"reactivation_recalibration_start_epoch": 0}, "counts epochs from 1"),
        ({"reactivation_recalibration_num_batches": 0}, "must be >= 1"),
        ({"reactivation_recalibration_ema_alpha": 0.0}, r"in \(0, 1\]"),
        ({"reactivation_update_mode": "sgd"}, "'sgd' is not supported"),
        ({"reactivation_recalibration_mode": "mad"}, "'mad' is not supported"),
    ],
)
def test_gate_update_options_are_checked_at_load_and_construction(
    tmp_path, options, message
):
    with pytest.raises(ValueError, match=message):
        _typed_config(tmp_path, {"common": options})
    with pytest.raises(ValueError, match=message):
        _make_trainer("standard", _mlp_model(), **options)


@pytest.mark.parametrize(
    "field, alias, canonical",
    [
        ("reactivation_update_mode", "bp", "backprop"),
        ("reactivation_update_mode", "freeze", "frozen"),
        ("reactivation_update_mode", "quantile-rule", "quantile"),
        ("reactivation_recalibration_mode", "empirical", "median_mad"),
    ],
)
def test_gate_update_mode_aliases_load_with_a_deprecation(field, alias, canonical):
    with pytest.warns(DeprecationWarning, match="is deprecated"):
        trainer, _ = _make_trainer(
            "standard",
            _mlp_model(),
            recalibrate_reactivation_every=1,
            **{field: alias},
        )
    assert getattr(trainer, field) == canonical


def test_training_results_record_the_data_parallel_batch_layout():
    model = _mlp_model()
    trainer, _ = _make_trainer("standard", model, epochs=1, batch_size=16)

    results = trainer.train(model, _data(40), _data(16, seed=1))

    assert results["data_parallel"] == {
        "world_size": 1,
        "batch_size_per_rank": 16,
        "global_batch_size": 16,
        "train_steps_per_epoch": 3,
    }


def test_the_global_batch_counts_every_rank():
    from torch.utils.data import DataLoader

    from dendritic_modeling.training.dataloader_utils import data_parallel_layout

    layout = data_parallel_layout(DataLoader(_data(40), batch_size=8), 4)
    assert layout["global_batch_size"] == 32
    assert layout["batch_size_per_rank"] == 8


def _two_stage_run(tmp_path, final_state, second_stage=None):
    model = _einet_model()
    stages = [
        {"learning_strategy": "standard", "epochs": 1, "stage_name": "first"},
        {"learning_strategy": "standard", "epochs": 1, "stage_name": "second"},
    ]
    stages[1].update(second_stage or {})
    trainer, _ = _make_trainer(
        "multi_stage",
        model,
        save_path=str(tmp_path),
        load_best_state_dict=True,
        multi_stage_config={"final_state": final_state, "stages": stages},
    )
    trainer.device = "cpu"
    return model, trainer


def test_multi_stage_ends_on_the_last_stage_in_stage_directories(tmp_path):
    model, trainer = _two_stage_run(tmp_path, "last_stage")

    results = trainer.train(model, _data(), _data(seed=1))

    assert results["final_state"] == "last_stage"
    assert results["final_stage"] == "second"
    second = results["stages"][1]["results"]
    assert results["best_loss"] == second["best_loss"]
    assert _max_abs_diff(second["best_state_dict"], model.state_dict()) == 0
    for name in ("first", "second"):
        assert (tmp_path / name / "stage_checkpoint.pt").is_file()


def test_multi_stage_best_stage_reloads_the_lowest_loss_stage(tmp_path):
    model, trainer = _two_stage_run(tmp_path, "best_stage")

    results = trainer.train(model, _data(), _data(seed=1))

    losses = {
        record["stage_name"]: record["results"]["best_loss"]
        for record in results["stages"]
    }
    best = min(losses, key=losses.get)
    assert results["final_stage"] == best
    best_state = results["stages"][list(losses).index(best)]["results"]
    assert _max_abs_diff(best_state["best_state_dict"], model.state_dict()) == 0


def test_multi_stage_best_stage_rejects_stages_with_different_losses(tmp_path):
    model, trainer = _two_stage_run(
        tmp_path, "best_stage", {"trainer_config": {"loss_function": "cat_nll"}}
    )
    with pytest.raises(ValueError, match="different loss functions"):
        trainer.train(model, _data(), _data(seed=1))
    assert not (tmp_path / "first").exists()  # rejected before training


def test_recurrent_grad_clip_precedence_is_explicit_strategy_then_common(tmp_path):
    common_only = _typed_config(
        tmp_path,
        {"strategy": "recurrent", "common": {"grad_clip_value": 3.0}},
        name="common_only",
    )
    explicit = _typed_config(
        tmp_path,
        {
            "strategy": "recurrent",
            "common": {"grad_clip_value": 3.0},
            "strategies": {"recurrent": {"grad_clip_value": 0.5}},
        },
        name="explicit",
    )
    model = _recurrent_model()

    first = get_trainer(
        "recurrent",
        _optimizer(model),
        _trainer_kwargs_from_typed(common_only, tmp_path),
    )
    second = get_trainer(
        "recurrent", _optimizer(model), _trainer_kwargs_from_typed(explicit, tmp_path)
    )

    assert first.grad_clip_value == 3.0
    assert second.grad_clip_value == 0.5


# ----------------------------------------------------------------------------
# 8. freeze_reactivation freezes the warm-up phase only
# ----------------------------------------------------------------------------


def _reactivation_parameters(model):
    from dendritic_modeling.networks import ParametricActivation

    return [
        param
        for module in model.modules()
        if isinstance(module, ParametricActivation)
        for param in module.parameters()
    ]


@pytest.mark.parametrize(
    "strategy", ["voltage_stabilization", "homeostatic_control", "ei_equilibrium"]
)
@pytest.mark.parametrize("freeze_reactivation", [True, False])
def test_freeze_reactivation_applies_to_warmup_phase_only(
    strategy, freeze_reactivation
):
    model = _einet_model_with_inhibition()
    trainer, _ = _make_trainer(
        strategy,
        model,
        epochs=2,
        pretrain_epochs=1,
        freeze_reactivation=freeze_reactivation,
    )
    phases = []
    original = trainer._epoch_train

    def recording(epoch_model, loader):
        before = [param.detach().clone() for param in _reactivation_parameters(model)]
        result = original(epoch_model, loader)
        change = max(
            float((param.detach() - previous).abs().max())
            for param, previous in zip(_reactivation_parameters(model), before)
        )
        phases.append((type(trainer.loss_function).__name__, change))
        return result

    trainer._epoch_train = recording
    trainer.train(model, _data(), _data(seed=1))

    (warmup_loss, warmup_change), *main_phase = phases
    assert warmup_loss != "CrossEntropyLoss"
    assert [loss for loss, _ in main_phase] == ["CrossEntropyLoss"] * 2
    if freeze_reactivation:
        assert warmup_change == 0.0
    else:
        assert warmup_change > 0.0
    assert all(change > 0.0 for _, change in main_phase)
    assert all(param.requires_grad for param in _reactivation_parameters(model))


def test_homeostatic_gate_occupancy_uses_the_gate_occupancy_objective():
    from dendritic_modeling.training.loss.functions import (
        ExcitationInhibitionBalanceLoss,
        HomeostaticControlLoss,
        VoltageStabilizationLoss,
    )

    model = _einet_model_with_inhibition()
    x = _data(8)[:][0]

    def value(loss_fn):
        handles = loss_fn.attach_forward_hooks(model)
        try:
            return float(loss_fn(model, x).detach())
        finally:
            loss_fn.remove_forward_hooks(handles)

    for mode in ("gate_occupancy", "vinf"):
        combined = value(HomeostaticControlLoss(stabilize_mode=mode))
        parts = value(VoltageStabilizationLoss(stabilize_mode=mode)) + value(
            ExcitationInhibitionBalanceLoss()
        )
        assert combined == pytest.approx(parts / 2, rel=1e-6)
    low = value(HomeostaticControlLoss("gate_occupancy", target_saturation=0.0))
    high = value(HomeostaticControlLoss("gate_occupancy", target_saturation=0.9))
    assert low != high


def test_homeostatic_control_needs_inhibitory_synapses():
    model = _einet_model()
    trainer, _ = _make_trainer("homeostatic_control", model, pretrain_epochs=1)
    with pytest.raises(ValueError, match="excitatory and inhibitory synapses"):
        trainer.train(model, _data(), _data(seed=1))


@pytest.mark.parametrize("strategy", ["voltage_stabilization", "homeostatic_control"])
def test_warmup_options_are_checked_at_load(tmp_path, strategy):
    for options, message in (
        ({"pretrain_epochs": 0}, "at least one epoch"),
        ({"stabilize_mode": "v_inf"}, "stabilize_mode='v_inf' is not supported"),
        ({"target_saturation": 1.5}, "fraction of gate outputs"),
    ):
        with pytest.raises(ValueError, match=message):
            _typed_config(
                tmp_path,
                {"strategy": strategy, "strategies": {strategy: options}},
            )
    model = _einet_model_with_inhibition()
    trainer, _ = _make_trainer(strategy, model, pretrain_epochs=0)
    with pytest.raises(ValueError, match="at least one epoch"):
        trainer.train(model, _data(), _data(seed=1))


def test_ei_equilibrium_trains_from_a_typed_config(tmp_path):
    config = _typed_config(
        tmp_path,
        {
            "strategy": "ei_equilibrium",
            "common": {"epochs": 1},
            "strategies": {"ei_equilibrium": {"pretrain_epochs": 2}},
        },
    )
    model = _einet_model_with_inhibition()
    trainer = get_trainer(
        "ei_equilibrium",
        _optimizer(model),
        _trainer_kwargs_from_typed(config, tmp_path),
    )
    trainer.device = "cpu"

    results = trainer.train(model, _data(), _data(seed=1))

    assert results["phases"][0] == {
        "name": "warmup",
        "epochs": 2,
        "loss": "Excitation-Inhibition Balance",
    }
    with pytest.raises(ValueError, match="at least one epoch"):
        _typed_config(
            tmp_path,
            {"strategies": {"ei_equilibrium": {"pretrain_epochs": 0}}},
            name="zero",
        )


def test_warmup_results_keep_every_history_aligned_and_name_the_phases():
    model = _einet_model_with_inhibition()
    trainer, _ = _make_trainer(
        "voltage_stabilization", model, epochs=2, pretrain_epochs=1
    )

    results = trainer.train(model, _data(), _data(seed=1))

    lengths = {
        key: len(results[key])
        for key in (
            "train_losses",
            "valid_losses",
            "train_losses_base",
            "train_losses_reg",
        )
    }
    assert set(lengths.values()) == {3}, lengths
    assert results["phases"] == [
        {"name": "warmup", "epochs": 1, "loss": "Voltage Stabilization"},
        {"name": "main", "epochs": 2},
    ]


@pytest.mark.parametrize(
    "strategy, options, phases",
    [
        (
            "freeze_layers",
            {"pretrain_epochs": 1, "epochs_per_layer": 1, "final_tune_epochs": 1},
            ["pretraining", "layers", "final_tuning"],
        ),
        (
            "freeze_branches",
            {"pretrain_epochs": 1, "epochs": 2},
            ["local_pretraining", "end_to_end"],
        ),
        (
            "voltage_stabilization",
            {"pretrain_epochs": 1, "epochs": 2},
            ["warmup", "main"],
        ),
    ],
)
def test_multi_phase_trainers_number_epochs_across_their_phases(
    strategy, options, phases
):
    model = _einet_model_with_inhibition()
    trainer, _ = _make_trainer(strategy, model, **options)
    epochs = []
    trainer._analyze = lambda: epochs.append(trainer.epoch_counter)

    results = trainer.train(model, _data(), _data(seed=1))

    assert epochs == [1, 2, 3]  # one epoch file per epoch, none overwritten
    assert trainer.total_epochs == 3
    assert [phase["name"] for phase in results["phases"]] == phases
    assert sum(phase["epochs"] for phase in results["phases"]) == 3
    assert len(results["train_losses"]) == 3
    assert 1 <= results["best_epoch"] <= 3


def test_the_main_phase_never_restores_a_warmup_state():
    import math

    model = _einet_model_with_inhibition()
    trainer, _ = _make_trainer(
        "voltage_stabilization",
        model,
        epochs=1,
        pretrain_epochs=1,
        load_best_state_dict=True,
    )
    values = iter([0.5, math.nan])  # warm-up finite, main phase never finite
    trainer._non_distributed_validation_loss = lambda *_args: next(values)
    with pytest.raises(RuntimeError, match="no epoch had a finite validation loss"):
        trainer.train(model, _data(), _data(seed=1))


# ----------------------------------------------------------------------------
# 9. Regularization / loss_function / clipping in recurrent, FA/DFA, local_ca
# ----------------------------------------------------------------------------


def _final_state_after_training(strategy, builder, *, sequence=False, **overrides):
    model = builder()
    trainer, _ = _make_trainer(strategy, model, epochs=1, **overrides)
    results = trainer.train(
        model, _data(sequence=sequence), _data(seed=1, sequence=sequence)
    )
    return _state(model), results


def test_recurrent_trainer_applies_regularization():
    plain, _ = _final_state_after_training("recurrent", _recurrent_model, sequence=True)
    regularized, results = _final_state_after_training(
        "recurrent",
        _recurrent_model,
        sequence=True,
        regularization={"l2_weight": 10.0},
    )

    assert _max_abs_diff(plain, regularized) > 0.0
    assert results["train_losses_reg"][0] > 0.0
    assert results["train_losses"][0] == pytest.approx(
        results["train_losses_base"][0] + results["train_losses_reg"][0]
    )


@pytest.mark.parametrize(
    ("task", "loss_function"),
    [("classification", "bce"), ("classification", "cat_nll"), ("regression", "bce")],
)
def test_recurrent_trainer_rejects_loss_function_it_would_replace(task, loss_function):
    with pytest.raises(ValueError, match="would not be applied"):
        _make_trainer(
            "recurrent", _recurrent_model(), task=task, loss_function=loss_function
        )


@pytest.mark.parametrize(
    ("task", "loss_function"),
    [
        ("classification", "ce"),
        ("classification", "cross_entropy"),
        ("regression", "mse"),
        ("regression", "ce"),
        ("classification", "temporal_event_ce"),
    ],
)
def test_recurrent_trainer_accepts_applied_loss_functions(task, loss_function):
    trainer, _ = _make_trainer(
        "recurrent", _recurrent_model(), task=task, loss_function=loss_function
    )
    assert trainer.task == task


@pytest.mark.parametrize(
    ("strategy", "builder"),
    [("dfa", _einet_model), ("shunting_dfa", _einet_model), ("fa", _mlp_model)],
)
def test_feedback_alignment_applies_regularization(strategy, builder):
    plain, _ = _final_state_after_training(strategy, builder)
    regularized, results = _final_state_after_training(
        strategy, builder, regularization={"l2_weight": 10.0}
    )

    assert _max_abs_diff(plain, regularized) > 0.0
    assert results["train_losses_reg"][0] > 0.0


def test_local_ca_rejects_regularization_it_does_not_apply():
    model = _einet_model()
    trainer, _ = _make_trainer(
        "local_ca", model, local_rule_config={}, regularization={"l2_weight": 1.0}
    )
    with pytest.raises(
        ValueError, match=r"does not apply training\.main\.regularization"
    ):
        trainer.train(model, _data(), _data(seed=1))


@pytest.mark.parametrize(
    ("common_clip", "local_clip", "rejected"),
    [
        (1.0, 5.0, True),
        (0.0, 5.0, True),
        (2.0, 5.0, False),  # typed common default: local value governs
        (5.0, 5.0, False),
        (3.0, 3.0, False),
        (0.0, 0.0, False),
    ],
)
def test_local_ca_common_grad_clip_value_must_match_local_clip(
    common_clip, local_clip, rejected
):
    model = _einet_model()
    trainer, _ = _make_trainer(
        "local_ca",
        model,
        grad_clip_value=common_clip,
        local_rule_config={"clip_grad_value": local_clip},
    )
    if rejected:
        with pytest.raises(ValueError, match="would not be applied"):
            trainer.train(model, _data(), _data(seed=1))
    else:
        trainer.train(model, _data(), _data(seed=1))


# ----------------------------------------------------------------------------
# G. Backward-hook gradient scaling with trainers that assign gradients directly
# ----------------------------------------------------------------------------


def _scaled_einet_model(seed=13, strategy="distal_upweight_by_idx"):
    config = deepcopy(_config("einet", "shunting", "param_tanh"))
    config["blocklinear"] = {"gradient_scaling": strategy}
    torch.manual_seed(seed)
    core = get_architecture("einet", config, input_dim=IN_DIM)
    return _wrap(core)


@pytest.mark.parametrize(
    ("strategy", "extra"),
    [
        ("dfa", {}),
        ("shunting_dfa", {}),
        ("local_ca", {"local_rule_config": {}}),
    ],
)
def test_manual_gradient_trainers_reject_gradient_scaling(strategy, extra):
    model = _scaled_einet_model()
    trainer, _ = _make_trainer(strategy, model, **extra)
    with pytest.raises(ValueError, match="gradient scaling"):
        trainer.train(model, _data(), _data(seed=1))


def test_soma_dfa_applies_backward_hook_gradient_scaling():
    states = []
    for builder in (_einet_model, _scaled_einet_model):
        model = builder()
        optimizer = CustomWeightDecayOptimizer(
            model, torch.optim.SGD(model.parameters(), lr=0.5), weight_decay=0.0
        )
        trainer, _ = _make_trainer("soma_dfa", model, optimizer, epochs=1)
        trainer.train(model, _data(), _data(seed=1))
        states.append(_state(model))

    assert _max_abs_diff(*states) > 0.0


# ----------------------------------------------------------------------------
# 4. FA/DFA under DDP keep replicas synchronized; E. FA on tree cores
# ----------------------------------------------------------------------------


def _ddp_feedback_alignment_worker(rank, world_size, init_file, queue):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        results = {}
        for strategy, builder in (
            ("dfa", _einet_model),
            ("shunting_dfa", _einet_model),
            ("fa", _mlp_model),
        ):
            model = builder()  # identical initialization on every rank
            ddp_model = torch.nn.parallel.DistributedDataParallel(model)
            trainer, _ = _make_trainer(strategy, model, epochs=1)
            trainer.train(ddp_model, _data(64), _data(32, seed=1))
            flat = torch.cat([param.detach().flatten() for param in model.parameters()])
            gathered = [torch.zeros_like(flat) for _ in range(world_size)]
            dist.all_gather(gathered, flat)
            results[strategy] = float((gathered[0] - gathered[1]).abs().max())
        if rank == 0:
            queue.put(results)
    finally:
        dist.destroy_process_group()


@pytest.mark.slow
@pytest.mark.multiprocess
def test_feedback_alignment_replicas_stay_synchronized_under_ddp(tmp_path):
    import torch.multiprocessing as mp

    context = mp.get_context("spawn")
    queue = context.SimpleQueue()
    mp.start_processes(
        _ddp_feedback_alignment_worker,
        args=(2, str(tmp_path / "process_group"), queue),
        nprocs=2,
        join=True,
        start_method="spawn",
    )

    assert queue.get() == {"dfa": 0.0, "shunting_dfa": 0.0, "fa": 0.0}


@pytest.mark.parametrize("builder", [_einet_model, _population_model])
@pytest.mark.parametrize("strategy", ["fa", "shunting_fa"])
def test_fa_rejects_dendritic_tree_cores(strategy, builder):
    model = builder()
    trainer, _ = _make_trainer(strategy, model)
    with pytest.raises(ValueError, match="sequential chain"):
        trainer.train(model, _data(), _data(seed=1))


@pytest.mark.parametrize("strategy", ["fa", "dfa"])
def test_feedback_alignment_trains_hidden_layers_of_mlp_cores(strategy):
    model = _mlp_model()
    initial = _state(model)
    trainer, _ = _make_trainer(strategy, model, epochs=1)

    trainer.train(model, _data(), _data(seed=1))

    hidden = [name for name in initial if name.startswith("core_network.")]
    assert hidden
    assert all(
        float((initial[name] - model.state_dict()[name]).abs().max()) > 0.0
        for name in hidden
        if name.endswith("weight")
    )


# ----------------------------------------------------------------------------
# 3. multi_stage stage optimizers follow training.main.optimizer, stay wrapped
# A. reset_model fails when it cannot reset every parameter
# ----------------------------------------------------------------------------

_GROUP_LRS = {
    "lr": 0.05,
    "topk_lr": 0.05,
    "blocklinear_lr": 0.05,
    "reactivation_lr": 0.05,
    "decoder_lr": 0.05,
}


def _record_stage_optimizers(monkeypatch):
    from dendritic_modeling.training.strategies import multi_stage

    recorded = []
    original = multi_stage._build_stage_optimizer

    def recording(*args, **kwargs):
        optimizer = original(*args, **kwargs)
        recorded.append(optimizer)
        return optimizer

    monkeypatch.setattr(multi_stage, "_build_stage_optimizer", recording)
    return recorded


def test_multi_stage_rebuilds_stage_optimizers_from_configured_optimizer(monkeypatch):
    from dendritic_modeling.config.training import OptimizerConfig, ParamGroupsConfig
    from dendritic_modeling.training.optimizers import create_optimizer

    model = _population_model()
    optimizer_config = OptimizerConfig(
        name="adamw", lr=0.05, weight_decay=0.0, betas=(0.8, 0.95), eps=1e-6
    )
    base = CustomWeightDecayOptimizer(
        model,
        create_optimizer(
            model.get_param_groups(ParamGroupsConfig(**_GROUP_LRS)), optimizer_config
        ),
        weight_decay=0.02,
        weight_boosting=True,
    )
    recorded = _record_stage_optimizers(monkeypatch)
    stage = {"learning_strategy": "standard", "epochs": 1, "save_checkpoint": False}
    trainer, _ = _make_trainer(
        "multi_stage",
        model,
        base,
        optimizer_config=optimizer_config,
        synapse_decay_rate=0.02,
        weight_boosting=True,
        param_groups=dict(_GROUP_LRS),
        multi_stage_config={"stages": [stage, {**stage, "reset_optimizer": True}]},
    )

    trainer.train(model, _data(), _data(seed=1))

    first, second = recorded
    assert first is base  # the configured optimizer, not AdamW(lr=0.001)
    assert isinstance(second, CustomWeightDecayOptimizer)
    assert second.optimizer is not base.optimizer
    assert isinstance(second.optimizer, torch.optim.AdamW)
    assert second.weight_decay == 0.02 and second.weight_boosting is True
    for group in second.optimizer.param_groups:
        assert group["lr"] == 0.05
        assert group["betas"] == (0.8, 0.95)
        assert group["eps"] == 1e-6
        assert group["weight_decay"] == 0.0
    assert {group["name"] for group in second.optimizer.param_groups} == {
        group["name"] for group in base.optimizer.param_groups
    }


def test_multi_stage_wraps_optimizer_without_weight_decay_so_rewiring_runs():
    from dendritic_modeling.networks.architectures.excitation_inhibition.synapse.indexed_rewire import (
        IndexedRewireLinear,
    )

    model = _population_model(
        topk_type="indexed_rewire",
        indexed_rewire_frequency=1,
        indexed_rewire_quantile=0.5,
    )
    banks = [m for m in model.modules() if isinstance(m, IndexedRewireLinear)]
    before = [bank.connection_indices.clone() for bank in banks]
    trainer, _ = _make_trainer(
        "multi_stage",
        model,
        _optimizer(model, synapse_decay_rate=0.0),
        synapse_decay_rate=0.0,
        multi_stage_config={
            "stages": [
                {
                    "learning_strategy": "standard",
                    "epochs": 1,
                    "reset_optimizer": True,
                    "save_checkpoint": False,
                }
            ]
        },
    )

    trainer.train(model, _data(), _data(seed=1))

    assert banks
    assert any(
        not torch.equal(previous.cpu(), bank.connection_indices.cpu())
        for previous, bank in zip(before, banks)
    )


def test_multi_stage_reset_model_rejects_parameters_it_cannot_reset():
    model = _einet_model()
    trainer, _ = _make_trainer(
        "multi_stage",
        model,
        multi_stage_config={
            "continue_on_failure": True,
            "stages": [
                {
                    "learning_strategy": "standard",
                    "epochs": 1,
                    "reset_model": True,
                    "save_checkpoint": False,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="pre_w"):
        trainer.train(model, _data(), _data(seed=1))


def test_multi_stage_reset_model_runs_for_linear_models():
    model = _mlp_model()
    trainer, _ = _make_trainer(
        "multi_stage",
        model,
        multi_stage_config={
            "stages": [
                {
                    "learning_strategy": "standard",
                    "epochs": 1,
                    "reset_model": True,
                    "save_checkpoint": False,
                }
            ]
        },
    )

    results = trainer.train(model, _data(), _data(seed=1))

    assert len(results["stages"]) == 1


# ----------------------------------------------------------------------------
# 10. freeze_branches(_kl) / freeze_layers: ignored fields and unsupported cores
# D. stage-1 local optimizers keep param-group LRs and sparse maintenance
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["freeze_branches", "freeze_branch_kl"])
@pytest.mark.parametrize(
    ("field", "value"),
    [("reverse_training", True), ("epochs_per_branch", 3), ("final_tune_epochs", 2)],
)
def test_two_step_rejects_unimplemented_strategy_fields(strategy, field, value):
    model = _einet_model()
    trainer, _ = _make_trainer(strategy, model, **{field: value})
    with pytest.raises(ValueError, match=f"does not implement {field}"):
        trainer.train(model, _data(), _data(seed=1))


@pytest.mark.parametrize("strategy", ["freeze_branches", "freeze_branch_kl"])
@pytest.mark.parametrize(
    ("builder", "sequence", "message"),
    [(_mlp_model, False, "no parameters"), (_recurrent_model, True, "recurrent")],
)
def test_two_step_rejects_unsupported_cores(strategy, builder, sequence, message):
    model = builder()
    trainer, _ = _make_trainer(strategy, model, pretrain_epochs=1)
    with pytest.raises(ValueError, match=message):
        trainer.train(model, _data(sequence=sequence), _data(seed=1, sequence=sequence))


@pytest.mark.parametrize(
    ("builder", "sequence"),
    [(_population_model, False), (_mlp_model, False), (_recurrent_model, True)],
)
def test_freeze_layers_rejects_non_dendrinet_cores(builder, sequence):
    model = builder()
    trainer, _ = _make_trainer("freeze_layers", model)
    with pytest.raises(ValueError, match="cores are not supported"):
        trainer.train(model, _data(sequence=sequence), _data(seed=1, sequence=sequence))


@pytest.mark.parametrize("builder", [_einet_model, _population_model])
def test_freeze_branch_kl_local_heads_match_model_outputs(builder):
    model = builder()
    trainer, _ = _make_trainer("freeze_branch_kl", model, epochs=0, pretrain_epochs=1)

    trainer.train(model, _data(), _data(seed=1))

    assert [head.out_features for head in trainer.local_heads] == [N_CLASSES] * len(
        trainer.local_heads
    )


def test_two_step_stage_one_optimizers_keep_group_lrs_and_maintenance():
    from dendritic_modeling.config.training import OptimizerConfig, ParamGroupsConfig
    from dendritic_modeling.training.optimizers import create_optimizer
    from dendritic_modeling.training.strategies.two_step import _LocalLayerMaintenance

    model = _einet_model()
    groups = ParamGroupsConfig(
        lr=0.06,
        topk_lr=0.02,
        blocklinear_lr=0.03,
        reactivation_lr=0.04,
        decoder_lr=0.05,
    )
    base = CustomWeightDecayOptimizer(
        model,
        create_optimizer(model.get_param_groups(groups), OptimizerConfig(lr=0.001)),
        weight_decay=0.01,
        weight_boosting=True,
    )
    trainer, _ = _make_trainer(
        "freeze_branches", model, base, epochs=0, pretrain_epochs=1
    )
    calls = []
    original = _LocalLayerMaintenance.decay_weights

    def counting(self, weight_decay, weight_boosting=False):
        calls.append((weight_decay, weight_boosting))
        return original(self, weight_decay, weight_boosting)

    _LocalLayerMaintenance.decay_weights = counting
    try:
        trainer.train(model, _data(), _data(seed=1))
    finally:
        _LocalLayerMaintenance.decay_weights = original

    (local,) = trainer.local_optimizers
    assert isinstance(local, CustomWeightDecayOptimizer)
    assert local.weight_decay == 0.01 and local.weight_boosting is True
    named = {
        group.get("name"): group["lr"]
        for group in local.optimizer.param_groups
        if group.get("name")
    }
    assert named == {"topk": 0.02, "blocklinear": 0.03, "reactivation": 0.04}
    assert local.optimizer.param_groups[-1]["lr"] == 0.05  # local head = readout lr
    assert calls and all(boost is True for _, boost in calls)
    assert calls[0][0] == pytest.approx(0.02 * 0.01)


# ----------------------------------------------------------------------------
# 11. training.main.optimizer.lr versus the param-group learning rates
# ----------------------------------------------------------------------------


def _run_main_phase(tmp_path, main):
    from types import SimpleNamespace

    from dendritic_modeling.scripts.training import train_experiments

    config = _typed_config(tmp_path, main)
    model = _einet_model()
    return train_experiments._run_main_training_phase(
        unwrapped_model=model,
        model=model,
        training_config=config.training,
        model_config=SimpleNamespace(task="classification"),
        experiment_config=config.experiment,
        run_save_path=str(tmp_path),
        encoded_train_ds=_data(),
        encoded_valid_ds=_data(seed=1),
        analysis_manager=None,
    )


_EQUAL_GROUP_LRS = {
    "lr": 3e-4,
    "split_params": True,
    "topk_lr": 3e-4,
    "blocklinear_lr": 3e-4,
    "reactivation_lr": 3e-4,
    "decoder_lr": 3e-4,
}


def test_optimizer_lr_equal_to_every_group_lr_is_accepted(tmp_path):
    run = _run_main_phase(
        tmp_path,
        {
            "optimizer": {"name": "adam", "lr": 3e-4},
            "common": {"param_groups": dict(_EQUAL_GROUP_LRS)},
        },
    )

    assert {group["lr"] for group in run.optimizer.param_groups} == {3e-4}


def test_optimizer_lr_equal_to_the_base_rate_keeps_component_rates(tmp_path):
    # optimizer.lr == param_groups.lr,
    # with deliberately different component rates.
    run = _run_main_phase(
        tmp_path,
        {
            "optimizer": {"name": "adam", "lr": 3e-4},
            "common": {"param_groups": {**_EQUAL_GROUP_LRS, "topk_lr": 1e-3}},
        },
    )

    named = {group["name"]: group["lr"] for group in run.optimizer.param_groups}
    assert named["topk"] == 1e-3 and named["decoder"] == 3e-4


@pytest.mark.parametrize("optimizer", ["adam", "muonh"])
@pytest.mark.parametrize(
    "param_groups",
    [
        {**_EQUAL_GROUP_LRS, "lr": 1e-2},  # the base rate differs
        {"lr": 1e-2, "split_params": False},  # the single group differs
    ],
)
def test_optimizer_lr_that_param_groups_override_is_rejected(
    tmp_path, param_groups, optimizer
):
    with pytest.raises(ValueError, match=r"optimizer\.lr=0\.0003 is not applied"):
        _run_main_phase(
            tmp_path,
            {
                "optimizer": {"name": optimizer, "lr": 3e-4},
                "common": {"param_groups": param_groups},
            },
        )


def test_muonh_trains_at_the_param_group_rates(tmp_path):
    run = _run_main_phase(
        tmp_path,
        {
            "optimizer": {"name": "muonh"},
            "common": {"param_groups": {**_EQUAL_GROUP_LRS, "topk_lr": 1e-3}},
        },
    )

    assert {group["lr"] for group in run.optimizer.param_groups} == {3e-4, 1e-3}


def test_default_optimizer_lr_is_not_checked(tmp_path):
    run = _run_main_phase(
        tmp_path,
        {"common": {"param_groups": {**_EQUAL_GROUP_LRS, "topk_lr": 1e-3}}},
    )

    assert {group["lr"] for group in run.optimizer.param_groups} == {3e-4, 1e-3}


# ----------------------------------------------------------------------------
# 12. experiment.fast_mode decides unset determinism/TF32 flags
# ----------------------------------------------------------------------------

_FAST = {
    "deterministic": False,
    "cudnn_benchmark": True,
    "allow_tf32": True,
    "float32_matmul_precision": "high",
}
_EXACT = {
    "deterministic": True,
    "cudnn_benchmark": False,
    "allow_tf32": False,
    "float32_matmul_precision": "highest",
}


@pytest.mark.parametrize(
    ("experiment", "expected"),
    [
        ({"fast_mode": True}, _FAST),
        ({"fast_mode": False}, _EXACT),
        ({}, _EXACT),
        (
            {"fast_mode": True, "deterministic": True},
            {**_FAST, "deterministic": True, "cudnn_benchmark": False},
        ),
        (
            {"fast_mode": False, "allow_tf32": True},
            {**_EXACT, "allow_tf32": True, "float32_matmul_precision": "high"},
        ),
        (
            {"fast_mode": True, "float32_matmul_precision": "highest"},
            {**_FAST, "allow_tf32": False, "float32_matmul_precision": "highest"},
        ),
        ({"fast_mode": True, "strict_deterministic": True}, _EXACT),
    ],
)
def test_fast_mode_decides_unset_runtime_flags(tmp_path, experiment, expected):
    from dataclasses import asdict

    from dendritic_modeling.scripts.script_utils.setup_utils import (
        resolve_runtime_performance_settings,
    )

    config = _typed_config(tmp_path, experiment=experiment)

    assert resolve_runtime_performance_settings(config.experiment) == expected
    resolve_runtime_performance_settings(config.experiment, write_back=True)
    recorded = asdict(config)["experiment"]
    assert {key: recorded[key] for key in expected} == expected


@pytest.mark.parametrize(
    ("experiment", "message"),
    [
        ({"allow_tf32": True, "float32_matmul_precision": "highest"}, "contradicts"),
        ({"allow_tf32": False, "float32_matmul_precision": "medium"}, "contradicts"),
        ({"float32_matmul_precision": "fast"}, "must be one of"),
        ({"strict_deterministic": True, "allow_tf32": True}, "requires exact math"),
        (
            {"strict_deterministic": True, "float32_matmul_precision": "high"},
            "requires exact math",
        ),
        ({"strict_deterministic": True, "cudnn_benchmark": True}, "exact math"),
    ],
)
def test_contradictory_runtime_flags_raise(experiment, message):
    from dendritic_modeling.utils.reproducibility import (
        resolve_runtime_performance_settings,
    )

    with pytest.raises(ValueError, match=message):
        resolve_runtime_performance_settings(experiment)


def test_setup_environment_applies_fast_mode(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from dendritic_modeling.config import DataConfig, ModelConfig, TrainingConfig
    from dendritic_modeling.config.config import ExperimentConfig
    from dendritic_modeling.scripts.script_utils import setup_utils

    dataset = TensorDataset(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long))
    seeded = []
    monkeypatch.setattr(
        setup_utils, "get_unified_datasets", lambda *, task_cfg: (dataset,) * 3
    )
    monkeypatch.setattr(
        setup_utils,
        "initialize_model",
        lambda _config: (torch.nn.Identity(), torch.nn.Identity()),
    )
    monkeypatch.setattr(
        setup_utils, "set_seed", lambda seed, **flags: seeded.append(flags)
    )
    experiment = ExperimentConfig(seed=3, fast_mode=True)

    setup_utils.setup_environment(
        model_config=ModelConfig(),
        training_config=TrainingConfig(),
        data_config=DataConfig(dataset_name="dummy", base_dir=""),
        outputs_config=SimpleNamespace(results_dir=str(tmp_path), run_name="fast"),
        experiment_config=experiment,
        is_main=False,
    )

    assert seeded and all(
        {key: flags[key] for key in _FAST} == _FAST for flags in seeded
    )
    assert experiment.deterministic is False and experiment.allow_tf32 is True


# ----------------------------------------------------------------------------
# 13. Retired experiment.enable_profiling / experiment.enable_hooks
# ----------------------------------------------------------------------------


def test_enable_profiling_true_cannot_be_loaded(tmp_path, caplog):
    with pytest.raises(ValueError, match="never implemented"):
        _typed_config(tmp_path, experiment={"enable_profiling": True})
    with caplog.at_level("WARNING"):
        config = _typed_config(tmp_path, experiment={"enable_profiling": False})
    assert "Ignoring experiment.enable_profiling" in caplog.text
    assert not hasattr(config.experiment, "enable_profiling")


@pytest.mark.parametrize("enable_hooks", [True, False])
def test_enable_hooks_is_ignored_and_scaling_hooks_still_run(
    tmp_path, caplog, enable_hooks
):
    with caplog.at_level("WARNING"):
        config = _typed_config(tmp_path, experiment={"enable_hooks": enable_hooks})
    assert "never disabled anything" in caplog.text
    kwargs = _trainer_kwargs_from_typed(config, tmp_path)
    assert "enable_hooks" not in kwargs
    model = _scaled_einet_model()
    trainer = get_trainer("standard", _optimizer(model), kwargs)
    trainer.train(model, _data(), _data(seed=1))


# ----------------------------------------------------------------------------
# 6. Explicit L1/L2 weight selection; B. pruning acts on the same selection
# ----------------------------------------------------------------------------


def test_regularized_weights_are_synapse_block_and_dense_weights_only():
    from dendritic_modeling.training.regularization import iter_regularized_weights

    torch.manual_seed(0)
    core = get_architecture(
        "einet", deepcopy(_config("einet", "shunting", "param_tanh")), input_dim=IN_DIM
    )
    model = Classifier(Identity(IN_DIM), core, nn.Linear(core.output_dim, N_CLASSES))
    selected = {name for name, _module, _param in iter_regularized_weights(model)}
    leaves = {name.rsplit(".", 1)[-1] for name in selected}

    assert leaves == {"pre_w", "log_weight", "weight"}
    assert "decoder_network.weight" in selected
    assert sum(name.endswith("pre_w") for name in selected) >= 1
    for name, _param in model.named_parameters():
        if name.endswith(("bias", "log_m", ".b", "log_output_scale")):
            assert name not in selected


def test_global_l2_regularization_reaches_synapse_weights():
    from dendritic_modeling.config.regularization import RegularizationConfig
    from dendritic_modeling.training.regularization import RegularizationManager

    model = _einet_model()
    manager = RegularizationManager(RegularizationConfig(l2_weight=1.0))
    named = dict(model.named_parameters())
    grads = dict(
        zip(
            named,
            torch.autograd.grad(
                manager.compute_regularization_loss(model),
                list(named.values()),
                allow_unused=True,
            ),
        )
    )

    def touched(name):
        return grads[name] is not None and float(grads[name].abs().sum()) > 0

    assert all(touched(name) for name in named if name.endswith("pre_w"))
    assert not any(
        touched(name)
        for name in named
        if name.endswith(("bias", "log_m", ".b", "log_output_scale"))
    )


def _pruning_manager(**selective):
    from dendritic_modeling.config.regularization import PruningConfig, SelectiveConfig
    from dendritic_modeling.training.regularization import PruningManager

    kwargs = {"selective": SelectiveConfig(**selective)} if selective else {}
    return PruningManager(PruningConfig(enabled=True, threshold=0.1, **kwargs))


def test_pruning_rejects_transformed_synapse_weights():
    with pytest.raises(ValueError, match="positive transform"):
        _pruning_manager(weight_types=["all"]).prune_model(_einet_model())


def test_pruning_rejects_a_selection_that_matches_nothing():
    with pytest.raises(ValueError, match="selects no weight tensor"):
        _pruning_manager().validate_model(_population_model())


def test_pruning_zeroes_identity_weights_and_keeps_biases():
    model = _mlp_model()
    biases = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if name.endswith("bias")
    }

    stats = _pruning_manager(weight_types=["all"]).prune_model(model)

    assert stats["pruned_params"] > 0
    assert all(
        torch.equal(param, biases[name])
        for name, param in model.named_parameters()
        if name in biases
    )


@pytest.mark.parametrize(
    ("pruning", "message"),
    [
        ({"enabled": True}, "positive transform"),
        ({"enabled": True, "dynamic_pruning": True}, "dynamic_pruning"),
    ],
)
def test_pruning_requests_fail_before_training(tmp_path, pruning, message):
    with pytest.raises(ValueError, match=message):
        _run_main_phase(tmp_path, {"pruning": pruning})


# ----------------------------------------------------------------------------
# 5. FSDP: configured optimizer, unsupported options, final weights
# C. FSDP parameter groups match synapse/BlockLinear subclasses
# F. FSDP rejects impact_rewiring
# ----------------------------------------------------------------------------

_FSDP_GROUP_LRS = {
    "split_params": True,
    "lr": 0.01,
    "topk_lr": 1e-3,
    "blocklinear_lr": 1e-4,
    "reactivation_lr": 1e-5,
    "decoder_lr": 2e-3,
}


def _fsdp_optimizer(model, optimizer_config):
    from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer

    trainer = FSDPTrainer.__new__(FSDPTrainer)
    trainer.rank = 0
    trainer.optimizer_config = optimizer_config
    return trainer.create_optimizer(model)


@pytest.mark.parametrize(
    "options",
    [
        {"topk_type": "dense_to_sparse"},
        {"topk_type": "stochastic"},
        {"topk_type": "variance"},
        {"topk_type": "indexed_release"},
        {"topk_type": "standard", "efficient_blocklinear": True},
    ],
)
def test_fsdp_param_groups_include_synapse_and_blocklinear_subclasses(options):
    from dendritic_modeling.networks import BlockLinear
    from dendritic_modeling.training.regularization import _SYNAPSE_WEIGHT_MODULES

    model = _population_model(**options)
    optimizer = _fsdp_optimizer(model, {"param_groups": dict(_FSDP_GROUP_LRS)})
    group_of = {
        id(param): (group["name"], group["lr"])
        for group in optimizer.param_groups
        for param in group["params"]
    }

    synapses = [m for m in model.modules() if isinstance(m, _SYNAPSE_WEIGHT_MODULES)]
    blocks = [m for m in model.modules() if isinstance(m, BlockLinear)]
    assert synapses and blocks
    assert {group_of[id(m.pre_w)] for m in synapses} == {("topk", 1e-3)}
    assert {group_of[id(m.log_weight)] for m in blocks} == {("blocklinear", 1e-4)}


def test_fsdp_optimizer_follows_training_main_optimizer(monkeypatch, tmp_path):
    from dendritic_modeling.scripts.training import train_experiments_fsdp

    # get_fsdp_config queries the CUDA device; it is irrelevant here.
    monkeypatch.setattr(train_experiments_fsdp, "get_fsdp_config", lambda **_: {})

    config = _typed_config(
        tmp_path,
        {
            "optimizer": {"name": "adamw", "betas": [0.8, 0.95], "eps": 1e-6},
            "common": {"synapse_decay_rate": 0.4, "param_groups": _FSDP_GROUP_LRS},
        },
        extra={"distributed": {"mode": "fsdp", "fsdp": {"mixed_precision": False}}},
    )
    _common, _fsdp, optimizer_config = (
        train_experiments_fsdp._build_fsdp_runtime_configs(config)
    )
    optimizer = _fsdp_optimizer(_einet_model(), optimizer_config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert "weight_decay" not in optimizer_config
    for group in optimizer.param_groups:
        assert tuple(group["betas"]) == (0.8, 0.95)
        assert group["eps"] == 1e-6
        assert group["weight_decay"] == 0.0  # synapse_decay_rate is not L2


_FSDP_SAFE_COMMON = {"synapse_decay_rate": 0.0, "load_best_state_dict": False}


@pytest.mark.parametrize(
    ("main", "experiment", "builder", "message"),
    [
        ({"common": {}}, {}, _einet_model, "synapse_decay_rate"),
        (
            {"common": {"lr_schedule": "cosine"}},
            {},
            _einet_model,
            "lr_schedule",
        ),
        ({"regularization": {"l2_weight": 1.0}}, {}, _einet_model, "regularization"),
        (
            {"common": {"enable_nan_checking": True}},
            {},
            _einet_model,
            "enable_nan_checking",
        ),
        (
            {"common": {"profile_dataloader": True}},
            {},
            _einet_model,
            "profile_dataloader",
        ),
        (
            {
                "common": {
                    "reactivation_update_mode": "quantile",
                    "recalibrate_reactivation_every": 1,
                }
            },
            {},
            _einet_model,
            "reactivation_update_mode",
        ),
        (
            {"common": {"impact_rewiring": {"eligible_fraction": 0.1}}},
            {},
            _einet_model,
            "impact_rewiring",
        ),
        ({"strategy": "local_ca"}, {}, _einet_model, "strategy"),
        ({"pruning": {"enabled": True}}, {}, _einet_model, "pruning"),
        (
            {},
            {},
            lambda: _population_model(
                topk_type="indexed_rewire",
                indexed_rewire_frequency=1,
                indexed_rewire_quantile=0.5,
            ),
            "IndexedRewireLinear",
        ),
        (
            {
                "optimizer": {"lr": 3e-4},
                "common": {"param_groups": {"lr": 1e-2, "split_params": False}},
            },
            {},
            _einet_model,
            "is not applied",
        ),
    ],
)
def test_fsdp_rejects_options_it_does_not_apply(
    tmp_path, main, experiment, builder, message
):
    from dendritic_modeling.scripts.training.train_experiments_fsdp import (
        _validate_fsdp_training_options,
    )

    if main != {"common": {}}:
        main = _deep_update({"common": _FSDP_SAFE_COMMON}, main)
    config = _typed_config(tmp_path, main, experiment=experiment)

    with pytest.raises(ValueError, match=message):
        _validate_fsdp_training_options(config, builder())


@pytest.mark.parametrize(
    "override, message",
    [
        ({"lr_schedule": "cosine"}, "recurrent.lr_schedule"),
        (
            {"lr_schedule": "cosine", "lr_warmup_epochs": 2},
            "recurrent.lr_warmup_epochs",
        ),
        ({"use_amp": True}, "recurrent.use_amp"),
        ({"grad_clip_value": 0.5}, "recurrent.grad_clip_value"),
    ],
)
def test_fsdp_rejects_recurrent_overrides_it_would_drop(tmp_path, override, message):
    from dendritic_modeling.scripts.training.train_experiments_fsdp import (
        _validate_fsdp_training_options,
    )

    config = _typed_config(
        tmp_path,
        {
            "strategy": "recurrent",
            "common": _FSDP_SAFE_COMMON,
            "strategies": {"recurrent": override},
        },
    )
    with pytest.raises(ValueError, match=message):
        _validate_fsdp_training_options(config, _einet_model())


def test_fsdp_accepts_recurrent_overrides_equal_to_what_it_applies(tmp_path):
    from dendritic_modeling.scripts.training.train_experiments_fsdp import (
        _validate_fsdp_training_options,
    )

    common = dict(_FSDP_SAFE_COMMON, grad_clip_value=1.5)
    config = _typed_config(
        tmp_path,
        {
            "strategy": "recurrent",
            "common": common,
            "strategies": {
                "recurrent": {
                    "grad_clip_value": 1.5,
                    "use_amp": False,
                    "lr_schedule": "none",
                }
            },
        },
    )
    _validate_fsdp_training_options(config, _einet_model())


def test_fsdp_accepts_supported_options(tmp_path):
    from dendritic_modeling.scripts.training.train_experiments_fsdp import (
        _validate_fsdp_training_options,
    )

    config = _typed_config(
        tmp_path, {"strategy": "recurrent", "common": _FSDP_SAFE_COMMON}
    )
    _validate_fsdp_training_options(config, _einet_model())


@pytest.mark.parametrize("requested", [True, False])
def test_fsdp_records_that_final_weights_are_kept(caplog, requested):
    import logging
    from types import SimpleNamespace

    from dendritic_modeling.scripts.training.train_experiments_fsdp import (
        _record_fsdp_final_weights,
    )

    results = {}
    with caplog.at_level(logging.WARNING):
        _record_fsdp_final_weights(
            results, SimpleNamespace(load_best_state_dict=requested), rank=0
        )

    assert results == {
        "final_weights": "last_completed_epoch",
        "load_best_state_dict_requested": requested,
        "load_best_state_dict_applied": False,
    }
    assert ("keeps the final weights" in caplog.text) is requested


# ----------------------------------------------------------------------------
# H. There is no separate encoder training phase
# ----------------------------------------------------------------------------


def test_training_encoder_section_is_ignored_with_a_warning(tmp_path, caplog):
    import yaml

    from dendritic_modeling.config import load_config

    path = tmp_path / "encoder.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "training": {
                    "encoder": {"common": {"epochs": 5}},
                    "main": _deep_update(_TYPED_MAIN, None),
                },
                "experiment": {"seed": 5},
                "analysis": {},
            }
        )
    )
    with caplog.at_level("WARNING"):
        config = load_config(str(path))
    assert "Ignoring training.encoder: it was retired on 2026-10-07" in caplog.text
    assert "training.encoder_pretraining pretrains autoencoder encoders" in caplog.text
    assert not hasattr(config.training, "encoder")


@pytest.mark.parametrize(
    ("encoder", "rejected"),
    [("autoencoder", True), ("linear", False), ("identity", False)],
)
def test_only_autoencoder_encoders_are_rejected(encoder, rejected):
    from dendritic_modeling.networks.architectures.classical.autoencoder import (
        MLPAutoencoder,
    )
    from dendritic_modeling.scripts.training import train_experiments

    network = {
        "autoencoder": lambda: MLPAutoencoder(
            input_dim=IN_DIM,
            encoder_hidden_dims=[4],
            latent_dim=3,
            decoder_hidden_dims=[4],
        ),
        "linear": lambda: nn.Linear(IN_DIM, 4),
        "identity": lambda: Identity(IN_DIM),
    }[encoder]()
    data = _data()

    def run():
        return train_experiments._encoder_datasets(
            encoder_network=network, train_ds=data, valid_ds=data, test_ds=data
        )

    if rejected:
        with pytest.raises(ValueError, match="must be trained before the main phase"):
            run()
    else:
        assert run() == (network, data, data, data)


# ----------------------------------------------------------------------------
# I. Step options a trainer's own loop does not apply fail at construction
# ----------------------------------------------------------------------------


def test_trainer_classes_declare_the_step_options_they_skip():
    from dendritic_modeling.training.strategies.feedback_alignment import (
        FeedbackAlignmentTrainer,
    )
    from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer
    from dendritic_modeling.training.strategies.local_learning import (
        LocalCreditAssignment,
    )
    from dendritic_modeling.training.strategies.presentation_stream import (
        PresentationStreamTrainer,
    )
    from dendritic_modeling.training.strategies.recurrent_trainer import (
        RecurrentTrainer,
    )
    from dendritic_modeling.training.strategies.soma_dfa import SomaDFATrainer
    from dendritic_modeling.training.strategies.standard import Trainer
    from dendritic_modeling.training.strategies.two_step import (
        TwoStepTrainer,
        TwoStepTrainerWithKL,
    )

    loop_options = {"enable_nan_checking"}
    assert not loop_options & Trainer.unapplied_options
    for cls in (
        FeedbackAlignmentTrainer,
        FSDPTrainer,
        LocalCreditAssignment,
        PresentationStreamTrainer,
        RecurrentTrainer,
        SomaDFATrainer,
        TwoStepTrainer,
        TwoStepTrainerWithKL,
    ):
        assert loop_options <= cls.unapplied_options, cls.__name__
    assert "log_first_step_diagnostics" not in FSDPTrainer.unapplied_options


@pytest.mark.parametrize("option", ["enable_nan_checking", "use_amp"])
def test_feedback_alignment_rejects_step_options_its_loop_skips(option):
    model = _mlp_model()
    _make_trainer("standard", model, **{option: True})
    with pytest.raises(
        ValueError, match=rf"FeedbackAlignmentTrainer does not apply {option}"
    ):
        _make_trainer("fa", model, **{option: True})


@pytest.mark.parametrize("strategy", ["freeze_branches", "freeze_branch_kl"])
def test_two_step_rejects_regularization_its_first_stage_skips(strategy):
    from dendritic_modeling.config.training import RegularizationConfig

    model = _einet_model()
    trainer, _ = _make_trainer(
        strategy, model, regularization=RegularizationConfig(l2_weight=1e-3)
    )
    with pytest.raises(ValueError, match="does not apply auxiliary regularization"):
        trainer.train(model, _data(), _data(seed=1))


def test_misspelled_trainer_options_raise_with_a_hint():
    from dendritic_modeling.config.keys import UnknownConfigKeyError

    model = _mlp_model()
    with pytest.raises(UnknownConfigKeyError, match="did you mean 'grad_clip_value'"):
        get_trainer("standard", _optimizer(model), {"grad_clip_valu": 1.0})


def test_strategy_options_belong_to_their_trainer():
    from dendritic_modeling.config.keys import UnknownConfigKeyError

    model = _einet_model()
    _make_trainer("freeze_branches", model, pretrain_epochs=2)
    with pytest.raises(UnknownConfigKeyError, match="pretrain_epochs"):
        _make_trainer("standard", model, pretrain_epochs=2)


def test_normalized_options_are_not_overwritten_by_raw_values():
    from dendritic_modeling.training.strategies.standard import Trainer

    model = _mlp_model()
    trainer = Trainer(_optimizer(model), seed=None, loader_seed=None)
    assert trainer.seed == 0
    assert trainer.loader_seed == 0
    assert trainer.regularization_config is not None


def test_each_trainer_applies_its_historical_clipping_operator_by_default():
    from dendritic_modeling.config.training import default_grad_clip_mode
    from dendritic_modeling.training.strategies.fsdp_standard import FSDPTrainer
    from dendritic_modeling.training.strategies.recurrent_trainer import (
        RecurrentTrainer,
    )
    from dendritic_modeling.training.strategies.standard import Trainer

    assert Trainer.default_grad_clip_mode == default_grad_clip_mode("standard")
    assert RecurrentTrainer.default_grad_clip_mode == default_grad_clip_mode(
        "recurrent"
    )
    assert FSDPTrainer.default_grad_clip_mode == default_grad_clip_mode(
        "standard", "fsdp"
    )
    assert default_grad_clip_mode("local_ca") == "value"


def test_adaptive_clipping_flag_loads_as_norm_clipping(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        config = _typed_config(tmp_path, {"common": {"enable_adaptive_clipping": True}})
    assert config.training.main.common.grad_clip_mode == "norm"
    assert "grad_clip_mode: norm" in caplog.text
    with pytest.raises(ValueError, match="contradicts"):
        _typed_config(
            tmp_path,
            {"common": {"enable_adaptive_clipping": True, "grad_clip_mode": "value"}},
            name="contradiction",
        )


def test_clip_gradients_and_nonfinite_repair():
    from dendritic_modeling.training.clipping import (
        clip_gradients,
        zero_nonfinite_gradients,
    )

    param = torch.nn.Parameter(torch.zeros(2))
    param.grad = torch.tensor([3.0, -4.0])
    clip_gradients([param], 1.0, "value")
    assert torch.equal(param.grad, torch.tensor([1.0, -1.0]))
    param.grad = torch.tensor([3.0, 4.0])
    clip_gradients([param], 1.0, "norm")
    assert torch.allclose(param.grad.norm(), torch.tensor(1.0))
    param.grad = torch.tensor([float("inf"), 2.0])
    assert zero_nonfinite_gradients([param])
    assert torch.equal(param.grad, torch.tensor([0.0, 2.0]))
    assert not zero_nonfinite_gradients([param])


def test_weight_decay_rate_loads_as_synapse_decay_rate(tmp_path, caplog):
    import logging

    from dendritic_modeling.config import keys

    keys._WARNED_RENAMED.discard("training.main.common.weight_decay_rate")
    with caplog.at_level(logging.WARNING):
        config = _typed_config(tmp_path, {"common": {"weight_decay_rate": 0.03}})
    assert config.training.main.common.synapse_decay_rate == 0.03
    assert "renamed training.main.common.synapse_decay_rate" in caplog.text
    with pytest.raises(ValueError, match="sets both"):
        _typed_config(
            tmp_path,
            {"common": {"weight_decay_rate": 0.03, "synapse_decay_rate": 0.02}},
        )


def test_local_ca_rejects_models_without_branch_layers():
    model = _mlp_model()
    trainer, _ = _make_trainer("local_ca", model, local_rule_config={})
    with pytest.raises(ValueError, match="has none; use 'standard'"):
        trainer.train(model, _data(), _data(seed=1))
